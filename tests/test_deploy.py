"""部署层全链路测试：导出 → 等价校验 → 量化 → 基准 → 回测。

用 toy 环境而不是真实机器人，理由与 test_toy_env.py 一样：这条链路要能在
几秒内跑完，才能在每次改动之后都跑一遍。链路里没有任何一处依赖 MuJoCo，
换成 G1 只是换个 spec 和一个更慢的环境。

策略在测试里手搓而不训练。导出、量化、回测三件事的正确性与策略训练得好不好
无关，用一个固定权重的 MLP 反而更容易复现：网络权重的随机性一旦引入，
"量化后回报掉了 3%" 这类结论就要先排除"这次初始化不好"的可能。

依赖说明：torch 是项目核心依赖，直接 import；onnxruntime 属于可选的 deploy
extra，缺失时整个模块跳过，而不是报一堆 collection error。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime", reason="部署链路依赖 onnxruntime（deploy extra）")

import torch  # noqa: E402

from robotrl.algorithms.base import Policy  # noqa: E402
from robotrl.assets.spec import Morphology, RobotSpec  # noqa: E402
from robotrl.configs.schema import DeployConfig  # noqa: E402
from robotrl.deploy import benchmark as bench  # noqa: E402
from robotrl.deploy import engine as engines  # noqa: E402
from robotrl.deploy import equivalence as eq  # noqa: E402
from robotrl.deploy import evaluate as backtest_mod  # noqa: E402
from robotrl.deploy import export_onnx as exporter  # noqa: E402
from robotrl.deploy import quantize as quant  # noqa: E402
from robotrl.envs.toy import TOY_SPEC, ToyVelocityEnv  # noqa: E402


class TinyPolicy(Policy):
    """两层 tanh MLP，权重由种子确定。

    输出层用 tanh 是为了满足 Policy 契约里 [-1, 1] 的值域要求；量化误差在
    饱和区会被压缩，所以这个网络里既要有饱和样本也要有线性区的样本——
    obs_std 刻意给成非平凡值，否则归一化等于没做，测不出"归一化进了图"。
    """

    def __init__(self, obs_dim: int, action_dim: int, *, hidden: int = 32, seed: int = 0) -> None:
        super().__init__(obs_dim, action_dim)
        gen = torch.Generator().manual_seed(seed)
        self.body = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, action_dim),
            torch.nn.Tanh(),
        )
        for layer in self.body:
            if isinstance(layer, torch.nn.Linear):
                bound = 1.0 / float(np.sqrt(layer.in_features))
                with torch.no_grad():
                    layer.weight.uniform_(-bound, bound, generator=gen)
                    layer.bias.uniform_(-bound, bound, generator=gen)

        with torch.no_grad():
            self.obs_mean.copy_(torch.linspace(-0.5, 0.5, obs_dim))
            self.obs_std.copy_(torch.linspace(0.8, 1.5, obs_dim))
        self.eval()

    def _policy_forward(self, obs_normalized: torch.Tensor) -> torch.Tensor:
        return self.body(obs_normalized)


# ---------------------------------------------------------------------------
# 夹具：整条链路的产物只做一次，各用例共用
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def deploy_cfg() -> DeployConfig:
    """用默认配置。容差、预热次数这些阈值都从这里取，保证测试与配置不脱节。"""
    return DeployConfig()


@pytest.fixture(scope="module")
def env() -> ToyVelocityEnv:
    return ToyVelocityEnv(seed=0)


@pytest.fixture(scope="module")
def policy(env) -> TinyPolicy:
    return TinyPolicy(env.obs_dim, env.action_dim)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory, policy) -> Path:
    """落一个与 Trainer.save() 同格式的 checkpoint。

    顺手把 config 也写进去，用来验证"能从 checkpoint 里认出形态"这条路径。
    """
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    torch.save(
        {
            "policy": policy.state_dict(),
            "num_timesteps": 0,
            "config": {"env": {"robot": "toy"}},
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def loaded(policy, checkpoint) -> TinyPolicy:
    """从 checkpoint 恢复出来的策略，后续所有用例都用它。"""
    return exporter.load_policy(  # type: ignore[return-value]
        checkpoint,
        policy_factory=lambda: TinyPolicy(policy.obs_dim, policy.action_dim),
    )


@pytest.fixture(scope="module")
def onnx_fp32(tmp_path_factory, loaded, env) -> Path:
    path = tmp_path_factory.mktemp("exported") / "policy.onnx"
    exporter.export_policy(loaded, path, spec=TOY_SPEC, contract=env.obs_contract)
    return path


@pytest.fixture(scope="module")
def onnx_int8(tmp_path_factory, onnx_fp32) -> Path:
    path = onnx_fp32.parent / "policy_int8.onnx"
    quant.quantize_dynamic_model(onnx_fp32, path)
    return path


@pytest.fixture(scope="module")
def onnx_qdq(tmp_path_factory, onnx_fp32, env) -> Path:
    """静态 QDQ 量化产物。某些 onnxruntime 版本对极小模型会拒绝量化，跳过即可。"""
    path = onnx_fp32.parent / "policy_qdq.onnx"
    fp32_engine = engines.load_engine(onnx_fp32)
    try:
        calibration = quant.collect_calibration_data(env, steps=256, seed=7, engine=fp32_engine)
        quant.quantize_static_model(onnx_fp32, path, calibration, batch_size=32)
    except Exception as exc:  # 量化器不支持就跳过，不让它挡住其他用例
        pytest.skip(f"静态 QDQ 量化在本环境不可用：{type(exc).__name__}: {exc}")
    finally:
        fp32_engine.close()
    return path


@pytest.fixture(scope="module")
def fp32_engine(onnx_fp32):
    engine = engines.load_engine(onnx_fp32)
    yield engine
    engine.close()


@pytest.fixture(scope="module")
def int8_engine(onnx_int8):
    engine = engines.load_engine(onnx_int8)
    yield engine
    engine.close()


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_preserves_weights(loaded, policy):
    before = policy.body[0].weight.detach().numpy()
    after = loaded.body[0].weight.detach().numpy()
    assert np.array_equal(before, after)


def test_spec_recovered_from_checkpoint(checkpoint):
    """checkpoint 里存了配置就能认出形态，调用方不必再传一次。"""
    spec = exporter.spec_from_checkpoint(checkpoint)
    assert spec is not None
    assert spec.name == TOY_SPEC.name


def test_loading_without_structure_fails_loudly(checkpoint):
    """state_dict 里没有网络结构，调用方不给结构就必须报错，而不是猜。"""
    with pytest.raises(ValueError, match="policy_factory"):
        exporter.load_policy(checkpoint)


def test_export_self_check_passes(onnx_fp32, env, deploy_cfg):
    report = exporter.inspect_onnx(onnx_fp32)

    assert report.checker_passed
    assert report.missing_meta == []
    assert report.inputs["obs"] == [None, env.obs_dim]  # 批轴动态，末维确定
    assert report.outputs["action"] == [None, env.action_dim]
    assert report.num_params > 0
    assert report.size_bytes > 0
    assert report.opset == deploy_cfg.opset


def test_export_writes_restorable_metadata(onnx_fp32, env):
    """metadata 是端侧还原关节目标角的唯一依据，逐项核对。"""
    engine = engines.load_engine(onnx_fp32)
    try:
        assert engine.default_angles.tolist() == pytest.approx(list(TOY_SPEC.default_angles))
        assert engine.action_scale.tolist() == pytest.approx(
            list(np.full(TOY_SPEC.n_dof, TOY_SPEC.action_scale))
        )

        action = np.linspace(-1.0, 1.0, env.action_dim)
        expected = np.asarray(TOY_SPEC.default_angles) + TOY_SPEC.action_scale_array * action
        assert engine.action_to_joint_targets(action) == pytest.approx(expected)

        # 观测分段布局也要在，端侧按段做误差归因要靠它
        segments = engine.meta_value("obs_segments")
        assert [name for name, _ in segments] == list(env.obs_contract.names)
        assert [dim for _, dim in segments] == [s.dim for s in env.obs_contract.segments]
        assert engine.meta_value("obs_mean") is not None
    finally:
        engine.close()


def test_export_rejects_joint_count_mismatch(tmp_path):
    """策略动作维度与形态关节数不一致时必须报错：导出的 metadata 会误导端侧。"""
    wrong = TinyPolicy(ToyVelocityEnv(seed=0).obs_dim, TOY_SPEC.n_dof + 1)
    with pytest.raises(ValueError, match="不一致"):
        exporter.export_policy(wrong, tmp_path / "wrong.onnx", spec=TOY_SPEC)


def test_export_requires_spec(loaded, tmp_path):
    with pytest.raises(ValueError, match="RobotSpec"):
        exporter.export_policy(loaded, tmp_path / "no_spec.onnx")


def test_metadata_survives_rewrite(onnx_fp32, tmp_path):
    """写 metadata 是量化后必须补的一步，先确认它本身是可读可写的。"""
    path = tmp_path / "copy.onnx"
    path.write_bytes(onnx_fp32.read_bytes())
    engines.write_onnx_metadata(path, {"default_angles": json.dumps([0.1, 0.2]), "note": "x"})

    meta = engines.read_onnx_metadata(path)
    assert meta["note"] == "x"
    assert engines.parse_metadata_value(meta["default_angles"]) == [0.1, 0.2]


def test_engine_requires_metadata_for_joint_targets(tmp_path, loaded):
    """没有 metadata 的图（例如别的工具导出的）要报错，而不是拿 0 当默认角。"""
    path = tmp_path / "bare.onnx"
    dummy = torch.zeros(1, loaded.obs_dim)
    torch.onnx.export(
        loaded,
        (dummy,),
        str(path),
        input_names=["obs"],
        output_names=["action"],
        dynamo=False,  # 老导出器不写 metadata，正合本用例需要
    )

    engine = engines.load_engine(path)
    try:
        with pytest.raises(KeyError, match="default_angles"):
            _ = engine.default_angles
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# 等价校验
# ---------------------------------------------------------------------------


def test_equivalence_on_random_inputs(loaded, fp32_engine, env, deploy_cfg):
    """导出必须是无损的：误差只应来自浮点累加顺序，量级 1e-7。"""
    inputs = eq.observation_sets(env.obs_dim, 128, seed=0, obs_clip=float(loaded.obs_clip))
    report = eq.check_equivalence(loaded, fp32_engine, inputs, tol=deploy_cfg.equivalence_tol)

    assert report.passed, report.summary()
    assert report.metrics.max_abs_err < deploy_cfg.equivalence_tol
    assert report.metrics.violation_ratio == 0.0
    assert set(report.per_input) == {"uniform", "normal", "beyond_clip"}


def test_equivalence_on_env_observations(loaded, fp32_engine, env, deploy_cfg):
    """真实观测上再验一遍。随机输入落不到策略真正会去的地方。"""
    observations = backtest_mod.collect_observations(env, steps=256, seed=1, engine=fp32_engine)
    report = eq.check_equivalence(loaded, fp32_engine, observations, tol=deploy_cfg.equivalence_tol)

    assert report.passed, report.summary()
    assert report.n_samples == 256


def test_equivalence_balances_the_normalizer(loaded, fp32_engine):
    """归一化统计量进图这件事，用一个刻意偏置的策略输入来验证。

    策略的 obs_mean 不为 0，若导出丢了归一化，输出的差异会是 O(1) 量级。
    """
    biased = np.full((1, loaded.obs_dim), 3.0, dtype=np.float32)
    torch_out = eq.policy_forward_numpy(loaded, biased)
    onnx_out = fp32_engine.infer_batch(biased)
    assert np.abs(torch_out - onnx_out).max() < 1e-6

    # 同一批输入，若不做归一化，输出会明显不同——证明确实做了这一步
    without_norm = loaded._policy_forward(torch.as_tensor(biased)).detach().numpy()
    assert np.abs(torch_out - without_norm).max() > 1e-3


def test_equivalence_detects_a_mismatch(env, fp32_engine, loaded):
    """校验必须有牙齿：换一个策略去比对，必须判不通过。"""
    other = TinyPolicy(env.obs_dim, env.action_dim, seed=99)
    report = eq.check_equivalence(other, fp32_engine, np.zeros((4, env.obs_dim)), tol=1e-5)

    assert not report.passed
    with pytest.raises(AssertionError, match="超过容差"):
        report.assert_passed()
    assert report.max_abs_err > 1e-3


def test_segment_attribution_ranks_observation_groups(fp32_engine, int8_engine, env):
    """按段做误差归因：各段占比之和为 1，且最敏感段必须来自契约。"""
    observations = backtest_mod.collect_observations(env, steps=256, seed=5, engine=fp32_engine)
    report = eq.compare_engines(
        fp32_engine, int8_engine, observations, contract=env.obs_contract, tol=float("inf")
    )

    assert [s.name for s in report.segments] == list(env.obs_contract.names)
    shares = [s.share for s in report.segments]
    assert sum(shares) == pytest.approx(1.0)
    assert all(s.output_max_delta >= 0 for s in report.segments)
    assert all(np.isfinite(s.input_rel_err) for s in report.segments)
    assert report.top_sensitive_segment in env.obs_contract.names


# ---------------------------------------------------------------------------
# 量化
# ---------------------------------------------------------------------------


def test_dynamic_quantization_replaces_ops_and_keeps_metadata(onnx_fp32, tmp_path):
    report = quant.quantize(onnx_fp32, tmp_path / "policy_int8.onnx", mode="dynamic")

    assert report.effective, report.summary()
    assert "MatMulInteger" in report.quantized_ops
    assert report.metadata_ok, report.summary()
    assert report.calibration_samples == 0  # 动态量化不需要校准集
    assert report.size_int8 > 0


def test_static_qdq_quantization_produces_quantize_nodes(onnx_qdq):
    ops = quant.count_quantized_ops(onnx_qdq)
    assert ops.get("QuantizeLinear", 0) > 0
    assert ops.get("DequantizeLinear", 0) > 0
    assert quant.read_onnx_metadata(onnx_qdq).get("default_angles")


def test_static_quantization_needs_calibration(onnx_fp32, tmp_path):
    with pytest.raises(ValueError, match="校准集"):
        quant.quantize(onnx_fp32, tmp_path / "x.onnx", mode="static")


def test_quantize_rejects_unknown_mode(onnx_fp32, tmp_path):
    with pytest.raises(ValueError, match="未知量化模式"):
        quant.quantize(onnx_fp32, tmp_path / "x.onnx", mode="int4")


def test_quantization_error_is_bounded(fp32_engine, int8_engine, env):
    """量化误差的宽松上限。

    门限定在 0.05 是有意义的：动作恒在 [-1, 1]，0.05 相当于满量程的 2.5%，
    真正把网络量化坏（例如 scale 估偏一个量级）时误差会远大于它；实测值在
    1e-3 量级，留了两个数量级的余量。
    """
    observations = backtest_mod.collect_observations(env, steps=512, seed=11, engine=fp32_engine)
    report = eq.compare_engines(fp32_engine, int8_engine, observations, tol=float("inf"))

    assert report.metrics.max_abs_err < 0.05, report.summary()
    assert report.metrics.mean_abs_err < 0.01, report.summary()


def test_calibration_data_is_drawn_from_the_env(env, fp32_engine):
    data = quant.collect_calibration_data(env, steps=128, seed=3, engine=fp32_engine)

    assert data.shape == (128, env.obs_dim)
    assert data.dtype == np.float32
    assert np.all(np.isfinite(data))


# ---------------------------------------------------------------------------
# 基准
# ---------------------------------------------------------------------------


def test_benchmark_reports_full_latency_distribution(fp32_engine, env):
    sample = bench.make_sample_input(env.obs_dim, seed=0)
    stats = bench.benchmark_engine(fp32_engine, sample, warmup=20, runs=200)

    assert stats.count == 200
    assert stats.p50_ms <= stats.p90_ms <= stats.p95_ms <= stats.p99_ms <= stats.max_ms
    assert stats.min_ms <= stats.p50_ms
    assert stats.mean_ms > 0 and stats.std_ms >= 0
    assert stats.throughput > 0
    assert stats.threads == 1
    assert stats.batch_size == 1


def test_benchmark_warmup_keeps_samples_out_of_the_stats(fp32_engine, env):
    """预热次数不能混进统计——这正是它存在的意义。"""
    sample = bench.make_sample_input(env.obs_dim, seed=0)
    stats = bench.benchmark_engine(fp32_engine, sample, warmup=50, runs=100)
    assert stats.count == 100


def test_benchmark_compares_fp32_and_int8(onnx_fp32, onnx_int8, env):
    report = bench.compare_benchmark(
        onnx_fp32, onnx_int8, warmup=10, runs=150, batch_size=1, threads=1
    )

    assert report.fp32.count == report.int8.count == 150
    assert report.fp32.threads == report.int8.threads == 1
    assert np.isfinite(report.speedup_mean)
    assert np.isfinite(report.speedup_p99)
    assert "INT8" in report.markdown_table()


def test_benchmark_thread_sweep(onnx_fp32, env):
    sample = bench.make_sample_input(env.obs_dim, seed=0)
    stats = bench.sweep_threads(onnx_fp32, sample=sample, threads=(1, 2), warmup=5, runs=60)

    assert sorted(stats) == [1, 2]
    assert all(s.count == 60 for s in stats.values())
    assert stats[1].threads == 1 and stats[2].threads == 2


def test_benchmark_warns_about_too_few_runs(onnx_fp32, onnx_int8):
    report = bench.compare_benchmark(onnx_fp32, onnx_int8, warmup=5, runs=20)
    assert any("p99" in note for note in report.notes)


def test_benchmark_single_round_reports_no_spread(fp32_engine, env):
    """只有一轮时没有"轮间波动"可言，不能凭一个数编出一个离散度。"""
    sample = bench.make_sample_input(env.obs_dim, seed=0)
    stats = bench.benchmark_engine(fp32_engine, sample, warmup=5, runs=50, repeats=1)

    assert stats.rounds == 1
    assert stats.p50_spread_ms == 0.0


def test_benchmark_repeats_pick_the_median_round(fp32_engine, env):
    """多轮取中位而不是取最快：取最快会把"最好一次"当成"通常情况"。"""
    sample = bench.make_sample_input(env.obs_dim, seed=0)
    stats = bench.benchmark_engine(fp32_engine, sample, warmup=5, runs=80, repeats=5)

    assert stats.rounds == 5
    assert stats.count == 80, "每轮的样本数不变，取中位不是把各轮样本拼起来"
    assert stats.p50_spread_ms >= 0.0


def test_benchmark_rejects_nonpositive_repeats(fp32_engine, env):
    sample = bench.make_sample_input(env.obs_dim, seed=0)
    with pytest.raises(ValueError, match="重复轮数必须为正"):
        bench.benchmark_engine(fp32_engine, sample, warmup=5, runs=10, repeats=0)


def test_representative_round_picks_median_and_reports_spread():
    """直接构造已知分布，验证选轮规则与波动计算。"""

    def make(p50: float, fastest: float | None = None) -> engines.LatencyStats:
        samples = [p50] * 99 + [fastest if fastest is not None else p50]
        return engines.LatencyStats.from_samples(samples, batch_size=1, threads=1, model="m")

    chosen = bench.representative_round([make(0.010), make(0.030, fastest=0.004), make(0.020)])

    assert chosen.p50_ms == pytest.approx(0.020), "取的是中位那一轮，不是最快那轮"
    assert chosen.rounds == 3
    assert chosen.p50_spread_ms == pytest.approx(0.020)


def test_representative_round_takes_min_across_all_rounds():
    """min 跨轮取全局最小，不跟着中位轮走。

    干扰只让测量变慢、不会让它变快，所以跨轮取最小最接近真实计算耗时。
    跟着中位轮走的话，一个没碰上快样本的轮次会把 min 抬高近一倍。
    """

    def make(typical: float, fastest: float) -> engines.LatencyStats:
        return engines.LatencyStats.from_samples(
            [typical] * 99 + [fastest], batch_size=1, threads=1, model="m"
        )

    # 最快的样本出现在"典型值最大"的那一轮里，也就是中位轮之外。
    rounds = [make(0.010, 0.0090), make(0.030, 0.0080), make(0.020, 0.0095)]
    chosen = bench.representative_round(rounds)

    assert chosen.p50_ms == pytest.approx(0.020), "分位数仍取自中位轮"
    assert chosen.min_ms == pytest.approx(0.0080), "min 取的是全局最小"


def test_representative_round_passes_single_round_through():
    stats = engines.LatencyStats.from_samples([0.01] * 50, model="m")
    assert bench.representative_round([stats]) is stats


def test_representative_round_rejects_empty():
    with pytest.raises(ValueError, match="没有可用的测量轮次"):
        bench.representative_round([])


def test_benchmark_flags_indistinguishable_difference(onnx_fp32, onnx_int8):
    """单轮测量下，轮间波动大于两模型差值时必须明说"本机分辨不出来"。

    这是报告里最容易被忽略、也最容易误导人的一句话：一个两位小数的加速比
    看起来像结论，实际上可能只是最后一次测量恰好落在哪。
    """
    report = bench.compare_benchmark(onnx_fp32, onnx_int8, warmup=5, runs=40, repeats=1)
    gap = abs(report.fp32.p50_ms - report.int8.p50_ms)
    spread = max(report.fp32.p50_spread_ms, report.int8.p50_spread_ms)

    flagged = any("分辨不出来" in note for note in report.notes)
    assert flagged == (spread > gap)
    if flagged:
        assert "p50 波动" in report.markdown_table()


def test_benchmark_single_round_has_no_paired_ratios(onnx_fp32, onnx_int8):
    report = bench.compare_benchmark(onnx_fp32, onnx_int8, warmup=5, runs=40, repeats=1)

    assert report.round_ratios == []
    assert report.ratio_spread == 1.0, "没有配对轮次时不该报出一个离散度"
    assert "单轮" in report.summary()


def test_benchmark_paired_rounds_report_a_ratio_per_round(onnx_fp32, onnx_int8):
    """配对测量：每轮两个模型各测一次，得出该轮自己的比值。"""
    repeats = 3
    report = bench.compare_benchmark(onnx_fp32, onnx_int8, warmup=5, runs=60, repeats=repeats)

    assert len(report.round_ratios) == repeats
    assert all(np.isfinite(r) and r > 0 for r in report.round_ratios)
    assert report.fp32.rounds == report.int8.rounds == repeats
    assert report.speedup_mean == pytest.approx(float(np.median(report.round_ratios)))
    assert "配对中位" in report.summary()


def test_benchmark_paired_ratio_spread_is_one_when_ratios_agree():
    """各轮比值一致时离散度为 1，且不该出现"机器状态不稳定"的提示。"""
    fp32 = engines.LatencyStats.from_samples([0.02] * 100, model="fp32")
    int8 = engines.LatencyStats.from_samples([0.01] * 100, model="int8")
    report = bench.BenchmarkReport(fp32=fp32, int8=int8, round_ratios=[2.0, 2.0, 2.0])

    assert report.ratio_spread == pytest.approx(1.0)
    assert report.speedup_mean == pytest.approx(2.0)


def test_benchmark_paired_ratio_spread_grows_with_disagreement():
    fp32 = engines.LatencyStats.from_samples([0.02] * 100, model="fp32")
    int8 = engines.LatencyStats.from_samples([0.01] * 100, model="int8")
    report = bench.BenchmarkReport(fp32=fp32, int8=int8, round_ratios=[1.0, 2.0, 4.0])

    assert report.ratio_spread == pytest.approx(4.0)
    assert report.speedup_mean == pytest.approx(2.0), "取中位，不被极端轮次带偏"


def test_benchmark_paired_median_beats_single_round_ratio():
    """配对中位与"两个中位轮的比值"不是一回事，报告里用的是前者。

    机器状态分段变化时，两个分别取中位的数可能来自不同的状态段，相除就错了。
    """
    fp32 = engines.LatencyStats.from_samples([0.020] * 100, model="fp32")
    int8 = engines.LatencyStats.from_samples([0.010] * 100, model="int8")
    single = bench.BenchmarkReport(fp32=fp32, int8=int8)
    paired = bench.BenchmarkReport(fp32=fp32, int8=int8, round_ratios=[3.0, 3.0])

    assert single.speedup_mean == pytest.approx(2.0)
    assert paired.speedup_mean == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 回测
# ---------------------------------------------------------------------------


def test_backtest_reproduces_initial_states(env, fp32_engine):
    """同种子必须给出逐位相同的初始状态，否则"同一批初始状态"不成立。"""
    first = backtest_mod.rollout(fp32_engine, env, episodes=2, seed=777)
    second = backtest_mod.rollout(fp32_engine, env, episodes=2, seed=777)

    assert np.array_equal(first.observations, second.observations)
    assert first.returns.tolist() == second.returns.tolist()


def test_backtest_compares_fp32_and_int8(fp32_engine, int8_engine, env, deploy_cfg):
    report = backtest_mod.backtest(
        fp32_engine,
        int8_engine,
        env,
        episodes=8,
        seed=12345,
        kl_threshold=deploy_cfg.quant_kl_threshold,
    )

    # 闭环回报：量化后的策略行为略有偏移，但不应崩掉
    assert abs(report.return_delta) < 0.5 * max(abs(report.fp32.return_mean), 1.0)
    assert report.int8.length_mean > 0

    # 配对动作偏差：这一组不涉及轨迹分叉，量的是纯粹的网络输出偏移
    assert report.deviation.mae < 0.02, report.summary()
    assert report.deviation.max_dev < 0.05, report.summary()

    # 动作分布：估计值已扣掉有限样本偏差，阈值取配置里的 0.1
    assert report.kl_mean < deploy_cfg.quant_kl_threshold, report.summary()
    assert report.kl is not None and report.kl.n_samples == report.fp32.num_steps


def test_backtest_report_is_readme_ready(fp32_engine, int8_engine, env):
    report = backtest_mod.backtest(fp32_engine, int8_engine, env, episodes=2, seed=1)
    table = report.markdown_table()

    assert table.startswith("回测条件：")
    assert "| FP32 | INT8 | 变化 |" in table
    assert "动作分布 KL" in table

    payload = report.to_dict()
    assert payload["episodes"] == 2
    assert len(payload["action_mae_per_dim"]) == env.action_dim
    assert len(payload["kl"]["per_dim"]) == env.action_dim


def test_kl_estimator_is_unbiased_on_identical_samples(env, fp32_engine):
    """同一个策略对同一个策略，KL 必须是 0，否则阈值无从谈起。"""
    actions = fp32_engine.infer_batch(
        backtest_mod.collect_observations(env, steps=600, seed=13, engine=fp32_engine)
    )
    estimate = backtest_mod.action_kl_divergence(actions, actions, bins=64)

    assert estimate.mean == pytest.approx(0.0, abs=1e-12)
    assert estimate.bias_mean > 0  # 偏差确实存在，只是被扣掉了


def test_kl_requires_matching_shapes(fp32_engine):
    a = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="形状不一致"):
        backtest_mod.action_kl_divergence(a, np.zeros((10, 3), dtype=np.float32))


def test_backtest_rejects_action_dim_mismatch(fp32_engine, env, tmp_path):
    """动作维度不同就不是同一个策略的两份实现，这种比对没有意义。"""
    wider = RobotSpec(
        name="toy_wide",
        morphology=Morphology.POINT,
        menagerie_dir="",
        controlled_joints=("推进_x", "推进_y", "推进_z"),
        default_angles=(0.0, 0.0, 0.0),
        pd_kp=(1.0, 1.0, 1.0),
        pd_kd=(0.0, 0.0, 0.0),
        torque_limits=(1.0, 1.0, 1.0),
        action_scale=2.0,
    )
    path = tmp_path / "wider.onnx"
    exporter.export_policy(TinyPolicy(env.obs_dim, 3, seed=1), path, spec=wider)

    other_engine = engines.load_engine(path)
    try:
        with pytest.raises(ValueError, match="动作维度不同"):
            backtest_mod.backtest(fp32_engine, other_engine, env, episodes=1)
    finally:
        other_engine.close()


# ---------------------------------------------------------------------------
# PolicyEngine 接口
# ---------------------------------------------------------------------------


def test_engine_shape_contract(fp32_engine, env):
    single = fp32_engine.infer(np.zeros(env.obs_dim, dtype=np.float32))
    assert single.shape == (env.action_dim,)

    batch = fp32_engine.infer(np.zeros((5, env.obs_dim), dtype=np.float32))
    assert batch.shape == (5, env.action_dim)

    with pytest.raises(ValueError, match="观测末维"):
        fp32_engine.infer(np.zeros(env.obs_dim + 1, dtype=np.float32))


def test_engine_dimensions_come_from_the_graph_or_metadata(fp32_engine, env):
    assert fp32_engine.obs_dim == env.obs_dim
    assert fp32_engine.action_dim == env.action_dim
    assert fp32_engine.input_name == "obs"
    assert fp32_engine.output_name == "action"


def test_engine_latency_recording_is_opt_in(onnx_fp32, env):
    """默认不记录：计时本身有开销，端侧监控才需要打开。"""
    quiet = engines.load_engine(onnx_fp32, record_latency=False)
    noisy = engines.load_engine(onnx_fp32, record_latency=True)
    sample = np.zeros(env.obs_dim, dtype=np.float32)
    try:
        quiet.infer(sample)
        assert quiet.latency_stats() is None

        for _ in range(5):
            noisy.infer(sample)
        stats = noisy.latency_stats()
        assert stats is not None and stats.count == 5
        noisy.reset_latency()
        assert noisy.latency_stats() is None
    finally:
        quiet.close()
        noisy.close()


def test_engine_factory_and_registry(onnx_fp32):
    assert "onnxruntime" in engines.available_backends()
    assert isinstance(engines.make_engine("onnxruntime"), engines.OnnxRuntimeEngine)

    with pytest.raises(KeyError, match="未注册的后端"):
        engines.make_engine("rknn")


def test_engine_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="模型文件不存在"):
        engines.load_engine(tmp_path / "nope.onnx")


def test_engine_is_usable_as_a_context_manager(onnx_fp32, env):
    with engines.load_engine(onnx_fp32) as engine:
        assert engine.path == onnx_fp32

    # 退出上下文之后会话已释放，再推理必须明确报错而不是给个野结果
    with pytest.raises(RuntimeError, match="会话还没建立"):
        engine.infer(np.zeros(env.obs_dim, dtype=np.float32))


def test_contract_dimension_matches_engine(env, fp32_engine):
    """契约、图、环境三者的维度必须一致，这是端侧按段切片归因的前提。"""
    assert env.obs_contract.total_dim == fp32_engine.obs_dim == env.obs_dim
    assert env.action_dim == fp32_engine.action_dim == TOY_SPEC.n_dof
