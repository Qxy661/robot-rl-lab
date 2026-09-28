"""评估工具与配置预设测试。

评估层经手的都是"用来下结论"的数字，所以这里盯的是两条性质：可复现，以及
能被别人读懂。前者靠固定种子与确定性动作保证，后者靠稳定的 JSON 格式和一份
能看出差异的对比表保证。

配置预设的用例是纯防守性的：预设是唯一手写的产物，键名写错、复制粘贴漏改
robot 字段，这类问题在训练跑到一半之前都不会暴露。这里一次性把 configs 目录
下的每个 YAML 都加载一遍，以后新增配置写错了 CI 会直接拦住。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from robotrl.algorithms.base import Policy
from robotrl.configs.schema import Config
from robotrl.envs import make
from robotrl.envs.toy import ToyVelocityEnv
from robotrl.eval.harness import (
    FINAL_SEEDS,
    EvalProtocol,
    evaluate_policy,
    evaluate_with_protocol,
    final_protocol,
    train_protocol,
)
from robotrl.eval.report import (
    SCHEMA,
    VERSION,
    Report,
    compare_reports,
    flatten_metrics,
    load_report,
    save_report,
)
from robotrl.utils.config import load_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "robotrl" / "configs"
ENV_CONFIGS = sorted((CONFIG_DIR / "envs").glob("*.yaml"))
TRAIN_CONFIGS = sorted((CONFIG_DIR / "train").glob("*.yaml"))
ALL_CONFIGS = sorted(CONFIG_DIR.rglob("*.yaml"))


# ---------------------------------------------------------------------------
# 测试用的最小策略
# ---------------------------------------------------------------------------


class _TinyPolicy(Policy):
    """几行就能跑的最小策略。只用来验证评估链路，不指望它学出任何东西。

    权重填成常数而不是随机初始化：评估的可复现性断言不该被"每次构造策略时
    初始化不一样"这种无关变量干扰。
    """

    def __init__(self, obs_dim: int, action_dim: int, *, scale: float = 0.3) -> None:
        super().__init__(obs_dim, action_dim)
        self.linear = nn.Linear(obs_dim, action_dim)
        with torch.no_grad():
            self.linear.weight.fill_(0.1 * scale)
            self.linear.bias.zero_()

    def _policy_forward(self, obs_normalized: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(obs_normalized))


class _SamplingPolicy(_TinyPolicy):
    """在 deterministic=False 时给动作加固定偏移的策略。

    基类的 act() 默认退化成 forward，看不出确定性与否的区别。这个子类把
    采样路径撑开，用来验证 evaluate 的 deterministic 参数真的接到了策略上，
    而不是写了个没人读的开关。
    """

    def act(self, obs: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        action = self.forward(obs)
        if deterministic:
            return action
        return torch.clamp(action + 0.3, -1.0, 1.0)


class _ConstantPolicy(Policy):
    """输出恒定动作，用来把某一维观测逼到量程边缘。"""

    def __init__(self, obs_dim: int, action_dim: int, value: float) -> None:
        super().__init__(obs_dim, action_dim)
        self.register_buffer("value", torch.full((action_dim,), float(value)))

    def _policy_forward(self, obs_normalized: torch.Tensor) -> torch.Tensor:
        return self.value.expand(obs_normalized.shape[0], -1)


@pytest.fixture
def env() -> ToyVelocityEnv:
    return ToyVelocityEnv(seed=0)


@pytest.fixture
def policy(env: ToyVelocityEnv) -> _TinyPolicy:
    return _TinyPolicy(env.obs_dim, env.action_dim)


# ---------------------------------------------------------------------------
# 可复现性
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_results(env, policy):
    """同一个种子跑两遍，回报必须逐个相等。不成立的话两次实验没法比。"""
    first = evaluate_policy(env, policy, 3, seed=7)
    second = evaluate_policy(env, policy, 3, seed=7)

    assert first.returns.tolist() == second.returns.tolist()
    assert first.lengths.tolist() == second.lengths.tolist()
    # 逐段统计也必须完全一致，否则说明有状态漏进了这次的评估之间
    for name, stats in first.segments.items():
        assert stats.mean == second.segments[name].mean
        assert stats.max_abs == second.segments[name].max_abs


def test_different_seeds_explore_different_initial_states(env, policy):
    """换种子就该换一批初始状态，否则"多种子"只是同一批数多跑了几遍。"""
    a = evaluate_policy(env, policy, 3, seed=1)
    b = evaluate_policy(env, policy, 3, seed=2)

    assert a.returns.tolist() != b.returns.tolist()


def test_multi_seed_reports_per_seed_and_spread(env, policy):
    """多种子时既要每组种子的均值，也要种子间的标准差。"""
    result = evaluate_policy(env, policy, 2, seed=(11, 12, 13))

    assert len(result.episodes) == 6
    assert set(result.seed_return_means) == {11, 12, 13}
    assert result.return_seed_std > 0
    assert result.summary()["eval/seeds"] == 3


def test_single_seed_has_no_between_seed_spread(env, policy):
    """一组种子时"种子间标准差"没有定义，必须是 0 而不是 nan。"""
    result = evaluate_policy(env, policy, 2, seed=5)

    assert result.return_seed_std == 0.0


def test_deterministic_flag_reaches_the_policy(env):
    """deterministic=False 必须真的走采样路径，否则评估里的随机性无从谈起。"""
    policy = _SamplingPolicy(env.obs_dim, env.action_dim)

    deterministic = evaluate_policy(env, policy, 2, seed=3, deterministic=True)
    sampled = evaluate_policy(env, policy, 2, seed=3, deterministic=False)

    assert deterministic.returns.tolist() != sampled.returns.tolist()


def test_policy_training_mode_is_restored(env, policy):
    """评估会把策略切到 eval，评估完必须切回去——调用方通常正在训练循环里。"""
    policy.train()
    evaluate_policy(env, policy, 1, seed=0)

    assert policy.training


# ---------------------------------------------------------------------------
# 协议：训练期评估与最终评估
# ---------------------------------------------------------------------------


def test_final_protocol_pins_the_seed_set():
    """最终评估的种子集合写死在模块里，谁跑都是同一批，这是"可比"的前提。"""
    protocol = final_protocol()

    assert protocol.seeds == FINAL_SEEDS
    assert len(protocol.seeds) > 1
    assert protocol.deterministic


def test_train_protocol_is_cheaper_than_final():
    """训练期评估要在循环里反复跑，回合数和种子数都该更省。"""
    cheap, full = train_protocol(n_episodes=5), final_protocol(n_episodes=20)

    assert cheap.total_episodes < full.total_episodes
    assert cheap.name != full.name


def test_protocol_rejects_empty_seed_set():
    with pytest.raises(ValueError, match="种子"):
        EvalProtocol(name="bad", seeds=(), n_episodes=1)
    with pytest.raises(ValueError, match="回合数"):
        EvalProtocol(name="bad", seeds=(1,), n_episodes=0)


def test_evaluate_with_protocol_records_the_protocol(env, policy):
    result = evaluate_with_protocol(env, policy, final_protocol(n_episodes=1))

    assert result.protocol.name == "final"
    assert result.protocol.to_dict() == {
        "name": "final",
        "seeds": list(FINAL_SEEDS),
        "n_episodes_per_seed": 1,
        "deterministic": True,
    }


# ---------------------------------------------------------------------------
# 逐段统计
# ---------------------------------------------------------------------------


def test_segments_follow_the_observation_contract(env, policy):
    """逐段统计的段名与维度必须与观测契约一一对上，否则误差归因会指错地方。"""
    result = evaluate_policy(env, policy, 2, seed=0)

    assert set(result.segments) == set(env.obs_contract.names)
    for name, stats in result.segments.items():
        assert stats.dim == env.obs_contract.dim_of(name)
        assert len(stats.mean) == stats.dim
        assert len(stats.std) == stats.dim
        assert len(stats.max_abs) == stats.dim


def test_segment_stats_are_internally_consistent(env, policy):
    """最大绝对值不可能小于均值，标准差不可能为负。这些数要能被人直接引用。"""
    result = evaluate_policy(env, policy, 3, seed=0)

    for stats in result.segments.values():
        assert all(m >= 0 for m in stats.std)
        assert all(a >= abs(m) - 1e-9 for a, m in zip(stats.max_abs, stats.mean, strict=True))


def test_saturation_responds_to_the_threshold(env, policy):
    """饱和比例是相对阈值算的：阈值放到天上就该是 0，收到地下就该是 1。"""
    loose = evaluate_policy(env, policy, 2, seed=0, saturation_thresholds={"vel": 1e6})
    tight = evaluate_policy(env, policy, 2, seed=0, saturation_thresholds={"vel": -1.0})

    assert all(s == 0.0 for s in loose.segments["vel"].saturation)
    # 初速度恰好是 0，用 0 当阈值会漏掉开头几步，所以取一个必然全中的负值
    assert all(s == 1.0 for s in tight.segments["vel"].saturation)


def test_saturation_detects_a_dimension_pushed_to_the_rail(env):
    """把动作顶到最大，速度这一维必然被推到量程边缘——这正是逐段统计要抓的。

    零动作的策略速度起不来，两者的饱和比例应当拉开差距。
    """
    idle = evaluate_policy(env, _ConstantPolicy(env.obs_dim, env.action_dim, 0.0), 3, seed=1)
    pushed = evaluate_policy(env, _ConstantPolicy(env.obs_dim, env.action_dim, 1.0), 3, seed=1)

    assert max(idle.segments["vel"].saturation) == 0.0
    assert max(pushed.segments["vel"].saturation) > 0.0


def test_segments_without_known_range_skip_saturation(env, policy):
    """量程表里没有的段不硬凑饱和比例，但最大绝对值仍然给出——量程可以自己判，
    最大值是客观的。"""
    result = evaluate_policy(env, policy, 1, seed=0, saturation_thresholds={})

    stats = result.segments["cmd"]
    assert stats.saturation == ()
    assert stats.threshold is None
    assert len(stats.max_abs) == stats.dim


def test_custom_threshold_table_replaces_the_builtin_one(env, policy):
    """换一套量程表就是换掉，不是逐键合并。合并的话，传空表会被静默当成默认值，
    一个参数同时表达两件事，早晚有人踩。"""
    result = evaluate_policy(env, policy, 1, seed=0, saturation_thresholds={"cmd": 1.0})

    assert result.segments["cmd"].threshold == 1.0
    assert result.segments["cmd"].saturation != ()
    assert result.segments["vel"].threshold is None


def test_summary_keys_match_trainer_evaluate(env, policy):
    """评估层的标量键要和 Trainer.evaluate() 对齐，两条路径的数才能并排看。"""
    summary = evaluate_policy(env, policy, 1, seed=0).summary()

    assert {
        "eval/return_mean",
        "eval/return_std",
        "eval/episode_length_mean",
        "eval/episodes",
    } <= set(summary)


def test_episode_results_keep_termination_reason(env, policy):
    """终止与超时分开记：混在一起会把超时也当成失败。"""
    result = evaluate_policy(env, policy, 2, seed=0)

    assert all(isinstance(e.terminated, bool) for e in result.episodes)
    assert 0.0 <= result.terminated_rate <= 1.0


# ---------------------------------------------------------------------------
# JSON 报告
# ---------------------------------------------------------------------------


def _sample_metrics() -> dict:
    return {
        "summary": {"eval/return_mean": 482.25, "eval/return_std": 12.5},
        "segments": {"cmd": {"dim": 2, "mean": [0.0, 0.125], "max_abs": [1.5, 1.5]}},
    }


def test_report_round_trip_is_lossless(tmp_path):
    """存进去再读出来必须一模一样，否则"不用重跑就能看结果"就打了折扣。"""
    metrics = _sample_metrics()
    path = save_report(metrics, tmp_path / "r.json", {"robot": "g1", "commit": "abc123"})
    report = load_report(path)

    assert report.metrics == metrics
    assert report.meta == {"robot": "g1", "commit": "abc123"}
    assert (report.schema, report.version) == (SCHEMA, VERSION)


def test_report_file_is_stable_and_diffable(tmp_path):
    """键排序、缩进、末尾换行——三件事都为了 git diff 里只剩数值变化。"""
    path = save_report(_sample_metrics(), tmp_path / "r.json")
    text = path.read_text(encoding="utf-8")

    assert text.endswith("\n")
    assert '\n  "metrics"' in text
    assert list(json.loads(text)) == sorted(json.loads(text))

    # 同样的输入写两次，字节完全一样
    other = save_report(_sample_metrics(), tmp_path / "r2.json")
    assert path.read_bytes() == other.read_bytes()


def test_report_round_trips_an_eval_result(tmp_path, env, policy):
    """评估结果直接落盘：带 to_dict() 的对象不必调用方自己转。"""
    path = save_report(evaluate_policy(env, policy, 1, seed=0), tmp_path / "r.json")
    report = load_report(path)

    assert report.metrics["protocol"]["name"] == "custom"
    assert report.metrics["summary"]["eval/episodes"] == 1.0


def test_report_refuses_non_finite_metrics(tmp_path):
    """NaN 不是合法 JSON，而且它出现就说明指标算错了，不该静默落盘。"""
    with pytest.raises(ValueError, match="有限值"):
        save_report({"eval/return_mean": float("nan")}, tmp_path / "r.json")


def test_load_report_rejects_foreign_json(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_report(path)


def test_load_report_rejects_other_versions(tmp_path):
    """格式改版后老报告要给明确报错，而不是解析到一半抛 KeyError。"""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema": SCHEMA, "version": 99, "metrics": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="99"):
        load_report(path)


def test_missing_report_reports_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.json"):
        load_report(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# 报告对比
# ---------------------------------------------------------------------------


def test_flatten_metrics_walks_nested_structures():
    flat = flatten_metrics(_sample_metrics())

    assert flat["summary.eval/return_mean"] == 482.25
    assert flat["segments.cmd.mean[1]"] == 0.125


def test_compare_reports_spots_a_change(tmp_path):
    """核心用途：数值变了要一眼看出来，并且知道是变好在变坏。"""
    baseline = save_report(_sample_metrics(), tmp_path / "a.json")
    changed = _sample_metrics()
    changed["summary"]["eval/return_mean"] = 500.0
    candidate = save_report(changed, tmp_path / "b.json")

    table = compare_reports(baseline, candidate, label_a="base", label_b="new")

    assert "summary.eval/return_mean" in table
    assert "500" in table
    assert "+3.68%" in table
    # 没变的项不该占地方
    assert "segments.cmd.mean[1]" not in table
    assert "1 项有变化" in table


def test_compare_reports_says_so_when_nothing_changed(tmp_path):
    a = save_report(_sample_metrics(), tmp_path / "a.json")
    b = save_report(_sample_metrics(), tmp_path / "b.json")

    assert "没有差异" in compare_reports(a, b)


def test_compare_reports_lists_one_sided_metrics(tmp_path):
    """只在一侧出现的项要单独列出来：它往往意味着评估跑了不同的回合数。"""
    a = save_report({"summary": {"eval/return_mean": 1.0}}, tmp_path / "a.json")
    b = save_report(
        {"summary": {"eval/return_mean": 1.0, "eval/terminated_rate": 0.5}},
        tmp_path / "b.json",
    )

    table = compare_reports(a, b, label_a="A", label_b="B")

    assert "只在 B 中出现" in table
    assert "eval/terminated_rate" in table


def test_compare_reports_flags_meta_differences(tmp_path):
    """数值对不上时，先看是不是配置或 commit 就不是同一个。"""
    a = save_report({"summary": {"eval/return_mean": 1.0}}, tmp_path / "a.json", {"robot": "g1"})
    b = save_report({"summary": {"eval/return_mean": 1.0}}, tmp_path / "b.json", {"robot": "go2"})

    assert "robot: g1 → go2" in compare_reports(a, b)


def test_compare_reports_accepts_loaded_reports(tmp_path):
    """路径、字典、Report 对象都该能比，省得每个调用点自己 load。"""
    path = save_report(_sample_metrics(), tmp_path / "a.json")
    loaded = load_report(path)

    assert isinstance(loaded, Report)
    assert "没有差异" in compare_reports(loaded, loaded)
    assert "没有差异" in compare_reports(path, loaded.to_dict())


# ---------------------------------------------------------------------------
# 配置预设
# ---------------------------------------------------------------------------


def _ids(paths: list[Path]) -> list[str]:
    return [p.parent.name + "/" + p.name for p in paths]


def test_config_directory_is_not_empty():
    """配置用例靠 parametrize 展开，目录空掉的话整套会静默通过。"""
    assert ENV_CONFIGS and TRAIN_CONFIGS


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=_ids(ALL_CONFIGS))
def test_every_config_loads(path):
    """configs 下每一份 YAML 都必须能被 load_config 解析。

    这是纯防守：预设是手写的，键名拼错、层级放歪、把字符串写成列表，这些
    问题在训练启动之前都不会暴露。新增配置写错，CI 在这里直接拦住。
    """
    cfg = load_config(path)

    assert isinstance(cfg, Config)
    assert cfg.env.control_dt >= cfg.env.sim_dt
    assert cfg.env.max_episode_steps > 0
    assert cfg.train.num_envs > 0


@pytest.mark.parametrize("path", ENV_CONFIGS, ids=_ids(ENV_CONFIGS))
def test_env_presets_agree_with_their_filenames(path):
    """文件名就是契约：g1_flat 必须是 g1 的平地配置，go2_rough 必须是崎岖地形。

    复制粘贴一份预设去改，最容易漏掉的就是 robot 字段。
    """
    robot, _, terrain = path.stem.partition("_")
    cfg = load_config(path)

    assert cfg.env.robot == robot
    assert cfg.terrain.kind == terrain
    assert cfg.env.task


def _registered_reward_names() -> set[str]:
    from robotrl.envs.managers.reward_manager import list_rewards

    return set(list_rewards())


def _velocity_reward_defaults() -> dict[str, float]:
    """速度跟踪任务的默认权重表。

    它是"配置里没写就用任务默认"这条约定的另一半，所以预设里出现的项应当与它
    不同——写一个一模一样的值只是让人误以为调了参。取不到就退回空表：那份表是
    任务层的实现细节，改名不该让评估层的测试变红。
    """
    from robotrl.envs.tasks import velocity

    return dict(getattr(velocity, "_DEFAULT_REWARD_SCALES", {}))


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=_ids(ALL_CONFIGS))
def test_reward_terms_are_registered(path):
    """YAML 里的奖励项名字必须真的注册过。

    这条不是洁癖：RewardManager 构造时碰到未注册的名字会直接抛 KeyError
    （见 robotrl/envs/managers/reward_manager.py），所以名字写错不是"这一项不
    生效"，而是环境根本建不起来。配置是唯一手写的产物，只能靠这里挡在训练之前。
    """
    cfg = load_config(path)
    unknown = set(cfg.reward.scales) - _registered_reward_names()

    assert not unknown, f"{path.name} 里这些奖励项没有注册：{sorted(unknown)}"


@pytest.mark.parametrize("path", ENV_CONFIGS, ids=_ids(ENV_CONFIGS))
def test_env_presets_only_list_deviations(path):
    """奖励段只写偏离项。列一个与任务默认完全相同的值，是"看着像调了参"的假动作，
    会把后来读配置的人带偏。"""
    cfg = load_config(path)
    defaults = _velocity_reward_defaults()
    redundant = {
        name: scale
        for name, scale in cfg.reward.scales.items()
        if name in defaults and defaults[name] == scale
    }

    assert not redundant, f"{path.name} 里这些权重与任务默认相同，不必写：{redundant}"


@pytest.mark.parametrize(
    ("name", "robot", "algo"),
    [
        ("g1_velocity", "g1", "ppo"),
        ("go2_velocity", "go2", "ppo"),
        ("go2_sac", "go2", "sac"),
    ],
)
def test_train_presets_agree_with_their_filenames(name, robot, algo):
    """文件名同样钉死了形态与算法。go2_sac 忘了写 algo: sac 会在这里挂掉。"""
    cfg = load_config(CONFIG_DIR / "train" / f"{name}.yaml")

    assert cfg.env.robot == robot
    assert cfg.train.algo == algo
    assert cfg.algo_config is (cfg.sac if algo == "sac" else cfg.ppo)
    assert cfg.train.total_timesteps > 0
    assert cfg.train.num_envs > 1


def test_ppo_and_sac_presets_are_a_fair_pair():
    """PPO/SAC 对照的两份预设，除了算法本身其余必须一致，否则比的不是算法。"""
    ppo = load_config(CONFIG_DIR / "train" / "go2_velocity.yaml")
    sac = load_config(CONFIG_DIR / "train" / "go2_sac.yaml")

    assert (ppo.train.algo, sac.train.algo) == ("ppo", "sac")
    assert ppo.env.robot == sac.env.robot
    assert ppo.train.total_timesteps == sac.train.total_timesteps
    assert ppo.reward.scales == sac.reward.scales


def test_rough_variant_actually_changes_the_terrain():
    """go2_rough 相对 go2_flat 的意义就在地形：地形没换，那份预设就是白写的。"""
    flat = load_config(CONFIG_DIR / "envs" / "go2_flat.yaml")
    rough = load_config(CONFIG_DIR / "envs" / "go2_rough.yaml")

    assert rough.terrain.kind == "rough"
    assert rough.terrain.height_range != flat.terrain.height_range
    assert rough.obs.terrain_samples > flat.obs.terrain_samples
    assert rough.env.max_episode_steps > flat.env.max_episode_steps


def test_smoke_config_is_small_and_runs_without_mujoco():
    """冒烟配置要能在没有 MuJoCo 的机器上几十秒跑完，否则 CI 用不了它。"""
    cfg = load_config(CONFIG_DIR / "train" / "smoke.yaml")

    assert cfg.train.total_timesteps <= 10_000
    assert cfg.env.max_episode_steps <= 100

    # toy 环境在注册表里以裸名注册，config 里的 robot 字段直接就是它
    env = make(cfg.env.robot, config=cfg)
    assert env.max_episode_steps == cfg.env.max_episode_steps


def test_smoke_config_supports_a_full_evaluation_round():
    """把评估链路挂到冒烟配置上跑一遍：CI 里唯一一次端到端的评估。"""
    cfg = load_config(CONFIG_DIR / "train" / "smoke.yaml")
    env = make(cfg.env.robot, config=cfg)
    policy = _TinyPolicy(env.obs_dim, env.action_dim)

    result = evaluate_policy(env, policy, 1, seed=0)

    assert result.returns.shape == (1,)
    assert set(result.segments) == set(env.obs_contract.names)
