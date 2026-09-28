"""端到端冒烟：一条命令跑完整条链路，分钟级出数字。

    训练 → 导出 ONNX → 等价校验 → INT8 量化 → 延迟基准 → 仿真回测

用质点速度跟踪这个不含 MuJoCo 的环境，是为了让"改完某一层之后全链路还通不通"
这件事能在几分钟内得到回答。真实机器人任务训练要几十分钟，不适合当冒烟测试。

这个脚本同时是 scripts/ 下各脚本的组装范例：每个 stage_* 函数对应工具链上的
一个环节，把其中任意一个换成读文件、换成别的算法都不影响其余部分。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from robotrl.algorithms import ActorCritic, make_trainer
from robotrl.assets.spec import RobotSpec, get_spec
from robotrl.configs.schema import Config, PPOConfig
from robotrl.contracts import ObsContract
from robotrl.deploy.benchmark import compare_benchmark
from robotrl.deploy.engine import load_engine
from robotrl.deploy.equivalence import check_equivalence, random_observations
from robotrl.deploy.evaluate import backtest
from robotrl.deploy.export_onnx import export_policy
from robotrl.deploy.quantize import collect_calibration_data, quantize
from robotrl.envs import make
from robotrl.envs.base_env import BaseEnv
from robotrl.utils.torch_runtime import describe

# ---------------------------------------------------------------------------
# 输出格式
# ---------------------------------------------------------------------------

_WIDTH = 78


def section(title: str) -> None:
    print(f"\n{'=' * _WIDTH}\n{title}\n{'=' * _WIDTH}")


def step_line(n: int, total: str, text: str) -> None:
    print(f"  [{n}/{total}] {text}")


# ---------------------------------------------------------------------------
# 各环节
# ---------------------------------------------------------------------------


def build(args: argparse.Namespace) -> tuple[list[BaseEnv], ActorCritic, Config]:
    """构造环境与策略。

    环境数就是并行度。这里用多个环境实例而不是向量化包装，是因为计算量本身
    很小，多进程调度的开销反而更大；换成 MuJoCo 任务时再考虑进程级并行。
    """
    cfg = Config()
    cfg.env.max_episode_steps = args.max_episode_steps
    cfg.env.seed = args.seed

    envs = [make("toy", config=cfg) for _ in range(args.num_envs)]
    first = envs[0]

    policy = ActorCritic(
        first.obs_dim,
        first.action_dim,
        critic_obs_dim=first.critic_obs_dim,
        actor_hidden=(64, 64),
        critic_hidden=(64, 64),
        init_noise_std=1.0,
    )
    step_line(1, "6", f"环境 {args.num_envs} 个，策略 {first.obs_dim}→{first.action_dim} 维")
    return envs, policy, cfg


def stage_train(
    envs: list[BaseEnv],
    policy: ActorCritic,
    cfg: Config,
    args: argparse.Namespace,
) -> Path:
    """训练并落盘。"""
    ppo_cfg = PPOConfig(
        num_steps_per_env=args.steps_per_env,
        num_learning_epochs=5,
        num_mini_batches=4,
    )
    trainer = make_trainer(
        envs,
        policy,
        algo_name="ppo",
        cfg=ppo_cfg,
        run_dir=args.out / "run",
        seed=args.seed,
        verbose=False,
        num_threads=args.num_threads,
    )
    print(f"        {describe()}")

    t0 = time.time()
    trainer.train(args.total_steps)
    elapsed = time.time() - t0

    ckpt = trainer.save(args.out / "run" / "final.pt")
    step_line(
        2,
        "6",
        f"训练 {args.total_steps:,} 步，{elapsed:.1f}s（{args.total_steps / elapsed:,.0f} 步/秒）",
    )
    print(f"        回报走势 {_return_trend(trainer.history)}")
    print(f"        checkpoint → {ckpt}")
    return ckpt


def _return_trend(history: list[Any], window: float = 0.1) -> str:
    """把学习曲线压缩成"开头 → 结尾"两个数。

    只统计跑完了的回合：回合很长时前若干轮可能一条都没跑完，那些轮的
    rollout/return_mean 根本不存在，直接取会得到一堆 nan。
    """
    values = [
        (m.iteration, m.metrics["rollout/return_mean"])
        for m in history
        if m.metrics.get("rollout/episodes", 0.0) > 0
    ]
    if not values:
        return f"{len(history)} 轮内没有回合跑完，策略尚未活到回合上限"

    k = max(1, int(len(values) * window))
    head = float(np.mean([v for _, v in values[:k]]))
    tail = float(np.mean([v for _, v in values[-k:]]))
    arrow = "↑" if tail > head else "↓"
    return f"{head:.3f} → {tail:.3f} {arrow}（{len(values)} 轮有完整回合）"


def stage_export(
    policy: ActorCritic,
    spec: RobotSpec,
    contract: ObsContract,
    args: argparse.Namespace,
) -> Path:
    """导出为自包含 ONNX。"""
    path = args.out / "exported" / "policy.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = export_policy(
        policy,
        path,
        spec=spec,
        contract=contract,
        opset=17,
        extra_metadata={"task": "toy-velocity", "algo": "ppo"},
    )
    step_line(3, "6", f"导出 ONNX {report.size_kb:.1f} KB → {path}")
    if not report.checker_passed:
        raise RuntimeError(f"导出后自检未通过：\n{report.summary()}")
    if report.missing_meta:
        raise RuntimeError(f"metadata 缺少 {report.missing_meta}，端侧无法还原关节目标")
    print("        自检通过：算子集合法、输入输出形状已推断、metadata 完整")
    return path


def stage_validate(
    policy: ActorCritic, onnx_path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """等价校验：导出本身不能引入误差。

    这一步必须排在量化之前。跳过它的话，回测出来的性能下降说不清是量化的
    锅还是导出的锅，调优也就无从下手。
    """
    inputs = {
        "uniform": random_observations(policy.obs_dim, 256, seed=0),
        "normal": random_observations(policy.obs_dim, 256, seed=1, distribution="normal"),
    }
    report = check_equivalence(policy, onnx_path, inputs, tol=1e-5)
    step_line(4, "6", f"等价校验 最大绝对误差 {report.max_abs_err:.3e}")
    if not report.passed:
        raise RuntimeError(f"ONNX 与 PyTorch 输出不一致：{report.summary()}")
    print("        两份实现逐元素一致，导出无损")
    return report.to_dict()


def stage_quantize(
    envs: list[BaseEnv],
    fp32_path: Path,
    args: argparse.Namespace,
) -> Path:
    """量化。静态量化需要校准集，动态量化不需要。"""
    int8_path = args.out / "exported" / f"policy_int8_{args.quant_format}.onnx"
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    calibration = None
    if args.quant_format == "static":
        # 校准集用 FP32 引擎闭环采出来，状态分布才与部署时一致。
        calibration = collect_calibration_data(
            envs[0], steps=args.calibration_steps, seed=args.seed
        )
        print(f"        校准集 {calibration.shape[0]} 条观测")

    report = quantize(fp32_path, int8_path, mode=args.quant_format, calibration=calibration)
    step_line(
        5,
        "6",
        f"{args.quant_format} 量化 {report.size_fp32 / 1024:.1f} → {report.size_int8 / 1024:.1f} KB"
        f"（压缩 {report.compression:.2f}×）",
    )
    if report.quantized_ops:
        ops = "，".join(f"{k} × {v}" for k, v in sorted(report.quantized_ops.items()))
        print(f"        已量化算子 {ops}")
    else:
        print("        警告：没有算子被替换，量化未生效")
    if not report.metadata_ok:
        print(f"        警告：metadata 缺少 {report.missing_metadata}，端侧无法还原关节目标")
    for note in report.notes:
        print(f"        注：{note}")
    return int8_path


def stage_benchmark(
    fp32_path: Path, int8_path: Path, obs_dim: int, args: argparse.Namespace
) -> dict[str, Any]:
    """延迟基准。同一个输入、同样的预热与次数——否则加速比不成立。

    冒烟脚本默认只跑 1 轮：这里要的是"链路通不通"，不是精确的延迟数。
    要看可信的加速比请跑 scripts/benchmark.py，那边默认 5 轮配对测量。
    """
    sample = random_observations(obs_dim, 1, seed=7)
    report = compare_benchmark(
        fp32_path,
        int8_path,
        sample=sample,
        warmup=args.warmup,
        runs=args.runs,
        threads=args.threads,
        repeats=args.repeats,
    )
    step_line(
        6,
        "6",
        f"延迟基准 {args.runs} 次 × {args.repeats} 轮（预热 {args.warmup}，{args.threads} 线程）",
    )
    print()
    for line in report.markdown_table().splitlines():
        print("        " + line)
    for note in report.notes:
        print(f"        注：{note}")
    return report.to_dict()


def stage_backtest(
    fp32_path: Path,
    int8_path: Path,
    cfg: Config,
    contract: ObsContract,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """回测：同一批初始状态下比奖励与动作偏差。

    用新建的环境实例，不复用训练时的那批——训练环境的状态已经被策略的
    探索搅乱了，拿它回测等于用有偏的初始分布比较两个策略。
    """
    env = make("toy", config=cfg)
    with load_engine(fp32_path) as fp32, load_engine(int8_path) as int8:
        report = backtest(
            fp32,
            int8,
            env,
            episodes=args.episodes,
            seed=args.seed + 20_000,
        )
    print()
    for line in report.markdown_table().splitlines():
        print("        " + line)
    print(f"\n        {report.summary()}")
    del contract  # 回测不需要切片归因，保留参数是为了将来加逐段误差分析
    return report.to_dict()


# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="端到端冒烟：训练 → 导出 → 校验 → 量化 → 基准 → 回测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=Path, default=Path("runs/smoke"), help="产物目录")
    p.add_argument("--total-steps", type=int, default=60_000, help="训练总步数")
    p.add_argument("--steps-per-env", type=int, default=24, help="每环境每轮采样步数")
    p.add_argument("--num-envs", type=int, default=8, help="并行环境数")
    p.add_argument("--max-episode-steps", type=int, default=200, help="回合长度上限")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=1, help="PyTorch 算子内并行度")
    p.add_argument("--quant-format", choices=["dynamic", "static"], default="dynamic")
    p.add_argument("--calibration-steps", type=int, default=512, help="静态量化的校准样本数")
    p.add_argument("--episodes", type=int, default=20, help="回测回合数")
    p.add_argument("--warmup", type=int, default=50, help="基准每轮预热次数")
    p.add_argument("--runs", type=int, default=1000, help="基准每轮统计次数")
    p.add_argument("--repeats", type=int, default=1, help="基准配对数，冒烟默认 1 轮")
    p.add_argument("--threads", type=int, default=1, help="推理线程数")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    section("端到端冒烟：toy 速度跟踪")
    envs, policy, cfg = build(args)
    spec = get_spec("toy")
    contract = envs[0].obs_contract
    print(f"        观测契约 {contract}")

    section("1-2  训练")
    stage_train(envs, policy, cfg, args)

    section("3-4  导出与校验")
    fp32_path = stage_export(policy, spec, contract, args)
    equivalence = stage_validate(policy, fp32_path, args)

    section("5  量化")
    int8_path = stage_quantize(envs, fp32_path, args)

    section("6  基准与回测")
    benchmark = stage_benchmark(fp32_path, int8_path, policy.obs_dim, args)

    section("回测：量化前后行为对比")
    backtest_report = stage_backtest(fp32_path, int8_path, cfg, contract, args)

    summary = {
        "total_steps": args.total_steps,
        "num_envs": args.num_envs,
        "quant_format": args.quant_format,
        "benchmark_repeats": args.repeats,
        "equivalence": equivalence,
        "benchmark": benchmark,
        "backtest": backtest_report,
    }
    summary_path = args.out / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * _WIDTH}")
    print(f"全链路通过。汇总 → {summary_path}")
    print(f"{'=' * _WIDTH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
