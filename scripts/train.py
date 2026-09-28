"""训练脚本：读配置，跑训练，落盘权重与曲线。

    python scripts/train.py --config robotrl/configs/train/g1_velocity.yaml
    python scripts/train.py --config <同上> --set train.num_envs=8 ppo.lr_actor=1e-4

配置里的 num_envs 决定并行度。纯 CPU 上的并行只能靠多开进程——MuJoCo 的
单步无法多线程化（详见 robotrl/utils/torch_runtime.py 与 vector_env.py）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 允许直接 `python scripts/train.py` 运行而不必先 pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotrl.utils.config import config_to_dict, dump_yaml  # noqa: E402
from scripts._shared import (  # noqa: E402
    add_config_args,
    build_envs,
    build_policy,
    build_trainer,
    dump_json,
    ensure_dir,
    header,
    info,
    load_cfg,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="训练一个运动策略",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(p)
    p.add_argument("--total-steps", type=int, default=None, help="覆写 train.total_timesteps")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true", help="不打印每轮指标")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_cfg(args)

    if args.total_steps is not None:
        cfg.train.total_timesteps = args.total_steps
    if args.seed is not None:
        cfg.env.seed = args.seed
    run_dir = ensure_dir(args.run_dir or cfg.train.run_dir)

    header(f"训练 {cfg.env.robot}-{cfg.env.task} · {cfg.train.algo.upper()}")
    info(
        f"总步数 {cfg.train.total_timesteps:,}｜并行 {cfg.train.num_envs} 环境（{cfg.train.vec_backend}）"
    )
    info(
        f"控制 {1 / cfg.env.control_dt:.0f} Hz｜仿真 {1 / cfg.env.sim_dt:.0f} Hz｜回合上限 {cfg.env.max_episode_steps} 步"
    )
    info(f"产物目录 {run_dir}")

    vec = build_envs(cfg)
    policy = build_policy(cfg, vec)
    info(f"观测 {vec.obs_dim} 维（critic {vec.critic_obs_dim}）→ 动作 {vec.action_dim} 维")
    if hasattr(vec, "num_workers"):
        info(f"多进程推进：{vec.num_workers} 个进程持 {vec.num_envs} 个环境")

    trainer = build_trainer(cfg, vec, policy, run_dir=run_dir, verbose=not args.quiet)
    info(
        f"PyTorch 线程 {trainer.torch_threads['num_threads']}（interop {trainer.torch_threads['interop_threads']}）"
    )

    header("训练中")
    t0 = time.perf_counter()
    history = trainer.train(cfg.train.total_timesteps)
    elapsed = time.perf_counter() - t0

    final = trainer.save(run_dir / "final.pt")
    curve = [m.to_dict() for m in history]
    curve_path = dump_json(run_dir / "metrics.json", curve)
    # 把这份配置一起存下来。导出、回放、回测都要知道形态、观测契约和动作
    # 尺度，让它们从产物目录里读，而不是靠人记住当初用的是哪份 YAML。
    cfg_path = dump_yaml(config_to_dict(cfg), run_dir / "config.yaml")

    header("完成")
    info(f"权重 {final}")
    info(f"曲线 {curve_path}")
    info(f"配置 {cfg_path}")
    info(f"{len(curve)} 轮，{elapsed:.1f}s（{cfg.train.total_timesteps / elapsed:,.0f} 步/秒）")
    info(f"回报走势 {_trend(curve)}")
    return 0


def _trend(curve: list[dict], window: float = 0.1) -> str:
    """把学习曲线压成"开头 → 结尾"。

    只统计有回合跑完的轮次：回合很长时前若干轮可能一条都没结束，那些轮里
    根本没有 rollout/return_mean 这个键。
    """
    values = [row["rollout/return_mean"] for row in curve if row.get("rollout/episodes", 0.0) > 0]
    if not values:
        return f"{len(curve)} 轮内没有回合跑完，策略尚未活到回合上限"

    k = max(1, int(len(values) * window))
    head = sum(values[:k]) / k
    tail = sum(values[-k:]) / k
    arrow = "↑" if tail > head else "↓"
    return f"{head:.3f} → {tail:.3f} {arrow}"


if __name__ == "__main__":
    raise SystemExit(main())
