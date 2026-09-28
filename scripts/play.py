"""回放脚本：在 MuJoCo 窗口里看策略走。

    python scripts/play.py --run-dir runs/g1_velocity
    python scripts/play.py --run-dir runs/g1_velocity --onnx runs/g1_velocity/exported/policy_int8.onnx

两个来源都支持：PyTorch checkpoint（训练刚结束，想看看到底学成什么样）和
ONNX（想确认导出/量化之后的策略在物理里还站得住）。对比着看是有意义的——
回测报告里的动作 MAE 是数字，这里是同一件事的直观版本。

按实时节奏播放：控制周期是 20 ms，那每步就等 20 ms，而不是能跑多快跑多快。
不然人眼看到的是快进，步态是否稳、有没有抖动都判断不出来。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from robotrl.deploy.engine import load_engine  # noqa: E402
from robotrl.deploy.equivalence import policy_forward_numpy  # noqa: E402
from robotrl.deploy.export_onnx import load_policy  # noqa: E402
from scripts._shared import (  # noqa: E402
    add_config_args,
    build_env,
    build_policy,
    header,
    info,
    latest_checkpoint,
    load_cfg,
    warn,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="在 MuJoCo 里回放策略",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(p)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None, help="PyTorch 权重；与 --onnx 二选一")
    p.add_argument("--onnx", type=Path, default=None, help="用 ONNX/INT8 模型回放")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=60.0, help="单个回合的最长播放秒数")
    p.add_argument(
        "--command",
        type=float,
        nargs=3,
        default=None,
        metavar=("VX", "VY", "YAW"),
        help="固定速度指令；不给则由任务自己采样",
    )
    return p.parse_args(argv)


def _load_actuator(args: argparse.Namespace, cfg, env):
    """按来源返回一个 (obs -> action) 的函数。

    两条路径都走确定性输出：PPO 的 forward() 取分布均值，ONNX 导出的本来
    就只有均值。回放要看的是策略学到的东西，不是噪声。
    """
    if args.onnx is not None:
        engine = load_engine(Path(args.onnx))
        engine.warmup(10)
        info(f"模型 {args.onnx}（观测 {engine.obs_dim} → 动作 {engine.action_dim}）")
        return (lambda obs: engine.infer(np.asarray(obs, dtype=np.float32).ravel())), engine

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else latest_checkpoint(Path(args.run_dir or cfg.train.run_dir))
    )
    policy = build_policy(cfg, env)
    load_policy(checkpoint, policy=policy)
    info(f"权重 {checkpoint}")

    def act(obs) -> np.ndarray:
        # 单条观测先补上批轴，求值后再去掉。
        batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return policy_forward_numpy(policy, batch)[0]

    return act, None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.config is None and args.run_dir is not None:
        saved = Path(args.run_dir) / "config.yaml"
        if saved.exists():
            args.config = saved
    cfg = load_cfg(args)
    if args.seed is not None:
        cfg.env.seed = args.seed

    try:
        import mujoco.viewer
    except ImportError:
        raise SystemExit("需要 mujoco 才能开窗口；本环境没装上") from None

    env = build_env(cfg)
    header(f"回放 {cfg.env.robot}-{cfg.env.task}")
    act, engine = _load_actuator(args, cfg, env)
    if args.command is not None:
        env.set_command(args.command)
        info(f"固定指令 vx={args.command[0]} vy={args.command[1]} yaw={args.command[2]}")

    step_dt = cfg.env.control_dt
    total_steps = 0
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for episode in range(args.episodes):
                if not viewer.is_running():
                    break
                obs, _ = env.reset(seed=cfg.env.seed + episode)

                episode_return = 0.0
                steps = int(args.max_seconds / step_dt)
                for _ in range(steps):
                    if not viewer.is_running():
                        break
                    t0 = time.perf_counter()
                    action = np.asarray(act(obs.policy), dtype=np.float64).ravel()
                    result = env.step(action)
                    obs = result.obs
                    episode_return += result.reward
                    total_steps += 1

                    viewer.sync()
                    # 按实时节奏播。推理比控制周期快时补足剩下的时间，慢时
                    # 不额外等待——真实部署里超时就是超时，这里如实反映。
                    remain = step_dt - (time.perf_counter() - t0)
                    if remain > 0:
                        time.sleep(remain)
                    if result.terminated or result.truncated:
                        break
                info(
                    f"回合 {episode + 1}：回报 {episode_return:.2f}，"
                    f"{'提前终止' if result.terminated else '走满'}"
                )
    except KeyboardInterrupt:
        warn("手动中断")
    finally:
        if engine is not None:
            engine.close()
        env.close()

    info(f"共 {total_steps} 步")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
