"""回放脚本：在 MuJoCo 窗口里看策略走，或者录成 GIF。

    python scripts/play.py --run-dir runs/g1_velocity
    python scripts/play.py --run-dir runs/g1_velocity --onnx runs/g1_velocity/exported/policy_int8.onnx
    python scripts/play.py --run-dir runs/g1_velocity --record docs/assets/g1.gif

两个来源都支持：PyTorch checkpoint（训练刚结束，想看看到底学成什么样）和
ONNX（想确认导出/量化之后的策略在物理里还站得住）。对比着看是有意义的——
回测报告里的动作 MAE 是数字，这里是同一件事的直观版本。

两种播放方式：

- 默认开窗口，按实时节奏播。控制周期 20 ms 就每步等 20 ms，而不是能跑多快
  跑多快——不然人眼看到的是快进，步态稳不稳、有没有抖动都判断不出来。
- `--record` 走离屏渲染，不开窗口，也不按实时节奏（GIF 自带帧率，等待只会让
  录制变慢）。这条路径不依赖显示器，所以无头的机器上照样能出素材。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
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

#: 录制画面的长宽比。机器人是竖着的，16:9 会浪费两侧，4:3 更贴。
_RECORD_ASPECT = 3 / 4

#: 相机距离相对站高的倍数。写死一个米数只能对一种形态好使：G1 站着约 0.79 m，
#: Go2 只有约 0.32 m，同一个 3 m 机位拍 Go2 会拍到一片空地。按站高缩放之后，
#: 三个形态的画面占比才一致。
_CAMERA_DISTANCE_RATIO = 2.8


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

    record = p.add_argument_group(
        "录制",
        '给了 --record 就走离屏渲染，不开窗口。需要 Pillow：pip install -e ".[record]"',
    )
    record.add_argument("--record", type=Path, default=None, help="输出 GIF 路径")
    record.add_argument("--record-fps", type=int, default=25, help="GIF 播放帧率")
    record.add_argument(
        "--record-seconds",
        type=float,
        default=4.0,
        help="录制的总时长上限；录满就停，不等回合跑完",
    )
    record.add_argument("--record-width", type=int, default=480, help="画面宽度，高度按 4:3 推出")
    record.add_argument("--record-camera-azimuth", type=float, default=90.0, help="相机方位角")
    record.add_argument("--record-camera-elevation", type=float, default=-8.0, help="相机俯仰角")
    record.add_argument(
        "--record-camera-distance",
        type=float,
        default=None,
        help="相机到机身的距离；不给则按站高的固定倍数推算，见 _CAMERA_DISTANCE_RATIO",
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


def _setup_recording(args: argparse.Namespace, cfg, env):
    """建离屏渲染器、相机和录制器。

    相机锁在机身上而不是架在固定位置：策略随时会走开甚至摔倒，固定机位拍到
    一半画面里就没东西了。
    """
    import mujoco

    from robotrl.utils.record import GifRecorder

    # 控制频率和想要的帧率不一定整除，只能取最近的整数个控制步一帧。这里
    # 把实际能达到的帧率算出来写进 GIF，而不是拿请求值充数——否则录出来的
    # 文件会以 25 fps 播放，却标着 30 fps，动作整体慢一拍。
    stride = max(1, round(1.0 / (cfg.env.control_dt * args.record_fps)))
    actual_fps = 1.0 / (cfg.env.control_dt * stride)
    if abs(actual_fps - args.record_fps) > 0.5:
        warn(
            f"控制周期 {cfg.env.control_dt * 1000:.0f} ms 凑不出 {args.record_fps} fps，"
            f"按 {actual_fps:.1f} fps 录（每 {stride} 个控制步一帧）"
        )
    fps = max(1, round(actual_fps))

    recorder = GifRecorder(
        args.record,
        fps=fps,
        width=args.record_width,
        max_frames=max(1, round(args.record_seconds * fps)),
    )
    height = max(1, round(args.record_width * _RECORD_ASPECT))
    renderer = mujoco.Renderer(env.model, height=height, width=args.record_width)

    distance = args.record_camera_distance
    if distance is None:
        # base_height_target 就是复位时的站高，是"这个形态多大"最直接的量。
        distance = _CAMERA_DISTANCE_RATIO * env.base_height_target

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.base_body_id
    camera.distance = distance
    camera.azimuth = args.record_camera_azimuth
    camera.elevation = args.record_camera_elevation

    info(f"录制 {args.record}（{args.record_width}×{height}，{fps} fps，每 {stride} 步一帧）")
    return recorder, renderer, camera, stride


def _play(
    args: argparse.Namespace,
    cfg,
    env,
    act,
    *,
    can_continue: Callable[[], bool],
    after_step: Callable[[int], None],
    pace: bool,
) -> int:
    """跑完若干回合，返回总控制步数。

    开窗口和录制共用这一段：两者的区别只有"每步之后做什么"（同步窗口还是
    渲染存帧）和"要不要等够控制周期"，所以差别被收进两个回调和一个开关，
    而不是复制两遍循环——复制出去的版本迟早只有一边会被改。
    """
    step_dt = cfg.env.control_dt
    total_steps = 0
    for episode in range(args.episodes):
        if not can_continue():
            break
        obs, _ = env.reset(seed=cfg.env.seed + episode)

        episode_return = 0.0
        steps = int(args.max_seconds / step_dt)
        for _ in range(steps):
            if not can_continue():
                return total_steps

            t0 = time.perf_counter()
            action = np.asarray(act(obs.policy), dtype=np.float64).ravel()
            result = env.step(action)
            obs = result.obs
            episode_return += result.reward
            total_steps += 1

            after_step(total_steps)

            # 按实时节奏播。推理比控制周期快时补足剩下的时间，慢时不额外
            # 等待——真实部署里超时就是超时，这里如实反映。录制不进这个
            # 分支：GIF 自带帧率，等待只会让录制变慢。
            if pace:
                remain = step_dt - (time.perf_counter() - t0)
                if remain > 0:
                    time.sleep(remain)
            if result.terminated or result.truncated:
                break
        info(
            f"回合 {episode + 1}：回报 {episode_return:.2f}，"
            f"{'提前终止' if result.terminated else '走满'}"
        )
    return total_steps


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.config is None and args.run_dir is not None:
        saved = Path(args.run_dir) / "config.yaml"
        if saved.exists():
            args.config = saved
    cfg = load_cfg(args)
    if args.seed is not None:
        cfg.env.seed = args.seed

    env = build_env(cfg)
    header(f"回放 {cfg.env.robot}-{cfg.env.task}")
    act, engine = _load_actuator(args, cfg, env)
    if args.command is not None:
        env.set_command(args.command)
        info(f"固定指令 vx={args.command[0]} vy={args.command[1]} yaw={args.command[2]}")

    total_steps = 0
    try:
        if args.record is not None:
            # 录制这条路不 import mujoco.viewer：窗口模块在没有显示器的机器上
            # 可能装不上或起不来，而离屏渲染恰恰是那种环境下唯一能出画面的方式。
            recorder, renderer, camera, stride = _setup_recording(args, cfg, env)

            def after_step(step: int) -> None:
                if step % stride:
                    return
                renderer.update_scene(env.data, camera=camera)
                recorder.add(renderer.render())

            try:
                total_steps = _play(
                    args,
                    cfg,
                    env,
                    act,
                    can_continue=lambda: not recorder.full,
                    after_step=after_step,
                    pace=False,
                )
            finally:
                # 中断也要把已经录到的帧写出来：崩之前那几帧往往正是要看的。
                if recorder.frame_count:
                    path = recorder.close()
                    info(f"已写出 {path}（{path.stat().st_size / 1024:.0f} KB）")
                else:
                    warn("一帧都没录到，没有写出文件")
        else:
            try:
                import mujoco.viewer
            except ImportError:
                raise SystemExit("需要 mujoco 才能开窗口；本环境没装上") from None

            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                total_steps = _play(
                    args,
                    cfg,
                    env,
                    act,
                    can_continue=viewer.is_running,
                    after_step=lambda _: viewer.sync(),
                    pace=True,
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
