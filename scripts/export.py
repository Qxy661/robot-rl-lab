"""导出脚本：checkpoint → 自包含 ONNX，并当场做等价校验。

    python scripts/export.py --run-dir runs/g1_velocity
    python scripts/export.py --checkpoint runs/g1_velocity/best.pt --robot g1

导出的模型是自包含的：观测归一化的统计量作为 buffer 随 forward 一起进图，
端侧直接喂原始观测即可，不需要复现任何预处理代码。

导出后立刻校验，是因为后面几步都建立在"导出无损"这个前提上。少了这一步，
量化的精度损失和导出的 bug 会混在一起，分不开。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotrl.deploy.equivalence import check_equivalence, observation_sets  # noqa: E402
from robotrl.deploy.evaluate import collect_observations  # noqa: E402
from robotrl.deploy.export_onnx import export_policy, load_policy  # noqa: E402
from scripts._shared import (  # noqa: E402
    add_config_args,
    build_env,
    build_policy,
    dump_json,
    ensure_dir,
    header,
    info,
    latest_checkpoint,
    load_cfg,
    warn,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="把训练好的策略导出为 ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(p)
    p.add_argument("--run-dir", type=Path, default=None, help="训练产物目录，用于定位权重与配置")
    p.add_argument("--checkpoint", type=Path, default=None, help="直接指定权重，优先于 --run-dir")
    p.add_argument("--out", type=Path, default=None, help="输出路径，默认 <run-dir>/exported/policy.onnx")
    p.add_argument("--opset", type=int, default=None)
    p.add_argument("--samples", type=int, default=256, help="每组校验输入的样本数")
    p.add_argument("--rollout-steps", type=int, default=512, help="真实观测采样步数")
    p.add_argument("--skip-equivalence", action="store_true", help="跳过校验（不推荐）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 配置的来源按可靠性排序：显式 --config > 训练时存下的 config.yaml > 默认值。
    # 训练产物目录里那份是最贴合这份权重的，所以它优先于默认值。
    if args.config is None and args.run_dir is not None:
        saved = Path(args.run_dir) / "config.yaml"
        if saved.exists():
            args.config = saved
    cfg = load_cfg(args)

    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
        run_dir = args.run_dir or checkpoint.parent
    else:
        run_dir = Path(args.run_dir or cfg.train.run_dir)
        checkpoint = latest_checkpoint(run_dir)

    if args.opset is not None:
        cfg.deploy.opset = args.opset
    out = Path(args.out or (run_dir / "exported" / "policy.onnx"))
    ensure_dir(out.parent)

    header(f"导出 {cfg.env.robot}-{cfg.env.task} · {cfg.train.algo.upper()}")
    info(f"权重 {checkpoint}")

    env = build_env(cfg)
    policy = build_policy(cfg, env)
    load_policy(checkpoint, policy=policy)
    info(f"结构 观测 {policy.obs_dim} → 动作 {policy.action_dim}")

    report = export_policy(
        policy,
        out,
        spec=env.spec,
        contract=env.obs_contract,
        opset=cfg.deploy.opset,
        extra_metadata={"task": cfg.env.task, "algo": cfg.train.algo},
    )
    header("导出结果")
    for line in report.summary().splitlines():
        info(line)
    if not report.checker_passed:
        raise SystemExit("ONNX checker 未通过，模型不可用")

    payload = {"export": report.to_dict()}

    if not args.skip_equivalence:
        header("等价校验")
        groups = observation_sets(policy.obs_dim, args.samples, seed=cfg.env.seed)
        # 随机输入覆盖数值边界，真实观测覆盖策略真正会走到的状态。两组都要，
        # 只测随机会漏掉策略自己引出的分布，只测真实观测会漏掉裁剪边界。
        groups["rollout"] = collect_observations(
            env, steps=args.rollout_steps, seed=cfg.env.seed, engine=None
        )
        equivalence = check_equivalence(policy, out, groups, tol=cfg.deploy.equivalence_tol)
        for line in equivalence.summary().splitlines():
            info(line)
        payload["equivalence"] = equivalence.to_dict()
        if not equivalence.passed:
            raise SystemExit(
                f"导出引入的误差超过 {cfg.deploy.equivalence_tol}，先修导出再看量化"
            )
    else:
        warn("跳过了等价校验；后续量化的精度结论将无法与导出误差分离")

    report_path = dump_json(out.parent / "export_report.json", payload)
    header("完成")
    info(f"模型 {out}")
    info(f"报告 {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
