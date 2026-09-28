"""回测脚本：量化前后在同一批初始状态下比行为。

    python scripts/evaluate.py --run-dir runs/g1_velocity
    python scripts/evaluate.py --run-dir runs/g1_velocity --episodes 50 --markdown

这是本项目想补上的那一环。业界普遍停在"导出 ONNX"，而部署真正要回答的是：
量化之后策略变成了什么样——奖励掉了多少，动作偏离多少，偏离集中在哪些维度。

报告里几个数字怎么读：

  平均回报变化   最直观，但注意它混着回合间的随机性。20 个回合、标准差 3 的
                 任务上，±1 以内的差异说明不了问题，报告里给出了标准差。
  动作 MAE       FP32 轨迹上的观测原样喂给两个引擎，得出的偏差不含轨迹分叉，
                 纯粹反映网络输出的差异。这是最能定位问题的指标。
  动作分布 KL    逐维比对两个引擎输出分布的差异。阈值是配置项，超了就说明
                 量化掉点太多，该退回去改量化方式，而不是硬上。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotrl.deploy.engine import load_engine  # noqa: E402
from robotrl.deploy.evaluate import backtest, rollout  # noqa: E402
from scripts._shared import (  # noqa: E402
    add_config_args,
    build_env,
    dump_json,
    ensure_dir,
    header,
    info,
    load_cfg,
    warn,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="量化前后的仿真回测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(p)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--fp32", type=Path, default=None, help="默认 <run-dir>/exported/policy.onnx")
    p.add_argument("--int8", type=Path, default=None, help="默认 <run-dir>/exported/policy_int8_<mode>.onnx")
    p.add_argument("--mode", choices=["dynamic", "static"], default=None,
                   help="要回测哪一份量化模型，默认取配置里的 quant_format")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--kl-threshold", type=float, default=None)
    p.add_argument("--markdown", action="store_true", help="额外输出一份 Markdown 对照表")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.config is None and args.run_dir is not None:
        saved = Path(args.run_dir) / "config.yaml"
        if saved.exists():
            args.config = saved
    cfg = load_cfg(args)

    run_dir = Path(args.run_dir or cfg.train.run_dir)
    exported = run_dir / "exported"
    mode = args.mode or cfg.deploy.quant_format
    fp32 = Path(args.fp32 or exported / "policy.onnx")
    int8 = Path(args.int8 or exported / f"policy_int8_{mode}.onnx")
    episodes = args.episodes or cfg.deploy.backtest_episodes
    kl_threshold = args.kl_threshold or cfg.deploy.quant_kl_threshold

    if not fp32.exists():
        raise SystemExit(f"找不到 {fp32}，先跑 scripts/export.py")

    header(f"回测 {cfg.env.robot}-{cfg.env.task} · {episodes} 回合")
    info(f"起始种子 {cfg.env.seed + 20_000}，两个策略逐回合同种子复位")

    # 回测用的环境必须是新建的。复用训练时的那些实例，等于在有偏的初始状态
    # 分布上比较两个策略，结论不可信。
    env = build_env(cfg)

    if not int8.exists():
        warn(f"找不到 {int8}，只跑 FP32 基线；要出量化对照请先跑 scripts/quantize.py")
        with load_engine(fp32) as engine:
            engine.warmup(10)
            result = rollout(engine, env, episodes=episodes, seed=cfg.env.seed + 20_000)
        header("FP32 基线")
        info(f"平均回报 {result.return_mean:.3f} ± {result.return_std:.3f}")
        info(f"平均回合长度 {result.length_mean:.1f}")
        path = dump_json(ensure_dir(exported) / "rollout_fp32.json", result.to_dict())
        info(f"报告 {path}")
        return 0

    with load_engine(fp32) as fp32_engine, load_engine(int8) as int8_engine:
        fp32_engine.warmup(10)
        int8_engine.warmup(10)
        if fp32_engine.action_dim != int8_engine.action_dim:
            raise SystemExit(
                f"两份模型的动作维度不同（{fp32_engine.action_dim} vs "
                f"{int8_engine.action_dim}），不是同一个策略的两份实现"
            )
        report = backtest(
            fp32_engine,
            int8_engine,
            env,
            episodes=episodes,
            seed=cfg.env.seed + 20_000,
            kl_threshold=kl_threshold,
        )

    header("对照表")
    for line in report.markdown_table().splitlines():
        print("  " + line)

    header("结论")
    info(report.summary())
    if not report.kl_ok:
        warn(
            f"动作分布 KL 超过阈值 {kl_threshold}，量化掉点过多。"
            "可以试：改 --mode static 走静态量化，或退回 FP32"
        )

    payload = report.to_dict()
    payload["robot"] = cfg.env.robot
    payload["task"] = cfg.env.task
    report_path = dump_json(ensure_dir(exported) / "backtest.json", payload)
    info(f"报告 {report_path}")

    if args.markdown:
        md_path = exported / "backtest.md"
        md_path.write_text(
            f"# {cfg.env.robot}-{cfg.env.task} 量化回测\n\n"
            f"控制频率 {1 / cfg.env.control_dt:.0f} Hz，{episodes} 回合，"
            f"起始种子 {cfg.env.seed + 20_000}。\n\n"
            f"{report.markdown_table()}\n\n"
            f"{report.summary()}\n",
            encoding="utf-8",
        )
        info(f"Markdown {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
