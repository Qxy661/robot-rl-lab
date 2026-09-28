"""量化脚本：FP32 ONNX → INT8 ONNX。

    python scripts/quantize.py --run-dir runs/g1_velocity
    python scripts/quantize.py --run-dir runs/g1_velocity --mode static --calibration-steps 1024

两种模式的区别，以及各自的适用场景：

    动态（dynamic）  权重离线量化成 INT8，激活值在推理时按批次动态定标。
                     不需要校准数据，开箱即用。代价是多插了一个动态定标算子，
                     在很小的网络（几万参数）上，这个算子的开销可能超过矩阵
                     乘法省下来的部分，量化后反而更慢——这是实测结论，不是
                     理论担忧，所以延迟必须量出来而不是假定。

    静态（static）   权重与激活都离线定标，需要一批校准数据。没有动态定标的
                     开销，延迟稳定更低，代价是校准集与真实部署分布不一致时
                     精度会掉。

选哪个不该靠猜。回测（scripts/evaluate.py）会给出两种模式各自的精度与
延迟数字，看数字决定。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotrl.deploy.engine import load_engine  # noqa: E402
from robotrl.deploy.evaluate import collect_observations  # noqa: E402
from robotrl.deploy.quantize import quantize  # noqa: E402
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
        description="把 FP32 ONNX 量化成 INT8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(p)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument(
        "--input", type=Path, default=None, help="FP32 ONNX，默认 <run-dir>/exported/policy.onnx"
    )
    p.add_argument(
        "--out", type=Path, default=None, help="INT8 ONNX，默认 <run-dir>/exported/policy_int8.onnx"
    )
    p.add_argument("--mode", choices=["dynamic", "static"], default=None)
    p.add_argument("--calibration-steps", type=int, default=1024, help="静态量化的校准样本数")
    p.add_argument("--per-channel", action="store_true", help="权重按通道定标（默认按张量）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.config is None and args.run_dir is not None:
        saved = Path(args.run_dir) / "config.yaml"
        if saved.exists():
            args.config = saved
    cfg = load_cfg(args)

    run_dir = Path(args.run_dir or cfg.train.run_dir)
    mode = args.mode or cfg.deploy.quant_format
    src = Path(args.input or (run_dir / "exported" / "policy.onnx"))
    # 文件名里带模式，两种量化方式各留一份。否则跑第二次会覆盖第一次，
    # 想对比"动态和静态哪个更划算"就得来回重跑。
    dst = Path(args.out or (run_dir / "exported" / f"policy_int8_{mode}.onnx"))
    ensure_dir(dst.parent)

    if not src.exists():
        raise SystemExit(f"找不到 FP32 模型 {src}，先跑 scripts/export.py")

    header(f"量化 {cfg.env.robot}-{cfg.env.task} · {mode}")
    info(f"输入 {src}")

    calibration = None
    if mode == "static":
        # 校准集必须来自策略自己走过的状态。用随机动作采样采到的是"环境能到
        # 的地方"，不是"策略会去的地方"，两者分布差得远，定标出来的范围会偏。
        env = build_env(cfg)
        with load_engine(src) as engine:
            engine.warmup(5)
            calibration = collect_observations(
                env,
                steps=args.calibration_steps,
                seed=cfg.env.seed,
                engine=engine,
            )
        info(f"校准集 {calibration.shape[0]} 条观测（由 FP32 策略闭环采出）")

    kwargs = {}
    if args.per_channel:
        kwargs["per_channel"] = True

    report = quantize(src, dst, mode=mode, calibration=calibration, **kwargs)

    header("量化结果")
    for line in report.summary().splitlines():
        info(line)

    if not report.effective:
        warn("没有任何算子被替换成整数实现，这份模型与 FP32 版本等价但更大")
    if not report.metadata_ok:
        warn(f"metadata 缺失 {report.missing_metadata}，端侧将无法还原关节目标角")

    report_path = dump_json(dst.parent / f"quant_report_{mode}.json", report.to_dict())
    header("完成")
    info(f"模型 {dst}")
    info(f"报告 {report_path}")
    info("下一步：python scripts/benchmark.py 与 python scripts/evaluate.py 量出代价")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
