"""延迟基准：并排测 FP32 与 INT8 的推理延迟。

    python scripts/benchmark.py --run-dir runs/g1_velocity
    python scripts/benchmark.py --run-dir runs/g1_velocity --threads 1 2 4 --batch 1 32

四件事会让延迟数字失去意义，这个脚本都做了处理：

  预热       CPU 频率和缓存要过一段时间才进入稳态，不预热的话前几次测量
              明显偏慢，把 p99 拖高。默认预热 50 次。
  重复轮次   策略网络一次推理只有几十微秒，这个尺度上单轮测量最高能差一倍
              （同一份模型、同一台机器、换个进程就从 0.011 变成 0.019 ms）。
              默认重复 5 轮取中位，并把轮间波动写进报告；波动大于两模型差值
              时会直接提示"本机分辨不出来"。
  p99 样本量 分位数是顺序统计量，1000 次采样里 p99 只由第 10 个最慢的样本
              决定，波动很大。样本太少时会给出提示。
  配对条件   两个模型必须用同一份输入、同样的预热与次数、同样的线程数。
              任何一项不一致，算出来的加速比就不成立。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from robotrl.deploy.benchmark import compare_benchmark, sweep_threads  # noqa: E402
from robotrl.deploy.engine import load_engine  # noqa: E402
from scripts._shared import (  # noqa: E402
    add_config_args,
    dump_json,
    ensure_dir,
    header,
    info,
    load_cfg,
    warn,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="测 FP32 与 INT8 的推理延迟",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(p)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--fp32", type=Path, default=None, help="默认 <run-dir>/exported/policy.onnx")
    p.add_argument("--int8", type=Path, default=None, help="默认 <run-dir>/exported/policy_int8_<mode>.onnx")
    p.add_argument("--mode", choices=["dynamic", "static"], default=None,
                   help="要比对哪一份量化模型，默认取配置里的 quant_format")
    p.add_argument("--runs", type=int, default=1000, help="每轮统计次数")
    p.add_argument("--warmup", type=int, default=50, help="每轮预热次数")
    p.add_argument(
        "--repeats", type=int, default=5,
        help="重复几轮取中位。亚毫秒网络的单轮测量波动可达一倍，默认 5 轮",
    )
    p.add_argument("--threads", type=int, nargs="+", default=[1], help="要测的线程数，可给多个")
    p.add_argument("--batch", type=int, default=1, help="批大小")
    p.add_argument("--sweep-batch", type=int, nargs="+", default=None, help="额外跑一组批大小扫描")
    return p.parse_args(argv)


def _sample(obs_dim: int, batch: int, seed: int = 7) -> np.ndarray:
    """基准输入用固定种子生成，两次运行之间的数字才可比。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(batch, obs_dim)).astype(np.float32)


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
    if not fp32.exists():
        raise SystemExit(f"找不到 {fp32}，先跑 scripts/export.py")
    has_int8 = int8.exists()
    if not has_int8:
        warn(f"找不到 {int8}，只测 FP32；要对比请先跑 scripts/quantize.py --mode {mode}")

    # 观测宽度从模型的 metadata 里读，不要求用户再传一次 --config。
    # 控制回路的输入形状必须与训练时完全一致，这里错了下面所有数字都没意义。
    with load_engine(fp32) as engine:
        obs_dim = int(engine.meta_value("obs_dim", 0))
        action_dim = int(engine.meta_value("action_dim", 0))
        if obs_dim <= 0:
            raise SystemExit("模型 metadata 里没有 obs_dim，无法构造基准输入")
        info(f"模型 {cfg.env.robot}-{cfg.env.task}｜观测 {obs_dim} → 动作 {action_dim}")

    payload = {
        "runs": args.runs,
        "warmup": args.warmup,
        "batch_size": args.batch,
        "obs_dim": obs_dim,
        "sweeps": {},
    }

    for threads in args.threads:
        header(f"延迟对比 · {threads} 线程 · batch {args.batch}")
        sample = _sample(obs_dim, args.batch)
        if has_int8:
            report = compare_benchmark(
                fp32,
                int8,
                sample=sample,
                warmup=args.warmup,
                runs=args.runs,
                threads=threads,
                batch_size=args.batch,
                repeats=args.repeats,
            )
            for line in report.markdown_table().splitlines():
                print("  " + line)
            for note in report.notes:
                warn(note)
            payload["sweeps"][f"threads{threads}_batch{args.batch}"] = report.to_dict()
            info(report.summary())
        else:
            from robotrl.deploy.benchmark import benchmark_path

            stats = benchmark_path(
                fp32,
                sample=sample,
                warmup=args.warmup,
                runs=args.runs,
                threads=threads,
                batch_size=args.batch,
                repeats=args.repeats,
            )
            info(stats.summary())
            payload["sweeps"][f"threads{threads}_batch{args.batch}"] = stats.to_dict()

    if args.sweep_batch:
        header("批大小扫描（FP32）")
        # 边测边落 JSON：LatencyStats 是 dataclass，不是 JSON 原生类型，
        # 攒到最后再统一序列化就会像这样在写文件那一步才炸掉。
        rows = []
        for batch in args.sweep_batch:
            stats_by_threads = sweep_threads(
                fp32,
                sample=_sample(obs_dim, batch),
                threads=tuple(args.threads),
                warmup=args.warmup,
                runs=max(args.runs // 2, 100),
                batch_size=batch,
                repeats=args.repeats,
            )
            for threads, stats in stats_by_threads.items():
                info(f"batch {batch:4d} · {threads} 线程：p50 {stats.p50_ms:.3f} ms｜p99 {stats.p99_ms:.3f} ms")
            rows.append({
                "batch_size": batch,
                "stats": {t: s.to_dict() for t, s in stats_by_threads.items()},
            })
        payload["sweeps"]["batch_sweep"] = rows

    report_path = dump_json(ensure_dir(run_dir / "exported") / "benchmark.json", payload)
    header("完成")
    info(f"报告 {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
