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
  固定到核   在大小核混合架构上，同一份模型换个逻辑核测，最小延迟能差 4 倍
              （实测 0.0108 到 0.0436 ms）。不绑核的话进程会在核之间迁移，
              待测的 15% 差异整个被淹掉，同一份模型能测出方向相反的两个结论。
              默认先逐个核试一遍，绑到最快的那个，并把核号写进报告。
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from robotrl.deploy.benchmark import compare_benchmark, sweep_threads  # noqa: E402
from robotrl.deploy.engine import load_engine  # noqa: E402
from robotrl.utils.cpu_affinity import (  # noqa: E402
    affinity_supported,
    cpu_count,
    fastest_cpu,
    pinned,
)
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
    p.add_argument(
        "--int8", type=Path, default=None, help="默认 <run-dir>/exported/policy_int8_<mode>.onnx"
    )
    p.add_argument(
        "--mode",
        choices=["dynamic", "static"],
        default=None,
        help="要比对哪一份量化模型，默认取配置里的 quant_format",
    )
    p.add_argument("--runs", type=int, default=1000, help="每轮统计次数")
    p.add_argument("--warmup", type=int, default=50, help="每轮预热次数")
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="重复几轮取中位。亚毫秒网络的单轮测量波动可达一倍，默认 5 轮",
    )
    p.add_argument("--threads", type=int, nargs="+", default=[1], help="要测的线程数，可给多个")
    p.add_argument("--batch", type=int, default=1, help="批大小")
    p.add_argument("--sweep-batch", type=int, nargs="+", default=None, help="额外跑一组批大小扫描")
    p.add_argument(
        "--cpu",
        default="auto",
        help="绑到哪个逻辑核上测。auto=先逐个核试一遍取最快的；none=不绑（结果不可跨次比较）；也可以直接给核号",
    )
    return p.parse_args(argv)


def _resolve_cpu(
    args: argparse.Namespace, probe
) -> tuple[int | None, dict[int, float] | None, bool]:
    """按 --cpu 决定绑哪个核，必要时先做一遍逐核试探。

    Returns:
        (要绑的核号或 None, 各核耗时, 是否成功绑上)。不绑核时前两项都是 None。

    Note:
        指定的核号超出范围时直接报错，不静默退回 auto——静默退回会让人以为
        测的是自己指定的那个核，而报告里的核号又不会说谎，两处对不上时更难查。
    """
    if str(args.cpu).lower() == "none":
        return None, None, False

    if not affinity_supported():
        warn("当前平台不支持设 CPU 亲和性，按不绑核测量：跨次运行的数字不可直接比较")
        return None, None, False

    if str(args.cpu).lower() == "auto":
        total = cpu_count()
        if total == 1:
            return 0, None, True
        info(f"逐核试探（{total} 个逻辑核）：先找出最快的一个，之后的测量都固定在它上面")
        cpu, timings, ok = fastest_cpu(
            probe,
            repeats=7,
            on_probe=lambda core, seconds: info(f"  CPU{core:<3d} {seconds * 1e3:.4f} ms"),
        )
        if not ok:
            warn("绑核失败，按不绑核测量：跨次运行的数字不可直接比较")
            return None, None, False
        spread = max(timings.values()) / min(timings.values())
        info(f"绑到最快的 CPU{cpu}（各核之间最快与最慢相差 {spread:.2f} 倍）")
        if spread > 2.0:
            # 差值这么大说明是大小核混合架构。不写出来，读者会以为 4 倍的差异
            # 是某个模型比另一个快，而不是核本身不同。
            warn(
                f"逻辑核之间的性能相差 {spread:.1f} 倍，是大小核混合架构。绝对延迟是"
                "所选这个核上的值，能迁移的结论只有两个模型的比值"
            )
        return cpu, timings, True

    try:
        cpu = int(args.cpu)
    except ValueError:
        raise SystemExit(f"--cpu 只接受 auto / none / 核号，得到 {args.cpu!r}") from None
    if not 0 <= cpu < cpu_count():
        raise SystemExit(f"--cpu {cpu} 超出范围，本机有 {cpu_count()} 个逻辑核")
    return cpu, None, True


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
        "mode": mode if has_int8 else None,
        "cpu": None,
        "sweeps": {},
    }

    # 逐核试探要真跑一次推理才有意义，所以探针取自 FP32 模型本身。用真实的
    # 算子（而不是空转）是因为"哪个核快"依赖指令集：FP32 的 GEMM 和 INT8 的
    # VNNI 内核在不同微架构上的相对快慢不一样。
    with load_engine(fp32) as probe_engine:
        probe_sample = _sample(obs_dim, 1)
        probe_engine.warmup(args.warmup)

        def probe() -> None:
            probe_engine.infer(probe_sample)

        cpu, timings, pinned_ok = _resolve_cpu(args, probe)

    payload["cpu"] = cpu
    if timings is not None:
        payload["cpu_probe_ms"] = {str(k): v * 1e3 for k, v in timings.items()}
    if not pinned_ok and cpu is not None:
        warn("没能绑核，这次结果只能当参考")

    with pinned(cpu) if cpu is not None else nullcontext():
        _measure_all(args, payload, fp32, int8, has_int8, obs_dim)

    # 同 evaluate.py：dynamic 和 static 是并列的两份结果，文件名必须区分开，
    # 否则后跑的那次静默覆盖先跑的，README 的两列数字就只剩一列有原始报告。
    suffix = mode if has_int8 else "fp32"
    report_path = dump_json(ensure_dir(run_dir / "exported") / f"benchmark_{suffix}.json", payload)
    header("完成")
    info(f"报告 {report_path}")
    if cpu is not None:
        info(f"测量固定在 CPU{cpu}；报告里记了核号，换核或换机器后绝对值不可比，比值可比")
    return 0


def _measure_all(
    args: argparse.Namespace,
    payload: dict,
    fp32: Path,
    int8: Path,
    has_int8: bool,
    obs_dim: int,
) -> None:
    """跑完所有线程数与批大小的测量，结果写进 payload["sweeps"]。"""
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
                info(
                    f"batch {batch:4d} · {threads} 线程：p50 {stats.p50_ms:.3f} ms｜p99 {stats.p99_ms:.3f} ms"
                )
            rows.append(
                {
                    "batch_size": batch,
                    "stats": {t: s.to_dict() for t, s in stats_by_threads.items()},
                }
            )
        payload["sweeps"]["batch_sweep"] = rows


if __name__ == "__main__":
    raise SystemExit(main())
