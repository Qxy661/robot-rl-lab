"""延迟基准：FP32 与 INT8 并排测。

三件容易做错的事，这里都按规矩来：

**预热**。CPU 平时跑在省电频率上，负载上来之后要过几十毫秒才升到稳态；
onnxruntime 首轮推理还要建内存池、为每个算子选一次内核。这些开销只发生
一次，但会落在最早的几次测量里。而延迟基准恰恰要报 p99——不预热的话，
p99 就不是"最慢的一次推理"，而是"启动开销有多大"。所以预热次数是可配的，
默认 50 次，且预热样本一律不进统计。

**线程数**。intra-op 并行是算子内部的（矩阵乘分块），inter-op 是算子之间的。
连续控制的策略网络是一条近乎串行的依赖链，算子之间没什么可并行的。所以：

    threads=1   测的是纯算子延迟，对应"一次推理要多久、控制周期够不够"
    threads>1   测的是吞吐，对应"一块板子能不能同时跑多路策略"

两个数不是一回事，报告里会带上线程数，跨实验比较时线程数必须一致。

**统计项，以及该信哪一个**。只有均值不够用：均值达标但偶发一次长尾，
机器人就会在那一拍踉跄。所以 min / mean / p50 / p90 / p95 / p99 / std 全报。
样本数也要够——1000 次测量下 p99 才有约 10 个样本落在尾部，样本再少的话
p99 就等于最大值。

但要拿两个模型比算力，**只能看 min**。干扰只会让一次测量变慢，不会让它变快，
所以最小值是对"这份计算真正需要多久"最干净的估计。其余几列反映的是机器有
多吵，不是模型有多快——同一台机器上实测 p50 能在 0.010 和 0.032 ms 之间跳，
而 min 六轮内稳定到四位有效数字。分位数那几列留给"最坏情况够不够用"的判断。

**重复轮次**。策略网络只有几万参数，一次推理在几十微秒量级，这个尺度上
频率调节、页错误、后台进程都能改变结果。所以 `repeats` 把整轮流程重复若干次，
并且**交替**测两个模型（FP32 → INT8 → FP32 → …），而不是先把 FP32 测完再测
INT8：串行测量下两个模型落在不同的时间窗口里，机器状态早就变了，算出来的
加速比可能整个是假的。各轮比值的中位、以及轮间比值的最大/最小之比都会写进
报告；波动过大时报告会直接说"只有 min 的比较还站得住"。

计时本身有开销（perf_counter 约 0.1 微秒），所以小于 10 微秒的模型测出来
的绝对值要把这个下限记在心里。这种场景建议加大 batch 或直接看相对加速比。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from robotrl.deploy.engine import LatencyStats, PolicyEngine, load_engine

#: 统计 p99 所需的样本量下限。低于这个数就给出提示，避免把 max 当成 p99 用。
_P99_MIN_RUNS = 100

#: 各轮加速比的最大/最小之比超过这个值，就认为机器状态不稳定、结论不可用。
#: 1.3 是经验值：同一台机器上稳定测量的多轮比值通常在 1.1 以内，而受干扰的
#: 测量动辄到 2 以上——中间留一段余量，免得把正常抖动误判成不可用。
_RATIO_SPREAD_LIMIT = 1.3


# ---------------------------------------------------------------------------
# 输入与单引擎测量
# ---------------------------------------------------------------------------


def make_sample_input(
    obs_dim: int,
    *,
    batch_size: int = 1,
    seed: int = 0,
    scale: float = 1.0,
) -> np.ndarray:
    """造一个基准用的输入张量 (batch_size, obs_dim)。

    用真实观测更好（数值分布真实，某些算子会走不同的分支），但基准关心的是
    稳定的计算量，随机输入足够，而且能脱离环境独立跑。
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(batch_size, obs_dim)).astype(np.float32)


def benchmark_engine(
    engine: PolicyEngine,
    sample: np.ndarray | None = None,
    *,
    warmup: int = 50,
    runs: int = 1000,
    batch_size: int = 1,
    seed: int = 0,
    repeats: int = 1,
) -> LatencyStats:
    """测一个已载入引擎的延迟分布。

    逐次计时而不是"总时间除以次数"：只有逐次样本才能给出分位数。代价是
    每次都要调一次 perf_counter。

    Args:
        repeats: 把"预热 + 统计"整个流程重复几轮，取其中位那一轮。
            单轮测量在这个量级上不够用——实测同一份 FP32 模型在同一进程里
            连测四次，p50 分别是 0.0126 / 0.0112 / 0.0113 / 0.0114 ms，
            但在另一个进程里第一轮测出 0.019 ms。亚毫秒级的推理容易被
            频率调节、页错误、后台进程打断，单轮结果最高能差一倍。
            多轮取中位，并把轮间波动一并报出来，读者才知道该信到哪一位。
    """
    if runs <= 0:
        raise ValueError(f"统计次数必须为正，得到 {runs}")
    if warmup < 0:
        raise ValueError(f"预热次数不能为负，得到 {warmup}")
    if repeats <= 0:
        raise ValueError(f"重复轮数必须为正，得到 {repeats}")
    if sample is None:
        if not engine.obs_dim:
            raise ValueError("引擎没有 obs_dim，必须显式给出 sample")
        sample = make_sample_input(engine.obs_dim, batch_size=batch_size, seed=seed)

    batch = np.asarray(sample, dtype=np.float32)
    if batch.ndim == 1:
        batch = batch[None, :]

    rounds: list[LatencyStats] = []
    for _ in range(repeats):
        engine.warmup(warmup)

        samples = np.empty(runs, dtype=np.float64)
        for i in range(runs):
            start = time.perf_counter()
            engine.infer_batch(batch)
            samples[i] = (time.perf_counter() - start) * 1e3

        rounds.append(
            LatencyStats.from_samples(
                samples,
                batch_size=int(batch.shape[0]),
                threads=engine.threads,
                backend=engine.backend,
                model=Path(engine.path).stem if engine.path else "",
            )
        )

    return representative_round(rounds)


def representative_round(rounds: Sequence[LatencyStats]) -> LatencyStats:
    """把多轮测量压成一个报告值。

    分位数取自"按 p50 排在中位的那一轮"，但 **min 取所有轮的全局最小**。
    两者取自不同的地方，是有意的：

    - 分位数（p50/p90/p99）描述的是"这台机器通常有多吵"，取中位轮是为了
      不被最吵或最闲的一轮带偏；
    - min 描述的是"这份计算真正需要多久"，它应该跨轮取最小——干扰只加不减，
      所以全局最小比任何单轮的最小都更接近真值。若跟着中位轮走，一个没碰上
      快样本的轮次会把 min 抬高近一倍（实测 0.0117 → 0.0181 ms）。

    轮间 p50 的最大差值一并带上：差值比两模型的差异还大时，结论就是
    "本机分辨不了"，这句话必须出现在报告里，而不是留给人猜。
    """
    if not rounds:
        raise ValueError("没有可用的测量轮次")
    if len(rounds) == 1:
        return rounds[0]

    ordered = sorted(rounds, key=lambda s: s.p50_ms)
    chosen = ordered[len(ordered) // 2]
    return replace(
        chosen,
        min_ms=min(r.min_ms for r in rounds),
        rounds=len(ordered),
        p50_spread_ms=ordered[-1].p50_ms - ordered[0].p50_ms,
    )


def benchmark_path(
    path: str | Path,
    *,
    sample: np.ndarray | None = None,
    warmup: int = 50,
    runs: int = 1000,
    threads: int = 1,
    batch_size: int = 1,
    backend: str = "onnxruntime",
    repeats: int = 1,
    **engine_options: Any,
) -> LatencyStats:
    """按路径测延迟。内部构造引擎、测完即释放。"""
    engine = load_engine(path, backend, threads=threads, **engine_options)
    try:
        return benchmark_engine(
            engine,
            sample,
            warmup=warmup,
            runs=runs,
            batch_size=batch_size,
            repeats=repeats,
        )
    finally:
        engine.close()


def sweep_threads(
    path: str | Path,
    *,
    sample: np.ndarray | None = None,
    threads: Sequence[int] = (1, 2, 4),
    warmup: int = 20,
    runs: int = 300,
    backend: str = "onnxruntime",
    repeats: int = 1,
    **engine_options: Any,
) -> dict[int, LatencyStats]:
    """同一模型在不同线程数下的延迟/吞吐。

    单线程那一点是延迟下限，往后每加一倍线程，吞吐涨不上去就说明这个模型
    太小、并行化的调度开销已经盖过收益——端侧部署时这类结论直接影响要不要
    给多个策略共享一块板子。
    """
    out: dict[int, LatencyStats] = {}
    for n in threads:
        out[int(n)] = benchmark_path(
            path,
            sample=sample,
            warmup=warmup,
            runs=runs,
            threads=int(n),
            backend=backend,
            repeats=repeats,
            **engine_options,
        )
    return out


# ---------------------------------------------------------------------------
# 对照报告
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    """FP32 与 INT8 的延迟对照。"""

    fp32: LatencyStats
    int8: LatencyStats
    notes: list[str] = field(default_factory=list)
    #: 每轮单独算出的加速比（FP32 均值 / INT8 均值）。重复轮数 > 1 时才有值。
    #: 它比"两个中位轮的比值"可信：同一轮里两个模型面对的是同一份机器状态，
    #: 频率、缓存、后台干扰都被这层配对消掉了。
    round_ratios: list[float] = field(default_factory=list)

    @property
    def speedup_mean(self) -> float:
        """按均值算的加速比。

        配对测量过就取各轮比值的中位——不取中位轮的比值。这两者差别不小：
        机器状态是分段的（一会儿快一会儿慢），把两个分别取中位的数相除，
        分子分母可能来自不同的状态段。
        """
        if self.round_ratios:
            return float(np.median(self.round_ratios))
        return self.int8.speedup_vs(self.fp32)

    @property
    def ratio_spread(self) -> float:
        """各轮加速比的最大/最小之比。1.0 表示各轮结论一致。"""
        if len(self.round_ratios) < 2:
            return 1.0
        low, high = min(self.round_ratios), max(self.round_ratios)
        if low <= 0:
            return float("inf")
        return high / low

    @property
    def speedup_min(self) -> float:
        """按最小值算的加速比——亚毫秒基准里唯一能复现的结论。

        干扰只会让一次测量变慢，不会让它变快（CPU 不会超频到设计频率以上）。
        所以最小值是对"这份计算真正需要多久"最干净的估计：一趟测量里哪怕
        九成样本被系统打断，最小值仍然落在真实值上。

        实测：同一台机器上 FP32 的 min 在六轮交替测量里全是 0.0100 ms，
        动态量化 0.0116–0.0117，静态量化 0.0127–0.0128——四位有效数字稳定。
        同一批测量的 p50 却在 0.010 到 0.032 之间跳。要下结论就看这个数。
        """
        if self.int8.min_ms <= 0:
            return float("nan")
        return self.fp32.min_ms / self.int8.min_ms

    @property
    def speedup_p99(self) -> float:
        """按 p99 算的加速比。尾部延迟才决定控制回路能不能稳定跟上。

        注意它同样受干扰污染：p99 是顺序统计量，一趟测量里混进几次系统
        打断就能把它抬高一倍。控制回路选型时它的意义在，但别拿它当算力结论。
        """
        if self.int8.p99_ms <= 0:
            return float("nan")
        return self.fp32.p99_ms / self.int8.p99_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "fp32": self.fp32.to_dict(),
            "int8": self.int8.to_dict(),
            "speedup_min": self.speedup_min,
            "speedup_mean": self.speedup_mean,
            "speedup_p99": self.speedup_p99,
            "round_ratios": list(self.round_ratios),
            "ratio_spread": self.ratio_spread,
            "notes": list(self.notes),
        }

    def markdown_table(self) -> str:
        """README 用的对照表。CPU 上的数字只在本机、本线程数下成立。"""
        rows = [
            ("FP32", self.fp32),
            ("INT8", self.int8),
        ]
        # min 排在最前：这个尺度上它是唯一能复现的统计量，其余几列是给
        # "最坏情况够不够用"做判断的，不该拿来做算力对比。
        lines = [
            "| 模型 | 线程 | batch | min (ms) | mean (ms) | p50 | p90 | p95 | p99 | std | 吞吐 (/s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, stats in rows:
            lines.append(
                f"| {name} | {stats.threads} | {stats.batch_size} | {stats.min_ms:.4f} | "
                f"{stats.mean_ms:.3f} | {stats.p50_ms:.3f} | {stats.p90_ms:.3f} | "
                f"{stats.p95_ms:.3f} | {stats.p99_ms:.3f} | {stats.std_ms:.3f} | "
                f"{stats.throughput:.1f} |"
            )
        lines.append(
            f"| 加速比 | {self.int8.threads} | {self.int8.batch_size} | "
            f"{self.speedup_min:.2f}× | {self.speedup_mean:.2f}× | — | — | — | "
            f"{self.speedup_p99:.2f}× | — | — |"
        )
        # 波动单独占一行而不是塞进表格：它是"这两个数能信到几位"的说明，
        # 不是又一个被测对象，混在数据行里容易被当成模型指标去比较。
        if self.fp32.rounds > 1 or self.int8.rounds > 1:
            lines.append(
                f"| 轮间 p50 波动 | — | — | — | — | "
                f"{max(self.fp32.p50_spread_ms, self.int8.p50_spread_ms):.4f} | — | — | — | — | — |"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [self.fp32.summary(), self.int8.summary()]
        label = "配对中位" if self.round_ratios else "单轮"
        lines.append(
            f"加速比    min {self.speedup_min:.2f}×（{self.fp32.min_ms:.4f} vs "
            f"{self.int8.min_ms:.4f} ms，看这个），mean {self.speedup_mean:.2f}×（{label}），"
            f"p99 {self.speedup_p99:.2f}×"
        )
        lines.extend(f"提示      {note}" for note in self.notes)
        return "\n".join(lines)


def compare_benchmark(
    fp32_path: str | Path,
    int8_path: str | Path,
    *,
    sample: np.ndarray | None = None,
    warmup: int = 50,
    runs: int = 1000,
    threads: int = 1,
    batch_size: int = 1,
    backend: str = "onnxruntime",
    repeats: int = 1,
    **engine_options: Any,
) -> BenchmarkReport:
    """并排测 FP32 与 INT8。

    两个模型用同一份输入、同样的预热与统计次数、同样的线程数——任何一项不
    一致，比出来的加速比都不成立。
    """
    common = dict(
        sample=sample,
        warmup=warmup,
        runs=runs,
        threads=threads,
        batch_size=batch_size,
        backend=backend,
        **engine_options,
    )

    if repeats <= 1:
        fp32 = benchmark_path(fp32_path, repeats=1, **common)
        int8 = benchmark_path(int8_path, repeats=1, **common)
        round_ratios: list[float] = []
    else:
        # 配对测量：每轮里先测 FP32 再测 INT8，从头交替 repeats 次。
        # 不这么做的话，"测完 FP32 再测 INT8"存在系统性偏置——两个模型落在
        # 不同的时间窗口里，机器频率和后台负载早就变了。实测同一份 FP32
        # 模型在不同时刻能测出 0.011 和 0.019 两个相差近一倍的值，串行测量
        # 出来的加速比因此可能整个是假的。
        fp32_rounds: list[LatencyStats] = []
        int8_rounds: list[LatencyStats] = []
        for _ in range(repeats):
            fp32_rounds.append(benchmark_path(fp32_path, repeats=1, **common))
            int8_rounds.append(benchmark_path(int8_path, repeats=1, **common))
        fp32 = representative_round(fp32_rounds)
        int8 = representative_round(int8_rounds)
        round_ratios = [
            quantized.speedup_vs(reference)
            for reference, quantized in zip(fp32_rounds, int8_rounds, strict=True)
        ]

    notes: list[str] = []
    if runs < _P99_MIN_RUNS:
        notes.append(
            f"统计次数只有 {runs}，p99 的尾部样本不足，这个 p99 更接近最大值而不是分位数"
        )
    if threads > 1:
        notes.append(
            f"线程数为 {threads}，测到的是吞吐而非单次推理延迟；"
            "要测纯算子延迟请用 threads=1"
        )
    if fp32.batch_size > 1:
        notes.append("batch 大于 1，单次控制步的延迟要按 batch=1 另测")

    # 各轮算出的加速比本身若不一致，说明这台机器的状态在测量期间一直在变，
    # 再给一个两位小数的加速比就是编数字。把范围摆出来，结论让人自己下。
    report = BenchmarkReport(fp32=fp32, int8=int8, notes=notes, round_ratios=round_ratios)

    if repeats > 1:
        if report.ratio_spread >= _RATIO_SPREAD_LIMIT:
            low, high = min(report.round_ratios), max(report.round_ratios)
            notes.append(
                f"{repeats} 轮之间的加速比在 {low:.2f}×–{high:.2f}× 之间波动"
                f"（最大/最小 = {report.ratio_spread:.2f}），测量期间机器状态一直在变。"
                f"这种情况下只有 min 的比较还站得住：{report.speedup_min:.2f}×，"
                "均值与 p99 那两列当参考，别写进结论"
            )
        else:
            notes.append(
                f"配对测量 {repeats} 轮，加速比中位 {report.speedup_mean:.2f}×，"
                f"各轮最大/最小 {report.ratio_spread:.2f}（越接近 1 越可信）"
            )
    else:
        # 单轮没有配对可言，只能退化回"比两个独立测量"。这时候轮间波动
        # 就是唯一的可信度线索，必须报出来。
        gap = abs(fp32.p50_ms - int8.p50_ms)
        spread = max(fp32.p50_spread_ms, int8.p50_spread_ms)
        if spread > gap:
            notes.append(
                f"轮间 p50 波动最大 {spread:.4f} ms，大于两模型的 p50 差值 {gap:.4f} ms："
                "这个量级的差异在本机分辨不出来。加大 --repeats 或换台更安静的机器再下结论"
            )

    return report
