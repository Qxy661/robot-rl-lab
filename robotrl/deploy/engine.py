"""推理后端抽象：PolicyEngine。

端侧部署的现实是同一个 ONNX 在不同机器上走不同的运行时——开发机上跑
onnxruntime，端侧可能要走 RKNN、BPU 或 TensorRT，而这几条路的 API 完全不搭。
把差异收在 PolicyEngine 后面，调用方只认三件事：

    engine.load(path)       载入模型
    engine.infer(obs)       观测进、动作出
    engine.latency_stats()  延迟分布

换后端时基准、回测这些上层代码一行都不用改。这不是为了好看：量化回测要
"同一批初始状态下跑两份策略"，如果回测代码里写死 onnxruntime 的
InferenceSession，将来接了 RKNN 就得把回测重写一遍。

metadata 是端侧的另一半。策略输出的是归一化动作，要变成关节目标角还得

    target = default_angles + action_scale * action

这两组常数存在 ONNX 的 metadata_props 里，不进计算图——融进图里会把量化
误差的来源搅在一起，反而不好做误差归因。端侧读出来即可还原。

本模块只在真正构造 OnnxRuntimeEngine 时才 import onnxruntime，因此
PolicyEngine 这份抽象本身是零依赖的：将来某个只有 RKNN 的端侧环境可以直接
继承它，不需要装 onnxruntime。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotrl.utils.onnx_meta import (
    parse_value,
    read_metadata,
    write_metadata,
)

# ---------------------------------------------------------------------------
# ONNX metadata 读写
#
# 实现放在 utils/onnx_meta.py。搬过去的原因是算法层也要写 metadata——基类
# `Policy.export_onnx()` 导出完就得补上，而算法层不能反过来 import 部署层。
# 放在 utils 里两层都能用，也不会让"算法层依赖部署层"这种倒挂在代码里生根。
#
# 这里保留同名入口：部署层读者习惯从本模块取，且 `engines.write_onnx_metadata`
# 是已有测试与外部调用点依赖的名字。别名指向同一个函数对象，行为完全一致。
# ---------------------------------------------------------------------------

read_onnx_metadata = read_metadata
write_onnx_metadata = write_metadata
parse_metadata_value = parse_value


# ---------------------------------------------------------------------------
# 延迟统计
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyStats:
    """一次延迟基准的分布。

    只报均值是不够的：足式策略跑在 50Hz 的控制回路里，均值达标但偶发一次
    长尾超时就会让机器人踉跄一下。所以 p95 / p99 是必报项，它们才对应
    "最坏情况下还能不能跟上控制周期"。

    Attributes:
        batch_size: 每次推理的样本数。batch=1 时延迟就是单次控制步的开销。
        threads: 推理线程数。1 表示纯算子延迟，大于 1 时测到的更接近吞吐。
    """

    count: int
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    batch_size: int = 1
    threads: int = 1
    backend: str = ""
    model: str = ""
    #: 测量重复了几轮。大于 1 时，下面的分布取的是"中位那一轮"。
    rounds: int = 1
    #: 轮与轮之间 p50 的最大差值（毫秒）。它比分布本身更重要——两个模型的
    #: p50 差得比这个值还小的话，本机根本分辨不出谁快。
    p50_spread_ms: float = 0.0

    @property
    def throughput(self) -> float:
        """每秒推理的样本数。多线程下这个值才是重点，单线程下它只是 1/延迟。"""
        if self.mean_ms <= 0:
            return float("nan")
        return self.batch_size * 1000.0 / self.mean_ms

    @classmethod
    def from_samples(
        cls,
        samples_ms: Sequence[float] | np.ndarray,
        *,
        batch_size: int = 1,
        threads: int = 1,
        backend: str = "",
        model: str = "",
        rounds: int = 1,
        p50_spread_ms: float = 0.0,
    ) -> LatencyStats:
        """从逐次测量的毫秒样本算分布。样本为空时报错而不是返回一堆 nan。"""
        arr = np.asarray(samples_ms, dtype=np.float64)
        if arr.size == 0:
            raise ValueError("延迟样本为空，无法统计")
        if np.any(~np.isfinite(arr)):
            raise ValueError("延迟样本里有非有限值，说明测量过程被中断过")
        return cls(
            count=int(arr.size),
            mean_ms=float(arr.mean()),
            std_ms=float(arr.std()),
            min_ms=float(arr.min()),
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            max_ms=float(arr.max()),
            batch_size=int(batch_size),
            threads=int(threads),
            backend=backend,
            model=model,
            rounds=int(rounds),
            p50_spread_ms=float(p50_spread_ms),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "count": self.count,
            "batch_size": self.batch_size,
            "threads": self.threads,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
            "throughput_per_s": self.throughput,
            "rounds": self.rounds,
            "p50_spread_ms": self.p50_spread_ms,
        }

    def speedup_vs(self, baseline: LatencyStats) -> float:
        """相对基线的加速比，按均值算。小于 1 表示变慢了。"""
        if self.mean_ms <= 0:
            return float("nan")
        return baseline.mean_ms / self.mean_ms

    def summary(self) -> str:
        text = (
            f"{self.model or 'model':<18} threads={self.threads} batch={self.batch_size} "
            f"min={self.min_ms:.4f}ms mean={self.mean_ms:.3f}ms p50={self.p50_ms:.3f}ms "
            f"p95={self.p95_ms:.3f}ms p99={self.p99_ms:.3f}ms "
            f"throughput={self.throughput:.1f}/s"
        )
        if self.rounds > 1:
            text += f"（{self.rounds} 轮取中位，轮间 p50 波动 {self.p50_spread_ms:.3f}ms）"
        return text


# ---------------------------------------------------------------------------
# 后端接口
# ---------------------------------------------------------------------------


class PolicyEngine(ABC):
    """策略推理后端的统一接口。

    子类只需实现两个钩子：`_load_backend` 和 `_run_batch`。批量/单条的形状
    处理、metadata 解析、延迟统计都在基类，因此换后端不会改变调用方代码，
    也不会出现"这个后端支持批量、那个不支持"的差异。

    Attributes:
        backend: 后端标识，会出现在报告里，用于区分同一模型在不同后端上的数字。
        path: 已载入的模型路径，未载入时为 None。
        metadata: 模型的 metadata_props，键值均为字符串。
        input_name / output_name: 图中的张量名。
        obs_dim / action_dim: 维度，优先由后端从图里读，读不到则回落到 metadata。
    """

    backend: str = "abstract"

    def __init__(self, *, record_latency: bool = False, latency_window: int = 2048) -> None:
        """
        Args:
            record_latency: 是否在每次 infer 时记录耗时。默认关闭——计时本身
                有开销，做基准时会污染测量结果；只有端侧运行时监控才需要打开。
            latency_window: 保留最近多少次延迟样本。用滑动窗口而不是累积，
                因为机器人跑几小时后，早期的样本已经不能代表当前状态了。
        """
        self.path: Path | None = None
        self.metadata: dict[str, str] = {}
        self.input_name: str = "obs"
        self.output_name: str = "action"
        self.obs_dim: int = 0
        self.action_dim: int = 0

        self._record_latency = record_latency
        self._latencies: deque[float] = deque(maxlen=latency_window)
        self._threads = 1
        self._loaded = False

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_backend(self, path: Path, **options: Any) -> None:
        """建立后端会话。实现里应当顺手填好 input_name / output_name。"""

    @abstractmethod
    def _run_batch(self, batch: np.ndarray) -> np.ndarray:
        """执行一次批量推理。输入 (B, obs_dim) float32，输出 (B, action_dim) float32。"""

    def _read_metadata(self, path: Path) -> dict[str, str]:
        """读模型的 metadata。默认返回空——不认 metadata 的后端可以不覆盖。"""
        return {}

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def load(self, path: str | Path, **options: Any) -> PolicyEngine:
        """载入模型。返回 self 以便链式调用。"""
        if self._loaded:
            self.close()

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"模型文件不存在：{self.path}")

        self._load_backend(self.path, **options)
        self.metadata = dict(self._read_metadata(self.path))

        if not self.obs_dim:
            self.obs_dim = int(self.metadata.get("obs_dim", 0) or 0)
        if not self.action_dim:
            self.action_dim = int(self.metadata.get("action_dim", 0) or 0)

        self._loaded = True
        # 第一次推理要建内存池、做首次常量分配，耗时可能是稳态的十几倍。
        # 在这里一次性付掉，免得混进后面的任何测量。批轴和特征轴都是动态
        # 的图推不出 obs_dim，那种情况下跳过——不值得为此让载入失败。
        if self.obs_dim:
            self.warmup(1)
        return self

    def infer(self, obs: np.ndarray | Sequence[float]) -> np.ndarray:
        """推理。

        接受单个观测 (obs_dim,) 或一批 (B, obs_dim)，输出秩与输入一致。
        保留一维入一维出是为了让控制回路里 `engine.infer(obs)` 能直接喂给
        env.step()，不必到处 squeeze。
        """
        arr = np.asarray(obs, dtype=np.float32)
        single = arr.ndim == 1
        batch = arr[None, :] if single else arr
        if batch.ndim != 2:
            raise ValueError(f"观测应为 (obs_dim,) 或 (B, obs_dim)，实际形状 {arr.shape}")
        if self.obs_dim and batch.shape[1] != self.obs_dim:
            raise ValueError(
                f"观测末维应为 {self.obs_dim}，实际 {batch.shape[1]}；"
                "维数对不上说明模型与环境的观测契约不一致"
            )

        prepared = np.ascontiguousarray(batch, dtype=np.float32)
        if self._record_latency:
            # 计时包住的是"观测进、动作出"的完整一段，包含 dtype 与内存布局的
            # 整理——端侧真正关心的就是这一段。
            start = time.perf_counter()
            out = self._run_batch(prepared)
            self._latencies.append((time.perf_counter() - start) * 1e3)
        else:
            out = self._run_batch(prepared)

        out = np.asarray(out, dtype=np.float32)
        return out[0] if single else out

    def infer_batch(self, obs: np.ndarray) -> np.ndarray:
        """批量推理的显式入口，恒返回 (B, action_dim)。基准测试用它。"""
        batch = np.asarray(obs, dtype=np.float32)
        if batch.ndim != 2:
            raise ValueError(f"批量推理需要 (B, obs_dim)，实际形状 {batch.shape}")
        return self._run_batch(np.ascontiguousarray(batch, dtype=np.float32))

    def warmup(self, n: int = 10) -> None:
        """空跑若干次，让 CPU 频率、内存池、算子内核选择都进入稳态。

        不预热的话头几次会明显偏慢，而延迟基准恰恰统计 p99——把这几百毫秒
        混进去，p99 就不是"最慢的一次推理"，而是"启动开销"了。
        """
        if n <= 0:
            return
        if not self.obs_dim:
            raise RuntimeError("载入之前拿不到 obs_dim，无法预热")
        dummy = np.zeros((1, self.obs_dim), dtype=np.float32)
        for _ in range(n):
            self._run_batch(dummy)

    def latency_stats(self) -> LatencyStats | None:
        """已记录延迟的分布。未开启记录或还没推理过时返回 None。"""
        if not self._latencies:
            return None
        return LatencyStats.from_samples(
            list(self._latencies),
            batch_size=1,
            threads=self._threads,
            backend=self.backend,
            model=self.path.stem if self.path else "",
        )

    def reset_latency(self) -> None:
        self._latencies.clear()

    @property
    def threads(self) -> int:
        """推理线程数。延迟报告必须带上它——1 线程和 4 线程的数字不可比。"""
        return self._threads

    # ---- metadata 派生量 ----

    def meta_value(self, key: str, default: Any = None) -> Any:
        """取 metadata 里的字段并解析成 Python 对象。"""
        raw = self.metadata.get(key)
        if raw is None:
            return default
        return parse_metadata_value(raw)

    @property
    def default_angles(self) -> np.ndarray:
        """默认关节角 (n_dof,)。读不到就直接报错，而不是悄悄用 0 顶替。"""
        angles = self.meta_value("default_angles")
        if angles is None:
            raise KeyError(
                "模型 metadata 里没有 default_angles，无法还原关节目标角；"
                "确认这个 ONNX 是由 robotrl.deploy.export_onnx 导出的"
            )
        return np.asarray(angles, dtype=np.float64)

    @property
    def action_scale(self) -> np.ndarray:
        """动作缩放 (n_dof,)。导出时标量会被展开成向量，这里统一按向量处理。"""
        scale = self.meta_value("action_scale")
        if scale is None:
            raise KeyError("模型 metadata 里没有 action_scale，无法还原关节目标角")
        arr = np.asarray(scale, dtype=np.float64)
        if arr.ndim == 0:
            n = self.action_dim or 1
            arr = np.full(n, float(arr))
        return arr

    def action_to_joint_targets(self, action: np.ndarray) -> np.ndarray:
        """把网络输出的归一化动作还原成关节目标角。

            target = default_angles + action_scale * action

        这一步在真机上由部署代码完成（仿真里由环境层的 _apply_action 完成）。
        放在引擎里是因为两者用的是同一份 metadata，写两遍迟早会不一致。
        """
        act = np.asarray(action, dtype=np.float64)
        angles, scale = self.default_angles, self.action_scale
        if act.shape[-1] != angles.shape[0]:
            raise ValueError(
                f"动作维度 {act.shape[-1]} 与 default_angles 长度 {angles.shape[0]} 不一致"
            )
        return angles + scale * act

    # ---- 生命周期 ----

    def close(self) -> None:
        """释放后端资源。无资源的后端可以不覆盖。"""
        self._loaded = False

    def __enter__(self) -> PolicyEngine:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        state = f"loaded={self.path}" if self._loaded else "empty"
        return f"{type(self).__name__}(backend={self.backend}, {state})"


class OnnxRuntimeEngine(PolicyEngine):
    """onnxruntime 后端。

    线程数的语义要分清楚：intra-op 是算子内部的并行（矩阵乘分块），
    inter-op 是算子之间的并行。连续控制的策略网络是一条近乎串行的依赖链，
    算子之间没什么可并行的，所以 inter-op 固定为 1——开了只会多出调度开销。

    因此 threads=1 测到的是"一次推理要多久"（纯算子延迟，对应控制周期够不够），
    threads>1 测到的是"多久能塞满 CPU"（吞吐，对应多路复用同一块板子的场景）。
    两者不是一回事，报告里必须说清测的是哪个。
    """

    backend = "onnxruntime"

    def __init__(
        self,
        *,
        threads: int = 1,
        providers: Sequence[str] | None = None,
        record_latency: bool = False,
        latency_window: int = 2048,
        graph_optimization: str = "all",
        execution_mode: str = "sequential",
    ) -> None:
        super().__init__(record_latency=record_latency, latency_window=latency_window)
        if threads <= 0:
            raise ValueError(f"线程数必须为正，得到 {threads}")
        self._threads = threads
        self._providers = list(providers) if providers else None
        self._graph_optimization = graph_optimization
        self._execution_mode = execution_mode
        self._session: Any = None
        self._input_name: str | None = None

    def _make_session_options(self) -> Any:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = self._threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }[self._graph_optimization]
        so.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
            if self._execution_mode == "sequential"
            else ort.ExecutionMode.ORT_PARALLEL
        )
        # 关掉内存 arena 之外的信息级别：ORT 默认会在 stderr 刷 provider 警告，
        # 混在基准输出里不好读。
        so.log_severity_level = 3
        return so

    def _load_backend(self, path: Path, **options: Any) -> None:
        import onnxruntime as ort

        providers = options.get("providers", self._providers) or ort.get_available_providers()
        session = ort.InferenceSession(
            str(path),
            sess_options=self._make_session_options(),
            providers=list(providers),
        )
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.input_name = self._input_name

        # 从图里读维度。批轴是动态的（None），只有末维是确定的。
        in_shape = session.get_inputs()[0].shape
        out_shape = session.get_outputs()[0].shape
        if isinstance(in_shape[-1], int):
            self.obs_dim = int(in_shape[-1])
        if isinstance(out_shape[-1], int):
            self.action_dim = int(out_shape[-1])

    def _read_metadata(self, path: Path) -> dict[str, str]:
        try:
            return read_onnx_metadata(path)
        except Exception:  # onnx 缺失或文件不完整时，没 metadata 也要能跑推理
            return {}

    def _run_batch(self, batch: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("会话还没建立，先调用 load()")
        outputs = self._session.run([self.output_name], {self._input_name: batch})
        return np.asarray(outputs[0], dtype=np.float32)

    @property
    def provider(self) -> str:
        """当前实际生效的 provider。CPU 上跑还是 CUDA 上跑，数字差很多。"""
        if self._session is None:
            return ""
        return self._session.get_providers()[0]

    def close(self) -> None:
        self._session = None
        super().close()

    def __repr__(self) -> str:
        return (
            f"OnnxRuntimeEngine(threads={self._threads}, "
            f"provider={self.provider or '-'}, path={self.path})"
        )


# ---------------------------------------------------------------------------
# 后端工厂：调用方按名字取引擎，将来加 RKNN 只需在这里注册一次
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, Callable[..., PolicyEngine]] = {}


def register_backend(name: str, factory: Callable[..., PolicyEngine]) -> None:
    """注册推理后端。名字重复直接报错，避免静默覆盖。"""
    if name in _BACKENDS and _BACKENDS[name] is not factory:
        raise ValueError(f"后端 {name!r} 已注册，不能覆盖")
    _BACKENDS[name] = factory


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def make_engine(name: str = "onnxruntime", **options: Any) -> PolicyEngine:
    """按名字构造引擎（未载入模型）。"""
    if name not in _BACKENDS:
        raise KeyError(f"未注册的后端 {name!r}，可用：{available_backends()}")
    return _BACKENDS[name](**options)


def load_engine(path: str | Path, name: str = "onnxruntime", **options: Any) -> PolicyEngine:
    """构造并载入。多数调用方要的是这个。"""
    return make_engine(name, **options).load(path)


register_backend("onnxruntime", OnnxRuntimeEngine)
