"""导出等价性校验：ONNX 的输出必须和 PyTorch 逐元素一致。

这一步在整条工具链里排在最前面，理由很直白：如果导出本身就有偏差，后面
量化测出来的性能下降就说不清是量化的锅还是导出的锅，调优也就无从下手。
先把导出这条路径钉死（误差量级 1e-6 以下，纯粹是浮点累加顺序的差异），
再谈量化。

比什么
------
- 最大绝对误差、平均绝对误差、RMSE；
- 相对误差：既有范数比 ||a-b||/||a||，也有逐元素相对误差的上界；
- 超差比例：有多少比例的元素的偏差超过容差——只看最大值容易被单个离群点
  带偏，比例能说明这是普遍现象还是个别点。

喂什么
------
两类输入都要测：
- 随机观测，且刻意跨越归一化的裁剪边界（|obs| > obs_clip）。裁剪是图里
  第一个非线性环节，导出前后在这里最容易出现不一致；
- 真实观测，从环境里用策略跑出来。随机观测落不到策略真正会遇到的区域，
  只测随机输入等于没测。

按段做误差归因
--------------
`attribute_by_segment()` 回答"哪一段观测对量化最敏感"：把某一段观测单独
按 INT8 的格点量化、其余段保持原值，喂进 FP32 模型看输出偏多少。逐段做一遍
就得到各段对输出误差的贡献占比。这是**一阶近似**——它忽略了下游的传播和
段间交互——但足够回答"该重点保护哪一段"这个问题，而且每段只跑一次推理。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robotrl.contracts import ObsContract
from robotrl.deploy.engine import PolicyEngine, make_engine

#: 逐元素相对误差里的分母下限。动作值可以接近 0，不设下限会把相对误差放大成
#: 无意义的巨大数值。
_REL_EPS = 1e-6


# ---------------------------------------------------------------------------
# 误差度量
# ---------------------------------------------------------------------------


@dataclass
class ErrorMetrics:
    """两组同形状数组之间的差异。"""

    n_elements: int
    max_abs_err: float
    mean_abs_err: float
    rmse: float
    rel_err_norm: float
    max_element_rel_err: float
    violation_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "n_elements": self.n_elements,
            "max_abs_err": self.max_abs_err,
            "mean_abs_err": self.mean_abs_err,
            "rmse": self.rmse,
            "rel_err_norm": self.rel_err_norm,
            "max_element_rel_err": self.max_element_rel_err,
            "violation_ratio": self.violation_ratio,
        }

    def summary(self) -> str:
        return (
            f"max_abs={self.max_abs_err:.3e} mean_abs={self.mean_abs_err:.3e} "
            f"rmse={self.rmse:.3e} rel_norm={self.rel_err_norm:.3e} "
            f"超差比例={self.violation_ratio:.3%}"
        )


def compare_arrays(a: np.ndarray, b: np.ndarray, *, tol: float) -> ErrorMetrics:
    """逐元素比较两组数组。形状必须一致——形状不同说明导出改了接口，直接报错。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"形状不一致：{a.shape} vs {b.shape}")

    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), _REL_EPS)
    norm = float(np.linalg.norm(a))

    return ErrorMetrics(
        n_elements=int(a.size),
        max_abs_err=float(diff.max()) if a.size else 0.0,
        mean_abs_err=float(diff.mean()) if a.size else 0.0,
        rmse=float(np.sqrt(np.mean(diff**2))) if a.size else 0.0,
        rel_err_norm=float(np.linalg.norm(diff) / norm) if norm > 0 else float("nan"),
        max_element_rel_err=float((diff / denom).max()) if a.size else 0.0,
        violation_ratio=float(np.mean(diff > tol)) if a.size else 0.0,
    )


# ---------------------------------------------------------------------------
# 输入构造
# ---------------------------------------------------------------------------


def random_observations(
    obs_dim: int,
    n: int = 256,
    *,
    seed: int = 0,
    scale: float = 1.0,
    distribution: str = "uniform",
) -> np.ndarray:
    """构造一批随机观测。

    distribution="normal" 时会生成少量远超常规范围的样本——策略在发散边缘
    才会遇到这种输入，而导出与量化的问题恰恰容易在这种边界上暴露。
    """
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        obs = rng.uniform(-scale, scale, size=(n, obs_dim))
    elif distribution == "normal":
        obs = rng.normal(0.0, scale, size=(n, obs_dim))
    else:
        raise ValueError(f"未知分布 {distribution!r}，只支持 uniform / normal")
    return obs.astype(np.float32)


def observation_sets(
    obs_dim: int,
    n: int = 128,
    *,
    seed: int = 0,
    obs_clip: float = 10.0,
) -> dict[str, np.ndarray]:
    """三组输入，覆盖策略真正会遇到的范围。

    归一化会把观测先裁剪到 ±obs_clip，所以特意造一组越界输入：裁剪前后的
    数值在图上走的是同一条路径，一旦导出把 Clip 换了实现，这里就会露出来。
    """
    return {
        "uniform": random_observations(obs_dim, n, seed=seed, scale=1.0),
        "normal": random_observations(obs_dim, n, seed=seed + 1, scale=2.0, distribution="normal"),
        "beyond_clip": random_observations(obs_dim, n, seed=seed + 2, scale=obs_clip * 1.5),
    }


# ---------------------------------------------------------------------------
# 推理：PyTorch 与 ONNX 两侧
# ---------------------------------------------------------------------------


def policy_forward_numpy(policy: Any, obs: np.ndarray) -> np.ndarray:
    """用 PyTorch 策略求值，返回 numpy。

    临时切 eval 再还原，是因为调用方可能正在训练循环里顺手做校验，
    把它的 train 模式改掉会静默影响后续的 dropout / BN 行为。
    """
    import torch

    was_training = getattr(policy, "training", False)
    if hasattr(policy, "eval"):
        policy.eval()

    with torch.no_grad():
        out = policy(torch.as_tensor(np.asarray(obs, dtype=np.float32)))

    if was_training and hasattr(policy, "train"):
        policy.train()
    return out.detach().cpu().numpy()


def resolve_engine(engine: PolicyEngine | str | Path, **options: Any) -> PolicyEngine:
    """允许调用方直接传路径，省掉"先 make_engine 再 load"这两行样板代码。"""
    if isinstance(engine, PolicyEngine):
        return engine
    return make_engine("onnxruntime", **options).load(engine)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


@dataclass
class EquivalenceReport:
    """等价性结论。

    passed 只看 max_abs_err：逐元素最大偏差是最严格的判据，它达标了，
    其余指标自然达标。超差比例与相对误差作为补充信息，用于判断偏差是
    普遍存在还是集中在个别维度。
    """

    tol: float
    n_samples: int
    metrics: ErrorMetrics
    per_input: dict[str, dict[str, float]] = field(default_factory=dict)
    label_a: str = "pytorch"
    label_b: str = "onnx"
    segments: list[SegmentError] = field(default_factory=list)

    @property
    def max_abs_err(self) -> float:
        return self.metrics.max_abs_err

    @property
    def passed(self) -> bool:
        return self.metrics.max_abs_err <= self.tol

    @property
    def top_sensitive_segment(self) -> str | None:
        """对输出误差贡献最大的观测段。量化时优先保护它，或给它单独留高精度。"""
        if not self.segments:
            return None
        return max(self.segments, key=lambda s: s.output_max_delta).name

    def assert_passed(self) -> None:
        if not self.passed:
            raise AssertionError(
                f"{self.label_a} 与 {self.label_b} 的最大偏差 {self.max_abs_err:.3e} "
                f"超过容差 {self.tol:.3e}；导出这一步就有问题，先别往下做量化"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "tol": self.tol,
            "passed": self.passed,
            "n_samples": self.n_samples,
            "metrics": self.metrics.to_dict(),
            "per_input": self.per_input,
            "top_sensitive_segment": self.top_sensitive_segment,
            "segments": [s.to_dict() for s in self.segments],
        }

    def summary(self) -> str:
        head = f"{self.label_a} vs {self.label_b}：{'通过' if self.passed else '未通过'}"
        lines = [
            f"{head}（容差 {self.tol:.1e}，样本 {self.n_samples}）",
            f"  总体    {self.metrics.summary()}",
        ]
        lines.extend(
            f"  {name:<12} {metrics['max_abs_err']:.3e}" for name, metrics in self.per_input.items()
        )
        if self.segments:
            lines.append(f"  最敏感段 {self.top_sensitive_segment}")
            lines.extend(
                f"  [{s.name:<14}] 输出偏差 {s.output_max_delta:.3e} "
                f"占比 {s.share:.1%} 输入相对误差 {s.input_rel_err:.3e}"
                for s in self.segments
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def check_equivalence(
    policy: Any,
    engine: PolicyEngine | str | Path,
    inputs: np.ndarray | Mapping[str, np.ndarray],
    *,
    tol: float = 1e-5,
    engine_options: Mapping[str, Any] | None = None,
) -> EquivalenceReport:
    """比对 PyTorch 与 ONNX 的输出。

    Args:
        policy: 任意可调用对象（通常是 Policy），接受 (B, obs_dim) 张量。
        engine: 已载入的引擎，或 ONNX 路径（内部按 onnxruntime 载入）。
        inputs: 一批观测，或 {名字: 观测} 的多组输入。
        tol: 最大绝对误差容差，对应 DeployConfig.equivalence_tol。

    Returns:
        EquivalenceReport。汇总的 max_abs_err 取各组输入里最差的那组——
        校验要按最坏情况判，不是按平均。
    """
    onnx_engine = resolve_engine(engine, **(engine_options or {}))

    if isinstance(inputs, Mapping):
        groups = {str(k): np.asarray(v, dtype=np.float32) for k, v in inputs.items()}
    else:
        groups = {"inputs": np.asarray(inputs, dtype=np.float32)}

    per_input: dict[str, dict[str, float]] = {}
    worst: ErrorMetrics | None = None
    total = 0
    for name, obs in groups.items():
        if obs.ndim == 1:
            obs = obs[None, :]
        expected = policy_forward_numpy(policy, obs)
        actual = onnx_engine.infer_batch(obs)
        metrics = compare_arrays(expected, actual, tol=tol)
        per_input[name] = metrics.to_dict()
        total += obs.shape[0]
        if worst is None or metrics.max_abs_err > worst.max_abs_err:
            worst = metrics

    assert worst is not None
    return EquivalenceReport(
        tol=tol,
        n_samples=total,
        metrics=worst,
        per_input=per_input,
        label_a="pytorch",
        label_b=Path(onnx_engine.path).name if onnx_engine.path else "onnx",
    )


def compare_engines(
    fp32_engine: PolicyEngine | str | Path,
    int8_engine: PolicyEngine | str | Path,
    inputs: np.ndarray | Mapping[str, np.ndarray],
    *,
    tol: float | None = None,
    contract: ObsContract | None = None,
    attribute: bool = True,
    engine_options: Mapping[str, Any] | None = None,
) -> EquivalenceReport:
    """比对量化前后的两个引擎，可选按观测段做误差归因。

    容差默认取 inf：量化必然会引入误差，这里不是为了判"过不过"，而是为了
    把误差量出来。真正的门槛在 evaluate.backtest() 里——看的是行为偏差和回报，
    不是逐元素偏差。
    """
    a = resolve_engine(fp32_engine, **(engine_options or {}))
    b = resolve_engine(int8_engine, **(engine_options or {}))
    threshold = float("inf") if tol is None else tol

    groups = (
        {str(k): np.asarray(v, dtype=np.float32) for k, v in inputs.items()}
        if isinstance(inputs, Mapping)
        else {"inputs": np.asarray(inputs, dtype=np.float32)}
    )

    per_input: dict[str, dict[str, float]] = {}
    worst: ErrorMetrics | None = None
    total = 0
    for name, obs in groups.items():
        if obs.ndim == 1:
            obs = obs[None, :]
        metrics = compare_arrays(a.infer_batch(obs), b.infer_batch(obs), tol=threshold)
        per_input[name] = metrics.to_dict()
        total += obs.shape[0]
        if worst is None or metrics.max_abs_err > worst.max_abs_err:
            worst = metrics

    assert worst is not None
    segments: list[SegmentError] = []
    if attribute and contract is not None:
        reference = next(iter(groups.values()))
        segments = attribute_by_segment(a, b, reference, contract)

    return EquivalenceReport(
        tol=threshold,
        n_samples=total,
        metrics=worst,
        per_input=per_input,
        label_a=f"fp32({Path(a.path).name if a.path else 'onnx'})",
        label_b=f"int8({Path(b.path).name if b.path else 'onnx'})",
        segments=segments,
    )


# ---------------------------------------------------------------------------
# 按观测段的误差归因
# ---------------------------------------------------------------------------


@dataclass
class SegmentError:
    """某一段观测对量化输出误差的贡献。

    Attributes:
        input_rel_err: 该段输入被量化后的平均相对误差。它解释"为什么是这一段"——
            量级小的段（例如 last_action）在 INT8 的固定格点下相对误差天然更大。
        output_max_delta: 仅该段被量化时，输出相对 FP32 的最大偏差。
        share: 该段的贡献占全部段之和的比例。一阶近似，各段之和≈总误差。
    """

    name: str
    dim: int
    index: slice
    input_rel_err: float
    output_max_delta: float
    output_mean_delta: float
    share: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "input_rel_err": self.input_rel_err,
            "output_max_delta": self.output_max_delta,
            "output_mean_delta": self.output_mean_delta,
            "share": self.share,
        }


def quantize_to_grid(
    x: np.ndarray,
    scale: float,
    zero_point: int = 0,
    *,
    qmin: int | None = None,
    qmax: int | None = None,
) -> np.ndarray:
    """把数值按 INT8 的格点取整再还原，模拟量化引入的误差。

    这就是量化本身做的事：round(x/scale)+zp 落到整数格点上，反量化回浮点。
    非对称量化（zero_point≠0，通常是 uint8 激活）的取值范围是 [0, 255]，
    对称量化是 [-128, 127]，边界弄错会凭空造出一批截断误差。
    """
    if scale <= 0:
        raise ValueError(f"量化 scale 必须为正，得到 {scale}")
    if qmin is None or qmax is None:
        lo, hi = (0, 255) if zero_point != 0 else (-128, 127)
        qmin = lo if qmin is None else qmin
        qmax = hi if qmax is None else qmax

    q = np.clip(np.round(np.asarray(x, dtype=np.float64) / scale) + zero_point, qmin, qmax)
    return (q - zero_point) * scale


def input_quant_params(
    engine: PolicyEngine,
    obs: np.ndarray,
    *,
    fallback: bool = True,
) -> tuple[float, int]:
    """取输入张量的量化参数 (scale, zero_point)。

    静态 QDQ 图里挂着 QuantizeLinear 节点，scale 与 zero_point 是常量，直接读。
    动态量化不存激活的量化参数（激活在运行时才量化），这时退回到按观测数据的
    最大值估一个对称 scale —— 这是一阶近似，报出来的数应当当成量级参考，
    而不是精确的端侧数值。
    """
    if engine.path is not None:
        params = _read_qdq_params(Path(engine.path), engine.input_name)
        if params is not None:
            return params
    if not fallback:
        raise RuntimeError(f"{engine.path} 里找不到输入量化参数，且不允许回退估算")
    peak = float(np.max(np.abs(np.asarray(obs, dtype=np.float64))))
    return max(peak, 1e-8) / 127.0, 0


def _read_qdq_params(path: Path, tensor_name: str) -> tuple[float, int] | None:
    """在 QDQ 图里找 `QuantizeLinear(tensor_name)` 并读出 scale / zero_point。"""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:  # onnx 缺失时直接走估算路径
        return None

    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception:
        return None

    initializers = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or not node.input or node.input[0] != tensor_name:
            continue
        if len(node.input) < 2:
            continue
        scale = initializers.get(node.input[1])
        if scale is None:
            continue
        zp = initializers.get(node.input[2]) if len(node.input) > 2 else None
        return float(np.asarray(scale).ravel()[0]), int(
            np.asarray(zp).ravel()[0]
        ) if zp is not None else 0
    return None


def attribute_by_segment(
    fp32_engine: PolicyEngine,
    int8_engine: PolicyEngine,
    obs: np.ndarray,
    contract: ObsContract,
    *,
    quant_params: Sequence[tuple[float, int]] | None = None,
) -> list[SegmentError]:
    """逐段量化观测、看输出偏多少，得到各段的误差贡献。

    做法：以 FP32 引擎为参照，每次只把一段观测替换成它被 INT8 量化后的值，
    其余段保持原样。输出偏差反映的就是这一段单独造成的影响。

    为什么用 FP32 引擎做扰动源而不是直接跑 INT8 引擎：INT8 引擎一次前向里
    所有段都被量化了，分不出是谁的锅。用 FP32 引擎逐段注入量化误差，才能
    把贡献拆开——代价是忽略了下游的误差传播，所以结论是一阶的。

    Args:
        quant_params: 逐段覆盖 (scale, zero_point)。默认从 ONNX 图里读，
            读不到就按各段自身的数据范围估对称 scale。
    """
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.shape[1] != contract.total_dim:
        raise ValueError(f"观测末维 {obs.shape[1]} 与契约总维 {contract.total_dim} 不一致")

    reference = fp32_engine.infer_batch(obs)

    errors: list[SegmentError] = []
    for i, segment in enumerate(contract.segments):
        index = contract.index(segment.name)
        if quant_params is not None:
            scale, zero_point = quant_params[i]
        else:
            scale, zero_point = _segment_quant_params(int8_engine, obs[:, index], segment.name)

        perturbed = obs.copy()
        perturbed[:, index] = quantize_to_grid(perturbed[:, index], scale, zero_point).astype(
            np.float32
        )
        delta = np.abs(fp32_engine.infer_batch(perturbed) - reference)

        original = np.abs(obs[:, index]).mean()
        introduced = np.abs(perturbed[:, index] - obs[:, index]).mean()
        errors.append(
            SegmentError(
                name=segment.name,
                dim=segment.dim,
                index=index,
                input_rel_err=float(introduced / original) if original > 0 else float("nan"),
                output_max_delta=float(delta.max()),
                output_mean_delta=float(delta.mean()),
                share=0.0,
            )
        )

    total = sum(e.output_max_delta for e in errors)
    if total > 0:
        for e in errors:
            e.share = e.output_max_delta / total
    return errors


def _segment_quant_params(engine: PolicyEngine, values: np.ndarray, name: str) -> tuple[float, int]:
    """单段的量化参数：优先读图里的输入 scale（全局），读不到就按该段自身范围估。

    两者含义不同：读到的全局 scale 是端侧真实用的那个，按段估的则假设每段
    各有一个独立量化器。前者更贴近实际，后者单看某一段的相对误差更准。
    报告里会标出用的是哪一种口径，避免把估算值当成实测值。
    """
    del name  # 段名只用于报错定位，这里不参与计算
    if engine.path is not None:
        params = _read_qdq_params(Path(engine.path), engine.input_name)
        if params is not None:
            return params
    peak = float(np.max(np.abs(np.asarray(values, dtype=np.float64))))
    return max(peak, 1e-8) / 127.0, 0
