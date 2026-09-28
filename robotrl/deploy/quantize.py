"""INT8 量化：把 FP32 的 ONNX 压成端侧能跑的 INT8。

两条路，代价和收益不一样：

**动态量化（dynamic）**：只量化权重，激活在推理时按当前 batch 的实际范围
现算 scale。不需要校准数据，一行代码就能跑，压缩率立竿见影。代价是激活的
量化是逐次推理做的，多一层运行时开销，而且每次推理的 scale 都不同——同一
个观测连喂两次，理论上结果一致，但和 FP32 的偏差不好预估。

**静态 QDQ（static）**：权重和激活都提前量化好，激活的 scale 由校准集统计
出来并写进图（QuantizeLinear / DequantizeLinear 节点）。推理时没有额外统计
开销，端侧 NPU 通常也要求这种形式。代价是需要一批有代表性的校准集：校准集
覆盖不到的状态，scale 就会估偏，那里掉点最狠。

校准集必须从环境里采，而且要用策略自己跑出来的观测——均匀随机动作采到的
状态分布和部署时遇到的不是一回事，拿它标定等于给错误的区间分配了量化精度。

量化器的产出有两件事必须复查，本模块在 quantize() 里自动做：
- 有没有真的量化到算子。模型太小时量化器可能一个算子都没换，文件还是那么大，
  但报告里写着"已量化"——这种假成功最坑。
- metadata 有没有丢。丢了端侧就还原不出关节目标角，而这个错误在推理时
  完全不报错。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from robotrl.deploy.engine import read_onnx_metadata, write_onnx_metadata
from robotrl.deploy.evaluate import collect_observations

#: 会被量化器替换掉的算子。用它们判断"到底量化成功了没有"。
_QUANT_OPS = (
    "MatMulInteger",
    "DynamicQuantizeLinear",
    "QLinearMatMul",
    "QuantizeLinear",
    "DequantizeLinear",
    "ConvInteger",
    "QLinearConv",
)

_WEIGHT_TYPES = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}
_ACT_TYPES = {"quint8": QuantType.QUInt8, "qint8": QuantType.QInt8}
_FORMATS = {"qdq": QuantFormat.QDQ, "qoperator": QuantFormat.QOperator}
_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}


def _lookup(table: Mapping[str, Any], key: str, what: str) -> Any:
    if key not in table:
        raise ValueError(f"未知的{what} {key!r}，可选：{sorted(table)}")
    return table[key]


# ---------------------------------------------------------------------------
# 校准集
# ---------------------------------------------------------------------------


class NumpyCalibrationReader(CalibrationDataReader):
    """把一堆 numpy 观测喂给静态量化器。

    onnxruntime 的校准接口是"反复调 get_next 直到返回 None"，所以这里用下标
    游标实现。分批而不是一条一条喂：QDQ 的 shape 推断在 batch 维上要一致，
    逐条喂会让每个 batch 都是 1，遇到带动态批轴的图反而更慢。
    """

    def __init__(
        self,
        data: np.ndarray | Sequence[np.ndarray],
        input_name: str = "obs",
        *,
        batch_size: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"批大小必须为正，得到 {batch_size}")

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data[None, :]
            if data.ndim != 2:
                raise ValueError(f"校准数据应为 (N, obs_dim)，实际形状 {data.shape}")
            batches = [
                data[i : i + batch_size].astype(np.float32)
                for i in range(0, data.shape[0], batch_size)
            ]
        else:
            batches = [np.asarray(b, dtype=np.float32) for b in data]

        if not batches:
            raise ValueError("校准数据为空，静态量化没有东西可标定")

        self._batches = batches
        self._input_name = input_name
        self._index = 0

    @property
    def num_batches(self) -> int:
        return len(self._batches)

    @property
    def num_samples(self) -> int:
        return int(sum(b.shape[0] for b in self._batches))

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._batches):
            return None
        batch = self._batches[self._index]
        self._index += 1
        return {self._input_name: batch}

    def rewind(self) -> None:
        self._index = 0


def collect_calibration_data(
    env: Any,
    *,
    steps: int = 512,
    seed: int = 0,
    engine: Any = None,
) -> np.ndarray:
    """从环境采一批观测作为校准集。

    转发到 evaluate.collect_observations，只是为了在调用点上读起来是校准语义。
    默认建议传入与待量化模型对应的引擎，让状态分布与部署时一致。
    """
    return collect_observations(env, steps=steps, seed=seed, engine=engine)


# ---------------------------------------------------------------------------
# 量化实现
# ---------------------------------------------------------------------------


def quantize_dynamic_model(
    src: str | Path,
    dst: str | Path,
    *,
    per_channel: bool = True,
    weight_type: str = "int8",
    reduce_range: bool = False,
    extra_options: Mapping[str, Any] | None = None,
) -> Path:
    """动态量化。不需要校准数据，适合先把链路跑通再谈精度。

    per_channel=True 时逐输出通道各用一个 scale，对权重的动态范围差异大的
    层（例如输出层）精度明显更好，代价是端侧 kernel 支持度不一是可能的。
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"待量化的模型不存在：{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        per_channel=per_channel,
        weight_type=_lookup(_WEIGHT_TYPES, weight_type, "权重量化类型"),
        reduce_range=reduce_range,
        extra_options=dict(extra_options or {}),
    )
    return dst


def quantize_static_model(
    src: str | Path,
    dst: str | Path,
    calibration: np.ndarray | Sequence[np.ndarray] | CalibrationDataReader,
    *,
    input_name: str = "obs",
    quant_format: str = "qdq",
    per_channel: bool = False,
    weight_type: str = "int8",
    activation_type: str = "quint8",
    calibrate_method: str = "minmax",
    batch_size: int = 1,
    preprocess: bool = False,
) -> Path:
    """静态 QDQ 量化。需要校准集。

    默认用 QDQ 格式（QuantizeLinear/DequantizeLinear 成对出现）而不是
    QOperator：QDQ 图在端侧编译器（TensorRT、RKNN、各家 NPU 工具链）里
    接受度更高，而且量化参数以节点形式显式挂在图上，读起来和调试都方便。

    Args:
        preprocess: 是否先跑一遍 quant_pre_process 做形状推断与图优化。
            ONNX Runtime 官方推荐开启，但它依赖额外的符号形状推断，模型简单时
            开着反而容易报错，所以默认关闭。
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"待量化的模型不存在：{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    reader = (
        calibration
        if isinstance(calibration, CalibrationDataReader)
        else NumpyCalibrationReader(calibration, input_name, batch_size=batch_size)
    )

    source = src
    if preprocess:
        source = Path(_preprocess(src))

    quantize_static(
        model_input=str(source),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=_lookup(_FORMATS, quant_format, "量化格式"),
        per_channel=per_channel,
        weight_type=_lookup(_WEIGHT_TYPES, weight_type, "权重量化类型"),
        activation_type=_lookup(_ACT_TYPES, activation_type, "激活量化类型"),
        calibrate_method=_lookup(_METHODS, calibrate_method, "校准方法"),
    )
    return dst


def _preprocess(src: Path) -> Path:
    """跑 onnxruntime 的图预处理。失败就退回原模型，不让它挡住主流程。"""
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out = src.with_name(f"{src.stem}_preprocessed{src.suffix}")
    quant_pre_process(
        input_model_path=str(src), output_model_path=str(out), skip_symbolic_shape=False
    )
    return out


# ---------------------------------------------------------------------------
# 复查与报告
# ---------------------------------------------------------------------------


def count_quantized_ops(path: str | Path) -> dict[str, int]:
    """数一数图里有哪些量化算子。全为 0 就说明量化器什么都没干。"""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type in _QUANT_OPS:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


@dataclass
class QuantReport:
    """量化结果与复查结论。"""

    mode: str
    src: Path
    dst: Path
    size_fp32: int
    size_int8: int
    quantized_ops: dict[str, int] = field(default_factory=dict)
    calibration_samples: int = 0
    weight_type: str = "int8"
    metadata_keys: list[str] = field(default_factory=list)
    missing_metadata: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def size_ratio(self) -> float:
        """INT8 文件相对 FP32 的体积比。越小压得越狠。"""
        if self.size_fp32 <= 0:
            return float("nan")
        return self.size_int8 / self.size_fp32

    @property
    def compression(self) -> float:
        return 1.0 / self.size_ratio if self.size_ratio > 0 else float("nan")

    @property
    def effective(self) -> bool:
        """量化是否真的生效。没换掉任何算子就不算。"""
        return bool(self.quantized_ops)

    @property
    def metadata_ok(self) -> bool:
        """端侧还原关节目标角所需的 metadata 是否齐全。"""
        return not self.missing_metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "src": str(self.src),
            "dst": str(self.dst),
            "size_fp32_bytes": self.size_fp32,
            "size_int8_bytes": self.size_int8,
            "size_ratio": self.size_ratio,
            "compression": self.compression,
            "quantized_ops": self.quantized_ops,
            "effective": self.effective,
            "calibration_samples": self.calibration_samples,
            "weight_type": self.weight_type,
            "metadata_keys": self.metadata_keys,
            "missing_metadata": self.missing_metadata,
            "notes": self.notes,
        }

    def summary(self) -> str:
        ops = ", ".join(f"{k}×{v}" for k, v in sorted(self.quantized_ops.items())) or "无"
        lines = [
            f"量化模式  {self.mode}（权重 {self.weight_type}）",
            f"体积      {self.size_fp32 / 1024:.1f} KB → {self.size_int8 / 1024:.1f} KB "
            f"（压缩 {self.compression:.2f}×）",
            f"量化算子  {ops}",
            f"metadata  {'完整' if self.metadata_ok else '缺少 ' + ', '.join(self.missing_metadata)}",
        ]
        if self.calibration_samples:
            lines.append(f"校准集    {self.calibration_samples} 条观测")
        lines.extend(f"提示      {note}" for note in self.notes)
        return "\n".join(lines)


def quantize(
    src: str | Path,
    dst: str | Path,
    *,
    mode: str = "dynamic",
    calibration: np.ndarray | Sequence[np.ndarray] | CalibrationDataReader | None = None,
    input_name: str = "obs",
    copy_metadata: bool = True,
    **options: Any,
) -> QuantReport:
    """量化主入口：量化 → 拷 metadata → 复查。

    Args:
        mode: "dynamic" 或 "static"，对应 DeployConfig.quant_format。
        calibration: 静态量化必需，动态量化忽略。可以是观测数组或自定义 reader。
        copy_metadata: 量化器不保证原样搬运 metadata，默认重写一遍。
    """
    src, dst = Path(src), Path(dst)
    if mode == "dynamic":
        quantize_dynamic_model(src, dst, **options)
        calibr_samples = 0
    elif mode == "static":
        if calibration is None:
            raise ValueError("静态量化需要校准集，请先用 collect_calibration_data() 采一批观测")
        quantize_static_model(src, dst, calibration, input_name=input_name, **options)
        calibr_samples = (
            calibration.num_samples
            if isinstance(calibration, NumpyCalibrationReader)
            else int(np.asarray(calibration).shape[0])
        )
    else:
        raise ValueError(f"未知量化模式 {mode!r}，只支持 dynamic / static")

    if copy_metadata:
        metadata = read_onnx_metadata(src)
        if metadata:
            write_onnx_metadata(dst, metadata, merge=False)

    report = inspect_quantized(src, dst, mode=mode)
    report.calibration_samples = calibr_samples
    return report


def inspect_quantized(
    src: str | Path,
    dst: str | Path,
    *,
    mode: str = "dynamic",
    weight_type: str = "int8",
) -> QuantReport:
    """复查量化产物：体积、量化算子、metadata。"""
    src, dst = Path(src), Path(dst)
    report = QuantReport(
        mode=mode,
        src=src,
        dst=dst,
        size_fp32=src.stat().st_size,
        size_int8=dst.stat().st_size,
        quantized_ops=count_quantized_ops(dst),
        weight_type=weight_type,
    )

    src_meta = read_onnx_metadata(src)
    dst_meta = read_onnx_metadata(dst)
    report.metadata_keys = sorted(dst_meta)
    report.missing_metadata = [k for k in src_meta if k not in dst_meta]

    if not report.effective:
        report.notes.append(
            "图里没有任何量化算子，量化器什么都没换掉——这个 INT8 模型的计算量"
            "与 FP32 相同，别把它当成量化结果"
        )
    if report.missing_metadata:
        report.notes.append("量化后 metadata 有缺失，端侧还原不出关节目标角")
    if "DequantizeLinear" in report.quantized_ops and "QuantizeLinear" not in report.quantized_ops:
        report.notes.append("只看到反量化节点，激活可能仍是 FP32 计算")

    return report
