"""ONNX 导出：把训练好的策略变成端侧能直接用的图。

`Policy.export_onnx()` 已经把最要紧的事做完了——观测归一化作为常量进图，
导出的东西自包含，端侧喂裸观测即可。这个模块在它外面补三件事：

1. **从 checkpoint 恢复策略**。checkpoint 里只有权重，没有网络结构，所以必须
   由调用方给出结构（一个空的 Policy 实例或一个工厂函数）。这里不猜、也不做
   "看见键名就猜是几层"的魔法——猜错的结果是导出成功但数值全错，比直接报错
   难查得多。

2. **把 default_angles / action_scale 写进 metadata**。策略吐的是 [-1, 1] 的
   归一化动作，端侧要还原成关节目标角需要

       target = default_angles + action_scale * action

   这两个常数不进计算图，因为融进去之后"量化误差"里就混了动作还原误差，
   误差归因就做不干净了。顺带把关节名、观测分段布局也写进去：端侧按段做
   误差归因时需要知道每一段占哪些下标。

3. **导出后自检**。onnx.checker 验证图合法、打印输入输出形状与算子数、
   核对 metadata 是否都落盘。跳过自检直接进量化，出了问题就得在量化后的
   一堆 QDQ 节点里找原因。

导出器选择
----------
基类 `Policy.export_onnx()` 固定走 TorchScript 导出器，并把 metadata 单独补写，
因此在正常环境里这里不会出意外。但基类终究是个可替换的抽象——调用方可能传入
自己实现的 Policy，或者落在某个 torch 版本上真的导不出来。所以本模块在基类
调用失败且失败原因指向"导出器/依赖不匹配"时，回落到自己这条 TorchScript 路径，
并把回落原因写进报告的 notes，不静默发生。

两条路径导出的图在数值上等价，等价校验会给出同样的结论——这一点由
`deploy/equivalence.py` 断言，而不是靠信任。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robotrl.algorithms.base import Policy
from robotrl.assets.spec import RobotSpec, get_spec
from robotrl.contracts import ObsContract
from robotrl.deploy.engine import write_onnx_metadata

#: 导出器写进 metadata 的键，端侧按这些键取值。列在这里是为了让"契约"可见。
META_KEYS = (
    "obs_dim",
    "action_dim",
    "joint_names",
    "default_angles",
    "action_scale",
    "obs_segments",
    "action_semantics",
)


# ---------------------------------------------------------------------------
# 从 checkpoint 恢复策略
# ---------------------------------------------------------------------------


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """从 checkpoint 里取策略权重。

    兼容三种落盘方式：Trainer.save() 的 {"policy": ...}、裸 state_dict、
    以及别的框架常用的 {"model": ...} / {"state_dict": ...}。识别不到时
    把顶层键列出来一起报错——比让 load_state_dict 抛一堆 missing keys 好查。
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint 应为 dict，实际是 {type(checkpoint).__name__}")

    for key in ("policy", "model", "state_dict", "actor"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and value and all(
            isinstance(v, torch.Tensor) for v in value.values()
        ):
            return dict(value)

    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return dict(checkpoint)

    raise KeyError(
        f"checkpoint 里找不到策略权重，顶层键为 {sorted(checkpoint)}；"
        "期望 policy / model / state_dict / actor 之一，或直接就是 state_dict"
    )


def load_policy(
    checkpoint: str | Path,
    *,
    policy: Policy | None = None,
    policy_factory: Callable[[], Policy] | None = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Policy:
    """从 checkpoint 加载策略。

    需要二选一：
      - policy：已经构造好的空策略（维度已知），本函数只做 load_state_dict；
      - policy_factory：无参工厂，本函数负责构造再填权重。

    为什么非要调用方给结构：state_dict 只记录张量，不记录层与层怎么接。
    从键名反推结构在简单 MLP 上碰巧能work，一旦换网络就会静默出错。

    Args:
        strict: 传给 load_state_dict。权重与结构对不上时报错，不静默吞掉。
    """
    if policy is None and policy_factory is None:
        raise ValueError("必须给出 policy 或 policy_factory 之一，否则无法确定网络结构")

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")

    ckpt = torch.load(checkpoint, map_location=map_location, weights_only=False)
    state_dict = extract_state_dict(ckpt)

    target = policy_factory() if policy is None else policy
    target.load_state_dict(state_dict, strict=strict)
    target.eval()
    return target


def spec_from_checkpoint(checkpoint: str | Path, *, key: str = "config") -> RobotSpec | None:
    """checkpoint 里若存了配置，就顺手把形态取出来，省得调用方再传一次。

    取不到返回 None 而不是报错：早期 checkpoint 没存配置，不该因此拒绝加载。
    """
    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return None
    config = ckpt.get(key)
    if not isinstance(config, Mapping):
        return None
    env_cfg = config.get("env")
    robot = env_cfg.get("robot") if isinstance(env_cfg, Mapping) else None
    if isinstance(robot, str):
        try:
            return get_spec(robot)
        except KeyError:
            return None
    return None


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def build_metadata(
    policy: Policy,
    *,
    spec: RobotSpec | None = None,
    contract: ObsContract | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """组装要写进 ONNX 的 metadata。

    default_angles 与 action_scale 是必需项（缺了端侧还原不出关节角），其余是
    诊断用信息：端侧要按段做误差归因，就得知道每段占哪些下标；要排查"是不是
    动作顺序错了"，就得有关节名。

    值统一用 JSON 序列化，读回来 parse_metadata_value 负责解析。ONNX 的
    metadata_props 只接受字符串，早期版本还要求必须是 ASCII——所以中文说明
    一律不往这里塞。
    """
    if spec is None:
        raise ValueError(
            "缺少 RobotSpec：default_angles / action_scale 只能从形态定义取，"
            "而端侧没有它们就还原不出关节目标角"
        )

    meta: dict[str, Any] = {
        "obs_dim": policy.obs_dim,
        "action_dim": policy.action_dim,
        "obs_mean": np.asarray(policy.obs_mean.detach().cpu()).tolist(),
        "obs_std": np.asarray(policy.obs_std.detach().cpu()).tolist(),
        "action_semantics": "target_pos = default_angles + action_scale * action",
    }

    if spec is not None:
        if spec.n_dof != policy.action_dim:
            raise ValueError(
                f"形态 {spec.name} 的关节数 {spec.n_dof} 与策略动作维度 "
                f"{policy.action_dim} 不一致，导出的 metadata 会误导端侧"
            )
        meta.update(
            {
                "robot": spec.name,
                "morphology": spec.morphology.value,
                "joint_names": list(spec.controlled_joints),
                "default_angles": list(spec.default_angles),
                # 标量 action_scale 展开成向量：端侧按逐关节乘法处理，不必分支。
                "action_scale": np.asarray(spec.action_scale_array).tolist(),
                "torque_limits": list(spec.torque_limits),
            }
        )

    if contract is not None:
        if contract.total_dim != policy.obs_dim:
            raise ValueError(
                f"观测契约总维 {contract.total_dim} 与策略 obs_dim {policy.obs_dim} 不一致"
            )
        # [[段名, 维度], ...]：端侧据此切片做误差归因
        meta["obs_segments"] = [[s.name, s.dim] for s in contract.segments]

    if extra:
        meta.update(extra)

    return {k: json.dumps(v) if not isinstance(v, str) else v for k, v in meta.items()}


# ---------------------------------------------------------------------------
# 自检报告
# ---------------------------------------------------------------------------


@dataclass
class ExportReport:
    """导出结果与自检结论。

    inputs / outputs 里的 None 表示该轴是动态的（批轴），端侧可以按任意
    batch 跑——控制回路用 1，离线批量回测可以更大。
    """

    path: Path
    inputs: dict[str, list[Any]] = field(default_factory=dict)
    outputs: dict[str, list[Any]] = field(default_factory=dict)
    opset: int = 0
    ir_version: int = 0
    num_nodes: int = 0
    num_params: int = 0
    size_bytes: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
    checker_passed: bool = False
    missing_meta: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def size_kb(self) -> float:
        return self.size_bytes / 1024.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "opset": self.opset,
            "ir_version": self.ir_version,
            "num_nodes": self.num_nodes,
            "num_params": self.num_params,
            "size_bytes": self.size_bytes,
            "checker_passed": self.checker_passed,
            "missing_metadata": self.missing_meta,
            "metadata": dict(self.metadata),
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        lines = [
            f"导出文件  {self.path}",
            f"体积      {self.size_kb:.1f} KB，参数 {self.num_params}，算子 {self.num_nodes}",
            f"opset     {self.opset}（ir_version={self.ir_version}）",
            f"输入      {self.inputs}",
            f"输出      {self.outputs}",
            f"checker   {'通过' if self.checker_passed else '未通过'}",
        ]
        if self.missing_meta:
            lines.append(f"metadata  缺少 {self.missing_meta}")
        else:
            angles = self.metadata.get("default_angles", "[]")
            scale = self.metadata.get("action_scale", "[]")
            lines.append(f"metadata  完整（default_angles 长度 {len(json.loads(angles))}，"
                         f"action_scale 长度 {len(json.loads(scale))}）")
        lines.extend(f"提示      {note}" for note in self.notes)
        return "\n".join(lines)


def inspect_onnx(path: str | Path, *, check: bool = True) -> ExportReport:
    """检查一个 ONNX 文件：图是否合法、输入输出形状、算子与参数量、metadata。

    量化之后同样可以用它复查——量化器有时会改掉图结构或丢掉 metadata，
    这两件事都会让端侧拿到一个"能跑但不对"的模型。
    """
    import onnx

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在：{path}")

    report = ExportReport(path=path, size_bytes=path.stat().st_size)
    model = onnx.load(str(path))

    if check:
        try:
            onnx.checker.check_model(model)
            report.checker_passed = True
        except Exception as exc:  # checker 的异常类型不稳定，统一转成报告结论
            report.checker_passed = False
            report.notes.append(f"onnx.checker 未通过：{exc}")

    report.ir_version = int(model.ir_version)
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            report.opset = int(opset.version)
    report.num_nodes = len(model.graph.node)
    report.metadata = {p.key: p.value for p in model.metadata_props}
    report.missing_meta = [k for k in META_KEYS if k not in report.metadata]

    # 参数量只数浮点/整型初始化器。量化后的模型还会多出 scale / zero_point
    # 这类小张量，数进去会让"模型有多大"这个数失真。
    report.num_params = int(
        sum(
            int(np.prod(t.dims))
            for t in model.graph.initializer
            if t.data_type in (1, 6, 7, 9, 10, 11) and t.dims
        )
    )

    def _shape(value_info: Any) -> list[Any]:
        dims = value_info.type.tensor_type.shape.dim
        return [d.dim_value if d.HasField("dim_value") else None for d in dims]

    graph = model.graph
    report.inputs = {v.name: _shape(v) for v in graph.input if v.name not in {i.name for i in graph.initializer}}
    report.outputs = {v.name: _shape(v) for v in graph.output}

    # 形状推断能查出"某一维推不出来"，这类图在端侧可能直接拒绝加载
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:
        report.notes.append(f"形状推断有告警（不一定影响推理）：{exc}")

    return report


# ---------------------------------------------------------------------------
# 导出主流程
# ---------------------------------------------------------------------------


def export_policy(
    policy: Policy,
    path: str | Path,
    *,
    spec: RobotSpec | None = None,
    contract: ObsContract | None = None,
    opset: int = 17,
    extra_metadata: Mapping[str, Any] | None = None,
    check: bool = True,
) -> ExportReport:
    """导出策略到 ONNX，写入 metadata，然后自检。

    归一化统计量随着 `Policy.forward` 一起进图，因此导出的模型是自包含的：
    端侧直接喂原始观测，不需要复现任何预处理代码。

    Args:
        spec: 形态定义，提供 default_angles / action_scale / 关节名。必填，
            否则端侧拿到动作也不知道该给关节发什么角度。
        contract: 观测契约，写进 metadata 供端侧切片。可选，但强烈建议给。
        opset: ONNX 算子集版本。17 是常见端侧运行时的安全选择。
        check: 导出后跑 onnx.checker 与形状推断。

    Returns:
        ExportReport，含路径、形状、参数量与 metadata。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(policy, spec=spec, contract=contract, extra=extra_metadata)

    fallback_reason: str | None = None
    try:
        policy.export_onnx(path, opset=opset, metadata=metadata)
    except Exception as exc:
        if not _needs_legacy_exporter(exc):
            raise
        # 走到这里通常是环境问题而不是模型问题：调用方可能传了自带导出的
        # Policy 实现，或落在某个 torch 版本上 dynamo 导出器与 onnxscript 对不上。
        # 回落到本模块的 TorchScript 路径，导出的图在数值上等价——等价性校验
        # 会证明这一点，失败原因也记进报告，不静默发生。
        fallback_reason = f"{type(exc).__name__}: {exc}"
        _export_with_torchscript(policy, path, opset=opset)

    # metadata 再写一遍。基类已经写过，这里不是补救而是兜底：本函数的对外承诺是
    # "导出完 metadata 一定完整"，不该依赖基类那条路径是否走通（回落路径同样会写，
    # 但调用方传入的 Policy 实现未必）。写重复是幂等的，代价只是多读一次文件。
    write_onnx_metadata(path, metadata)

    report = inspect_onnx(path, check=check)
    if fallback_reason:
        report.notes.append(f"基类导出不可用，已回落到 torchscript 导出器（{fallback_reason}）")
    return report


def _needs_legacy_exporter(exc: Exception) -> bool:
    """判断导出失败是不是"导出器/依赖不匹配"这类环境问题，而不是模型本身的问题。

    真出现模型层面的问题（例如 forward 里有不支持的控制流）时，回落路径同样会
    失败，那个错误会被抛出去，所以这里的宽容不会把真问题吞掉。
    """
    if isinstance(exc, TypeError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    hints = (
        "onnxscript",
        "metadata_props",
        "dynamo",
        "onnx_export",
        "not installed",
        "no module named",
    )
    return any(h in text for h in hints)


def _export_with_torchscript(policy: Policy, path: Path, *, opset: int) -> Path:
    """老导出器路径。

    单独写一份而不是包一层 wrapper，是因为 torch.onnx.export 的 dynamo 开关
    必须由这里显式传 False，通过基类调用拿不到这个参数。metadata 不在这一步
    写，由调用方统一处理——这个导出器在部分版本上不接受 metadata_props。
    """
    policy.eval()
    dummy = torch.zeros(1, policy.obs_dim, dtype=torch.float32)
    torch.onnx.export(
        policy,
        (dummy,),
        str(path),
        input_names=["obs"],
        output_names=["action"],
        opset_version=opset,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        dynamo=False,
    )
    return path
