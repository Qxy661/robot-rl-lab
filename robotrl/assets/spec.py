"""机器人形态契约：RobotSpec。

RobotSpec 描述一个机器人身上"不随任务变化"的静态属性——有哪些关节、默认
姿态是什么、PD 增益多少、力矩上限多少。任务代码只通过它读取维度，因此换
机器人不需要改任务代码，只换一个 spec。

两个设计要点值得说明：

1. 只描述"被控关节"。MJCF 里其余关节（例如 G1 的手臂）不进入 RobotSpec，
   由环境层施加姿态保持 PD，锁在 keyframe 默认角度上。这样冻结任意自由度
   都不用改模型文件，对三种形态用同一套逻辑。

2. 数值字段统一用 tuple 而不是 np.ndarray。frozen dataclass 要求字段可比较
   且行为可预期，而 numpy 数组的 `==` 返回逐元素结果而非布尔值，相等性断言
   和去重都会出问题。需要数组时用 .xxx_array 属性现场转换。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Morphology(str, Enum):
    """形态分类。

    影响任务层的默认 reward 项启用（例如足式都关心的步态对称项）、
    默认站姿高度约定、以及是否含腰关节。

    POINT 用于质点、简化模型和非足式形态（如无人机、车辆）。框架的契约
    本身与形态无关，这个枚举只是让任务层能按大类启用不同的奖励项。
    """

    BIPED = "biped"
    QUADRUPED = "quadruped"
    POINT = "point"


@dataclass(frozen=True)
class RobotSpec:
    """机器人的静态属性契约。

    Attributes:
        name: 形态唯一标识，也是注册表键，例如 "g1"。
        morphology: 足式形态分类。
        menagerie_dir: 在 MuJoCo Menagerie 仓库中的子目录名，loader 据此定位模型。
        controlled_joints: 被策略控制的关节名，顺序即 action 顺序，长度即 n_dof。
        default_angles: 各被控关节的默认角度（弧度），与 controlled_joints 一一对应。
        pd_kp: 位置环比例增益，逐关节。
        pd_kd: 位置环微分增益，逐关节。
        torque_limits: 力矩上限（N·m），逐关节，用于动作裁剪。
        action_scale: 策略输出到关节位置增量的缩放。标量表示所有关节共用，
            也可给逐关节序列（例如腿部大步幅、腰部小步幅）。
        notes: 这个形态值得记一笔的说明，会出现在 `python scripts/info.py` 输出里。
    """

    name: str
    morphology: Morphology
    menagerie_dir: str
    controlled_joints: tuple[str, ...]
    default_angles: tuple[float, ...]
    pd_kp: tuple[float, ...]
    pd_kd: tuple[float, ...]
    torque_limits: tuple[float, ...]
    action_scale: float | tuple[float, ...] = 0.25
    notes: str = ""

    def __post_init__(self) -> None:
        n = len(self.controlled_joints)
        if n == 0:
            raise ValueError(f"{self.name}: controlled_joints 不能为空")

        for field_name in ("default_angles", "pd_kp", "pd_kd", "torque_limits"):
            got = len(getattr(self, field_name))
            if got != n:
                raise ValueError(
                    f"{self.name}: {field_name} 长度 {got} 与 controlled_joints 长度 {n} 不一致"
                )

        if isinstance(self.action_scale, tuple) and len(self.action_scale) != n:
            raise ValueError(
                f"{self.name}: action_scale 长度 {len(self.action_scale)} 与关节数 {n} 不一致"
            )

        if len(set(self.controlled_joints)) != n:
            dupes = {j for j in self.controlled_joints if self.controlled_joints.count(j) > 1}
            raise ValueError(f"{self.name}: controlled_joints 有重复项 {sorted(dupes)}")

        if np.any(np.asarray(self.pd_kp) < 0) or np.any(np.asarray(self.pd_kd) < 0):
            raise ValueError(f"{self.name}: PD 增益不能为负")

        if np.any(np.asarray(self.torque_limits) <= 0):
            raise ValueError(f"{self.name}: torque_limits 必须为正")

    # ---- 维度 ----

    @property
    def n_dof(self) -> int:
        """被控关节数。obs/action 中所有逐关节分段的长度都是它。

        由 controlled_joints 推导而非独立字段，避免两者不一致。
        """
        return len(self.controlled_joints)

    # ---- 数组视图：训练/仿真热路径上按需转换 ----

    @property
    def default_angles_array(self) -> np.ndarray:
        return np.asarray(self.default_angles, dtype=np.float64)

    @property
    def pd_kp_array(self) -> np.ndarray:
        return np.asarray(self.pd_kp, dtype=np.float64)

    @property
    def pd_kd_array(self) -> np.ndarray:
        return np.asarray(self.pd_kd, dtype=np.float64)

    @property
    def torque_limits_array(self) -> np.ndarray:
        return np.asarray(self.torque_limits, dtype=np.float64)

    @property
    def action_scale_array(self) -> np.ndarray:
        """(n_dof,) 的缩放向量，标量输入会被广播成向量。"""
        return (
            np.full(self.n_dof, self.action_scale, dtype=np.float64)
            if isinstance(self.action_scale, (int, float))
            else np.asarray(self.action_scale, dtype=np.float64)
        )


# --------------------------------------------------------------------------
# 注册表：任务层用 get_spec("g1") 拿形态，不直接 import 具体机器人模块，
# 这样加一种新形态只需要 import 一次（在 assets/__init__.py 里），
# 任务代码一行都不用动。
# --------------------------------------------------------------------------

_REGISTRY: dict[str, RobotSpec] = {}


def register(spec: RobotSpec) -> RobotSpec:
    """注册一个形态。重复注册同名 spec 会报错，避免静默覆盖。"""
    if spec.name in _REGISTRY and _REGISTRY[spec.name] is not spec:
        raise ValueError(f"形态 {spec.name!r} 已被注册过，不能覆盖")
    _REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> RobotSpec:
    """按名字取形态。名字错误时列出所有可用项，省去翻源码。"""
    if name not in _REGISTRY:
        raise KeyError(f"未注册的形态 {name!r}，可用：{sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_specs() -> list[str]:
    return sorted(_REGISTRY)
