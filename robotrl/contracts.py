"""观测与动作契约：环境层、算法层、部署层三方共用的接口定义。

这个文件刻意放在包顶层而不是 envs/ 下面。原因是部署层（量化、基准、回测）
需要按名字切分观测向量来做误差归因，但它不应该因此被拖入 MuJoCo 依赖——
放在顶层，只依赖 numpy，谁都能用。

动作约定
--------
策略输出 action ∈ [-1, 1]^n_dof，是**归一化的关节位置增量**，不是力矩。
下发到仿真器前按下面两式还原：

    target_pos = default_angles + action_scale * action
    torque     = pd_kp * (target_pos - q) - pd_kd * qd

第一式在环境层做，第二式交给 MuJoCo 的 position actuator。把策略输出限制在
[-1, 1] 有两个好处：一是动作尺度与形态解耦，换机器人不用重调网络输出层；
二是导出的 ONNX 天然带上了输出范围约束，量化部署时不用再补 clip。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObsSegment:
    """观测向量里的一段。

    Attributes:
        name: 段名，也是误差归因时的键。
        dim: 长度。逐关节段的 dim 等于 spec.n_dof。
        description: 这一段装的是什么，写给人看。
    """

    name: str
    dim: int
    description: str = ""


@dataclass(frozen=True)
class ObsContract:
    """观测向量的布局契约。

    段与段之间靠拼接顺序定义，顺序即内存布局，不可随意调整——训练好的策略
    权重和导出的 ONNX 都依赖这个顺序。改动顺序等同于让旧 checkpoint 失效，
    所以这里提供 with_extra() 而不是"插到中间"的能力。

    所有取值方法都带 `...` 广播，因此同一份契约既能处理单个观测 (dim,)，
    也能处理批量观测 (num_envs, dim)。
    """

    segments: tuple[ObsSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("观测契约至少要有一段")

        names = [s.name for s in self.segments]
        if len(set(names)) != len(names):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"观测契约段名重复：{sorted(dupes)}")

        for s in self.segments:
            if s.dim <= 0:
                raise ValueError(f"观测段 {s.name!r} 维度必须为正，得到 {s.dim}")

    # ---- 维度与索引 ----

    @property
    def total_dim(self) -> int:
        return sum(s.dim for s in self.segments)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.segments)

    def index(self, name: str) -> slice:
        """取某一段在观测向量中的切片，用于 `obs[..., contract.index("cmd")]`。"""
        offset = 0
        for s in self.segments:
            if s.name == name:
                return slice(offset, offset + s.dim)
            offset += s.dim
        raise KeyError(f"观测契约中没有段 {name!r}，可用：{list(self.names)}")

    def dim_of(self, name: str) -> int:
        for s in self.segments:
            if s.name == name:
                return s.dim
        raise KeyError(f"观测契约中没有段 {name!r}，可用：{list(self.names)}")

    # ---- 拆装 ----

    def split(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """按段拆开观测。返回结果是视图，不复制内存。"""
        obs = np.asarray(obs)
        if obs.shape[-1] != self.total_dim:
            raise ValueError(
                f"观测末维应为 {self.total_dim}，实际 {obs.shape[-1]}"
            )
        return {s.name: obs[..., self.index(s.name)] for s in self.segments}

    def concat(self, parts: Mapping[str, np.ndarray]) -> np.ndarray:
        """按契约顺序把各段拼成观测向量。

        环境层每步都调用它，所以实现上先转换一次再堆叠，避免每段各转一次。
        """
        missing = set(self.names) - set(parts)
        if missing:
            raise KeyError(f"缺少观测段：{sorted(missing)}")
        extra = set(parts) - set(self.names)
        if extra:
            raise KeyError(f"契约外的观测段：{sorted(extra)}")

        arrays = [np.asarray(parts[s.name], dtype=np.float32) for s in self.segments]
        for s, arr in zip(self.segments, arrays, strict=True):
            if arr.shape[-1] != s.dim:
                raise ValueError(f"段 {s.name!r} 期望末维 {s.dim}，实际 {arr.shape[-1]}")
        return np.concatenate(arrays, axis=-1)

    # ---- 扩展 ----

    def with_extra(self, *segments: ObsSegment) -> ObsContract:
        """在末尾追加若干段，返回新契约（原契约不变）。

        critic 观测就是 policy 观测追加特权信息（真实机身速度、地形高度等）
        得到的，用这个方法构造，可以保证两者的公共前缀完全对齐。
        """
        return ObsContract(segments=self.segments + tuple(segments))

    def __repr__(self) -> str:
        body = ", ".join(f"{s.name}:{s.dim}" for s in self.segments)
        return f"ObsContract(dim={self.total_dim} | {body})"


# --------------------------------------------------------------------------
# 标准足式观测契约
#
# 十二维本体感觉（线速度/角速度/重力投影/速度指令）+ 三段逐关节量。
# 这套划分来自足式运动控制的通行做法：前三段回答"我的身体现在什么状态"，
# cmd 回答"我想去哪"，后三段回答"我的关节在哪、在怎么动、上一步做了什么"。
# 逐关节段排在后面，是因为它们的长度随身形态变化，放末尾能让固定段的下标
# 在所有形态上保持一致。
# --------------------------------------------------------------------------

_BASE_SEGMENTS: tuple[ObsSegment, ...] = (
    ObsSegment("lin_vel", 3, "机体系线速度估计"),
    ObsSegment("ang_vel", 3, "机体系角速度"),
    ObsSegment("proj_gravity", 3, "重力方向在机体系下的投影，表征姿态"),
    ObsSegment("cmd", 3, "速度指令 [vx, vy, yaw_rate]"),
)

_PER_JOINT_SEGMENTS: tuple[ObsSegment, ...] = (
    ObsSegment("dof_pos", 0, "关节位置与默认姿态之差"),
    ObsSegment("dof_vel", 0, "关节速度"),
    ObsSegment("last_action", 0, "上一步的动作，让策略能感知自己的输出历史"),
)

#: 固定段的段数，部署层做误差归因时用来区分"与形态无关"和"与形态有关"的段。
NUM_FIXED_SEGMENTS = len(_BASE_SEGMENTS)


def make_obs_contract(n_dof: int) -> ObsContract:
    """构造标准足式观测契约。n_dof 由 RobotSpec 提供，因此天然跨形态。"""
    if n_dof <= 0:
        raise ValueError(f"n_dof 必须为正，得到 {n_dof}")

    per_joint = tuple(
        ObsSegment(s.name, n_dof, s.description) for s in _PER_JOINT_SEGMENTS
    )
    return ObsContract(segments=_BASE_SEGMENTS + per_joint)


def make_privileged_obs_contract(
    n_dof: int,
    *,
    num_feet: int = 4,
    terrain_samples: int = 9,
) -> ObsContract:
    """构造 critic 专用的特权观测契约。

    额外给三段训练期真值：机身线速度、足端接触标志、脚下地形高度采样。
    critic 只在训练时存在、不参与部署，所以可以拿到策略拿不到的信息——
    这是 asymmetric actor-critic 的核心，用廉价的特权信息换更稳的梯度。

    Args:
        num_feet: 双足填 2，四足填 4，由调用方从 morphology 推出后传入。
        terrain_samples: 地形高度采样点数，常见做法是在机身下方取 3x3 网格。
    """
    return make_obs_contract(n_dof).with_extra(
        ObsSegment("base_lin_vel_gt", 3, "机身线速度真值，critic 特权信息"),
        ObsSegment("feet_contact", num_feet, "各足接触状态标志"),
        ObsSegment("terrain_height", terrain_samples, "机身下方地形高度采样"),
    )
