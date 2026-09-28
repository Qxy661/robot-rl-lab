"""质点速度跟踪：不依赖 MuJoCo 的最小环境。

这个环境存在的意义是**验证契约本身**，而不是验证物理。它完全不碰 MuJoCo，
只用几行解析动力学，因此几秒钟就能跑完一次训练，可以在改完任何一层之后
立刻确认"全链路还是通的"——训练、导出、量化、回测，四个环节都能在分钟级
跑完并给出数字。

顺带它也是环境的参考实现：8 段管线每一段都填了，且都保持在一两行以内。
写一个新任务环境时照着它抄结构即可。

任务本身是速度跟踪：一个质点在平面上运动，每 10 步换一次目标速度，奖励
速度与目标的接近程度。这个任务与足式的速度跟踪任务同构（同样的指令-跟踪
结构），但去掉了平衡、接触、步态这些难点，因此 PPO 能在几十秒内学到明显
的策略，学习曲线肉眼可见地上升——这对新环境是否接对是个很有效的信号。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robotrl.assets.spec import Morphology, RobotSpec, register
from robotrl.contracts import ObsContract, ObsSegment
from robotrl.envs.base_env import BaseEnv, Obs
from robotrl.envs.registry import register_env

# 质点用两个虚拟关节表示平面上的两个推进方向。morphology 标为 POINT，
# 任务层据此跳过一切与足式相关的奖励项。这份 spec 也说明 RobotSpec 并不
# 绑真实机器人——只要给得出维度，环境层就能工作。
TOY_SPEC = register(
    RobotSpec(
        name="toy",
        morphology=Morphology.POINT,
        menagerie_dir="",
        controlled_joints=("推进_x", "推进_y"),
        default_angles=(0.0, 0.0),
        pd_kp=(1.0, 1.0),
        pd_kd=(0.0, 0.0),
        torque_limits=(1.0, 1.0),
        action_scale=2.0,
        notes="质点速度跟踪，不依赖 MuJoCo，用于分钟级验证全链路。",
    )
)


class ToyVelocityEnv(BaseEnv):
    """二维质点速度跟踪。

    状态量只有位置和速度，动作直接解释为加速度，没有中间的执行器模型——
    真实环境里这一层由 MuJoCo 的 position actuator 承担。
    """

    #: 速度误差的容忍尺度，奖励 = exp(-误差²/σ)。σ 越小，策略越要贴紧目标。
    _SIGMA = 0.25
    #: 动作代价权重，抑制无意义的抖动输出。
    _ACTION_COST = 0.01
    #: 位置越界判定，防止策略学会"跑出地图逃避跟踪"。
    _BOUND = 5.0
    #: 每隔多少步重采样一次指令。指令变化太快，策略来不及响应，价值估计
    #: 也会因为回报的非平稳性变差。
    _CMD_HOLD_STEPS = 10

    def __init__(
        self,
        *,
        spec: RobotSpec | None = None,
        config: Any = None,
        max_episode_steps: int = 200,
        control_dt: float = 0.05,
        seed: int = 0,
    ) -> None:
        # config 优先于构造参数，这样命令行覆写才有意义；显式传参又优先于
        # config，方便在测试里精确控制单个用例。
        if config is not None:
            max_episode_steps = config.env.max_episode_steps
            control_dt = config.env.control_dt
            seed = config.env.seed
        super().__init__(spec or TOY_SPEC, max_episode_steps=max_episode_steps)
        self.dt = control_dt
        self._np_random = np.random.default_rng(seed)
        self._reset_sim()
        self._update_command()
        self._apply_events(reset=True)

    # ------------------------------------------------------------------
    # 观测契约：这里刻意不套用足式契约，改用与任务匹配的三段式。
    # 契约本身是通用的，足式那套只是 make_obs_contract() 这个工厂的产物。
    # ------------------------------------------------------------------

    @property
    def obs_contract(self) -> ObsContract:
        return ObsContract(
            segments=(
                ObsSegment("pos", 2, "平面位置"),
                ObsSegment("vel", 2, "平面速度"),
                ObsSegment("cmd", 2, "目标速度"),
            )
        )

    @property
    def critic_obs_contract(self) -> ObsContract:
        # 阻尼系数是随机的、策略观察不到，但 critic 知道它就更容易解释回报差异。
        return self.obs_contract.with_extra(
            ObsSegment("damping", 1, "真实阻尼系数，策略不可见，仅 critic 使用")
        )

    # ------------------------------------------------------------------
    # 8 段管线
    # ------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray) -> None:
        # 动作语义：归一化加速度，乘 action_scale 后落到 ±2 m/s²
        self._accel = action * self.spec.action_scale
        self._last_action = action.copy()

    def _simulate(self) -> None:
        # 一步欧拉积分。阻尼乘在速度上模拟空气阻力，让"不施加加速度就减速"。
        self._vel = self._vel * self._damping + self._accel * self.dt
        self._pos = self._pos + self._vel * self.dt

    def _terminate(self) -> bool:
        return bool(np.linalg.norm(self._pos) > self._BOUND)

    def _reward(self) -> float:
        tracking = np.exp(-np.sum((self._vel - self._cmd) ** 2) / self._SIGMA)
        effort = self._ACTION_COST * float(np.sum(self._accel**2))
        return float(tracking - effort)

    def _reset_sim(self) -> None:
        # 初始位置加一点噪声，避免所有回合从同一点起步导致策略过拟合初始状态
        self._pos = self.np_random.uniform(-0.5, 0.5, size=2)
        self._vel = np.zeros(2)
        self._accel = np.zeros(2)
        self._last_action = np.zeros(2)
        self._cmd = np.zeros(2)
        self._damping = 0.98
        self._steps_since_cmd = 0

    def _update_command(self) -> None:
        self._steps_since_cmd += 1
        if self._steps_since_cmd >= self._CMD_HOLD_STEPS:
            self._cmd = self.np_random.uniform(-1.0, 1.0, size=2)
            self._steps_since_cmd = 0

    def _apply_events(self, *, reset: bool) -> None:
        if reset:
            # 每回合随机阻尼，让策略不要依赖某一个固定的减速规律
            self._damping = float(self.np_random.uniform(0.96, 0.995))

    def _observe(self) -> Obs:
        policy = np.concatenate([self._pos, self._vel, self._cmd]).astype(np.float32)
        critic = np.concatenate([policy, [self._damping]]).astype(np.float32)
        return Obs(policy=policy, critic=critic)


def _factory(**kwargs: Any) -> BaseEnv:
    return ToyVelocityEnv(**kwargs)


register_env("toy", _factory)
