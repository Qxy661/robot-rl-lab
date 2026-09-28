"""域随机化：把训练分布摊得比真机更宽。

仿真是确定的，真机不是。同一份策略在仿真里行走如飞、上真机就抖，最常见的原因
是策略把仿真里的某个常数当成了物理规律——地面摩擦恰好是 1.0、电池电压恰好够、
电机恰好能输出额定力矩。训练时把这些量按分布随机，策略就被迫学一种对参数不敏感
的步态，这是 sim2real 里性价比最高的一招。

两种时机，按 `reset` 分流：

- `reset=True`  每回合一次。摩擦、质量、电机强度、PD 增益这类**模型常数**，
  同一回合内保持不变（回合内外都在变的参数会让策略面对一个无解的非平稳问题）。
- `reset=False` 每步都可能发生。周期性推力这类**瞬时扰动**，随时可能来，策略
  必须能当场恢复而不是等到下一回合。

改的是 `model` 上的数组（摩擦、质量、增益），所以这里有个前提：环境必须持有
自己的一份模型。RobotModel.model 在多个环境实例间共享是允许的（loader 的注释
写明了），共享着做随机化会互相覆写，因此 MujocoEnv 在随机化生效时先复制模型。
"""

from __future__ import annotations

import mujoco
import numpy as np

from robotrl.assets.spec import RobotSpec
from robotrl.configs.schema import EventsConfig, ObsConfig

#: 摩擦系数在 geom_friction 里的哪一列。1=扭转摩擦，2=滚动摩擦，这两个不动。
_SLIDING_FRICTION = 0


class DomainRandomizer:
    """域随机化集合。

    每个组件都先记下模型里的标称值，随机化时写"标称值 × 因子"而不是"当前值 ×
    因子"。后者会一步步漂移：回合数一多，摩擦系数就慢慢爬到了分布之外，而且
    完全不可复现。
    """

    def __init__(
        self,
        config: EventsConfig,
        obs_config: ObsConfig,
        *,
        model: mujoco.MjModel,
        spec: RobotSpec,
        base_body_id: int,
        base_dof_adr: int,
    ) -> None:
        self.config = config
        self.obs_config = obs_config
        self.spec = spec
        self._model = model
        self._base_body_id = base_body_id
        self._base_dof_adr = base_dof_adr

        # 标称值快照。复制而不是引用：model 里的数组会被就地改写。
        self._nominal_geom_friction = model.geom_friction.copy()
        self._nominal_body_mass = model.body_mass.copy()
        self._nominal_gainprm = model.actuator_gainprm.copy()
        self._nominal_biasprm = model.actuator_biasprm.copy()
        self._nominal_gear = model.actuator_gear.copy()

        # 距离下一次推力还有多少步。None 表示尚未采样（首次 reset 时补上）。
        self._steps_until_push: int | None = None

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------

    @staticmethod
    def affects_model(config: EventsConfig) -> bool:
        """是否有组件会改写模型数组。

        环境用它决定要不要为每个实例复制一份模型，见模块文档。
        """
        return bool(
            config.randomize_friction
            or config.randomize_base_mass
            or config.randomize_motor_strength
            or config.randomize_pd_gain
        )

    def observation_noise(self, segment: str) -> float:
        """某观测段的噪声标准差。

        噪声尺度放在这里而不是观测组装那一段，是为了让"训练时加了多少噪声"和
        "加了哪些随机化"在同一个地方读得全，调 sim2real 时不用两头找。
        """
        if not self.obs_config.use_obs_noise:
            return 0.0
        return float(self.obs_config.noise_scale.get(segment, 0.0))

    # ------------------------------------------------------------------
    # 施加
    #
    # 两个公开方法对应两类时机，由环境在合适的钩子里调用：apply_model 在物理
    # 复位时，apply_push 在控制步的事件段。不合成一个 apply(reset=...) 是因为
    # 环境里的复位点不止一处（外部 reset 与回合结束时的内部复位），合成之后
    # 总有一处调不到。
    # ------------------------------------------------------------------

    def apply_model(self, data: mujoco.MjData, np_random: np.random.Generator) -> None:
        """随机化模型常数：摩擦、机身质量、电机强度、PD 增益。

        Args:
            data: 当前仿真状态。质量改动后要让 MuJoCo 重算派生常数。
            np_random: 环境的随机源，保证 reset(seed=...) 能复现整回合。
        """
        config = self.config
        model = self._model

        if config.randomize_friction:
            factors = np_random.uniform(*config.friction_range, size=model.ngeom)
            model.geom_friction[:, _SLIDING_FRICTION] = (
                self._nominal_geom_friction[:, _SLIDING_FRICTION] * factors
            )

        if config.randomize_base_mass:
            delta = np_random.uniform(*config.base_mass_delta)
            model.body_mass[:] = self._nominal_body_mass
            # 质量不能随机到非正数，否则惯性矩阵失去正定性，求解器会直接发散。
            model.body_mass[self._base_body_id] = max(
                1e-3, self._nominal_body_mass[self._base_body_id] + delta
            )

        if config.randomize_motor_strength:
            factors = np_random.uniform(*config.motor_strength_range, size=model.nu)
            # 缩 gear 等价于缩力矩常数：同样的 ctrl 得到成比例的力矩，
            # 模拟电池电压下降、电机个体差异。
            model.actuator_gear[:, 0] = self._nominal_gear[:, 0] * factors

        if config.randomize_pd_gain:
            # 位置执行器在 MuJoCo 里就是 general 执行器：出力 = kp*ctrl - kp*q - kd*qd。
            # 因此 kp 出现在 gainprm[0] 和 biasprm[1] 两处，必须同时改，只改一处
            # 会得到一个物理上不存在的执行器（例如刚度变了但稳态误差没变）。
            kp_factor = np_random.uniform(*config.pd_gain_range, size=model.nu)
            kd_factor = np_random.uniform(*config.pd_gain_range, size=model.nu)
            model.actuator_gainprm[:, 0] = self._nominal_gainprm[:, 0] * kp_factor
            model.actuator_biasprm[:, 1] = self._nominal_biasprm[:, 1] * kp_factor
            model.actuator_biasprm[:, 2] = self._nominal_biasprm[:, 2] * kd_factor

        # 质量进了派生常数（求解器用的逆权重等），改完必须让 MuJoCo 重算一遍。
        if config.randomize_base_mass:
            mujoco.mj_setConst(model, data)

    # ---- 瞬时扰动 ----

    def reschedule_push(self, np_random: np.random.Generator) -> None:
        """重排下一次推力的时刻。回合开始时调用，让推力不必卡在固定周期上。"""
        low, high = self.config.push_interval_steps
        self._steps_until_push = int(np_random.integers(low, high + 1))

    def apply_push(self, data: mujoco.MjData, np_random: np.random.Generator) -> None:
        """周期性给机身一个水平速度冲击。

        注意是直接改速度而不是施加外力：外力要靠求解器积分成速度，推力大小
        还要除以质量才说得清，而"被推了一下，速度突变 0.5 m/s"这句话本身就是
        工程上想表达的意思。
        """
        if not self.config.push_robot or self._base_dof_adr < 0:
            return
        if self._steps_until_push is None:
            self.reschedule_push(np_random)
        self._steps_until_push -= 1
        if self._steps_until_push > 0:
            return

        push = np_random.uniform(*self.config.push_velocity_xy, size=2)
        data.qvel[self._base_dof_adr : self._base_dof_adr + 2] += push
        self.reschedule_push(np_random)
