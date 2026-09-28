"""速度跟踪任务：平地上跟住随机的机身速度指令。

这是足式运动里最基础也最该先跑通的任务。它只要求"按指令走"，不涉及地形、
不涉及参考动作，因此形态之间可以直接对比——G1 和 Go2 用同一份代码，差别只在
spec 给的关节数与默认站姿。

指令是机身坐标系下的 [vx, vy, yaw_rate]，每若干步换一次。之所以要换：固定指令
下策略可以退化成"找到一个朝那个方向的步态然后死记硬背"，换指令逼它学会"根据
目标调整步态"这件真正的事。也不能换太快，策略来不及响应时，回报里混进的是
它学不会的噪声。默认 5 秒一换（50Hz 下 250 步）。

任务代码里没有任何一处写死了具体机器人的关节数、默认角度或增益——那些全部
来自 RobotSpec。换机器人改的是配置里的 `env.robot`，这份文件一行不用动。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robotrl.assets.spec import Morphology, RobotSpec
from robotrl.envs.mujoco_env import MujocoEnv
from robotrl.envs.registry import register_env

#: 速度指令的采样范围 [min, max]（米/秒，弧度/秒）。
#: 侧向范围比前后小：横着走对足式而言代价高得多，一开始就要求 ±1 m/s 的侧移
#: 会让大部分回合在"起步就把自己绊倒"里消耗掉，学习信号被淹没。
_DEFAULT_LIN_VEL_X = (-1.0, 1.0)
_DEFAULT_LIN_VEL_Y = (-0.5, 0.5)
_DEFAULT_ANG_VEL_YAW = (-1.0, 1.0)

#: 指令保持步数。50Hz 下 250 步 = 5 秒，够走出一段稳定的步态。
_DEFAULT_CMD_HOLD_STEPS = 250

#: 指令置零的概率。总要有一部分回合格指令是"原地站住"，否则策略学到的
#: 永远是"只要有指令就往前冲"，而静止站立恰恰是上板后最常被要求的能力。
_DEFAULT_ZERO_COMMAND_PROB = 0.1

#: 默认奖励权重。量级参考主流足式实现，正值是任务目标，负值是代价：
#: 跟踪两项之和最大 1.5，各项惩罚加起来在零点几的量级，既不喧宾夺主，
#: 又足以让"抖着走"和"蹲着挪"的解拿不到高分。
_DEFAULT_REWARD_SCALES: dict[str, float] = {
    "track_lin_vel_xy": 1.0,   # 线速度跟踪，任务主目标
    "track_ang_vel_z": 0.5,    # 角速度跟踪，转不了弯就走不了曲线
    "lin_vel_z": -2.0,         # 抑制上下弹跳，弹跳会被策略当成推进手段
    "ang_vel_xy": -0.05,       # 抑制机身俯仰横滚的摇摆
    "orientation": -1.0,       # 机身保持竖直
    "base_height": -10.0,      # 保持站高，防止"蹲着挪"这种退化解
    "action_rate": -0.01,      # 动作平滑，真机的冲击来自相邻步的跳变
    "action_l2": -0.005,       # 动作幅度，抑制无意义的持续输出
    "joint_vel": -0.001,       # 关节速度，过快时真机的编码器延迟会被放大
    "torques": -0.0001,        # 力矩平方，能耗与发热的代理量
    "feet_slip": -0.1,         # 足端滑动，落地拖拽最伤减速器
}


class VelocityEnv(MujocoEnv):
    """速度跟踪。平地、随机指令、足式通用奖励项。

    失败判据沿用环境基类的四条：状态发散、机身过低、倾角过大、非足端触地。
    平地速度跟踪不需要额外的判据——这四条已经覆盖了"摔了"的全部情形。

    指令相关的三个范围都可以从构造参数覆写（`make("Go2-Velocity", lin_vel_range=...)`），
    因为不同机器人的能力差得多：让 Go2 追 1 m/s 是热身，让 G1 追同样的速度
    一开始就是不可能任务。
    """

    def __init__(
        self,
        *,
        lin_vel_x_range: tuple[float, float] | None = None,
        lin_vel_y_range: tuple[float, float] | None = None,
        ang_vel_yaw_range: tuple[float, float] | None = None,
        cmd_hold_steps: int = _DEFAULT_CMD_HOLD_STEPS,
        zero_command_prob: float = _DEFAULT_ZERO_COMMAND_PROB,
        **kwargs: Any,
    ) -> None:
        # 先赋值再交给基类构造：基类构造的末尾会调用 _reset_sim，那里要用到
        # 指令保持步数。
        self.lin_vel_x_range = tuple(lin_vel_x_range or _DEFAULT_LIN_VEL_X)
        self.lin_vel_y_range = tuple(lin_vel_y_range or _DEFAULT_LIN_VEL_Y)
        self.ang_vel_yaw_range = tuple(ang_vel_yaw_range or _DEFAULT_ANG_VEL_YAW)
        self.cmd_hold_steps = int(cmd_hold_steps)
        self.zero_command_prob = float(zero_command_prob)
        if self.cmd_hold_steps <= 0:
            raise ValueError("cmd_hold_steps 必须为正")

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # 任务配置
    # ------------------------------------------------------------------

    def default_reward_scales(self) -> dict[str, float]:
        scales = dict(_DEFAULT_REWARD_SCALES)
        if self.spec.morphology is Morphology.POINT:
            # 没有足端的形态算不出足端滑动，留着权重只会得到一个恒为 0 的项。
            scales.pop("feet_slip", None)
        return scales

    @property
    def command_ranges(self) -> dict[str, tuple[float, float]]:
        """三个指令维度的采样范围，供日志与文档引用。"""
        return {
            "lin_vel_x": self.lin_vel_x_range,
            "lin_vel_y": self.lin_vel_y_range,
            "ang_vel_yaw": self.ang_vel_yaw_range,
        }

    # ------------------------------------------------------------------
    # 管线：指令
    # ------------------------------------------------------------------

    def _reset_sim(self) -> None:
        super()._reset_sim()
        # 指令清零，并把计数推到阈值上，让复位后的第一次 _update_command
        # 立刻采出一条新指令——否则第一步的观测里 cmd 是上一回合的残留值。
        self._command = np.zeros(3)
        self._steps_since_command = self.cmd_hold_steps

    def _update_command(self) -> None:
        """按保持步数重采样指令。"""
        self._steps_since_command += 1
        if self._steps_since_command < self.cmd_hold_steps:
            return
        self._steps_since_command = 0

        if self.np_random.random() < self.zero_command_prob:
            self._command = np.zeros(3)
            return

        self._command = np.array(
            [
                self.np_random.uniform(*self.lin_vel_x_range),
                self.np_random.uniform(*self.lin_vel_y_range),
                self.np_random.uniform(*self.ang_vel_yaw_range),
            ]
        )


def _factory(*, spec: RobotSpec | None = None, config: Any = None, **kwargs: Any) -> VelocityEnv:
    return VelocityEnv(spec=spec, config=config, **kwargs)


register_env("velocity", _factory)
