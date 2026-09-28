"""足式运动任务的通用奖励项。

这些项与形态无关，只通过环境暴露的工具方法读状态，因此 G1、H1、Go2 以及以后
新增的形态共用同一份实现。分段的判据是"它想约束什么"：

- **跟住指令**：线速度跟踪、角速度跟踪。任务的骨架，没有它们剩下的项会退化成
  "站着不动最省事"。
- **身体别乱动**：机身竖直速度、俯仰横滚角速度、姿态平直。策略有个偷懒的捷径：
  原地高抬腿也能让跟踪项拿到分。这几项把与指令无关的运动逐一卖贵。
- **站得对**：机身高度保持。人形和四足都有一个"省力的站高"，蹲着或者站得过高
  都不该拿满分。
- **动作别抖**：动作变化率、动作幅度、关节速度、力矩。真机上的抖动来自策略
  输出的高频成分，仿真里的能量代价先把它压下去；力矩项同时是能耗的代理。
- **脚别蹭地**：足端滑动。落地瞬间的横向拖拽是仿真里最常见、真机上最伤
  电机与减速器的动作。

每项都返回"越大越好"的量：奖励项直接为正，惩罚项自带负号。这样总奖励就是
加权和，不用在心里记住哪些项是负的。
"""

from __future__ import annotations

import numpy as np

from robotrl.envs.managers.reward_manager import register_reward

#: 跟踪项的高斯核宽度。奖励取 exp(-误差²/σ)，σ 越小越"只有贴着指令才有分"。
#: 0.25 对应速度误差 0.5 m/s 时奖励降到约 0.37，是足式里的通行取值：既要求跟得住，
#: 又不会因为起步阶段的必然误差把奖励压成 0 从而使梯度消失。
_TRACKING_SIGMA = 0.25

#: 竖直速度惩罚里"1 m/s 的弹跳"值多少，与其它项对比使用的量级参考。
_VZ_SCALE = 1.0


@register_reward("track_lin_vel_xy")
def track_lin_vel_xy(env) -> float:
    """线速度跟踪：exp(-‖v_xy - cmd_xy‖²/σ)。

    在机体系比较而不是世界系：策略看到的状态就在机体系，指令也定义成"朝我正在
    面对的方向走多快"，两边一致才不用在策略里额外做一次坐标变换。
    """
    err = env.base_linear_velocity_body()[:2] - env.command[:2]
    return float(np.exp(-float(np.sum(err**2)) / _TRACKING_SIGMA))


@register_reward("track_ang_vel_z")
def track_ang_vel_z(env) -> float:
    """角速度跟踪：exp(-(ω_z - cmd_yaw)²/σ)。转弯速度跟不住，路径就走不准。"""
    err = env.base_angular_velocity_body()[2] - env.command[2]
    return float(np.exp(-float(err**2) / _TRACKING_SIGMA))


@register_reward("lin_vel_z")
def lin_vel_z(env) -> float:
    """机身竖直速度惩罚：-v_z²。

    走路的质心本来就该上下起伏，但策略会把"弹跳"当成推进手段——蹬地起跳再落地，
    平均速度也能上去。这项让弹跳的收益转负。
    """
    return -_VZ_SCALE * float(env.base_linear_velocity_body()[2] ** 2)


@register_reward("ang_vel_xy")
def ang_vel_xy(env) -> float:
    """俯仰与横滚角速度惩罚：-‖ω_xy‖²。抑制机身摇摆，与姿态平直项互补：
    那项管"歪不歪"，这项管"晃不晃"。"""
    return -float(np.sum(env.base_angular_velocity_body()[:2] ** 2))


@register_reward("orientation")
def orientation(env) -> float:
    """姿态平直：-‖g_xy‖²，g 是重力方向在机体系下的投影。

    ‖g_xy‖² = sin²(倾角)，近似等于倾角的平方，所以在小倾角附近和 -θ² 等价，
    无需开方也不必处理三角函数的分支。机身直立时该项为 0，是它可能取到的最大值。
    """
    gravity = env.projected_gravity()
    return -float(np.sum(gravity[:2] ** 2))


@register_reward("base_height")
def base_height(env) -> float:
    """机身高度保持：-(h - h_target)²。

    目标高度由形态给出（spec 换形态就换值），不平地地形上可以按脚下高度调整。
    没有这项，策略会找到一个"蹲着慢慢挪"的解：跟踪项还能拿分，姿态项也满足，
    只是站姿已经不像个机器人了。
    """
    err = env.base_height() - env.base_height_target
    return -float(err**2)


@register_reward("action_rate")
def action_rate(env) -> float:
    """动作变化率惩罚：-‖a_t - a_{t-1}‖²。动作平滑。

    惩罚的是"变化"而不是"大小"，所以策略仍然可以自由地输出大幅度动作，只要它
    连续。真机上的电流尖峰、减速器冲击基本都来自相邻两步之间的跳变。
    """
    diff = env.last_action - env.previous_action
    return -float(np.sum(diff**2))


@register_reward("action_l2")
def action_l2(env) -> float:
    """动作幅度惩罚：-‖a‖²。与力矩项的区别：这项约束指令层，力矩项约束物理层。

    有 PD 控制的机器人上两者不等价——同样的动作幅度，在关节被外力顶住时力矩会
    大得多。一般只用其中一项，两项都开要一起把权重调小。
    """
    return -float(np.sum(env.last_action**2))


@register_reward("joint_vel")
def joint_vel(env) -> float:
    """关节速度惩罚：-‖q̇‖²。关节转得太快时，真机上编码器延迟、控制周期
    离散带来的误差都被放大，仿真里看不出来，上板就是抖。"""
    return -float(np.sum(env.joint_velocities() ** 2))


@register_reward("torques")
def torques(env) -> float:
    """力矩惩罚：-‖τ‖²。能耗的代理量。

    用平方而不是绝对值：平方对大输出更敏感，而大输出正是发热和电池消耗的
    主要来源。真实能耗与 ∫|τ·q̇| 成正比，但那要额外乘关节速度，且对静止站立
    的抖动更宽容，不如平方项稳。
    """
    return -float(np.sum(env.joint_efforts() ** 2))


@register_reward("feet_slip")
def feet_slip(env) -> float:
    """足端滑动惩罚：-Σ_c 触地足 ‖v_xy‖²。

    只算触地的脚：摆动腿正该快速移动，算进去等于惩罚抬腿。落地瞬间的横向拖拽
    在仿真里只表现为摩擦损耗，在真机上直接磨损减速器。
    """
    contacts = env.foot_contacts()
    if contacts.size == 0:
        return 0.0
    velocities = env.foot_velocities()[:, :2]
    return -float(np.sum(np.sum(velocities**2, axis=1) * contacts))
