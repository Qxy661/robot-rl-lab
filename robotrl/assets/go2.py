"""Unitree Go2 四足机器人形态定义。

Go2 是这里唯一的四足形态，也是三形态里最容易训的：12 个自由度、静态稳定、
摔倒不心疼。它在本项目里承担两个角色——一是让算法在简单形态上先跑通，
二是作为「同一套代码换 YAML 就能换形态」的证明。

四足与双足的差异落在数据里：默认姿态是蹲伏的（大腿前摆、小腿后折），
PD 增益比人形低一个量级（质量小、关节力矩小），没有腰关节。
"""

from __future__ import annotations

from robotrl.assets.spec import Morphology, RobotSpec, register

# FL/FR/RL/RR = 前左 / 前右 / 后左 / 后右，每个腿三个关节：
# hip 负责侧摆，thigh 负责前后摆，calf 负责膝部伸缩。
_CONTROLLED_JOINTS = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

# 取自 Menagerie `go2.xml` 的 keyframe `home`，逐项核对一致。大腿前摆 0.9 弧度、
# 小腿后折 -1.8 弧度，两者相抵后足端大致落在髋部正下方——足端在髋正下方时，
# 地面反作用力不产生绕髋力矩，静立最省力。这个姿态是四足的通用起点。
_DEFAULT_ANGLES = (
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
    0.0, 0.9, -1.8,
)

# 估算：沿用 Unitree Go2 RL 配置的量级。上游 MJCF 里 Go2 用的是纯力矩电机，
# 模型文件里没有位置环增益可抄。四足整机质量远小于人形，关节力矩需求也小，
# 增益相应低一个量级。
_PD_KP = (20.0,) * 12
_PD_KD = (0.5,) * 12

# 取自 Menagerie `go2.xml` 各 <motor> 的 ctrlrange：髋外摆、大腿 23.7，小腿
# 45.43（小腿是唯一要独自撑住整机重量的关节，电机规格高一档）。
_TORQUE_LIMITS = (
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
)

GO2_SPEC = register(
    RobotSpec(
        name="go2",
        morphology=Morphology.QUADRUPED,
        menagerie_dir="unitree_go2",
        controlled_joints=_CONTROLLED_JOINTS,
        default_angles=_DEFAULT_ANGLES,
        pd_kp=_PD_KP,
        pd_kd=_PD_KD,
        torque_limits=_TORQUE_LIMITS,
        action_scale=0.25,
        notes=(
            "12 自由度四足，全关节受控，没有需要冻结的关节；三形态中训练成本"
            "最低，适合先跑通流程。默认姿态与力矩上限均已按 Menagerie "
            "go2.xml 核对。"
        ),
    )
)
