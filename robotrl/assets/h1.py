"""Unitree H1 人形机器人形态定义。

H1 是比 G1 更重、更早一代的人形平台，全身 19 个自由度，腿部结构与 G1 接近
（每腿 5 个关节 + 躯干 1 个），但腿更长、质量更大，对 PD 增益和动作幅度的
要求不同。

与 G1 的差异集中体现在几处，恰好说明「RobotSpec 作为契约」的价值：默认姿态
弯得更深（膝关节 0.8 对 0.3）、髋部增益更高（150 对 100）、有躯干关节而没有
腰部的三个关节、力矩上限大一档（膝盖 300 N·m 对 139）。这些差异全部收在数据
里，环境层和算法层一行代码都不用改。
"""

from __future__ import annotations

from robotrl.assets.spec import Morphology, RobotSpec, register

# H1 每腿 5 个关节：没有独立的踝 roll，靠髋 roll 完成侧向平衡。
# 关节名没有 `_joint` 后缀，和 G1、Go2 的命名习惯不同，写错一个字符 loader 就会
# 报错并列出模型里实际的关节名。
#
# 髋部按 pitch/roll/yaw 排，与 G1 一致而不是照抄 MJCF 里的 yaw/roll/pitch。
# 两处理由：默认角度的数值是按这个顺序填的，顺序换了数据就对不上（曾经踩过，
# 表现为机器人一站起来就歪）；双足形态用同一套顺序，共用的人形奖励项也能直接
# 按下标取关节，不必先查名字。
_CONTROLLED_JOINTS = (
    # 左腿
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle",
    # 右腿
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle",
    # 躯干
    "torso",
)

# 取自 Menagerie `scene.xml` 的 keyframe `home`，逐项核对一致（腿三关节之和为
# -0.4 + 0.8 - 0.4 = 0，足底水平）。H1 腿长明显大于 G1，同样的膝关节角能提供
# 更大的离地高度，所以站姿更直。
#
# 这个姿态的重心几乎压在脚跟边缘（实测机体系 x 偏后 8 cm，足底长约 15 cm），
# 只靠关节 PD 站不住，仿真里约 1 秒就会向后倒。这是模型自带的 home 姿态的性质，
# 不是增益问题——把 kp 提高到 16 倍同样会倒。它仍然是合适的默认姿态：双足本来
# 就要靠策略闭环维持平衡，默认姿态只决定回合从哪儿开始。任务层的终止判定要把
# 摔倒兜住。
_DEFAULT_ANGLES = (
    -0.40,
    0.0,
    0.0,
    0.80,
    -0.40,
    -0.40,
    0.0,
    0.0,
    0.80,
    -0.40,
    0.0,
)

# 估算：沿用 Unitree H1 RL 配置的量级。上游 MJCF 里 H1 用的是纯力矩电机
# （<motor> + ctrlrange），模型文件里没有位置环增益可抄，只能按整机调校经验给。
# 整机质量大，重力矩对关节的压迫更强，因此髋部增益高于 G1；踝关节仍保持
# 较低刚度，避免地面接触力被放大成高频抖动。
_PD_KP = (
    150.0,
    150.0,
    150.0,
    200.0,
    40.0,
    150.0,
    150.0,
    150.0,
    200.0,
    40.0,
    300.0,
)

_PD_KD = (
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    6.0,
)

# 取自 Menagerie `h1.xml` 各 <motor> 的 ctrlrange：髋 200、膝 300、踝 40、
# 躯干 200。loader 会把它从 ctrlrange 挪到 forcerange，语义不变。
_TORQUE_LIMITS = (
    200.0,
    200.0,
    200.0,
    300.0,
    40.0,
    200.0,
    200.0,
    200.0,
    300.0,
    40.0,
    200.0,
)

H1_SPEC = register(
    RobotSpec(
        name="h1",
        morphology=Morphology.BIPED,
        menagerie_dir="unitree_h1",
        controlled_joints=_CONTROLLED_JOINTS,
        default_angles=_DEFAULT_ANGLES,
        pd_kp=_PD_KP,
        pd_kd=_PD_KD,
        torque_limits=_TORQUE_LIMITS,
        action_scale=0.25,
        notes=(
            "19 自由度人形，本项目控制腿部 10 + 躯干 1，手臂 8 个关节冻结在 "
            "keyframe `home` 的角度上（全 0，即自然下垂）。关节名无 `_joint` "
            "后缀，与 G1、Go2 不同。"
        ),
    )
)
