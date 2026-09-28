"""Unitree G1 人形机器人形态定义。

G1 全身 29 个自由度，本项目只控制其中 15 个：双腿 12 + 腰 3。手臂 14 个关节
不进入 RobotSpec，加载时由 loader 锁在模型 keyframe `stand` 给出的角度上（肘部
弯 1.28 rad）。这么做的原因很实在：手臂对行走本身贡献很小，却会让动作空间从
15 维膨胀到 29 维，样本效率明显下降。臂部动作是单独的任务（操作、抓取），
不该和行走混在一起训。

冻结方式用的是"不控但施加保持力矩"，而不是改 MJCF 删掉 actuator。好处是
模型文件不动，换回全自由度只是改这一份 spec 的事。
"""

from __future__ import annotations

from robotrl.assets.spec import Morphology, RobotSpec, register

# 被控关节的顺序就是动作向量的顺序，训练好的权重与这个顺序绑定，
# 调整顺序等于让旧 checkpoint 失效。
_CONTROLLED_JOINTS = (
    # 左腿
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # 右腿
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # 腰
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

# 略屈膝的站姿，来源是 Unitree 自家 RL 配置里 G1 的默认姿态，与模型 keyframe
# `stand` 不同（那个是直腿站立）。取前者是因为完全伸直的腿在奇异位形附近，
# 雅可比退化，控制会变得敏感；保持一点弯曲既远离奇异，也给落脚留出缓冲余量。
# 三处角度之和为 0（-0.1 + 0.3 - 0.2），足底保持水平。
_DEFAULT_ANGLES = (
    -0.10,
    0.0,
    0.0,
    0.30,
    -0.20,
    0.0,
    -0.10,
    0.0,
    0.0,
    0.30,
    -0.20,
    0.0,
    0.0,
    0.0,
    0.0,
)

# 估算：沿用 Unitree G1 RL 配置的量级。上游 MJCF 里是 kp=500、dampratio=1 的
# 位置伺服（biasprm 推出 kd 约 43），那是按位置伺服调的一套很硬的增益；RL 里
# 用这么高的刚度会把接触力放大成抖动，所以按行业惯例给 100 量级。
# 膝的增益明显高于踝：它承重最大、需要更强的位置保持；踝关节靠近地面，
# 过高的刚度会把地面反作用力直接放大成抖动。
_PD_KP = (
    100.0,
    100.0,
    100.0,
    150.0,
    40.0,
    40.0,
    100.0,
    100.0,
    100.0,
    150.0,
    40.0,
    40.0,
    100.0,
    100.0,
    100.0,
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
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
)

# 逐关节额定力矩，取自 Menagerie `g1.xml` 各 <joint> 的 actuatorfrcrange：
# 髋俯仰/偏航 88、髋侧摆 139、膝 139、踝 50、腰偏航 88、腰侧摆/俯仰 50。
# loader 会把它写成伺服的 forcerange，伺服在这个力矩处饱和，而不是无限出力。
_TORQUE_LIMITS = (
    88.0,
    139.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    139.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    50.0,
    50.0,
)

G1_SPEC = register(
    RobotSpec(
        name="g1",
        morphology=Morphology.BIPED,
        menagerie_dir="unitree_g1",
        controlled_joints=_CONTROLLED_JOINTS,
        default_angles=_DEFAULT_ANGLES,
        pd_kp=_PD_KP,
        pd_kd=_PD_KD,
        torque_limits=_TORQUE_LIMITS,
        action_scale=0.25,
        notes=(
            "29 自由度人形，本项目控制腿部 12 + 腰部 3，手臂 14 个关节冻结在 "
            "keyframe `stand` 的角度上（肘 1.28 rad）。关节名与力矩上限均已按 "
            "Menagerie g1.xml 核对。"
        ),
    )
)
