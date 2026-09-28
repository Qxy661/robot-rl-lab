"""内嵌的最小 MJCF：单腿加一条胳膊的浮动机器人。

环境层的测试不该依赖 Menagerie。拉模型要网络，模型大了每个用例还慢，而这些
用例要验的是管线——动作换算、观测拼装、复位与随机化的时序、终止与超时的分流
——跟机器人长什么样没有关系。所以这里直接内嵌一段 MJCF，编译成本不到一毫秒。

模型的选型都是为了让某条管线可测，不是为了让机器人好走：

- 一条腿两个铰链（hip / knee），用来验"动作 → ctrl"的换算和逐关节观测。
- 一条胳膊一个铰链，**不在 RobotSpec 里**，用来验未受控关节的姿态保持：
  它必须每步都被写进 ctrl，否则会被受控关节的写入覆盖成 0。
- 足端是名为 foot 的叶节点，用来验足端发现与接触检测。
- keyframe 给了几个关节的参考角度，loader 的 held_targets 约定就是从它取的。

机器人站不稳也无所谓：这里没有一条用例在考察"能不能走路"。
"""

from __future__ import annotations

import mujoco
import numpy as np

from robotrl.assets.loader import RobotModel
from robotrl.assets.spec import Morphology, RobotSpec, register

#: 内嵌 MJCF。基座带自由关节（和真实机器人一样），三个铰链各有一个 position
#: 执行器。ctrlrange 有意留成默认的 (0,0)，即不限幅，这样测动作换算时不会
#: 被裁剪切掉差异。
MINI_XML = """
<mujoco model="mini">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <default>
    <geom friction="1.0 0.005 0.0001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <body name="base" pos="0 0 0.45">
      <freejoint name="base_free"/>
      <geom name="base_geom" type="capsule" fromto="0 0 0 0 0 0.12" size="0.05"/>
      <body name="thigh" pos="0 0 0">
        <joint name="hip_joint" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03"/>
        <body name="shank" pos="0 0 -0.2">
          <joint name="knee_joint" type="hinge" axis="0 1 0" range="-2.5 0"/>
          <geom name="shank_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03"/>
          <body name="foot" pos="0 0 -0.2">
            <geom name="foot_geom" type="sphere" size="0.04"/>
          </body>
        </body>
      </body>
      <body name="arm" pos="0 0 0.12">
        <joint name="arm_joint" type="hinge" axis="0 1 0" range="-1.0 1.0"/>
        <geom name="arm_geom" type="capsule" fromto="0 0 0 0 0 0.15" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="hip_actuator" joint="hip_joint" kp="40" kv="1.5"/>
    <position name="knee_actuator" joint="knee_joint" kp="40" kv="1.5"/>
    <position name="arm_actuator" joint="arm_joint" kp="5" kv="0.5"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0.45 1 0 0 0 0.15 -0.35 0.2"/>
  </keyframe>
</mujoco>
"""

#: 只控两个腿关节；arm_joint 交给环境做姿态保持。这个不对称是刻意的，
#: 它让"未受控关节"这条路径在测试里始终有覆盖。
MINI_SPEC = register(
    RobotSpec(
        name="mini",
        morphology=Morphology.BIPED,
        menagerie_dir="",
        controlled_joints=("hip_joint", "knee_joint"),
        default_angles=(0.15, -0.35),
        pd_kp=(40.0, 40.0),
        pd_kd=(1.5, 1.5),
        torque_limits=(20.0, 20.0),
        action_scale=0.25,
        notes="测试用的最小形态，单腿加一条胳膊，不依赖 Menagerie。",
    )
)

#: keyframe 里 arm_joint 的角度。未受控关节的保持目标就该是它。
MINI_HELD_TARGET = 0.2

#: 与 keyframe 一致的初始机身高度。
MINI_BASE_HEIGHT = 0.45


def build_robot_model(
    spec: RobotSpec = MINI_SPEC,
    xml: str = MINI_XML,
    *,
    model: mujoco.MjModel | None = None,
    keyframe: int = 0,
) -> RobotModel:
    """从 MJCF 字符串建出与 loader 字段语义一致的 RobotModel。

    这里复刻的是 loader 的"约定"而不是它的代码：关节名到 qpos/dof/actuator 下标
    的映射、未受控执行器与它们的保持目标。因此它既能支撑环境层的测试，也能在
    loader 就绪后当作一份可对照的参考实现。

    差别只有来源：模型来自内嵌字符串，而不是 Menagerie 的场景文件。传入已编译
    好的 `model` 可以跳过编译，用于"地形已经注入进去"的那一类用例。

    Args:
        model: 已编译的模型。给了就用它，关节名按名字查。
    """
    model = model if model is not None else mujoco.MjModel.from_xml_string(xml)
    actuator_ids: list[int] = []
    qpos_ids: list[int] = []
    dof_ids: list[int] = []

    for joint_name in spec.controlled_joints:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint < 0:
            raise ValueError(f"模型里没有关节 {joint_name!r}")
        qpos_ids.append(int(model.jnt_qposadr[joint]))
        dof_ids.append(int(model.jnt_dofadr[joint]))
        actuator = _actuator_for_joint(model, joint)
        if actuator < 0:
            raise ValueError(f"关节 {joint_name!r} 没有对应的执行器")
        actuator_ids.append(actuator)

    controlled = set(actuator_ids)
    held = [actuator for actuator in range(model.nu) if actuator not in controlled]
    held_targets = [
        float(model.key_qpos[keyframe][model.jnt_qposadr[model.actuator_trnid[actuator, 0]]])
        for actuator in held
    ]

    return RobotModel(
        model=model,
        spec=spec,
        actuator_ids=np.asarray(actuator_ids, dtype=int),
        qpos_ids=np.asarray(qpos_ids, dtype=int),
        dof_ids=np.asarray(dof_ids, dtype=int),
        held_actuator_ids=np.asarray(held, dtype=int),
        held_targets=np.asarray(held_targets, dtype=float),
    )


def compile_with_terrain(terrain_config, *, xml: str = MINI_XML) -> mujoco.MjModel:
    """把地形注入场景后编译，用来验地形与环境层的衔接。

    走的是与真实加载相同的路径（MjSpec → mutate → compile），只是模型换成
    内嵌的那一份。
    """
    from robotrl.envs.terrain import apply_terrain

    scene = mujoco.MjSpec.from_string(xml)
    apply_terrain(scene, terrain_config)
    return scene.compile()


def _actuator_for_joint(model: mujoco.MjModel, joint: int) -> int:
    """找出驱动某关节的执行器。关节到执行器不是一一对应，必须查表。"""
    for actuator in range(model.nu):
        if model.actuator_trnid[actuator, 0] == joint:
            return actuator
    return -1
