"""MuJoCo 模型加载：把 RobotSpec 变成可仿真的模型，并建立关节索引映射。

这一层要解决的问题只有一个，但很关键：**RobotSpec 里的关节顺序和 MuJoCo 内部
的关节顺序不是一回事**。RobotSpec 的 15 个 G1 关节在 MuJoCo 模型里散落在 29
个关节中间，而且 qpos、qvel、actuator 三套数组各有一套编号。策略输出的第 3 个
动作对应的是哪根关节、该写到 ctrl 的哪个位置，全部要靠这里的映射来回答。

映射一旦错位，仿真照跑、不报错，只是学不会——这是最难查的一类 bug，所以
索引映射必须在加载时就确定下来，并作为 RobotModel 的一部分交给环境层，
而不是让环境层每步现算。

另外，被控关节之外的关节（例如 G1 的手臂）需要施加"姿态保持"力矩锁在默认
角度。这些关节的索引也在这里一并算好。

ctrl 的语义
-----------
加载完成后，`data.ctrl` 的每一个位置都统一是**目标关节角**。上游模型并不是
按这个约定给的：G1 配的是 position 伺服（gainprm/biasprm 里带 kp=500 和阻尼），
H1、Go2 配的是纯力矩电机（ctrlrange 直接是 ±200 N·m 这样的力矩上限）。语义不
统一，环境层就得按机器人分支，正是这个框架要避免的事。

所以 loader 在编译前把全部关节执行器统一成 MuJoCo 的 position 伺服：

    力矩 = pd_kp * (ctrl - q) - pd_kd * qd，并在 torque_limits 处饱和

pd_kp / pd_kd 取自 RobotSpec，写进 gainprm/biasprm；torque_limits 写进
forcerange，让伺服在额定力矩处饱和而不是无限出力。于是环境层的动作换算只剩
一行，换机器人不用改：

    data.ctrl[model.actuator_ids] = default_angles + action_scale * action
    data.ctrl[model.held_actuator_ids] = model.held_targets

覆盖上游增益是刻意的：RobotSpec 声明了 PD 增益，如果模型里还留着上游那套
kp=500，spec 里的数字就成了没人读的死数据，读代码的人会被误导。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from robotrl.assets.spec import RobotSpec

#: 环境变量名。设置后优先从它指向的目录找 Menagerie，便于把模型放在别处。
MENAGERIE_ENV_VAR = "ROBOTRL_MENAGERIE"

#: 未设置环境变量时的默认位置，与 scripts/fetch_assets.py 的检出目标一致。
DEFAULT_MENAGERIE = Path(__file__).resolve().parent.parent.parent / "assets" / "menagerie"

#: 找不到模型时给出的修复命令。
_FETCH_CMD = "python scripts/fetch_assets.py"

#: 保持环的增益。上游只给了力矩电机（H1）时没有现成的位置环增益可抄，
#: 按额定力矩折算：每 N·m 额定力矩配 10 的比例增益，即 0.1 rad 的跟踪误差
#: 就出满额定力矩。这是估算值，够硬能压住手臂自重，又不至于把接触力放大。
_HELD_KP_PER_NM = 10.0

#: 保持环的 kd/kp 比例。G1 的 position 类用 dampratio=1 自动推出的阻尼约为
#: kp 的 0.086 倍，这里取 0.1，与上游的临界阻尼同量级。
_HELD_KD_RATIO = 0.1

#: 从 keyframe 取保持角度时优先用的名字，都没有则退回第 0 个 keyframe。
_PREFERRED_KEYS = ("home", "stand")


def find_menagerie() -> Path:
    """定位 Menagerie 仓库根目录。

    找不到时给出可执行的修复命令，而不是抛一个光秃秃的 FileNotFoundError——
    首次运行的失败信息质量，直接决定这个项目能不能被陌生人跑起来。

    这里只确认目录存在，不检查里面有没有具体某个机器人：那是 model_path 的
    事，分层报错才能说清"是仓库没拉，还是这个机器人没检出"。
    """
    candidates: list[Path] = []
    env_dir = os.environ.get(MENAGERIE_ENV_VAR)
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(DEFAULT_MENAGERIE)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        f"找不到 MuJoCo Menagerie 目录，已尝试：\n{tried}\n"
        f"在仓库根目录运行 `{_FETCH_CMD}` 拉取模型（稀疏检出，只取用到的三个），\n"
        f"或设置环境变量 {MENAGERIE_ENV_VAR} 指向已有的检出目录。"
    )


def model_path(spec: RobotSpec, *, scene: str = "scene.xml") -> Path:
    """给出某形态的场景文件路径。

    报错分成两级：子目录不存在（可能是稀疏检出漏了）和目录在但文件名不对，
    分别列出实际有什么，省得翻仓库。
    """
    root = find_menagerie()
    model_dir = root / spec.menagerie_dir

    if not model_dir.is_dir():
        available = sorted(p.name for p in root.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"{root} 下没有 {spec.menagerie_dir!r}（形态 {spec.name!r}）。\n"
            f"已检出的模型目录：{available}\n"
            f"在仓库根目录运行 `{_FETCH_CMD}` 补齐。"
        )

    path = model_dir / scene
    if not path.is_file():
        available = sorted(p.name for p in model_dir.glob("*.xml"))
        raise FileNotFoundError(
            f"{model_dir} 下没有 {scene!r}。该目录已有的 XML：{available}"
        )
    return path


@dataclass
class RobotModel:
    """加载完成的机器人模型，以及 RobotSpec 顺序与 MuJoCo 内部顺序的映射。

    Attributes:
        model: 编译好的 MuJoCo 模型，可被多个 MjData 共享（只读）。
        spec: 对应的形态定义。
        actuator_ids: (n_dof,) 被控关节对应的 actuator 索引，动作按此顺序写入 ctrl。
        qpos_ids: (n_dof,) 被控关节在 qpos 中的位置索引。
        dof_ids: (n_dof,) 被控关节在 qvel 中的速度索引。
        held_actuator_ids: 其余 actuator 的索引，用于施加姿态保持力矩。
        held_targets: 这些关节要保持的角度，取自模型 keyframe。
    """

    model: mujoco.MjModel
    spec: RobotSpec
    actuator_ids: np.ndarray
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    held_actuator_ids: np.ndarray
    held_targets: np.ndarray

    def new_data(self) -> mujoco.MjData:
        """新建一份仿真状态。每个环境实例都需要独立的 MjData，模型可以共享。"""
        return mujoco.MjData(self.model)

    @property
    def n_dof(self) -> int:
        return self.spec.n_dof


def load_robot(
    spec: RobotSpec,
    *,
    scene: str = "scene.xml",
    mutate: Callable[[mujoco.MjSpec], None] | None = None,
) -> RobotModel:
    """加载机器人模型并建立关节映射。

    Args:
        spec: 形态定义。
        scene: 场景文件名，默认用 Menagerie 提供的 scene.xml。
        mutate: 可选的编辑钩子，在编译前对 MjSpec 做修改。环境层用它在场景里
            插入地形，这样 loader 不需要知道地形是怎么回事。
            钩子在 loader 改完执行器之后调用，因此地形代码看到的是一个已经
            按 RobotSpec 配好增益的模型。钩子不要增删关节或执行器——映射是
            按名字建立的，增删会让下面的索引与动作对不上。

    Returns:
        RobotModel，含模型与全部索引映射。
    """
    path = model_path(spec, scene=scene)
    mj_spec = mujoco.MjSpec.from_file(str(path))

    _configure_actuators(mj_spec, spec, path)

    if mutate is not None:
        mutate(mj_spec)

    model = mj_spec.compile()
    return _build_model(model, spec, path)


# --------------------------------------------------------------------------
# 执行器配置：统一 ctrl 语义（见模块文档）
# --------------------------------------------------------------------------


def _joint_map(mj_spec: mujoco.MjSpec) -> dict[str, mujoco.MjsJoint]:
    """按名字索引关节。匿名关节（例如 Go2 的浮动基座）不参与映射。"""
    return {j.name: j for j in mj_spec.joints if j.name}


def _rated_torque(act: mujoco.MjsActuator, joint: mujoco.MjsJoint) -> float | None:
    """执行器的额定力矩，用于伺服的出力饱和。

    两个来源，按可信度排序：关节自己声明了 actuatorfrcrange 就用它（G1 的做法，
    逐关节写死在 XML 里）；否则退回执行器的 ctrlrange 幅值——H1、Go2 把力矩电机
    的额定力矩直接写在 ctrlrange 上。位置伺服的 ctrlrange 是关节角范围，不能
    当力矩用，所以只在 biastype 为 none（纯力矩驱动）时才看它。
    """
    if np.any(np.asarray(joint.actfrcrange) != 0.0):
        return float(abs(joint.actfrcrange[0]))

    ctrl = np.asarray(act.ctrlrange, dtype=np.float64)
    if act.biastype == mujoco.mjtBias.mjBIAS_NONE and ctrl[1] > ctrl[0]:
        return float(max(abs(ctrl[0]), abs(ctrl[1])))
    return None


def _configure_actuators(mj_spec: mujoco.MjSpec, spec: RobotSpec, path: Path) -> None:
    """把所有关节执行器改成 position 伺服，并按 RobotSpec 配好增益与饱和力矩。

    受控关节的增益一律用 spec 的值；被保持的关节（G1 的手臂）spec 里没有数据，
    上游本身就是位置伺服就沿用它自己的增益，是力矩电机则按额定力矩估算。
    """
    joints = _joint_map(mj_spec)
    unknown = [n for n in spec.controlled_joints if n not in joints]
    if unknown:
        raise ValueError(
            f"{spec.name}: {path.name} 里没有这些关节：{unknown}\n"
            f"模型中实际的关节名：{sorted(joints)}"
        )

    # strict=True：长度一致是 RobotSpec 已经校验过的契约，真被破坏就该在这里炸
    gains = dict(
        zip(
            spec.controlled_joints,
            zip(spec.pd_kp, spec.pd_kd, spec.torque_limits, strict=True),
            strict=True,
        )
    )

    for act in mj_spec.actuators:
        if act.trntype != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(
                f"{spec.name}: 执行器 {act.name!r} 不是关节执行器（trntype={act.trntype}）。"
                "映射与保持力矩都建立在「一个执行器驱动一个关节」之上，"
                "其他类型的执行器无法给出目标关节角。"
            )
        joint = joints.get(act.target)
        if joint is None:
            raise ValueError(
                f"{spec.name}: 执行器 {act.name!r} 指向未知关节 {act.target!r}，"
                f"模型中实际的关节名：{sorted(joints)}"
            )

        torque = _rated_torque(act, joint)
        if joint.name in gains:
            kp, kd, torque = gains[joint.name]
        elif act.biastype == mujoco.mjtBias.mjBIAS_AFFINE:
            # 上游已经是位置伺服：gainprm[0]=kp，biasprm[1]=-kp，biasprm[2]=-kd。
            # 保持类关节的刚度是上游按整机调过的，比我们估算的可信。
            kp = float(act.gainprm[0])
            kd = float(-act.biasprm[2])
        elif torque is not None:
            kp = _HELD_KP_PER_NM * torque
            kd = _HELD_KD_RATIO * kp
        else:
            raise ValueError(
                f"{spec.name}: 执行器 {act.name!r} 既没有位置环增益，也没有可用的"
                "力矩上限，无法推算保持增益。请把它加入 controlled_joints，"
                "或在模型里给关节补上 actuatorfrcrange。"
            )

        act.set_to_position(kp=kp, kv=kd)

        # 目标角限制在关节行程内。MuJoCo 会自己裁剪 ctrl，等于给动作又加了一层
        # 安全网：即使策略输出越界，伺服也不会追一个到不了的角。
        joint_range = np.asarray(joint.range, dtype=np.float64)
        if joint_range[1] > joint_range[0]:
            act.ctrlrange = joint_range

        if torque is not None:
            act.forcerange = np.array([-torque, torque], dtype=np.float64)


# --------------------------------------------------------------------------
# 索引映射
# --------------------------------------------------------------------------


def _hold_targets(model: mujoco.MjModel, joint_ids: list[int]) -> np.ndarray:
    """被保持关节的目标角，取自模型 keyframe。

    keyframe 是模型作者留下的一个自洽位形（Go2 的蹲伏、H1 的微屈膝），拿它当
    保持目标，手臂就不会和腿打架。没有 keyframe 时退回 qpos0，那是 MJCF 里
    定义的参考位形，也比凭空写 0 强。
    """
    key_id: int | None = None
    for preferred in _PREFERRED_KEYS:
        for k in range(model.nkey):
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) == preferred:
                key_id = k
                break
        if key_id is not None:
            break
    if key_id is None and model.nkey > 0:
        key_id = 0

    source = model.key_qpos[key_id] if key_id is not None else model.qpos0
    return np.array([source[model.jnt_qposadr[j]] for j in joint_ids], dtype=np.float64)


def _build_model(model: mujoco.MjModel, spec: RobotSpec, path: Path) -> RobotModel:
    """按 RobotSpec 的顺序建立三套索引，并找出所有被保持的关节。"""
    joint_id_by_name: dict[str, int] = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name:
            joint_id_by_name[name] = j

    unknown = [n for n in spec.controlled_joints if n not in joint_id_by_name]
    if unknown:
        raise ValueError(
            f"{spec.name}: {path.name} 里没有这些关节：{unknown}\n"
            f"模型中实际的关节名：{sorted(joint_id_by_name)}"
        )

    actuators_of: dict[int, list[int]] = {}
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(
                f"{spec.name}: 执行器 {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)!r} "
                "不是关节执行器，无法映射到关节角。"
            )
        actuators_of.setdefault(int(model.actuator_trnid[a][0]), []).append(a)

    actuator_ids: list[int] = []
    qpos_ids: list[int] = []
    dof_ids: list[int] = []
    for name in spec.controlled_joints:
        jid = joint_id_by_name[name]
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(
                f"{spec.name}: 受控关节 {name!r} 不是铰链关节，"
                "它的 qpos 不是标量，无法用单个索引读写。"
            )
        acts = actuators_of.get(jid, [])
        if not acts:
            raise ValueError(f"{spec.name}: 受控关节 {name!r} 在模型里没有对应的执行器")
        if len(acts) > 1:
            raise ValueError(
                f"{spec.name}: 受控关节 {name!r} 挂了 {len(acts)} 个执行器 {acts}，"
                "一个动作该写给谁无法确定"
            )
        actuator_ids.append(acts[0])
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        dof_ids.append(int(model.jnt_dofadr[jid]))

    controlled_acts = set(actuator_ids)
    held_actuator_ids: list[int] = []
    held_joint_ids: list[int] = []
    for a in range(model.nu):
        if a in controlled_acts:
            continue
        jid = int(model.actuator_trnid[a][0])
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            raise ValueError(
                f"{spec.name}: 被保持的执行器 {name!r} 驱动的不是铰链关节，"
                "无法用单个目标角锁住它。要么把它加入 controlled_joints，"
                "要么在模型里去掉它。"
            )
        held_actuator_ids.append(a)
        held_joint_ids.append(jid)

    return RobotModel(
        model=model,
        spec=spec,
        actuator_ids=np.asarray(actuator_ids, dtype=np.int64),
        qpos_ids=np.asarray(qpos_ids, dtype=np.int64),
        dof_ids=np.asarray(dof_ids, dtype=np.int64),
        held_actuator_ids=np.asarray(held_actuator_ids, dtype=np.int64),
        held_targets=_hold_targets(model, held_joint_ids),
    )
