"""模型加载与关节映射测试。

这一层最值得测的不是"能不能跑起来"，而是**索引有没有指向正确的关节**。映射
错位的典型症状是仿真照跑、不报错、策略学不会——训练几个小时才发现，代价很高。
所以这里用 mujoco 自己的关节名反查一遍：反查对得上，映射就是对的。

需要真实模型文件的用例标了 slow：它们依赖 `python scripts/fetch_assets.py`
检出的 Menagerie。没检出时自动跳过，而不是报一堆 FileNotFoundError——首次克隆
仓库的人跑 `pytest` 不该被没拉模型挡住。
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robotrl.assets import RobotSpec, get_spec
from robotrl.assets import loader as loader_module
from robotrl.assets.loader import find_menagerie, load_robot, model_path

# 三种内置形态，参数化时反复用。
MORPHOLOGIES = ("g1", "h1", "go2")


def _menagerie_available() -> bool:
    try:
        find_menagerie()
    except FileNotFoundError:
        return False
    return True


MENAGERIE_AVAILABLE = _menagerie_available()

#: 需要真实模型文件的用例：既标 slow（语义上属于慢用例），又能在没拉模型时跳过。
#: 两个都需要——slow 说明它为什么慢，skipif 保证缺模型时测试仍然是绿的。
needs_models = pytest.mark.skipif(
    not MENAGERIE_AVAILABLE,
    reason="未检出 Menagerie，先运行 python scripts/fetch_assets.py",
)


def _joint_name_of(model, actuator_id: int) -> str:
    """反查执行器驱动的关节名。映射校验全靠它。"""
    joint_id = int(model.actuator_trnid[actuator_id][0])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    assert name is not None, f"关节 {joint_id} 没有名字，测试无法反查"
    return name


# ---------------------------------------------------------------------------
# 路径定位：不依赖模型文件，报错信息本身就是要测的东西
# ---------------------------------------------------------------------------


def test_env_var_takes_priority(tmp_path, monkeypatch):
    """环境变量指向已有目录时优先用它，便于把模型放在仓库外。"""
    monkeypatch.setenv(loader_module.MENAGERIE_ENV_VAR, str(tmp_path))
    assert find_menagerie() == tmp_path.resolve()


def test_missing_menagerie_error_is_actionable(tmp_path, monkeypatch):
    """找不到模型时的报错要给出修复命令，而不是一句"文件不存在"。"""
    monkeypatch.delenv(loader_module.MENAGERIE_ENV_VAR, raising=False)
    monkeypatch.setattr(loader_module, "DEFAULT_MENAGERIE", tmp_path / "nope")

    with pytest.raises(FileNotFoundError) as excinfo:
        find_menagerie()

    message = str(excinfo.value)
    assert "fetch_assets.py" in message
    assert loader_module.MENAGERIE_ENV_VAR in message


def test_model_path_reports_missing_robot_dir(tmp_path, monkeypatch):
    """稀疏检出漏了某个机器人目录时，列出实际检出了哪些，省得翻仓库。"""
    (tmp_path / "unitree_g1").mkdir()
    monkeypatch.setenv(loader_module.MENAGERIE_ENV_VAR, str(tmp_path))

    with pytest.raises(FileNotFoundError, match="unitree_go2"):
        model_path(get_spec("go2"))


def test_model_path_reports_missing_scene(tmp_path, monkeypatch):
    """目录在但文件名不对时，列出该目录下有哪些 XML。"""
    model_dir = tmp_path / "unitree_g1"
    model_dir.mkdir()
    (model_dir / "g1.xml").write_text("<mujoco/>", encoding="utf-8")
    monkeypatch.setenv(loader_module.MENAGERIE_ENV_VAR, str(tmp_path))

    with pytest.raises(FileNotFoundError) as excinfo:
        model_path(get_spec("g1"))

    message = str(excinfo.value)
    assert "scene.xml" in message
    assert "g1.xml" in message


@needs_models
def test_model_path_points_into_menagerie():
    """路径规则：<menagerie>/<menagerie_dir>/<scene>。"""
    spec = get_spec("g1")
    assert model_path(spec).name == "scene.xml"
    assert model_path(spec).parent.name == spec.menagerie_dir


# ---------------------------------------------------------------------------
# 三种形态都能加载，映射长度与关节数一致
# ---------------------------------------------------------------------------


@needs_models
@pytest.mark.slow
@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_loads_and_mapping_lengths_match(name):
    spec = get_spec(name)
    robot = load_robot(spec)

    n = spec.n_dof
    assert robot.actuator_ids.shape == (n,)
    assert robot.qpos_ids.shape == (n,)
    assert robot.dof_ids.shape == (n,)
    assert robot.held_targets.shape == (len(robot.held_actuator_ids),)
    assert robot.n_dof == n


@needs_models
@pytest.mark.slow
@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_every_actuator_is_either_controlled_or_held(name):
    """执行器不能被漏掉：漏掉的那个会一直停在 ctrl=0，等于关节失控。"""
    robot = load_robot(get_spec(name))
    covered = np.concatenate([robot.actuator_ids, robot.held_actuator_ids])
    assert sorted(covered.tolist()) == list(range(robot.model.nu))
    # 两个集合不能有交集，否则同一个执行器会被写两种目标
    assert not set(robot.actuator_ids) & set(robot.held_actuator_ids)


@needs_models
@pytest.mark.slow
@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_mapping_points_at_the_named_joint(name):
    """核心用例：反查 mujoco 的关节名，逐个确认映射没有错位。

    actuator 顺序、qpos 位置、qvel 位置三套编号各查一遍。少查一套，就可能出现
    "动作写对了关节、读状态读错了关节"这种半对半错的映射。
    """
    spec = get_spec(name)
    robot = load_robot(spec)
    model = robot.model

    for i, joint_name in enumerate(spec.controlled_joints):
        actuator_id = int(robot.actuator_ids[i])
        assert _joint_name_of(model, actuator_id) == joint_name

        joint_id = int(model.actuator_trnid[actuator_id][0])
        assert robot.qpos_ids[i] == model.jnt_qposadr[joint_id]
        assert robot.dof_ids[i] == model.jnt_dofadr[joint_id]


@needs_models
@pytest.mark.slow
@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_qpos_ids_address_the_right_coordinate(name):
    """再验一层：把探针值写进 qpos_ids，用名字取回来的状态必须跟着变。

    上一条测的是索引算得对不对，这一条测的是"按这个索引读写真的落在预期关节
    上"——两件事分开测，出错时能直接定位到是哪一层的问题。
    """
    spec = get_spec(name)
    robot = load_robot(spec)
    model, data = robot.model, robot.new_data()

    probes = np.linspace(-0.13, 0.13, spec.n_dof)
    data.qpos[robot.qpos_ids] = probes
    data.qvel[robot.dof_ids] = probes * 2.0

    for i, joint_name in enumerate(spec.controlled_joints):
        joint_id = int(model.actuator_trnid[robot.actuator_ids[i]][0])
        assert model.jnt_qposadr[joint_id] == robot.qpos_ids[i]
        assert data.qpos[robot.qpos_ids[i]] == pytest.approx(probes[i])
        assert data.qvel[robot.dof_ids[i]] == pytest.approx(probes[i] * 2.0)
        # 关节名与 spec 声明一致，排除"索引对、名字不对"的情况
        assert _joint_name_of(model, robot.actuator_ids[i]) == joint_name


@needs_models
@pytest.mark.slow
@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_data_instances_are_independent(name):
    """模型可共享，状态必须独立：多个环境实例共用一份编译好的模型。"""
    robot = load_robot(get_spec(name))
    first, second = robot.new_data(), robot.new_data()
    first.qpos[robot.qpos_ids] = 0.5

    assert not np.allclose(first.qpos, second.qpos)
    assert np.allclose(second.qpos, robot.model.qpos0)


# ---------------------------------------------------------------------------
# ctrl 语义：统一成位置伺服，增益来自 RobotSpec
# ---------------------------------------------------------------------------


@needs_models
@pytest.mark.slow
@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_controlled_actuators_are_position_servos_with_spec_gains(name):
    """spec 是增益的唯一出处：模型里的 kp/kd 必须等于 spec 里的值。

    上游 G1 的 position 类自带 kp=500。如果不覆盖，RobotSpec.pd_kp 就是没人读
    的死数据，读代码的人会以为策略跑的是 100 的刚度。
    """
    spec = get_spec(name)
    robot = load_robot(spec)
    model = robot.model
    ids = robot.actuator_ids

    assert np.all(model.actuator_gaintype[ids] == mujoco.mjtGain.mjGAIN_FIXED)
    assert np.all(model.actuator_biastype[ids] == mujoco.mjtBias.mjBIAS_AFFINE)
    assert np.allclose(model.actuator_gainprm[ids, 0], spec.pd_kp_array)
    assert np.allclose(model.actuator_biasprm[ids, 1], -spec.pd_kp_array)
    assert np.allclose(model.actuator_biasprm[ids, 2], -spec.pd_kd_array)


@needs_models
@pytest.mark.slow
@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_torque_limits_become_force_saturation(name):
    """torque_limits 写进 forcerange：伺服在额定力矩处饱和，而不是无限出力。"""
    spec = get_spec(name)
    robot = load_robot(spec)
    model = robot.model
    ids = robot.actuator_ids

    assert np.all(model.actuator_forcelimited[ids])
    assert np.allclose(model.actuator_forcerange[ids, 1], spec.torque_limits_array)


@needs_models
@pytest.mark.slow
@pytest.mark.parametrize("name", MORPHOLOGIES)
def test_servo_targets_are_clamped_to_joint_range(name):
    """目标角不能追到关节行程之外，否则策略一激进就顶死限位。"""
    spec = get_spec(name)
    robot = load_robot(spec)
    model = robot.model

    for actuator_id in robot.actuator_ids:
        joint_id = int(model.actuator_trnid[actuator_id][0])
        assert model.actuator_ctrllimited[actuator_id]
        assert np.allclose(
            model.actuator_ctrlrange[actuator_id], model.jnt_range[joint_id]
        )


@needs_models
@pytest.mark.slow
def test_position_control_actually_moves_the_joint():
    """写进 ctrl 的是目标角：给定目标后关节应该朝它走，而不是朝别处。

    这是整条链路的冒烟测试。前面几条都在查数据对不对，这一条查行为对不对——
    如果 ctrl 语义其实还是力矩，这里会立刻看出来。
    """
    spec = get_spec("go2")
    robot = load_robot(spec)
    model, data = robot.model, robot.new_data()

    mujoco.mj_resetDataKeyframe(model, data, 0)
    target = spec.default_angles_array.copy()
    target[0] += 0.2  # 只推一个关节，看它有没有跟着动
    data.ctrl[robot.actuator_ids] = target
    data.ctrl[robot.held_actuator_ids] = robot.held_targets

    before = data.qpos[robot.qpos_ids[0]]
    for _ in range(100):
        mujoco.mj_step(model, data)
    after = data.qpos[robot.qpos_ids[0]]

    assert after > before
    assert after == pytest.approx(target[0], abs=0.2)


# ---------------------------------------------------------------------------
# 被保持的关节：索引与目标角
# ---------------------------------------------------------------------------


@needs_models
@pytest.mark.slow
def test_g1_holds_exactly_the_arms():
    """G1 冻结的是 14 个手臂关节，腿部与腰部一个都不能落到保持列表里。"""
    spec = get_spec("g1")
    robot = load_robot(spec)
    model = robot.model

    held_names = {
        _joint_name_of(model, int(a))
        for a in robot.held_actuator_ids
    }
    arm_names = {
        n for n in held_names if "shoulder" in n or "elbow" in n or "wrist" in n
    }
    assert held_names == arm_names
    assert len(held_names) == 14


@needs_models
@pytest.mark.slow
def test_go2_holds_nothing():
    """四足 12 个关节全受控，没有需要冻结的自由度。"""
    robot = load_robot(get_spec("go2"))
    assert robot.held_actuator_ids.size == 0
    assert robot.held_targets.size == 0


@needs_models
@pytest.mark.slow
def test_held_targets_come_from_keyframe():
    """保持角度取自 keyframe。拿 0 当目标会把手臂慢慢掰到不同姿态。"""
    spec = get_spec("g1")
    robot = load_robot(spec)
    model = robot.model

    key_id = next(
        k
        for k in range(model.nkey)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) == "stand"
    )
    for a, target in zip(robot.held_actuator_ids, robot.held_targets, strict=True):
        joint_id = int(model.actuator_trnid[int(a)][0])
        assert target == pytest.approx(model.key_qpos[key_id][model.jnt_qposadr[joint_id]])

    # 抽一个具体的数：keyframe 里肘部弯 1.28 rad，如果保持目标全是 0 就会露馅
    assert np.max(np.abs(robot.held_targets)) == pytest.approx(1.28)


# ---------------------------------------------------------------------------
# mutate 钩子与报错
# ---------------------------------------------------------------------------


@needs_models
@pytest.mark.slow
def test_mutate_hook_runs_before_compile():
    """环境层靠这个钩子插地形，所以钩子必须在编译前生效。"""
    def insert_terrain(mj_spec):
        body = mj_spec.worldbody.add_body(name="terrain_body", pos=[0, 0, 0])
        body.add_geom(
            name="terrain_box",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[2.0, 2.0, 0.02],
            pos=[0, 0, -0.02],
        )

    robot = load_robot(get_spec("go2"), mutate=insert_terrain)
    geom_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_box")
    assert geom_id >= 0
    assert robot.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX


@needs_models
@pytest.mark.slow
def test_unknown_joint_name_lists_actual_names():
    """名字写错时要把模型里真实存在的关节名列出来，而不是只说不认识。

    三种形态的命名习惯不一样（H1 没有 `_joint` 后缀），没有这份清单就得去翻
    几百行 XML。
    """
    bad_spec = RobotSpec(
        name="bad",
        morphology=get_spec("go2").morphology,
        menagerie_dir="unitree_go2",
        controlled_joints=("FL_hip",),  # 实际叫 FL_hip_joint
        default_angles=(0.0,),
        pd_kp=(20.0,),
        pd_kd=(0.5,),
        torque_limits=(23.7,),
    )

    with pytest.raises(ValueError) as excinfo:
        load_robot(bad_spec)

    message = str(excinfo.value)
    assert "FL_hip" in message
    assert "FL_hip_joint" in message  # 实际存在的关节名


@needs_models
@pytest.mark.slow
def test_missing_robot_dir_mentions_the_robot():
    """形态指向的目录不存在时，报错要带上目录名，便于确认是没拉模型还是拼错了。"""
    bad_spec = RobotSpec(
        name="ghost",
        morphology=get_spec("go2").morphology,
        menagerie_dir="unitree_nonexistent",
        controlled_joints=("FL_hip_joint",),
        default_angles=(0.0,),
        pd_kp=(20.0,),
        pd_kd=(0.5,),
        torque_limits=(23.7,),
    )

    with pytest.raises(FileNotFoundError, match="unitree_nonexistent"):
        load_robot(bad_spec)
