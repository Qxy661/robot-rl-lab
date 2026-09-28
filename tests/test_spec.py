"""RobotSpec 契约测试。

重点不是"字段能不能存进去"，而是**非法数据能不能被挡住**。形态定义是手写的
数据，一个长度不对的列表、一个写错的增益，如果放到训练中途才炸，排查成本
会高得多。这些校验必须在构造那一刻就拦住。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotrl.assets import Morphology, RobotSpec, get_spec, list_specs, register
from robotrl.assets import spec as spec_module


@pytest.fixture
def isolated_registry():
    """注册表是全局单例，测试往里写东西必须还原，否则会污染后面的用例。

    没有这层隔离，一个用例注册的 "dup" 会出现在另一个用例的可用形态列表里，
    测试之间就产生了顺序依赖——这类问题排查起来非常费劲。
    """
    saved = dict(spec_module._REGISTRY)
    yield
    spec_module._REGISTRY.clear()
    spec_module._REGISTRY.update(saved)


def _valid_kwargs(**overrides):
    """一份合法的最小 spec 参数，测试里按需覆写单项。"""
    base = dict(
        name="dummy",
        morphology=Morphology.BIPED,
        menagerie_dir="dummy_dir",
        controlled_joints=("a", "b", "c"),
        default_angles=(0.0, 0.0, 0.0),
        pd_kp=(1.0, 1.0, 1.0),
        pd_kd=(0.1, 0.1, 0.1),
        torque_limits=(10.0, 10.0, 10.0),
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


def test_builtin_morphologies_registered():
    """资产层内置三种真实机器人形态。

    toy 是环境层为冒烟测试造的量点替身，注册发生在 envs 导入时，
    不属于资产层的内置形态，因此不在这里断言。
    """
    assert set(list_specs()) >= {"g1", "h1", "go2"}


@pytest.mark.parametrize("name", ["g1", "h1", "go2"])
def test_specs_are_internally_consistent(name):
    """三种真实形态的字段长度必须自洽，且数值在合理范围。"""
    spec = get_spec(name)

    n = spec.n_dof
    assert n > 0
    assert len(spec.default_angles) == n
    assert len(spec.pd_kp) == n
    assert len(spec.pd_kd) == n
    assert len(spec.torque_limits) == n
    assert len(spec.action_scale_array) == n

    assert np.all(spec.pd_kp_array >= 0)
    assert np.all(spec.pd_kd_array >= 0)
    assert np.all(spec.torque_limits_array > 0)
    assert np.all(spec.action_scale_array > 0)


def test_morphology_matches_expected_joint_count():
    """关节数随形态变化，这正是跨形态要处理的东西。"""
    assert get_spec("g1").morphology is Morphology.BIPED
    assert get_spec("go2").morphology is Morphology.QUADRUPED
    assert get_spec("go2").n_dof == 12
    # 人形自由度明显多于四足，且两者不相等——换形态时 obs/action 维度确实会变
    assert get_spec("g1").n_dof != get_spec("go2").n_dof


def test_get_spec_unknown_name_lists_alternatives():
    with pytest.raises(KeyError, match="g1"):
        get_spec("nonexistent")


def test_register_rejects_duplicate_name(isolated_registry):
    register(RobotSpec(**_valid_kwargs(name="dup")))
    with pytest.raises(ValueError, match="已被注册"):
        register(RobotSpec(**_valid_kwargs(name="dup")))


def test_isolation_fixture_restores_registry(isolated_registry):
    """确认隔离夹具确实还原了注册表，否则上面那个用例的断言是假通过。"""
    register(RobotSpec(**_valid_kwargs(name="tmp_morph")))
    assert "tmp_morph" in list_specs()


# ---------------------------------------------------------------------------
# 非法数据必须被挡住
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["default_angles", "pd_kp", "pd_kd", "torque_limits"],
)
def test_rejects_length_mismatch(field):
    kwargs = _valid_kwargs(**{field: (0.0, 0.0)})  # 三个关节却只给两个值
    with pytest.raises(ValueError, match=field):
        RobotSpec(**kwargs)


def test_rejects_empty_joint_list():
    with pytest.raises(ValueError, match="不能为空"):
        RobotSpec(**_valid_kwargs(controlled_joints=(), default_angles=(), pd_kp=(),
                                  pd_kd=(), torque_limits=()))


def test_rejects_duplicate_joint_names():
    with pytest.raises(ValueError, match="重复"):
        RobotSpec(**_valid_kwargs(controlled_joints=("a", "a", "b")))


def test_rejects_negative_gain():
    with pytest.raises(ValueError, match="不能为负"):
        RobotSpec(**_valid_kwargs(pd_kp=(-1.0, 1.0, 1.0)))


def test_rejects_nonpositive_torque_limit():
    with pytest.raises(ValueError, match="必须为正"):
        RobotSpec(**_valid_kwargs(torque_limits=(10.0, 0.0, 10.0)))


def test_rejects_action_scale_length_mismatch():
    with pytest.raises(ValueError, match="action_scale"):
        RobotSpec(**_valid_kwargs(action_scale=(0.1, 0.2)))


# ---------------------------------------------------------------------------
# 数组视图
# ---------------------------------------------------------------------------


def test_scalar_action_scale_broadcasts_to_all_joints():
    spec = RobotSpec(**_valid_kwargs(action_scale=0.5))
    assert spec.action_scale_array.shape == (spec.n_dof,)
    assert np.allclose(spec.action_scale_array, 0.5)


def test_per_joint_action_scale_is_preserved():
    spec = RobotSpec(**_valid_kwargs(action_scale=(0.1, 0.2, 0.3)))
    assert np.allclose(spec.action_scale_array, [0.1, 0.2, 0.3])


def test_spec_is_hashable_and_frozen():
    """frozen 保证形态定义不会被训练代码顺手改掉，这是并行开发的安全前提。"""
    import dataclasses

    spec = get_spec("go2")
    assert hash(spec) == hash(get_spec("go2"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "changed"  # type: ignore[misc]
