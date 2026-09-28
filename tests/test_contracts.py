"""观测契约测试。

观测布局是三层共用的接口，一旦拼错位置，错误会以"训练不收敛"这种最难查的
形式表现出来，而不是抛异常。所以拆装的一致性必须由测试保证，尤其要覆盖
"拆了再拼回来是否逐位相等"。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotrl.assets import get_spec
from robotrl.contracts import (
    NUM_FIXED_SEGMENTS,
    ObsContract,
    ObsSegment,
    make_obs_contract,
    make_privileged_obs_contract,
)

# ---------------------------------------------------------------------------
# 布局与索引
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("robot", ["g1", "h1", "go2"])
def test_total_dim_matches_formula(robot):
    """总维度 = 12 维本体感觉 + 3 段逐关节量。"""
    n = get_spec(robot).n_dof
    contract = make_obs_contract(n)
    assert contract.total_dim == 12 + 3 * n


def test_fixed_segments_keep_stable_offsets_across_morphologies():
    """固定段的切片位置不能随形态变化。

    这是把逐关节段排在末尾的原因：让与形态无关的量在任意机器人上占同样的
    下标，部署层做误差归因时就不用为每种形态单独写一套索引。
    """
    small = make_obs_contract(12)
    large = make_obs_contract(29)

    for name in ("lin_vel", "ang_vel", "proj_gravity", "cmd"):
        assert small.index(name) == large.index(name)

    # 逐关节段的起点随形态变化，这是预期的
    assert small.index("dof_pos") != large.index("dof_pos")


def test_segment_order_is_the_documented_one():
    contract = make_obs_contract(12)
    assert contract.names[:NUM_FIXED_SEGMENTS] == ("lin_vel", "ang_vel", "proj_gravity", "cmd")
    assert contract.names[NUM_FIXED_SEGMENTS:] == ("dof_pos", "dof_vel", "last_action")


def test_index_unknown_segment_lists_alternatives():
    with pytest.raises(KeyError, match="cmd"):
        make_obs_contract(12).index("no_such_segment")


# ---------------------------------------------------------------------------
# 拆装往返
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_dof", [2, 12, 15, 19, 29])
def test_split_concat_roundtrip_is_exact(n_dof):
    """拆开再拼回来必须逐位相等。这是观测管线正确性的底线。"""
    contract = make_obs_contract(n_dof)
    original = np.arange(contract.total_dim, dtype=np.float32)

    parts = contract.split(original)
    rebuilt = contract.concat(parts)

    assert rebuilt.dtype == np.float32
    assert np.array_equal(rebuilt, original)


def test_split_supports_batched_observations():
    """向量化环境会传 (num_envs, obs_dim)，契约要能直接吃下去。"""
    contract = make_obs_contract(12)
    batch = np.arange(8 * contract.total_dim, dtype=np.float32).reshape(8, -1)

    parts = contract.split(batch)
    assert all(v.shape == (8, contract.dim_of(k)) for k, v in parts.items())
    assert np.array_equal(contract.concat(parts), batch)


def test_split_rejects_wrong_width():
    with pytest.raises(ValueError, match="末维"):
        make_obs_contract(12).split(np.zeros(10, dtype=np.float32))


def test_concat_rejects_missing_and_extra_segments():
    contract = make_obs_contract(2)
    parts = {name: np.zeros(contract.dim_of(name), dtype=np.float32)
             for name in contract.names}

    with pytest.raises(KeyError, match="缺少"):
        contract.concat({k: v for k, v in parts.items() if k != "cmd"})

    extra = dict(parts)
    extra["bogus"] = np.zeros(1, dtype=np.float32)
    with pytest.raises(KeyError, match="契约外"):
        contract.concat(extra)


def test_concat_rejects_wrong_segment_width():
    contract = make_obs_contract(2)
    parts = {name: np.zeros(contract.dim_of(name), dtype=np.float32)
             for name in contract.names}
    parts["cmd"] = np.zeros(5, dtype=np.float32)  # 应是 3

    with pytest.raises(ValueError, match="cmd"):
        contract.concat(parts)


# ---------------------------------------------------------------------------
# 契约自身的校验
# ---------------------------------------------------------------------------


def test_rejects_duplicate_segment_names():
    with pytest.raises(ValueError, match="重复"):
        ObsContract(segments=(ObsSegment("a", 1), ObsSegment("a", 2)))


@pytest.mark.parametrize("bad_dim", [0, -1])
def test_rejects_nonpositive_segment_dim(bad_dim):
    with pytest.raises(ValueError, match="维度必须为正"):
        ObsContract(segments=(ObsSegment("a", bad_dim),))


def test_rejects_empty_contract():
    with pytest.raises(ValueError, match="至少要有一段"):
        ObsContract(segments=())


def test_rejects_nonpositive_n_dof():
    with pytest.raises(ValueError, match="n_dof"):
        make_obs_contract(0)


# ---------------------------------------------------------------------------
# 特权观测：critic 用的扩展契约
# ---------------------------------------------------------------------------


def test_privileged_contract_extends_without_perturbing_prefix():
    """critic 观测必须与 policy 观测共享完全一致的前缀。

    否则 actor 和 critic 对同一个状态的解读会在前若干维上错位，
    这类 bug 不报错、只是学得慢，最难发现。
    """
    n = get_spec("g1").n_dof
    policy = make_obs_contract(n)
    privileged = make_privileged_obs_contract(n, num_feet=2)

    assert privileged.total_dim > policy.total_dim
    for name in policy.names:
        assert privileged.index(name) == policy.index(name)
    assert privileged.names[: len(policy.names)] == policy.names


def test_privileged_contract_feet_dim_follows_morphology():
    """双足与四足的接触标志维度不同，这正是需要参数化的地方。"""
    biped = make_privileged_obs_contract(15, num_feet=2)
    quadruped = make_privileged_obs_contract(12, num_feet=4)

    assert biped.dim_of("feet_contact") == 2
    assert quadruped.dim_of("feet_contact") == 4


def test_with_extra_returns_new_object():
    """with_extra 不能就地修改——契约是共享的，改一处会污染所有使用者。"""
    base = make_obs_contract(12)
    extended = base.with_extra(ObsSegment("extra", 1))

    assert base.total_dim != extended.total_dim
    assert "extra" not in base.names
