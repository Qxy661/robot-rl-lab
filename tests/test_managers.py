"""奖励与终止条件的注册表。

这里只测管理器的机制：注册、查找、启停、累加顺序。具体每一项奖励算得对不对、
终止判据在管线的哪一段被调用，分别在 test_rewards.py 式的数值用例与
test_mujoco_env.py 的管线用例里，不在这里重复。

注册表是进程级全局状态，所以凡是要塞东西进去的用例都用 monkeypatch，退出时
自动还原——否则一条用例注册的名字会漏给后面所有用例，失败位置还会飘。
"""

from __future__ import annotations

import pytest

from robotrl.configs.schema import RewardConfig
from robotrl.envs.managers import reward_manager as reward_manager_module
from robotrl.envs.managers import termination_manager as termination_manager_module
from robotrl.envs.managers.reward_manager import (
    RewardManager,
    get_reward,
    list_rewards,
    register_reward,
)
from robotrl.envs.managers.termination_manager import (
    TerminationManager,
    get_termination,
    list_terminations,
    register_termination,
)

# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


def _zero(env) -> float:
    return 0.0


def test_builtin_reward_terms_are_registered():
    """奖励模块 import 进来就该完成注册，不需要谁显式调用一遍注册函数。"""
    names = list_rewards()

    assert names == sorted(names)  # 列表用于报错信息，稳定顺序更好读
    for expected in (
        "action_rate",
        "base_height",
        "feet_slip",
        "joint_vel",
        "orientation",
        "track_lin_vel_xy",
    ):
        assert expected in names
        assert callable(get_reward(expected))


def test_builtin_termination_terms_are_registered():
    names = list_terminations()

    assert "nan_state" in names
    assert "tilt_too_large" in names
    assert callable(get_termination("tilt_too_large"))


def test_unknown_reward_name_lists_the_options():
    with pytest.raises(KeyError, match="未注册的奖励项"):
        get_reward("no_such_reward")


def test_unknown_termination_name_lists_the_options():
    with pytest.raises(KeyError, match="未注册的终止条件"):
        get_termination("no_such_termination")


def test_register_reward_refuses_to_overwrite(monkeypatch):
    """重名静默覆盖的后果是"权重调了但没反应"，查起来要翻整个注册流程。"""
    monkeypatch.setitem(reward_manager_module._REWARD_REGISTRY, "dup_reward", lambda env: 0.0)

    with pytest.raises(ValueError, match="已被注册"):
        register_reward("dup_reward")(lambda env: 1.0)


def test_register_termination_refuses_to_overwrite(monkeypatch):
    monkeypatch.setitem(
        termination_manager_module._TERMINATION_REGISTRY, "dup_term", lambda env: False
    )

    with pytest.raises(ValueError, match="已被注册"):
        register_termination("dup_term")(lambda env: True)


def test_register_reward_accepts_a_new_name(monkeypatch):
    """注册的新名字必须立刻能被查到——装饰器写完却忘了写进表里，是很容易犯的错。"""
    monkeypatch.setitem(reward_manager_module._REWARD_REGISTRY, "fresh", _zero)

    assert register_reward("fresh")(_zero) is _zero
    assert get_reward("fresh") is _zero


# ---------------------------------------------------------------------------
# 奖励组合
# ---------------------------------------------------------------------------


def probe_registry(order: list[str]) -> dict[str, object]:
    """三个会记录调用顺序的探针。名字留了空档，能看出排序是按名字而不是按插入。"""

    def make(name: str, value: float):
        def fn(env) -> float:
            order.append(name)
            return value

        return fn

    return {
        "a_first": make("a_first", 1.0),
        "b_second": make("b_second", 2.0),
        "c_third": make("c_third", 3.0),
    }


def test_terms_are_summed_in_a_stable_order():
    """累加顺序按名字排序，不随配置里的书写顺序变化。

    浮点加法不满足结合律，顺序变了末位就有差异。同一份配置跑两次得到不同的
    总奖励，实验对比会凭空多出噪声，而且只在低位，最难查。
    """
    order: list[str] = []
    registry = probe_registry(order)
    manager = RewardManager(
        RewardConfig(scales={"c_third": 1.0, "a_first": 1.0, "b_second": 1.0}),
        registry=registry,
    )

    total, breakdown = manager.compute(env=None)

    assert order == ["a_first", "b_second", "c_third"]
    assert manager.active_names == ("a_first", "b_second", "c_third")
    assert total == pytest.approx(6.0)
    assert breakdown == {"a_first": 1.0, "b_second": 2.0, "c_third": 3.0}


def test_scale_multiplies_only_the_total():
    """分项值回传的是原始量，权重只作用在总和上。

    分项值要跟日志里的物理量对齐（比如足端滑移的速度平方），乘上权重之后
    量级就没法横向比较了，也就失去了"这项是不是在偷偷补偿"的诊断价值。
    """
    manager = RewardManager(RewardConfig(scales={"a_first": -2.5}), registry=probe_registry([]))

    total, breakdown = manager.compute(env=None)

    assert total == pytest.approx(-2.5)
    assert breakdown["a_first"] == pytest.approx(1.0)


def test_scale_of_disabled_term_is_zero():
    order: list[str] = []
    manager = RewardManager(RewardConfig(scales={"a_first": 0.0}), registry=probe_registry(order))

    assert manager.scale_of("a_first") == 0.0
    assert manager.scale_of("never_mentioned") == 0.0
    assert len(manager) == 0


def test_unknown_term_from_config_is_rejected():
    """配置里写错名字必须当场报错。

    静默忽略的话，"我明明加权了"要等到看收敛曲线才怀疑，那时已经烧掉一轮训练。
    """
    with pytest.raises(KeyError, match="typo_term"):
        RewardManager(RewardConfig(scales={"typo_term": 1.0}), registry=probe_registry([]))


def test_repr_shows_every_active_term():
    manager = RewardManager(RewardConfig(scales={"a_first": 0.5}), registry=probe_registry([]))

    assert "a_first" in repr(manager)
    assert "0.5" in repr(manager)


# ---------------------------------------------------------------------------
# 终止组合
# ---------------------------------------------------------------------------


def termination_registry(seen: list[str]) -> dict[str, object]:
    def make(name: str, result: bool):
        def fn(env) -> bool:
            seen.append(name)
            return result

        return fn

    return {
        "a_fires": make("a_fires", True),
        "b_never": make("b_never", False),
    }


def test_termination_check_short_circuits():
    """任一条件为真就停：后面的条件不再求值。

    条件之间是布尔或，短路不丢信息；而某些判据（接触扫描、射线）不便宜，
    每一步都全算一遍纯属浪费。
    """
    seen: list[str] = []
    manager = TerminationManager(["a_fires", "b_never"], registry=termination_registry(seen))

    assert manager.check(env=None)
    assert seen == ["a_fires"]


def test_termination_check_runs_every_term_when_none_fires():
    seen: list[str] = []
    manager = TerminationManager(["b_never"], registry=termination_registry(seen))

    assert not manager.check(env=None)
    assert seen == ["b_never"]


def test_termination_mapping_uses_the_boolean_value():
    """映射形式用值当开关，写成 False 的名字等同于没写。

    这样配置里能保留全部判据的名字并逐个开关，不用靠注释来"关掉"某一行。
    """
    seen: list[str] = []
    registry = termination_registry(seen)
    manager = TerminationManager({"a_fires": False, "b_never": True}, registry=registry)

    assert manager.active_names == ("b_never",)
    assert not manager.check(env=None)


def test_no_termination_conditions_means_never_terminate():
    """默认全关。调试时想看策略能撑多久，就得让"摔倒"不结束回合。"""
    manager = TerminationManager()

    assert len(manager) == 0
    assert not manager.check(env=None)
    assert manager.active_names == ()


def test_duplicate_termination_names_are_deduplicated():
    manager = TerminationManager(["a_fires", "a_fires"], registry=termination_registry([]))

    assert manager.active_names == ("a_fires",)
    assert len(manager) == 1


def test_unknown_termination_is_rejected():
    with pytest.raises(KeyError, match="typo_termination"):
        TerminationManager(["typo_termination"], registry=termination_registry([]))


def test_termination_repr_lists_active_names():
    manager = TerminationManager(["a_fires"], registry=termination_registry([]))

    assert "a_fires" in repr(manager)
