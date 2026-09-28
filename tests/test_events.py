"""域随机化。

直接对着 DomainRandomizer 测，不经过环境的 8 段管线：这样"哪个组件随机了哪些
数组"与"随机化在管线的哪一段被调用"是两个独立的问题，分开测，出问题时不用
在两者之间猜。
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from mini_robot import MINI_HELD_TARGET, MINI_SPEC, build_robot_model

from robotrl.configs.schema import Config, EventsConfig, ObsConfig
from robotrl.envs.events import DomainRandomizer
from robotrl.envs.mujoco_env import MujocoEnv

#: 内嵌模型里的标称摩擦系数（MJCF 的 default 段给了 1.0）。
NOMINAL_FRICTION = 1.0


def make_randomizer(
    events: EventsConfig | None = None,
    obs: ObsConfig | None = None,
    *,
    robot_model=None,
    base_dof_adr: int | None = None,
) -> tuple[DomainRandomizer, mujoco.MjModel, mujoco.MjData]:
    """建一组独立的随机化器与仿真状态。

    模型不经过环境，这样"改模型数组"这件事与环境的其它职责完全隔离开。
    """
    robot_model = robot_model or build_robot_model()
    model = robot_model.model
    data = mujoco.MjData(model)
    base_body, _, dof_adr = MujocoEnv._find_base(model)
    randomizer = DomainRandomizer(
        events or EventsConfig(),
        obs or ObsConfig(),
        model=model,
        spec=MINI_SPEC,
        base_body_id=base_body,
        base_dof_adr=dof_adr if base_dof_adr is None else base_dof_adr,
    )
    return randomizer, model, data


def apply_reset(randomizer: DomainRandomizer, data: mujoco.MjData, seed: int = 0) -> None:
    randomizer.apply_model(data, np.random.default_rng(seed))


# ---------------------------------------------------------------------------
# 模型常数
# ---------------------------------------------------------------------------


def test_friction_randomization_covers_every_geom():
    randomizer, model, data = make_randomizer()
    apply_reset(randomizer, data, seed=1)

    low, high = EventsConfig().friction_range
    assert np.all(model.geom_friction[:, 0] >= low * NOMINAL_FRICTION)
    assert np.all(model.geom_friction[:, 0] <= high * NOMINAL_FRICTION)
    # 扭转与滚动摩擦不动：随机化它们的效果在仿真里几乎不可见，却会让调参多两个维度。
    assert np.allclose(model.geom_friction[:, 1:], [0.005, 0.0001])


def test_randomization_is_applied_on_top_of_nominal_not_current_value():
    """基准是标称值，不是"当前值 × 因子"。

    后者会漂移：来回随机几十次之后，摩擦系数爬到配置范围之外，而且同一份配置
    不再可复现。逐次检查范围上界就能直接暴露漂移。
    """
    randomizer, model, data = make_randomizer()
    low, high = EventsConfig().friction_range
    nominal_mass = build_robot_model().model.body_mass[1]

    for seed in range(30):
        apply_reset(randomizer, data, seed=seed)
        friction = float(model.geom_friction[1, 0])
        assert low * NOMINAL_FRICTION <= friction <= high * NOMINAL_FRICTION
        delta = float(model.body_mass[1]) - nominal_mass
        assert EventsConfig().base_mass_delta[0] <= delta <= EventsConfig().base_mass_delta[1]


def test_base_mass_randomization_never_goes_non_positive():
    """质量被随机到 0 或负数会让惯性矩阵失去正定性，求解器直接发散。"""
    config = EventsConfig(base_mass_delta=(-1e6, -4.0))
    randomizer, model, data = make_randomizer(config)
    apply_reset(randomizer, data)

    assert model.body_mass[1] > 0.0


def test_motor_strength_scales_every_actuator():
    randomizer, model, data = make_randomizer()
    apply_reset(randomizer, data, seed=5)

    low, high = EventsConfig().motor_strength_range
    gear = model.actuator_gear[:, 0]
    assert np.all(gear >= low - 1e-9)
    assert np.all(gear <= high + 1e-9)


def test_pd_gain_randomization_keeps_position_actuator_consistent():
    """kp 同时出现在 gainprm[0] 和 biasprm[1]，必须同步改。

    位置执行器在 MuJoCo 里是 general 执行器：出力 = kp*ctrl - kp*q - kd*qd。
    只改 gainprm 会得到刚度变了、稳态误差却按旧刚度算的执行器——物理上不存在，
    而且它不会报错，只会让 sim2real 的差距莫名其妙地大。
    """
    randomizer, model, data = make_randomizer()
    apply_reset(randomizer, data, seed=3)

    kp = model.actuator_gainprm[:, 0]
    assert np.allclose(kp, -model.actuator_biasprm[:, 1])
    assert np.all(-model.actuator_biasprm[:, 2] >= 0)  # kd 仍是阻尼，不能变负

    low, high = EventsConfig().pd_gain_range
    nominal = build_robot_model().model.actuator_gainprm[:, 0]
    assert np.all(kp >= low * nominal * 0.999)
    assert np.all(kp <= high * nominal * 1.001)


def test_each_episode_draws_fresh_model_constants():
    randomizer, model, data = make_randomizer()
    frictions = set()
    for seed in range(12):
        apply_reset(randomizer, data, seed=seed)
        frictions.add(round(float(model.geom_friction[1, 0]), 6))

    assert len(frictions) > 1


# ---------------------------------------------------------------------------
# 瞬时扰动
# ---------------------------------------------------------------------------


def test_push_is_applied_only_when_its_time_comes():
    """推力按随机间隔触发，不是每步都推。

    每步都推相当于给策略加了一个恒定的速度偏置，它能学会抵消掉；间隔随机才
    逼它学会"随时可能被打断，被打断后要能恢复"。
    """
    events = EventsConfig(push_interval_steps=(5, 5), push_velocity_xy=(0.5, 0.5))
    randomizer, model, data = make_randomizer(events)
    randomizer.reschedule_push(np.random.default_rng(0))
    data.qvel[:] = 0.0

    for _ in range(4):
        randomizer.apply_push(data, np.random.default_rng(0))
        assert data.qvel[0] == pytest.approx(0.0)

    randomizer.apply_push(data, np.random.default_rng(0))
    assert data.qvel[0] > 0.0


def test_push_moves_only_the_base_and_only_in_the_plane():
    """冲击是水平的速度突变，不该给机身加竖直速度或角速度。"""
    low, high = 0.3, 0.4
    events = EventsConfig(push_interval_steps=(1, 1), push_velocity_xy=(low, high))
    randomizer, model, data = make_randomizer(events)
    data.qvel[:] = 0.0
    randomizer.reschedule_push(np.random.default_rng(0))

    randomizer.apply_push(data, np.random.default_rng(0))

    assert low <= data.qvel[0] <= high
    assert low <= data.qvel[1] <= high
    assert data.qvel[2] == pytest.approx(0.0)  # 竖直分量不动
    assert np.all(data.qvel[3:6] == 0.0)  # 角速度不动


def test_push_can_be_disabled():
    events = EventsConfig(push_robot=False, push_interval_steps=(1, 1))
    randomizer, model, data = make_randomizer(events)
    data.qvel[:] = 0.0
    randomizer.reschedule_push(np.random.default_rng(0))

    for _ in range(5):
        randomizer.apply_push(data, np.random.default_rng(0))

    assert np.all(data.qvel == 0.0)


def test_push_does_nothing_without_a_floating_base():
    """固定基座的模型没有浮动自由度可推，应当安静地跳过而不是越界写。

    base_dof_adr = -1 正是 _find_base 在找不到自由关节时给出的值，这里直接把它
    喂进来，比为了造一个固定基座场景再写一段 MJCF 更贴近要验的那个分支。
    """
    events = EventsConfig(push_interval_steps=(1, 1), push_velocity_xy=(0.5, 0.5))
    randomizer, model, data = make_randomizer(events, base_dof_adr=-1)
    data.qvel[:] = 0.0
    randomizer.reschedule_push(np.random.default_rng(0))

    for _ in range(3):
        randomizer.apply_push(data, np.random.default_rng(0))

    assert np.all(data.qvel == 0.0)


# ---------------------------------------------------------------------------
# 观测噪声
# ---------------------------------------------------------------------------


def test_observation_noise_scales_come_from_obs_config():
    """噪声尺度按段名从 ObsConfig.noise_scale 读，没列出的段不加噪声。

    cmd 与 last_action 没有噪声是刻意的：它们不是测量值，是策略自己产生或
    环境直接给定的，给它们加噪声等于告诉策略"你的指令不可信"。
    """
    randomizer, _, _ = make_randomizer()
    assert randomizer.observation_noise("dof_pos") == pytest.approx(
        ObsConfig().noise_scale["dof_pos"]
    )
    assert randomizer.observation_noise("cmd") == 0.0
    assert randomizer.observation_noise("last_action") == 0.0


def test_observation_noise_can_be_turned_off():
    randomizer, _, _ = make_randomizer(obs=ObsConfig(use_obs_noise=False))
    assert randomizer.observation_noise("dof_pos") == 0.0


# ---------------------------------------------------------------------------
# 与环境层的衔接
# ---------------------------------------------------------------------------


def test_shared_model_is_copied_when_randomization_is_on():
    """随机化会就地改模型数组，所以环境必须持有自己的一份。

    RobotModel.model 允许在多个实例间共享；不复制的后果是环境之间互相覆写参数，
    而这种 bug 只在并行训练时出现，单环境调试永远撞不到。
    """
    shared = build_robot_model()
    nominal_friction = shared.model.geom_friction.copy()
    nominal_mass = shared.model.body_mass.copy()

    env = MujocoEnv(spec=MINI_SPEC, robot_model=shared, init_base_height=0.45, seed=0)
    env.reset(seed=1)

    assert env.model is not shared.model
    assert shared.model.geom_friction == pytest.approx(nominal_friction)
    assert shared.model.body_mass == pytest.approx(nominal_mass)


def test_model_is_not_copied_when_nothing_touches_it():
    """随机化全关时不必复制模型，省下每个环境一份模型的内存。"""
    config = Config()
    config.events.randomize_friction = False
    config.events.randomize_base_mass = False
    config.events.randomize_motor_strength = False
    config.events.randomize_pd_gain = False
    shared = build_robot_model()

    env = MujocoEnv(spec=MINI_SPEC, robot_model=shared, config=config, seed=0)

    assert not DomainRandomizer.affects_model(config.events)
    assert env.model is shared.model


def test_model_randomization_reapplies_after_an_in_episode_reset():
    """回合在 step 内部结束时也要重新随机化。

    基类只在外部 reset() 时会带着 reset=True 走到事件段，所以随机化不能挂在那里；
    向量化训练里环境通常只在开头 reset 一次，挂错地方等于整场训练用同一批参数，
    还必须靠逐回合比较才能发现。
    """
    config = Config()
    config.terrain.kind = "flat"
    env = MujocoEnv(spec=MINI_SPEC, robot_model=build_robot_model(), config=config, seed=0)
    env.max_episode_steps = 1  # 第一步就超时，触发 step 内部的复位
    env.reset(seed=0)
    before = float(env.model.geom_friction[1, 0])

    env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))

    after = float(env.model.geom_friction[1, 0])
    assert after != pytest.approx(before)


def test_held_targets_are_untouched_by_randomization():
    """保持目标来自 keyframe，是几何量，不属于随机化的范围。"""
    randomizer, model, data = make_randomizer()
    apply_reset(randomizer, data, seed=2)

    assert pytest.approx(0.2) == MINI_HELD_TARGET
    assert model.key_qpos[0][9] == pytest.approx(MINI_HELD_TARGET)
