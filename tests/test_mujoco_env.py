"""MuJoCo 环境的 8 段管线。

用 tests/mini_robot.py 里内嵌的最小 MJCF，不依赖 Menagerie 与网络，因此这些
用例在任何机器上都能秒级跑完。默认模板、真实形态的用例标了 slow，见本文件末尾。
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from mini_robot import MINI_BASE_HEIGHT, MINI_HELD_TARGET, MINI_SPEC, build_robot_model

from robotrl.configs.schema import Config, RewardConfig
from robotrl.envs.managers import RewardManager, TerminationManager
from robotrl.envs.mujoco_env import MujocoEnv


def make_env(**kwargs) -> MujocoEnv:
    """构造一个测试环境。

    默认关掉奖励与失败判据：这些用例验的是管线时序，而这个机器人站不稳，
    几分钟仿真后必然会倒——真让判据生效，测到的东西就变成了"它倒没倒"。
    需要判据的用例显式传进来。
    """
    params = dict(
        spec=MINI_SPEC,
        robot_model=build_robot_model(),
        init_base_height=MINI_BASE_HEIGHT,
        reward_manager=RewardManager(),
        termination_manager=TerminationManager(),
        seed=0,
    )
    params.update(kwargs)
    return MujocoEnv(**params)


@pytest.fixture
def env() -> MujocoEnv:
    return make_env()


def tilt_over(env: MujocoEnv) -> None:
    """把机身放倒（绕 x 轴转 90 度），用于制造"摔了"的状态。

    用放倒而不是压低机身：压低会把自己压进地面，接触求解器会在下一步把机器人
    弹出来，"摔了"这个状态就自己消失了。放倒则不会自愈，仿真跑十几步也还是躺着的。
    """
    half = np.pi / 4.0
    env.data.qpos[env._base_qpos_adr + 3 : env._base_qpos_adr + 7] = [
        np.cos(half),
        np.sin(half),
        0.0,
        0.0,
    ]
    mujoco.mj_forward(env.model, env.data)


# ---------------------------------------------------------------------------
# 观测契约
# ---------------------------------------------------------------------------


def test_reset_returns_contract_shaped_observation(env):
    obs, info = env.reset(seed=0)

    assert obs.policy.shape == (env.obs_dim,)
    assert obs.critic.shape == (env.critic_obs_dim,)
    assert obs.policy.dtype == np.float32
    assert np.all(np.isfinite(obs.policy))
    assert info["step"] == 0


def test_observation_matches_declared_contract(env):
    """观测必须能被契约按段拆开，且逐关节段的宽度跟着 spec 走。"""
    obs, _ = env.reset(seed=0)
    parts = env.obs_contract.split(obs.policy)

    assert parts["cmd"].shape == (3,)
    assert parts["proj_gravity"].shape == (3,)
    assert parts["dof_pos"].shape == (MINI_SPEC.n_dof,)
    assert parts["last_action"].shape == (MINI_SPEC.n_dof,)
    assert env.obs_contract.concat(parts).tolist() == pytest.approx(obs.policy.tolist())


def test_critic_observation_extends_policy_prefix(env):
    obs, _ = env.reset(seed=0)
    fixed = env.obs_contract.total_dim

    assert env.critic_obs_dim == fixed + 3 + env.num_feet + env.obs_config.terrain_samples
    assert np.allclose(obs.critic[:fixed], obs.policy)
    assert env.critic_obs_contract.index("cmd") == env.obs_contract.index("cmd")


def test_dof_observations_are_relative_to_default_pose(env):
    """逐关节观测是"与默认姿态之差"。

    用绝对角度的话，同一段观测在不同形态上的数值中心不一样，观测归一化等于
    每次换机器人都要重新学一遍。
    """
    env.reset(seed=0)
    env.data.qpos[env.qpos_ids] = MINI_SPEC.default_angles_array
    mujoco.mj_forward(env.model, env.data)

    parts = env.obs_contract.split(env._observe().policy)
    # 默认姿态下应当只有观测噪声（0.01 量级），不是默认角度本身（0.15 弧度）。
    assert np.allclose(parts["dof_pos"], 0.0, atol=0.05)


def test_last_action_segment_tracks_previous_action(env):
    """last_action 段是上一步的动作：策略需要它来感知自己的输出历史。"""
    env.reset(seed=0)
    action = np.array([0.4, -0.6], dtype=np.float32)
    env.step(action)

    parts = env.obs_contract.split(env._observe().policy)
    assert parts["last_action"].tolist() == pytest.approx(action.tolist())


# ---------------------------------------------------------------------------
# 动作与仿真
# ---------------------------------------------------------------------------


def test_action_is_clipped_then_scaled_into_joint_target(env):
    """动作是归一化增量：target = default + action_scale * action。

    超范围的动作先被裁到 ±1，所以 ±5 与 ±1 得到同一个目标——这条契约是部署
    环节的前置条件（导出的模型带输出范围约束），不能被绕开。
    """
    env.reset(seed=0)
    env.step(np.full(MINI_SPEC.n_dof, 5.0, dtype=np.float32))

    expected = MINI_SPEC.default_angles_array + MINI_SPEC.action_scale * 1.0
    assert env.data.ctrl[env.actuator_ids] == pytest.approx(expected)
    assert env.target_joint_positions == pytest.approx(expected)


def test_negative_action_moves_target_below_default(env):
    env.reset(seed=0)
    env.step(np.full(MINI_SPEC.n_dof, -0.5, dtype=np.float32))

    expected = MINI_SPEC.default_angles_array - 0.5 * MINI_SPEC.action_scale
    assert env.data.ctrl[env.actuator_ids] == pytest.approx(expected)


def test_held_joints_receive_their_hold_target_every_step(env):
    """未受控关节（这里是胳膊）每步都要写保持目标。

    只写一次的话，受控关节那一次写入之后它们就被晾着了——ctrl 数组是共享的，
    不写等于让它自由下垂。
    """
    env.reset(seed=0)
    assert env.held_targets == pytest.approx([MINI_HELD_TARGET])

    for _ in range(3):
        env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))
        assert env.data.ctrl[env.held_actuator_ids] == pytest.approx(env.held_targets)


def test_simulate_advances_decimation_physics_steps(env):
    env.reset(seed=0)
    start = env.data.time
    env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))

    assert env.decimation == 10  # 0.02 / 0.002
    assert env.data.time == pytest.approx(start + env.decimation * env.sim_dt)


def test_previous_action_lags_last_action_by_one_step(env):
    """动作平滑项要的是相邻两步之差，所以两个动作都要留着。"""
    env.reset(seed=0)
    env.step(np.array([0.2, 0.2], dtype=np.float32))
    env.step(np.array([0.5, -0.5], dtype=np.float32))

    assert env.last_action == pytest.approx([0.5, -0.5])
    assert env.previous_action == pytest.approx([0.2, 0.2])


# ---------------------------------------------------------------------------
# 终止与超时分流
# ---------------------------------------------------------------------------


def test_truncation_is_not_termination():
    """超时走 truncated，失败走 terminated，两者不能合并。

    价值估计上这是两件事：超时的那一步还要 bootstrap，失败的那一步价值是 0。
    """
    env = make_env(max_episode_steps=3)
    env.reset(seed=0)

    for _ in range(2):
        result = env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))
        assert not result.truncated
        assert not result.terminated

    result = env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))
    assert result.truncated
    assert not result.terminated
    assert result.info["is_terminal_step"]
    assert result.info["step"] == 0  # 复位后计数归零


def test_termination_resets_the_episode_inside_the_step():
    env = make_env(termination_manager=TerminationManager(["tilt_too_large"]))
    env.reset(seed=0)
    tilt_over(env)

    result = env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))

    assert result.terminated
    assert not result.truncated
    # 观测已经是复位后的新状态：机身重新立起来了
    assert env.projected_gravity()[2] == pytest.approx(-1.0, abs=0.1)


def test_reward_is_computed_before_reset():
    """终止那一步的奖励必须基于"失败时刻"的状态，而不是复位后的新状态。

    顺序反了的话，策略会学到"摔倒前那一步的奖励来自新回合的初始状态"，
    这个梯度是有害的，而且极难从曲线上看出来。
    """
    seen: dict[str, float] = {}

    def probe(env: MujocoEnv) -> float:
        seen["gravity_z"] = float(env.projected_gravity()[2])
        return 1.0

    manager = RewardManager(defaults={"probe": 1.0}, registry={"probe": probe})
    env = make_env(
        reward_manager=manager,
        termination_manager=TerminationManager(["tilt_too_large"]),
    )
    env.reset(seed=0)
    tilt_over(env)

    env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))

    assert seen["gravity_z"] > -0.5  # 奖励看到的是躺倒时的姿态
    assert env.projected_gravity()[2] < -0.9  # 观测看到的是复位后的直立姿态


def test_termination_terms_are_individually_switchable():
    """判据是具名开关：关掉倾角判据后，躺着也不结束。"""
    env_on = make_env(termination_manager=TerminationManager(["tilt_too_large"]))
    env_on.reset(seed=0)
    tilt_over(env_on)

    env_off = make_env(termination_manager=TerminationManager())
    env_off.reset(seed=0)
    tilt_over(env_off)

    assert env_on.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32)).terminated
    assert not env_off.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32)).terminated


def test_step_counter_and_return_reset_on_new_episode():
    env = make_env(max_episode_steps=2)
    env.reset(seed=0)
    env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))
    result = env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))

    assert result.truncated
    assert result.info["episode_return"] == pytest.approx(result.reward)


# ---------------------------------------------------------------------------
# 观测噪声
# ---------------------------------------------------------------------------


def test_observation_noise_is_resampled_every_observe(env):
    """同一状态下连读两次观测应当不同——噪声是在每次组装观测时重新采的。"""
    env.reset(seed=0)
    first = env._observe().policy
    second = env._observe().policy

    # 差异必须是噪声量级：超过几个标准差就说明状态本身变了，那不是这条用例想测的。
    noise_bound = 5.0 * max(env.obs_config.noise_scale.values())
    assert not np.array_equal(first, second)
    assert 0.0 < np.abs(first - second).max() < noise_bound


def test_observation_noise_can_be_disabled():
    config = Config()
    config.obs.use_obs_noise = False
    env = make_env(config=config)
    env.reset(seed=0)

    assert np.array_equal(env._observe().policy, env._observe().policy)


# ---------------------------------------------------------------------------
# 奖励管理器
# ---------------------------------------------------------------------------


def test_zero_weight_reward_terms_are_never_computed():
    """权重为 0 的项在构造时就被剔除：不进调用列表，也不在热路径上被调用。"""
    calls: list[int] = []

    def counter(env: MujocoEnv) -> float:
        calls.append(1)
        return 1.0

    registry = {"counter": counter}
    disabled = RewardManager(defaults={"counter": 0.0}, registry=registry)
    enabled = RewardManager(defaults={"counter": 1.0}, registry=registry)

    assert disabled.active_names == ()
    assert enabled.active_names == ("counter",)

    env = make_env(reward_manager=disabled)
    env.reset(seed=0)
    env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))
    assert calls == []

    env = make_env(reward_manager=enabled)
    env.reset(seed=0)
    env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))
    assert len(calls) == 1


def test_config_scale_overrides_task_default():
    registry = {"counter": lambda env: 1.0}
    kept = RewardManager(
        RewardConfig(scales={"counter": 2.5}), defaults={"counter": 5.0}, registry=registry
    )
    zeroed = RewardManager(
        RewardConfig(scales={"counter": 0.0}), defaults={"counter": 5.0}, registry=registry
    )

    assert kept.scale_of("counter") == pytest.approx(2.5)
    assert zeroed.active_names == ()


def test_reward_manager_rejects_unknown_name():
    with pytest.raises(KeyError, match="no_such_term"):
        RewardManager(RewardConfig(scales={"no_such_term": 1.0}))


def test_reward_breakdown_reports_raw_values(env):
    """逐项值回传的是未乘权重的原始量，日志里才能横向比较各项的量级。"""
    env.reward_manager = RewardManager(
        defaults={"counter": 3.0}, registry={"counter": lambda e: 2.0}
    )
    env.reset(seed=0)
    env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))

    assert env.reward_terms == {"counter": 2.0}
    assert env.reward_manager.scale_of("counter") == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 可复现性
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_rollouts():
    """同一份代码同一种子必须走出同一条轨迹，否则实验结论无法比较。

    回报里带上机身高度，因为它同时受初始状态与域随机化（质量、增益）影响，
    任何一种随机源没接上环境自己的生成器都会让这条用例失败。
    """
    probe = {"height": lambda env: env.base_height()}

    def rollout(seed: int) -> list[float]:
        env = make_env(
            seed=seed, reward_manager=RewardManager(defaults={"height": 1.0}, registry=probe)
        )
        env.reset(seed=seed)
        return [env.step(np.full(MINI_SPEC.n_dof, 0.3, dtype=np.float32)).reward for _ in range(6)]

    assert rollout(11) == rollout(11)
    assert rollout(11) != rollout(12)


def test_reset_seed_overrides_construction_seed():
    env = make_env(seed=999)
    obs_a, _ = env.reset(seed=4)
    obs_b, _ = env.reset(seed=4)

    assert np.array_equal(obs_a.policy, obs_b.policy)


# ---------------------------------------------------------------------------
# 工具方法
# ---------------------------------------------------------------------------


def test_base_state_accessors_are_consistent(env):
    """机身姿态、速度、重力投影三者的关系必须自洽，否则奖励项会集体算错。"""
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.zeros(MINI_SPEC.n_dof, dtype=np.float32))

    rotation = env.base_rotation()
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-9)
    # 机身大致竖直：重力投影的 z 分量接近 -1
    assert env.projected_gravity()[2] == pytest.approx(-1.0, abs=0.2)
    assert env.base_height() == pytest.approx(env.base_position()[2])
    # 机体系与世界系线速度之间差一个姿态旋转
    assert env.base_linear_velocity_body() == pytest.approx(rotation.T @ env.base_linear_velocity())


def test_foot_contacts_and_slip_require_ground_contact(env):
    env.reset(seed=0)
    env.data.qpos[env._base_qpos_adr + 2] = 0.2  # 踩穿地面，必定接触
    mujoco.mj_forward(env.model, env.data)

    assert env.num_feet == 1
    assert env.foot_contacts().shape == (1,)
    assert env.foot_contacts()[0]
    assert env.foot_velocities().shape == (1, 3)


def test_illegal_contact_detects_non_foot_ground_contact(env):
    """大腿、机身这些部位触地即"摔了"。

    判定按 body 名字找部位，所以换机器人不用改代码；只有足端排除在外。
    """
    env.reset(seed=0)
    assert not env.illegal_contact()

    env.data.qpos[env._base_qpos_adr + 2] = 0.15  # 整条腿贴地
    mujoco.mj_forward(env.model, env.data)

    assert env.illegal_contact()


def test_terrain_sampling_falls_back_to_ground_level_without_terrain(env):
    """模型里没有地形几何时射线打空，返回地面高度而不是 NaN。

    填 NaN 会让整条 critic 观测报废，填 0 只是少了一条信息。
    """
    env.reset(seed=0)
    heights = env.sample_terrain_heights(9)

    assert heights.shape == (9,)
    assert np.all(np.isfinite(heights))
    assert np.all(heights == 0.0)


def test_repr_mentions_morphology_and_feet(env):
    text = repr(env)
    assert MINI_SPEC.name in text
    assert "feet=1" in text
