"""速度跟踪任务。

分两段：上面用内嵌的最小形态验任务逻辑（指令采样、默认权重、注册表接线），
秒级可跑；末尾标了 slow，用真实的 G1 / Go2 验"同一份任务代码在两个形态上都
能跑起来"——这条只有真加载模型才有意义，而它依赖 Menagerie，所以两条路都在
缺失资源时跳过，而不是失败。
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from mini_robot import MINI_BASE_HEIGHT, MINI_SPEC, build_robot_model

from robotrl.assets.spec import Morphology
from robotrl.configs.schema import Config, RewardConfig
from robotrl.envs import list_envs, make
from robotrl.envs import mujoco_env as mujoco_env_module
from robotrl.envs import registry as registry_module
from robotrl.envs import tasks as tasks_module
from robotrl.envs.managers import RewardManager, TerminationManager
from robotrl.envs.managers.reward_manager import list_rewards
from robotrl.envs.registry import env_name
from robotrl.envs.tasks.velocity import (
    _DEFAULT_LIN_VEL_Y,
    _DEFAULT_REWARD_SCALES,
    VelocityEnv,
)

velocity_module = tasks_module.velocity


def make_velocity_env(**kwargs) -> VelocityEnv:
    """构造一个测试用的速度跟踪环境。

    失败判据默认全关：最小机器人站不住，判据一开就会在几步内摔倒结束回合，
    而这些用例要验的是指令采样与权重，不是它站不站得住。
    """
    params: dict = dict(
        spec=MINI_SPEC,
        robot_model=build_robot_model(),
        init_base_height=MINI_BASE_HEIGHT,
        termination_manager=TerminationManager(),
        seed=0,
    )
    params.update(kwargs)
    return VelocityEnv(**params)


def no_action(env: VelocityEnv) -> np.ndarray:
    return np.zeros(env.action_dim, dtype=np.float32)


# ---------------------------------------------------------------------------
# 注册表接线
# ---------------------------------------------------------------------------


def test_velocity_is_registered_as_a_shared_task():
    """任务按任务名注册一次，所有形态共用。

    若按「形态-任务」逐个注册，加一个机器人就要加一个工厂，而工厂里除了
    形态名没有任何差别——那份重复迟早会分叉。
    """
    assert "velocity" in list_envs()


def test_env_name_joins_morphology_and_task():
    assert env_name("g1", "velocity") == "G1-Velocity"
    assert env_name("h1", "rough") == "H1-Rough"
    assert env_name("g1", "Velocity") == "G1-Velocity"


def test_env_name_keeps_the_documented_spelling_of_each_robot():
    """形态名的拼写要和文档、日志、报错信息里的一致。

    `go2` 是这里唯一的坑：`upper()` 会得到 `GO2`，而三份文档里写的都是 `Go2`。
    注册表查找不区分大小写，所以拼错了也不会报错——正因如此才需要一个用例把它
    钉住，否则它会一直错到有人去读日志的那天。
    """
    assert env_name("go2", "velocity") == "Go2-Velocity"
    for robot in ("g1", "h1", "go2"):
        assert env_name(robot, "velocity").startswith(robot.capitalize())


def test_name_lookup_is_case_insensitive(monkeypatch):
    """配置里写 Go2-Velocity 还是 go2-velocity 都要能构造出来。

    名字是给人读的，大小写写错不该变成一次"环境未注册"的报错。
    """

    def fake_load_robot(spec, *, scene="scene.xml", mutate=None):
        return build_robot_model(spec)

    monkeypatch.setattr(registry_module, "get_spec", lambda name: MINI_SPEC)
    monkeypatch.setattr(mujoco_env_module, "load_robot", fake_load_robot)

    for name in ("Go2-Velocity", "go2-velocity", "GO2-VELOCITY"):
        assert isinstance(make(name, config=Config()), VelocityEnv)


def test_make_resolves_morphology_and_builds_the_task(monkeypatch):
    """make("Go2-Velocity") 的两件事：按前半段取形态、按后半段找任务工厂。

    这里把形态与加载都换成内嵌的那份，验的是接线对不对——名字拆得对不对、
    spec 有没有传进工厂。真实模型加载由末尾的 slow 用例负责。
    """
    calls: list[tuple[object, str]] = []

    def fake_load_robot(spec, *, scene="scene.xml", mutate=None):
        calls.append((spec, scene))
        return build_robot_model(spec)

    monkeypatch.setattr(registry_module, "get_spec", lambda name: MINI_SPEC)
    monkeypatch.setattr(mujoco_env_module, "load_robot", fake_load_robot)

    env = make("Go2-Velocity", config=Config())

    assert isinstance(env, VelocityEnv)
    assert env.spec is MINI_SPEC
    assert calls == [(MINI_SPEC, "scene.xml")]


def test_registered_factory_takes_spec_not_a_robot_name():
    """注册进去的工厂只认 spec，不认机器人名字。

    工厂一旦自己去认机器人，加一种形态就要动所有工厂，注册表也就白建了。
    这里走的是注册表实际持有的那个入口，不是直接构造类。
    """
    entry = inspect.signature(velocity_module._factory).parameters
    assert "spec" in entry
    assert "config" in entry

    env = velocity_module._factory(
        spec=MINI_SPEC,
        robot_model=build_robot_model(),
        init_base_height=MINI_BASE_HEIGHT,
        termination_manager=TerminationManager(),
    )

    assert isinstance(env, VelocityEnv)
    assert env.spec is MINI_SPEC


# ---------------------------------------------------------------------------
# 指令
# ---------------------------------------------------------------------------


def test_command_is_sampled_at_reset_not_left_at_zero():
    """复位后必须有指令。

    第一步的 cmd 段若是 0，策略会把"站着不动"当成当前目标；指令换得再勤，
    每个回合的第一步都喂了一个错误目标。
    """
    env = make_velocity_env(zero_command_prob=0.0)
    env.reset(seed=0)

    assert np.any(env.command != 0.0)
    parts = env.obs_contract.split(env._observe().policy)
    assert parts["cmd"].tolist() == pytest.approx(env.command.tolist())


def test_command_holds_until_the_hold_count_elapses():
    """指令按保持步数换，不是每步都换。

    每步都换等于把指令变成噪声：策略来不及响应，回报里混进的是它学不会的部分。
    """
    env = make_velocity_env(cmd_hold_steps=5, zero_command_prob=0.0)
    env.reset(seed=0)
    initial = env.command.copy()

    for _ in range(4):
        env.step(no_action(env))
        assert env.command == pytest.approx(initial)

    env.step(no_action(env))
    assert env.command != pytest.approx(initial)


def test_zero_command_probability_produces_a_stand_still_command():
    """一部分回合要以"原地站住"为目标。

    没有这一类样本，策略学到的永远是"有指令就冲"，而上板后最常被要求的能力
    恰恰是站着不动。
    """
    env = make_velocity_env(cmd_hold_steps=1, zero_command_prob=1.0)
    env.reset(seed=0)

    for _ in range(5):
        env.step(no_action(env))
        assert env.command == pytest.approx([0.0, 0.0, 0.0])


def test_command_components_stay_within_their_ranges():
    env = make_velocity_env(
        lin_vel_x_range=(0.3, 0.6),
        lin_vel_y_range=(-0.2, 0.2),
        ang_vel_yaw_range=(0.5, 0.5),
        cmd_hold_steps=1,
        zero_command_prob=0.0,
    )
    env.reset(seed=0)

    for _ in range(20):
        env.step(no_action(env))
        vx, vy, yaw = env.command
        assert 0.3 <= vx <= 0.6
        assert -0.2 <= vy <= 0.2
        assert yaw == pytest.approx(0.5)


def test_command_ranges_report_the_overrides():
    """范围要能被读出来：日志和文档都靠它，写死了就对不上实际的采样范围。"""
    env = make_velocity_env(lin_vel_x_range=(0.1, 0.2), ang_vel_yaw_range=(0.0, 0.0))

    assert env.command_ranges["lin_vel_x"] == (0.1, 0.2)
    assert env.command_ranges["ang_vel_yaw"] == (0.0, 0.0)
    # 没覆写的那个维度保持默认：侧移范围比前后小，这是刻意的
    assert env.command_ranges["lin_vel_y"] == _DEFAULT_LIN_VEL_Y


def test_zero_hold_steps_is_rejected():
    """保持 0 步意味着每步都换指令，是配置写错的典型形态，当场报错更好查。"""
    with pytest.raises(ValueError, match="cmd_hold_steps"):
        make_velocity_env(cmd_hold_steps=0)


def test_command_is_reset_when_the_episode_restarts():
    """回合结束时指令要清掉并立刻重采。

    留着上一回合的指令，新回合的前几步会带着过期的目标——而这几步恰好是
    策略最脆弱的时候。
    """
    env = make_velocity_env(cmd_hold_steps=1000, zero_command_prob=0.0, max_episode_steps=2)
    env.reset(seed=0)
    first = env.command.copy()

    for _ in range(2):
        env.step(no_action(env))

    assert env.command != pytest.approx(first)


def test_fixed_command_survives_the_task_resampling():
    """set_command 之后，任务自己的指令采样不能把它盖掉。

    回放与评估靠这个入口指定"让机器人怎么走"。一旦被采样盖掉，回放出来的
    行为与训练时不一致，而症状只是"结果对不上"，很难往指令上想。
    """
    fixed = [0.7, -0.3, 0.2]
    env = make_velocity_env(cmd_hold_steps=1, zero_command_prob=0.0)
    env.set_command(fixed)
    env.reset(seed=0)

    assert env.command == pytest.approx(fixed)
    for _ in range(3):
        result = env.step(no_action(env))
        assert env.command == pytest.approx(fixed)
        assert env.obs_contract.split(result.obs.policy)["cmd"] == pytest.approx(fixed)

    env.clear_command()
    env.step(no_action(env))
    assert env.command != pytest.approx(fixed)


# ---------------------------------------------------------------------------
# 奖励与判据
# ---------------------------------------------------------------------------


def test_default_reward_scales_are_all_registered():
    """默认权重里出现未注册的名字，会在构造环境时报错——先在这里挡住。"""
    registered = set(list_rewards())

    assert set(_DEFAULT_REWARD_SCALES) <= registered
    assert all(scale != 0.0 for scale in _DEFAULT_REWARD_SCALES.values())


def test_env_uses_the_task_default_scales():
    env = make_velocity_env()

    assert env.reward_manager.active_names == tuple(sorted(_DEFAULT_REWARD_SCALES))
    for name, scale in _DEFAULT_REWARD_SCALES.items():
        assert env.reward_manager.scale_of(name) == pytest.approx(scale)


def test_config_can_zero_out_a_default_term():
    """配置写的权重优先于任务默认值，写 0 就是关掉。

    这条路径是调参的主入口：关掉一项惩罚不该需要改任务代码。
    """
    config = Config()
    config.reward = RewardConfig(scales={"feet_slip": 0.0})
    env = make_velocity_env(config=config)

    assert "feet_slip" not in env.reward_manager.active_names
    assert env.reward_manager.scale_of("feet_slip") == 0.0


def test_point_morphology_drops_the_foot_slip_term():
    """没有足端的形态算不出足端滑动，留着只会得到一个恒为 0 的项。

    恒为 0 的项不会影响梯度，但会让"我明明加权了"这类疑问多绕一圈。
    """
    point_spec = replace(MINI_SPEC, name="mini_point", morphology=Morphology.POINT)
    env = make_velocity_env(spec=point_spec, robot_model=build_robot_model(point_spec))

    assert "feet_slip" not in env.default_reward_scales()
    assert "feet_slip" not in env.reward_manager.active_names
    assert "track_lin_vel_xy" in env.reward_manager.active_names


def test_default_termination_terms_cover_the_generic_failures():
    env = make_velocity_env(termination_manager=None)

    assert set(env.termination_manager.active_names) == {
        "base_height_low",
        "illegal_contact",
        "nan_state",
        "tilt_too_large",
    }


def test_reward_terms_match_the_active_terms():
    """分项日志的键必须与启用项一致，否则曲线对不上配置。"""
    env = make_velocity_env()
    env.reset(seed=0)
    env.step(no_action(env))

    assert tuple(sorted(env.reward_terms)) == env.reward_manager.active_names


def test_rollout_produces_finite_rewards():
    env = make_velocity_env()
    env.reset(seed=0)

    for _ in range(10):
        result = env.step(np.full(env.action_dim, 0.2, dtype=np.float32))
        assert np.isfinite(result.reward)
        assert np.all(np.isfinite(result.obs.policy))


def test_reward_manager_in_can_be_overridden_for_debugging():
    """环境允许注入管理器，调试时能只留一项看它单独的样子。"""
    env = make_velocity_env(
        reward_manager=RewardManager(defaults={"track_lin_vel_xy": 1.0}),
    )

    assert env.reward_manager.active_names == ("track_lin_vel_xy",)


# ---------------------------------------------------------------------------
# 真实形态
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("robot", ["go2", "g1"])
def test_real_robot_velocity_env_runs(robot):
    """同一份任务代码在两个形态上都能跑起来，且维度全部跟着 spec 走。

    这是"任务不写死机器人"的最终验收：Go2 是 12 个关节的机器狗，G1 是 15 个
    关节的人形，两边的观测维度、动作维度、足数都不一样，代码一行没改。
    """
    try:
        env = make(env_name(robot, "velocity"))
    except (FileNotFoundError, NotImplementedError) as exc:
        pytest.skip(f"模型或加载器未就绪：{exc}")

    obs, _ = env.reset(seed=0)
    result = env.step(np.zeros(env.action_dim, dtype=np.float32))

    assert env.action_dim == env.spec.n_dof
    assert obs.policy.shape == (env.obs_dim,)
    assert result.obs.policy.shape == (env.obs_dim,)
    assert np.isfinite(result.reward)
    assert np.all(np.isfinite(result.obs.policy))

    parts = env.obs_contract.split(result.obs.policy)
    assert parts["dof_pos"].shape == (env.spec.n_dof,)
    assert parts["cmd"].tolist() == pytest.approx(env.command.tolist())


@pytest.mark.slow
def test_real_robot_action_moves_the_controlled_joints():
    """动作确实写到了 ctrl 上，且换算用的是 spec 的默认角与 action_scale。

    内嵌模型上验过同一件事，这里再验一遍是因为 ctrl 的语义由 loader 统一
    （见 assets/loader.py 的模块文档）——把上游的力矩电机改造成位置伺服是
    loader 做的，两边必须对得上。
    """
    try:
        env = make("Go2-Velocity")
    except (FileNotFoundError, NotImplementedError) as exc:
        pytest.skip(f"模型或加载器未就绪：{exc}")

    env.reset(seed=0)
    env.step(np.full(env.action_dim, 0.5, dtype=np.float32))

    expected = env.spec.default_angles_array + 0.5 * env.spec.action_scale
    assert env.data.ctrl[env.actuator_ids] == pytest.approx(expected)
    assert env.data.ctrl[env.held_actuator_ids] == pytest.approx(env.held_targets)

    # 位置伺服：出力 = kp*ctrl - kp*q - kd*qd。kp 必须同时出现在 gainprm 与
    # biasprm 里，且量级来自 spec（这里带域随机化，所以只看区间）。
    kp = env.model.actuator_gainprm[env.actuator_ids, 0]
    assert np.allclose(kp, -env.model.actuator_biasprm[env.actuator_ids, 1])
    low, high = env.events_config.pd_gain_range
    assert np.all(kp >= low * env.spec.pd_kp_array * 0.999)
    assert np.all(kp <= high * env.spec.pd_kp_array * 1.001)


@pytest.mark.slow
def test_real_robot_terrain_reaches_the_model():
    """地形配置真的走到了模型里，不是只在配置对象里躺着。"""
    config = Config()
    config.terrain.kind = "rough"
    try:
        env = make("Go2-Velocity", config=config)
    except (FileNotFoundError, NotImplementedError) as exc:
        pytest.skip(f"模型或加载器未就绪：{exc}")

    assert env.model.nhfield == 1
    heights = env.sample_terrain_heights(9)
    assert np.all(np.isfinite(heights))
    assert heights.max() - heights.min() > 0.0


@pytest.mark.slow
def test_real_robot_reset_and_step_are_reproducible():
    """真实模型上也要可复现：域随机化、初始噪声、推力都得走环境自己的随机源。"""
    try:
        env_a = make("Go2-Velocity", seed=3)
        env_b = make("Go2-Velocity", seed=3)
    except (FileNotFoundError, NotImplementedError) as exc:
        pytest.skip(f"模型或加载器未就绪：{exc}")

    def rollout(env) -> list[float]:
        env.reset(seed=3)
        action = np.full(env.action_dim, 0.1, dtype=np.float32)
        return [env.step(action).reward for _ in range(5)]

    assert rollout(env_a) == rollout(env_b)


@pytest.mark.slow
def test_real_robot_falls_and_terminates():
    """把机身放倒，回合必须以 terminated 结束（超时不算失败）。

    这条走的是完整判据链：倾角判据为真 → terminated；而超时必须是 truncated，
    两者在 PPO 里的价值估计处理不同。
    """
    try:
        env = make("Go2-Velocity")
    except (FileNotFoundError, NotImplementedError) as exc:
        pytest.skip(f"模型或加载器未就绪：{exc}")

    env.reset(seed=0)
    half = np.pi / 4.0
    env.data.qpos[env._base_qpos_adr + 3 : env._base_qpos_adr + 7] = [
        np.cos(half),
        np.sin(half),
        0.0,
        0.0,
    ]
    mujoco.mj_forward(env.model, env.data)

    result = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert result.terminated
    assert not result.truncated
