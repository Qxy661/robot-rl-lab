"""toy 环境与 BaseEnv 管线测试。

toy 环境不依赖 MuJoCo，因此这些用例能在任何机器上秒级跑完，可以覆盖那些
"逻辑上必须成立"的性质：观测维度与契约一致、超时与终止分开、种子可复现、
复位发生在奖励之后。真实机器人环境沿用同一套基类，这些性质同样受保护。
"""

from __future__ import annotations

import numpy as np
import pytest

from robotrl.configs.schema import Config
from robotrl.envs import list_envs, make
from robotrl.envs.toy import TOY_SPEC, ToyVelocityEnv


@pytest.fixture
def env() -> ToyVelocityEnv:
    return ToyVelocityEnv(seed=0)


# ---------------------------------------------------------------------------
# 观测与动作的维度契约
# ---------------------------------------------------------------------------


def test_reset_returns_valid_observation(env):
    obs, info = env.reset(seed=0)

    assert obs.policy.shape == (env.obs_dim,)
    assert obs.critic.shape == (env.critic_obs_dim,)
    assert obs.policy.dtype == np.float32
    assert np.all(np.isfinite(obs.policy))
    assert info["step"] == 0


def test_step_returns_well_formed_result(env):
    obs, _ = env.reset(seed=0)
    result = env.step(np.zeros(env.action_dim, dtype=np.float32))

    assert result.obs.policy.shape == (env.obs_dim,)
    assert isinstance(result.reward, float)
    assert isinstance(result.terminated, bool)
    assert isinstance(result.truncated, bool)
    assert np.isfinite(result.reward)


def test_observation_matches_declared_contract(env):
    """观测必须能被契约按段拆开，且各段宽度正确。"""
    obs, _ = env.reset(seed=0)
    parts = env.obs_contract.split(obs.policy)

    assert parts["pos"].shape == (2,)
    assert parts["vel"].shape == (2,)
    assert parts["cmd"].shape == (2,)
    assert env.obs_contract.concat(parts).tolist() == pytest.approx(obs.policy.tolist())


def test_critic_observation_extends_policy_prefix(env):
    """critic 观测的前缀必须与 policy 完全一致，只有尾部多出特权信息。"""
    obs, _ = env.reset(seed=0)
    n = env.obs_contract.total_dim

    assert env.critic_obs_dim == n + 1
    assert np.allclose(obs.critic[:n], obs.policy)
    assert env.critic_obs_contract.index("cmd") == env.obs_contract.index("cmd")


def test_action_dim_follows_spec(env):
    assert env.action_dim == TOY_SPEC.n_dof == 2


# ---------------------------------------------------------------------------
# 管线行为
# ---------------------------------------------------------------------------


def test_actions_are_clipped(env):
    """超出范围的动作要按契约裁到 [-1, 1]，再乘 action_scale 变成加速度。

    这里用"走了多远"间接验证：给定同样的初速度，动作越大加速度越大。
    大动作与小动作如果产生同样的位移，说明裁剪或缩放没生效。
    """

    def travel(action_value: float) -> float:
        e = ToyVelocityEnv(seed=0)
        e.reset(seed=0)
        e._vel = np.array([0.0, 0.0])
        e._pos = np.array([0.0, 0.0])
        for _ in range(5):
            e.step(np.full(2, action_value, dtype=np.float32))
        return float(np.linalg.norm(e._pos))

    assert travel(1.0) > travel(0.5) > travel(0.0)
    # 超大动作应被裁到与 1.0 相同
    assert travel(5.0) == pytest.approx(travel(1.0))


def test_episode_truncates_at_max_steps():
    env = ToyVelocityEnv(max_episode_steps=10, seed=0)
    env.reset(seed=0)

    for i in range(10):
        result = env.step(np.zeros(2, dtype=np.float32))
        if i < 9:
            assert not result.truncated
    assert result.truncated
    assert not result.terminated  # 超时不是失败


def test_out_of_bounds_terminates_not_truncates():
    """跑出边界的语义是"任务失败"，与超时严格区分——两者的价值估计处理不同。"""
    env = ToyVelocityEnv(max_episode_steps=10_000, seed=0)
    env.reset(seed=0)

    env._pos = np.array([env._BOUND + 1.0, 0.0])
    result = env.step(np.zeros(2, dtype=np.float32))

    assert result.terminated
    assert not result.truncated


def test_reset_happens_after_reward_so_terminal_state_is_observable():
    """终止那一步的奖励必须在复位之前算出来，否则奖励会基于新回合的初始状态。"""
    env = ToyVelocityEnv(max_episode_steps=10_000, seed=0)
    env.reset(seed=0)

    env._pos = np.array([env._BOUND + 1.0, 0.0])
    env._vel = np.array([0.5, 0.5])
    result = env.step(np.zeros(2, dtype=np.float32))

    # 终止时奖励基于越界前那一步的状态，而观测已经是复位后的新状态
    assert result.terminated
    assert np.linalg.norm(env._pos) < env._BOUND
    assert result.info["is_terminal_step"]


def test_step_counter_resets_on_new_episode():
    env = ToyVelocityEnv(max_episode_steps=5, seed=0)
    env.reset(seed=0)
    for _ in range(5):
        result = env.step(np.zeros(2, dtype=np.float32))
    assert result.info["step"] == 0


# ---------------------------------------------------------------------------
# 可复现性
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_trajectories():
    """同一份代码同一种子必须走出同一条轨迹，否则实验结论无法比较。"""

    def rollout(seed: int) -> list[float]:
        e = ToyVelocityEnv(seed=seed)
        e.reset(seed=seed)
        rewards = []
        for _ in range(20):
            results = e.step(np.full(2, 0.3, dtype=np.float32))
            rewards.append(results.reward)
        return rewards

    assert rollout(7) == rollout(7)
    assert rollout(7) != rollout(8)


def test_reset_seed_overrides_construction_seed():
    env = ToyVelocityEnv(seed=999)
    obs_a, _ = env.reset(seed=1)
    obs_b, _ = env.reset(seed=1)
    assert np.array_equal(obs_a.policy, obs_b.policy)


def test_commands_are_resampled_periodically():
    """指令每若干步换一次；换得太频繁策略来不及响应，一直不换则学不到跟踪。"""
    env = ToyVelocityEnv(seed=0)
    env.reset(seed=0)

    commands = []
    for _ in range(env._CMD_HOLD_STEPS * 4):
        env.step(np.zeros(2, dtype=np.float32))
        commands.append(tuple(np.round(env._cmd, 6)))

    assert len(set(commands)) > 1


def test_domain_randomization_varies_across_episodes():
    """每回合的阻尼系数应当不同，这是域随机化生效的直接证据。"""
    env = ToyVelocityEnv(seed=0)
    dampings = set()
    for i in range(20):
        env.reset(seed=i)
        dampings.add(round(env._damping, 6))
    assert len(dampings) > 1


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


def test_toy_is_registered_and_constructible():
    assert "toy" in list_envs()
    env = make("toy")
    obs, _ = env.reset(seed=0)
    assert obs.policy.shape == (env.obs_dim,)


def test_make_accepts_config():
    cfg = Config()
    cfg.env.max_episode_steps = 42
    env = make("toy", config=cfg)
    assert env.max_episode_steps == 42


def test_make_unknown_name_lists_available():
    with pytest.raises(KeyError, match="toy"):
        make("Nonexistent-Task")
