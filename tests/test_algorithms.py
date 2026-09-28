"""算法层测试。

这里断言的不是"训练能跑通"，而是**那些错了也不报错的实现细节**。PPO 和 SAC
的绝大多数 bug 都不会抛异常，只会让曲线悄悄变平或缓慢发散，等到发现时已经
烧掉几小时机时。所以每个用例都对准一个具体的坑：

- tanh 压缩后漏掉 log 概率的雅可比项；
- 把超时截断当成真终止，导致价值凭空掉一截；
- 用 done 而不是 terminated 做自举掩码；
- 归一化统计量没进 state_dict，续训时观测尺度突变；
- 联合对数概率没有按动作维求和，形状悄悄广播成 (batch, action_dim)。

toy 环境不依赖 MuJoCo，全部用例都能在秒级跑完。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from robotrl.algorithms.networks import (
    ActorCritic,
    SACActor,
    SquashedGaussian,
    TwinQNetwork,
)
from robotrl.algorithms.ppo import PPO
from robotrl.algorithms.sac import SAC
from robotrl.algorithms.storage import ReplayBuffer, RolloutStorage
from robotrl.algorithms.trainer import PPOTrainer, SACTrainer, make_trainer
from robotrl.configs.schema import PPOConfig, SACConfig
from robotrl.envs.toy import ToyVelocityEnv

OBS_DIM = 6
CRITIC_OBS_DIM = 7
ACTION_DIM = 2


@pytest.fixture
def env() -> ToyVelocityEnv:
    return ToyVelocityEnv(seed=0)


# ---------------------------------------------------------------------------
# 构造存储的辅助函数
# ---------------------------------------------------------------------------


def make_storage(num_steps: int, num_envs: int = 1) -> RolloutStorage:
    return RolloutStorage(num_steps, num_envs, OBS_DIM, CRITIC_OBS_DIM, ACTION_DIM)


def fill_storage(
    storage: RolloutStorage,
    *,
    rewards: list[list[float]],
    values: list[list[float]],
    terminated: list[list[bool]] | None = None,
    truncated: list[list[bool]] | None = None,
) -> None:
    """按时间步往 storage 里灌数据。

    迭代次数取 rewards 的长度而不是 storage.num_steps，这样可以故意只写一半，
    用来测"rollout 没采满就求优势"的行为。
    """
    num_envs = storage.num_envs
    zeros = torch.zeros(num_envs, OBS_DIM)
    for t in range(len(rewards)):
        storage.add(
            obs=zeros,
            critic_obs=torch.zeros(num_envs, CRITIC_OBS_DIM),
            action=torch.zeros(num_envs, ACTION_DIM),
            log_prob=torch.zeros(num_envs),
            reward=torch.tensor(rewards[t], dtype=torch.float32),
            terminated=torch.tensor(
                terminated[t] if terminated else [False] * num_envs
            ),
            truncated=torch.tensor(truncated[t] if truncated else [False] * num_envs),
            value=torch.tensor(values[t], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# PPO：核心不变量
# ---------------------------------------------------------------------------


def test_ppo_stored_log_prob_matches_current_policy_before_update():
    """更新开始前，KL 必须恰好是 0。

    这条不变量说的是：策略对存下来的动作算出的 log 概率，必须与采样那一刻
    存进去的完全一致。它同时管住了四件事——tanh 修正有没有在两条路径上都加、
    观测归一化有没有在两条路径上都用、联合对数概率有没有按动作维求和、存进
    storage 的动作有没有被改写。任何一处不一致，比值就不再是 1，PPO 的重要性
    采样从第一个 mini-batch 起就是错的。
    """
    torch.manual_seed(0)
    policy = ActorCritic(OBS_DIM, ACTION_DIM, critic_obs_dim=CRITIC_OBS_DIM)
    cfg = PPOConfig(num_steps_per_env=8, num_learning_epochs=1, num_mini_batches=2)
    algo = PPO(policy, cfg)

    # 故意先动一下归一化统计量，把"两处归一化必须一致"也纳入考察。
    policy.update_normalizer(torch.randn(64, OBS_DIM) * 3.0 + 1.0)
    policy.update_critic_normalizer(torch.randn(64, CRITIC_OBS_DIM))

    storage = make_storage(8)
    for _ in range(8):
        obs = torch.randn(1, OBS_DIM)
        critic_obs = torch.randn(1, CRITIC_OBS_DIM)
        action, log_prob, value = algo.act(obs, critic_obs)
        storage.add(
            obs=obs,
            critic_obs=critic_obs,
            action=action,
            log_prob=log_prob,
            reward=torch.zeros(1),
            terminated=torch.tensor([False]),
            truncated=torch.tensor([False]),
            value=value,
        )

    data = storage.flatten()
    log_prob_now = policy.distribution(data["obs"]).log_prob(data["actions"])

    assert log_prob_now.shape == data["log_probs"].shape
    assert torch.allclose(log_prob_now, data["log_probs"], atol=1e-5)


def test_ppo_mini_batches_cover_every_sample_exactly_once():
    """每个 epoch 必须把全部样本过一遍且不重复。

    不整除时若直接丢掉余数，被丢的永远是索引最大的那批样本，也就是 rollout
    末尾的时间步——那里恰恰是回合结束、终止与截断标志出现的地方。
    """
    storage = make_storage(num_steps=5, num_envs=3)  # 15 个样本，用 4 个 batch 切
    fill_storage(
        storage,
        rewards=[[1.0] * 3 for _ in range(5)],
        values=[[0.0] * 3 for _ in range(5)],
    )
    storage.compute_gae(torch.zeros(3), gamma=0.99, lam=0.95)

    for num_mini_batches in (1, 2, 4, 7, 15):
        seen = torch.cat(list(storage.mini_batch_indices(num_mini_batches, 1)))
        assert sorted(seen.tolist()) == list(range(15))


def test_rollout_storage_refuses_overflow():
    """写满后再写必须报错，而不是静默覆盖第一帧。

    静默覆盖会让优势计算用上已经不存在的后继——错得很隐蔽。
    """
    storage = make_storage(num_steps=2)
    fill_storage(storage, rewards=[[1.0], [1.0]], values=[[0.0], [0.0]])
    assert storage.full
    with pytest.raises(RuntimeError, match="写满"):
        storage.add(
            obs=torch.zeros(1, OBS_DIM),
            critic_obs=torch.zeros(1, CRITIC_OBS_DIM),
            action=torch.zeros(1, ACTION_DIM),
            log_prob=torch.zeros(1),
            reward=torch.zeros(1),
            terminated=torch.tensor([False]),
            truncated=torch.tensor([False]),
            value=torch.zeros(1),
        )


# ---------------------------------------------------------------------------
# PPO：GAE 与终止 / 截断
# ---------------------------------------------------------------------------


def test_gae_bootstraps_on_timeout_but_not_on_termination():
    """同样一串奖励，末步超时和末步真终止的优势必须不同。

    超时是"不采样了"，状态还在，价值要自举；真终止是任务失败，状态价值按
    定义就是 0。把两者合并成一个 done，超时的那些回合每一条都会凭空少掉
    gamma * V(s') 的回报，而且超时越频繁偏差越大。
    """
    gamma, lam = 1.0, 1.0
    rewards = [[1.0], [1.0], [1.0]]
    values = [[0.0], [0.0], [0.0]]
    last_values = torch.tensor([10.0])

    timeout = make_storage(3)
    fill_storage(
        timeout,
        rewards=rewards,
        values=values,
        truncated=[[False], [False], [True]],
    )
    _, adv_timeout = timeout.compute_gae(last_values, gamma, lam)

    failed = make_storage(3)
    fill_storage(
        failed,
        rewards=rewards,
        values=values,
        terminated=[[False], [False], [True]],
    )
    _, adv_failed = failed.compute_gae(last_values, gamma, lam)

    # 超时：末步优势 = 1 + 1·V(s') = 11，往前每步再加 1
    assert adv_timeout[:, 0].tolist() == pytest.approx([13.0, 12.0, 11.0])
    # 真终止：末步优势 = 1，自举项被完全去掉
    assert adv_failed[:, 0].tolist() == pytest.approx([3.0, 2.0, 1.0])


def test_gae_does_not_leak_next_episode_value_across_a_reset():
    """回合中段超时后，新回合的价值不能算进上一回合的优势。

    环境在 step() 内部就复位了，所以缓冲里下一帧的观测属于新回合。不把这一
    帧屏蔽掉的话，上一回合的优势会被塞进一个毫不相干的 V(s')。
    """
    storage = make_storage(3)
    fill_storage(
        storage,
        rewards=[[1.0], [1.0], [1.0]],
        # values[1] 与 values[2] 属于新回合，数值故意给大，泄漏进来一眼能看出
        values=[[0.0], [100.0], [100.0]],
        truncated=[[True], [False], [False]],
    )
    _, advantages = storage.compute_gae(torch.zeros(1), gamma=1.0, lam=1.0)

    # t=0 是超时那一步：拿不到 s'，自举项为 0，advantage 就是这一步的奖励
    assert advantages[0, 0].item() == pytest.approx(1.0)
    # 新回合内部照常自举，末步用 last_values = 0
    assert advantages[1, 0].item() == pytest.approx(-98.0)


def test_gae_rejects_incomplete_rollout():
    storage = make_storage(4)
    fill_storage(storage, rewards=[[1.0], [1.0]], values=[[0.0], [0.0]])
    with pytest.raises(RuntimeError, match="采满"):
        storage.compute_gae(torch.zeros(1), gamma=0.99, lam=0.95)


def test_returns_equal_advantages_plus_values():
    storage = make_storage(3)
    fill_storage(
        storage,
        rewards=[[0.5], [0.5], [0.5]],
        values=[[1.0], [2.0], [3.0]],
    )
    returns, advantages = storage.compute_gae(torch.zeros(1), gamma=0.99, lam=0.95)
    assert torch.allclose(returns, advantages + storage.values)


# ---------------------------------------------------------------------------
# PPO：更新规则
# ---------------------------------------------------------------------------


def ppo_update_setup(**cfg_kwargs):
    """造一轮随机 rollout，返回 (policy, algo, storage)。"""
    torch.manual_seed(0)
    policy = ActorCritic(OBS_DIM, ACTION_DIM, critic_obs_dim=CRITIC_OBS_DIM)
    cfg = PPOConfig(num_steps_per_env=8, num_learning_epochs=2, num_mini_batches=2)
    for key, value in cfg_kwargs.items():
        setattr(cfg, key, value)
    algo = PPO(policy, cfg)

    storage = make_storage(8, num_envs=2)
    for _ in range(8):
        obs = torch.randn(2, OBS_DIM)
        critic_obs = torch.randn(2, CRITIC_OBS_DIM)
        action, log_prob, value = algo.act(obs, critic_obs)
        storage.add(
            obs=obs,
            critic_obs=critic_obs,
            action=action,
            log_prob=log_prob,
            reward=torch.rand(2),
            terminated=torch.tensor([False, False]),
            truncated=torch.tensor([False, False]),
            value=value,
        )
    storage.compute_gae(torch.zeros(2), cfg.gamma, cfg.lam)
    return policy, algo, storage


def test_ppo_update_produces_finite_metrics():
    _, algo, storage = ppo_update_setup()
    metrics = algo.update(storage)

    assert set(metrics) >= {
        "ppo/actor_loss", "ppo/value_loss", "ppo/clip_fraction", "ppo/kl", "ppo/lr_actor"
    }
    assert all(math.isfinite(v) for v in metrics.values())


def test_ppo_zero_epochs_leaves_policy_untouched():
    """学习轮数为 0 时一次梯度都不该走——这是"更新本身有没有副作用"的对照。"""
    policy, algo, storage = ppo_update_setup(num_learning_epochs=0)
    before = [p.detach().clone() for p in policy.parameters()]
    metrics = algo.update(storage)

    for old, new in zip(before, policy.parameters(), strict=True):
        assert torch.equal(old, new)
    assert metrics["ppo/kl"] == 0.0


def test_ppo_dual_learning_rates_are_independent():
    """actor 与 critic 的学习率各自独立，且参数集合不重叠。

    重叠会让同一份参数每步被更新两次，等效学习率翻倍，且完全静默。
    """
    policy = ActorCritic(OBS_DIM, ACTION_DIM, critic_obs_dim=CRITIC_OBS_DIM)
    cfg = PPOConfig(lr_actor=3e-4, lr_critic=1e-3)
    algo = PPO(policy, cfg)

    actor_ids = {id(p) for p in policy.actor_parameters()}
    critic_ids = {id(p) for p in policy.critic_parameters()}

    assert actor_ids.isdisjoint(critic_ids)
    assert len(actor_ids) + len(critic_ids) == len(list(policy.parameters()))
    assert algo.optimizer_actor.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert algo.optimizer_critic.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_ppo_adaptive_lr_reacts_to_kl():
    """KL 超标降学习率，KL 偏小涨学习率，且两组同比例变化。"""
    policy = ActorCritic(OBS_DIM, ACTION_DIM, critic_obs_dim=CRITIC_OBS_DIM)
    cfg = PPOConfig(lr_actor=3e-4, lr_critic=1e-3, desired_kl=0.01, use_lr_schedule=True)
    algo = PPO(policy, cfg)

    ratio_before = 1e-3 / 3e-4
    assert algo._adapt_learning_rate(0.01 * 2.5) is True

    assert algo.optimizer_actor.param_groups[0]["lr"] < 3e-4
    assert algo.optimizer_critic.param_groups[0]["lr"] < 1e-3
    ratio_after = (
        algo.optimizer_critic.param_groups[0]["lr"] / algo.optimizer_actor.param_groups[0]["lr"]
    )
    assert ratio_after == pytest.approx(ratio_before)

    small = PPO(policy, cfg)
    small._adapt_learning_rate(0.01 / 5.0)
    assert small.optimizer_actor.param_groups[0]["lr"] > 3e-4


def test_ppo_lr_schedule_has_floors_and_ceilings():
    """反复降/升学习率不会把它推到 0 或推到不稳定区。"""
    policy = ActorCritic(OBS_DIM, ACTION_DIM, critic_obs_dim=CRITIC_OBS_DIM)
    algo = PPO(policy, PPOConfig())
    for _ in range(100):
        algo._scale_lr(1.0 / 1.5)
    assert algo.optimizer_actor.param_groups[0]["lr"] >= 1e-5
    for _ in range(100):
        algo._scale_lr(1.5)
    assert algo.optimizer_actor.param_groups[0]["lr"] <= 1e-2


# ---------------------------------------------------------------------------
# SAC：tanh 压缩与 log 概率
# ---------------------------------------------------------------------------


def standard_normal_squashed() -> SquashedGaussian:
    """mean=0、std=1 的 tanh 高斯，方便手算校验。"""
    return SquashedGaussian(torch.zeros(1, 3), torch.zeros(1, 3))


def test_tanh_log_prob_carries_the_jacobian_correction():
    """log π(a) = log N(u) - log(1 - tanh²(u))，逐维求和。

    这是 SAC 里最容易漏的一项。漏掉之后训练照样跑，但熵的尺度整体错位，
    alpha 会朝错误方向调，表现为策略过早收敛到确定性动作。
    """
    dist = standard_normal_squashed()
    u = torch.tensor([[0.3, -1.2, 2.0]])
    action = torch.tanh(u)

    log_normal = (-0.5 * u**2 - 0.5 * math.log(2 * math.pi)).sum(dim=-1)
    correction = -torch.log(1.0 - action**2).sum(dim=-1)
    expected = log_normal + correction

    assert dist.log_prob(action).item() == pytest.approx(expected.item(), abs=1e-4)
    # 修正项不是可以忽略的小量：这里它比高斯部分还大
    assert abs(correction.item()) > 0.5


def test_tanh_log_prob_is_consistent_between_sample_and_evaluate_paths():
    """采样路径（直接拿 u）与求值路径（atanh 反解）必须给出同一个值。

    容差按 float32 给：两条路径差了一趟 tanh/atanh 的往返，末位的舍入误差
    是结构性的，不是实现分歧。这里要保证的是"没有漏项"，不是比特级相等。
    """
    dist = SquashedGaussian(torch.randn(1, 4), torch.randn(1, 4))
    u = torch.randn(1, 4)
    action = torch.tanh(u)

    assert dist.log_prob_raw(u).item() == pytest.approx(dist.log_prob(action).item(), abs=1e-4)


@pytest.mark.parametrize(
    "action",
    [
        [1.0, -1.0, 1.0],
        [1.0 - 1e-7, -1.0 + 1e-7, 0.0],
        [0.999999, -0.999999, 1e-8],
        [0.0, 0.0, 0.0],
    ],
    ids=["boundary", "floating_point_edge", "near_boundary", "center"],
)
def test_tanh_log_prob_stays_finite_at_extreme_actions(action):
    """动作贴着 ±1 时 log 概率不能变成 inf 或 nan。

    atanh 在 ±1 处发散，而动作经 float32 舍入后完全可能正好落在边界上。
    一个 nan 会顺着损失污染整轮更新的所有参数，是不可恢复的。
    """
    for log_std in (0.0, -5.0, 2.0):  # 覆盖夹取区间的两端
        dist = SquashedGaussian(torch.zeros(1, 3), torch.full((1, 3), log_std))
        log_prob = dist.log_prob(torch.tensor([action]))
        assert torch.isfinite(log_prob).all(), (action, log_std, log_prob)


def test_squashed_sample_respects_action_range():
    """采样动作绝不越界；均值很大时会有样本正好落在边界上。

    后者不是实现缺陷，而是 float32 的必然：tanh 在 |u| 超过约 9 之后就饱和
    到 1.0。这说明"边界上的动作"在真实数据里会出现，log_prob 的夹取边距
    不是防御性冗余。契约写的是闭区间 [-1, 1]，环境层也会再裁一次。
    """
    dist = SquashedGaussian(torch.randn(512, 2) * 5.0, torch.zeros(512, 2))
    action, log_prob = dist.sample()

    assert action.abs().max().item() <= 1.0
    assert (action.abs() == 1.0).any(), "均值给到 ±5 时应当出现饱和样本"
    assert torch.isfinite(log_prob).all()
    # 饱和样本走求值路径也必须给出有限值
    assert torch.isfinite(dist.log_prob(action)).all()


def test_log_std_is_clamped():
    """标准差被夹在可用区间内，不会塌到 0 也不会涨到失控。"""
    dist = SquashedGaussian(torch.zeros(1, 2), torch.full((1, 2), -100.0))
    assert dist.log_std.min().item() == pytest.approx(-5.0)
    assert dist.std.min().item() > 0.0

    dist = SquashedGaussian(torch.zeros(1, 2), torch.full((1, 2), 100.0))
    assert dist.log_std.max().item() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# SAC：网络与目标值
# ---------------------------------------------------------------------------


def test_policy_outputs_are_bounded_everywhere():
    """两个算法的部署语义一致：输出严格落在 (-1, 1)。"""
    torch.manual_seed(0)
    obs = torch.randn(256, OBS_DIM) * 10.0
    for policy in (
        ActorCritic(OBS_DIM, ACTION_DIM, critic_obs_dim=CRITIC_OBS_DIM),
        SACActor(OBS_DIM, ACTION_DIM, hidden_dims=(32, 32)),
    ):
        action = policy(obs)
        assert action.shape == (256, ACTION_DIM)
        assert action.abs().max().item() < 1.0
        assert torch.isfinite(action).all()


def test_sac_actor_owns_a_single_normalizer_shared_with_q():
    """SAC 只有一份归一化统计量，藏在 actor 里，随它一起导出。

    统计量是滑动平均（momentum=0.99），单次调用几乎不动——这是刻意的，
    它要跟着观测分布慢慢漂移，而不是被某一批数据带跑。所以断言写成收敛
    之后的性质。
    """
    actor = SACActor(OBS_DIM, ACTION_DIM, hidden_dims=(16, 16))
    assert "obs_mean" in actor.state_dict()

    raw = torch.randn(1024, OBS_DIM) * 1.5 + 0.5
    for _ in range(600):
        actor.update_normalizer(raw)

    assert torch.allclose(actor.obs_mean, raw.mean(dim=0), atol=0.1)
    normalized = actor.normalize_obs(raw)
    assert normalized.mean().abs().item() < 0.05
    assert normalized.std().item() == pytest.approx(1.0, abs=0.05)


def test_twin_q_networks_are_independently_initialized():
    """两个 Q 必须独立初始化。

    共用一个初始值的话它们的误差是相关的，取 min 就退化成单 Q，压不住高估。
    """
    torch.manual_seed(0)
    twin = TwinQNetwork(OBS_DIM, ACTION_DIM, hidden_dims=(16, 16))
    q1, q2 = twin(torch.randn(32, OBS_DIM), torch.rand(32, ACTION_DIM) * 2 - 1)

    assert not torch.allclose(q1, q2)
    assert not torch.allclose(
        next(twin.q1.parameters()), next(twin.q2.parameters())
    )


def test_twin_q_min_matches_elementwise_min():
    torch.manual_seed(0)
    twin = TwinQNetwork(OBS_DIM, ACTION_DIM, hidden_dims=(16, 16))
    obs, action = torch.randn(32, OBS_DIM), torch.rand(32, ACTION_DIM) * 2 - 1
    q1, q2 = twin(obs, action)
    assert torch.allclose(twin.min_q(obs, action), torch.minimum(q1, q2))


def sac_setup() -> tuple[SAC, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    actor = SACActor(OBS_DIM, ACTION_DIM, hidden_dims=(16, 16), use_layer_norm=False)
    critic = TwinQNetwork(OBS_DIM, ACTION_DIM, hidden_dims=(16, 16), use_layer_norm=False)
    algo = SAC(actor, critic, SACConfig(gamma=0.5))

    batch_size = 32
    return algo, {
        "obs": torch.randn(batch_size, OBS_DIM),
        "action": torch.rand(batch_size, ACTION_DIM) * 2 - 1,
        "reward": torch.rand(batch_size),
        "next_obs": torch.randn(batch_size, OBS_DIM),
        "terminated": torch.zeros(batch_size, dtype=torch.bool),
        "truncated": torch.zeros(batch_size, dtype=torch.bool),
        "not_terminated": torch.ones(batch_size),
    }


def test_sac_target_skips_bootstrap_on_termination():
    """真终止的转移，目标值必须精确等于奖励本身。"""
    algo, batch = sac_setup()
    batch["terminated"] = torch.ones_like(batch["terminated"])
    batch["not_terminated"] = torch.zeros_like(batch["not_terminated"])

    target = algo.compute_targets(batch)
    assert torch.allclose(target, batch["reward"])


def test_sac_target_bootstraps_on_truncation():
    """超时截断的转移必须带上后继价值——这是 terminated 与 truncated 分开的理由。"""
    algo, batch = sac_setup()
    batch["truncated"] = torch.ones_like(batch["truncated"])
    batch["not_terminated"] = torch.ones_like(batch["not_terminated"])

    target = algo.compute_targets(batch)
    assert not torch.allclose(target, batch["reward"])


def test_sac_update_produces_finite_metrics():
    algo, batch = sac_setup()
    for _ in range(3):
        metrics = algo.update(batch)
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["sac/alpha"] > 0.0


def test_sac_target_network_tracks_critic_without_copying():
    """软更新让目标网络缓慢跟随 critic，而不是一步到位。"""
    algo, batch = sac_setup()
    for _ in range(5):
        algo.update(batch)

    target_param = next(algo.critic_target.parameters())
    param = next(algo.critic.parameters())
    assert not torch.allclose(target_param, param), "tau 太大会退化成硬更新"

    before = target_param.detach().clone()
    algo._soft_update_target()
    moved = (target_param - before).abs().mean()
    gap = (param - before).abs().mean()
    assert 0 < moved.item() < gap.item()


def test_sac_actor_updates_less_often_than_critic():
    """策略延迟更新：critic 每动 policy_frequency 次，actor 才动一次。

    具体节奏是"更新计数整除 policy_frequency 时才动"，所以在一个周期里
    actor 恰好只被更新一次。策略跑在没估准的价值上会被带偏，而这个偏差又
    会喂回价值，来回几轮就发散了。
    """
    algo, batch = sac_setup()
    algo.cfg.policy_frequency = 4

    changed_at = []
    for step in range(9):
        before = [p.detach().clone() for p in algo.actor.parameters()]
        algo.update(batch)
        if any(
            not torch.equal(old, new)
            for old, new in zip(before, algo.actor.parameters(), strict=True)
        ):
            changed_at.append(step)

    assert changed_at == [0, 4, 8]


@pytest.mark.parametrize(
    ("log_prob", "should_grow"),
    [
        (5.0, True),  # 分布很尖，熵远低于目标熵 → 温度该涨，鼓励探索
        (-10.0, False),  # 分布很平，熵远高于目标熵 → 温度该降，开始利用
    ],
    ids=["low_entropy", "high_entropy"],
)
def test_sac_alpha_moves_toward_target_entropy(log_prob, should_grow):
    """自动温度调节的方向：熵不够就加温度，熵过剩就减温度。

    推导一下括号里的符号。样本的熵用 -log π 估计，目标熵是 -action_dim，
    于是括号 = log π + target_entropy = -H + action_dim。熵低于目标时它为正，
    损失 -(log α · 括号) 对 log α 的梯度为负，梯度下降就把 log α 推大。
    """
    algo, _ = sac_setup()
    alpha_before = algo.alpha.item()

    algo._update_alpha(torch.full((32,), log_prob))
    if should_grow:
        assert algo.alpha.item() > alpha_before
    else:
        assert algo.alpha.item() < alpha_before


# ---------------------------------------------------------------------------
# 回放池
# ---------------------------------------------------------------------------


def test_replay_buffer_samples_batch_with_bootstrap_mask():
    buffer = ReplayBuffer(capacity=16, obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=0)
    for i in range(8):
        buffer.add(
            obs=np.full(OBS_DIM, i, dtype=np.float32),
            action=np.zeros(ACTION_DIM, dtype=np.float32),
            reward=float(i),
            next_obs=np.full(OBS_DIM, i + 1, dtype=np.float32),
            terminated=(i % 2 == 0),
            truncated=(i % 3 == 0),
        )

    batch = buffer.sample(4)
    assert batch["obs"].shape == (4, OBS_DIM)
    assert batch["action"].shape == (4, ACTION_DIM)
    # 自举掩码必须只看 terminated：截断的样本仍然要自举
    for i in range(4):
        assert batch["not_terminated"][i].item() == float(not batch["terminated"][i].item())


def test_replay_buffer_overwrites_oldest_when_full():
    """写满后覆盖最旧的，池里始终是最新的 capacity 条。

    这里查 state_dict 而不是 sample：sample 是有放回随机抽取，取出的内容
    不保证是全集（可能重复、可能漏掉），用它断言"池里到底存了什么"不成立。
    """
    buffer = ReplayBuffer(capacity=4, obs_dim=2, action_dim=1, seed=0)
    for i in range(6):
        buffer.add(
            obs=np.full(2, i, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            reward=float(i),
            next_obs=np.zeros(2, dtype=np.float32),
            terminated=False,
            truncated=False,
        )
    assert len(buffer) == 4
    assert buffer.full
    # 按时间顺序取出，最旧的 0、1 已经被 4、5 覆盖
    assert buffer.state_dict()["rewards"].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_replay_buffer_batch_add_matches_single_adds():
    """批量写入与逐条写入必须落成同一份数据。"""
    single = ReplayBuffer(capacity=5, obs_dim=2, action_dim=1, seed=0)
    batched = ReplayBuffer(capacity=5, obs_dim=2, action_dim=1, seed=0)

    obs = np.arange(6, dtype=np.float32).reshape(3, 2)
    actions = np.arange(3, dtype=np.float32).reshape(3, 1)
    rewards = [1.0, 2.0, 3.0]
    next_obs = obs + 10
    terminated = [False, True, False]
    truncated = [False, False, True]

    for i in range(3):
        single.add(obs[i], actions[i], rewards[i], next_obs[i], terminated[i], truncated[i])
    batched.add_batch(obs[:2], actions[:2], rewards[:2], next_obs[:2], terminated[:2],
                      truncated[:2])
    batched.add_batch(obs[2:], actions[2:], rewards[2:], next_obs[2:], terminated[2:],
                      truncated[2:])

    for key in ("obs", "actions", "rewards", "terminated", "truncated"):
        assert np.array_equal(single.state_dict()[key], batched.state_dict()[key])


def test_replay_buffer_batch_add_wraps_around_capacity():
    buffer = ReplayBuffer(capacity=4, obs_dim=2, action_dim=1, seed=0)
    obs = np.arange(8, dtype=np.float32).reshape(4, 2)
    buffer.add_batch(obs[:3], np.zeros((3, 1), np.float32), [0.0, 1.0, 2.0],
                     obs[:3], [False] * 3, [False] * 3)
    buffer.add_batch(obs[3:], np.zeros((1, 1), np.float32), [3.0], obs[3:], [False], [False])

    assert len(buffer) == 4
    assert buffer.state_dict()["rewards"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_replay_buffer_state_dict_roundtrip():
    buffer = ReplayBuffer(capacity=8, obs_dim=2, action_dim=1, seed=0)
    for i in range(5):
        buffer.add(np.full(2, i, np.float32), np.zeros(1, np.float32), float(i),
                   np.full(2, i + 1, np.float32), i == 0, False)

    restored = ReplayBuffer(capacity=8, obs_dim=2, action_dim=1, seed=1)
    restored.load_state_dict(buffer.state_dict())

    assert len(restored) == len(buffer)
    for key in ("obs", "actions", "rewards", "terminated", "truncated"):
        assert np.array_equal(restored.state_dict()[key], buffer.state_dict()[key])


# ---------------------------------------------------------------------------
# 归一化统计量的存取
# ---------------------------------------------------------------------------


def test_normalizer_buffers_survive_state_dict_roundtrip():
    """统计量必须在 checkpoint 里。丢了它，续训时观测尺度突变，价值估计崩。

    这条性质是靠"注册成 buffer"结构上保证的，不是靠记得手写存取代码。
    """
    policy = ActorCritic(OBS_DIM, ACTION_DIM, critic_obs_dim=CRITIC_OBS_DIM)
    policy.update_normalizer(torch.randn(512, OBS_DIM) * 4.0 + 2.0)
    policy.update_critic_normalizer(torch.randn(512, CRITIC_OBS_DIM) * 2.0 - 1.0)

    state = policy.state_dict()
    assert {"obs_mean", "obs_std", "critic_obs_mean", "critic_obs_std"} <= set(state)

    restored = ActorCritic(OBS_DIM, ACTION_DIM, critic_obs_dim=CRITIC_OBS_DIM)
    restored.load_state_dict(state)

    assert torch.allclose(restored.obs_mean, policy.obs_mean)
    assert torch.allclose(restored.critic_obs_std, policy.critic_obs_std)
    obs = torch.randn(4, OBS_DIM)
    assert torch.allclose(restored.normalize_obs(obs), policy.normalize_obs(obs))


def test_obs_normalization_actually_changes_the_input():
    """归一化统计量收敛后，观测的各维被拉回零均值、单位方差。

    统计量是滑动平均，单次调用几乎不动，所以这里跑到收敛再断言——这也是
    训练里的实际情形：几万步之后统计量才真正贴合观测分布。
    """
    policy = ActorCritic(OBS_DIM, ACTION_DIM, critic_obs_dim=CRITIC_OBS_DIM)
    # 尺度取小一点，让数据整体落在 obs_clip 之内——裁剪会削掉尾部、压低方差，
    # 那是另一条性质，不该混进这条断言里。
    obs = torch.randn(1024, OBS_DIM) * 1.5 + 0.5
    assert torch.allclose(policy.obs_std, torch.ones(OBS_DIM))

    for _ in range(600):
        policy.update_normalizer(obs)

    assert not torch.allclose(policy.obs_std, torch.ones(OBS_DIM))
    normalized = policy.normalize_obs(obs)
    assert normalized.mean().abs().max().item() < 0.05
    assert normalized.std().item() == pytest.approx(1.0, abs=0.05)


def test_normalize_obs_clips_before_standardizing():
    """先夹再归一化。顺序反过来的话，一个异常大的观测会被归一化成很大的
    数值送进网络，第一层的激活直接饱和。"""
    policy = ActorCritic(OBS_DIM, ACTION_DIM, critic_obs_dim=CRITIC_OBS_DIM)
    huge = torch.full((1, OBS_DIM), 1e6)
    normalized = policy.normalize_obs(huge)

    assert normalized.abs().max().item() <= policy.obs_clip.item() + 1e-3
    assert torch.isfinite(normalized).all()


def test_normalizer_ignores_single_sample_batches():
    """只有一条观测时不更新。单条样本的"标准差"是 0，会把统计量毒化。"""
    policy = ActorCritic(OBS_DIM, ACTION_DIM, critic_obs_dim=CRITIC_OBS_DIM)
    policy.update_normalizer(torch.randn(1, OBS_DIM) * 100.0)
    assert torch.allclose(policy.obs_mean, torch.zeros(OBS_DIM))
    assert torch.allclose(policy.obs_std, torch.ones(OBS_DIM))


# ---------------------------------------------------------------------------
# 集成：在 toy 环境上真跑
# ---------------------------------------------------------------------------


def toy_envs(num_envs: int) -> list[ToyVelocityEnv]:
    return [ToyVelocityEnv(seed=i) for i in range(num_envs)]


def test_ppo_trains_on_toy_env_without_nan():
    """短训几十轮，指标必须全部是有限值，动作必须始终在契约范围内。

    这条用例不检查"学得好不好"——几十轮还看不出趋势，硬断言学习效果只会
    做出一个随机失败的测试。它检查的是整条链路上没有 inf/nan 泄漏，以及
    策略输出始终满足动作契约。
    """
    envs = toy_envs(2)
    policy = ActorCritic(envs[0].obs_dim, envs[0].action_dim,
                         critic_obs_dim=envs[0].critic_obs_dim)
    cfg = PPOConfig(num_steps_per_env=32, num_learning_epochs=3, num_mini_batches=2)
    trainer = PPOTrainer(envs, policy, cfg, seed=0, verbose=False)

    history = trainer.train(1500)

    assert history, "一轮都没跑"
    assert trainer.num_timesteps >= 1500
    for record in history:
        for key, value in record.metrics.items():
            assert math.isfinite(value), f"{key} 不是有限值：{value}"

    action = policy(torch.randn(64, envs[0].obs_dim) * 3.0)
    assert action.abs().max().item() < 1.0
    assert torch.isfinite(torch.tensor(trainer.evaluate(2)["eval/return_mean"]))


def test_sac_trains_on_toy_env_without_nan():
    """SAC 的短训集成测试。网络刻意开小，几十秒内跑完。"""
    envs = toy_envs(2)
    actor = SACActor(envs[0].obs_dim, envs[0].action_dim, hidden_dims=(32, 32))
    cfg = SACConfig(buffer_size=4000, batch_size=32, learning_starts=100, lr=1e-3)
    trainer = SACTrainer(envs, actor, cfg, seed=0, steps_per_iteration=250, verbose=False)

    history = trainer.train(750)

    assert trainer.num_timesteps >= 750
    for record in history:
        for key, value in record.metrics.items():
            assert math.isfinite(value), f"{key} 不是有限值：{value}"

    assert len(trainer.buffer) > 0
    action = actor(torch.randn(64, envs[0].obs_dim) * 3.0)
    assert action.abs().max().item() < 1.0
    assert math.isfinite(trainer.evaluate(2)["eval/return_mean"])


def test_sac_warmup_uses_random_actions():
    """学习开始前动作是随机的，不是策略给出的近似常量。"""
    envs = toy_envs(1)
    actor = SACActor(envs[0].obs_dim, envs[0].action_dim, hidden_dims=(16, 16))
    cfg = SACConfig(buffer_size=100, batch_size=8, learning_starts=10_000)
    trainer = SACTrainer(envs, actor, cfg, steps_per_iteration=50, verbose=False)
    trainer._prepare_run_dir()

    actions = np.stack([trainer._select_actions(torch.zeros(1, envs[0].obs_dim))
                        for _ in range(20)])
    assert actions.std() > 0.1, "预热阶段应该是均匀随机动作"
    assert np.abs(actions).max() <= 1.0


@pytest.mark.parametrize("algo_name", ["ppo", "sac"])
def test_trainer_checkpoint_roundtrip_reproduces_evaluation(algo_name, tmp_path):
    """存取往返后，同样的评估种子必须给出同样的回报。

    这条用例把"训练状态存全了"变成可验证的：漏存归一化统计量、漏存优化器
    或者漏存 Q 网络，评估结果都会对不上。
    """
    envs = toy_envs(1)
    env = envs[0]
    if algo_name == "ppo":
        policy = ActorCritic(env.obs_dim, env.action_dim, critic_obs_dim=env.critic_obs_dim)
        cfg = PPOConfig(num_steps_per_env=16, num_learning_epochs=1, num_mini_batches=1)
        trainer = PPOTrainer(envs, policy, cfg, run_dir=tmp_path, seed=0, verbose=False)
    else:
        policy = SACActor(env.obs_dim, env.action_dim, hidden_dims=(16, 16))
        cfg = SACConfig(buffer_size=500, batch_size=16, learning_starts=50)
        trainer = SACTrainer(envs, policy, cfg, run_dir=tmp_path, seed=0,
                             steps_per_iteration=100, verbose=False)

    trainer.train(200)
    path = trainer.save(tmp_path / "ckpt.pt")
    before = trainer.evaluate(3)

    if algo_name == "ppo":
        fresh = ActorCritic(env.obs_dim, env.action_dim, critic_obs_dim=env.critic_obs_dim)
        loaded = PPOTrainer(toy_envs(1), fresh, cfg, run_dir=tmp_path, seed=0, verbose=False)
    else:
        fresh = SACActor(env.obs_dim, env.action_dim, hidden_dims=(16, 16))
        loaded = SACTrainer(toy_envs(1), fresh, cfg, run_dir=tmp_path, seed=0,
                            steps_per_iteration=100, verbose=False)
    loaded.load(path)

    assert loaded.num_timesteps == trainer.num_timesteps
    assert loaded.evaluate(3) == pytest.approx(before)


def test_make_trainer_dispatches_by_name():
    envs = toy_envs(1)
    env = envs[0]
    ppo = make_trainer(
        envs,
        ActorCritic(env.obs_dim, env.action_dim, critic_obs_dim=env.critic_obs_dim),
        algo_name="ppo",
    )
    sac = make_trainer(
        envs, SACActor(env.obs_dim, env.action_dim, hidden_dims=(16, 16)), algo_name="sac"
    )
    assert isinstance(ppo, PPOTrainer)
    assert isinstance(sac, SACTrainer)

    with pytest.raises(ValueError, match="未知算法"):
        make_trainer(envs, ppo.policy, algo_name="td3")


def test_shared_trunk_requires_matching_obs_dims():
    """共享躯干时 actor 与 critic 输入必须同维——这是结构限制，不是参数问题。"""
    with pytest.raises(ValueError, match="共享躯干"):
        ActorCritic(OBS_DIM, ACTION_DIM, critic_obs_dim=CRITIC_OBS_DIM, separate_trunks=False)


def test_shared_trunk_actor_and_critic_share_parameters():
    policy = ActorCritic(OBS_DIM, ACTION_DIM, separate_trunks=False)
    actor_ids = {id(p) for p in policy.actor_parameters()}
    critic_ids = {id(p) for p in policy.critic_parameters()}
    trunk_ids = {id(p) for p in policy.trunk.parameters()}

    assert trunk_ids <= actor_ids
    assert trunk_ids.isdisjoint(critic_ids)
    assert policy(torch.randn(8, OBS_DIM)).abs().max().item() < 1.0
    assert policy.value(torch.randn(8, OBS_DIM)).shape == (8,)


def test_actor_critic_matches_deployment_contract(env):
    """基类契约：forward 输出确定性动作，act 默认带随机性。"""
    policy = ActorCritic(env.obs_dim, env.action_dim, critic_obs_dim=env.critic_obs_dim)
    obs = torch.as_tensor(env.reset(seed=0)[0].policy).unsqueeze(0)

    assert torch.allclose(policy(obs), policy.act(obs, deterministic=True))
    assert not torch.allclose(policy(obs), policy.act(obs), atol=1e-6)
    assert policy.obs_dim == env.obs_dim
    assert policy.action_dim == env.action_dim
