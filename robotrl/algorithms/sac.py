"""SAC 算法本体：从回放池里取一个批次，更新四个网络。

SAC 比 PPO 多出的东西几乎都是为了同一个目的——**在异策略下把自举做稳**。
PPO 的数据是当前策略刚采的，价值目标最多偏一点点；SAC 的数据可能是几万步
之前的老策略采的，而价值目标里又带着自己对后继状态的估计，偏差会自我放大。
下面这些细节都是对着这个问题来的。

1. tanh 变换后的 log 概率修正
   log π(a|s) = log μ(u|s) - log(1 - tanh²(u))。最后这项最容易漏，而它恰好
   是策略梯度里的熵项和 alpha 的目标值。漏掉的后果不是"差一点"，是熵的尺度
   整体错位——动作越靠近边界错得越多，而策略恰恰需要学会把动作推到边界
   （比如关节满力矩），于是整条学习曲线在需要大动作的任务上明显变差。

2. 双 Q 取 min
   自举目标里带 max/min 时，估计噪声会被单向挑出来：噪声为正的那次估计被
   选中，误差随每轮自举累积。两个独立初始化的 Q 网络误差不同步，取 min
   就近似削掉了高估的部分。

3. 目标网络软更新
   硬更新（每隔 N 步整体复制）会让目标值每 N 步跳一次，Q 网络刚追上又被打飞。
   软更新每次只挪一小步（tau=0.005），目标值的变化速度被限制住，自举才收敛。

4. 自动温度调节
   alpha 决定"探索换回报"的汇率。手动调的话，不同任务、甚至同一任务的不同
   阶段都需要不同的值——早期要探索、后期要收敛。把它也当成一个被优化的参数，
   目标熵定为 -action_dim，即每个动作维度上的平均熵约为 1 个 nat。

5. 策略延迟更新
   价值没估准之前更新策略，策略会朝着一个错误的方向跑，而错误的价值又会被
   新策略带得更偏。让 critic 先更新 policy_frequency 次再动 actor，是拿计算
   换稳定。

6. 观测归一化只有一份
   归一化统计量放在 actor 里（部署时只导出 actor，统计量必须跟着它走），
   Q 网络复用同一份。各存一份的话两者会各更各的，Q 看到的观测尺度与策略
   不一致，目标值会系统性偏移。
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor

from robotrl.algorithms.networks import SACActor, TwinQNetwork
from robotrl.configs.schema import SACConfig


class SAC:
    """SAC 的更新器。

    持有一个 actor、一对 Q、一对目标 Q、三个优化器（actor / critic / alpha）。
    与 PPO 一样不继承 Trainer：这里只回答"给定一个批次怎么更新"。

    Attributes:
        actor: 策略网络，也是最终要导出部署的那个。
        critic: 双 Q 网络。
        critic_target: critic 的滞后副本，只用于算目标值。
    """

    def __init__(
        self,
        actor: SACActor,
        critic: TwinQNetwork,
        cfg: SACConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # 目标网络用 deepcopy 而不是重新构造再 load_state_dict：两者的
        # 结构参数、buffer（LayerNorm 的统计量）都要一致，deepcopy 一次到位。
        self.critic_target = copy.deepcopy(self.critic)
        for param in self.critic_target.parameters():
            param.requires_grad_(False)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, eps=1e-8)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr, eps=1e-8)

        self.target_entropy = -cfg.target_entropy_ratio * self.actor.action_dim
        if cfg.auto_alpha:
            # 优化 log_alpha 而不是 alpha：alpha 必须为正，直接优化就得每步
            # 投影到正数区间，而 log 参数化让正性自动成立，梯度也无界。
            self.log_alpha = torch.tensor(math.log(cfg.alpha), device=self.device,
                                          requires_grad=True)
            self.optimizer_alpha: torch.optim.Adam | None = torch.optim.Adam(
                [self.log_alpha], lr=cfg.lr, eps=1e-8
            )
        else:
            self.log_alpha = None
            self.optimizer_alpha = None
            self._fixed_alpha = cfg.alpha

        self.num_updates = 0

    # ------------------------------------------------------------------

    @property
    def alpha(self) -> Tensor:
        """当前温度系数。关掉自动调节时就是一个常数张量。"""
        if self.log_alpha is None:
            return torch.tensor(self._fixed_alpha, device=self.device)
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: Tensor, *, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        """选动作。返回 (动作, log 概率)。随机模式下动作带重参数化噪声。"""
        dist = self.actor.distribution(obs.to(self.device))
        if deterministic:
            action = dist.mode()
            return action, dist.log_prob(action)
        return dist.sample()

    # ------------------------------------------------------------------

    def compute_targets(self, batch: dict[str, Tensor]) -> Tensor:
        """算 Q 的自举目标，形状 (batch,)。

            y = r + γ · (1 - terminated) · ( min Q̄(s', a') - α · log π(a'|s') )

        单独拆成一个方法是为了能直接测这个公式：目标值算错是 SAC 最隐蔽的
        一类 bug，它不会报错，只会让曲线慢慢变平。

        自举掩码用 terminated 而不是 done：超时截断的回合下面还有状态，它的
        价值必须算进来。用 done 的话，每一条超时样本都会让 Q 凭空少掉一截远期
        回报，而且超时越频繁偏差越大（回合上限设得越短越明显）。
        """
        next_obs_norm = self.actor.normalize_obs(batch["next_obs"])
        next_action, next_log_prob = self.actor.distribution(next_obs_norm).sample()
        # 目标值这一路必须停在梯度之外：Q 的目标里不该有对 Q 自身的梯度，
        # 否则就是在优化一个自己定义的目标，会发散。
        q_next = self.critic_target.min_q(next_obs_norm, next_action)
        return (
            batch["reward"]
            + self.cfg.gamma * batch["not_terminated"] * (q_next - self.alpha.detach() * next_log_prob)
        )

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """更新一轮。critic 每步都更，actor 与 alpha 按 policy_frequency 更。"""
        metrics = self._update_critic(batch)
        # 延迟更新：价值还没估准就动策略，策略会朝着错误方向跑，而错误的价值
        # 又被新策略带得更偏。
        if self.num_updates % self.cfg.policy_frequency == 0:
            metrics.update(self._update_actor(batch))
        self._soft_update_target()
        self.num_updates += 1
        return metrics

    def _update_critic(self, batch: dict[str, Tensor]) -> dict[str, float]:
        target = self.compute_targets(batch)
        obs_norm = self.actor.normalize_obs(batch["obs"])
        q1, q2 = self.critic(obs_norm, batch["action"])
        # 两个 Q 各自朝同一个目标回归。目标里已经取过 min，所以这是"教两个
        # 学生同一份标准答案"，它们之间的差异只剩初始化带来的那份。
        critic_loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(
            q2, target
        )

        self.optimizer_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optimizer_critic.step()
        return {
            "sac/critic_loss": float(critic_loss.detach()),
            "sac/q_mean": float(torch.min(q1, q2).mean().detach()),
        }

    def _update_actor(self, batch: dict[str, Tensor]) -> dict[str, float]:
        obs_norm = self.actor.normalize_obs(batch["obs"])
        # 重新采样动作（而不是复用 critic 那步的），策略的梯度才是在当前参数
        # 下算的。重参数化让梯度能穿过采样回到网络的 mean 与 log_std。
        action, log_prob = self.actor.distribution(obs_norm).sample()
        # 只用 q1 做策略梯度：目标里取 min 是为了压高估，但把 min 也放进策略
        # 梯度里会让策略倾向于去钻两个 Q 之间那个被低估的缝隙。
        q = self.critic.q1(obs_norm, action)
        # alpha 必须 detach：不 detach 的话这一项会把梯度灌进 log_alpha，
        # 而 log_alpha 有自己的优化器，等于用错误的梯度更新它。
        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.optimizer_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.optimizer_actor.step()

        metrics = {
            "sac/actor_loss": float(actor_loss.detach()),
            "sac/entropy": float(-log_prob.mean().detach()),
            "sac/alpha": float(self.alpha.detach()),
        }
        metrics.update(self._update_alpha(log_prob))
        return metrics

    def _update_alpha(self, log_prob: Tensor) -> dict[str, float]:
        """自动温度调节。

        损失是 -(log α · (log π + H_target))，只对 log α 求梯度。样本的熵用
        -log π 估计，H_target 取 -action_dim，于是括号 = -H + action_dim：
        熵低于目标时为正，梯度下降把 log α 推大，温度升高、探索增强；熵过剩
        时反过来。整个过程不需要人工判断"现在是该探索还是该利用"。

        目标熵取 -action_dim 的含义是每个动作维度各留约 1 个 nat 的不确定性。
        取 0 就成了"完全确定"，那是最容易陷进局部最优的设定；取得太大则策略
        永远收敛不了。
        """
        if self.optimizer_alpha is None or self.log_alpha is None:
            return {}
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.optimizer_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.optimizer_alpha.step()
        return {"sac/alpha_loss": float(alpha_loss.detach())}

    @torch.no_grad()
    def _soft_update_target(self) -> None:
        """目标网络朝 critic 挪一小步：θ̄ ← (1-τ)θ̄ + τθ。

        τ 取 0.005 意味着目标网络的"记忆"大约 200 步，比任何一次单独更新的
        影响都长得多。这个时间尺度要明显长于 critic 自身的收敛速度，目标才
        是"滞后但稳定"的；τ 太大就退化成硬更新。
        """
        tau = self.cfg.tau
        # strict=True：两边结构必须完全一致，长度不同就是构建出了问题，
        # 静默地少更新几层会让目标网络悄悄滞后到错误的参数上。
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters(), strict=True
        ):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
            "optimizer_alpha": (
                self.optimizer_alpha.state_dict() if self.optimizer_alpha else None
            ),
            "log_alpha": None if self.log_alpha is None else self.log_alpha.detach().cpu(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.optimizer_actor.load_state_dict(state["optimizer_actor"])
        self.optimizer_critic.load_state_dict(state["optimizer_critic"])
        if self.optimizer_alpha is not None and state.get("optimizer_alpha") is not None:
            self.optimizer_alpha.load_state_dict(state["optimizer_alpha"])
        if self.log_alpha is not None and state.get("log_alpha") is not None:
            self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.num_updates = state.get("num_updates", 0)
