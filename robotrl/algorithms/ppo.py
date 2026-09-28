"""PPO 算法本体：把一轮装满的 rollout 变成一次参数更新。

这个文件只做一件事——给定 RolloutStorage 里的数据，算出损失、更新参数。
采样循环、日志、存档都在 trainer.py 里，网络结构在 networks.py 里。分开写
的代价是多了一层调用，收益是这一百多行可以逐条对着论文核对，不必在采样
代码里翻找哪一行是损失。

下面按重要性列出实现的细节，每条都注明了为什么必须这么做——PPO 的坑几乎
全在细节里，漏掉任何一条都能训出"看起来在降但学不出东西"的曲线。

1. 优势归一化
   优势的绝对尺度随奖励量纲变化（速度跟踪的奖励在 [0, 1]，力矩惩罚可能是
   几十）。不归一化的话 clip_ratio=0.2 这个阈值在不同任务上含义完全不同：
   优势大的任务里所有比值都被裁掉，梯度恒为 0；优势小的任务里裁剪形同虚设。
   归一化把"0.2"变成一个可跨任务复用的数。

2. 观测归一化
   观测各维的量纲差着几个数量级（关节角 0.1 弧度级、速度 10 级、重力投影
   1 级）。不归一化时第一层的有效学习率被最大那一维支配，小量纲的维度等于
   没在学。统计量按滑动平均在线更新，因为观测分布会随策略漂移。

3. 价值损失裁剪
   价值网络拟合的是不断移动的目标，偶尔会一步迈太大，把后续所有自举都拖偏。
   裁剪把单次更新幅度限制在 old_value ± clip_ratio 内，并且取裁与不裁的
   较大损失——方向是"宁可更新不足，也不要更新过头"。

4. 双学习率
   价值函数的拟合难度显著高于策略：它的目标是自举出来的、每轮都在变，而策略
   的目标是相对稳定的优势加权。用同一个学习率时，迁就 critic 就会让 actor
   步子太小，迁就 actor 就会让 critic 发散。分开设。

5. 梯度裁剪
   优势归一化之后仍有少量样本会给出很大的比值，进而在某一层产生尖峰梯度。
   按全局范数裁剪，代价是这一步的方向略有偏差，收益是训练不会因为一个批次
   而崩掉。

6. 按实测 KL 自适应调学习率
   固定的学习率只在"策略变化幅度刚好"时才合适，而这个幅度随训练阶段变化：
   早期策略差、梯度大，需要小步；中期可以大步；接近收敛又得收小。按训练
   进度衰减是一种办法，但它需要预先知道总步数，而且对"这一轮步子迈大了"
   这种即时问题反应迟钝。用实测 KL 直接反馈调节，是在线的、任务无关的。

7. 正交初始化
   见 networks.py 的模块文档。

8. tanh 高斯策略的 log 概率修正
   动作经 tanh 压缩到 [-1, 1]，密度要乘雅可比。漏掉修正的后果是重要性比值
   系统性偏移，且偏移量随动作饱和程度变化——动作越靠近边界错得越多，恰恰
   是策略最需要被纠正的地方。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from robotrl.algorithms.networks import ActorCritic
from robotrl.algorithms.storage import RolloutStorage
from robotrl.configs.schema import PPOConfig

#: 学习率的上下限。一次异常的 KL 不该把学习率打到 0（再涨不回来）或飙到
#: 不稳定区，所以乘性调整两头都要有闸。
MIN_LR = 1e-5
MAX_LR = 1e-2


class PPO:
    """PPO 的更新器。

    不继承 Trainer：它只负责"拿数据更新参数"，不管数据从哪来。这样想换一种
    采样方式（比如并行环境换成子进程）时，这个文件一行都不用改。

    Attributes:
        policy: 被更新的 actor-critic。
        kl: 最近一次更新实测到的 KL，自适应学习率的输入。
    """

    def __init__(
        self,
        policy: ActorCritic,
        cfg: PPOConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.num_updates = 0

        # 双优化器：actor 组和 critic 组。两者的参数集合不重叠，所以各自
        # step 不会互相干扰。
        self.optimizer_actor = torch.optim.Adam(
            self.policy.actor_parameters(), lr=cfg.lr_actor, eps=1e-8
        )
        self.optimizer_critic = torch.optim.Adam(
            self.policy.critic_parameters(), lr=cfg.lr_critic, eps=1e-8
        )
        self.kl = 0.0

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        critic_obs: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """选动作。返回 (动作, log 概率, 价值)，三者都按 (batch,) 对齐。

        log 概率必须和动作同时取出来存进 rollout：更新时要用它算重要性比值，
        事后重算的话，只要策略参数变过一次，算出来的就是另一个分布的 log
        概率，比值恒等于 1，整个 PPO 退化成普通的策略梯度。
        """
        obs = obs.to(self.device)
        dist = self.policy.distribution(obs)
        if deterministic:
            action = dist.mode()
            log_prob = dist.log_prob(action)
        else:
            action, log_prob = dist.sample()
        return action, log_prob, self.policy.value(critic_obs.to(self.device))

    # ------------------------------------------------------------------
    # 更新
    # ------------------------------------------------------------------

    def update(self, storage: RolloutStorage) -> dict[str, float]:
        """用一轮 rollout 更新策略，返回本轮的诊断指标。

        流程：展平 -> 归一化优势 -> 轮流做 num_learning_epochs 遍 mini-batch
        -> 每个 epoch 结束按实测 KL 调一次学习率。
        """
        cfg = self.cfg
        data = storage.flatten()
        advantages = data["advantages"]
        if cfg.use_adv_norm:
            # 在整个 batch 上算均值方差，而不是每个 mini-batch 各自算：
            # 按 mini-batch 归一化会让同一份数据在不同 batch 里被除以不同的
            # 标准差，等价于给优势加了噪声。
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totals: dict[str, float] = {}
        kls: list[float] = []
        num_mini_batches = 0
        early_stop = False
        for _ in range(cfg.num_learning_epochs):
            for idx in storage.mini_batch_indices(cfg.num_mini_batches, 1):
                for key, value in self._update_mini_batch(data, advantages, idx).items():
                    totals[key] = totals.get(key, 0.0) + value
                num_mini_batches += 1

            # 每个 epoch 量一次 KL：策略偏离采样策略有多远。这正是 PPO 想控制
            # 的那个量，比"训了多少步"更能说明当前步子合不合适。
            self.kl = self._measure_kl(data)
            kls.append(self.kl)
            if self._adapt_learning_rate(self.kl):
                # 这一轮已经偏离太多，后面几个 epoch 只会在更偏的策略上继续跑。
                # 早停保住的是稳定性，不是时间。
                early_stop = True
                break

        metrics = {f"ppo/{k}": v / max(num_mini_batches, 1) for k, v in totals.items()}
        # num_learning_epochs 为 0 时一次梯度都没走，KL 就没有观测量——返回 0
        # 而不是让除零异常把整个训练打断。调参时把 epoch 调成 0 来看"不更新
        # 时的基线表现"是很常见的用法。
        metrics["ppo/kl"] = sum(kls) / len(kls) if kls else 0.0
        metrics["ppo/kl_early_stop"] = float(early_stop)
        metrics["ppo/lr_actor"] = self.optimizer_actor.param_groups[0]["lr"]
        metrics["ppo/lr_critic"] = self.optimizer_critic.param_groups[0]["lr"]
        self.num_updates += 1
        return metrics

    def _update_mini_batch(
        self,
        data: dict[str, Tensor],
        advantages: Tensor,
        idx: Tensor,
    ) -> dict[str, float]:
        cfg = self.cfg
        dist = self.policy.distribution(data["obs"][idx])
        log_prob = dist.log_prob(data["actions"][idx])
        old_log_prob = data["log_probs"][idx]
        # 用 log 空间相减再 exp，而不是直接算概率之比：概率可能小到 1e-20，
        # 两个小量相除的浮点误差远大于对数相减。
        ratio = torch.exp(log_prob - old_log_prob)
        adv = advantages[idx]

        surrogate = torch.min(
            ratio * adv,
            torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv,
        )
        actor_loss = -surrogate.mean()
        if cfg.entropy_coef != 0.0:
            # 减号：损失是往下走的，熵要往上走，所以带负号进损失。
            actor_loss = actor_loss - cfg.entropy_coef * dist.entropy().mean()

        value = self.policy.value(data["critic_obs"][idx])
        returns = data["returns"][idx]
        old_value = data["values"][idx]
        value_loss = (value - returns) ** 2
        if cfg.use_value_clip:
            clipped = old_value + torch.clamp(value - old_value, -cfg.clip_ratio, cfg.clip_ratio)
            value_loss = torch.max(value_loss, (clipped - returns) ** 2)
        value_loss = value_loss.mean()

        self.optimizer_actor.zero_grad(set_to_none=True)
        self.optimizer_critic.zero_grad(set_to_none=True)
        # 一次 backward 把两路的梯度都算出来。分两次 backward 也能得到同样的
        # 结果（梯度是累加的），但共享躯干时两路的梯度会在躯干上相加，分开
        # 调用就得小心别在中间清零。
        (actor_loss + cfg.value_coef * value_loss).backward()
        self._clip_and_step()

        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()
        return {
            "actor_loss": float(actor_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "clip_fraction": float(clip_fraction),
        }

    def _clip_and_step(self) -> None:
        """按参数组分别裁剪梯度范数，然后各自更新。

        两组分开裁而不是拼成一个大向量裁一次：价值损失是平方误差、可以到几十，
        策略损失是有界的重要性加权、通常在 1 以内，两者合起来的范数会被大的
        那一组主导，小的那一组等于没裁到。
        """
        nn.utils.clip_grad_norm_(self.policy.actor_parameters(), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self.policy.critic_parameters(), self.cfg.max_grad_norm)
        self.optimizer_actor.step()
        self.optimizer_critic.step()

    @torch.no_grad()
    def _measure_kl(self, data: dict[str, Tensor]) -> float:
        """整批数据上量一次新旧策略的 KL。

        用 k1 估计量 mean(log π_old - log π_new)，它是 KL 的一阶无偏估计，
        只需要一次前向。精确 KL 要按新旧两个分布逐样本求积分，代价大得多，
        而这个数只用来看趋势、调学习率，精度要求没那么高。
        """
        log_prob = self.policy.distribution(data["obs"]).log_prob(data["actions"])
        return float((data["log_probs"] - log_prob).mean())

    def _adapt_learning_rate(self, kl: float) -> bool:
        """按实测 KL 乘性调整学习率。返回是否应该提前结束本轮 epoch 循环。

        阈值取 desired_kl 的 1.5 和 2 倍，是 legged_gym 沿用下来的一组经验值：
        1.5 倍以内算正常波动，调一下就行；超过 2 倍说明这一步确实迈大了，除了
        降学习率还要立刻停下，因为剩下的 epoch 是在已经偏了的策略上继续跑。
        """
        if not self.cfg.use_lr_schedule:
            return False
        desired = self.cfg.desired_kl
        if kl > desired * 2.0:
            self._scale_lr(1.0 / 1.5)
            return True
        if kl > desired * 1.5:
            self._scale_lr(1.0 / 1.5)
        elif 0.0 < kl < desired / 1.5:
            self._scale_lr(1.5)
        return False

    def _scale_lr(self, factor: float) -> None:
        """两组学习率乘同一个因子。

        必须同乘：双学习率的比值（actor 慢、critic 快）是刻意的设计，只调其中
        一组会让这个比值漂移，最后又退化成"用一个学习率"。
        """
        for opt in (self.optimizer_actor, self.optimizer_critic):
            lr = opt.param_groups[0]["lr"] * factor
            opt.param_groups[0]["lr"] = min(max(lr, MIN_LR), MAX_LR)

    # ------------------------------------------------------------------
    # 存取
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
            "num_updates": self.num_updates,
            "kl": self.kl,
        }

    def load_state_dict(self, state: dict) -> None:
        self.optimizer_actor.load_state_dict(state["optimizer_actor"])
        self.optimizer_critic.load_state_dict(state["optimizer_critic"])
        self.num_updates = state.get("num_updates", 0)
        self.kl = state.get("kl", 0.0)
