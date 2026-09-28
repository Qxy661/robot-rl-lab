"""经验存放：PPO 的 rollout 缓冲与 SAC 的回放池。

两个类的差别不是"一个能存得多一个存得少"，而是数据的使用方式不同：

    RolloutStorage  同策略。数据采完立刻就用来算一次梯度，然后整批丢掉。
                    所以容量是固定的（一轮 rollout），写入是顺序的，关键是
                    能在采完后一次性算出优势。
    ReplayBuffer    异策略。数据反复复用，越老的数据越可能与当前策略脱节。
                    所以是环形缓冲，写满就覆盖最旧的，采样靠随机索引。

把两者放在同一个文件里，是因为它们共享同一条纪律：**终止与截断必须分开存**。
价值自举时二者的处理相反——真终止的状态后继价值是 0，超时截断的仍要自举——
合并成一个 done 布尔量，超时那一步的回报会凭空掉一截。这个坑在两种算法里
是同一个，所以修复方式也放在一起写。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor


class RolloutStorage:
    """固定长度的 rollout 缓冲，按时间步顺序写入。

    形状约定：所有张量都是 (num_steps, num_envs, dim)。时间维在前，是因为
    优势计算沿时间反向递推，取 values[t] 比取 values[:, t] 的访存更顺，
    而 mini-batch 采样本来就要先展平，前后顺序无所谓。

    Attributes:
        num_steps: 一轮 rollout 的步数。
        num_envs: 并行环境数。
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_steps <= 0 or num_envs <= 0:
            raise ValueError(f"num_steps 与 num_envs 必须为正，得到 {num_steps}, {num_envs}")

        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._ptr = 0

        def zeros(*shape: int) -> Tensor:
            return torch.zeros(*shape, device=self.device)

        self.obs = zeros(num_steps, num_envs, obs_dim)
        self.critic_obs = zeros(num_steps, num_envs, critic_obs_dim)
        self.actions = zeros(num_steps, num_envs, action_dim)
        self.log_probs = zeros(num_steps, num_envs)
        self.rewards = zeros(num_steps, num_envs)
        self.values = zeros(num_steps, num_envs)
        # 两个布尔量分开存。done 用于"后面那个观测属不属于同一回合"，
        # terminated 用于"这一步要不要自举"。
        self.terminated = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=self.device)

        # compute_gae 的产物，形状展平后是 (num_steps * num_envs,)。
        self.advantages = zeros(num_steps, num_envs)
        self.returns = zeros(num_steps, num_envs)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add(
        self,
        obs: Tensor,
        critic_obs: Tensor,
        action: Tensor,
        log_prob: Tensor,
        reward: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        value: Tensor,
    ) -> None:
        """顺序写入一个时间步。标量字段的形状是 (num_envs,)。

        逐字段 copy_ 而不是赋值张量：缓冲区是预分配的，赋值会让旧张量失去
        引用但内存不清空，一轮轮下来显存占用只增不减。
        """
        if self._ptr >= self.num_steps:
            raise RuntimeError(
                f"RolloutStorage 已写满 {self.num_steps} 步，请先 compute_gae 再 reset"
            )
        t = self._ptr
        self.obs[t].copy_(torch.as_tensor(obs, device=self.device))
        self.critic_obs[t].copy_(torch.as_tensor(critic_obs, device=self.device))
        self.actions[t].copy_(torch.as_tensor(action, device=self.device))
        self.log_probs[t].copy_(torch.as_tensor(log_prob, device=self.device))
        self.rewards[t].copy_(torch.as_tensor(reward, device=self.device))
        self.values[t].copy_(torch.as_tensor(value, device=self.device))
        self.terminated[t].copy_(torch.as_tensor(terminated, dtype=torch.bool, device=self.device))
        self.truncated[t].copy_(torch.as_tensor(truncated, dtype=torch.bool, device=self.device))
        self._ptr += 1

    @property
    def full(self) -> bool:
        return self._ptr >= self.num_steps

    def reset(self) -> None:
        """复用同一块内存开始下一轮。优势与回报会一并清掉。

        不重新分配张量是刻意的：一轮 rollout 动辄几十万个数，每轮重新分配
        会让内存碎片化，长训练里这部分开销并不小。
        """
        self._ptr = 0
        self.advantages.zero_()
        self.returns.zero_()

    def __len__(self) -> int:
        return self._ptr * self.num_envs

    # ------------------------------------------------------------------
    # 优势估计
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_gae(
        self,
        last_values: Tensor,
        gamma: float,
        lam: float,
    ) -> tuple[Tensor, Tensor]:
        """广义优势估计。返回 (returns, advantages)，形状同 values。

        GAE 的递推是

            δ_t = r_t + γ V(s_{t+1}) - V(s_t)
            A_t = δ_t + γλ A_{t+1}

        其中 V(s_{t+1}) 能不能用，取决于回合在 t 步之后是否还在继续：

        - **真终止**（摔倒）：s_{t+1} 的价值确实是 0，δ 里那一项直接去掉；
        - **超时截断**：回合只是不采样了，s_{t+1} 仍在，价值要自举。

        Args:
            last_values: rollout 最后一个观测的价值估计，形状 (num_envs,)。
                只有回合没结束（或最后一步是超时）时才会被用上，真终止的
                环境上会被屏蔽掉。
        """
        if not self.full:
            raise RuntimeError(f"rollout 只写了 {self._ptr}/{self.num_steps} 步，先采满再算优势")
        last_values = torch.as_tensor(last_values, device=self.device).reshape(self.num_envs)

        last_gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_values
                next_gae = torch.zeros_like(last_gae)
            else:
                # 本步之后回合就结束了的话，values[t+1] 记的是新回合复位后的
                # 观测，它和 s_{t+1} 无关，用进来等于把下一回合的价值算到这一
                # 回合头上。mask 掉。
                continues = (~self.terminated[t] & ~self.truncated[t]).float()
                next_value = self.values[t + 1] * continues
                next_gae = self.advantages[t + 1] * continues

            # 真终止时后继价值不存在，直接归零——不是"近似为 0"，而是终止状态
            # 按定义没有后继，价值为 0。
            alive = (~self.terminated[t]).float()
            delta = self.rewards[t] + gamma * next_value * alive - self.values[t]
            last_gae = delta + gamma * lam * alive * next_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values
        return self.returns, self.advantages

    # ------------------------------------------------------------------
    # 取样
    # ------------------------------------------------------------------

    def flatten(self) -> dict[str, Tensor]:
        """把 (num_steps, num_envs, ...) 展平成 (num_steps * num_envs, ...)。

        展平只在这里做一次：mini-batch 循环里每轮都展平的话，同一份数据会
        被复制 num_learning_epochs 次。
        """

        def flat(x: Tensor) -> Tensor:
            return x.reshape(-1, *x.shape[2:])

        return {
            "obs": flat(self.obs),
            "critic_obs": flat(self.critic_obs),
            "actions": flat(self.actions),
            "log_probs": flat(self.log_probs),
            "values": flat(self.values),
            "returns": flat(self.returns),
            "advantages": flat(self.advantages),
        }

    def mini_batch_indices(
        self,
        num_mini_batches: int,
        num_epochs: int,
    ) -> Iterator[Tensor]:
        """按 epoch 打乱后切分，逐个产出索引。

        产出索引而不是切好的张量，是为了让调用方从已经展平的那份数据上取，
        避免每切一刀就复制一次数据。

        切分点用 round(i * n / k) 算，而不是先整除再丢余数：n 不整除 k 时
        余数会被均匀摊到各个 batch 上，没有样本被丢掉。丢掉尾部样本看着
        无害，但被丢的总是同一批（索引最大的那些，对应 rollout 末尾），
        长期下来末尾时间步的数据从来没参与过更新。
        """
        if num_mini_batches <= 0 or num_epochs <= 0:
            return
        n = self.num_steps * self.num_envs
        bounds = [round(i * n / num_mini_batches) for i in range(num_mini_batches + 1)]
        for _ in range(num_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(num_mini_batches):
                yield perm[bounds[i] : bounds[i + 1]]


class ReplayBuffer:
    """环形回放池，SAC 用。

    预先分配定长数组再循环覆盖，而不是用 list 追加：SAC 训练上百万步，每步
    追加一条经验意味着百万次对象分配和一次巨大的内存增长。定长数组的代价是
    容量必须提前定好，但回放池容量本来就该按显存预算定，不该随训练时长涨。

    Attributes:
        capacity: 容量（转移条数）。
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        *,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"回放池容量必须为正，得到 {capacity}")
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        def arrays(*shape: int) -> np.ndarray:
            return np.zeros((capacity, *shape), dtype=np.float32)

        self._obs = arrays(obs_dim)
        self._next_obs = arrays(obs_dim)
        self._actions = arrays(action_dim)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=bool)
        self._truncated = np.zeros(capacity, dtype=bool)

        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size >= self.capacity

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """写入一条转移。写满后从头部覆盖，池里永远是最新的 capacity 条。

        最新的数据比最老的数据更贴近当前策略，丢弃顺序按写入顺序来是对的。
        """
        i = self._ptr
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._terminated[i] = terminated
        self._truncated[i] = truncated
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: Sequence[float],
        next_obs: np.ndarray,
        terminated: Sequence[bool],
        truncated: Sequence[bool],
    ) -> None:
        """一次写入 num_envs 条转移。

        单独提供一个批量入口，是因为环形缓冲的指针在跨过容量边界时分成两段，
        逐条调用要走两遍 Python 循环和两次边界判断；批量写入只在断点处切一刀。
        """
        n = len(rewards)
        if n > self.capacity:
            raise ValueError(f"一次写入 {n} 条，超过回放池容量 {self.capacity}")
        if n == 0:
            return

        end = self._ptr + n
        if end <= self.capacity:
            span = slice(self._ptr, end)
            self._write(span, obs, actions, rewards, next_obs, terminated, truncated)
        else:
            # 跨过尾部：先写到容量末尾，剩下的绕回头部。
            head = self.capacity - self._ptr
            self._write(
                slice(self._ptr, self.capacity),
                obs[:head],
                actions[:head],
                rewards[:head],
                next_obs[:head],
                terminated[:head],
                truncated[:head],
            )
            tail = n - head
            self._write(
                slice(0, tail),
                obs[head:],
                actions[head:],
                rewards[head:],
                next_obs[head:],
                terminated[head:],
                truncated[head:],
            )
        self._ptr = end % self.capacity
        self._size = min(self._size + n, self.capacity)

    def _write(
        self,
        span: slice | np.ndarray,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: Sequence[float],
        next_obs: np.ndarray,
        terminated: Sequence[bool],
        truncated: Sequence[bool],
    ) -> None:
        self._obs[span] = obs
        self._next_obs[span] = next_obs
        self._actions[span] = actions
        self._rewards[span] = rewards
        self._terminated[span] = terminated
        self._truncated[span] = truncated

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        """随机采样一个批次。

        返回字典而不是元组：字段一多，位置参数调用几乎必然出现"顺序写反了
        却能跑"的错误，而且反了以后数值上完全看不出来。

        `not_terminated` 是自举掩码。超时截断在这里被**当作未终止**处理——
        回合结束只是采样停了，s' 的价值仍在，目标值要带上它。这正是不能把
        terminated 和 truncated 合并的原因。
        """
        if self._size < batch_size:
            raise ValueError(f"回放池只有 {self._size} 条，取不出 {batch_size} 条")
        idx = self._rng.integers(0, self._size, size=batch_size)

        def to_tensor(x: np.ndarray) -> Tensor:
            return torch.as_tensor(x, device=self.device)

        return {
            "obs": to_tensor(self._obs[idx]),
            "action": to_tensor(self._actions[idx]),
            "reward": to_tensor(self._rewards[idx]),
            "next_obs": to_tensor(self._next_obs[idx]),
            "terminated": to_tensor(self._terminated[idx]),
            "truncated": to_tensor(self._truncated[idx]),
            "not_terminated": to_tensor(1.0 - self._terminated[idx].astype(np.float32)),
        }

    # ------------------------------------------------------------------
    # 存取
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """回放池的完整状态。

        只保留已写入的前 _size 条：环形缓冲的物理布局依赖 _ptr，直接按物理
        顺序存会让读出来的人以为索引 0 是最老的数据。
        """
        if self._size < self.capacity:
            keep: Any = slice(0, self._size)
        else:
            keep = np.arange(self._ptr, self._ptr + self.capacity) % self.capacity
        return {
            "capacity": self.capacity,
            "size": self._size,
            "obs": self._obs[keep],
            "next_obs": self._next_obs[keep],
            "actions": self._actions[keep],
            "rewards": self._rewards[keep],
            "terminated": self._terminated[keep],
            "truncated": self._truncated[keep],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError(f"存档里有 {size} 条经验，超过当前容量 {self.capacity}")
        self._obs[:size] = state["obs"]
        self._next_obs[:size] = state["next_obs"]
        self._actions[:size] = state["actions"]
        self._rewards[:size] = state["rewards"]
        self._terminated[:size] = state["terminated"]
        self._truncated[:size] = state["truncated"]
        self._size = size
        self._ptr = size % self.capacity
