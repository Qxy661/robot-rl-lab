"""算法层契约：Policy 与 Trainer。

算法层的对外承诺只有两件事：给一个观测能吐一个动作（Policy），给一个环境
能把它训出来（Trainer）。PPO 和 SAC 的内部结构差别很大——前者同策略、要
rollout 缓冲；后者异策略、要回放池和四个网络——但它们对外的样子必须一致，
否则部署层和评估层就得为每个算法写一套分支。

Policy 的 forward 语义
----------------------
forward 不是"训练时的前向"，而是**部署语义**：输入观测，输出归一化的确定性
动作。这个选择让三件事自然统一：

- 训练时 `act()` 负责采样（带随机性），forward 负责求值；
- 评估时直接 forward 就是确定性策略；
- 导出 ONNX 时不需要包 wrapper，`torch.onnx.export(policy, dummy_obs, path)`
  导出的就是能上板的东西。

观测归一化也放在 forward 里、统计量注册成 buffer，而不是导出时再拼一个
预处理节点。这样"导出的图自带归一化"是结构上保证的，不会有人漏掉。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from robotrl.utils.onnx_meta import encode, write_metadata


class Policy(nn.Module, ABC):
    """可部署策略的统一接口。

    子类实现 `_policy_forward`，输入是**已归一化**的观测，输出是 [-1, 1]
    区间的确定性动作。归一化和输出限幅由基类负责，子类不用操心——这样
    "自包含"和"动作有界"两条性质对所有算法一致成立。

    Attributes:
        obs_dim: policy 观测维度。
        action_dim: 动作维度，等于被控关节数。
    """

    def __init__(self, obs_dim: int, action_dim: int, *, obs_clip: float = 10.0) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 归一化统计量注册为 buffer，才会随 state_dict 一起存取、随 ONNX 一起导出。
        # obs_clip 防止策略早期输出极端值时，归一化把数值放大到爆炸。
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("obs_clip", torch.tensor(float(obs_clip)))

    # ------------------------------------------------------------------
    # 子类实现
    # ------------------------------------------------------------------

    @abstractmethod
    def _policy_forward(self, obs_normalized: Tensor) -> Tensor:
        """由归一化观测算出 [-1, 1] 的动作。子类必须保证值域。"""

    # ------------------------------------------------------------------
    # 训练 / 评估 / 部署三条路径
    # ------------------------------------------------------------------

    def normalize_obs(self, obs: Tensor) -> Tensor:
        """标准化观测。用 buffer 里的统计量，导出后这部分会变成图里的常量。"""
        clipped = torch.clamp(obs, -self.obs_clip, self.obs_clip)
        return (clipped - self.obs_mean) / (self.obs_std + 1e-8)

    def forward(self, obs: Tensor) -> Tensor:
        """确定性动作，值域 [-1, 1]。部署和 ONNX 导出走这条路。"""
        return self._policy_forward(self.normalize_obs(obs))

    def act(self, obs: Tensor, *, deterministic: bool = False) -> Tensor:
        """产生用于与环境交互的动作。

        默认策略是确定性的，因此基类实现直接退化为 forward。随机策略
        （PPO 的高斯、SAC 的 tanh 高斯）覆盖本方法做采样。把"求值"和
        "采样"分开的好处是评估阶段不需要额外的开关分支。
        """
        if deterministic:
            return self.forward(obs)
        return self.forward(obs)

    # ------------------------------------------------------------------
    # 归一化统计量的在线估计
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_normalizer(self, obs: np.ndarray | Tensor, *, momentum: float = 0.99) -> None:
        """用一批观测的统计量滑动更新归一化参数。

        用滑动平均而不是一次性统计全量数据，是因为观测分布会随训练漂移
        （策略变强后走的地方不一样、地形不同），固定统计量会让后期观测
        落在归一化范围之外。
        """
        obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.reshape(-1, self.obs_dim)
        if obs.shape[0] < 2:
            return
        self.obs_mean.mul_(momentum).add_(obs.mean(dim=0), alpha=1 - momentum)
        self.obs_std.mul_(momentum).add_(obs.std(dim=0).clamp_min(1e-6), alpha=1 - momentum)

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        path: str | Path,
        *,
        opset: int = 17,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """导出为 ONNX。

        图的输入是 policy 观测 (batch, obs_dim)，输出是归一化动作
        (batch, action_dim)，值域 [-1, 1]。归一化统计量已经作为常量进图，
        所以推理端只需要喂裸观测，不需要预处理。

        默认姿态和 action_scale 写进图的 metadata 而不是塞进计算图：
        它们是"动作 → 关节角"这一层的参数，融进图里会把量化误差的来源
        搅在一起，反而不好做误差归因。端侧读 metadata 即可还原。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.eval()
        dummy = torch.zeros(1, self.obs_dim, dtype=torch.float32)

        # 显式指定 TorchScript 导出器（dynamo=False），不跟随版本默认值。
        # torch 2.6 起 dynamo 导出成为默认路径，但它依赖 onnxscript，没装就直接
        # 报错；torch 2.14 又移除了本函数的 metadata_props 参数——跟着默认值走
        # 意味着同一份代码在不同 torch 上时灵时不灵。本项目的策略是纯前馈 MLP，
        # 没有 dynamo 导出器才擅长的动态控制流，两条路径导出的图数值等价。
        torch.onnx.export(
            self,
            (dummy,),
            str(path),
            input_names=["obs"],
            output_names=["action"],
            opset_version=opset,
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            dynamo=False,
        )

        # metadata 用 onnx 单独补写，不指望导出器搬运。原因同上：不同 torch 版本
        # 对 metadata_props 的支持不一致，而补写只是一次文件读写，代价极小，
        # 换来的是"无论走哪个导出器、无论什么 torch 版本，metadata 一定在"。
        # 部署层 export_policy() 会再写一次，那是为了兜住量化后的情况，写重复是幂等的。
        meta: dict[str, Any] = {"action_dim": self.action_dim, "obs_dim": self.obs_dim}
        if metadata:
            meta.update(metadata)
        write_metadata(path, encode(meta))
        return path

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim}, params={sum(p.numel() for p in self.parameters())})"
        )


@dataclass
class TrainMetrics:
    """一次训练迭代的指标，写成 dict 便于直接落盘成 JSON。"""

    iteration: int = 0
    total_timesteps: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "total_timesteps": self.total_timesteps,
            **self.metrics,
        }


class Trainer(ABC):
    """把一个 Policy 在一个环境上训练出来。

    PPO 和 SAC 的差异全部收在 train() 内部。评估、保存、加载这些所有算法
    都要做的事放在基类，保证行为一致——尤其是 evaluate() 的确定性动作和
    train/eval 模式切换，这两个地方一旦各写各的，跑出来的数就不可比了。
    """

    def __init__(
        self,
        env: Any,
        policy: Policy,
        *,
        run_dir: str | Path = "runs",
        seed: int = 0,
    ) -> None:
        self.env = env
        self.policy = policy
        self.run_dir = Path(run_dir)
        self.seed = seed
        self.num_timesteps = 0

    @abstractmethod
    def train(self, total_timesteps: int) -> list[TrainMetrics]:
        """训练到指定总步数，返回每轮的指标记录。"""

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 10, *, seed: int = 12345) -> dict[str, float]:
        """评估策略。

        用独立的评估种子、固定的回合数、确定性动作。评估必须可复现，否则
        两条训练曲线之间的差异说不清是策略变好了还是评估时运气好。
        """
        was_training = self.policy.training
        self.policy.eval()

        returns, lengths = [], []
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=seed + ep)
            done, ep_return, ep_len = False, 0.0, 0
            while not done:
                action = self.policy(torch.as_tensor(obs.policy, dtype=torch.float32).unsqueeze(0))
                result = self.env.step(action.squeeze(0).numpy())
                obs, done = result.obs, (result.terminated or result.truncated)
                ep_return += result.reward
                ep_len += 1
            returns.append(ep_return)
            lengths.append(ep_len)

        if was_training:
            self.policy.train()

        returns_arr = np.asarray(returns, dtype=np.float64)
        return {
            "eval/return_mean": float(returns_arr.mean()),
            "eval/return_std": float(returns_arr.std()),
            "eval/episode_length_mean": float(np.mean(lengths)),
            "eval/episodes": float(n_episodes),
        }

    # ---- 存取 ----

    def save(self, path: str | Path) -> Path:
        """保存 checkpoint。训练状态（优化器、步数）交给子类补。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint(), path)
        return path

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.num_timesteps = ckpt.get("num_timesteps", 0)
        self._load_extra(ckpt)

    def _checkpoint(self) -> dict[str, Any]:
        return {"policy": self.policy.state_dict(), "num_timesteps": self.num_timesteps}

    def _load_extra(self, ckpt: dict[str, Any]) -> None:
        """子类恢复自己的额外状态（优化器、回放池等）。"""
