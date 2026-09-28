"""网络结构：PPO 的 actor-critic 与 SAC 的双 Q + 策略网络。

所有策略网络都继承 Policy，因此"输入观测、输出 [-1, 1] 动作"这条对外承诺
与具体算法无关。算法之间的差异收在两个方法里：

    distribution(obs)  训练时的动作分布，可采样、可求 log 概率
    value(critic_obs)  PPO 专有的价值估计，SAC 不用

于是 ppo.py 和 sac.py 里看不到任何 nn.Linear——换网络结构只动这个文件。

三条全局约定，理由都写在这里，子类不再重复
------------------------------------------

正交初始化
    每层权重若是独立同分布的随机数，前向传播的方差会随层数逐层放大或缩小，
    几十层下来梯度不是爆掉就是消失。正交初始化让每层输出的方差保持不变，
    深层网络才训得动。隐藏层的增益按激活函数取（ReLU 的方差是输入的一半，
    所以增益取 sqrt(2) 补回来），这是把"方差守恒"这件事算准，不是凑数。

输出层的增益压到 0.01
    隐藏层要保持方差，输出层反过来要压小。策略输出层直接决定动作，按常规
    尺度初始化的话训练第一步就会输出接近满幅的动作——在足式上等同于开机
    瞬间让所有关节猛甩，价值网络也会因为回报剧烈波动而估不准。压到 0.01
    意味着初始策略近似恒定输出，学习从"原地不动"附近开始。

动作标准差的上下界
    log_std 夹在 [LOG_STD_MIN, LOG_STD_MAX]。上界 2 对应 std≈7.4，已经远超
    [-1, 1] 的动作区间，再大就等于均匀乱动，优势信号被噪声淹没；下界 -5
    对应 std≈0.007，已经足够接近确定性。SAC 原文用 -20，对应的 std 是 2e-9，
    比 float32 在 log 概率上的舍入误差还小，夹在那里没有意义。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from robotrl.algorithms.base import Policy

#: 动作标准差对数的取值区间，PPO 与 SAC 共用。
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}

#: 各激活函数对应的正交初始化增益。nn.init.calculate_gain 不认识 elu，
#: 且它的默认值都是按"线性层后接该激活"推导的，这里显式写出来更好核对。
_GAINS: dict[str, float] = {
    "relu": math.sqrt(2.0),
    "elu": 1.0,
    "gelu": math.sqrt(2.0),
    "tanh": 5.0 / 3.0,
}


def _check_activation(name: str) -> str:
    if name not in _ACTIVATIONS:
        raise ValueError(f"未知激活函数 {name!r}，可用：{sorted(_ACTIVATIONS)}")
    return name


def make_activation(name: str) -> nn.Module:
    return _ACTIVATIONS[_check_activation(name)]()


def _linear(
    in_dim: int,
    out_dim: int,
    *,
    gain: float,
    layer_norm: bool = False,
) -> list[nn.Module]:
    """一层 Linear（可选 LayerNorm），权重正交初始化、偏置置零。

    偏置置零而不是照搬权重的初始化：正交初始化已经把激活值的方差管住了，
    再加一个随机偏置等于给每个神经元的均值也加噪声，反而把前面算好的尺度
    搅乱。
    """
    linear = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(linear.weight, gain=gain)
    nn.init.zeros_(linear.bias)
    layers: list[nn.Module] = [linear]
    if layer_norm:
        # LayerNorm 放在激活之前：先把这一层的输出尺度拉回标准正态，
        # 激活函数拿到的输入分布才稳定。
        layers.append(nn.LayerNorm(out_dim))
    return layers


def hidden_stack(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    activation: str = "elu",
    layer_norm: bool = False,
) -> tuple[nn.Sequential, int]:
    """只搭隐藏层。返回 (网络, 输出维度)。

    单独拆出来是给"共享躯干"用的：那种结构要先算出一段公共特征，再接两个
    互不相干的输出头，所以躯干和输出层得分开建。
    """
    layers: list[nn.Module] = []
    gain = _GAINS[_check_activation(activation)]
    last_dim = input_dim
    for hidden in hidden_dims:
        layers += _linear(last_dim, hidden, gain=gain, layer_norm=layer_norm)
        layers.append(make_activation(activation))
        last_dim = hidden
    return nn.Sequential(*layers), last_dim


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    activation: str = "elu",
    output_gain: float = 0.01,
    layer_norm: bool = False,
) -> nn.Sequential:
    """搭一个全连接网络，隐层用正交初始化。

    Args:
        hidden_dims: 各隐藏层宽度。空元组表示线性模型，便于写小规模测试。
        output_gain: 输出层增益。策略网络取 0.01（见模块文档），价值与 Q 网络
            取 1.0——它们的输出是标量估计，压小只会让初始价值估计恒为 0，
            白白浪费前几百步去把值域撑开。
        layer_norm: 每个隐层后是否加 LayerNorm。SAC 从回放池里取旧数据训练，
            观测分布随策略漂移，逐层重新归一能让它更稳。
    """
    trunk, last_dim = hidden_stack(
        input_dim, hidden_dims, activation=activation, layer_norm=layer_norm
    )
    return nn.Sequential(*trunk, *_linear(last_dim, output_dim, gain=output_gain))


# ---------------------------------------------------------------------------
# 动作分布
# ---------------------------------------------------------------------------


def _tanh_log_det(u: Tensor) -> Tensor:
    """log(1 - tanh²(u))，写成数值稳定的形式。

    直接算 torch.log(1 - torch.tanh(u) ** 2) 在 |u| 稍大时就会得到
    log(0) = -inf：tanh 饱和到 1.0 是 float32 精度内的必然结果，1 - 1 = 0。
    而 u 大恰恰是策略刚开始探索时的常态。

    用恒等式改写：

        1 - tanh²u = 4 e^{-2u} / (1 + e^{-2u})²

    取对数得 2 log 2 - 2u - 2 softplus(-2u)，softplus 自己就是稳定的。
    """
    return 2.0 * (math.log(2.0) - u - torch.nn.functional.softplus(-2.0 * u))


class SquashedGaussian:
    """对角高斯经 tanh 压缩到 (-1, 1) 的动作分布。

    tanh 压缩解决两件事：动作天然落在 [-1, 1]，不用在输出层补裁剪；动作分布
    被明确定义在有界区间上，部署时取 tanh(mean) 就是确定性策略，与基类的
    forward 语义严丝合缝。

    代价是 log 概率要加一项雅可比修正。压缩后的密度是

        p(a) = p(u) / |da/du|,   a = tanh(u),   da/du = 1 - tanh²(u)

    所以 log p(a) = log p(u) - log(1 - tanh²(u))。这一项最容易漏，漏了之后
    训练照样能跑，但熵的尺度是错的：SAC 的自动温度会朝错误的方向调 alpha，
    典型表现是策略过早收敛到确定性动作、探索消失。
    """

    def __init__(self, mean: Tensor, log_std: Tensor) -> None:
        self.mean = mean
        # 夹住之后广播到 mean 的形状，避免调用方还要自己对齐 batch 维。
        self.log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)

    @property
    def std(self) -> Tensor:
        return self.log_std.exp()

    def sample(self) -> tuple[Tensor, Tensor]:
        """重参数化采样。返回 (压缩后的动作, 它的 log 概率)。

        重参数化（而不是直接按分布采样）让梯度能穿过采样这一步回到 mean 和
        log_std，这是 SAC 策略梯度成立的前提。
        """
        u = self.mean + self.std * torch.randn_like(self.mean)
        return torch.tanh(u), self.log_prob_raw(u)

    def mode(self) -> Tensor:
        """分布的众数。压缩后取 tanh(mean)，值域严格在 (-1, 1) 内。"""
        return torch.tanh(self.mean)

    def log_prob(self, action: Tensor) -> Tensor:
        """求**已压缩**动作的对数概率，用于重要性比值和熵项。"""
        # 反解 u = atanh(a)。a 恰好等于 ±1 时 atanh 发散——采样的理论值取不到
        # ±1（tanh 只在无穷远饱和），但动作经 float32 舍入后可能正好落到边界上，
        # 也可能是外部传进来的。夹一个边距：边界处的密度略有偏差，换来的是
        # 整条更新不会被一个 inf/nan 污染，后者不可恢复。
        eps = torch.finfo(action.dtype).eps
        u = torch.atanh(action.clamp(-1.0 + eps, 1.0 - eps))
        return self.log_prob_raw(u)

    def log_prob_raw(self, u: Tensor) -> Tensor:
        """求未压缩变量 u 上的对数概率，含 tanh 修正。返回形状 (batch,)。

        采样路径直接拿 u 算，避免 atanh(tanh(u)) 这一趟来回丢精度。

        各维独立，联合对数概率是逐维相加。漏掉这一步的话返回的是 (batch,
        action_dim)，在 storage 里和 (batch,) 的字段相减会广播出一张 (batch,
        action_dim) 的比值表——不会报错，但每一维的比值都掺进了别维的信息，
        重要性采样整个是错的。
        """
        var = (2.0 * self.log_std).exp()
        normal = -0.5 * ((u - self.mean) ** 2 / var + 2.0 * self.log_std + math.log(2.0 * math.pi))
        return (normal - _tanh_log_det(u)).sum(dim=-1)

    def entropy(self) -> Tensor:
        """熵的估计，逐样本返回。

        压缩后的分布没有解析熵，这里用未压缩高斯的解析熵 + 压缩项在 mean 处
        的取值作为近似：E[log(1 - tanh²u)] 有界且对 std 单调，所以这个近似在
        "标准差该变大还是变小"这个方向上不会给出相反的信号——熵正则要的正是
        这个方向。
        """
        unsquashed = self.log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))
        return unsquashed.sum(dim=-1) + _tanh_log_det(self.mean).sum(dim=-1)


# ---------------------------------------------------------------------------
# PPO：actor-critic
# ---------------------------------------------------------------------------


class ActorCritic(Policy):
    """PPO 的 actor-critic。

    critic 用特权观测（真实机身速度、地形高度等），维度往往与 policy 观测
    不同，所以默认**不共享躯干**：共享躯干要求两者输入维度一致，一旦开了
    特权观测就共享不了，到那时再改结构等于把已经训好的权重作废。默认独立
    还有第二个好处：双学习率才有意义（见 ppo.py），共享躯干时躯干参数只能
    归到其中一组，另一组的"独立学习率"就名存实亡。

    actor 与 critic 各自持有一份观测归一化统计量。看似重复，但两者的输入
    分布本来就不同（critic 多了几维真值），共用一份只能按前缀对齐，多出的
    维度没有统计量可用。

    Attributes:
        critic_obs_dim: critic 输入维度，可以是特权观测的宽度。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        critic_obs_dim: int | None = None,
        actor_hidden: Sequence[int] = (256, 128),
        critic_hidden: Sequence[int] = (256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        separate_trunks: bool = True,
    ) -> None:
        super().__init__(obs_dim, action_dim)
        self.critic_obs_dim = critic_obs_dim if critic_obs_dim is not None else obs_dim
        self.separate_trunks = separate_trunks

        # critic 的归一化统计量同样注册成 buffer：checkpoint 里少了它，续训
        # 时 critic 看到的观测尺度会突变，价值估计直接崩一截。
        self.register_buffer("critic_obs_mean", torch.zeros(self.critic_obs_dim))
        self.register_buffer("critic_obs_std", torch.ones(self.critic_obs_dim))
        self.register_buffer("critic_obs_clip", torch.tensor(10.0))

        # log_std 是独立的可学习参数，不接在躯干上：动作噪声的大小与当前观测
        # 基本无关，让它当状态函数只会多一层拟合负担，还容易在少见观测上输出
        # 离谱的噪声值。
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_noise_std)))

        self.trunk: nn.Sequential | None = None
        self.actor_head: nn.Linear | None = None
        self.critic_head: nn.Linear | None = None
        self.actor_body: nn.Sequential | None = None
        self.critic_body: nn.Sequential | None = None

        if separate_trunks:
            self.actor_body = build_mlp(
                obs_dim, actor_hidden, action_dim, activation=activation, output_gain=0.01
            )
            self.critic_body = build_mlp(
                self.critic_obs_dim,
                critic_hidden,
                1,
                activation=activation,
                output_gain=1.0,
            )
        else:
            if self.critic_obs_dim != obs_dim:
                raise ValueError(
                    f"共享躯干要求 actor 与 critic 输入同维，得到 {obs_dim} 与 "
                    f"{self.critic_obs_dim}；critic 用特权观测时必须 separate_trunks=True"
                )
            self.trunk, hidden = hidden_stack(obs_dim, actor_hidden, activation=activation)
            self.actor_head = nn.Linear(hidden, action_dim)
            self.critic_head = nn.Linear(hidden, 1)
            for head, gain in ((self.actor_head, 0.01), (self.critic_head, 1.0)):
                nn.init.orthogonal_(head.weight, gain=gain)
                nn.init.zeros_(head.bias)

    # ---- 归一化 ----

    def normalize_critic_obs(self, obs: Tensor) -> Tensor:
        clipped = torch.clamp(obs, -self.critic_obs_clip, self.critic_obs_clip)
        return (clipped - self.critic_obs_mean) / (self.critic_obs_std + 1e-8)

    @torch.no_grad()
    def update_critic_normalizer(self, obs, *, momentum: float = 0.99) -> None:
        """滑动更新 critic 的归一化统计量。与基类同法，只是换一组 buffer。

        没有复用基类的 update_normalizer，是因为它把维度写死在 self.obs_dim 上，
        而 critic 的输入更宽。硬套会静默地把特权那几维统计错。
        """
        obs = torch.as_tensor(obs, dtype=torch.float32).reshape(-1, self.critic_obs_dim)
        if obs.shape[0] < 2:
            return
        self.critic_obs_mean.mul_(momentum).add_(obs.mean(dim=0), alpha=1 - momentum)
        self.critic_obs_std.mul_(momentum).add_(obs.std(dim=0).clamp_min(1e-6), alpha=1 - momentum)

    # ---- 前向 ----

    def _actor_mean(self, obs_normalized: Tensor) -> Tensor:
        if self.trunk is not None:
            return self.actor_head(self.trunk(obs_normalized))
        return self.actor_body(obs_normalized)

    def _policy_forward(self, obs_normalized: Tensor) -> Tensor:
        """部署语义：tanh 之后值域严格落在 (-1, 1)。"""
        return torch.tanh(self._actor_mean(obs_normalized))

    def value(self, critic_obs: Tensor) -> Tensor:
        """状态价值，返回形状 (batch,)。"""
        obs_normalized = self.normalize_critic_obs(critic_obs)
        if self.trunk is not None:
            return self.critic_head(self.trunk(obs_normalized)).squeeze(-1)
        return self.critic_body(obs_normalized).squeeze(-1)

    def distribution(self, obs: Tensor) -> SquashedGaussian:
        """训练时的动作分布。输入原始观测，归一化在内部完成。"""
        mean = self._actor_mean(self.normalize_obs(obs))
        return SquashedGaussian(mean, self.log_std)

    def act(self, obs: Tensor, *, deterministic: bool = False) -> Tensor:
        """覆盖基类：PPO 的探索来自高斯噪声，不是确定性输出。"""
        dist = self.distribution(obs)
        if deterministic:
            return dist.mode()
        action, _ = dist.sample()
        return action

    # ---- 参数分组 ----

    def actor_parameters(self) -> list[nn.Parameter]:
        """actor 优化器管的参数：actor 躯干（或共享躯干）+ 输出头 + log_std。

        共享躯干时把躯干划给 actor 组，理由不是"躯干更属于策略"，而是两组
        参数不能重叠——同一份参数出现在两个优化器里，每步会被更新两次，
        等效学习率翻倍，而且是静默发生的。
        """
        if self.trunk is None:
            return [self.log_std, *self.actor_body.parameters()]
        return [self.log_std, *self.trunk.parameters(), *self.actor_head.parameters()]

    def critic_parameters(self) -> list[nn.Parameter]:
        """critic 优化器管的参数。共享躯干时只剩输出头。"""
        if self.trunk is None:
            return list(self.critic_body.parameters())
        return list(self.critic_head.parameters())


# ---------------------------------------------------------------------------
# SAC：策略网络与双 Q
# ---------------------------------------------------------------------------


class SACActor(Policy):
    """SAC 的策略网络：输出 tanh 高斯的 mean 与 log_std。

    隐藏层默认 ReLU，对齐原论文与其主流实现，方便和数据对照；换成 ELU 也能
    跑，那属于调参不属于结构。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__(obs_dim, action_dim)
        self.action_dim = action_dim
        # 一次吐出 mean 与 log_std 再对半切，比接两个头省一次前向。
        self.body = build_mlp(
            obs_dim,
            hidden_dims,
            2 * action_dim,
            activation=activation,
            output_gain=0.01,
            layer_norm=use_layer_norm,
        )

    def _split(self, obs_normalized: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self.body(obs_normalized).split(self.action_dim, dim=-1)
        # log_std 不再单独注册参数，而是由网络输出：SAC 的噪声大小与状态相关
        # 是必要的——机械臂在接近目标时要收小噪声，在开阔区域可以放大。
        return mean, log_std

    def _policy_forward(self, obs_normalized: Tensor) -> Tensor:
        mean, _ = self._split(obs_normalized)
        return torch.tanh(mean)

    def distribution(self, obs: Tensor) -> SquashedGaussian:
        mean, log_std = self._split(self.normalize_obs(obs))
        return SquashedGaussian(mean, log_std)

    def act(self, obs: Tensor, *, deterministic: bool = False) -> Tensor:
        dist = self.distribution(obs)
        if deterministic:
            return dist.mode()
        action, _ = dist.sample()
        return action


class SACQNetwork(nn.Module):
    """单个动作价值网络 Q(s, a)。

    输入是归一化观测与动作的拼接。动作不做归一化也不能归一化：它已经是
    [-1, 1] 的归一化增量，本身就是标准尺度。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = build_mlp(
            obs_dim + action_dim,
            hidden_dims,
            1,
            activation=activation,
            # Q 的输出是回报估计，增益给 1.0：压小会让初始 Q 恒为 0，
            # 而 SAC 的 Q 值要跨越整条回报量程，从 0 涨起来要多花很多步。
            output_gain=1.0,
            layer_norm=use_layer_norm,
        )

    def forward(self, obs_normalized: Tensor, action: Tensor) -> Tensor:
        """返回 (batch,) 的 Q 值。观测必须是**已归一化**的。"""
        return self.net(torch.cat([obs_normalized, action], dim=-1)).squeeze(-1)


class TwinQNetwork(nn.Module):
    """双 Q：两个独立初始化的 Q 网络。

    取 min 是为了对付 Q 的高估。单 Q 的估计误差是单向的——目标里的 max 操作
    会把噪声挑出来当信号，误差随自举一轮轮放大。两个网络初始化不同、见过的
    批次也不同，它们的误差不会同时偏同一侧，取 min 就近似把高估那部分削掉。
    这也解释了为什么两个 Q 必须独立初始化：共用一个初始值的话它们的误差是
    相关的，min 就退化成单 Q 了。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        kwargs = {
            "hidden_dims": hidden_dims,
            "activation": activation,
            "use_layer_norm": use_layer_norm,
        }
        self.q1 = SACQNetwork(obs_dim, action_dim, **kwargs)
        self.q2 = SACQNetwork(obs_dim, action_dim, **kwargs)

    def forward(self, obs_normalized: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(obs_normalized, action), self.q2(obs_normalized, action)

    def min_q(self, obs_normalized: Tensor, action: Tensor) -> Tensor:
        q1, q2 = self.forward(obs_normalized, action)
        return torch.min(q1, q2)
