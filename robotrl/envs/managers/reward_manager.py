"""奖励项的组合器：具名注册、按权重开关。

奖励是最容易写乱的一块。任务一多，`_reward()` 就长成几十行，每一项都直接
`reward += self.cfg.scale_xxx * ...`，于是三件事同时失控：想知道"奖励由哪些项
组成"得读完全文；想关掉某一项得改代码；关掉之后它照算不误，白花算力。

这里的做法来自 legged_gym：每项奖励是一个具名函数，注册在注册表里；配置给
权重，权重为 0 的项在构造时就被剔除，连函数都不进调用列表。于是
`env.reward_manager.active_names` 一句话回答"训练时到底在奖励什么"，而调参
时改 YAML 就能开关任意一项，零代码改动、零多余计算。

权重来自两处，优先级明确：

1. `RewardConfig.scales` 里显式写了名字 → 用它（写 0 就是关闭）。
2. 没写 → 用任务给的默认权重。

这样任务的默认配置是"开箱能训"，而任何一项都能被配置单独覆盖。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from robotrl.configs.schema import RewardConfig

if TYPE_CHECKING:
    from robotrl.envs.mujoco_env import MujocoEnv

#: 奖励项函数的签名。参数是环境本身，需要什么状态自己取。
RewardFn = Callable[["MujocoEnv"], float]


@dataclass(frozen=True)
class RewardTerm:
    """一项启用中的奖励。scale 为 0 的项不会出现在这里。"""

    name: str
    scale: float
    fn: RewardFn


_REWARD_REGISTRY: dict[str, RewardFn] = {}


def register_reward(name: str) -> Callable[[RewardFn], RewardFn]:
    """把函数注册为具名奖励项。

    名字是配置里的键，也是日志里的键，因此不允许同名覆盖——静默覆盖会让
    "权重调了但没反应"这类问题查很久。
    """
    def decorator(fn: RewardFn) -> RewardFn:
        if name in _REWARD_REGISTRY and _REWARD_REGISTRY[name] is not fn:
            raise ValueError(f"奖励项 {name!r} 已被注册，不能覆盖")
        _REWARD_REGISTRY[name] = fn
        return fn

    return decorator


def get_reward(name: str) -> RewardFn:
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"未注册的奖励项 {name!r}，可用：{list_rewards()}")
    return _REWARD_REGISTRY[name]


def list_rewards() -> list[str]:
    return sorted(_REWARD_REGISTRY)


class RewardManager:
    """按权重叠加启用中的奖励项。

    Attributes:
        terms: 启用中的 (名字, 权重, 函数)，顺序按名字排序以保证数值可复现
            （浮点加法不满足结合律，累加顺序变了末位会差）。
    """

    def __init__(
        self,
        config: RewardConfig | None = None,
        *,
        defaults: Mapping[str, float] | None = None,
        registry: Mapping[str, RewardFn] | None = None,
    ) -> None:
        self.config = config or RewardConfig()
        self.defaults = dict(defaults or {})
        available = _REWARD_REGISTRY if registry is None else registry

        names = set(self.defaults) | set(self.config.scales)
        terms: list[RewardTerm] = []
        for name in sorted(names):
            scale = self._resolve_scale(name)
            if scale == 0.0:
                # 跳过发生在构造时，不是每次调用时判断——热路径上少一次判断
                # 不重要，重要的是行为上真的没调用这个函数。
                continue
            if name not in available:
                raise KeyError(f"未知奖励项 {name!r}，可用：{sorted(available)}")
            terms.append(RewardTerm(name=name, scale=scale, fn=available[name]))
        self.terms = tuple(terms)

    def _resolve_scale(self, name: str) -> float:
        if name in self.config.scales:
            return float(self.config.scales[name])
        return float(self.defaults.get(name, 0.0))

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.terms)

    def scale_of(self, name: str) -> float:
        for term in self.terms:
            if term.name == name:
                return term.scale
        return 0.0

    def compute(self, env: MujocoEnv) -> tuple[float, dict[str, float]]:
        """算出本步总奖励，并返回逐项的值。

        逐项值回传是为了日志：只看总奖励曲线，分不清是"走得更稳了"还是
        "姿态项在偷偷补偿"。分项值能让每一次调参都有依据。
        """
        total = 0.0
        breakdown: dict[str, float] = {}
        for term in self.terms:
            value = float(term.fn(env))
            total += term.scale * value
            breakdown[term.name] = value
        return total, breakdown

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        body = ", ".join(f"{t.name}*{t.scale:g}" for t in self.terms)
        return f"RewardManager({len(self.terms)} 项 | {body})"
