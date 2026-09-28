"""终止条件的组合器：与奖励同一套具名机制。

终止条件不做加权（回合结束就是结束，没有"一半结束"），所以这里没有权重，
只有开关。但"具名 + 注册 + 不启用就不计算"的结构和奖励一致，好处也一样：
任务想要哪些失败判据一目了然，调参时能单独关掉某一项而不动代码。

一个典型用法是课程学习。刚起步时策略一定会摔，`illegal_contact` 一开，回合
长度只有几十步，价值函数还没学会就被截断。把膝、髋触地先关掉，等站稳了再开，
这条曲线会好看很多——而这一切只是改一个布尔量。

判定顺序上，`check` 一遇到 True 就返回：终止判定在热路径上，也因为它只是
布尔或运算，短路不损失信息。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotrl.envs.mujoco_env import MujocoEnv

#: 终止条件函数的签名：真值表示本回合失败，应当结束。
TerminationFn = Callable[["MujocoEnv"], bool]


_TERMINATION_REGISTRY: dict[str, TerminationFn] = {}


def register_termination(name: str) -> Callable[[TerminationFn], TerminationFn]:
    """把函数注册为具名终止条件。"""
    def decorator(fn: TerminationFn) -> TerminationFn:
        if name in _TERMINATION_REGISTRY and _TERMINATION_REGISTRY[name] is not fn:
            raise ValueError(f"终止条件 {name!r} 已被注册，不能覆盖")
        _TERMINATION_REGISTRY[name] = fn
        return fn

    return decorator


def get_termination(name: str) -> TerminationFn:
    if name not in _TERMINATION_REGISTRY:
        raise KeyError(f"未注册的终止条件 {name!r}，可用：{list_terminations()}")
    return _TERMINATION_REGISTRY[name]


def list_terminations() -> list[str]:
    return sorted(_TERMINATION_REGISTRY)


class TerminationManager:
    """按开关组合终止条件。

    Args:
        enabled: 启用哪些条件。映射形式用值表示开关，序列形式表示"列出的都开"。
            None 表示全部关闭——空集合在调试时很有用：想看策略能撑多久而不被
            提前截断，就把失败判据全关掉，只留超时。
        registry: 可注入的注册表，主要给测试用，方便塞入探针函数。
    """

    def __init__(
        self,
        enabled: Mapping[str, bool] | Iterable[str] | None = None,
        *,
        registry: Mapping[str, TerminationFn] | None = None,
    ) -> None:
        available = _TERMINATION_REGISTRY if registry is None else registry
        if enabled is None:
            names: list[str] = []
        elif isinstance(enabled, Mapping):
            names = [name for name, on in enabled.items() if on]
        else:
            names = list(enabled)

        terms: list[tuple[str, TerminationFn]] = []
        for name in sorted(set(names)):
            if name not in available:
                raise KeyError(f"未知终止条件 {name!r}，可用：{sorted(available)}")
            terms.append((name, available[name]))
        self.terms = tuple(terms)

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.terms)

    def check(self, env: MujocoEnv) -> bool:
        """任一条件为真即终止。any 自带短路，一遇到真就不再求值后面的条件。"""
        return any(fn(env) for _, fn in self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        return f"TerminationManager({list(self.active_names)})"
