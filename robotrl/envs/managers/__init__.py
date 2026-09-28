"""奖励项与终止条件的管理器。

对外只需要两个类：

    env.reward_manager.compute(env)        # 本步总奖励 + 逐项值
    env.termination_manager.check(env)     # 是否失败

以及两个注册装饰器，供任务层补充自己特有的项：

    @register_reward("my_term")
    def my_term(env) -> float: ...

模块导入时会把内置的通用项注册进注册表，因此任务代码只需要写奖励函数的
名字和权重。

    from robotrl.envs.managers import RewardManager, TerminationManager
"""

from robotrl.envs.managers.reward_manager import (
    RewardFn,
    RewardManager,
    RewardTerm,
    get_reward,
    list_rewards,
    register_reward,
)
from robotrl.envs.managers.termination_manager import (
    TerminationFn,
    TerminationManager,
    get_termination,
    list_terminations,
    register_termination,
)

# 导入即注册内置项。放在最后：上面两个模块定义注册表，这两个往表里填。
from robotrl.envs.managers import rewards, terminations  # noqa: F401  isort:skip

__all__ = [
    "RewardFn",
    "RewardManager",
    "RewardTerm",
    "TerminationFn",
    "TerminationManager",
    "get_reward",
    "get_termination",
    "list_rewards",
    "list_terminations",
    "register_reward",
    "register_termination",
]
