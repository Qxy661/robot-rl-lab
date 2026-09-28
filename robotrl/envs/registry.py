"""环境注册表：按名字构造环境。

任务代码不 import 具体环境类，而是

    env = make("G1-Velocity")

名字由「形态-任务」拼成。这样从配置文件里读到 robot=g1、task=velocity
就能直接构造出环境，命令行换一个词就换了一套机器人加任务，不需要写
任何 if/else 分支——分支一多，加一种形态就要动所有分支，正是要避免的。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from robotrl.assets.spec import get_spec
from robotrl.envs.base_env import BaseEnv

_ENV_REGISTRY: dict[str, Callable[..., BaseEnv]] = {}


def register_env(name: str, factory: Callable[..., BaseEnv]) -> None:
    """注册环境工厂。名字重复直接报错，避免后注册的静默覆盖先注册的。"""
    if name in _ENV_REGISTRY and _ENV_REGISTRY[name] is not factory:
        raise ValueError(f"环境 {name!r} 已被注册，不能覆盖")
    _ENV_REGISTRY[name] = factory


def list_envs() -> list[str]:
    return sorted(_ENV_REGISTRY)


def env_name(robot: str, task: str) -> str:
    """把形态名和任务名拼成注册键，例如 ("g1", "velocity") → "G1-Velocity"。

    形态名用 `capitalize()` 而不是 `upper()`：`go2` 的正确写法是 `Go2`，
    `upper()` 会拼出 `GO2`——注册表的查找不区分大小写，所以这个错拼不会导致
    任何功能问题，但它会出现在日志、报错信息和文档里，而文档里写的是 `Go2`。
    这种"能跑但写错"的差异最有腐蚀性，宁可现在就对齐。
    """
    return f"{robot.capitalize()}-{task.capitalize()}"


def make(
    name: str,
    *,
    config: Any = None,
    **kwargs: Any,
) -> BaseEnv:
    """按名字构造环境。

    优先在注册表里找 `形态-任务` 这个键；找不到时只按任务名再找一次，
    这样"所有形态共用的任务"只需注册一遍。

    形态由名字的前半段解析，并作为 spec 传给工厂——因此注册的工厂不需要
    知道自己要被哪个机器人用，加形态时不用改任何已注册的工厂。

    Args:
        name: 形如 "G1-Velocity" 或 "toy" 的名字。
        config: 可选的 Config 对象，环境自行从中读取需要的小节。
        **kwargs: 直接传给环境构造函数的参数，优先级最高。
    """
    key = name.lower()
    robot, _, task = name.partition("-")

    if key in _ENV_REGISTRY:
        factory = _ENV_REGISTRY[key]
    elif task and task.lower() in _ENV_REGISTRY:
        factory = _ENV_REGISTRY[task.lower()]
    else:
        raise KeyError(f"未注册的环境 {name!r}，可用：{list_envs()}")

    if task and "spec" not in kwargs:
        kwargs["spec"] = get_spec(robot.lower())

    return factory(config=config, **kwargs)
