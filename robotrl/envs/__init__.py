"""环境层：观测、奖励、终止、指令与地形。

对外推荐用注册表按名字构造环境：

    from robotrl.envs import make
    env = make("G1-Velocity")

这样调用方不必 import 具体环境类，配置里换个名字就换了一套形态加任务。
"""

from robotrl.envs.base_env import BaseEnv, Obs, StepResult
from robotrl.envs.mujoco_env import MujocoEnv
from robotrl.envs.registry import env_name, list_envs, make, register_env

# 导入即注册内置任务。toy 不依赖 MuJoCo，用来分钟级验证全链路。
from robotrl.envs import toy  # noqa: F401  isort:skip
from robotrl.envs import tasks  # noqa: F401  isort:skip

__all__ = [
    "BaseEnv",
    "MujocoEnv",
    "Obs",
    "StepResult",
    "env_name",
    "list_envs",
    "make",
    "register_env",
]
