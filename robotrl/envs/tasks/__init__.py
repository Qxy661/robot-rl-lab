"""任务层：在 MuJoCo 环境上定义一个具体的运动任务。

一个任务要做的事只有三件：决定指令怎么来、奖励怎么配、失败怎么判。物理、
观测组装、域随机化这些与环境有关的部分由 MujocoEnv 提供，任务不重复实现。

导入本包即注册内置任务，因此上层只需要

    from robotrl.envs import make
    env = make("Go2-Velocity")
"""

from robotrl.envs.tasks import velocity  # noqa: F401  isort:skip

__all__: list[str] = []
