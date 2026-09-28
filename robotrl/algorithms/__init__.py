"""算法层：从零实现的 PPO 与 SAC。

四个部件彼此正交，换算法只换其中两个：

    networks.py   网络结构（PPO 用 actor-critic，SAC 用双 Q + 策略）
    storage.py    经验存放（PPO 用 rollout，SAC 用回放池）
    ppo.py/sac.py 算法本体（损失与更新规则）
    trainer.py    训练循环（采样、日志、评估、存档）

这个划分的用处在于：想换成另一种同策略算法，只需新写一个算法文件；
想试试不同的网络结构，只需换 networks.py。两者互不牵扯。

具体到依赖方向：算法文件只依赖 networks 与 storage，不知道 trainer 的存在；
trainer 只依赖算法文件暴露的 act/update 两个方法，不知道损失是怎么算的。
于是"在测试里手工填一个 storage 然后调 update"这种用法是天然的，不需要
先把整个训练循环跑起来。
"""

from robotrl.algorithms.base import Policy, Trainer, TrainMetrics
from robotrl.algorithms.networks import (
    ActorCritic,
    SACActor,
    SACQNetwork,
    SquashedGaussian,
    TwinQNetwork,
)
from robotrl.algorithms.ppo import PPO
from robotrl.algorithms.sac import SAC
from robotrl.algorithms.storage import ReplayBuffer, RolloutStorage
from robotrl.algorithms.trainer import PPOTrainer, SACTrainer, make_trainer

__all__ = [
    "ActorCritic",
    "PPO",
    "PPOTrainer",
    "Policy",
    "ReplayBuffer",
    "RolloutStorage",
    "SAC",
    "SACActor",
    "SACQNetwork",
    "SACTrainer",
    "SquashedGaussian",
    "TrainMetrics",
    "Trainer",
    "TwinQNetwork",
    "make_trainer",
]
