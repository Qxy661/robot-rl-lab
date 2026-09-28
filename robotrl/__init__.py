"""robotrl —— 机器人强化学习全栈框架。

分层结构，每层只依赖下面一层：

    assets      物理层：MuJoCo 模型 + RobotSpec
    envs        环境层：观测 / 奖励 / 终止 + 地形域随机化
    algorithms  算法层：从零实现的 PPO / SAC
    deploy      部署层：ONNX 导出 → INT8 量化 → 延迟基准 → 仿真回测
    eval        评估层：固定种子的可复现基准

只有这个顶层包和 contracts 是轻量的——不 import torch 和 mujoco，
因此可以快速拿到版本文档、配置模式等内容而不必等重量级依赖加载。
真正要跑仿真时再 import 对应的子模块。
"""

from robotrl.assets.spec import Morphology, RobotSpec, get_spec, list_specs, register
from robotrl.configs.schema import Config
from robotrl.contracts import ObsContract, ObsSegment, make_obs_contract
from robotrl.utils.config import load_config

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Morphology",
    "ObsContract",
    "ObsSegment",
    "RobotSpec",
    "get_spec",
    "list_specs",
    "load_config",
    "make_obs_contract",
    "register",
    "__version__",
]
