"""配置：dataclass 定义 + YAML 预设。

本包只负责"配置长什么样"，加载与覆写逻辑在 robotrl/utils/config.py。
分开是为了让 schema 保持零依赖——它要能被文档工具、配置校验脚本
独立引用，不该拖上 yaml 之外的东西。
"""

from robotrl.configs.schema import (
    Config,
    DeployConfig,
    EnvConfig,
    EventsConfig,
    ObsConfig,
    PPOConfig,
    RewardConfig,
    SACConfig,
    TerrainConfig,
    TrainConfig,
)

__all__ = [
    "Config",
    "DeployConfig",
    "EnvConfig",
    "EventsConfig",
    "ObsConfig",
    "PPOConfig",
    "RewardConfig",
    "SACConfig",
    "TerrainConfig",
    "TrainConfig",
]
