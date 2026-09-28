"""物理层：机器人形态定义与 MuJoCo 模型加载。

import 本包会触发三种内置形态的注册，因此上层代码只需

    from robotrl.assets import get_spec
    spec = get_spec("g1")

就能拿到形态，不必知道 g1 定义在哪个模块里。新增形态时只需要在本文件
补一行 import。
"""

from robotrl.assets.spec import Morphology, RobotSpec, get_spec, list_specs, register

# 导入即注册。这三个模块的作用就是把数据登记进注册表。
from robotrl.assets import g1, go2, h1  # noqa: F401  isort:skip

__all__ = ["Morphology", "RobotSpec", "get_spec", "list_specs", "register"]
