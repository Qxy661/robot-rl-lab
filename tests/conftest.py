"""测试公共配置。

把仓库根加进 sys.path，这样即使没有 pip install -e . 也能跑测试——
贡献者克隆下来第一件事通常是跑测试，不该在这里被环境问题挡住。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
