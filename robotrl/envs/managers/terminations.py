"""通用失败判据。

分四类，覆盖足式任务里绝大多数"这一回合已经没救了"的情形：

- 状态发散——数值上的失败，再跑下去只会产出垃圾样本。
- 摔到地上——机身高度掉到站姿的一半以下，或者躯干、髋、膝这些不该碰地的部位
  碰了地。
- 姿态翻掉——倾角超过阈值，此时"恢复"已经不在策略的能力范围内。

这些判据的阈值都放在环境实例上（`base_height_min`、`max_tilt`），因为同一个
形态在不同任务里的合理阈值不同：上下楼梯时轻微触地是正常的，平地上则说明摔了。
任务层改属性即可，不必重写判定函数。

注意这里**不**判定超时。超时是 truncated，与失败严格分开，理由见 base_env 的
StepResult 文档。所有判据都走"越界即失败"的单侧条件，没有临界带——临界带会
让同一个状态在两步之间反复横跳，回合长度出现不可解释的抖动。
"""

from __future__ import annotations

import numpy as np

from robotrl.envs.managers.termination_manager import register_termination


@register_termination("nan_state")
def nan_state(env) -> bool:
    """状态里出现 NaN 或 inf。

    仿真发散是数值问题不是策略问题，但放任下去 mj_step 会把垃圾值传播到整个
    模型，之后所有样本都不可信，所以宁可当作失败提前结束。
    """
    return not (np.all(np.isfinite(env.data.qpos)) and np.all(np.isfinite(env.data.qvel)))


@register_termination("base_height_low")
def base_height_low(env) -> bool:
    """机身高度低于站姿的一半。"""
    return env.base_height() < env.base_height_min


@register_termination("tilt_too_large")
def tilt_too_large(env) -> bool:
    """倾角超过阈值：-g_z < cos(阈值) 即判失败。

    g 是重力方向在机体系下的投影，直立时 g_z ≈ -1，倒下时接近 0。
    """
    return float(-env.projected_gravity()[2]) < float(np.cos(env.max_tilt))


@register_termination("illegal_contact")
def illegal_contact(env) -> bool:
    """非足端部位触地（躯干、髋、膝等）。判哪些部位算"非法"由环境按形态给出。"""
    return env.illegal_contact()
