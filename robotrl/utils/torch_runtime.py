"""PyTorch 运行时设置。目前只做一件事：把线程数按网络规模调对。

为什么需要单独一个文件
----------------------
PyTorch 默认把 intra-op 线程数设成 CPU 核数。对小网络来说这是灾难性的：
16 核机器上，一个 64×64 的 MLP 做一次 backward 要走 258 ms；把线程数降到 1，
同样的 backward 只要 0.92 ms——相差 280 倍。

原因在于算子内部的并行是按张量切分的。参数量只有几万时，单个算子的计算量
远小于线程同步的开销，核越多、同步的代价越大。足式策略网络（256,128）在这个
尺度上同样适用，所以这不是"小玩具才有的问题"。

这个坑值得专门写一个文件，是因为它**不报错、不崩溃，只是慢**。训练慢十倍时，
第一反应往往是"MuJoCo 太慢"或者"CPU 不行"，很难想到是线程数设错了。

约定
----
显式传给 `configure_torch()` 的线程数优先；没传时读环境变量
``ROBOTRL_TORCH_THREADS``；都没有则用 `DEFAULT_NUM_THREADS`。
这样默认情况下训练是快的，而想调的人不需要改代码。
"""

from __future__ import annotations

import contextlib
import os

#: 默认线程数。理由见模块文档：这类规模的网络，单线程几乎总是更快。
DEFAULT_NUM_THREADS = 1
#: interop 线程控制算子之间的并行，同样受同步开销拖累，默认也压到 1。
DEFAULT_INTEROP_THREADS = 1

ENV_VAR = "ROBOTRL_TORCH_THREADS"
INTEROP_ENV_VAR = "ROBOTRL_TORCH_INTEROP_THREADS"

_configured: bool = False


def resolve_num_threads(requested: int | None = None) -> int:
    """按"显式参数 → 环境变量 → 默认值"的顺序决定线程数。

    单独抽出来是为了让脚本能在设置之前先把结果打印出来——训练日志里
    "用了几个线程"应该是一眼能看到的，否则线程数配错时无从察觉。
    """
    if requested is not None:
        if requested <= 0:
            raise ValueError(f"线程数必须为正，得到 {requested}")
        return requested

    raw = os.environ.get(ENV_VAR)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{ENV_VAR}={raw!r} 不是整数") from None
        if value <= 0:
            raise ValueError(f"{ENV_VAR} 必须为正，得到 {value}")
        return value

    return DEFAULT_NUM_THREADS


def resolve_interop_threads(requested: int | None = None) -> int:
    """interop 线程数的解析规则与 intra-op 一致。"""
    if requested is not None:
        if requested <= 0:
            raise ValueError(f"interop 线程数必须为正，得到 {requested}")
        return requested

    raw = os.environ.get(INTEROP_ENV_VAR)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{INTEROP_ENV_VAR}={raw!r} 不是整数") from None
        if value <= 0:
            raise ValueError(f"{INTEROP_ENV_VAR} 必须为正，得到 {value}")
        return value

    return DEFAULT_INTEROP_THREADS


def configure_torch(
    num_threads: int | None = None,
    *,
    interop_threads: int | None = None,
    force: bool = False,
) -> dict[str, int]:
    """设置 PyTorch 线程数，返回实际生效的值。

    Args:
        num_threads: intra-op 线程数，即单个算子内部的并行度。
        interop_threads: interop 线程数，即算子之间的并行度。
        force: 重复调用时是否重新设置。默认为 False——训练器构造时会调一次，
            而同一个进程里构造多个训练器是常见的（多种子基准就是），
            每次都重设会让上层的显式设置被覆盖掉。

    Returns:
        {"num_threads": ..., "interop_threads": ...}，便于调用方写进日志。

    Note:
        `torch.set_num_interop_threads` 一旦有并行工作跑过就不能再改，会抛
        RuntimeError。这里捕获它并保留原值，而不是让训练在启动阶段崩掉——
        线程数没设上是性能问题，不该升级成可用性问题。
    """
    global _configured

    import torch

    threads = resolve_num_threads(num_threads)
    interop = resolve_interop_threads(interop_threads)

    if _configured and not force:
        return {
            "num_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        }

    torch.set_num_threads(threads)
    # 已经跑过并行工作时 interop 线程数会被锁死，此时维持现状即可。
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(interop)

    effective = {
        "num_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }
    _configured = True
    return effective


def describe(cpu_count: int | None = None) -> str:
    """一行说明当前线程设置与机器核数的关系，用于训练日志开头。"""
    import torch

    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    threads = torch.get_num_threads()
    note = "（单线程，小网络下最快）" if threads == 1 else ""
    return f"PyTorch 线程 {threads}/{cores} 核{note}"
