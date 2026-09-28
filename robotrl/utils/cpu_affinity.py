"""把测量绑到某一个逻辑核上。

为什么需要
----------
微秒级的推理基准里，"最小延迟"这个统计量只有在**同一个核上**才可比。这一点在
大小核混合架构的 CPU 上尤其致命：本项目实测的这台 Intel Core Ultra 7 255H，
同一个模型、同一份输入，逐个逻辑核测最小延迟，最快 0.0108 ms、最慢 0.0436 ms，
差 4 倍。

不绑核时进程由调度器在核之间迁移。待测的差异（INT8 与 FP32 之间约 15%）比核
迁移带来的波动（400%）小两个量级，于是测出来的结论跟着进程落在哪个核上翻来
覆去：同一份模型在不同次运行里得到过 0.86× 和 1.24×，方向相反。这不是模型变了，
是进程换了核。

"按 min 下结论"这条一般性的原则救不了这个：min 排除的是**时间维度**的干扰
（别的进程抢了一会儿 CPU），而核迁移是**空间维度**的——它改变的是基准性能本身，
不是某一次测量被拖慢了，所以最小值一样会跟着变。

绑核之后（实测，12 轮交替）：FP32 min 0.0097 ms、INT8 min 0.0115 ms，各轮比值
0.839–0.904，轮间波动 1.08，落回阈值以内。

怎么读这个结果
--------------
绝对值是"这个核上的绝对值"。换一个核，0.0097 ms 就不成立（实测能到 0.0436 ms），
换一台机器更不成立。所以能迁移的结论是**比值**，不是绝对延迟。报告里因此把
用的是哪个核一并记下来。

绑核不是万能的：如果别的进程正占着这个核，测量照样会被拖慢。它的作用是把
"哪个核"从随机变量变成一个被记录下来的常量。
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import time
from collections.abc import Callable, Iterator

#: 这台机器上可用的逻辑核数量。用作不指定候选核时的全集。
__all__ = [
    "affinity_supported",
    "cpu_count",
    "current_affinity",
    "fastest_cpu",
    "pinned",
]


def cpu_count() -> int:
    """逻辑核数量。取不到时按 1 处理，绑核退化成空操作。"""
    return os.cpu_count() or 1


def affinity_supported() -> bool:
    """当前平台能不能设 CPU 亲和性。

    Windows 走 SetProcessAffinityMask，Linux 走 sched_setaffinity。容器里
    sched_setaffinity 常被受限（返回 EINVAL），所以这个函数只回答"这个平台
    有没有这个接口"，实际能不能设由 `set_affinity` 的返回值说了算。
    """
    return sys.platform == "win32" or hasattr(os, "sched_setaffinity")


def _windows_mask(cpus: set[int]) -> int:
    mask = 0
    for cpu in cpus:
        mask |= 1 << cpu
    return mask


def set_affinity(cpus: set[int]) -> bool:
    """把当前进程绑到给定的逻辑核集合上。成功返回 True。

    失败返回 False 而不是抛异常：设不上亲和性是测量质量问题，不该让整个基准
    跑不下去。调用方应当把这个结果记进报告，而不是当没事发生。
    """
    if not cpus:
        return False
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        # 句柄必须是 c_void_p。ctypes 默认按 32 位 int 传，GetCurrentProcess
        # 返回的伪句柄 -1 被截断后就不是合法句柄了，调用会静默失败。
        handle = ctypes.c_void_p(kernel32.GetCurrentProcess())
        return bool(kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(_windows_mask(cpus))))
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, cpus)
        except OSError:
            return False
        return True
    return False


def current_affinity() -> set[int] | None:
    """当前进程的亲和性；取不到返回 None（表示"未限制"）。"""
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        handle = ctypes.c_void_p(kernel32.GetCurrentProcess())
        process_mask = ctypes.c_size_t(0)
        system_mask = ctypes.c_size_t(0)
        ok = kernel32.GetProcessAffinityMask(
            handle, ctypes.byref(process_mask), ctypes.byref(system_mask)
        )
        if not ok:
            return None
        return {i for i in range(cpu_count()) if process_mask.value >> i & 1}
    if hasattr(os, "sched_getaffinity"):
        with contextlib.suppress(OSError):
            return set(os.sched_getaffinity(0))
    return None


@contextlib.contextmanager
def pinned(cpu: int) -> Iterator[bool]:
    """测量期间把进程绑到 `cpu` 这个逻辑核上，退出时恢复原来的亲和性。

    恢复放在 finally 里：测量中途报错时也必须还原，否则异常会带着"进程只能
    用一个核"这个副作用扩散到调用方后面的代码里，而且极难联想到原因。

    产出值为是否真的绑上了。绑不上时不抛异常，只让调用方知道该在报告里标注。
    """
    before = current_affinity()
    ok = set_affinity({cpu})
    try:
        yield ok
    finally:
        if ok and before is not None:
            set_affinity(before)


def fastest_cpu(
    probe: Callable[[], None],
    *,
    cpus: list[int] | None = None,
    repeats: int = 5,
    on_probe: Callable[[int, float], None] | None = None,
) -> tuple[int, dict[int, float], bool]:
    """逐个逻辑核跑一遍 `probe`，返回最快的那个核。

    Args:
        probe: 一次待测操作。它应当自带预热——预热放在这里做的话，每个核都要
            重新预热一遍，慢得多，而且预热本身要在一个固定的核上完成才有意义。
        cpus: 候选核。默认全部逻辑核。
        repeats: 每个核测几次，取最小值。
        on_probe: 每测完一个核回调一次 (核号, 最小耗时秒)，用于打进度。

    Returns:
        (最快的核号, 每个核的最小耗时, 是否真的绑核成功)。绑核失败时第一个
        元素是 `cpus[0]`，第三个元素是 False——调用方据此决定要不要把这次
        结果当"已绑核"来解读。

    Note:
        逐个核测这件事本身有偏：先测的核可能正好赶上别的进程忙。所以每个核
        之间没有休息，靠取多次最小值来压低偶发干扰，而不是靠测量顺序——顺序
        带来的偏差没有便宜的办法消除，只能如实记录测的是哪一轮。
    """
    candidates = list(range(cpu_count())) if cpus is None else list(cpus)
    if not candidates:
        raise ValueError("候选核不能为空")

    best_cpu = candidates[0]
    best_time = float("inf")
    timings: dict[int, float] = {}
    pinned_ok = True

    for cpu in candidates:
        with pinned(cpu) as ok:
            pinned_ok = pinned_ok and ok
            best = float("inf")
            for _ in range(max(1, repeats)):
                t0 = time.perf_counter()
                probe()
                best = min(best, time.perf_counter() - t0)
        timings[cpu] = best
        if on_probe is not None:
            on_probe(cpu, best)
        if best < best_time:
            best_time = best
            best_cpu = cpu

    return best_cpu, timings, pinned_ok
