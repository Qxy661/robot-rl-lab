"""CPU 亲和性。

这一组测的是"测量的前提"而不是测量本身：延迟基准要靠绑核才能得到可比的数字，
绑核的保存/恢复如果写错，症状是**别的**测试或调用方莫名其妙变慢——那是最难
查的一类问题。所以这里重点验恢复，而不是验绑上了没有。

容器里 `sched_setaffinity` 常被限制，绑不上时跳过：这不是代码错，是环境不给。
绑不上这件事本身有专门的用例覆盖（`pinned` 如实报告失败，而不是抛异常）。
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import pytest

from robotrl.utils.cpu_affinity import (
    affinity_supported,
    cpu_count,
    current_affinity,
    fastest_cpu,
    pinned,
    set_affinity,
)


@pytest.fixture
def one_cpu() -> Iterator[int]:
    """能真正绑上去的某个逻辑核；绑不上就跳过。"""
    if not affinity_supported():
        pytest.skip("当前平台不支持设 CPU 亲和性")
    with pinned(0) as ok:
        if not ok:
            pytest.skip("环境不允许设 CPU 亲和性（容器里常见）")
    return 0


# ---------------------------------------------------------------------------
# 平台能力
# ---------------------------------------------------------------------------


def test_cpu_count_is_at_least_one():
    """核数是逐个核试探的上界来源，取不到时也必须是 1 而不是 0。

    0 会让 `fastest_cpu` 的默认候选集为空，而那里的报错信息是"候选核不能为空"，
    看着像调用方传错了参数——实际是核数没读到。
    """
    assert cpu_count() >= 1


def test_set_affinity_with_an_empty_set_is_a_no_op():
    """空集合不可绑，且要如实返回 False 而不是抛异常。"""
    assert set_affinity(set()) is False


def test_current_affinity_is_none_or_a_nonempty_set():
    """取不到时返回 None，取到时不能是空集——空集在调用方眼里等于"哪个核都不许用"。"""
    affinity = current_affinity()

    assert affinity is None or len(affinity) >= 1


# ---------------------------------------------------------------------------
# 绑与恢复
# ---------------------------------------------------------------------------


def test_pinned_reports_whether_it_actually_bound(one_cpu):
    """`pinned` 的产出值必须反映真实情况。

    绑不上却报成功，调用方就会把一份"进程在核之间乱跑"的结果当成可比的基准
    写进报告——这正是引入绑核要解决的那个问题，报错了反而更糟。
    """
    with pinned(one_cpu) as ok:
        assert ok is True
        assert current_affinity() == {one_cpu}


def test_affinity_is_restored_after_the_block(one_cpu):
    """退出后必须回到原来的亲和性。

    不恢复的话，整个进程（训练、回放、后续的测量）都被压在一个核上，症状是
    别处变慢，而原因在几十行之前的这段测量代码里。
    """
    before = current_affinity()

    with pinned(one_cpu):
        pass

    assert current_affinity() == before


def test_affinity_is_restored_even_when_the_body_raises(one_cpu):
    """异常路径同样要恢复：测量中途报错是最容易漏掉恢复的路径。"""
    before = current_affinity()

    with pytest.raises(RuntimeError, match="测到一半炸了"), pinned(one_cpu) as ok:
        assert ok is True
        raise RuntimeError("测到一半炸了")

    assert current_affinity() == before


def test_pinning_actually_takes_effect(one_cpu):
    """绑上之后，进程真的只在一个核上跑。"""
    with pinned(one_cpu):
        assert current_affinity() == {one_cpu}
        assert os.getpid() > 0  # 触发一次系统调用，确认进程还活着且可用


# ---------------------------------------------------------------------------
# 挑核
# ---------------------------------------------------------------------------


def test_fastest_cpu_rejects_an_empty_candidate_list():
    with pytest.raises(ValueError, match="候选核不能为空"):
        fastest_cpu(lambda: None, cpus=[])


def test_fastest_cpu_probes_every_candidate_the_requested_number_of_times(one_cpu):
    """每个候选核都要测 `repeats` 次：少测了会让"最快"变成抽签结果。"""
    calls = []

    cpu, timings, ok = fastest_cpu(lambda: calls.append(1), cpus=[one_cpu], repeats=4)

    assert len(calls) == 4
    assert cpu == one_cpu
    assert set(timings) == {one_cpu}
    assert ok is True


def test_fastest_cpu_picks_the_shortest_probe_time(one_cpu):
    """排序逻辑本身：让探针的耗时随所在核号递增，最快的核应该是 0 号。

    探针要能感知自己在哪个核上，所以它读 `current_affinity`。这也是这条用例
    必须依赖绑核的原因——绑不上时探针读到的永远是全集。
    """
    candidates = [0, 1, 2]

    def probe() -> None:
        affinity = current_affinity() or {0}
        cpu = min(affinity)
        # 忙等而不是 sleep：sleep 的粒度在毫秒以上，三次会一样长，排不出序。
        deadline = time.perf_counter() + 0.0003 * (cpu + 1)
        while time.perf_counter() < deadline:
            pass

    cpu, timings, ok = fastest_cpu(probe, cpus=candidates, repeats=2)

    assert ok is True
    assert cpu == 0
    assert timings[0] == min(timings.values())
    assert timings[0] < timings[2]
