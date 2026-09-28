"""GIF 录制。

不加载 MuJoCo：`robotrl/utils/record.py` 只认 numpy 数组，所以这一组用例是
毫秒级的。测的是"写出来的东西能不能被读回来"——帧数、尺寸、颜色、以及几处
容易悄悄写错的边界（一帧没录、录满之后、缺 Pillow）。

Pillow 是 record extra，本地只装 `.[dev]` 时会整组跳过；CI 装的是
`.[dev,deploy,record]`，那里会真跑。
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("PIL", reason='录制依赖 Pillow，装法：pip install -e ".[record]"')

from PIL import Image  # noqa: E402

from robotrl.utils.record import GifRecorder  # noqa: E402


def _frame(height: int = 8, width: int = 8, shift: int = 0) -> np.ndarray:
    """造一帧有梯度和色偏的图。

    不能全用纯色：纯色下不管调色板怎么选都对，测不出量化有没有做坏。
    """
    ramp = np.linspace(0, 255, width, dtype=np.uint8)
    plane = np.tile(ramp, (height, 1))
    return np.ascontiguousarray(
        np.stack([plane, np.roll(plane, shift, axis=1), 255 - plane], axis=-1)
    )


def _read_frames(path) -> list[np.ndarray]:
    """把 GIF 读回来，逐帧转成 RGB 数组。"""
    with Image.open(path) as image:
        frames = []
        for index in range(image.n_frames):
            image.seek(index)
            frames.append(np.asarray(image.convert("RGB")).copy())
        return frames


def _write(tmp_path, frames, **kwargs):
    recorder = GifRecorder(tmp_path / "out.gif", **kwargs)
    for frame in frames:
        assert recorder.add(frame)
    return recorder.close()


def test_gif_round_trips_with_the_frame_count_it_was_given(tmp_path):
    """写出去的帧数要和喂进去的一致。

    这条看着显然，但它实际验的是 GIF 的 `save_all`/`append_images` 那条路径
    真的接上了——只给首帧的话 Pillow 会安安静静写出一张单帧图，不报错。
    """
    frames = [_frame(shift=i * 3) for i in range(5)]

    path = _write(tmp_path, frames)

    assert len(_read_frames(path)) == 5


def test_recorded_colors_stay_close_to_the_source(tmp_path):
    """量化到 256 色允许有误差，但不该把画面整变色。"""
    source = _frame()

    restored = _read_frames(_write(tmp_path, [source]))[0]

    # 中位误差而不是均值：单像素的边缘误差不影响观感，平均值会把它放大成结论。
    assert float(np.median(np.abs(restored.astype(int) - source.astype(int)))) < 12


def test_all_frames_share_one_palette(tmp_path):
    """全片共用首帧的调色板。

    逐帧自适应量化单看每帧更准，但相邻帧的最优调色板不同，播放时颜色会来回
    跳，整片闪烁——比颜色少难看得多。这条用例把那个取舍钉住：如果哪次改动让
    每帧各带一张调色板，这里会红。
    """
    frames = [_frame(shift=i * 11) for i in range(4)]

    path = _write(tmp_path, frames)

    with Image.open(path) as image:
        palettes = []
        for index in range(image.n_frames):
            image.seek(index)
            palettes.append(bytes(image.palette.palette))
    assert len(set(palettes)) == 1


def test_width_scales_and_keeps_the_aspect_ratio(tmp_path):
    """按宽度缩放，高度跟着比例走。"""
    path = _write(tmp_path, [_frame(height=60, width=80)], width=40)

    with Image.open(path) as image:
        assert image.size == (40, 30)


def test_frame_that_exceeds_the_limit_is_refused_not_stored(tmp_path):
    """录满之后不再收帧，并且如实返回 False。

    上限的意义就是防"忘了停"，所以超出的帧必须丢掉；悄悄收下等于没有上限。
    """
    recorder = GifRecorder(tmp_path / "out.gif", max_frames=2)

    assert recorder.add(_frame()) is True
    assert recorder.add(_frame()) is True
    assert recorder.full is True
    assert recorder.add(_frame()) is False
    assert recorder.frame_count == 2


def test_closing_without_any_frame_fails_loudly_and_writes_nothing(tmp_path):
    """一帧都没录到就报错，而不是写出一个 0 帧的空文件。

    空 GIF 能被很多查看器"成功"打开，然后什么都不显示——这种失败最费时间。
    """
    path = tmp_path / "empty.gif"
    recorder = GifRecorder(path)

    with pytest.raises(ValueError, match="一帧都没录到"):
        recorder.close()

    assert not path.exists()


def test_close_is_idempotent(tmp_path):
    """重复 close 返回同一路径，不重写文件。"""
    recorder = GifRecorder(tmp_path / "out.gif")
    recorder.add(_frame())

    first = recorder.close()
    stamp = first.stat().st_mtime_ns
    second = recorder.close()

    assert first == second
    assert second.stat().st_mtime_ns == stamp


def test_adding_after_close_is_an_error(tmp_path):
    """写盘之后再喂帧要报错：那些帧不会被写进去，静默丢掉就是骗人。"""
    recorder = GifRecorder(tmp_path / "out.gif")
    recorder.add(_frame())
    recorder.close()

    with pytest.raises(RuntimeError, match="录制已经结束"):
        recorder.add(_frame())


def test_rgba_frames_are_accepted_with_alpha_dropped(tmp_path):
    """四通道输入照收。

    渲染器给不给 alpha 是它的事，录制方不该为了这个改代码；GIF 本来也没有
    alpha 通道，丢掉是唯一的正确做法。
    """
    rgba = np.concatenate([_frame(), np.full((8, 8, 1), 128, dtype=np.uint8)], axis=-1)
    recorder = GifRecorder(tmp_path / "out.gif")

    assert recorder.add(rgba) is True

    assert len(_read_frames(recorder.close())) == 1


def test_frame_with_a_wrong_shape_is_rejected(tmp_path):
    """形状不对直接报错，并说清楚期望什么。"""
    recorder = GifRecorder(tmp_path / "out.gif")

    with pytest.raises(ValueError, match=r"\(高, 宽, 3\)"):
        recorder.add(np.zeros((4, 4), dtype=np.uint8))


def test_invalid_limits_are_rejected_at_construction(tmp_path):
    """参数错在建的时候就报，不要等到录了一大轮才发现帧率是负的。"""
    for kwargs in ({"fps": 0}, {"max_frames": 0}, {"width": -1}):
        with pytest.raises(ValueError):
            GifRecorder(tmp_path / "out.gif", **kwargs)


def test_context_manager_writes_what_was_recorded_before_an_exception(tmp_path):
    """异常路径也落盘：崩之前那几帧往往正是要看的证据。"""
    path = tmp_path / "out.gif"

    with pytest.raises(RuntimeError, match="模拟中途失败"), GifRecorder(path) as recorder:
        recorder.add(_frame(shift=0))
        recorder.add(_frame(shift=20))
        raise RuntimeError("模拟中途失败")

    assert len(_read_frames(path)) == 2


def test_identical_consecutive_frames_merge_but_total_duration_is_kept(tmp_path):
    """连续相同帧会被合并成一帧，但总时长不变。

    这是 Pillow 的行为，不是我们加的，但必须知道：回放里机器人静止时连着
    好几帧是逐像素相同的（相机跟踪 + 物理状态不变），它们会被合成一帧并把
    时长叠上去。所以判断"录全了没有"要看总时长，不能看帧数——按帧数看会
    以为丢了帧，其实播放出来一模一样，文件还更小。
    """
    path = _write(tmp_path, [_frame() for _ in range(6)], fps=25)

    with Image.open(path) as image:
        durations = []
        for index in range(image.n_frames):
            image.seek(index)
            durations.append(image.info["duration"])

    # 帧数塌了，但 6 帧 × 40 ms 的总时长原样保留。
    assert len(durations) < 6
    assert sum(durations) == 6 * 40


def test_missing_pillow_names_the_extra_to_install(tmp_path, monkeypatch):
    """没装 Pillow 时的报错要给出能直接粘的命令。

    `import PIL` 会因为 sys.modules 里是 None 而抛 ImportError，正好拿来模拟
    "这台机器上没装"——不用真的把包装卸一遍。
    """
    recorder = GifRecorder(tmp_path / "out.gif")
    recorder.add(_frame())
    monkeypatch.setitem(sys.modules, "PIL", None)

    with pytest.raises(ImportError, match=r"\[record\]"):
        recorder.close()
