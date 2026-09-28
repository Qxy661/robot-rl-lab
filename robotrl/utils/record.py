"""把回放帧录成 GIF。

这一层刻意和 MuJoCo 无关：帧以 numpy RGB 数组进来，出去是一个 GIF 文件。
分开的理由有两层——录制的正确性（帧数、尺寸、没有 Pillow 时报什么错）值得
单独测，而测它不该需要加载一个机器人模型；换渲染源（别的仿真器、真机相机）
时这个模块原样能用。

GIF 只有 256 色，调色板怎么选决定了成片质量。这里用首帧的调色板贯穿全片：
逐帧各自量化单看每帧都更准，但相邻帧的最优调色板并不相同，播放时颜色会
来回跳，整片闪烁——比颜色少本身难看得多。

一个要知道的下游行为：写盘时 Pillow 会把**连续相同的帧合并成一帧**并把时长
叠上去。回放机器人静止时，相机跟踪加上物理状态不变，连着几帧是逐像素相同的，
于是帧数会明显少于喂进去的数目。这不是丢帧——总时长不变，播放效果一样，
文件更小。所以判断录制是否完整要看时长，不能看帧数。
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

#: 默认帧率。控制周期 20 ms（50 Hz）时，25 fps 正好是"两个控制步一帧"。
DEFAULT_FPS = 25

#: GIF 的颜色表是 256 格，留一格做索引用，剩下的给颜色。
_PALETTE_COLORS = 255

_INSTALL_HINT = 'pip install -e ".[record]"'


def _pillow() -> Any:
    """延迟导入 Pillow。

    录制是可选功能，没装它的人应该只在真的要用录制时才看到这条依赖，
    而不是在 import robotrl 的时候就撞上一句 ImportError。
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(f"录制 GIF 需要 Pillow。装法：{_INSTALL_HINT}") from None
    return Image


class GifRecorder:
    """攒帧，然后写成一个 GIF。

    Args:
        path: 输出文件，父目录不存在会自动建。
        fps: 播放帧率。它只写进 GIF 的帧间隔元数据，和你喂帧的快慢无关。
        width: 缩放到这个宽度（保持长宽比）；None 表示原尺寸。
        max_frames: 帧数上限。录满之后 `add` 返回 False 并丢弃该帧——
            这是防止"忘了停"录出一个几十兆的 GIF。默认值按 25 fps 算
            大约是 24 秒。

    用法上是个上下文管理器，异常路径也会把已经录到的帧写出来：

        with GifRecorder("docs/assets/g1.gif", width=480) as rec:
            rec.add(renderer.render())
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fps: int = DEFAULT_FPS,
        width: int | None = None,
        max_frames: int = 600,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"帧率必须为正，实际是 {fps}")
        if max_frames <= 0:
            raise ValueError(f"帧数上限必须为正，实际是 {max_frames}")
        if width is not None and width <= 0:
            raise ValueError(f"宽度必须为正，实际是 {width}")

        self.path = Path(path)
        self.fps = int(fps)
        self.width = None if width is None else int(width)
        self.max_frames = int(max_frames)
        self._frames: list[np.ndarray] = []
        self._closed = False

    # ------------------------------------------------------------------
    # 收帧
    # ------------------------------------------------------------------

    def add(self, frame: np.ndarray) -> bool:
        """收一帧。返回 False 表示已录满，这一帧被丢掉了。"""
        if self._closed:
            raise RuntimeError("录制已经结束，不能再加帧")
        if self.full:
            return False

        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(
                f"帧应当是 (高, 宽, 3) 或 (高, 宽, 4) 的 RGB(A) 数组，实际拿到 {array.shape}"
            )
        # GIF 没有 alpha 通道，四通道的输入把 alpha 丢掉而不是报错：渲染器
        # 给不给 alpha 是它的事，录制方不该为此改代码。
        self._frames.append(array[..., :3])
        return True

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def full(self) -> bool:
        return len(self._frames) >= self.max_frames

    def __len__(self) -> int:
        return len(self._frames)

    def __repr__(self) -> str:
        state = "已结束" if self._closed else f"{self.frame_count}/{self.max_frames} 帧"
        return f"GifRecorder({self.path.name}, {self.fps} fps, {state})"

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------

    def close(self) -> Path:
        """写出 GIF，返回文件路径。重复调用不会重写文件。"""
        if self._closed:
            return self.path
        self._closed = True

        if not self._frames:
            raise ValueError(f"一帧都没录到，不写 {self.path}")

        image = _pillow()
        images = [self._to_image(frame, image) for frame in self._frames]

        # 首帧自适应量化，其余帧映射到同一张调色板，避免逐帧变色。
        palette = images[0].quantize(colors=_PALETTE_COLORS, method=image.Quantize.MEDIANCUT)
        rest = [
            frame.quantize(palette=palette, dither=image.Dither.FLOYDSTEINBERG)
            for frame in images[1:]
        ]

        self.path.parent.mkdir(parents=True, exist_ok=True)
        palette.save(
            self.path,
            format="GIF",
            save_all=True,
            append_images=rest,
            duration=round(1000 / self.fps),
            loop=0,
        )
        # 帧已经进了文件，留着只是占内存。
        self._frames.clear()
        return self.path

    def _to_image(self, frame: np.ndarray, image: Any) -> Any:
        picture = image.fromarray(frame)
        if self.width is None or self.width == picture.width:
            return picture
        height = max(1, round(picture.height * self.width / picture.width))
        return picture.resize((self.width, height), image.Resampling.LANCZOS)

    def __enter__(self) -> GifRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # 异常路径也写盘：崩之前的那些帧往往正是要看的证据。
        if self._frames:
            self.close()


__all__ = ["DEFAULT_FPS", "GifRecorder"]
