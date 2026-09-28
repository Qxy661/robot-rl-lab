"""JSON 基线：让结果不重跑也能看、能比。

实验数据最常见的浪费是"只有跑过那次的人知道"。把评估结果写成一个固定格式的
JSON 提交进仓库，别人 clone 下来就能看数、能 diff、能把新结果和旧基线并排
比一遍，不必先复现整个训练。

格式设计上的三个取舍：

- **键排序 + 缩进 + 末尾换行**。同一个实验导出两次应该得到逐字节相同的文件，
  否则 git diff 里全是无意义的行序变化，真正的数值变化反而被淹掉。
- **浮点截到固定精度**。报告的读者是人和 diff 工具，不是下一个计算步骤。
  6 位小数对回报量级（几百到几千）已经是 1e-9 的相对分辨率，远小于换一批
  种子的波动，留着全精度只会让 diff 噪声变大。
- **schema/version 显式写进文件**。格式以后改了，老报告还能被认出来并给出
  明确报错，而不是解析到一半抛 KeyError。
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: 报告格式标识与版本。改结构时递增 version，load_report 会据此拒绝旧格式。
SCHEMA = "robotrl.eval.report"
VERSION = 1

#: 默认保留的小数位数。
DEFAULT_PRECISION = 6


# ---------------------------------------------------------------------------
# 浮点整理
# ---------------------------------------------------------------------------


def _round_floats(value: Any, precision: int, path: str = "") -> Any:
    """递归地把浮点截到指定精度，顺路挡住非有限值。

    NaN / inf 写进 JSON 是非法字面量（Python 的 json 会写出裸的 NaN，严格
    解析器读不了）。更关键的是，指标里出现 NaN 就说明计算本身出了问题，
    静默落盘等于把它藏起来，不如在这里直接报错、把键名指出来。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"指标 {path or '<root>'} 不是有限值（{value}），拒绝写入报告")
        return round(value, precision)
    if isinstance(value, Mapping):
        return {
            str(k): _round_floats(v, precision, f"{path}.{k}" if path else str(k))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_round_floats(v, precision, f"{path}[{i}]") for i, v in enumerate(value)]
    return value


# ---------------------------------------------------------------------------
# 报告对象
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Report:
    """一份评估基线。

    Attributes:
        meta: 描述这份结果是怎么来的——配置、checkpoint、commit、时间。它不参与
            数值对比，但少了它，三个月后没人说得清这个数对应哪次实验。
        metrics: 指标本身，结构由调用方决定（评估层用 EvalResult.to_dict()）。
    """

    meta: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA
    version: int = VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "meta": self.meta,
            "metrics": self.metrics,
        }


def save_report(
    metrics: Any,
    path: str | Path,
    meta: Mapping[str, Any] | None = None,
    *,
    precision: int = DEFAULT_PRECISION,
) -> Path:
    """把指标写成 JSON 报告。

    Args:
        metrics: 指标字典。带 to_dict() 的对象（EvalResult、TrainMetrics）会先
            转成字典，省得每个调用点都写一遍 .to_dict()。
        path: 目标路径。父目录不存在会自动创建。
        meta: 来源信息。工具不擅自塞时间戳——自动写进去会让同一个实验的两次
            导出产生无意义的 diff，需要就由调用方显式传。
        precision: 浮点保留的小数位数。

    Returns:
        实际写入的路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(metrics, Mapping) and hasattr(metrics, "to_dict"):
        metrics = metrics.to_dict()
    if not isinstance(metrics, Mapping):
        raise TypeError(f"metrics 需要映射或带 to_dict() 的对象，实际是 {type(metrics).__name__}")

    report = Report(
        meta=dict(meta) if meta else {},
        metrics=_round_floats(dict(metrics), precision, "metrics"),
    )
    # sort_keys 是"可 diff"的关键：键序稳定，diff 里才只剩下数值变化。
    # ensure_ascii=False 让中文 meta 直接可读，末尾补换行免得 diff 报 \ No newline。
    text = json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def load_report(path: str | Path) -> Report:
    """读一份报告。格式不对时给出能直接定位问题的报错。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"报告不存在：{path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 不是合法 JSON：{exc}") from exc

    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(
            f"{path} 不是 {SCHEMA} 报告（读到 schema={data.get('schema')!r}）"
            if isinstance(data, dict)
            else f"{path} 的顶层应为对象，实际是 {type(data).__name__}"
        )

    version = data.get("version")
    if version != VERSION:
        raise ValueError(
            f"{path} 的报告版本是 {version}，本版工具只认 {VERSION}；"
            "请用生成它的那版 robotrl 读取，或重新生成"
        )

    return Report(meta=data.get("meta") or {}, metrics=data.get("metrics") or {})


# ---------------------------------------------------------------------------
# 拍平与取值
# ---------------------------------------------------------------------------


def flatten_metrics(
    metrics: Mapping[str, Any],
    *,
    skip_prefixes: Sequence[str] = (),
    _prefix: str = "",
) -> dict[str, Any]:
    """把嵌套指标拍成 `a.b[0].c` 形式的扁键。

    嵌套结构适合存盘，扁键适合对比：两边的键一一对上才能逐项算差值，也才能
    一眼说出"是 summary 变了还是逐段的某一维变了"。
    """
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        path = f"{_prefix}.{key}" if _prefix else str(key)
        if any(path == p or path.startswith(p + ".") for p in skip_prefixes):
            continue
        if isinstance(value, Mapping):
            flat.update(flatten_metrics(value, skip_prefixes=skip_prefixes, _prefix=path))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                item_path = f"{path}[{i}]"
                if isinstance(item, Mapping):
                    flat.update(
                        flatten_metrics(item, skip_prefixes=skip_prefixes, _prefix=item_path)
                    )
                elif isinstance(item, (int, float)) and not isinstance(item, bool):
                    flat[item_path] = item
        else:
            flat[path] = value
    return flat


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# 终端表格
# ---------------------------------------------------------------------------


def _display_width(text: str) -> int:
    """字符串在终端里占几列。中文按 len() 算会少一半，表格就全歪了。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, *, align: str = "left") -> str:
    fill = " " * max(0, width - _display_width(text))
    return fill + text if align == "right" else text + fill


def _format_number(value: Any) -> str:
    """数值的显示格式。量级跨度大，太小太大的走科学计数法，其余保持可读。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if not _is_number(value):
        return str(value)
    value = float(value)
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5 or magnitude < 1e-4:
        return f"{value:.4e}"
    return f"{value:.6g}"


def _format_delta(delta: float) -> str:
    return f"{delta:+.6g}" if delta else "0"


def _format_ratio(a: Any, b: Any) -> str:
    """相对变化。基线为 0 时相对变化没有定义，只能留空。"""
    if not (_is_number(a) and _is_number(b)) or float(a) == 0:
        return "—"
    return f"{(float(b) - float(a)) / abs(float(a)) * 100:+.2f}%"


def _render_table(
    header: Sequence[str], rows: Sequence[Sequence[str]], aligns: Sequence[str]
) -> str:
    widths = [_display_width(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _display_width(cell))

    lines = ["  ".join(_pad(h, widths[i], align=aligns[i]) for i, h in enumerate(header)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    lines.extend(
        "  ".join(_pad(c, widths[i], align=aligns[i]) for i, c in enumerate(row)).rstrip()
        for row in rows
    )
    return "\n".join(lines)


def _as_report(source: Report | Mapping[str, Any] | str | Path) -> Report:
    """接受报告对象、字典或路径，统一成 Report。"""
    if isinstance(source, Report):
        return source
    if isinstance(source, (str, Path)):
        return load_report(source)
    if isinstance(source, Mapping):
        return Report(
            meta=dict(source.get("meta") or {}), metrics=dict(source.get("metrics") or {})
        )
    raise TypeError(f"无法把 {type(source).__name__} 当作报告处理")


def compare_reports(
    a: Report | Mapping[str, Any] | str | Path,
    b: Report | Mapping[str, Any] | str | Path,
    *,
    label_a: str = "A",
    label_b: str = "B",
    include_unchanged: bool = False,
    include_episodes: bool = False,
    max_rows: int = 200,
) -> str:
    """对比两份报告，返回可直接 print 的差异表。

    默认只列有变化的项——两份报告里绝大多数行都是一样的，全列出来会把真正
    变了的那几行淹掉。想看全量就传 include_unchanged=True。

    Args:
        a: 基线报告（对象、字典或路径）。
        b: 待比较的报告。
        label_a / label_b: 表格列名，通常是两次实验的简称。
        include_unchanged: 是否连没变化的项一起列。
        include_episodes: 是否展开逐回合明细。默认关掉：几十上百个回合会把
            表格撑成几千行，而那些数在报告文件里随时能查。
        max_rows: 表格最多显示多少行，超出部分只给计数。
    """
    report_a, report_b = _as_report(a), _as_report(b)
    skip = () if include_episodes else ("episodes",)
    flat_a = flatten_metrics(report_a.metrics, skip_prefixes=skip)
    flat_b = flatten_metrics(report_b.metrics, skip_prefixes=skip)

    only_a = sorted(set(flat_a) - set(flat_b))
    only_b = sorted(set(flat_b) - set(flat_a))
    shared = sorted(set(flat_a) & set(flat_b))

    rows: list[list[str]] = []
    changed = 0
    for key in shared:
        va, vb = flat_a[key], flat_b[key]
        if _is_number(va) and _is_number(vb):
            delta = float(vb) - float(va)
            is_changed = delta != 0
            if not is_changed and not include_unchanged:
                continue
            changed += int(is_changed)
            rows.append(
                [
                    key,
                    _format_number(va),
                    _format_number(vb),
                    _format_delta(delta),
                    _format_ratio(va, vb),
                ]
            )
        else:
            if va == vb and not include_unchanged:
                continue
            changed += int(va != vb)
            rows.append([key, _format_number(va), _format_number(vb), "", ""])

    lines: list[str] = []
    lines.append(f"报告对比：{label_a} → {label_b}")
    lines.append(f"schema={report_a.schema} v{report_a.version}")
    if include_episodes:
        lines.append("（含逐回合明细）")
    lines.append("")

    shown = rows[:max_rows]
    if shown:
        lines.append(
            _render_table(
                ["指标", label_a, label_b, "Δ", "Δ%"],
                shown,
                ["left", "right", "right", "right", "right"],
            )
        )
    else:
        lines.append("共同指标中没有差异。" if not include_unchanged else "两份报告没有共同指标。")
    if len(rows) > len(shown):
        lines.append(
            f"... 另有 {len(rows) - len(shown)} 行未显示（共 {len(rows)} 行有变化/被列出）"
        )

    lines.append("")
    lines.append(f"共同指标 {len(shared)} 项，其中 {changed} 项有变化。")
    if only_a:
        lines.append(f"只在 {label_a} 中出现（{len(only_a)} 项）：{_preview(only_a)}")
    if only_b:
        lines.append(f"只在 {label_b} 中出现（{len(only_b)} 项）：{_preview(only_b)}")

    meta_diff = _meta_differences(report_a.meta, report_b.meta)
    if meta_diff:
        lines.append("")
        lines.append("来源信息差异（不参与数值对比，但数值变了先看这里）：")
        lines.extend(
            f"  {key}: {_format_number(va)} → {_format_number(vb)}" for key, va, vb in meta_diff
        )

    return "\n".join(lines)


def _preview(keys: Sequence[str], limit: int = 8) -> str:
    head = ", ".join(keys[:limit])
    return head if len(keys) <= limit else f"{head}, ..."


def _meta_differences(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    """列出 meta 里不一致的项。数值对不上时，先看是不是配置或 commit 就不是同一个。"""
    out: list[tuple[str, Any, Any]] = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            out.append((key, a.get(key, "—"), b.get(key, "—")))
        elif a[key] != b[key]:
            out.append((key, a[key], b[key]))
    return out
