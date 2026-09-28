"""配置加载：YAML 文件 + 命令行覆写 → 类型安全的 Config 对象。

支持三层叠加，后面的覆盖前面的：

    Config 默认值  →  YAML 文件  →  命令行 --set 覆写

这样一份 YAML 可以只写"和默认不同的项"，命令行可以只写"这次实验要临时改的
那一项"，而不必在每个 YAML 里复制一遍全部字段。

命令行覆写用点号路径：

    python scripts/train.py --config configs/train/g1_velocity.yaml \
        --set train.num_envs=128 --set env.robot=go2 --set ppo.lr_actor=1e-4

类型从 dataclass 的字段注解来推：字段声明为 int，覆写值 "128" 就会被转成
int 而不是留着字符串——省掉每个脚本手写 argparse 参数的重复劳动，也避免
"值传进去了但类型不对，跑一半才炸"。
"""

from __future__ import annotations

import dataclasses
import re
import typing
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from robotrl.configs.schema import Config

#: 匹配 1e-4 / 2.5E+3 / -1.5e-6 这类科学计数法。
_SCIENTIFIC = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


# ---------------------------------------------------------------------------
# YAML 读写
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读 YAML 文件。文件不存在时给出明确报错，而不是抛 FileNotFoundError。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} 的顶层必须是映射（键值对），实际是 {type(data).__name__}")
    return data


def dump_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """写 YAML。allow_unicode 保证中文注释类字段可读，不转成转义序列。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return path


# ---------------------------------------------------------------------------
# 合并与覆写
# ---------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个字典，override 优先。返回新字典，不修改入参。"""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """应用 "a.b.c=value" 形式的命令行覆写。

    值的类型在这里先不管——统一按字符串写进去，交给 build_config 时
    依据 dataclass 字段注解做转换。这样类型转换只有一处，不会出现
    "这里转成 int 那里忘了转"。
    """
    result = dict(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆写项 {item!r} 格式不对，应为 路径=值，例如 train.num_envs=128")
        path, _, raw = item.partition("=")
        keys = path.strip().split(".")
        if not all(keys):
            raise ValueError(f"覆写项 {item!r} 的路径有空段")

        cursor = result
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = _parse_scalar(raw.strip())
    return result


def _parse_scalar(raw: str) -> Any:
    """把命令行里的裸字符串转成合适的 Python 字面量。

    用 YAML 的解析器做这件事，比自己写一堆 int()/float() 的 try-except 可靠：
    "128" → int，"true" → bool，"[0.5, 1.25]" → list，"[a, b]" → 字符串列表。
    转换失败就原样返回字符串。
    """
    # YAML 1.1 的浮点字面量要求小数点前有数字，所以 "1e-4" 会被 safe_load
    # 原样当成字符串。而科学计数法是学习率这类超参最常见的写法，必须单独认。
    if _SCIENTIFIC.match(raw):
        return float(raw)

    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    # safe_load 会把裸词也解析成字符串，这里没有歧义风险
    return parsed if parsed is not None else raw


# ---------------------------------------------------------------------------
# dict → dataclass
# ---------------------------------------------------------------------------


def build_config(data: dict[str, Any]) -> Config:
    """从字典构造 Config，缺失字段用默认值，类型按字段注解转换。"""
    return _build_dataclass(Config, data)  # type: ignore[return-value]


def _build_dataclass(cls: type, data: Any) -> Any:
    """递归地把嵌套字典构造为 dataclass 实例。

    用 get_type_hints 而不是 f.type，因为模块开了 `from __future__ import
    annotations`，字段注解在运行时是字符串，直接比较类型会全部落空。
    """
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} 需要映射，实际拿到 {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}

    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__} 不认识这些配置项：{sorted(unknown)}；"
            f"可用项：{sorted(known)}"
        )

    kwargs = {}
    for name, field in ((f.name, f) for f in dataclasses.fields(cls)):
        if name not in data:
            continue
        kwargs[name] = _coerce(data[name], hints.get(name, field.type), field_name=name)

    return cls(**kwargs)


def _coerce(value: Any, hint: Any, *, field_name: str = "") -> Any:
    """按类型注解转换单个值。"""
    if dataclasses.is_dataclass(hint):
        return _build_dataclass(hint, value)

    origin = typing.get_origin(hint)

    if origin is tuple:
        if isinstance(value, str):
            # 允许 "0.5,1.25" 这种写法，命令行里比 [0.5, 1.25] 少敲几个字符
            value = [v.strip() for v in value.split(",")]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field_name} 需要序列，实际拿到 {value!r}")
        args = typing.get_args(hint)
        if args and args[-1] is not Ellipsis and len(args) != len(value):
            raise ValueError(
                f"{field_name} 需要 {len(args)} 个元素，实际给了 {len(value)} 个：{value!r}"
            )
        caster = args[0] if args else str
        return tuple(_cast_scalar(v, caster) for v in value)

    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} 需要映射，实际拿到 {value!r}")
        args = typing.get_args(hint)
        val_type = args[1] if len(args) == 2 else str
        return {str(k): _cast_scalar(v, val_type) for k, v in value.items()}

    return _cast_scalar(value, hint)


def _cast_scalar(value: Any, hint: Any) -> Any:
    """标量转换。已经是目标类型就原样返回，避免 bool("false") 这类误转。"""
    if hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "yes", "1"):
                return True
            if low in ("false", "no", "0"):
                return False
            raise ValueError(f"无法把 {value!r} 解释为布尔值")
        return bool(value)

    if hint in (int, float) and isinstance(value, str):
        return hint(float(value)) if hint is int else float(value)

    if hint is str:
        return str(value)

    return value


# ---------------------------------------------------------------------------
# dataclass → dict
# ---------------------------------------------------------------------------


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """把 Config 递归转成纯字典，用于存档和日志。"""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}
    if isinstance(cfg, dict):
        return {k: config_to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [config_to_dict(v) for v in cfg]
    if isinstance(cfg, Path):
        return str(cfg)
    return cfg


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Sequence[str] = (),
) -> Config:
    """对外主入口：读 YAML（可选）、叠加命令行覆写、构造 Config。"""
    data: dict[str, Any] = {}
    if path is not None:
        data = load_yaml(path)
    if overrides:
        data = apply_overrides(data, overrides)
    return build_config(data)
