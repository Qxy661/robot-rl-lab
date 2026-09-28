"""ONNX metadata 的读写。

metadata 装的是"图里放不下、但端侧必须知道"的那部分契约：动作还原公式的两个
常数（default_angles / action_scale）、观测分段布局、关节顺序。它们不参与计算，
所以不该进计算图——融进去之后量化误差里就混了动作还原误差，误差归因就做不干净。

两个层都要写它：算法层在 `Policy.export_onnx()` 里写，部署层在量化之后补写
（量化器不保证原样搬运 metadata，而端侧拿到一份没有 default_angles 的 INT8
模型就还原不出关节角）。所以它放在这里——既不属于算法层，也不属于部署层。

本模块只依赖 onnx，不依赖 torch：量化、基准这些环节读写 metadata 时不该被迫
把 torch 拖进来。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_metadata(path: str | Path) -> dict[str, str]:
    """读 ONNX 的 metadata_props。文件不存在时抛出明确错误。

    load_external_data=False：只读 metadata，不需要把外置的大权重一起搬进内存。
    """
    import onnx

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在：{path}")
    model = onnx.load(str(path), load_external_data=False)
    return {p.key: p.value for p in model.metadata_props}


def write_metadata(
    path: str | Path,
    metadata: dict[str, Any],
    *,
    merge: bool = True,
) -> Path:
    """把 metadata 写进 ONNX 文件。

    值统一转成字符串：metadata_props 只接受字符串，这是 ONNX 的规定。

    merge=True 时同名键覆盖、其余保留。量化器产出的模型未必原样携带输入模型的
    metadata，所以量化之后要再写一次，否则端侧拿到 INT8 模型就还原不出关节角了。

    Args:
        merge: False 表示先清空原有 metadata 再写。
    """
    import onnx

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在：{path}")

    model = onnx.load(str(path))
    if not merge:
        del model.metadata_props[:]

    position = {p.key: i for i, p in enumerate(model.metadata_props)}
    for key, value in metadata.items():
        key, value = str(key), str(value)
        if key in position:
            model.metadata_props[position[key]].value = value
        else:
            entry = model.metadata_props.add()
            entry.key, entry.value = key, value

    onnx.save(model, str(path))
    return path


def parse_value(raw: str) -> Any:
    """metadata 里的值都是字符串。能当 JSON 解析的按 JSON 解析，其余原样返回。

    数值型字段（default_angles、action_scale、obs_mean）写入时统一 json.dumps，
    读回来也要统一解析，否则端侧会拿到 "[0.1, 0.2]" 这样的字符串。
    """
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def encode(metadata: dict[str, Any]) -> dict[str, str]:
    """把任意值编码成 metadata_props 能存的字符串。

    字符串原样保留，其余 json.dumps。与 `parse_value` 是一对，两者必须成对使用，
    否则会出现"写进去是 1，读出来是 '1'"这种难查的错位。
    """
    return {str(k): v if isinstance(v, str) else json.dumps(v) for k, v in metadata.items()}
