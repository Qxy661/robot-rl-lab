#!/usr/bin/env python
"""拉取 MuJoCo Menagerie 中本项目用到的机器人模型。

Menagerie 是一个完整的模型仓库，含几十种机器人，整个克隆下来数百 MB。
本项目只用到三个，所以用 git 的稀疏检出（sparse-checkout）只取需要的目录，
其余文件不进工作区、也不下载数据。

检出目标默认为仓库根下的 assets/menagerie/。模型不随本仓库分发：它们是
BSD-3 许可、可以再分发，但跟着上游走能拿到修正和新增，而且不会让本仓库
随时间膨胀。

    python scripts/fetch_assets.py            # 首次拉取
    python scripts/fetch_assets.py --update   # 更新到上游最新
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"

#: 与本项目内置三种形态对应的 Menagerie 子目录，与 RobotSpec.menagerie_dir 一致。
SUBDIRS = ("unitree_g1", "unitree_h1", "unitree_go2")

DEFAULT_DEST = Path(__file__).resolve().parent.parent / "assets" / "menagerie"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    """执行 git 命令，失败时把原始输出抛出来，不要吞掉。"""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(f"命令失败：{' '.join(cmd)}\n")
        sys.stderr.write(result.stderr)
        raise SystemExit(1)


def _check_git() -> None:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.stderr.write("找不到 git。请先安装 git 再运行本脚本。\n")
        raise SystemExit(1) from None


def fetch(dest: Path, update: bool = False) -> Path:
    dest = dest.resolve()

    if (dest / ".git").exists():
        if not update:
            print(f"模型目录已存在：{dest}")
            print("如需更新到上游最新版本，加 --update 重新运行。")
            return dest
        print(f"更新已有仓库：{dest}")
        _run(["git", "pull", "--ff-only"], cwd=dest)
        return dest

    if dest.exists() and any(dest.iterdir()):
        sys.stderr.write(
            f"目标目录 {dest} 已存在且非空，但不是 git 仓库。\n请换个 --dest，或先清空该目录。\n"
        )
        raise SystemExit(1)

    dest.parent.mkdir(parents=True, exist_ok=True)

    # --filter=blob:none 让克隆时只取提交历史不取文件内容，
    # --sparse 让工作区初始为空。两者叠加，只有下面指定的目录会被真正下载。
    print(f"从 {REPO_URL} 拉取模型（稀疏检出，仅 {len(SUBDIRS)} 个目录）...")
    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--sparse",
            "--depth",
            "1",
            REPO_URL,
            str(dest),
        ]
    )

    print(f"检出目录：{', '.join(SUBDIRS)}")
    _run(["git", "sparse-checkout", "set", *SUBDIRS], cwd=dest)

    missing = [d for d in SUBDIRS if not (dest / d).is_dir()]
    if missing:
        sys.stderr.write(f"以下目录未检出：{missing}\n")
        raise SystemExit(1)

    print(f"完成。模型位于 {dest}")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="拉取机器人模型（MuJoCo Menagerie 稀疏检出）")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"检出目标目录（默认 {DEFAULT_DEST}）",
    )
    parser.add_argument("--update", action="store_true", help="更新已有仓库到上游最新")
    args = parser.parse_args()

    _check_git()
    fetch(args.dest, update=args.update)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
