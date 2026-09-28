"""脚本层共用的装配代码：配置 → 环境 → 策略 → 训练器。

几个脚本都要做同一件事——把一份 YAML 变成能跑的对象图。这段逻辑放在这里
而不是塞进库里的原因，是它属于**应用组装**：库里各模块只认契约，谁把它们
拼起来、拼成什么规模，是脚本这一层决定的。将来想做超参搜索、多任务批量
启动，改的也是这一层，库不用动。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from robotrl.algorithms import ActorCritic, SACActor, make_trainer
from robotrl.algorithms.base import Policy, Trainer
from robotrl.assets.spec import RobotSpec, get_spec
from robotrl.configs.schema import Config
from robotrl.envs import env_name, list_envs, make
from robotrl.envs.base_env import BaseEnv
from robotrl.envs.vector_env import VecEnv, make_vec_env
from robotrl.utils.config import load_config

WIDTH = 78


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def header(title: str) -> None:
    print(f"\n{'=' * WIDTH}\n{title}\n{'=' * WIDTH}")


def info(text: str) -> None:
    print(f"  {text}")


def warn(text: str) -> None:
    print(f"  ! {text}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """加上所有脚本都有的配置参数。

    `--set a.b.c=value` 是主要工作方式：一份 preset 覆盖大部分场景，个别
    参数想试别的值时不必新建文件、也不必改代码。作用范围只限本次进程。
    """
    parser.add_argument("--config", type=Path, default=None, help="YAML 配置；不给则全用默认值")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆写配置项，可重复，如 --set train.num_envs=16",
    )


def load_cfg(args: argparse.Namespace) -> Config:
    """读配置并做一次基本检查。"""
    if args.config is not None and not Path(args.config).exists():
        raise SystemExit(f"配置文件不存在：{args.config}")
    cfg = load_config(args.config, overrides=args.overrides)
    return cfg


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------


def resolve_env_name(cfg: Config) -> str:
    """决定用哪个注册键构造环境。

    命名规则是「形态-任务」，但两类环境只注册了一半：
    - 所有形态共用的任务（velocity）只注册任务名，加形态不用重新注册；
    - 没有形态概念的环境（toy）只注册形态名。

    回落顺序是「完整名 → 形态名 → 任务名」，形态优先于任务。这不是随手定的
    顺序：形态级注册比任务级更具体，它说的是"这个形态的默认环境"。反过来
    排的话，toy 这种"形态自带任务"的环境会被同名的通用任务抢走——toy 的配置
    里 task 字段写的是 velocity，而恰好存在一个通用的 velocity 任务，先查
    任务名就会构造出一个 MuJoCo 环境去跑质点任务。
    """
    available = set(list_envs())
    full = env_name(cfg.env.robot, cfg.env.task)
    if full.lower() in available:
        return full
    if cfg.env.robot.lower() in available:
        return cfg.env.robot
    if cfg.env.task.lower() in available:
        return cfg.env.task
    raise SystemExit(
        f"没有可用的环境（robot={cfg.env.robot!r}, task={cfg.env.task!r}）；"
        f"已注册的有：{sorted(available)}"
    )


def build_envs(cfg: Config, num_envs: int | None = None) -> VecEnv:
    """按配置构造批式环境。

    返回的是 VecEnv 而不是实例列表：训练循环需要"一次下发一批动作"，而多进程
    后端只有在整批下发时才是真并行（见 envs/vector_env.py）。返回列表的话，
    这个关键区别就藏在了调用方手里。
    """
    n = num_envs if num_envs is not None else cfg.train.num_envs
    if n <= 0:
        raise SystemExit(f"环境数必须为正，得到 {n}")

    name = resolve_env_name(cfg)
    try:
        return make_vec_env(cfg, name, num_envs=n)
    except Exception as exc:  # noqa: BLE001 - 转成更可读的提示
        raise SystemExit(f"构造环境 {name} 失败：{exc}") from exc


def build_env(cfg: Config) -> BaseEnv:
    """构造单个环境。评估、回放、回测用这个——它们要的是干净的状态序列。"""
    return make(resolve_env_name(cfg), config=cfg)


def build_policy(cfg: Config, env: BaseEnv | VecEnv) -> Policy:
    """按算法名建对应的策略网络。

    PPO 用 actor-critic，SAC 用单独的随机策略网络加一对 Q 网络（Q 网络由
    训练器自己建）。两者输入维度都取自观测契约，因此换形态不用改这里。

    参数同时接受单个环境和批式环境——两者都暴露 obs_dim / action_dim /
    critic_obs_dim，策略只关心维度，不关心环境是怎么推进的。
    """
    algo = cfg.train.algo
    if algo == "ppo":
        return ActorCritic(
            env.obs_dim,
            env.action_dim,
            # 特权观测的宽度与策略观测不同，必须显式告诉网络，否则 critic
            # 会在第一层就因为维度不匹配而报错。
            critic_obs_dim=env.critic_obs_dim,
        )
    if algo == "sac":
        return SACActor(env.obs_dim, env.action_dim)
    raise SystemExit(f"未知算法 {algo!r}，只支持 ppo / sac")


def build_trainer(
    cfg: Config,
    envs: VecEnv,
    policy: Policy,
    *,
    run_dir: str | Path | None = None,
    **extra: Any,
) -> Trainer:
    """组装训练器。算法相关的超参配置按 cfg.train.algo 取对应那一份。"""
    run_dir = Path(run_dir if run_dir is not None else cfg.train.run_dir)
    return make_trainer(
        envs,
        policy,
        algo_name=cfg.train.algo,
        cfg=cfg.algo_config,
        run_dir=run_dir,
        seed=cfg.env.seed,
        device=cfg.train.device,
        log_interval=cfg.train.log_interval,
        eval_interval=cfg.train.eval_interval,
        save_interval=cfg.train.save_interval,
        num_threads=cfg.train.num_threads,
        **extra,
    )


def resolve_spec(cfg: Config) -> RobotSpec:
    return get_spec(cfg.env.robot)


# ---------------------------------------------------------------------------
# 产物
# ---------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: str | Path, data: Any) -> Path:
    """把结果写成 JSON。

    带 ensure_ascii=False：这些文件是给人看的，中文说明被转成 \\uXXXX 之后就
    没人愿意读了。
    """
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def latest_checkpoint(run_dir: str | Path) -> Path:
    """在 run 目录里找最近一次训练的权重。

    优先 best.pt：它按评估回报挑，比 latest.pt 更能代表策略的真实水平。
    latest.pt 只是"最后一步"，可能正处在一次失败的探索之后。
    """
    run_dir = Path(run_dir)
    for name in ("best.pt", "latest.pt", "final.pt"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate

    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"))
    if checkpoints:
        return checkpoints[-1]

    raise SystemExit(f"{run_dir} 里没有 checkpoint，先跑 scripts/train.py")


__all__: Sequence[str] = (
    "WIDTH",
    "add_config_args",
    "build_env",
    "build_envs",
    "build_policy",
    "build_trainer",
    "dump_json",
    "ensure_dir",
    "header",
    "info",
    "latest_checkpoint",
    "load_cfg",
    "resolve_env_name",
    "resolve_spec",
    "warn",
)
