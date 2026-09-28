"""批式环境：把 N 个环境实例包成一个"一次推进一步"的对象。

为什么必须批式
--------------
训练循环要的是"给 N 个动作、拿 N 个结果"。顺序地一个个调 `env.step()`，
多进程起不到任何作用——每次调用都要等这一轮 IPC 返回，N 次等待串起来，
总时间与单进程相同，还多付了通信开销。只有**先把 N 个动作全部发出去、
再统一收结果**，N 个进程才真的在同时干活。

这一点容易被忽略，因为"开了多进程"看起来就该快。实测数字：G1 的单步
约 1.5 ms，64 个环境顺序推进，一个控制步要 96 ms；两千万步的训练因此要
8 小时以上。批式分发把它压到 40 分钟量级。

为什么 MuJoCo 只能靠多进程
--------------------------
MuJoCo 的单步是纯 C 计算，但环境推进的大头在 Python 侧：读传感器、算奖励、
拼观测、判终止。这些是 Python 字节码，受 GIL 约束，多线程拿不到加速。
真正能并行的只有进程。这也是配置里 vec_backend 存在的理由。

子系统怎么切分
--------------
进程数取 min(环境数, CPU 核数)，每个进程持有环境数/进程数 个实例。不按
"一个环境一个进程"开，是因为每个进程要持有一份完整的 MjModel（几 MB 到
几十 MB），64 个进程光模型就吃掉大量内存，进程启动与 IPC 的固定开销也会
反过来压过收益。一个进程跑几个环境，批内是顺序的，批间才是并行——这个
粒度足够，代价小得多。
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from robotrl.contracts import ObsContract
from robotrl.envs.base_env import BaseEnv, Obs, StepResult

#: 单次 IPC 最长等待时间（秒）。不给上限的话，子进程若因段错误静默死掉，
#: 父进程会永远挂在 recv 上，连报错都没有。
_DEFAULT_TIMEOUT = 300.0


class VecEnv(ABC):
    """批式环境接口。

    对外暴露的观测/动作维度取自第一个环境，同构假设是成立的：同一份配置
    构造出来的实例维度必然一致。异构环境需要各自独立的观测契约，那是另一
    类问题（多任务联合训练），不在这个接口的范围内。
    """

    def __init__(self, num_envs: int) -> None:
        if num_envs <= 0:
            raise ValueError(f"环境数必须为正，得到 {num_envs}")
        self.num_envs = num_envs

    # ---- 维度与契约 ----

    @property
    @abstractmethod
    def spec(self) -> Any:
        """形态定义。导出 ONNX 时要用它写 default_angles / action_scale。"""

    @property
    @abstractmethod
    def obs_contract(self) -> ObsContract: ...

    @property
    @abstractmethod
    def critic_obs_contract(self) -> ObsContract: ...

    @property
    def obs_dim(self) -> int:
        return self.obs_contract.total_dim

    @property
    def critic_obs_dim(self) -> int:
        return self.critic_obs_contract.total_dim

    @property
    def action_dim(self) -> int:
        return self.spec.n_dof

    # ---- 推进 ----

    @abstractmethod
    def reset(self, seeds: list[int] | None = None) -> list[Obs]:
        """复位全部环境，返回各自的观测。

        每个环境给不同的种子是必须的：种子相同的话所有环境走同一条轨迹，
        并行等于白开，而且 PPO 的批次里全是重复样本。
        """

    @abstractmethod
    def step_batch(self, actions: np.ndarray) -> list[StepResult]:
        """下发一批动作，返回一批结果。actions 形状 (num_envs, action_dim)。"""

    @abstractmethod
    def close(self) -> None: ...

    # ---- 评估用的单环境 ----

    @property
    @abstractmethod
    def eval_env(self) -> BaseEnv:
        """供 Trainer.evaluate() 使用的单个环境。

        评估要的是"干净、可复现、逐回合"的状态序列，批式接口不适合承载它
        （评估时只有一个环境在跑，批式的那套分发反而绕）。所以单独给一个。
        """

    def __enter__(self) -> VecEnv:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# 串行：把一批本地实例包装成批式接口
# ---------------------------------------------------------------------------


class SerialVecEnv(VecEnv):
    """把一批本地环境实例包成批式接口。

    它不带来任何加速，存在的意义是**让训练循环只有一条代码路径**。否则
    "单进程"和"多进程"要走两套推进逻辑，两套都要维护、都要测，而它们的
    行为必须逐位一致——这种事一旦分叉，就会出现"换后端导致结果不同"的
    问题，排查成本极高。
    """

    def __init__(self, envs: list[BaseEnv]) -> None:
        if not envs:
            raise ValueError("环境列表不能为空")
        super().__init__(len(envs))
        self.envs = list(envs)

    @property
    def spec(self) -> Any:
        return self.envs[0].spec

    @property
    def obs_contract(self) -> ObsContract:
        return self.envs[0].obs_contract

    @property
    def critic_obs_contract(self) -> ObsContract:
        return self.envs[0].critic_obs_contract

    @property
    def eval_env(self) -> BaseEnv:
        # 与训练用的是同一个实例，行为与改造前一致。评估会改变环境状态，
        # 但那本来就是训练循环里"评估环境"的语义。
        return self.envs[0]

    def reset(self, seeds: list[int] | None = None) -> list[Obs]:
        if seeds is None:
            return [env.reset()[0] for env in self.envs]
        if len(seeds) != self.num_envs:
            raise ValueError(f"种子数 {len(seeds)} 与环境数 {self.num_envs} 不一致")
        return [env.reset(seed=s)[0] for env, s in zip(self.envs, seeds, strict=True)]

    def step_batch(self, actions: np.ndarray) -> list[StepResult]:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"动作形状应为 {(self.num_envs, self.action_dim)}，实际 {actions.shape}"
            )
        return [env.step(a) for env, a in zip(self.envs, actions, strict=True)]

    def close(self) -> None:
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# 多进程
# ---------------------------------------------------------------------------


def _worker_main(conn: Any, cfg: Any, name: str, count: int) -> None:
    """子进程主循环：持有 count 个环境，按指令推进。

    这个函数必须是模块级可导入的（不能是闭包或实例方法）：Windows 上多进程
    用 spawn 启动，子进程要重新 import 本模块再按名字找到它。闭包在那边
    反序列化不出来。
    """
    from robotrl.envs import make

    try:
        envs = [make(name, config=cfg) for _ in range(count)]
    except Exception as exc:  # noqa: BLE001 - 启动失败要让父进程看见
        conn.send(("error", f"子进程构造环境失败：{exc!r}"))
        conn.close()
        return

    try:
        first = envs[0]
        conn.send(("ready", (first.obs_dim, first.critic_obs_dim, first.action_dim)))
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                conn.send(("ok", [e.reset(seed=s)[0] for e, s in zip(envs, payload, strict=True)]))
            elif cmd == "step":
                conn.send(("ok", [e.step(a) for e, a in zip(envs, payload, strict=True)]))
            elif cmd == "close":
                conn.send(("ok", None))
                break
    except (EOFError, BrokenPipeError):
        # 父进程先退出了。这不是错误，安静收摊。
        pass
    finally:
        for e in envs:
            e.close()
        conn.close()


def split_counts(num_envs: int, num_workers: int) -> list[int]:
    """把 num_envs 个环境分给 num_workers 个进程，返回每个进程分到几个。

    余数给前几个进程：每个进程内部是顺序推进，各进程的环境数差不超过 1，
    "批间并行"的负载才均衡——否则最慢的那个进程决定每个控制步的耗时。

    进程数超过环境数时，多出来的进程没有环境可分，直接丢掉。空转的进程
    要白白 import 一遍依赖、持一份解释器内存，是纯亏损。
    """
    if num_envs <= 0:
        raise ValueError(f"环境数必须为正，得到 {num_envs}")
    if num_workers <= 0:
        raise ValueError(f"进程数必须为正，得到 {num_workers}")
    effective = min(num_workers, num_envs)
    base, extra = divmod(num_envs, effective)
    return [base + (1 if i < extra else 0) for i in range(effective)]


class _WorkerGroup:
    """一组子进程，负责分发与收集。

    构造与启动分成两步：`__init__` 只算好怎么分，`start()` 才真正开进程。
    这样"没启动过"的组是个合法状态——出错时 close() 要在任何阶段都能安全
    调用，而 close() 的清理逻辑不该依赖子进程是否已经起来。
    """

    def __init__(self, cfg: Any, name: str, num_envs: int, num_workers: int, seed: int) -> None:
        self.counts = split_counts(num_envs, num_workers)
        self.num_workers = len(self.counts)

        self._cfg = cfg
        self._name = name
        self._seed = seed

        self._conns: list[Any] = []
        self._procs: list[Any] = []
        # 每个环境在全局的起始序号，用来给种子错开且与串行路径一致。
        self._offsets: list[int] = []
        offset = 0
        for count in self.counts:
            self._offsets.append(offset)
            offset += count

        self.obs_dim = 0
        self.critic_obs_dim = 0
        self.action_dim = 0
        self._started = False

    def start(self) -> None:
        """拉起子进程，并确认它们报告的环境维度一致。"""
        if self._started:
            raise RuntimeError("这组子进程已经启动过，重复启动会多开一倍进程")

        ctx = mp.get_context("spawn")
        try:
            for count in self.counts:
                parent_conn, child_conn = ctx.Pipe(duplex=True)
                proc = ctx.Process(
                    target=_worker_main,
                    args=(child_conn, self._cfg, self._name, count),
                    daemon=True,
                )
                proc.start()
                child_conn.close()  # 父进程这一侧要关掉子端的句柄，否则收不到 EOF
                self._conns.append(parent_conn)
                self._procs.append(proc)

            dims = []
            for conn in self._conns:
                status, payload = self._recv(conn)
                if status == "error":
                    raise RuntimeError(payload)
                dims.append(payload)
        except Exception:
            self.close()
            raise

        # 各进程报告的维度必须一致，否则拼出来的批数据是错的。与其等训练
        # 中途在某个矩阵乘法上炸掉，不如在启动阶段就说清楚。
        if len(set(map(tuple, dims))) != 1:
            self.close()
            raise RuntimeError(f"各子进程的环境维度不一致：{dims}")
        self.obs_dim, self.critic_obs_dim, self.action_dim = dims[0]
        self._started = True

    def _recv(self, conn: Any, timeout: float = _DEFAULT_TIMEOUT) -> tuple[str, Any]:
        if not conn.poll(timeout):
            raise TimeoutError(
                f"等待子进程超过 {timeout:.0f} 秒。多半是子进程已崩溃——"
                "先确认单个环境能否在本进程里正常构造，再看内存是否够用。"
            )
        return conn.recv()

    def _require_started(self) -> None:
        """没启动就分发会静默返回空结果——宁可直接报错。"""
        if not self._started:
            raise RuntimeError("子进程尚未启动，先调用 start()")

    def reset(self, seeds: list[int]) -> list[Obs]:
        self._require_started()
        # 先把所有指令发出去，再统一收。这一步是并行成立的前提，见模块文档。
        for conn, off, count in zip(self._conns, self._offsets, self.counts, strict=True):
            conn.send(("reset", seeds[off : off + count]))
        out: list[Obs] = []
        for conn in self._conns:
            status, payload = self._recv(conn)
            if status == "error":
                raise RuntimeError(payload)
            out.extend(payload)
        return out

    def step(self, actions: np.ndarray) -> list[StepResult]:
        self._require_started()
        for conn, off, count in zip(self._conns, self._offsets, self.counts, strict=True):
            conn.send(("step", actions[off : off + count]))
        out: list[StepResult] = []
        for conn in self._conns:
            status, payload = self._recv(conn)
            if status == "error":
                raise RuntimeError(payload)
            out.extend(payload)
        return out

    def close(self) -> None:
        # 子进程可能已经自己退了，这时候 send/close 会抛 BrokenPipe。关闭流程
        # 本来就要把每条路都走完，单个连接出错不该中断其余的清理。
        for conn in self._conns:
            with contextlib.suppress(BrokenPipeError, OSError):
                conn.send(("close", None))
        for proc in self._procs:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
        for conn in self._conns:
            with contextlib.suppress(OSError):
                conn.close()
        self._conns.clear()
        self._procs.clear()


class SubprocessVecEnv(VecEnv):
    """多进程推进的环境批。

    构造参数里带一个 `eval_env` 用的本地实例：Trainer.evaluate() 需要单个
    环境做逐回合评估，而批式接口不适合承载它。多开一个环境实例的代价
    （一份 MjModel）远小于给批式接口硬塞一个"单步模式"的复杂度。
    """

    def __init__(
        self,
        cfg: Any,
        name: str,
        *,
        num_envs: int,
        seed: int = 0,
        num_workers: int | None = None,
    ) -> None:
        super().__init__(num_envs)
        if num_workers is None:
            num_workers = min(num_envs, mp.cpu_count() or 1)
        if num_workers <= 0:
            raise ValueError(f"进程数必须为正，得到 {num_workers}")

        self._cfg = cfg
        self._name = name
        self._seed = seed
        self.group = _WorkerGroup(cfg, name, num_envs, num_workers, seed)

        # 评估环境在本进程里另建一份。不要在子进程里远程驱动它——评估是
        # 逐回合、带随机性的，往返一轮一步会让 1000 步的回合慢得离谱。
        from robotrl.envs import make

        try:
            self._eval_env = make(name, config=cfg)
        except Exception:
            self.group.close()
            raise

        # 先建好评估环境再拉子进程：本地这一步失败得最快，放在前面就不会留下
        # "子进程已经起来、主进程却正在报错"的中间状态。
        try:
            self.group.start()
        except Exception:
            self._eval_env.close()
            raise

        self.cfg = cfg
        self.env_name = name

    @property
    def spec(self) -> Any:
        return self._eval_env.spec

    @property
    def obs_contract(self) -> ObsContract:
        return self._eval_env.obs_contract

    @property
    def critic_obs_contract(self) -> ObsContract:
        return self._eval_env.critic_obs_contract

    @property
    def eval_env(self) -> BaseEnv:
        return self._eval_env

    @property
    def num_workers(self) -> int:
        return self.group.num_workers

    def reset(self, seeds: list[int] | None = None) -> list[Obs]:
        if seeds is None:
            seeds = [self._seed + i for i in range(self.num_envs)]
        if len(seeds) != self.num_envs:
            raise ValueError(f"种子数 {len(seeds)} 与环境数 {self.num_envs} 不一致")
        return self.group.reset(seeds)

    def step_batch(self, actions: np.ndarray) -> list[StepResult]:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"动作形状应为 {(self.num_envs, self.action_dim)}，实际 {actions.shape}"
            )
        return self.group.step(actions)

    def close(self) -> None:
        self.group.close()
        self._eval_env.close()

    def __repr__(self) -> str:
        return (
            f"SubprocessVecEnv({self.num_envs} 环境 / {self.num_workers} 进程，"
            f"观测 {self.obs_dim} → 动作 {self.action_dim})"
        )


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------


def default_num_workers(num_envs: int) -> int:
    return min(num_envs, mp.cpu_count() or 1)


def make_vec_env(
    cfg: Any,
    name: str,
    *,
    num_envs: int | None = None,
    seed: int | None = None,
    backend: str | None = None,
) -> VecEnv:
    """按配置构造批式环境，自动挑后端。

    串行与多进程之间不做"自动加速"式的猜测：环境数不多时多进程的启动开销
    （每个进程要 import torch、编译 MjModel）比省下的时间还多，所以由配置
    明说，而不是让运行时去猜。
    """
    from robotrl.envs import make

    n = num_envs if num_envs is not None else cfg.train.num_envs
    s = seed if seed is not None else cfg.env.seed
    kind = backend if backend is not None else cfg.train.vec_backend

    if kind == "subprocess":
        return SubprocessVecEnv(cfg, name, num_envs=n, seed=s)
    if kind == "serial":
        envs = [make(name, config=cfg) for _ in range(n)]
        return SerialVecEnv(envs)
    raise ValueError(f"未知向量化后端 {kind!r}，只支持 serial / subprocess")


__all__ = [
    "SerialVecEnv",
    "SubprocessVecEnv",
    "VecEnv",
    "default_num_workers",
    "make_vec_env",
    "split_counts",
]
