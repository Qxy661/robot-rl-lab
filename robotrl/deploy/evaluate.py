"""量化回测：让 FP32 与 INT8 策略在同一批初始状态下跑完整回合。

这是本项目的核心产出。业界的机器人 RL 仓库大多停在"导出 ONNX"，量化之后
到底掉不掉点、掉多少，很少有人给出对照数据——因为"跑一遍看看"是不够的：
策略是闭环的，两次运行的初始状态不一样，回报差异里就混着初始状态差异，
结论没法用。

这里的做法是两条：

1. **相同种子**。FP32 与 INT8 各自从头跑同样的 episode 数，第 i 个回合用
   同一个种子复位，于是两边面对的是逐位相同的初始状态。轨迹会在某个时刻
   因为动作的微小差异分叉——这正是量化在闭环里的真实后果，也是回测要
   量出来的东西，不是要消除的噪声。

2. **同一批观测上的配对比较**。闭环回报里分不清"动作错了多少"和"动作错
   导致轨迹跑偏了多少"。所以另外把 FP32 轨迹上的观测原样喂给 INT8 引擎，
   得到一组配对动作，在这组配对上算逐维 MAE、最大偏差和分布 KL。这组数字
   与轨迹无关，是纯粹的策略输出偏差。

两类数字各有各的用处：配对偏差说明"量化把网络改了多少"，闭环回报说明
"这点改动在控制回路里放大了多少"。后者通常比前者大一两个量级。

分布 KL 用逐维直方图估计，两个分布共用同一组 bin 边界——边界不一致的话
两个直方图不可比，KL 就没有意义。直方图估计对 bin 数敏感，报告里会记下
bin 数，跨实验对比时要保持一致。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from robotrl.deploy.engine import PolicyEngine

#: 直方图 KL 的默认 bin 数。64 对 [-1, 1] 的动作区间来说每格宽约 0.03。
#: 再细分不会更准——直方图 KL 的偏差随 bin 数线性增长，随样本数反比下降。
DEFAULT_KL_BINS = 64

#: 每 bin 建议的最少样本数。低于这个比例就提示"这次 KL 估计不够可靠"。
SAMPLES_PER_BIN_MIN = 20

#: 直方图平滑项。KL 里 log(p/q) 遇到 q=0 会发散，而有限样本下总有些格子是空的。
_KL_EPS = 1e-9


# ---------------------------------------------------------------------------
# 回合展开
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    """一批回合的展开结果。

    observations 与 actions 是逐控制步平铺的（不是按回合分组的），因为
    后续的配对比较、KL 估计都把它们当成一个样本池用。
    """

    returns: np.ndarray
    lengths: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    @property
    def return_mean(self) -> float:
        return float(self.returns.mean()) if self.returns.size else float("nan")

    @property
    def return_std(self) -> float:
        return float(self.returns.std()) if self.returns.size else float("nan")

    @property
    def length_mean(self) -> float:
        return float(self.lengths.mean()) if self.lengths.size else float("nan")

    @property
    def num_steps(self) -> int:
        return int(self.observations.shape[0])

    def to_dict(self) -> dict[str, float]:
        return {
            "episodes": int(self.returns.size),
            "return_mean": self.return_mean,
            "return_std": self.return_std,
            "length_mean": self.length_mean,
            "steps": self.num_steps,
            "reward_mean": float(self.rewards.mean()) if self.rewards.size else float("nan"),
        }


def rollout(
    engine: PolicyEngine,
    env: Any,
    *,
    episodes: int = 20,
    seed: int = 12345,
    max_steps: int | None = None,
) -> RolloutResult:
    """用引擎跑若干个完整回合，记录回报、长度与逐步的观测/动作。

    回合 i 用 seed+i 复位。种子显式传给 env.reset 而不是设一次全局随机种子，
    是为了让"同一批初始状态"这件事可验证：换个引擎、换个机器，只要种子序列
    一样，起点就一样。

    Args:
        max_steps: 单回合步数上限，防止环境不终止时卡死。默认取环境的
            max_episode_steps。
    """
    if episodes <= 0:
        raise ValueError(f"回合数必须为正，得到 {episodes}")

    cap = max_steps if max_steps is not None else int(getattr(env, "max_episode_steps", 1000))
    returns: list[float] = []
    lengths: list[int] = []
    rewards: list[float] = []
    obs_log: list[np.ndarray] = []
    act_log: list[np.ndarray] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        ep_return, ep_len = 0.0, 0
        while not done and ep_len < cap:
            observation = np.asarray(obs.policy, dtype=np.float32)
            action = np.asarray(engine.infer(observation), dtype=np.float32)

            # 观测与动作在 step 之前记录：配对比较要求"面对这个观测时输出了
            # 什么动作"，若在 step 之后记录，观测已经被环境推进过一步了。
            obs_log.append(observation)
            act_log.append(action)

            result = env.step(action)
            obs = result.obs
            ep_return += float(result.reward)
            rewards.append(float(result.reward))
            ep_len += 1
            done = bool(result.terminated or result.truncated)

        returns.append(ep_return)
        lengths.append(ep_len)

    return RolloutResult(
        returns=np.asarray(returns, dtype=np.float64),
        lengths=np.asarray(lengths, dtype=np.int64),
        observations=np.stack(obs_log).astype(np.float32),
        actions=np.stack(act_log).astype(np.float32),
        rewards=np.asarray(rewards, dtype=np.float64),
    )


def collect_observations(
    env: Any,
    *,
    steps: int = 512,
    seed: int = 0,
    act: Callable[[np.ndarray], np.ndarray] | None = None,
    engine: PolicyEngine | None = None,
) -> np.ndarray:
    """从环境里采一批真实观测，返回 (steps, obs_dim)。

    真实观测与随机观测不是一回事：策略只会把它自己引到的那片状态空间走一遍，
    量化要在这片区域上评估才有意义。校准集（静态量化）和误差归因都用这里的
    输出。

    act 与 engine 二选一：给 engine 就用策略本身驱动（状态分布与部署时一致，
    这是更可取的做法）；两者都不给时用均匀随机动作，采到的是"环境能到的地方"
    而不是"策略会去的地方"，只适合冒烟测试。
    """
    if steps <= 0:
        raise ValueError(f"采样步数必须为正，得到 {steps}")

    def _default_act(obs: np.ndarray) -> np.ndarray:
        return env.np_random.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)

    if engine is not None:

        def _act(obs: np.ndarray) -> np.ndarray:
            return np.asarray(engine.infer(obs), dtype=np.float32)
    else:
        _act = act or _default_act

    obs, _ = env.reset(seed=seed)
    collected: list[np.ndarray] = []
    while len(collected) < steps:
        observation = np.asarray(obs.policy, dtype=np.float32)
        collected.append(observation)
        result = env.step(_act(observation))
        if result.terminated or result.truncated:
            # 回合结束就复位并换种子，避免总在同一个初始状态附近打转
            obs, _ = env.reset(seed=seed + len(collected))
        else:
            obs = result.obs

    return np.asarray(collected[:steps], dtype=np.float32)


# ---------------------------------------------------------------------------
# 动作偏差
# ---------------------------------------------------------------------------


@dataclass
class ActionDeviation:
    """两组配对动作之间的偏差。逐维统计，因为不同关节的量纲和敏感度不同。"""

    mae: float
    rmse: float
    max_dev: float
    mae_per_dim: np.ndarray
    max_dev_per_dim: np.ndarray

    def worst_dims(self, k: int = 3) -> list[tuple[int, float]]:
        """偏差最大的几个维度。维度号即动作顺序，对应 spec.controlled_joints。"""
        order = np.argsort(self.max_dev_per_dim)[::-1][:k]
        return [(int(i), float(self.max_dev_per_dim[i])) for i in order]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "max_dev": self.max_dev,
            "mae_per_dim": self.mae_per_dim.tolist(),
            "max_dev_per_dim": self.max_dev_per_dim.tolist(),
            "worst_dims": self.worst_dims(),
        }


def action_deviation(reference: np.ndarray, candidate: np.ndarray) -> ActionDeviation:
    """逐元素比较两组动作。形状不同直接报错——静默广播会把结论算错。"""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"动作形状不一致：{a.shape} vs {b.shape}")

    diff = np.abs(a - b)
    return ActionDeviation(
        mae=float(diff.mean()),
        rmse=float(np.sqrt(np.mean(diff**2))),
        max_dev=float(diff.max()),
        mae_per_dim=diff.mean(axis=0),
        max_dev_per_dim=diff.max(axis=0),
    )


# ---------------------------------------------------------------------------
# 动作分布 KL
# ---------------------------------------------------------------------------


@dataclass
class KLEstimate:
    """逐维 KL(参考 ‖ 候选) 的估计值，连同估计偏差一起给出。

    直方图 KL 是有偏的：两个分布完全相同，有限样本下每个格子的计数也不会
    相等，log 比值自然非零。偏差大致按（格子数-1)/样本数 增长——样本少的时候
    它能比真实 KL 还大，一个恒等映射都能测出可观的"分布偏移"。

    所以这里报的是扣掉偏差之后的值，同时把偏差量与样本数一并留下，
    让读的人自己判断这个数可不可信。
    """

    per_dim: np.ndarray
    bias_per_dim: np.ndarray
    bins: int
    n_samples: int

    @property
    def mean(self) -> float:
        return float(self.per_dim.mean())

    @property
    def max(self) -> float:
        return float(self.per_dim.max())

    @property
    def bias_mean(self) -> float:
        return float(self.bias_per_dim.mean())

    @property
    def underpowered(self) -> bool:
        """样本数是否少到 KL 估计不足为凭。"""
        return self.n_samples < SAMPLES_PER_BIN_MIN * self.bins

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_dim": self.per_dim.tolist(),
            "bias_per_dim": self.bias_per_dim.tolist(),
            "bins": self.bins,
            "n_samples": self.n_samples,
            "mean": self.mean,
            "max": self.max,
            "bias_mean": self.bias_mean,
            "underpowered": self.underpowered,
        }


def action_kl_divergence(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    bins: int = DEFAULT_KL_BINS,
    bias_correction: bool = True,
) -> KLEstimate:
    """逐维估计 KL(参考 ‖ 候选)。

    用直方图而不是核密度估计：动作维度高、样本量有限，直方图在相同样本下
    方差更小，而且 bin 边界显式可控，两个分布能严格共用一套边界。

    两个分布共用边界这件事是必须的——各自用自己的取值范围分箱，得到的两组
    概率在同一个格子上指的就不是同一段动作区间，KL 会变成一个没有意义的数。

    Args:
        bins: 每维的直方图格数。**跨实验比较时这个值必须一致**，否则 KL 不可比：
        bin 越细，同一个分布对的 KL 估计越大。
        bias_correction: 是否扣掉有限样本偏差。默认开启——不扣的话，量化前后
            完全相同的策略也会测出一个非零的"分布偏移"，阈值就没法定了。
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"动作形状不一致：{a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise ValueError(f"动作应为 (N, action_dim)，实际形状 {a.shape}")
    if a.shape[0] < 2:
        raise ValueError("KL 估计至少需要两个样本")

    action_dim = a.shape[1]
    n_samples = a.shape[0]
    values = np.zeros(action_dim, dtype=np.float64)
    biases = np.zeros(action_dim, dtype=np.float64)

    for dim in range(action_dim):
        lo = min(a[:, dim].min(), b[:, dim].min())
        hi = max(a[:, dim].max(), b[:, dim].max())
        if hi <= lo:
            # 该维动作恒定（例如输出饱和在 ±1），分布退化成单点，KL 无定义。
            # 记 0 而不是 nan，逐维数组里能直接看出是哪一维。
            continue

        edges = np.linspace(lo, hi, bins + 1)
        # 逐维处理而不是 histogramdd：这样才能给出"哪一维偏得最多"
        raw = _kl(_hist_probs(a[:, dim], edges), _hist_probs(b[:, dim], edges))

        bias = _sample_size_floor(a[:, dim], edges) / 2.0 if bias_correction else 0.0
        biases[dim] = bias
        values[dim] = max(raw - bias, 0.0)

    return KLEstimate(per_dim=values, bias_per_dim=biases, bins=bins, n_samples=n_samples)


def _hist_probs(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """按给定边界统计直方图并归一化成概率。平滑项避免空格子让 log 发散。"""
    counts = np.histogram(values, bins=edges)[0].astype(np.float64) + _KL_EPS
    return counts / counts.sum()


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def _sample_size_floor(values: np.ndarray, edges: np.ndarray) -> float:
    """估计有限样本带来的 KL 偏差地板。

    做法是把参考样本按奇偶下标分成两半（不是前后对半：回测数据是按时序排的，
    前后两半可能对应策略的不同阶段，分出来的分布会带着真实差异），各自估一个
    分布，它们之间的 KL 就是"同一个分布"下这个估计器会给出的非零读数。

    偏差与样本数成反比，半样本的地板是完整样本偏差的两倍，所以调用方要
    除以 2 再用。用测出来的地板而不是套 (K-1)/(2N) 这类公式，是因为格子数 K
    在数据稀疏时本身也是估计出来的，实测更稳。
    """
    even, odd = values[0::2], values[1::2]
    if even.size < 2 or odd.size < 2:
        return 0.0
    return _kl(_hist_probs(even, edges), _hist_probs(odd, edges))


# ---------------------------------------------------------------------------
# 回测报告
# ---------------------------------------------------------------------------


@dataclass
class BacktestReport:
    """量化前后的对照数据，可直接生成 README 的表格。"""

    episodes: int
    seed: int
    fp32: RolloutResult
    int8: RolloutResult
    deviation: ActionDeviation
    kl: KLEstimate
    kl_threshold: float

    # ---- 派生指标 ----

    @property
    def return_delta(self) -> float:
        """INT8 减 FP32 的平均回报差。负数表示量化掉点。"""
        return self.int8.return_mean - self.fp32.return_mean

    @property
    def return_delta_ratio(self) -> float:
        """相对掉点比例。FP32 回报接近 0 时这个数没有意义，此时返回 nan。"""
        base = self.fp32.return_mean
        if abs(base) < 1e-8:
            return float("nan")
        return self.return_delta / abs(base)

    @property
    def length_delta(self) -> float:
        return self.int8.length_mean - self.fp32.length_mean

    @property
    def kl_mean(self) -> float:
        return self.kl.mean

    @property
    def kl_max(self) -> float:
        """偏得最狠的那一维。动作维之间差异大，只看均值会漏掉局部崩坏。"""
        return self.kl.max

    @property
    def kl_ok(self) -> bool:
        return self.kl_mean <= self.kl_threshold

    @property
    def kl_reliable(self) -> bool:
        """KL 的样本量是否够。不够时这个数只能当趋势看，不能当判据。"""
        return not self.kl.underpowered

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "seed": self.seed,
            "fp32": self.fp32.to_dict(),
            "int8": self.int8.to_dict(),
            "return_delta": self.return_delta,
            "return_delta_ratio": self.return_delta_ratio,
            "length_delta": self.length_delta,
            "action_mae": self.deviation.mae,
            "action_rmse": self.deviation.rmse,
            "action_max_dev": self.deviation.max_dev,
            "action_mae_per_dim": self.deviation.mae_per_dim.tolist(),
            "action_max_dev_per_dim": self.deviation.max_dev_per_dim.tolist(),
            "worst_dims": self.deviation.worst_dims(),
            "kl": self.kl.to_dict(),
            "kl_threshold": self.kl_threshold,
            "kl_ok": self.kl_ok,
        }

    def markdown_table(self) -> str:
        """生成 README 里的对照表。数字带单位与变化量，便于直接引用。"""
        rows = [
            (
                "平均回报",
                f"{self.fp32.return_mean:.3f} ± {self.fp32.return_std:.3f}",
                f"{self.int8.return_mean:.3f} ± {self.int8.return_std:.3f}",
                f"{self.return_delta:+.3f} ({self.return_delta_ratio:+.2%})",
            ),
            (
                "平均回合长度",
                f"{self.fp32.length_mean:.1f}",
                f"{self.int8.length_mean:.1f}",
                f"{self.length_delta:+.1f}",
            ),
            ("逐维动作 MAE", "—", f"{self.deviation.mae:.4f}", ""),
            ("动作最大偏差", "—", f"{self.deviation.max_dev:.4f}", ""),
            ("动作 RMSE", "—", f"{self.deviation.rmse:.4f}", ""),
        ]
        kl_label = f"动作分布 KL（{self.kl.bins} bins，{self.kl.n_samples} 样本）"
        rows.append((kl_label, "—", f"{self.kl_mean:.4f}（均值）", ""))
        rows.append(("", "—", f"{self.kl_max:.4f}（最大维）", ""))
        rows.append(("KL 估计偏差", "—", f"{self.kl.bias_mean:.4f}", "已从上面的值里扣除"))

        lines = [
            f"回测条件：{self.episodes} 回合，起始种子 {self.seed}，两个策略逐回合同种子复位",
            "",
            "| 指标 | FP32 | INT8 | 变化 |",
            "| --- | --- | --- | --- |",
        ]
        lines.extend(f"| {name} | {a} | {b} | {delta} |" for name, a, b, delta in rows)
        return "\n".join(lines)

    def summary(self) -> str:
        verdict = "可接受" if self.kl_ok else f"KL 超过阈值 {self.kl_threshold}"
        caution = "" if self.kl_reliable else "（样本偏少，KL 只能看趋势）"
        return (
            f"量化回测（{self.episodes} 回合，种子 {self.seed}）："
            f"平均回报 {self.fp32.return_mean:.3f} → {self.int8.return_mean:.3f}"
            f"（{self.return_delta:+.3f}，{self.return_delta_ratio:+.2%}）；"
            f"动作 MAE {self.deviation.mae:.4f}，最大偏差 {self.deviation.max_dev:.4f}；"
            f"KL 均值 {self.kl_mean:.4f}，最大维 {self.kl_max:.4f}"
            f"（已扣偏差 {self.kl.bias_mean:.4f}{caution}）——{verdict}"
        )


def evaluate_engine(
    engine: PolicyEngine,
    env: Any,
    *,
    episodes: int = 20,
    seed: int = 12345,
) -> dict[str, float]:
    """单个引擎的评估指标。格式与算法层 Trainer.evaluate() 对齐，便于并排比较。"""
    result = rollout(engine, env, episodes=episodes, seed=seed)
    return {
        "eval/return_mean": result.return_mean,
        "eval/return_std": result.return_std,
        "eval/episode_length_mean": result.length_mean,
        "eval/episodes": float(result.returns.size),
    }


def backtest(
    fp32_engine: PolicyEngine,
    int8_engine: PolicyEngine,
    env: Any,
    *,
    episodes: int = 20,
    seed: int = 12345,
    kl_bins: int = DEFAULT_KL_BINS,
    kl_threshold: float = 0.1,
) -> BacktestReport:
    """量化回测主流程。

    顺序是固定的：先各自闭环跑完（同种子），再做同一批观测上的配对比较。

    Args:
        episodes: 各自跑的回合数。太少的话回报差异会被回合间的随机性淹没；
            足式任务上一般 20 个回合起。
        seed: 起始种子。回合 i 用 seed+i 复位，两个引擎用同一套。
        kl_threshold: 动作分布 KL 的容忍上限，超过就认为量化掉点过多。

    Returns:
        BacktestReport，含对照表生成能力。
    """
    if (
        fp32_engine.action_dim
        and int8_engine.action_dim
        and fp32_engine.action_dim != int8_engine.action_dim
    ):
        raise ValueError(
            f"两个引擎的动作维度不同（{fp32_engine.action_dim} vs "
            f"{int8_engine.action_dim}），不是同一个策略的两份实现"
        )

    fp32 = rollout(fp32_engine, env, episodes=episodes, seed=seed)
    int8 = rollout(int8_engine, env, episodes=episodes, seed=seed)

    # 配对比较：把 FP32 轨迹上的观测原样喂给两个引擎。这一组动作不涉及轨迹
    # 分叉，量出来的纯粹是网络输出的偏差。
    paired_fp32 = fp32_engine.infer_batch(fp32.observations)
    paired_int8 = int8_engine.infer_batch(fp32.observations)

    deviation = action_deviation(paired_fp32, paired_int8)
    kl = action_kl_divergence(paired_fp32, paired_int8, bins=kl_bins)

    return BacktestReport(
        episodes=episodes,
        seed=seed,
        fp32=fp32,
        int8=int8,
        deviation=deviation,
        kl=kl,
        kl_threshold=kl_threshold,
    )
