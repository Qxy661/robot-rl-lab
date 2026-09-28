"""可复现评估：固定种子集合、多回合重复、逐段观测量程统计。

评估的唯一目的是让两次实验可比。做到这点要守住三件事：

1. **种子固定且成组**。单一种子跑出来的均值本身就带运气成分——换一批初始
   状态，均值差出几个身位是常事。最终评估用一组固定种子，报的是"换一批
   初始条件策略还稳不稳"，而不是"某一次跑得有多好"。
2. **训练期评估与最终评估分开**。训练中每隔若干迭代看一眼，用单一种子、
   少量回合，只为发现"学崩了"这种量级的问题；下结论用的是最终评估，
   种子集合写死在模块里，谁在什么时候跑都是同一批。
3. **观测按契约拆段统计**。只看总回报看不出策略是怎么拿到这个分的：把观测
   按段拆开算均值/方差/饱和比例，才能看出策略是不是把某一维推到了极端
   （例如 last_action 长期贴 ±1，那就是执行器饱和的典型症状）。

动作一律走确定性路径。随机采样会把探索噪声掺进回报，两次评估的差异就
说不清是策略变了还是采样运气不同。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from robotrl.algorithms.base import Policy
from robotrl.contracts import ObsContract
from robotrl.envs.base_env import BaseEnv

# ---------------------------------------------------------------------------
# 阈值表：判断"某一维是不是被推到了极端"需要先知道它的正常量程
# ---------------------------------------------------------------------------

#: 各观测段的典型量程。样本绝对值超过它就是"贴边"。键与观测契约的段名一致。
#: 表外的段没有先验量程，不做饱和统计，但 max_abs 仍然给出——量程可以自己判，
#: 最大值是客观的。要换一套量程就传 saturation_thresholds。
_SATURATION_THRESHOLDS: dict[str, float] = {
    "lin_vel": 2.0,        # m/s。常规行走到小跑的量级，再高基本是失控
    "ang_vel": 3.0,        # rad/s
    "proj_gravity": 0.9,   # 单位向量分量。接近 1 表示机身几乎躺平
    "cmd": 1.0,            # 指令采样范围，超出说明指令本身有问题
    "dof_pos": 1.0,        # rad。偏离默认姿态 1 弧度已经很夸张
    "dof_vel": 10.0,       # rad/s
    "last_action": 1.0,    # 动作契约定死在 [-1, 1]，贴边即饱和
    "pos": 5.0,            # 与 ToyVelocityEnv._BOUND 一致
    "vel": 2.0,
    "damping": 1.0,        # toy 环境的特权段，量程本来就在 1 附近
}


# ---------------------------------------------------------------------------
# 评估协议
# ---------------------------------------------------------------------------

#: 最终评估的固定种子集合。写死在这里而不是每次现取，是为了让"上周的 480 分"
#: 和"这周的 500 分"真的可比——换了种子，两个数之间就混进了初始状态的差异。
FINAL_SEEDS: tuple[int, ...] = (12345, 12346, 12347, 12348, 12349)

#: 训练期评估的种子。只用一组，图的就是便宜，它要在训练循环里反复跑。
TRAIN_SEED: int = 12345


@dataclass(frozen=True)
class EvalProtocol:
    """一次评估的完整约定。

    Attributes:
        name: 协议名，会写进报告，用来区分"训练中看的数"和"最终结论用的数"。
        seeds: 种子集合。多组种子是"报告 mean ± std"的前提。
        n_episodes: 每个种子跑多少回合。
        deterministic: 是否用确定性动作。关掉只应该出现在专门研究随机性的场合。
    """

    name: str
    seeds: tuple[int, ...]
    n_episodes: int
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("评估协议至少要有一个种子")
        if self.n_episodes <= 0:
            raise ValueError(f"每个种子的回合数必须为正，得到 {self.n_episodes}")

    @property
    def total_episodes(self) -> int:
        return len(self.seeds) * self.n_episodes

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seeds": list(self.seeds),
            "n_episodes_per_seed": self.n_episodes,
            "deterministic": self.deterministic,
        }


def train_protocol(n_episodes: int = 5) -> EvalProtocol:
    """训练期评估：单一种子、少量回合。便宜、够用来发现学崩。"""
    return EvalProtocol(name="train", seeds=(TRAIN_SEED,), n_episodes=n_episodes)


def final_protocol(n_episodes: int = 20) -> EvalProtocol:
    """最终评估：固定种子集合、更多回合。写进报告和文档的数字从这里出。"""
    return EvalProtocol(name="final", seeds=FINAL_SEEDS, n_episodes=n_episodes)


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentStats:
    """一段观测量程的统计量。逐维给出，长度等于该段的 dim。

    字段都是 tuple 而不是 ndarray：frozen dataclass 需要可比较、可哈希的字段，
    而 ndarray 的 `==` 返回逐元素结果，相等性断言会直接失效。

    Attributes:
        saturation: 各维绝对值超过阈值 threshold 的样本比例。这个数比均值更能
            说明问题：某一维均值为 0、饱和比例却很高，意味着策略在两个极端
            之间来回跳，而不是稳定工作。
        threshold: 判定饱和用的量程。None 表示这一段没有先验量程，未做统计。
    """

    name: str
    dim: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    max_abs: tuple[float, ...]
    saturation: tuple[float, ...] = ()
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "mean": list(self.mean),
            "std": list(self.std),
            "max_abs": list(self.max_abs),
            "saturation": list(self.saturation),
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class EpisodeResult:
    """一个回合的结果。

    terminated 与 truncated 分开记：前者是任务失败（跑出边界、摔倒），后者是
    超时。两者的价值处理不同，混在一起看会低估策略的失败率。
    """

    seed: int
    episode: int
    episode_return: float
    length: int
    terminated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "episode": self.episode,
            "return": self.episode_return,
            "length": self.length,
            "terminated": self.terminated,
        }


@dataclass(frozen=True)
class EvalResult:
    """一次完整评估的产物。既是给人看的结果，也是落盘 JSON 的来源。

    `return_std` 和 `return_seed_std` 的区别值得记一笔：前者是回合间的波动，
    衡量评估内部有多吵；后者是种子间（每组种子均值之间）的波动，衡量"换一批
    初始状态结论还成不成立"。做实验对比时该用的是后者——回合间波动会随回合数
    增加而变小，但它变小的只是噪声，不代表结论更稳。
    """

    protocol: EvalProtocol
    episodes: tuple[EpisodeResult, ...]
    segments: dict[str, SegmentStats]

    # ---- 基本统计 ----

    @property
    def returns(self) -> np.ndarray:
        return np.asarray([e.episode_return for e in self.episodes], dtype=np.float64)

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray([e.length for e in self.episodes], dtype=np.float64)

    @property
    def return_mean(self) -> float:
        return float(self.returns.mean())

    @property
    def return_std(self) -> float:
        return float(self.returns.std())

    @property
    def length_mean(self) -> float:
        return float(self.lengths.mean())

    @property
    def seed_return_means(self) -> dict[int, float]:
        """每个种子的平均回报。多组种子时，实验结论的误差棒应该从这里算。"""
        out: dict[int, float] = {}
        for seed in self.protocol.seeds:
            values = [e.episode_return for e in self.episodes if e.seed == seed]
            if values:
                out[seed] = float(np.mean(values))
        return out

    @property
    def return_seed_std(self) -> float:
        """种子间标准差。只有一组种子时没有意义，返回 0。"""
        means = list(self.seed_return_means.values())
        return float(np.std(means)) if len(means) > 1 else 0.0

    @property
    def terminated_rate(self) -> float:
        """因任务失败提前结束的回合比例。"""
        if not self.episodes:
            return 0.0
        return float(np.mean([e.terminated for e in self.episodes]))

    # ---- 导出 ----

    def summary(self) -> dict[str, float]:
        """标量指标。键名与 Trainer.evaluate() 对齐，两条路径的数可以并排看。"""
        return {
            "eval/return_mean": self.return_mean,
            "eval/return_std": self.return_std,
            "eval/return_seed_std": self.return_seed_std,
            "eval/episode_length_mean": self.length_mean,
            "eval/terminated_rate": self.terminated_rate,
            "eval/episodes": float(len(self.episodes)),
            "eval/seeds": float(len(self.protocol.seeds)),
        }

    def to_dict(self) -> dict[str, Any]:
        """可直接喂给 save_report 的结构。

        标量拍平成 `eval/xxx` 的键，和 Trainer.evaluate() 的指标键同名；逐段的
        细节放在 segments 下。拍平是为了让报告 diff 时一眼看到哪一项变了。
        """
        return {
            "protocol": self.protocol.to_dict(),
            "summary": self.summary(),
            "segments": {name: s.to_dict() for name, s in sorted(self.segments.items())},
            "episodes": [e.to_dict() for e in self.episodes],
            "per_seed_return_mean": {
                str(seed): value for seed, value in sorted(self.seed_return_means.items())
            },
        }

    def __str__(self) -> str:
        return (
            f"EvalResult[{self.protocol.name}] "
            f"return={self.return_mean:.2f} ± {self.return_std:.2f} "
            f"(种子间 ± {self.return_seed_std:.2f}) "
            f"len={self.length_mean:.1f} "
            f"失败率={self.terminated_rate:.0%} "
            f"回合={len(self.episodes)}"
        )


# ---------------------------------------------------------------------------
# 逐段统计的在线累积
# ---------------------------------------------------------------------------


class _SegmentAccumulator:
    """按契约分段累积观测量程统计。

    在线累积而不是把整条轨迹存下来再算：一个回合动辄上千步、观测上百维，
    存全量在长评估里会白吃几百 MB 内存，而均值/方差/最大绝对值/饱和计数
    都是可以增量维护的。
    """

    def __init__(self, contract: ObsContract, thresholds: dict[str, float | None]) -> None:
        self._contract = contract
        self._thresholds = thresholds
        self._count = 0
        self._sum = {s.name: np.zeros(s.dim) for s in contract.segments}
        self._sumsq = {s.name: np.zeros(s.dim) for s in contract.segments}
        self._maxabs = {s.name: np.zeros(s.dim) for s in contract.segments}
        self._sat = {s.name: np.zeros(s.dim) for s in contract.segments}

    def update(self, obs: np.ndarray) -> None:
        """累积一帧观测。传进来的是策略当步实际看到的那一帧。"""
        parts = self._contract.split(np.asarray(obs, dtype=np.float64))
        for name, values in parts.items():
            self._sum[name] += values
            self._sumsq[name] += values**2
            self._maxabs[name] = np.maximum(self._maxabs[name], np.abs(values))
            threshold = self._thresholds.get(name)
            if threshold is not None:
                self._sat[name] += np.abs(values) > threshold
        self._count += 1

    def finish(self) -> dict[str, SegmentStats]:
        if self._count == 0:
            raise RuntimeError("没有累积到任何观测，无法计算逐段统计")

        out: dict[str, SegmentStats] = {}
        for segment in self._contract.segments:
            name = segment.name
            mean = self._sum[name] / self._count
            # 方差用 E[x²] - E[x]²，再夹到非负：浮点误差会让本该为 0 的方差
            # 变成 -1e-18，开根号直接得到 nan。
            var = np.maximum(self._sumsq[name] / self._count - mean**2, 0.0)
            threshold = self._thresholds.get(name)
            out[name] = SegmentStats(
                name=name,
                dim=segment.dim,
                mean=tuple(float(v) for v in mean),
                std=tuple(float(v) for v in np.sqrt(var)),
                max_abs=tuple(float(v) for v in self._maxabs[name]),
                saturation=(
                    tuple(float(v) for v in self._sat[name] / self._count)
                    if threshold is not None
                    else ()
                ),
                threshold=threshold,
            )
        return out


def _resolve_thresholds(
    contract: ObsContract, overrides: dict[str, float] | None
) -> dict[str, float | None]:
    """给契约里的每一段定出饱和阈值。没量程的段填 None，表示不统计。

    传了 overrides 就以它为准，而不是与内置表逐键合并：合并的话，调用方想
    关掉某一段的统计只能靠"传一个很大的阈值"这种拐弯写法，而传空表会被
    静默理解成"用默认值"——一个参数同时表达两件事，早晚会有人踩。
    """
    table = _SATURATION_THRESHOLDS if overrides is None else overrides
    return {
        segment.name: float(table[segment.name]) if segment.name in table else None
        for segment in contract.segments
    }


# ---------------------------------------------------------------------------
# 评估主流程
# ---------------------------------------------------------------------------


def _select_action(policy: Policy, obs: np.ndarray, deterministic: bool) -> np.ndarray:
    """把一帧观测喂给策略，取回动作。走 act() 而不是 forward()，随机策略
    才有机会在 deterministic=False 时真正采样。"""
    tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
    action = policy.act(tensor, deterministic=deterministic)
    return action.squeeze(0).detach().cpu().numpy()


def _rollout(
    env: BaseEnv,
    policy: Policy,
    *,
    seed: int,
    episode: int,
    deterministic: bool,
    accumulator: _SegmentAccumulator,
) -> EpisodeResult:
    """跑一个回合。"""
    obs, _ = env.reset(seed=seed)

    episode_return = 0.0
    length = 0
    terminated = False

    while True:
        # 先记账再看动作：累积的观测序列恰好是"策略每一步的输入"，不复不漏
        accumulator.update(obs.policy)
        action = _select_action(policy, obs.policy, deterministic)
        result = env.step(action)
        episode_return += result.reward
        length += 1
        obs = result.obs
        if result.terminated or result.truncated:
            terminated = bool(result.terminated)
            break

    return EpisodeResult(
        seed=seed,
        episode=episode,
        episode_return=float(episode_return),
        length=length,
        terminated=terminated,
    )


@torch.no_grad()
def evaluate_with_protocol(
    env: BaseEnv,
    policy: Policy,
    protocol: EvalProtocol,
    *,
    saturation_thresholds: dict[str, float] | None = None,
) -> EvalResult:
    """按给定协议评估策略。

    评估前后恢复 policy 的训练模式：调用方通常在训练循环里顺手评估，忘了切回去
    会让 dropout、批归一化这类在训练/推理下行为不同的层悄悄出问题。

    Args:
        env: 环境。回合之间用协议里的种子复位，保证轨迹可复现。
        policy: 策略。按确定性路径求值。
        protocol: 评估协议。种子集合与回合数都在里面。
        saturation_thresholds: 换一套饱和量程表，键是段名。不传用内置表；传了
            就以传入的为准，表里没有的段不做饱和统计。
    """
    was_training = policy.training
    policy.eval()

    contract = env.obs_contract
    accumulator = _SegmentAccumulator(contract, _resolve_thresholds(contract, saturation_thresholds))

    episodes: list[EpisodeResult] = []
    for seed in protocol.seeds:
        for episode in range(protocol.n_episodes):
            episodes.append(
                _rollout(
                    env,
                    policy,
                    seed=seed,
                    episode=episode,
                    deterministic=protocol.deterministic,
                    accumulator=accumulator,
                )
            )

    if was_training:
        policy.train()

    return EvalResult(
        protocol=protocol,
        episodes=tuple(episodes),
        segments=accumulator.finish(),
    )


def evaluate_policy(
    env: BaseEnv,
    policy: Policy,
    n_episodes: int = 20,
    *,
    seed: int | Sequence[int] = 0,
    deterministic: bool = True,
    saturation_thresholds: dict[str, float] | None = None,
) -> EvalResult:
    """便捷入口：直接给回合数和种子。

    seed 可以是单个整数，也可以是一组。给一组时，每个种子各跑 n_episodes 个
    回合，返回的 `return_seed_std` 就是可以写进报告的那个误差棒。

    需要"训练期 / 最终评估"这种有名字、可追溯的约定时，用
    `evaluate_with_protocol(env, policy, final_protocol())`——种子集合固定，
    两次实验的数才真的可比。
    """
    seeds = (seed,) if isinstance(seed, (int, np.integer)) else tuple(int(s) for s in seed)
    protocol = EvalProtocol(
        name="custom",
        seeds=seeds,
        n_episodes=n_episodes,
        deterministic=deterministic,
    )
    return evaluate_with_protocol(
        env, policy, protocol, saturation_thresholds=saturation_thresholds
    )
