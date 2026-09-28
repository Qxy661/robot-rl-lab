"""训练循环：把环境、算法、日志、评估、存档串起来。

PPO 与 SAC 的循环结构不同，这个差异是算法性质决定的，藏不掉：

    PPO  采满一轮 rollout -> 算优势 -> 更新几遍 -> 丢掉数据 -> 重来
         同一个样本在一个 epoch 里被看 num_learning_epochs 遍，之后不再用。
    SAC  采一步 -> 存进回放池 -> 从池里随机取一批 -> 更新
         数据反复复用，所以"轮次"这个概念只是为了让日志和评估有统一的刻度。

两者共用的部分放在 _TrainerBase 里：环境推进、指标落盘、评估与存档。这些
事情一旦各写各的，两条曲线就不可比了。

关于并行：这里的"向量化"指的是持有一批同步推进的环境实例，不是多进程。
MuJoCo 的单步无法用多线程加速，真正的并行要开多个进程，那是环境层
（vec_backend）的事，训练循环只需要知道"有一批环境、每步都给动作、都返回
结果"。
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from robotrl.algorithms.base import Policy, Trainer, TrainMetrics
from robotrl.algorithms.networks import ActorCritic, SACActor, TwinQNetwork
from robotrl.algorithms.ppo import PPO
from robotrl.algorithms.sac import SAC
from robotrl.algorithms.storage import ReplayBuffer, RolloutStorage
from robotrl.configs.schema import PPOConfig, SACConfig
from robotrl.envs.vector_env import SerialVecEnv, VecEnv
from robotrl.utils.torch_runtime import configure_torch

#: 训练循环一次推进多少个环境步。设得比回合上限小很多，评估才来得及在
#: 策略明显变差之前发现它。
DEFAULT_STEPS_PER_ITERATION = 1000


def _as_vec_env(env: Any) -> VecEnv:
    """把环境统一成批式接口。

    三种输入都接受：单个环境、一批环境实例、已经包好的 VecEnv。前两种在
    这里套上 SerialVecEnv，于是训练循环**只有一条推进路径**。

    这一点值得强调：如果串行和多进程各走一套逻辑，两套都必须维护、必须测，
    而且行为要逐位一致。一旦哪天分叉，就会出现"换个后端结果就变了"的
    问题——这种问题没有报错、只有数字不同，排查代价极高。
    """
    if isinstance(env, VecEnv):
        return env
    if isinstance(env, (list, tuple)):
        if not env:
            raise ValueError("环境列表不能为空")
        return SerialVecEnv(list(env))
    return SerialVecEnv([env])


def _stack(values: Sequence[np.ndarray]) -> Tensor:
    return torch.as_tensor(np.stack(values), dtype=torch.float32)


class _TrainerBase(Trainer):
    """PPO 与 SAC 共用的骨架：环境推进、日志、评估、存档。"""

    def __init__(
        self,
        env: Any,
        policy: Policy,
        *,
        run_dir: str | Path = "runs",
        seed: int = 0,
        device: str | torch.device = "cpu",
        log_interval: int = 1,
        eval_interval: int = 0,
        save_interval: int = 0,
        eval_episodes: int = 10,
        verbose: bool = True,
        num_threads: int | None = None,
    ) -> None:
        vec = _as_vec_env(env)
        # 线程数在构造时就定下来。默认单线程——这类规模的策略网络用多线程
        # 反而慢一个量级，原因见 utils/torch_runtime.py。
        self.torch_threads = configure_torch(num_threads)
        # 基类的 evaluate() 只接受单个环境，所以这里用 vec 提供的评估环境。
        # 评估固定用同一个实例、固定的种子序列，两条曲线才有可比性。
        super().__init__(vec.eval_env, policy, run_dir=run_dir, seed=seed)
        self.vec = vec
        self.num_envs = vec.num_envs
        self.device = torch.device(device)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.eval_episodes = eval_episodes
        self.verbose = verbose

        self.iteration = 0
        self.history: list[TrainMetrics] = []
        self._episode_returns: list[float] = []
        self._running_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._running_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._best_return = -np.inf
        self._start_time = time.time()

    # ------------------------------------------------------------------

    def _reset_envs(self) -> tuple[Tensor, Tensor]:
        """复位所有环境。种子错开，否则所有环境走同一条轨迹，并行等于白开。"""
        obs_list = self.vec.reset([self.seed + i for i in range(self.num_envs)])
        return (
            _stack([o.policy for o in obs_list]),
            _stack([o.critic for o in obs_list]),
        )

    def _advance(
        self,
        actions: np.ndarray,
    ) -> tuple[Tensor, Tensor, list[float], list[bool], list[bool]]:
        """把一批动作下发给所有环境，收集结果。

        返回 (下一步 policy 观测, 下一步 critic 观测, 奖励, terminated, truncated)。

        整批一起下发（step_batch）而不是逐个 step。多进程后端下，逐个调用
        会让每次都要等一轮 IPC 返回，N 次等待串起来等于没有并行，还白付
        通信开销。详见 envs/vector_env.py 的模块文档。
        """
        results = self.vec.step_batch(actions)
        for i, result in enumerate(results):
            self._running_returns[i] += result.reward
            self._running_lengths[i] += 1
            if result.terminated or result.truncated:
                self._episode_returns.append(float(self._running_returns[i]))
                self._running_returns[i] = 0.0
                self._running_lengths[i] = 0
        return (
            _stack([r.obs.policy for r in results]),
            _stack([r.obs.critic for r in results]),
            [r.reward for r in results],
            [r.terminated for r in results],
            [r.truncated for r in results],
        )

    # ------------------------------------------------------------------
    # 日志与存档
    # ------------------------------------------------------------------

    def _rollout_stats(self) -> dict[str, float]:
        """取走这轮采到的回合回报。取走而不是累计，指标才是"本轮的"。

        回报统计的是**已完成**的回合，所以回合很长的任务上前几十轮可能是空的。
        这正是要看的信号：没有一条回合跑完，说明策略还没活到回合上限。
        """
        finished = self._episode_returns
        self._episode_returns = []
        if not finished:
            return {"rollout/episodes": 0.0}
        returns = np.asarray(finished, dtype=np.float64)
        return {
            "rollout/episodes": float(len(finished)),
            "rollout/return_mean": float(returns.mean()),
            "rollout/return_std": float(returns.std()),
            "rollout/return_min": float(returns.min()),
            "rollout/return_max": float(returns.max()),
        }

    def _log(self, metrics: dict[str, float]) -> TrainMetrics:
        record = TrainMetrics(
            iteration=self.iteration,
            total_timesteps=self.num_timesteps,
            metrics={
                **metrics,
                "time/elapsed": time.time() - self._start_time,
                "time/fps": self.num_timesteps / max(time.time() - self._start_time, 1e-6),
            },
        )
        self.history.append(record)
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        if self.verbose and self.iteration % self.log_interval == 0:
            body = " ".join(
                f"{k.split('/')[-1]}={v:.3g}"
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            )
            print(f"[{self.iteration}] steps={self.num_timesteps} {body}", flush=True)
        return record

    def _maybe_evaluate(self) -> dict[str, float]:
        if not self.eval_interval or self.iteration % self.eval_interval:
            return {}
        return self.evaluate(self.eval_episodes, seed=self.seed + 10_000)

    def _maybe_save(self, eval_metrics: dict[str, float]) -> None:
        if not self.save_interval or self.iteration % self.save_interval:
            return
        self.save(self.run_dir / f"checkpoint_{self.iteration:06d}.pt")
        self.save(self.run_dir / "latest.pt")
        # best.pt 按评估回报挑，不按训练回报：训练回报里混着探索噪声和当轮
        # 初始状态的运气，用它挑出来的"最好"往往只是运气好的那几轮。
        score = eval_metrics.get("eval/return_mean")
        if score is not None and score >= self._best_return:
            self._best_return = score
            self.save(self.run_dir / "best.pt")

    def _finish_iteration(self) -> dict[str, float]:
        """迭代收尾：评估、存档。返回这一步的评估指标。"""
        eval_metrics = self._maybe_evaluate()
        self._maybe_save(eval_metrics)
        return eval_metrics

    def _prepare_run_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


class PPOTrainer(_TrainerBase):
    """同策略训练循环。

    一轮迭代的结构是固定的：采满 num_steps_per_env 步 -> 算 GAE -> 更新。
    采样和更新不能交错，因为更新一旦动了参数，之前采的数据就不再是"当前策略
    采的"，重要性比值的前提就没了。
    """

    def __init__(
        self,
        env: Any,
        policy: ActorCritic,
        cfg: PPOConfig | None = None,
        *,
        algo: PPO | None = None,
        run_dir: str | Path = "runs",
        seed: int = 0,
        device: str | torch.device = "cpu",
        log_interval: int = 1,
        eval_interval: int = 0,
        save_interval: int = 0,
        eval_episodes: int = 10,
        verbose: bool = True,
        num_threads: int | None = None,
    ) -> None:
        super().__init__(
            env,
            policy,
            run_dir=run_dir,
            seed=seed,
            device=device,
            log_interval=log_interval,
            eval_interval=eval_interval,
            save_interval=save_interval,
            eval_episodes=eval_episodes,
            verbose=verbose,
            num_threads=num_threads,
        )
        self.cfg = cfg or PPOConfig()
        self.policy: ActorCritic = policy.to(self.device)
        # 允许注入现成的更新器：测试里想固定随机性、或从别的训练里接管时有用。
        self.algo = algo or PPO(self.policy, self.cfg, device=self.device)

    def train(self, total_timesteps: int) -> list[TrainMetrics]:
        """训练到指定总步数。

        总步数是循环的退出条件而不是迭代次数：PPO 的样本量等于
        num_steps_per_env × num_envs，按迭代次数计会让"训了多久"依赖于并行度，
        换机器就跑出不同的结果。
        """
        cfg = self.cfg
        self._prepare_run_dir()
        obs, critic_obs = self._reset_envs()

        num_envs = self.num_envs
        storage = RolloutStorage(
            cfg.num_steps_per_env,
            num_envs,
            self.policy.obs_dim,
            self.policy.critic_obs_dim,
            self.policy.action_dim,
            device=self.device,
        )

        while self.num_timesteps < total_timesteps:
            self.iteration += 1
            for _ in range(cfg.num_steps_per_env):
                action, log_prob, value = self.algo.act(obs, critic_obs)
                next_obs, next_critic, rewards, terminateds, truncateds = self._advance(
                    action.cpu().numpy()
                )
                storage.add(
                    obs=obs,
                    critic_obs=critic_obs,
                    action=action,
                    log_prob=log_prob,
                    reward=torch.as_tensor(rewards),
                    terminated=torch.as_tensor(terminateds),
                    truncated=torch.as_tensor(truncateds),
                    value=value,
                )
                obs, critic_obs = next_obs, next_critic
                self.num_timesteps += num_envs

            # rollout 末尾那个观测的价值。它只在"回合还没结束"或"最后一步恰好
            # 超时"时被用上，真终止的环境上会被 compute_gae 屏蔽掉。
            with torch.no_grad():
                last_values = self.policy.value(critic_obs)

            if cfg.use_obs_norm:
                # 按整轮 rollout 更新统计量，不是按每步：单环境采样时每步只有
                # 一条观测，滑动平均会被单条样本的噪声主导；整轮的样本量是
                # num_steps × num_envs，估出来的均值方差稳定得多。
                self.policy.update_normalizer(storage.obs)
                self.policy.update_critic_normalizer(storage.critic_obs)

            storage.compute_gae(last_values, cfg.gamma, cfg.lam)
            metrics = self.algo.update(storage)
            storage.reset()

            metrics.update(self._rollout_stats())
            eval_metrics = self._finish_iteration()
            metrics.update(eval_metrics)
            self._log(metrics)

        return self.history

    def _checkpoint(self) -> dict[str, Any]:
        return {
            **super()._checkpoint(),
            "algo": self.algo.state_dict(),
            "iteration": self.iteration,
        }

    def _load_extra(self, ckpt: dict[str, Any]) -> None:
        self.algo.load_state_dict(ckpt["algo"])
        self.iteration = ckpt.get("iteration", 0)


# ---------------------------------------------------------------------------
# SAC
# ---------------------------------------------------------------------------


class SACTrainer(_TrainerBase):
    """异策略训练循环。

    没有"采满一轮再更新"这回事——每采一步就能更新一次。迭代在这里只是日志和
    评估的刻度，用 steps_per_iteration 划出来。
    """

    def __init__(
        self,
        env: Any,
        actor: SACActor,
        cfg: SACConfig | None = None,
        *,
        critic: TwinQNetwork | None = None,
        algo: SAC | None = None,
        steps_per_iteration: int = DEFAULT_STEPS_PER_ITERATION,
        num_updates_per_step: int = 1,
        save_replay_buffer: bool = False,
        run_dir: str | Path = "runs",
        seed: int = 0,
        device: str | torch.device = "cpu",
        log_interval: int = 1,
        eval_interval: int = 0,
        save_interval: int = 0,
        eval_episodes: int = 10,
        verbose: bool = True,
        num_threads: int | None = None,
    ) -> None:
        super().__init__(
            env,
            actor,
            run_dir=run_dir,
            seed=seed,
            device=device,
            log_interval=log_interval,
            eval_interval=eval_interval,
            save_interval=save_interval,
            eval_episodes=eval_episodes,
            verbose=verbose,
            num_threads=num_threads,
        )
        self.cfg = cfg or SACConfig()
        self.actor: SACActor = actor.to(self.device)
        self.critic = (
            critic
            or TwinQNetwork(
                self.actor.obs_dim,
                self.actor.action_dim,
                use_layer_norm=self.cfg.use_layer_norm,
            )
        ).to(self.device)
        self.algo = algo or SAC(self.actor, self.critic, self.cfg, device=self.device)
        self.steps_per_iteration = steps_per_iteration
        self.num_updates_per_step = num_updates_per_step
        self.save_replay_buffer = save_replay_buffer
        self.buffer = ReplayBuffer(
            self.cfg.buffer_size,
            self.actor.obs_dim,
            self.actor.action_dim,
            device=self.device,
            seed=seed,
        )
        self._rng = np.random.default_rng(seed)

    def train(self, total_timesteps: int) -> list[TrainMetrics]:
        cfg = self.cfg
        self._prepare_run_dir()
        obs, _ = self._reset_envs()

        totals: dict[str, float] = {}
        num_updates = 0
        while self.num_timesteps < total_timesteps:
            self.iteration += 1
            for _ in range(self.steps_per_iteration):
                actions = self._select_actions(obs)
                next_obs, _, rewards, terminateds, truncateds = self._advance(actions)
                self.buffer.add_batch(
                    obs.numpy(), actions, rewards, next_obs.numpy(), terminateds, truncateds
                )
                obs = next_obs
                self.num_timesteps += self.num_envs

                # 回放池还没铺开时不学习：一份几十条、且几乎全来自同一个初始
                # 状态的数据，只够 critic 记住那一个点的价值，接着就会被用来
                # 指导策略，典型的先过拟合再发散。
                if len(self.buffer) < cfg.learning_starts:
                    continue
                for _ in range(self.num_updates_per_step):
                    for key, value in self.algo.update(self.buffer.sample(cfg.batch_size)).items():
                        totals[key] = totals.get(key, 0.0) + value
                    num_updates += 1

            metrics = {k: v / max(num_updates, 1) for k, v in totals.items()}
            totals, num_updates = {}, 0
            metrics.update(self._rollout_stats())
            metrics["sac/buffer_size"] = float(len(self.buffer))
            metrics.update(self._finish_iteration())
            self._log(metrics)

        return self.history

    def _select_actions(self, obs: Tensor) -> np.ndarray:
        """选动作。学习开始前用均匀随机动作铺回放池。

        随机动作而不是策略动作：未训练的策略输出近似恒定（输出层增益被压到
        0.01），用它采出来的是高度重复的状态，回放池的"多样性"是假的。
        """
        if len(self.buffer) < self.cfg.learning_starts:
            return self._rng.uniform(-1.0, 1.0, size=(self.num_envs, self.actor.action_dim))
        with torch.no_grad():
            action, _ = self.algo.act(obs)
        return action.cpu().numpy().astype(np.float32)

    def _checkpoint(self) -> dict[str, Any]:
        ckpt = {
            **super()._checkpoint(),
            "critic": self.critic.state_dict(),
            "algo": self.algo.state_dict(),
            "iteration": self.iteration,
        }
        if self.save_replay_buffer:
            ckpt["replay_buffer"] = self.buffer.state_dict()
        return ckpt

    def _load_extra(self, ckpt: dict[str, Any]) -> None:
        if "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
        self.algo.load_state_dict(ckpt["algo"])
        self.iteration = ckpt.get("iteration", 0)
        if self.save_replay_buffer and "replay_buffer" in ckpt:
            self.buffer.load_state_dict(ckpt["replay_buffer"])


def make_trainer(
    env: Any,
    policy: Policy,
    *,
    algo_name: str = "ppo",
    cfg: PPOConfig | SACConfig | None = None,
    **kwargs: Any,
) -> Trainer:
    """按算法名构造训练器。

    存在的意义是让上层（脚本、评估层）不必 import 具体的训练器类，也就不会
    因为新增一个算法而到处加分支。
    """
    if algo_name == "ppo":
        return PPOTrainer(env, policy, cfg, **kwargs)
    if algo_name == "sac":
        return SACTrainer(env, policy, cfg, **kwargs)
    raise ValueError(f"未知算法 {algo_name!r}，只支持 ppo / sac")
