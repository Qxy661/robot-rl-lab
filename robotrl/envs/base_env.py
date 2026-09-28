"""环境层契约：BaseEnv 与它钉死的 8 段管线。

环境是强化学习里最容易写乱的一层：奖励、终止、复位、指令、域随机化、观测
构造全挤在一个 step() 里，几百行下来没人说得清执行顺序，也没法单独测试
其中任何一项。这里的做法是把顺序钉死在基类，子类只填空——每个环节一个
方法，各自可以独立单测、独立替换。

一个 step() 的执行顺序（不可更改）：

    1. Act        _clip_action  动作裁剪
    2. Simulate   _apply_action → _simulate   换算 PD 目标并推进物理
    3. Terminate  _terminate     是否本回合结束
    4. Reward     _reward        本步奖励
    5. Reset      _reset_sim     若结束则复位（在 reward 之后，避免奖励用错状态）
    6. Command    _update_command 重采样速度指令
    7. Events     _apply_events  域随机化，按低频触发
    8. Observe    _observe       组装观测

顺序不是随便定的，有两处是踩过坑才这么排：复位放在 reward 之后，是为了让
终止那一步的奖励还在结束前的状态上计算；观测放在最后，是为了让它看到的
一定是复位和随机化都完成之后的状态。

注意 BaseEnv 不依赖 MuJoCo——它只是一份管线契约，用 numpy 就能实现一个合规
的环境。真正的 MuJoCo 实现在 envs/mujoco_env.py。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robotrl.assets.spec import RobotSpec
from robotrl.contracts import ObsContract


@dataclass(frozen=True)
class Obs:
    """一次观测。policy 给策略网络，critic 给价值网络。

    分开是刻意的：PPO 的 actor 和 critic 输入维度可以不同，critic 能拿到
    训练期才存在的特权信息（真实机身速度、地形高度等）。部署时只导出
    policy，所以额外信息不会成为上板负担。
    """

    policy: np.ndarray
    critic: np.ndarray


@dataclass
class StepResult:
    """一次 step 的完整结果。

    terminated 和 truncated 分开，不能合并成一个 done。原因在 PPO 的价值估计：
    因任务失败而终止（摔倒）状态价值确实是 0，可以截断；因超时而截断则是
    "这一回合没结束，只是不继续采样了"，最后一步仍要 bootstrap 价值。
    合并成一个布尔量会让超时被当成真终止，价值估计凭空掉一个台阶。
    """

    obs: Obs
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


class BaseEnv(ABC):
    """环境基类。子类实现 8 段管线里的各段，不实现 step() 本身。

    Attributes:
        spec: 机器人形态，任务代码通过它读维度，不写死具体机器人。
        max_episode_steps: 回合上限，超时算 truncated 而非 terminated。

    Note:
        指令的存放位置约定为 `self._command`（形如 `[vx, vy, yaw_rate]` 的
        三维向量），由 `_update_command` 每步写入。基类不实现它——有的任务
        没有指令——但复位、回放、固定指令评估都要用到，所以把这个约定写进
        接口，而不是让每个调用方去猜属性名。
    """

    def __init__(self, spec: RobotSpec, max_episode_steps: int = 1000) -> None:
        self.spec = spec
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._episode_return = 0.0
        # 非 None 时压制任务自己的指令采样，见 set_command()。
        self._command_override: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 观测契约：由 n_dof 推出，因此天然跨形态
    # ------------------------------------------------------------------

    @property
    def obs_contract(self) -> ObsContract:
        """policy 观测布局。子类可覆盖以增删段。"""
        from robotrl.contracts import make_obs_contract

        return make_obs_contract(self.spec.n_dof)

    @property
    def critic_obs_contract(self) -> ObsContract:
        """critic 观测布局。默认与 policy 相同，子类可扩展特权信息。"""
        return self.obs_contract

    @property
    def obs_dim(self) -> int:
        return self.obs_contract.total_dim

    @property
    def critic_obs_dim(self) -> int:
        return self.critic_obs_contract.total_dim

    @property
    def action_dim(self) -> int:
        """动作维度恒等于被控关节数。"""
        return self.spec.n_dof

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> tuple[Obs, dict[str, Any]]:
        """复位环境。返回 (观测, 信息)。

        复位也要走一遍指令采样和域随机化——否则第一步的观测里 cmd 是残留值、
        随机化参数还是上一回合的，训练会在这一个样本上反复吃到脏数据。
        """
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self._step_count = 0
        self._episode_return = 0.0
        self._reset_sim()
        self._update_command()
        self._apply_command_override()
        self._apply_events(reset=True)
        return self._observe(), {"step": 0}

    def step(self, action: np.ndarray) -> StepResult:
        """推进一个控制步。顺序见模块文档，子类不应覆盖本方法。"""
        # 1. Act —— 裁剪后动作的语义是归一化增量，见 contracts.py
        action = self._clip_action(action)

        # 2. Simulate —— 换算成关节位置目标并推进物理
        self._apply_action(action)
        self._simulate()

        # 3. Terminate
        terminated = self._terminate()

        # 4. Reward —— 在复位之前算，保证用的是结束时刻的真实状态
        reward = self._reward()

        # 5. Reset —— 结束则复位
        self._step_count += 1
        self._episode_return += reward
        truncated = self._truncate()
        if terminated or truncated:
            self._reset_sim()
            self._step_count = 0

        # 6. Command —— 重采样指令，让复位后的第一步就有有效目标
        self._update_command()
        self._apply_command_override()

        # 7. Events —— 域随机化按低频触发
        self._apply_events(reset=False)

        # 8. Observe —— 组装观测，此时看到的必定是复位完成后的状态
        obs = self._observe()

        info = {
            "step": self._step_count,
            "episode_return": self._episode_return,
            "is_terminal_step": terminated or truncated,
        }
        return StepResult(
            obs=obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    # ------------------------------------------------------------------
    # 指令
    # ------------------------------------------------------------------

    @property
    def command(self) -> np.ndarray:
        """当前速度指令 [vx, vy, yaw_rate]。无指令任务返回零向量。"""
        cmd = getattr(self, "_command", None)
        if cmd is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(cmd, dtype=np.float64)

    def set_command(self, command: Sequence[float]) -> None:
        """固定速度指令，覆盖任务自己的采样。

        训练时指令要随机采样，策略才不会只学会跟一条指令。但另外两个场景
        需要固定：回放时想指定让机器人怎么走，评估时想确认"这条指令跟得准
        不准"这个具体问题。用同一个入口而不是两套代码，观察到的行为才与
        训练时一致。

        覆盖会一直生效到 clear_command()，跨回合保持。
        """
        vec = np.asarray(command, dtype=np.float64).ravel()
        if vec.shape != (3,):
            raise ValueError(f"速度指令应为三个分量，得到形状 {vec.shape}")
        self._command_override = vec.copy()
        self._command = vec.copy()

    def clear_command(self) -> None:
        """恢复任务自己的指令采样。"""
        self._command_override = None

    def _apply_command_override(self) -> None:
        """在任务采样之后覆盖指令。管线的第 6 段调用它。"""
        if self._command_override is not None:
            self._command = self._command_override.copy()

    def close(self) -> None:
        """释放资源。无资源的实现可以直接不覆盖。"""

    # ------------------------------------------------------------------
    # 8 段管线：子类按需覆盖
    # ------------------------------------------------------------------

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """把动作裁剪到 [-1, 1]。这是动作契约的一部分，不要绕开。"""
        return np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

    @abstractmethod
    def _apply_action(self, action: np.ndarray) -> None:
        """把归一化动作换算成执行器目标并写入仿真器。"""

    @abstractmethod
    def _simulate(self) -> None:
        """推进物理仿真若干个内部步。"""

    def _terminate(self) -> bool:
        """任务失败判定（例如躯干触地、姿态发散）。默认永不为真。"""
        return False

    def _reward(self) -> float:
        """本步奖励。默认 0，便于只验证管线时跑通。"""
        return 0.0

    def _truncate(self) -> bool:
        """超时判定。默认按 max_episode_steps 截断。"""
        return self._step_count >= self.max_episode_steps

    @abstractmethod
    def _reset_sim(self) -> None:
        """复位物理状态与回合内累积量。"""

    def _update_command(self) -> None:
        """重采样速度指令。无指令任务可以不覆盖。"""

    def _apply_events(self, *, reset: bool) -> None:
        """域随机化。reset=True 时施加每次回合开始才做的随机化。

        参数不是"是否只在 reset 时执行"，而是"当前处于哪种时机"——
        每步都做的扰动（如观测噪声、外力推动）和每回合才做一次的随机化
        （如质量、摩擦系数）在同一个方法里按 reset 分流，避免拆成两个
        容易漏调的钩子。
        """

    @abstractmethod
    def _observe(self) -> Obs:
        """组装 policy / critic 观测。"""

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @property
    def np_random(self) -> np.random.Generator:
        """环境私有随机源。用它而不是全局 np.random，保证种子可控可复现。"""
        if not hasattr(self, "_np_random"):
            self._np_random = np.random.default_rng()
        return self._np_random
