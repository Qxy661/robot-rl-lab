"""配置模式：一套 dataclass 描述整个训练流程的可调项。

配置系统要解决的核心问题是**跨形态**：任务代码不能写死机器人。所以这里
只有 `env.robot` 一个字符串字段指向形态名，关节数、默认姿态、PD 增益全部
由 RobotSpec 提供，配置里一个维度都不出现。换机器人就是把 "g1" 改成 "go2"。

第二个作用是**可复现**：所有影响训练结果的量都在这里，没有散落在代码里的
魔法数字。一次实验的完整配置可以直接序列化成 YAML 存档，三个月后还能
复现出同样的数。

字段划分按"谁决定它"来分：env/terrain/obs/reward/events 由任务决定，
ppo/sac/train 由算法决定，deploy 由部署目标决定。三个群体之间不交叉引用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 环境与任务
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    """环境与仿真的基本设定。"""

    robot: str = "g1"
    """形态名，对应 assets 注册表里的键。换机器人只改这一行。"""

    task: str = "velocity"
    """任务名，对应环境注册表里的键（velocity / rough / stairs / mimic）。"""

    control_dt: float = 0.02
    """控制周期（秒）。50Hz 是足式运动控制的通行选择：够快能及时修正，
    又不会让策略网络每几毫秒就得推一次。"""

    sim_dt: float = 0.002
    """物理步长（秒）。500Hz。与控制周期的比值就是 decimation，即每个控制步
    内部要跑 10 次物理积分。接触这类刚性约束需要比控制频率高得多的积分频率。"""

    max_episode_steps: int = 1000
    """回合上限。50Hz 下 1000 步 = 20 秒。"""

    seed: int = 0

    @property
    def decimation(self) -> int:
        """每个控制步内的物理步数。"""
        return max(1, round(self.control_dt / self.sim_dt))

    def __post_init__(self) -> None:
        if self.control_dt <= 0 or self.sim_dt <= 0:
            raise ValueError("控制周期与物理步长必须为正")
        if self.control_dt < self.sim_dt:
            raise ValueError(
                f"控制周期 {self.control_dt} 小于物理步长 {self.sim_dt}，"
                "意味着控制比物理还快，没有意义"
            )


@dataclass
class TerrainConfig:
    """地形生成。"""

    kind: str = "flat"
    """flat（平地）/ rough（崎岖高度场）/ stairs（楼梯）。"""

    terrain_size: tuple[float, float] = (8.0, 8.0)
    """地形平面的尺寸（米）。"""

    cell_size: float = 0.1
    """高度场网格边长（米）。越小越精细，内存和碰撞检测开销也越大。"""

    height_range: tuple[float, float] = (0.0, 0.15)
    """崎岖地形的高度起伏范围（米）。0.15 大致是能走但不至于摔的量级。"""

    step_height: float = 0.08
    """楼梯单级高度（米）。"""

    step_width: float = 0.30
    """楼梯单级进深（米）。"""

    num_steps: int = 10
    """楼梯级数。"""


@dataclass
class ObsConfig:
    """观测构造选项。"""

    use_obs_noise: bool = True
    """给观测加噪声。真机上编码器、IMU 都有噪声，训练时不见过噪声，
    策略就会把仿真里的干净读数当成物理规律。"""

    noise_scale: dict[str, float] = field(
        default_factory=lambda: {
            "lin_vel": 0.1,
            "ang_vel": 0.2,
            "proj_gravity": 0.05,
            "dof_pos": 0.01,
            "dof_vel": 0.5,
        }
    )
    """各观测段的噪声标准差，键名与观测契约里的段名一致。"""

    use_privileged_critic: bool = True
    """critic 是否使用带真值的特权观测。开启后是 asymmetric actor-critic，
    训练更稳；关闭则 actor 与 critic 输入相同，便于对照。"""

    num_feet: int = 2
    """足数，用于特权观测里的接触标志。换形态时由环境层按 morphology 覆写。"""

    terrain_samples: int = 9
    """机身下方地形高度采样点数。"""


@dataclass
class RewardConfig:
    """奖励项权重。

    键是 reward 项的名字，值是权重。**权重为 0 的项会被直接跳过不计算**，
    这样调试时可以随手关掉某一项，既不用改代码也不付计算开销。
    """

    scales: dict[str, float] = field(default_factory=dict)

    def scale_of(self, name: str) -> float:
        """取某项权重。未列出的项返回 0，即默认关闭。"""
        return self.scales.get(name, 0.0)

    def is_enabled(self, name: str) -> bool:
        return self.scale_of(name) != 0.0


@dataclass
class EventsConfig:
    """域随机化。

    每一项都是"训练分布比真机更宽"的刻意做法：真机上电机响应、地面摩擦、
    负载、外部扰动都不确定，训练时见过足够宽的分布，策略才不会一上真机就崩。
    """

    randomize_friction: bool = True
    friction_range: tuple[float, float] = (0.5, 1.25)

    randomize_base_mass: bool = True
    base_mass_delta: tuple[float, float] = (-1.0, 3.0)
    """机身附加质量范围（公斤）。负值表示减重（模拟负载估计偏差）。"""

    randomize_motor_strength: bool = True
    motor_strength_range: tuple[float, float] = (0.8, 1.2)
    """电机输出力矩的缩放。模拟电池电压下降、电机个体差异。"""

    randomize_pd_gain: bool = True
    pd_gain_range: tuple[float, float] = (0.8, 1.2)
    """PD 增益扰动。模拟真实执行器与理想 PD 的偏差，是 sim2real 的关键项之一。"""

    push_robot: bool = True
    push_interval_steps: tuple[int, int] = (200, 400)
    push_velocity_xy: tuple[float, float] = (-0.5, 0.5)
    """每隔一段时间给机身一个随机水平速度冲击（米/秒），模拟被推、踩到凸起。"""


# ---------------------------------------------------------------------------
# 算法
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """PPO 超参。

    默认值对齐论文与主流实现，不是随手填的：
    - clip_ratio 0.2、gamma 0.99、lam 0.95 来自 PPO 原论文；
    - 双学习率（actor 3e-4 / critic 1e-3）是因为价值函数的拟合难度显著高于策略；
    - 观测归一化、优势归一化、价值损失裁剪等实现细节，见 docs/04。
    """

    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.0
    """熵正则系数。足式运动通常设 0：动作维度高，熵奖励容易让策略变得抖动。"""

    value_coef: float = 1.0
    num_steps_per_env: int = 24
    """每个环境每次迭代采样多少步。这个值乘以环境数就是一次迭代的样本量。"""

    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    max_grad_norm: float = 1.0
    use_obs_norm: bool = True
    use_adv_norm: bool = True
    use_value_clip: bool = True
    use_lr_schedule: bool = True
    desired_kl: float = 0.01
    """自适应学习率的 KL 目标。实测 KL 超了就降学习率，是最稳的调参手段之一。"""


@dataclass
class SACConfig:
    """SAC 超参。"""

    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    """目标网络软更新系数。0.005 相当于约 200 步把目标网络更新一遍。"""

    auto_alpha: bool = True
    """自动调节温度系数。手动调 alpha 是 SAC 最费劲的部分，自动调节基本免掉。"""

    alpha: float = 0.2
    target_entropy_ratio: float = 1.0
    """目标熵 = -ratio * action_dim。"""

    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5000
    """先随机采集这么多步再开始学习。回放池里样本太单一就更新，容易过拟合早期数据。"""

    policy_frequency: int = 2
    """策略更新频率低于价值网络。价值估计还不准时更新策略，会引入偏差。"""

    use_layer_norm: bool = True
    use_obs_norm: bool = True


# ---------------------------------------------------------------------------
# 训练与部署
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """训练流程控制。"""

    algo: str = "ppo"
    """ppo 或 sac。决定用上面哪一份超参。"""

    total_timesteps: int = 1_000_000
    num_envs: int = 64
    vec_backend: str = "serial"
    """serial（单进程顺序推进）/ subprocess（多进程并行）。前者好调试，
    后者在纯 CPU 上能拿到接近核数的加速——MuJoCo 的单步无法用多线程加速，
    并行只能靠多开环境实例。"""

    device: str = "cpu"
    run_dir: str = "runs"
    save_interval: int = 50
    eval_interval: int = 50
    log_interval: int = 1

    num_threads: int = 1
    """PyTorch 算子内并行度。默认 1，不是保守，是这类网络下实测最快。

    策略网络的参数量在几万量级，单个算子被切到多核后，线程同步的开销
    超过了省下来的计算时间。16 核机器上一个 64×64 MLP 的 backward 走
    16 线程要 258 ms，走单线程只要 0.92 ms；网络放大到 (256,128) 结论不变。
    真要开到核数，填本机核数即可。"""

    def __post_init__(self) -> None:
        if self.algo not in ("ppo", "sac"):
            raise ValueError(f"未知算法 {self.algo!r}，只支持 ppo / sac")
        if self.vec_backend not in ("serial", "subprocess"):
            raise ValueError(f"未知向量化后端 {self.vec_backend!r}")
        if self.num_envs <= 0:
            raise ValueError("num_envs 必须为正")
        if self.num_threads <= 0:
            raise ValueError("num_threads 必须为正")


@dataclass
class DeployConfig:
    """导出、量化、基准、回测四步的设定。"""

    checkpoint: str = ""
    """待导出的 checkpoint 路径。"""

    onnx_path: str = "exported/policy.onnx"
    quantized_path: str = "exported/policy_int8.onnx"

    opset: int = 17

    quantize: bool = True
    quant_format: str = "dynamic"
    """dynamic（动态量化，只需要 onnxruntime，无需校准数据）/
    static（静态 QDQ 量化，需要校准集，精度更好但流程更重）。"""

    equivalence_tol: float = 1e-5
    """ONNX 与 PyTorch 输出的最大允许偏差。超过就说明导出过程出了问题，
    必须先查清楚再谈量化精度——不然量化的误差里混着导出的 bug。"""

    benchmark_warmup: int = 50
    benchmark_runs: int = 1000
    """延迟基准的预热次数与统计次数。预热是为了让 CPU 频率和缓存进入稳态，
    不预热测出来的前几次会明显偏慢，把 p99 拖高。"""

    backtest_episodes: int = 20
    """仿真回测的回合数。量化前后各跑这么多回合，对比奖励和动作偏差。"""

    quant_kl_threshold: float = 0.1
    """量化前后动作分布的 KL 散度上限。超过就认为量化掉点太多，需要回退。"""

    def __post_init__(self) -> None:
        if self.quant_format not in ("dynamic", "static"):
            raise ValueError(f"未知量化格式 {self.quant_format!r}")


@dataclass
class Config:
    """顶层配置。各子配置之间不交叉引用，因此可以按模块独立覆写。"""

    env: EnvConfig = field(default_factory=EnvConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)

    @property
    def algo_config(self) -> PPOConfig | SACConfig:
        """按 train.algo 取对应的超参块。"""
        return self.ppo if self.train.algo == "ppo" else self.sac
