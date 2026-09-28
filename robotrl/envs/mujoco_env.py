"""MuJoCo 环境：8 段管线的物理实现。

BaseEnv 把执行顺序钉死了，这里填的是每一段在 MuJoCo 上具体做什么：

    1. Act        _clip_action   基类实现，裁到 [-1, 1]
    2. Simulate   _apply_action  default + scale*action 写进 ctrl，未受控关节写保持目标
                  _simulate      按 decimation 次 mj_step
    3. Terminate  _terminate     委托终止管理器，判据具名可开关
    4. Reward     _reward        委托奖励管理器，权重为 0 的项不参与
    5. Reset      _reset_sim     物理复位 + 模型随机化（见下）
    6. Command    _update_command 基类留空，速度跟踪任务覆写
    7. Events     _apply_events  周期性推力
    8. Observe    _observe       按契约拼装，逐段加噪声

有两处宿主类契约之外的安排值得说明。

**模型随机化放在 _reset_sim 而不是 _apply_events(reset=True)。** 基类只在外部
调用 reset() 时会带着 reset=True 走到事件段，而 step() 内部的复位不经过那里。
随机化若只挂在那一个点上，向量化训练里环境通常只在开头 reset 一次，此后整场
训练都在用第一批摩擦和质量——"每回合随机一次"名存实亡。放进 _reset_sim 则
两处复位都覆盖到。

**未受控关节每步都写保持目标。** RobotSpec 只描述被控关节（G1 只控 15 个，
手臂 14 个不在其中），其余关节靠 position 执行器锁在 keyframe 默认角度上。
这比改 MJCF 删执行器灵活：想换受控集合只改 spec，模型文件不动。
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from robotrl.assets.loader import RobotModel, load_robot
from robotrl.assets.spec import Morphology, RobotSpec, get_spec
from robotrl.configs.schema import (
    Config,
    EventsConfig,
    ObsConfig,
    RewardConfig,
    TerrainConfig,
)
from robotrl.contracts import (
    ObsContract,
    make_obs_contract,
    make_privileged_obs_contract,
)
from robotrl.envs.base_env import BaseEnv, Obs
from robotrl.envs.events import DomainRandomizer
from robotrl.envs.managers import RewardManager, TerminationManager
from robotrl.envs.terrain import TERRAIN_GEOM_GROUP, apply_terrain

#: 各形态的默认站姿机身高度（米）。复位时把机身摆到这里，也是高度保持奖励的
#: 目标值。之所以按形态给默认值而不是塞进 RobotSpec，是因为"站多高"取决于
#: 任务（蹲伏前进、爬楼梯都会改它），而 RobotSpec 描述的是不随任务变化的属性。
_DEFAULT_BASE_HEIGHT: dict[Morphology, float] = {
    Morphology.BIPED: 0.75,
    Morphology.QUADRUPED: 0.32,
    Morphology.POINT: 0.20,
}

#: 足端 body 的名字线索。抓不住也无所谓，取不到名字时还有按高度回退的一手。
_FOOT_NAME_HINTS = ("foot", "ankle")

#: 触地即失败的部位名字线索。真正的判据是"除了足端，谁都不许碰地"，
#: 这里给的是排除足端之后剩下的那些部位。
_PENALIZED_NAME_HINTS = (
    "base",
    "torso",
    "trunk",
    "pelvis",
    "waist",
    "hip",
    "knee",
    "thigh",
    "shank",
    "calf",
)

#: 机身下方的地形采样网格间距（米）与射线起点高度（米）。起点要够高，越过
#: 机身自身的碰撞体，否则射线先打到自己的背。
_TERRAIN_SAMPLE_SPACING = 0.3
_TERRAIN_RAY_START = 1.5


class MujocoEnv(BaseEnv):
    """MuJoCo 物理环境基类。

    子类（任务）通常只需要覆写 `_update_command`、`default_reward_scales`、
    `default_termination_terms`，以及需要时覆写 `_terminate`。

    Attributes:
        spec: 形态。任务代码所有维度都从它读，不写死机器人。
        model: 本实例独占的 MjModel。随机化会改写它的数组，所以不能与其他
            实例共享（共享的那一份由 loader 持有）。
        data: 本实例的仿真状态。
        base_height_target: 高度保持奖励的目标高度，任务可按地形调整。
        base_height_min: 低于它判失败。
        max_tilt: 倾角超过它判失败（弧度）。
    """

    #: 复位时的随机化幅度。零初值有助于复现，但所有回合从同一状态起步会让策略
    #: 过拟合初始状态；幅度取到"能扰动但不至于站不稳"的量级。
    _RESET_JOINT_POS_NOISE = 0.05
    _RESET_JOINT_VEL_NOISE = 0.2
    _RESET_XY_NOISE = 0.05
    _RESET_HEIGHT_NOISE = 0.01

    def __init__(
        self,
        *,
        spec: RobotSpec | None = None,
        config: Config | None = None,
        robot_model: RobotModel | None = None,
        max_episode_steps: int = 1000,
        control_dt: float = 0.02,
        sim_dt: float = 0.002,
        seed: int = 0,
        scene: str = "scene.xml",
        init_base_height: float | None = None,
        reward_manager: RewardManager | None = None,
        termination_manager: TerminationManager | None = None,
    ) -> None:
        # config 优先于构造参数（命令行覆写才有效），显式传参又优先于默认值。
        # robot_model 可以注入：测试用它绕开 assets/loader 直接塞一份自建模型，
        # 这样环境层的管线不依赖 Menagerie 是否就位。
        if config is not None:
            control_dt = config.env.control_dt
            sim_dt = config.env.sim_dt
            max_episode_steps = config.env.max_episode_steps
            seed = config.env.seed
            self.terrain_config = config.terrain
            self.obs_config = config.obs
            self.events_config = config.events
            self.reward_config = config.reward
        else:
            self.terrain_config = TerrainConfig()
            self.obs_config = ObsConfig()
            self.events_config = EventsConfig()
            self.reward_config = RewardConfig()

        if spec is None:
            spec = get_spec(config.env.robot) if config is not None else get_spec("g1")

        super().__init__(spec, max_episode_steps=max_episode_steps)
        self.dt = control_dt
        self.sim_dt = sim_dt
        self.decimation = max(1, round(control_dt / sim_dt))
        self._np_random = np.random.default_rng(seed)

        self._build_model(robot_model, scene)
        self._build_index_maps()
        self._discover_feet()

        base_height = (
            init_base_height
            if init_base_height is not None
            else _DEFAULT_BASE_HEIGHT.get(spec.morphology, 0.5)
        )
        self.base_height_target = float(base_height)
        self.base_height_min = float(base_height) * 0.5
        self.max_tilt = float(np.deg2rad(60.0))

        self.randomizer = DomainRandomizer(
            self.events_config,
            self.obs_config,
            model=self.model,
            spec=spec,
            base_body_id=self._base_body_id,
            base_dof_adr=self._base_dof_adr,
        )
        self.reward_manager = (
            reward_manager
            if reward_manager is not None
            else RewardManager(self.reward_config, defaults=self.default_reward_scales())
        )
        self.termination_manager = (
            termination_manager
            if termination_manager is not None
            else TerminationManager(self.default_termination_terms())
        )

        self._command = np.zeros(3)
        self._last_reward_terms: dict[str, float] = {}
        self._reset_sim()

    # ------------------------------------------------------------------
    # 子类可覆写：默认奖励项与终止条件
    # ------------------------------------------------------------------

    def default_reward_scales(self) -> dict[str, float]:
        """本任务的默认奖励权重。配置里的同名字段会覆盖这里。"""
        return {}

    def default_termination_terms(self) -> dict[str, bool]:
        """本任务默认启用的失败判据。"""
        return {
            "nan_state": True,
            "base_height_low": True,
            "tilt_too_large": True,
            "illegal_contact": True,
        }

    # ------------------------------------------------------------------
    # 观测契约
    # ------------------------------------------------------------------

    @property
    def obs_contract(self) -> ObsContract:
        """标准足式契约：十二维本体感觉 + 三段逐关节量。"""
        return make_obs_contract(self.spec.n_dof)

    @property
    def critic_obs_contract(self) -> ObsContract:
        """critic 观测：policy 观测后面追加训练期才有的真值。

        没有足端的形态（例如质点）拿不到接触标志，契约里那一维宽度为 0 会直接
        报错，所以这类形态干脆不给特权观测——本来也没有可用信息。
        """
        if not self._use_privileged:
            return self.obs_contract
        return make_privileged_obs_contract(
            self.spec.n_dof,
            num_feet=self.num_feet,
            terrain_samples=self.obs_config.terrain_samples,
        )

    # ------------------------------------------------------------------
    # 模型与索引
    # ------------------------------------------------------------------

    def _build_model(self, robot_model: RobotModel | None, scene: str) -> None:
        if robot_model is None:
            robot_model = load_robot(
                self.spec, scene=scene, mutate=lambda scene_spec: self._mutate_scene(scene_spec)
            )
        self.robot_model = robot_model

        model = robot_model.model
        if DomainRandomizer.affects_model(self.events_config):
            # RobotModel.model 允许在多个环境实例间共享，而随机化就地改写
            # geom_friction / body_mass 这些数组。不复制的后果是环境之间互相
            # 覆写参数：A 的摩擦被 B 的随机化改掉，且这种 bug 只在并行训练时出现。
            model = copy.deepcopy(model)

        self.model = model
        self.data = mujoco.MjData(model)

    def _mutate_scene(self, scene_spec: mujoco.MjSpec) -> None:
        """传给 loader 的编译前钩子：把地形注入场景。

        loader 不需要知道地形是什么——它只管在编译前把 spec 交出来一次。
        """
        apply_terrain(scene_spec, self.terrain_config)

    def _build_index_maps(self) -> None:
        """把 RobotSpec 的关节顺序翻译成 MuJoCo 的数组下标。

        映射错位的后果是仿真照跑、不报错、永远学不会，所以这些下标在构造时一次
        算好，热路径上只做查表。
        """
        robot_model = self.robot_model
        model = self.model
        self.actuator_ids = np.asarray(robot_model.actuator_ids, dtype=int)
        self.qpos_ids = np.asarray(robot_model.qpos_ids, dtype=int)
        self.dof_ids = np.asarray(robot_model.dof_ids, dtype=int)
        self.held_actuator_ids = np.asarray(robot_model.held_actuator_ids, dtype=int)
        self.held_targets = np.asarray(robot_model.held_targets, dtype=float)
        self.defaults = self.spec.default_angles_array
        self.action_scale = self.spec.action_scale_array

        self._base_body_id, self._base_qpos_adr, self._base_dof_adr = self._find_base(model)

        # 足端与"非法触地"部位都按 geom 预展开成布尔/索引表，接触检测就不用
        # 每步做 body→geom 的换算。
        self._is_penalized_geom = np.zeros(model.ngeom, dtype=bool)

    @staticmethod
    def _find_base(model: mujoco.MjModel) -> tuple[int, int, int]:
        """定位浮动基座：返回 (body_id, qpos 起始下标, qvel 起始下标)。

        没有自由关节（固定基座）时后两项返回 -1，调用方据此跳过基座相关的操作。
        MuJoCo 里浮动基座必须用 free 关节，所以找第一个 free 关节就是找基座。
        """
        for jnt in range(model.njnt):
            if model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE:
                return (
                    int(model.jnt_bodyid[jnt]),
                    int(model.jnt_qposadr[jnt]),
                    int(model.jnt_dofadr[jnt]),
                )
        return 1, -1, -1

    def _discover_feet(self) -> None:
        """找出足端 body。

        优先按名字找叶节点：G1 的腿部链条里有 ankle_pitch_link 和 ankle_roll_link
        两节，都含 "ankle"，只有叶节点才是真正接地的那一个。

        名字对不上时退回"参考站姿下位置最低的几个叶节点"。命名约定不是契约的一部分，
        不该因为某个模型少写了个 foot 就让环境整个不可用。
        """
        model = self.model
        leaves = self._leaf_bodies(model)
        candidates = [
            body
            for body in leaves
            if any(hint in self._body_name(model, body).lower() for hint in _FOOT_NAME_HINTS)
        ]

        if candidates:
            feet = sorted(candidates, key=lambda body: self._body_name(model, body))
        else:
            expected = {Morphology.BIPED: 2, Morphology.QUADRUPED: 4}.get(self.spec.morphology, 0)
            heights = self._rest_heights(model)
            ordered = sorted(
                (body for body in leaves if body != self._base_body_id),
                key=lambda body: heights[body],
            )
            feet = ordered[:expected]

        self.foot_body_ids = np.asarray(feet, dtype=int)
        self.num_feet = len(feet)

        foot_of_geom = np.full(model.ngeom, -1, dtype=int)
        for index, body in enumerate(self.foot_body_ids):
            for geom in self._geoms_of_body(model, int(body)):
                foot_of_geom[geom] = index
        self._foot_of_geom = foot_of_geom

        penalized = {self._base_body_id}
        foot_bodies = set(self.foot_body_ids.tolist())
        for body in range(1, model.nbody):
            if body in foot_bodies:
                continue
            name = self._body_name(model, body).lower()
            if any(hint in name for hint in _PENALIZED_NAME_HINTS):
                penalized.add(body)
        for body in penalized:
            for geom in self._geoms_of_body(model, int(body)):
                self._is_penalized_geom[geom] = True

        # 没有足端的形态（POINT）没有接触信息可给 critic，关掉特权观测。
        self._use_privileged = bool(self.obs_config.use_privileged_critic) and self.num_feet > 0

        terrain_groups = np.zeros(6, dtype=np.uint8)
        terrain_groups[TERRAIN_GEOM_GROUP] = 1
        self._terrain_geomgroup = terrain_groups
        # mj_ray 的输出缓冲区，就地复用，免得每个采样点分配两块小数组。
        self._ray_geomid = np.zeros(1, dtype=np.int32)
        self._ray_normal = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _leaf_bodies(model: mujoco.MjModel) -> list[int]:
        """没有子 body 的 body（不含 world）。"""
        used_as_parent = set(int(p) for p in model.body_parentid[1:])
        return [body for body in range(1, model.nbody) if body not in used_as_parent]

    @staticmethod
    def _geoms_of_body(model: mujoco.MjModel, body: int) -> range:
        start = int(model.body_geomadr[body])
        return range(start, start + int(model.body_geomnum[body])) if start >= 0 else range(0)

    @staticmethod
    def _body_name(model: mujoco.MjModel, body: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""

    @staticmethod
    def _rest_heights(model: mujoco.MjModel) -> np.ndarray:
        """参考位形下各 body 的高度。

        用 qpos0 前向算一次，而不是把 body_pos 沿父子链累加：hinge 关节在参考
        位形下可能已经有非零角度，累加会算出一个不存在的位形。
        """
        reference = mujoco.MjData(model)
        reference.qpos[:] = model.qpos0
        mujoco.mj_forward(model, reference)
        return reference.xpos[:, 2].copy()

    # ------------------------------------------------------------------
    # 8 段管线
    # ------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray) -> None:
        """归一化增量 → 关节位置目标。

        target = default_angles + action_scale * action，物理侧的 PD 由 position
        执行器承担。动作的语义在 contracts.py 里定义，这里只是它的执行者。
        """
        self._previous_action = self._last_action
        self._last_action = action.copy()
        self._target_pos = self.defaults + self.action_scale * action

        self.data.ctrl[self.actuator_ids] = self._target_pos
        if self.held_actuator_ids.size:
            # 未受控关节每步都要写：ctrl 只写一次的话，被控关节的写入会把它们
            # 覆盖成 0，手臂就会垂下来。
            self.data.ctrl[self.held_actuator_ids] = self.held_targets

    def _simulate(self) -> None:
        """推进 decimation 个物理步。接触约束需要比控制频率高得多的积分频率，
        控制周期内只跑一步物理，脚底会在两次控制之间穿透地面。"""
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

    def _terminate(self) -> bool:
        """委托终止管理器。子类通过 default_termination_terms 增删判据。"""
        return self.termination_manager.check(self)

    def _reward(self) -> float:
        total, terms = self.reward_manager.compute(self)
        self._last_reward_terms = terms
        return total

    def _reset_sim(self) -> None:
        """复位物理状态，并重新随机化模型常数。"""
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)

        if self._base_qpos_adr >= 0:
            noise = self.np_random.uniform(-1.0, 1.0, size=2) * self._RESET_XY_NOISE
            data.qpos[self._base_qpos_adr : self._base_qpos_adr + 2] = noise
            data.qpos[self._base_qpos_adr + 2] = self.base_height_target + (
                self.np_random.uniform(-1.0, 1.0) * self._RESET_HEIGHT_NOISE
            )
            # 四元数按 wxyz 排列，直立即单位四元数。
            data.qpos[self._base_qpos_adr + 3 : self._base_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        joint_noise = self.np_random.uniform(-1.0, 1.0, size=self.spec.n_dof)
        data.qpos[self.qpos_ids] = self.defaults + joint_noise * self._RESET_JOINT_POS_NOISE
        data.qvel[:] = 0.0
        data.qvel[self.dof_ids] = (
            self.np_random.uniform(-1.0, 1.0, size=self.spec.n_dof) * self._RESET_JOINT_VEL_NOISE
        )

        self._last_action = np.zeros(self.spec.n_dof)
        self._previous_action = np.zeros(self.spec.n_dof)
        self._target_pos = self.defaults.copy()

        # 模型常数的随机化挂在这里而不是事件段，理由见模块文档。
        self.randomizer.apply_model(data, self.np_random)
        mujoco.mj_forward(model, data)

    def _update_command(self) -> None:
        """基类不做指令采样。速度跟踪类任务覆写本方法。"""

    def _apply_events(self, *, reset: bool) -> None:
        """域随机化的瞬时部分：周期性推力。

        reset=True 时只重排推力的时刻而不立刻推——复位后马上被推的话，
        第一步的观测里初速度已经变了，而策略还没做任何动作。
        """
        if reset:
            self.randomizer.reschedule_push(self.np_random)
            return
        self.randomizer.apply_push(self.data, self.np_random)

    def _observe(self) -> Obs:
        parts = self._policy_parts()
        policy = self.obs_contract.concat(parts)

        critic_parts: dict[str, np.ndarray] = dict(parts)
        if self._use_privileged:
            critic_parts.update(self._privileged_parts())
        critic = self.critic_obs_contract.concat(critic_parts)
        return Obs(policy=policy, critic=critic)

    def _policy_parts(self) -> dict[str, np.ndarray]:
        """policy 观测的各段。逐段加噪声，尺度读 ObsConfig.noise_scale。"""
        raw = {
            "lin_vel": self.base_linear_velocity_body(),
            "ang_vel": self.base_angular_velocity_body(),
            "proj_gravity": self.projected_gravity(),
            "cmd": self._command,
            "dof_pos": self.joint_positions(),
            "dof_vel": self.joint_velocities(),
            "last_action": self._last_action,
        }
        parts = {name: self._add_obs_noise(name, value) for name, value in raw.items()}
        parts.update(self._extra_policy_parts())
        return parts

    def _extra_policy_parts(self) -> dict[str, np.ndarray]:
        """子类扩展观测契约时在这里补新段，键名与契约段名一致。"""
        return {}

    def _privileged_parts(self) -> dict[str, np.ndarray]:
        """critic 独有的真值段。训练期才存在，导出策略时不会带上。"""
        return {
            "base_lin_vel_gt": self.base_linear_velocity(),
            "feet_contact": self.foot_contacts().astype(np.float64),
            "terrain_height": self.sample_terrain_heights(self.obs_config.terrain_samples),
        }

    def _add_obs_noise(self, segment: str, value: np.ndarray) -> np.ndarray:
        """给某一段观测加高斯噪声。

        噪声在观测组装时逐段加，而不是在状态读取处加：状态是物理真值，奖励、
        终止判定都要用干净的那一份，被噪声污染的只能是喂给策略的观测。
        """
        scale = self.randomizer.observation_noise(segment)
        if scale <= 0.0:
            return value
        return value + self.np_random.normal(0.0, scale, size=np.shape(value))

    # ------------------------------------------------------------------
    # 状态读取：任务层与奖励项通过它们取物理量
    # ------------------------------------------------------------------

    # command 由基类提供：它还要照顾 set_command 的覆盖逻辑，这里不再重复实现。

    @property
    def last_action(self) -> np.ndarray:
        """本步施加的动作。"""
        return self._last_action

    @property
    def previous_action(self) -> np.ndarray:
        """上一步施加的动作。动作平滑项用它与本步作差。"""
        return self._previous_action

    @property
    def target_joint_positions(self) -> np.ndarray:
        """本步写进 ctrl 的关节位置目标，调试与测试用。"""
        return self._target_pos

    @property
    def reward_terms(self) -> dict[str, float]:
        """上一步各奖励项的原始值（未乘权重），用于日志。"""
        return self._last_reward_terms

    def base_position(self) -> np.ndarray:
        """机身位置，世界系。"""
        return self.data.xpos[self._base_body_id].copy()

    def base_height(self) -> float:
        return float(self.data.xpos[self._base_body_id][2])

    def base_rotation(self) -> np.ndarray:
        """机身姿态矩阵，机体系到世界系。"""
        return self.data.xmat[self._base_body_id].reshape(3, 3).copy()

    def base_quaternion(self) -> np.ndarray:
        """机身姿态四元数，wxyz。"""
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.xmat[self._base_body_id])
        return quat

    def base_linear_velocity(self) -> np.ndarray:
        """机身线速度，世界系。

        自由关节的 qvel 前三项就是世界系线速度，直接用；固定基座的模型没有
        这个自由度，退回通用接口算。
        """
        if self._base_dof_adr >= 0:
            return self.data.qvel[self._base_dof_adr : self._base_dof_adr + 3].copy()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._base_body_id, velocity, 0
        )
        return velocity[3:6]

    def base_linear_velocity_body(self) -> np.ndarray:
        """机身线速度，机体系。指令也定义在机体系，两者在同一坐标系下才好比。"""
        return self.base_rotation().T @ self.base_linear_velocity()

    def base_angular_velocity_body(self) -> np.ndarray:
        """机身角速度，机体系。

        自由关节的 qvel 后三项存的**就是机体系**角速度（MuJoCo 的约定：线速度
        在世界系、角速度在体坐标系）。直接读 qvel 在这里恰好是想要的那一份，
        但这一点很容易想当然地搞反，故记一笔。
        """
        if self._base_dof_adr >= 0:
            return self.data.qvel[self._base_dof_adr + 3 : self._base_dof_adr + 6].copy()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._base_body_id, velocity, 1
        )
        return velocity[:3]

    def projected_gravity(self) -> np.ndarray:
        """重力方向在机体系下的投影。直立时约 (0, 0, -1)，倒下时接近 (0, -1, 0)。"""
        gravity = np.zeros(3)
        gravity[2] = -1.0
        return self.base_rotation().T @ gravity

    def joint_positions(self) -> np.ndarray:
        """被控关节位置与默认姿态之差。"""
        return self.data.qpos[self.qpos_ids] - self.defaults

    def joint_velocities(self) -> np.ndarray:
        """被控关节速度。"""
        return self.data.qvel[self.dof_ids].copy()

    def joint_efforts(self) -> np.ndarray:
        """被控关节当前力矩。position 执行器的输出力，用于能耗惩罚与调试。"""
        return self.data.actuator_force[self.actuator_ids].copy()

    def foot_positions(self) -> np.ndarray:
        """各足位置，世界系，(num_feet, 3)。"""
        return self.data.xpos[self.foot_body_ids].copy()

    def foot_velocities(self) -> np.ndarray:
        """各足速度，世界系，(num_feet, 3)。"""
        if self.num_feet == 0:
            return np.zeros((0, 3))
        velocity = np.zeros((self.num_feet, 3))
        buffer = np.zeros(6)
        for index, body in enumerate(self.foot_body_ids):
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, int(body), buffer, 0
            )
            velocity[index] = buffer[3:6]
        return velocity

    def foot_contacts(self) -> np.ndarray:
        """各足是否触地，(num_feet,) 布尔数组。

        逐接触点判断而不是读传感器：接触对里只要有一个 geom 属于某只脚，就算
        这只脚踩住了。传感器要改模型，接触对不用。
        """
        if self.num_feet == 0:
            return np.zeros(0, dtype=bool)
        contacts = np.zeros(self.num_feet, dtype=bool)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            for geom in (contact.geom1, contact.geom2):
                foot = self._foot_of_geom[geom]
                if foot >= 0:
                    contacts[foot] = True
        return contacts

    def illegal_contact(self) -> bool:
        """是否有非足端部位触地。"""
        if not self._is_penalized_geom.any():
            return False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self._is_penalized_geom[contact.geom1] or self._is_penalized_geom[contact.geom2]:
                return True
        return False

    def sample_terrain_heights(self, num_samples: int) -> np.ndarray:
        """机身下方地形高度采样，世界系绝对高度。

        用竖直射线投射而不是读高度场数组：高度场只对 rough 地形成立，楼梯是一堆
        方块，读数组给不出台阶的真实高度。射线同时对三种地形有效。

        射线只打地形所在的 geom group（见 terrain 模块），否则先命中的会是机器人
        自己的脚。打空时返回 0，即地面高度——比起填 NaN 让整条观测报废，填一个
        "和地面齐平"的保守值更安全。
        """
        if num_samples <= 0:
            return np.zeros(0)
        side = int(np.ceil(np.sqrt(num_samples)))
        offsets = (np.arange(side) - (side - 1) / 2.0) * _TERRAIN_SAMPLE_SPACING
        base = self.base_position()
        origin = base.copy()
        origin[2] = base[2] + _TERRAIN_RAY_START
        down = np.array([0.0, 0.0, -1.0])

        heights = np.zeros(side * side)
        for i, dx in enumerate(offsets):
            for j, dy in enumerate(offsets):
                point = origin + np.array([dx, dy, 0.0])
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    point,
                    down,
                    self._terrain_geomgroup,
                    True,
                    -1,
                    self._ray_geomid,
                    self._ray_normal,
                )
                if distance >= 0.0:
                    heights[i * side + j] = point[2] - distance
        return heights[:num_samples]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.spec.name}, n_dof={self.spec.n_dof}, "
            f"feet={self.num_feet}, dt={self.dt})"
        )
