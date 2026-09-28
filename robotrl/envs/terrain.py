"""地形生成：平地 / 崎岖高度场 / 楼梯。

地形不改模型文件，而是在编译前通过 loader 的 mutate 钩子往 MjSpec 里插几何体。
这样地形与机器人彻底解耦：同一个 Go2 场景配 rough 就是崎岖地形，配 stairs 就是
楼梯，改的只是配置里的一个词。若改成维护三份场景 XML，加一种地形要动三个文件，
而且机器人的改动得同步三遍。

地面几何统一放进 TERRAIN_GEOM_GROUP 这个 geom group。原因在机身下方的地形高度
采样：那里靠竖直射线投射问"我脚下多高"，射线必须只看地面、不看机器人自己的脚。
mj_ray 支持的过滤手段只有 geomgroup，所以地形必须独占一组。

高度场是随机生成的，但用固定种子的生成器——同一份配置必须生成同一片地形，
否则两次训练的对比里混进了地形差异，结论就不可信了。地形课程（越训越难）
是另一个话题，等有需要时再引入。
"""

from __future__ import annotations

import mujoco
import numpy as np

from robotrl.configs.schema import TerrainConfig

#: 地形几何独占的 geom group。射线投射靠它把机器人本体排除在外。
#:
#: 这个值曾经是 3，那是错的。MuJoCo Menagerie 的约定是**视觉几何放 group 2、
#: 碰撞几何放 group 3**，所以地形和机器人自己的碰撞体挤在同一组里，射线过滤
#: 根本排除不掉本体：平地（真值处处为 0）上实测 9 条射线里有 1 条打中 G1 的
#: torso_link，读回 1.32 米——这个数完全是机器人自身几何到射线的距离，却被
#: 当成"机身下方的地形高度"喂给了 critic。三个形态都中招（H1 读到 1.80 米，
#: Go2 读到 0.50 米）。
#:
#: 现在选 1：三个形态都只用到 group 2 和 3，group 0/1/4/5 是空的。选 1 而不是 0
#: 是因为 0 是"默认组"，将来往场景里加东西的人多半会顺手用 0。另外 group 1
#: 在 MjvOption 的默认设置里是**渲染开启**的，所以地面顺带恢复了可见——在那之前
#: 回放窗口里机器人是悬浮在一片蓝色里的，看不出它站在哪。
#:
#: 代价是这条不变式无法从代码本身保证：换一个把碰撞体放在 group 1 的机器人就会
#: 静默复现这个 bug。所以 MujocoEnv 构造时会当场核对地形组是不是独占了，见
#: `_check_terrain_group_is_exclusive`。
TERRAIN_GEOM_GROUP = 1

#: 地形几何名字的统一前缀。MujocoEnv 构造时靠它确认地形组里没有混进机器人的
#: 几何——这条检查是上面那个 bug 唯一的防线，见
#: `MujocoEnv._check_terrain_group_is_exclusive`。
TERRAIN_NAME_PREFIX = "terrain_"

#: 场景里原有的地面名字。注入地形前先删掉，否则会和注入的地面叠在一起，
#: 接触点倍增、摩擦系数取两者较大值，物理变得难以解释。
_EXISTING_GROUND_NAMES = ("floor", "ground")

#: 高度场的 base_z 必须为正（MuJoCo 的硬性要求）。0.01 让高度场的底面略低于
#: 下面的平面，脚踩到高度场最低处时不会同时压到平面。
_HFIELD_BASE_Z = 0.01

#: 楼梯前的助跑段长度（米）。起点就是台阶的话，复位时机器人贴着垂直面，
#: 第一步必须先抬腿再前进，学步态之前先学会撞墙。
_STAIRS_RUN_UP = 0.6

#: 地形几何的摩擦系数，与 Menagerie 场景里的默认地面一致。
_TERRAIN_FRICTION = (1.0, 0.005, 0.0001)

#: 高度场左下角的生成种子。固定值是为了可复现，见模块文档。
_HFIELD_SEED = 0


def apply_terrain(
    scene: mujoco.MjSpec,
    config: TerrainConfig,
    *,
    rng: np.random.Generator | None = None,
) -> None:
    """把地形注入场景。签名即 loader 的 mutate 钩子。

    Args:
        scene: 编译前的场景 spec，就地修改。
        config: 地形参数。
        rng: 高度场噪声源。默认固定种子，保证同一份配置生成同一片地形。
    """
    builders = {
        "flat": _build_flat,
        "rough": _build_rough,
        "stairs": _build_stairs,
    }
    if config.kind not in builders:
        raise ValueError(f"未知地形 {config.kind!r}，可用：{sorted(builders)}")

    _remove_existing_ground(scene)
    builders[config.kind](scene, config, rng)


# ---------------------------------------------------------------------------
# 三种地形
# ---------------------------------------------------------------------------


def _build_flat(
    scene: mujoco.MjSpec, config: TerrainConfig, rng: np.random.Generator | None
) -> None:
    """平地。尺寸取自配置，比 Menagerie 自带的 10x10 更容易按需收缩。"""
    _add_ground_plane(scene, config, name="terrain_flat")


def _build_rough(
    scene: mujoco.MjSpec, config: TerrainConfig, rng: np.random.Generator | None
) -> None:
    """崎岖地形。用高度场而不是一堆方块拼，因为高度场是连续曲面，脚落上去
    没有竖直的接缝，接触力平滑，策略不会学到"踩缝"这种只在这里成立的技巧。

    高度值先随机再平滑。不平滑的逐格独立噪声是一片尖刺，落脚点变成拼运气。

    高度范围由两处共同决定：几何体的 z 偏移给出最低高度，size 的 elevation
    分量给出起伏幅度。之所以不用"把数据缩放到 [low, high]"来一步到位，是因为
    MuJoCo 编译时会把高度场数据重新归一化到 [0, 1]——自己缩放的数据会被抹掉，
    只有几何体的位置和尺寸是说了算的。
    """
    # 高度场下面仍然铺一层平面：走出地形范围时不至于掉进虚空。
    _add_ground_plane(scene, config, name="terrain_rough_base")

    size_x, size_y = config.terrain_size
    nrow = max(2, int(size_x / config.cell_size) + 1)
    ncol = max(2, int(size_y / config.cell_size) + 1)
    low, high = config.height_range
    if high <= low:
        raise ValueError(f"height_range 的上界必须大于下界，得到 {config.height_range}")

    generator = rng if rng is not None else np.random.default_rng(_HFIELD_SEED)
    heights = _smooth(generator.uniform(low, high, size=(nrow, ncol)))

    hfield = scene.add_hfield(
        name="terrain_hfield",
        nrow=nrow,
        ncol=ncol,
        size=[size_x / 2.0, size_y / 2.0, high - low, _HFIELD_BASE_Z],
        userdata=_normalize(heights).ravel().tolist(),
    )
    geom = scene.worldbody.add_geom(
        name="terrain_hfield_geom",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        pos=[0.0, 0.0, low],
        rgba=[0.45, 0.42, 0.38, 1.0],
    )
    geom.hfieldname = hfield.name
    geom.group = TERRAIN_GEOM_GROUP
    geom.friction = list(_TERRAIN_FRICTION)


def _build_stairs(
    scene: mujoco.MjSpec, config: TerrainConfig, rng: np.random.Generator | None
) -> None:
    """楼梯。每级是一个实心方块，从地面一直堆到该级顶面。

    用实心而非薄板：薄板下面有缝隙，脚趾或者碰撞体边缘可能卡进去，接触求解
    会给出方向诡异的力。实心块没有内表面，接触只会发生在顶面和侧面。
    """
    _add_ground_plane(scene, config, name="terrain_stairs_base")

    size_y = config.terrain_size[1]
    for i in range(config.num_steps):
        top = (i + 1) * config.step_height
        x_center = _STAIRS_RUN_UP + (i + 0.5) * config.step_width
        body = scene.worldbody.add_body(name=f"terrain_step_{i}", pos=[x_center, 0.0, 0.0])
        geom = body.add_geom(
            name=f"terrain_step_geom_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.0, top / 2.0],
            size=[config.step_width / 2.0, size_y / 2.0, top / 2.0],
            rgba=[0.5, 0.5, 0.5, 1.0],
        )
        geom.group = TERRAIN_GEOM_GROUP
        geom.friction = list(_TERRAIN_FRICTION)

    # 顶上加一段平台。没有它，走到最后一级的尽头就是悬崖，策略得在"上楼梯"
    # 之外再学一个"别掉下去"，两件事混在一起训不出来。
    landing_top = config.num_steps * config.step_height
    landing_len = (
        config.terrain_size[0] / 2.0 - _STAIRS_RUN_UP - config.num_steps * config.step_width
    )
    if landing_len > 0:
        x_center = _STAIRS_RUN_UP + config.num_steps * config.step_width + landing_len / 2.0
        body = scene.worldbody.add_body(name="terrain_stairs_landing", pos=[x_center, 0.0, 0.0])
        geom = body.add_geom(
            name="terrain_stairs_landing_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.0, landing_top / 2.0],
            size=[landing_len / 2.0, size_y / 2.0, landing_top / 2.0],
            rgba=[0.5, 0.5, 0.5, 1.0],
        )
        geom.group = TERRAIN_GEOM_GROUP
        geom.friction = list(_TERRAIN_FRICTION)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _add_ground_plane(scene: mujoco.MjSpec, config: TerrainConfig, *, name: str) -> None:
    """在 z=0 铺一张平面。平面在 MuJoCo 里是无限大的，尺寸只影响显示。"""
    size_x, size_y = config.terrain_size
    geom = scene.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, 0.0],
        size=[size_x / 2.0, size_y / 2.0, 0.1],
        rgba=[0.4, 0.42, 0.4, 1.0],
    )
    geom.group = TERRAIN_GEOM_GROUP
    geom.friction = list(_TERRAIN_FRICTION)


def _remove_existing_ground(scene: mujoco.MjSpec) -> None:
    """删掉场景自带的 `floor` / `ground`。

    按名字找而不是按"位置最低的 geom"找：删除错了对象的代价是场景少一块地面，
    症状会延迟到第一次接触才暴露，很难查。
    """
    for geom in list(scene.worldbody.geoms):
        if geom.name in _EXISTING_GROUND_NAMES:
            scene.delete(geom)


def _smooth(field: np.ndarray, passes: int = 1) -> np.ndarray:
    """3x3 均值滤波，边界按边缘值延拓。"""
    for _ in range(passes):
        padded = np.pad(field, 1, mode="edge")
        acc = np.zeros_like(field)
        for dx in range(3):
            for dy in range(3):
                acc += padded[dx : dx + field.shape[0], dy : dy + field.shape[1]]
        field = acc / 9.0
    return field


def _normalize(field: np.ndarray) -> np.ndarray:
    """把高度场线性缩放到 [0, 1]，即 MuJoCo 期望的 userdata 形式。

    平滑之后重新归一化是必须的：平滑会把极值抹平，直接拿原始量去归一化，
    地形的实际起伏会比配置的窄，而且随格点密度变化——同一份 height_range
    在不同的 cell_size 下会生成不同陡峭的地形。
    """
    span = float(field.max() - field.min())
    if span < 1e-9:
        return np.zeros_like(field)
    return (field - field.min()) / span
