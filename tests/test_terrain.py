"""地形注入。

地形通过 loader 的 mutate 钩子在编译前改 MjSpec。这里直接调 apply_terrain
再 compile，走的是与真实加载完全相同的那条路径，只是模型换成内嵌的最小 MJCF，
所以不需要 Menagerie。
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from mini_robot import (
    MINI_BASE_HEIGHT,
    MINI_SPEC,
    MINI_XML,
    build_robot_model,
    compile_with_terrain,
)

from robotrl.configs.schema import Config, TerrainConfig
from robotrl.envs import mujoco_env as mujoco_env_module
from robotrl.envs.mujoco_env import MujocoEnv
from robotrl.envs.terrain import TERRAIN_GEOM_GROUP


def geom_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        for geom in range(model.ngeom)
    ]


def geom_by_name(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def world_geom_pos(model: mujoco.MjModel, geom: int) -> np.ndarray:
    """几何体的世界位置。

    `model.geom_pos` 是相对父 body 的，直接用会把台阶的位置算少一个 body 的偏移，
    而地形刚好是"body 带位置、geom 在原点"的写法，很容易踩到。
    """
    body = model.geom_bodyid[geom]
    return np.asarray(model.body_pos[body]) + np.asarray(model.geom_pos[geom])


def stair_tops(model: mujoco.MjModel) -> list[float]:
    """每个台阶几何的顶面高度，按台阶序号排好。"""
    tops = []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if name.startswith("terrain_step_geom_"):
            tops.append(float(world_geom_pos(model, geom)[2] + model.geom_size[geom][2]))
    return sorted(tops)


# ---------------------------------------------------------------------------
# 三种地形
# ---------------------------------------------------------------------------


def test_flat_replaces_the_scene_floor():
    """自带的地面必须删掉。

    留着的话两块地面重叠在 z=0：接触点翻倍，摩擦取两者的较大值，
    物理变得难以解释——而症状只会表现为"策略学得慢"。
    """
    model = compile_with_terrain(TerrainConfig(kind="flat"))
    names = geom_names(model)

    assert "floor" not in names
    assert "terrain_flat" in names
    assert model.nhfield == 0


def test_rough_terrain_builds_a_height_field():
    model = compile_with_terrain(TerrainConfig(kind="rough"))

    assert model.nhfield == 1
    data = model.hfield_data
    assert data.max() - data.min() > 0.0  # 有起伏，不是一张平面


def test_rough_terrain_grid_follows_cell_size():
    config = TerrainConfig(kind="rough", terrain_size=(4.0, 4.0), cell_size=0.5)
    model = compile_with_terrain(config)

    assert model.hfield_nrow[0] == 9  # 4 / 0.5 + 1
    assert model.hfield_ncol[0] == 9
    assert model.hfield_size[0][0] == pytest.approx(2.0)  # 半边长


def test_rough_terrain_height_range_is_respected():
    """起伏范围由几何体的 z 偏移与 size 的 elevation 分量共同决定。

    MuJoCo 编译时会把高度场数据重新归一化到 [0, 1]，所以"最低处也有 low 这么高"
    这件事只能靠几何体整体抬高来实现，靠缩放数据是做不到的。
    """
    config = TerrainConfig(kind="rough", height_range=(0.05, 0.2))
    model = compile_with_terrain(config)
    low, high = config.height_range

    geom = geom_by_name(model, "terrain_hfield_geom")
    assert world_geom_pos(model, geom)[2] == pytest.approx(low)
    assert model.hfield_size[0][2] == pytest.approx(high - low)
    assert model.hfield_data.min() == pytest.approx(0.0)
    assert model.hfield_data.max() == pytest.approx(1.0)


def test_rough_terrain_is_reproducible():
    """同一份配置生成同一片地形，否则两次训练的对比里混进了地形差异。"""
    first = compile_with_terrain(TerrainConfig(kind="rough"))
    second = compile_with_terrain(TerrainConfig(kind="rough"))

    assert first.hfield_data == pytest.approx(second.hfield_data)


def test_stairs_stack_one_box_per_step():
    config = TerrainConfig(kind="stairs", num_steps=4, step_height=0.1, step_width=0.4)

    model = compile_with_terrain(config)

    assert stair_tops(model) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    # 每级都是从地面堆到顶面的实心块，没有薄板下面的缝隙可卡
    terrain_geoms = [name for name in geom_names(model) if name.startswith("terrain_")]
    assert len(terrain_geoms) == 1 + config.num_steps + 1  # 基准平面 + 台阶 + 顶部平台


def test_stairs_start_after_a_run_up():
    """楼梯不从头开始：复位时贴着垂直面的话，第一步就成了抬腿上墙。"""
    config = TerrainConfig(kind="stairs", num_steps=3, step_width=0.3)
    model = compile_with_terrain(config)

    first_step = geom_by_name(model, "terrain_step_geom_0")
    front_edge = world_geom_pos(model, first_step)[0] - model.geom_size[first_step][0]

    assert front_edge > 0.3


def test_unknown_terrain_kind_is_rejected():
    with pytest.raises(ValueError, match="未知地形"):
        compile_with_terrain(TerrainConfig(kind="swamp"))


# ---------------------------------------------------------------------------
# 地形几何与环境层的约定
# ---------------------------------------------------------------------------


def test_terrain_geoms_live_in_their_own_group():
    """地形几何必须独占一个 geom group。

    机身下方的高度采样靠竖直射线，射线只认 geomgroup 这一个过滤器；不独占的话，
    射线会先打到机器人自己的脚，采样出来的"地形高度"永远等于脚的高度。
    """
    for kind in ("flat", "rough", "stairs"):
        model = compile_with_terrain(TerrainConfig(kind=kind))
        terrain_geoms = [
            geom
            for geom in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith(
                "terrain_"
            )
        ]
        assert terrain_geoms
        assert all(model.geom_group[geom] == TERRAIN_GEOM_GROUP for geom in terrain_geoms)


def test_height_sampling_hits_real_terrain():
    """地形就位后，机身下方的采样应当读到地形本身的高度。

    采样值必须落在配置的高度范围内，且九点之间有起伏：只验"非零"的话，
    射线打到机器人自己的脚也能通过。
    """
    config = TerrainConfig(kind="rough", height_range=(0.05, 0.15))
    model = compile_with_terrain(config)
    env = MujocoEnv(
        spec=MINI_SPEC,
        robot_model=build_robot_model(model=model),
        init_base_height=MINI_BASE_HEIGHT * 2.0,  # 悬在地形上方，只管采样
        seed=0,
    )
    env.reset(seed=0)

    heights = env.sample_terrain_heights(9)

    assert heights.shape == (9,)
    assert np.all(np.isfinite(heights))
    assert np.all(heights >= config.height_range[0] - 1e-6)
    assert np.all(heights <= config.height_range[1] + 1e-6)
    assert heights.max() - heights.min() > 0.0


def test_height_sampling_works_on_stairs():
    """楼梯是方块不是高度场，采样同样要能读出台阶的真实高度。

    采样值必须等于某级台阶的顶面：只要"非零"，射线打到机器人自己的脚也能通过；
    只有对得上台阶高度，才说明它读到的确实是地形。
    """
    config = TerrainConfig(kind="stairs", num_steps=6, step_height=0.1, step_width=0.3)
    model = compile_with_terrain(config)
    env = MujocoEnv(
        spec=MINI_SPEC,
        robot_model=build_robot_model(model=model),
        init_base_height=1.2,
        seed=0,
    )
    env.reset(seed=0)
    # 机身挪到第三级上方（台阶从 0.6 m 起，每级 0.3 m 宽），
    # 3x3 采样网格铺开 ±0.3 m，横跨第一到第四级。
    env.data.qpos[env._base_qpos_adr] = 1.2
    mujoco.mj_forward(env.model, env.data)

    heights = env.sample_terrain_heights(9)

    step_tops = [i * config.step_height for i in range(config.num_steps + 1)]
    assert np.all(np.isfinite(heights))
    for height in heights:
        assert min(abs(height - top) for top in step_tops) < 0.02
    assert heights.max() == pytest.approx(0.4, abs=0.02)


def test_robot_joints_survive_terrain_injection():
    """地形只往 worldbody 里加几何和 body，机器人的关节映射不该受影响。

    台阶是有 body 的（为了带位置），所以这条用例真正在防的是"注入地形把 body
    下标顶掉了"这类错误。
    """
    plain = build_robot_model()
    with_stairs = build_robot_model(model=compile_with_terrain(TerrainConfig(kind="stairs")))

    assert with_stairs.actuator_ids.tolist() == plain.actuator_ids.tolist()
    assert with_stairs.qpos_ids.tolist() == plain.qpos_ids.tolist()
    assert with_stairs.dof_ids.tolist() == plain.dof_ids.tolist()


def test_terrain_reaches_the_scene_through_the_loader_hook(monkeypatch):
    """地形是通过 loader 的 mutate 钩子进到场景里的，不需要 loader 知道地形是什么。

    这里用一个假的 loader 复现真实加载的两步：拿到场景 spec、调一次钩子、编译。
    验的是环境交给 loader 的东西对不对，而不是 loader 自己怎么实现。
    """
    captured: dict[str, object] = {}

    def fake_load_robot(spec, *, scene="scene.xml", mutate=None):
        scene_spec = mujoco.MjSpec.from_string(MINI_XML)
        if mutate is not None:
            mutate(scene_spec)
        compiled = scene_spec.compile()
        captured["scene"] = scene
        captured["nhfield"] = compiled.nhfield
        return build_robot_model(spec, model=compiled)

    monkeypatch.setattr(mujoco_env_module, "load_robot", fake_load_robot)

    config = Config()
    config.terrain.kind = "rough"
    env = MujocoEnv(spec=MINI_SPEC, config=config, seed=0)

    assert captured["scene"] == "scene.xml"
    assert captured["nhfield"] == 1  # 钩子确实在编译前把地形注了进去
    assert env.model.nhfield == 1
