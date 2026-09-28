"""配置系统测试。

配置是唯一"用户直接手写"的接口，因此它的容错行为决定了调试体验：写错键名
如果能立刻报错，用户五秒钟就改好了；如果被静默忽略，用户会花半天怀疑算法。
这里主要验证三件事——类型转换正确、非法输入被挡住、覆写优先级符合预期。
"""

from __future__ import annotations

import pytest

from robotrl.configs.schema import Config, EnvConfig
from robotrl.utils.config import (
    apply_overrides,
    build_config,
    config_to_dict,
    deep_merge,
    dump_yaml,
    load_config,
    load_yaml,
)

# ---------------------------------------------------------------------------
# 默认值与派生量
# ---------------------------------------------------------------------------


def test_defaults_are_valid():
    cfg = Config()
    assert cfg.env.robot == "g1"
    assert cfg.train.algo == "ppo"
    assert cfg.algo_config is cfg.ppo


def test_algo_config_follows_algo_switch():
    cfg = Config()
    cfg.train.algo = "sac"
    assert cfg.algo_config is cfg.sac


def test_decimation_derives_from_periods():
    """每个控制步内部跑多少次物理积分，由两个周期之比推出。"""
    cfg = EnvConfig(control_dt=0.02, sim_dt=0.002)
    assert cfg.decimation == 10


def test_rejects_control_slower_than_physics():
    with pytest.raises(ValueError, match="小于物理步长"):
        EnvConfig(control_dt=0.001, sim_dt=0.01)


def test_rejects_unknown_algo_and_backend():
    from robotrl.configs.schema import TrainConfig

    with pytest.raises(ValueError, match="未知算法"):
        TrainConfig(algo="dqn")
    with pytest.raises(ValueError, match="未知向量化后端"):
        TrainConfig(vec_backend="mpi")


# ---------------------------------------------------------------------------
# 字典 → Config
# ---------------------------------------------------------------------------


def test_build_config_from_nested_dict():
    cfg = build_config({"env": {"robot": "go2"}, "train": {"algo": "sac", "num_envs": 32}})
    assert cfg.env.robot == "go2"
    assert cfg.train.algo == "sac"
    assert cfg.train.num_envs == 32
    # 未提到的字段保持默认
    assert cfg.ppo.lr_actor == 3e-4


def test_unknown_key_is_rejected_with_helpful_message():
    """写错键名要报错并列出可用键，而不是静默忽略。"""
    with pytest.raises(ValueError, match="不认识"):
        build_config({"env": {"robots": "g1"}})


def test_tuple_fields_coerce_from_yaml_lists():
    """YAML 里写的是列表，字段声明是 tuple，需要转过来。"""
    cfg = build_config({"terrain": {"height_range": [0.0, 0.2]}})
    assert cfg.terrain.height_range == (0.0, 0.2)
    assert isinstance(cfg.terrain.height_range, tuple)


def test_tuple_field_rejects_wrong_arity():
    with pytest.raises(ValueError, match="需要 2 个元素"):
        build_config({"terrain": {"height_range": [0.0, 0.1, 0.2]}})


def test_dict_field_coerces_value_types():
    cfg = build_config({"reward": {"scales": {"tracking_lin_vel": 1.5, "action_rate": -0.01}}})
    assert cfg.reward.scale_of("tracking_lin_vel") == pytest.approx(1.5)
    assert cfg.reward.is_enabled("tracking_lin_vel")
    assert cfg.reward.is_enabled("action_rate")
    # 未列出的项默认为 0，即关闭
    assert not cfg.reward.is_enabled("not_listed")


# ---------------------------------------------------------------------------
# 覆写
# ---------------------------------------------------------------------------


def test_overrides_parse_scalar_types():
    """命令行传进来的都是字符串，要按目标类型还原。"""
    data = apply_overrides(
        {},
        [
            "train.num_envs=128",
            "ppo.lr_actor=1e-4",
            "obs.use_obs_noise=false",
            "env.robot=go2",
        ],
    )
    assert data["train"]["num_envs"] == 128
    assert isinstance(data["train"]["num_envs"], int)
    assert data["ppo"]["lr_actor"] == pytest.approx(1e-4)
    assert data["obs"]["use_obs_noise"] is False
    assert data["env"]["robot"] == "go2"


def test_overrides_survive_round_trip_through_config():
    cfg = load_config(overrides=["train.num_envs=256", "ppo.clip_ratio=0.1"])
    assert cfg.train.num_envs == 256
    assert cfg.ppo.clip_ratio == pytest.approx(0.1)


def test_overrides_create_missing_intermediate_paths():
    data = apply_overrides({}, ["a.b.c=1"])
    assert data == {"a": {"b": {"c": 1}}}


def test_override_requires_equals_sign():
    with pytest.raises(ValueError, match="格式不对"):
        apply_overrides({}, ["train.num_envs"])


def test_deep_merge_prefers_override_and_keeps_siblings():
    base = {"env": {"robot": "g1", "control_dt": 0.02}, "train": {"num_envs": 64}}
    over = {"env": {"robot": "go2"}}
    merged = deep_merge(base, over)

    assert merged["env"]["robot"] == "go2"
    assert merged["env"]["control_dt"] == 0.02  # 同级字段不丢
    assert merged["train"]["num_envs"] == 64  # 未涉及的子树原样保留
    assert base["env"]["robot"] == "g1"  # 不修改入参


# ---------------------------------------------------------------------------
# 文件读写
# ---------------------------------------------------------------------------


def test_yaml_round_trip(tmp_path):
    cfg = Config()
    cfg.env.robot = "h1"
    cfg.reward.scales = {"tracking_lin_vel": 1.5}

    path = dump_yaml(config_to_dict(cfg), tmp_path / "cfg.yaml")
    reloaded = build_config(load_yaml(path))

    assert reloaded.env.robot == "h1"
    assert reloaded.reward.scale_of("tracking_lin_vel") == pytest.approx(1.5)


def test_missing_config_file_reports_path(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_yaml(missing)


def test_empty_yaml_falls_back_to_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert build_config(load_yaml(path)).env.robot == "g1"
