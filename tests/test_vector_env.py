"""批式环境测试。

这个模块要守住的性质只有一条，但它决定了训练能不能信：**换后端不换结果**。
串行和上多进程是同一份配置的两种推进方式，喂同样的种子、同样的动作，
必须得到逐位相同的观测与回报。一旦分叉，训练曲线的差异就说不清是算法
调参的效果还是并行方式带来的，实验也就白做了。

其余用例围绕接口的边界：动作形状不对要当场报错而不是让矩阵乘法在后面
炸掉、进程数超过环境数时不多开空转的进程、子进程不吭声时要有超时而不是
永远挂着。

toy 环境不依赖 MuJoCo，所以这些用例在没抓 Menagerie 模型的机器上照样跑得动。
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from robotrl.configs.schema import Config
from robotrl.envs import make

# _WorkerGroup 是私有类，但这里测的正是它的分组与启动状态机——通过
# SubprocessVecEnv 间接测会慢上几十倍，还得靠时序去碰错误分支。
from robotrl.envs.vector_env import (
    SerialVecEnv,
    SubprocessVecEnv,
    VecEnv,
    _WorkerGroup,
    default_num_workers,
    make_vec_env,
    split_counts,
)


def _cfg(num_envs: int = 2, backend: str = "serial") -> Config:
    cfg = Config()
    cfg.env.max_episode_steps = 40
    cfg.env.seed = 0
    cfg.train.num_envs = num_envs
    cfg.train.vec_backend = backend
    return cfg


def _toy_envs(n: int) -> list:
    cfg = _cfg(n)
    return [make("toy", config=cfg) for _ in range(n)]


# ---------------------------------------------------------------------------
# 进程划分
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_envs", "num_workers", "expected"),
    [
        (8, 4, [2, 2, 2, 2]),
        (10, 4, [3, 3, 2, 2]),  # 余数给前几个，各进程差不超过 1
        (3, 8, [1, 1, 1]),  # 进程数超过环境数，多出来的丢掉
        (1, 1, [1]),
        (7, 1, [7]),
    ],
)
def test_split_counts_distributes_evenly(num_envs, num_workers, expected):
    assert split_counts(num_envs, num_workers) == expected


@pytest.mark.parametrize(("num_envs", "num_workers"), [(13, 4), (64, 16), (5, 3)])
def test_split_counts_conserves_envs_and_balances(num_envs, num_workers):
    counts = split_counts(num_envs, num_workers)

    assert sum(counts) == num_envs, "分出去的环境总数必须不增不减"
    assert max(counts) - min(counts) <= 1, "各进程环境数差超过 1 会让最慢的进程拖住整批"
    assert all(c > 0 for c in counts), "不该出现分不到环境却照样启动的进程"


@pytest.mark.parametrize(("num_envs", "num_workers"), [(0, 4), (-1, 2), (4, 0), (4, -2)])
def test_split_counts_rejects_nonpositive(num_envs, num_workers):
    with pytest.raises(ValueError):
        split_counts(num_envs, num_workers)


def test_worker_group_never_starts_more_processes_than_envs():
    """进程数超过环境数时，空转的进程要被丢掉而不是照开。

    构造不启动进程，所以这里能直接检查分组结果，不必真开 8 个进程。
    """
    group = _WorkerGroup(_cfg(), "toy", num_envs=2, num_workers=8, seed=0)

    assert group.num_workers == 2
    assert group.counts == [1, 1]
    assert group._offsets == [0, 1], "各环境在全局的起始序号要连续，种子才错得开"


def test_default_num_workers_capped_by_env_count():
    """进程数取 min(环境数, 核数)：环境只有 2 个时开 16 个进程是纯亏。"""
    assert default_num_workers(2) == 2
    assert default_num_workers(10_000) == min(10_000, mp.cpu_count() or 1)


# ---------------------------------------------------------------------------
# 串行后端
# ---------------------------------------------------------------------------


def test_vec_env_rejects_nonpositive_num_envs():
    class _Bare(VecEnv):
        def __init__(self, n: int) -> None:
            super().__init__(n)

        spec = obs_contract = critic_obs_contract = eval_env = None

        def reset(self, seeds=None):  # pragma: no cover - 抽象方法占位
            raise NotImplementedError

        def step_batch(self, actions):  # pragma: no cover - 抽象方法占位
            raise NotImplementedError

        def close(self) -> None:
            pass

    with pytest.raises(ValueError, match="环境数必须为正"):
        _Bare(0)


def test_serial_rejects_empty_env_list():
    with pytest.raises(ValueError, match="环境列表不能为空"):
        SerialVecEnv([])


def test_serial_dims_and_contracts_come_from_first_env():
    envs = _toy_envs(3)
    vec = SerialVecEnv(envs)

    assert vec.num_envs == 3
    assert vec.obs_dim == envs[0].obs_dim
    assert vec.critic_obs_dim == envs[0].critic_obs_dim
    assert vec.action_dim == envs[0].action_dim
    # 用等值而不是同一：契约可能是每次访问现构造的，比的是内容不是对象身份。
    assert vec.obs_contract == envs[0].obs_contract


def test_serial_eval_env_is_the_first_instance():
    """评估环境复用第一个实例，与改造前的行为一致。"""
    envs = _toy_envs(2)
    vec = SerialVecEnv(envs)
    assert vec.eval_env is envs[0]


def test_serial_reset_returns_one_observation_per_env():
    vec = SerialVecEnv(_toy_envs(4))
    obs = vec.reset([0, 1, 2, 3])

    assert len(obs) == vec.num_envs
    for o in obs:
        assert o.policy.shape == (vec.obs_dim,)
        assert o.critic.shape == (vec.critic_obs_dim,)


def test_serial_reset_rejects_wrong_seed_count():
    vec = SerialVecEnv(_toy_envs(3))
    with pytest.raises(ValueError, match="种子数 2 与环境数 3 不一致"):
        vec.reset([0, 1])


@pytest.mark.parametrize(
    "shape",
    [(3, 2), (1, 2), (2, 3), (2,), (2, 2, 1)],
)
def test_serial_step_batch_rejects_wrong_action_shape(shape):
    """形状不对要在入口报错。放到后面就是矩阵乘法里一个难查的维度错误。"""
    vec = SerialVecEnv(_toy_envs(2))
    with pytest.raises(ValueError, match="动作形状应为"):
        vec.step_batch(np.zeros(shape, dtype=np.float32))


def test_serial_step_batch_accepts_list_and_castable_dtype():
    vec = SerialVecEnv(_toy_envs(2))
    results = vec.step_batch([[0.0, 0.0], [0.0, 0.0]])

    assert len(results) == 2
    assert all(isinstance(r.reward, float) for r in results)


def test_serial_close_closes_every_env():
    class _Stub:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    stubs = [_Stub() for _ in range(3)]
    SerialVecEnv(stubs).close()  # type: ignore[arg-type]

    assert all(s.closed for s in stubs)


def test_vec_env_is_a_context_manager():
    """with 退出时自动收摊。训练脚本靠这个保证异常退出后不留子进程。"""
    with SerialVecEnv(_toy_envs(2)) as vec:
        assert vec.num_envs == 2
    assert vec.num_envs == 2


# ---------------------------------------------------------------------------
# 换后端不换结果
# ---------------------------------------------------------------------------


def test_backend_choice_is_read_from_config():
    cfg = _cfg(2, backend="subprocess")
    with make_vec_env(cfg, "toy") as vec:
        assert isinstance(vec, SubprocessVecEnv)

    cfg = _cfg(2, backend="serial")
    with make_vec_env(cfg, "toy") as vec:
        assert isinstance(vec, SerialVecEnv)


def test_make_vec_env_rejects_unknown_backend():
    with pytest.raises(ValueError, match="未知向量化后端"):
        make_vec_env(_cfg(2), "toy", backend="thread")


@pytest.fixture(scope="module")
def both_backends():
    """同一份配置的串行与多进程两份实现，用完一起关掉。"""
    cfg = _cfg(2)
    serial = make_vec_env(cfg, "toy", num_envs=2, seed=0, backend="serial")
    parallel = make_vec_env(cfg, "toy", num_envs=2, seed=0, backend="subprocess")
    try:
        yield serial, parallel
    finally:
        serial.close()
        parallel.close()


def test_two_backends_report_identical_dims(both_backends):
    serial, parallel = both_backends

    assert parallel.num_envs == serial.num_envs
    assert parallel.obs_dim == serial.obs_dim
    assert parallel.critic_obs_dim == serial.critic_obs_dim
    assert parallel.action_dim == serial.action_dim
    assert parallel.obs_contract == serial.obs_contract


def test_two_backends_reset_identically(both_backends):
    """同种子复位必须逐位相同。差一点，训练曲线就不可比了。"""
    serial, parallel = both_backends
    seeds = [0, 1]

    a = serial.reset(seeds)
    b = parallel.reset(seeds)

    assert len(a) == len(b) == 2
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.policy, y.policy)
        np.testing.assert_array_equal(x.critic, y.critic)


def test_two_backends_roll_out_identically(both_backends):
    """同样的动作序列推进若干步，观测、回报、终止标志都要逐位相同。

    这是整个模块最要紧的一条断言：训练循环只有一条代码路径，区别只在
    "谁来推进环境"。两者若不等价，换后端就等于换了个任务。
    """
    serial, parallel = both_backends
    seeds = [0, 1]
    serial.reset(seeds)
    parallel.reset(seeds)

    rng = np.random.default_rng(0)
    n_steps = 25
    for step in range(n_steps):
        actions = rng.uniform(-1.0, 1.0, size=(2, serial.action_dim)).astype(np.float32)
        ra = serial.step_batch(actions)
        rb = parallel.step_batch(actions)

        for i, (x, y) in enumerate(zip(ra, rb, strict=True)):
            assert x.reward == pytest.approx(y.reward, abs=0.0), f"第 {step} 步 env{i} 回报不同"
            assert x.terminated == y.terminated, f"第 {step} 步 env{i} 终止标志不同"
            assert x.truncated == y.truncated, f"第 {step} 步 env{i} 截断标志不同"
            np.testing.assert_array_equal(x.obs.policy, y.obs.policy)
            np.testing.assert_array_equal(x.obs.critic, y.obs.critic)


# ---------------------------------------------------------------------------
# 多进程后端的接口细节
# ---------------------------------------------------------------------------


def test_subprocess_workers_are_materially_fewer_than_envs_when_capped():
    cfg = _cfg(4)
    with SubprocessVecEnv(cfg, "toy", num_envs=4, num_workers=2) as vec:
        assert vec.num_workers == 2
        assert vec.group.counts == [2, 2]


def test_subprocess_eval_env_lives_in_this_process(both_backends):
    """评估环境是本进程里的独立实例，不经过 IPC。

    逐回合评估如果每步都要往返一次子进程，1000 步的回合会慢到不可用。
    """
    _, parallel = both_backends
    obs, _ = parallel.eval_env.reset(seed=123)
    assert obs.policy.shape == (parallel.obs_dim,)


def test_subprocess_reset_defaults_to_seed_offset(both_backends):
    """不给种子时按 seed + i 铺开，保证各环境走不同轨迹。"""
    _, parallel = both_backends
    obs = parallel.reset()
    assert len(obs) == parallel.num_envs


def test_subprocess_repr_states_env_and_worker_counts(both_backends):
    _, parallel = both_backends
    text = repr(parallel)

    assert f"{parallel.num_envs} 环境" in text
    assert f"{parallel.num_workers} 进程" in text


def test_worker_group_recv_times_out_instead_of_hanging():
    """子进程静默死掉时，父进程必须报错，不能永远等在 recv 上。

    真实场景是子进程段错误：管道没关、消息永远不来，没有超时的实现会
    在 recv 上挂到天荒地老，连一句错误都打不出来。
    """

    class _SilentConn:
        def poll(self, timeout: float | None = None) -> bool:
            return False

        def recv(self):  # pragma: no cover - 不该被调到
            raise AssertionError("超时路径不应该走到 recv")

    group = _WorkerGroup(_cfg(), "toy", num_envs=1, num_workers=1, seed=0)
    with pytest.raises(TimeoutError, match="等待子进程超过"):
        group._recv(_SilentConn(), timeout=0.01)


def test_worker_group_refuses_to_dispatch_before_start():
    """没启动就分发会返回空批，那是个静默的错误答案。"""
    group = _WorkerGroup(_cfg(), "toy", num_envs=2, num_workers=2, seed=0)

    with pytest.raises(RuntimeError, match="尚未启动"):
        group.reset([0, 1])
    with pytest.raises(RuntimeError, match="尚未启动"):
        group.step(np.zeros((2, 2)))


def test_worker_group_cannot_be_started_twice(both_backends):
    """重复启动会多开一倍进程，要当场挡住。"""
    _, parallel = both_backends
    with pytest.raises(RuntimeError, match="已经启动过"):
        parallel.group.start()


def test_close_is_safe_without_any_running_worker():
    """close() 会被异常处理路径重复调用，不能第二次就炸。"""
    group = _WorkerGroup(_cfg(), "toy", num_envs=2, num_workers=2, seed=0)

    group.close()
    group.close()
