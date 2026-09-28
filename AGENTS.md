# AGENTS.md

给在这个仓库里干活的 AI agent（以及第一次上手的人）的说明。只写从这里看不出来
的东西：约定、禁区、验证方式。代码结构不重复，README 里已经写了。

## 先跑这条

```bash
pip install -e ".[dev,deploy,record]"
ruff format --check . && ruff check . && pytest -m "not slow"
```

改完任何一层都跑一遍。`-m "not slow"` 会跳过需要 MuJoCo Menagerie 模型的用例；
要跑全套先 `python scripts/fetch_assets.py`，再 `pytest`。

三个 extra 都要装。少 `deploy` 会让导出/量化/基准/回测整条链被 `importorskip`
跳过，少 `record` 会让录制那组跳过——CI 照样绿灯，但那些代码一次都没验证过。
**绿灯如果不是"跑过且通过"，就没有意义。**

## 这个项目的约定

### 注释写"为什么"，不写"是什么"

仓库里注释密度很高，但几乎不解释代码在做什么——那看代码就知道。每条注释回答的
是"为什么是这个写法"：为什么这个数是 3 不是 4（`TERRAIN_GEOM_GROUP`），为什么
这个字段叫 `speedup_min` 而不叫 `speedup`，为什么这里宁可报错也不静默退回。
每一条几乎都对应一次真实的踩坑。

写新注释时对着这条检查：如果删掉它，读者会不会把代码改成错的？不会就别写。
反过来，如果你为了某个选择纠结过，那个纠结必须写下来——下一个人会重新纠结一遍。

### 中文，专业，说人话

面向的读者是中文母语、有工程背景的人。不要大白话，也不要 AI 味。判断标准是
"这句话像不像一个懂行的人写给同事看的"。

### 失败要响，不要静默

这个仓库里几条最主要的不变式都用"构造时检查"守着，而不是靠文档提醒：

- `MujocoEnv._check_terrain_group_is_exclusive`：地形组里混进机器人的几何时当场
  报错。这条挡的 bug 不报错、不崩溃，只是往 critic 的输入里塞一个假信号。
- `GifRecorder.close`：一帧没录到时抛错，而不是写出一个 0 帧的空 GIF（空 GIF
  能被很多查看器"成功"打开）。
- `GifRecorder.add`：写盘之后再喂帧要报错，而不是悄悄丢掉。
- `_resolve_cpu`：`--cpu` 给的核号超范围直接退出，不静默退回 `auto`。

加新代码时优先考虑这一条：一个会在错误发生时变红的检查，比一段说明为什么不能
那么写的注释有用得多。

### 每个数字都要能被一条命令复现

README 和 `docs/` 里的任何数字，都必须对应仓库里的一条命令。写文档时如果要引用
一个数，先跑一遍，把命令一起写下来。做不到就别写这个数。

有个具体的坑：**训练默认不可复现**。策略权重来自 torch 的全局随机源，不调用
`robotrl.utils.torch_runtime.seed_torch` 的话，同一份配置跑两次会得到两个不同的
策略（实测第一轮回报就差 3%）。`scripts/train.py` 和 `examples/toy_train.py` 都
在建策略之前调了它；写新的训练入口时必须照做，否则你产出的数字没人能复现。

同理，跑延迟基准前先读 `robotrl/utils/cpu_affinity.py`：不绑核的话，同一份模型
在大小核之间迁移能让最小延迟差近 8 倍。

## 不要做的事

- **不要提交 `runs/`、`assets/menagerie/`、`*.onnx`、`*.pt`**。都在 `.gitignore`
  里，理由是体积和可重新生成。要分享结果就提交 JSON 报告。
- **不要引入 RL 库**（stable-baselines3、tianshou 之类）。算法层是从零写的，
  这是项目定位的一部分，不是没来得及换。
- **不要为了跑通而放宽断言**。测试红的时候先确认是代码错了还是测试错了；这个
  仓库里两者都发生过（详见 `tests/test_terrain.py` 里那条单向断言的注释）。
- **不要动 `tests/mini_robot.py` 里的最小形态**，除非你要测的正是形态加载。
  它存在的意义是让环境层的用例不依赖 Menagerie，天级能跑几百次。

## 加东西时的落点

| 要加什么 | 落在哪 |
| --- | --- |
| 一种机器人 | `robotrl/assets/<name>.py` + `assets/__init__.py` 一行 import + `scripts/fetch_assets.py` 的 `SUBDIRS`，见 `docs/05` |
| 一种地形 | `robotrl/envs/terrain.py` 的 `builders` 里加一个函数 |
| 一种奖励项 | `robotrl/envs/managers/rewards.py`，再在任务的默认权重里挂上 |
| 一种域随机化 | `robotrl/envs/events.py`，一个事件一个类 |
| 一种量化方式 | `robotrl/deploy/quantize.py`；如果多一种模式，记得 `scripts/evaluate.py` 与 `scripts/benchmark.py` 的报告文件名带 mode，别互相覆盖 |
| 一个推理后端 | `robotrl/deploy/engine.py` 的 `register_backend` |
| 一个脚本 | `scripts/`，公开参数用 `scripts/_shared.py` 的 `add_config_args` |

## 测试怎么分

- 默认套件（`pytest -m "not slow"`）必须快、且不依赖外部下载。目前是几十秒。
- 要加载真实模型或真训练一会的，标 `@pytest.mark.slow`。
- 需要 GPU 的标 `@pytest.mark.gpu`。
- 断言尽量盯"可观察的后果"而不是实现细节：`tests/test_terrain.py` 里验的是
  "采样值等于某级台阶的高度"，而不是"射线用了哪个 group"。前者在实现换掉之后
  依然有效，后者不会。

## 提交

提交信息写清"改了什么、为什么"，不要写"fix bug"这类。格式上没有硬要求，
但参考已有的两条提交：一条是初始搭建，一条是规范检查后的修正。
