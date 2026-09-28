# robot-rl-lab

A full-stack reinforcement learning framework for robot locomotion:
MuJoCo simulation → PPO/SAC implemented from scratch → ONNX export and INT8
quantization → on-device deployment backtest.

Demonstrated on three Unitree platforms (G1, H1, Go2). The emphasis is on a
**complete loop** and on **every layer being explainable**.

```
train ──▶ policy ──▶ ONNX ──▶ INT8 ──▶ latency ──▶ sim backtest
  │         │         │        │         │             │
  └ sim env └ from-scratch └ self-contained └ quantized └ p50/p95/p99 └ reward / action drift
```

![G1 velocity-tracking policy](docs/assets/g1_velocity.gif)

G1 velocity tracking, trained with the minimal preset shipped in this
repository: it stands for about 1.6 seconds and falls on its back. The clip is
itself a statement of fact — **there is no walking robot in this repository**.
See the table below for why.

## Status

This repository is about a complete, explainable pipeline, not about producing a
controller you could put on hardware. What is verified, what merely runs, and
what is missing:

| Part | Status | Notes |
| --- | --- | --- |
| train → export → quantize → benchmark → backtest | Verified | Toy task (no MuJoCo) runs end to end in minutes; CI runs the default suite every commit: 413 selected, 1 skips by design |
| PPO / SAC from scratch | Verified | 49 tests covering GAE, ratio clipping, the tanh correction, target networks, entropy temperature |
| G1 / H1 / Go2 morphologies | Verified | One `env.robot` field to switch; terrain sampling and joint mapping are under test |
| Quantization backtest and KL gate | Verified | 20 paired episodes; a drifted action distribution is reported as a failure, not smoothed over |
| G1 velocity-tracking policy | Smoke only | 8448 steps, about 7 seconds of training; **every number below comes from it** |
| The full-scale presets | Not run | G1 PPO 20M steps / 64 envs, Go2 SAC 30M steps / 32 envs. Hours on CPU; the numbers here did not come from them |
| Rough terrain / stairs | Environment only | Terrain generation and sampling are tested; no trained policy |
| Real hardware / NPU deployment | Not done | Stops at ONNX + INT8 + simulation backtest. `PolicyEngine.register_backend` is the hook; nothing implements it |

Read the two numbers together: **8448 steps is not 20 million**. The performance
figures below are evidence that the pipeline reproduces, not that the policy is
any good.

The one test CI skips by design is `test_model_path_points_into_menagerie`: it
only checks the path rule, but it is gated on Menagerie being fetched (see
`needs_models` in `tests/test_loader.py`). Run `scripts/fetch_assets.py` locally
and all 413 pass.

Two more may skip depending on which machine CI lands on. The quantisation
accuracy and quantised-backtest conclusions only hold where the runtime executes
int8 faithfully. On some CPUs onnxruntime's int8 kernels compute the wrong
answer; those two tests then skip with the CPU model attached rather than
reporting a wrong conclusion. CI runs `pytest -rs` so skip reasons reach the log
instead of hiding inside a green run. See item 3 in
[07](docs/07_未决问题与风险.md) (Chinese).

## What this project addresses

Most open-source robot RL projects stop at "export to ONNX". The questions that
on-device deployment actually has to answer — how much accuracy does quantization
cost, how much latency does it save, how far does the policy's behavior drift —
rarely come with published comparison data. This project closes that loop:
FP32 and INT8 policies are evaluated on identical initial states, and the
resulting reward delta, action deviation and latency distribution are reported.
The pipeline is hardware-agnostic.

The framework itself is not tied to any robot. Switching platforms is a
one-string config change; adding a new morphology means adding one spec file.
This is demonstrated on three legged platforms and leaves the door open for
non-legged ones.

## Features

- **PPO and SAC written from scratch**, 294 and 260 lines respectively including
  comments, no RL library. Hyperparameters follow the papers and mainstream
  implementations, with the implementation details commented one by one.
- **Cross-morphology**: G1 (29-DoF humanoid), H1 (19-DoF humanoid) and Go2 (12-DoF
  quadruped) share one environment, one algorithm stack and one deployment pipeline.
  Switching is `env.robot`.
- **Self-contained ONNX**: observation normalization is baked into the graph, so the
  inference side feeds raw observations with no preprocessing code.
- **Quantization backtest loop**: export → equivalence check → INT8 quantization →
  latency benchmark → simulation backtest, each step producing reproducible numbers.
- **Systematic Chinese documentation**: two reading tracks, one for readers with an
  ML background and one starting from zero.

## Quick start

Two paths. The first needs no model download and takes a few minutes; the second
uses the real robot models.

### Path 1: no download, full loop

```bash
git clone https://github.com/Qxy661/robot-rl-lab.git && cd robot-rl-lab
pip install -e ".[dev]"

python examples/toy_train.py
```

The toy environment is a point-mass velocity tracking task that does not need
MuJoCo. It trains to a visible improvement in seconds and then walks the whole
export → quantize → benchmark → backtest chain. Its purpose is to let you confirm
the pipeline still works after touching any layer.

### Path 2: real robots

```bash
pip install -e ".[deploy,dev]"
python scripts/fetch_assets.py     # sparse checkout of MuJoCo Menagerie
```

Reproducing the table below (about 7 seconds of training, a few minutes overall):

```bash
python scripts/train.py     --config robotrl/configs/train/g1_smoke.yaml
python scripts/export.py    --run-dir runs/g1_smoke
python scripts/quantize.py  --run-dir runs/g1_smoke --mode dynamic
python scripts/quantize.py  --run-dir runs/g1_smoke --mode static
python scripts/benchmark.py --run-dir runs/g1_smoke --mode dynamic
python scripts/benchmark.py --run-dir runs/g1_smoke --mode static
python scripts/evaluate.py  --run-dir runs/g1_smoke --mode dynamic
python scripts/evaluate.py  --run-dir runs/g1_smoke --mode static
```

Watch the policy at real-time pace, or record it:

```bash
python scripts/play.py --run-dir runs/g1_smoke
python scripts/play.py --run-dir runs/g1_smoke --record docs/assets/g1_velocity.gif
```

To see a policy that actually learned to walk, run the full preset (hours on CPU):

```bash
python scripts/train.py --config robotrl/configs/train/g1_velocity.yaml
```

## Layout

```
robotrl/
  assets/      physical layer: RobotSpec definitions + MuJoCo loading
  envs/        environment layer: 8-stage pipeline
  algorithms/  algorithm layer: PPO and SAC, four orthogonal pieces
  deploy/      deployment layer: export, equivalence, quantize, benchmark, backtest
  eval/        evaluation layer: reproducible baselines
  configs/     config schema and presets
docs/          Chinese documentation, two reading tracks
```

Four contracts separate the layers, so changing one does not disturb the others:

| Contract | Location | Purpose |
| --- | --- | --- |
| `RobotSpec` | `assets/spec.py` | joints, default pose, PD gains of a platform |
| `BaseEnv` | `envs/base_env.py` | environment interface and the 8-stage pipeline order |
| `Policy` / `Trainer` | `algorithms/base.py` | observation-to-action mapping, train and eval |
| obs/action layout | `contracts.py` | shared observation segments and action semantics |

## Where to read for what

| I want to | Start at |
| --- | --- |
| Switch platforms | `assets/spec.py`; the three specs sit next to it, loading is in `envs/mujoco_env.py` |
| Change rewards | `envs/tasks/velocity.py` holds the task defaults, `reward.scales` in config overrides them |
| Change observations | Layout in `contracts.py`, assembly in `BaseEnv._observe` |
| Add terrain | `envs/terrain.py`, one builder plus a name in the config |
| Add domain randomization | `envs/events.py`, one class per event |
| Read PPO / SAC | `algorithms/ppo.py` (294 lines) / `sac.py` (260 lines) |
| See the training loop | `algorithms/trainer.py` — eval, checkpointing, logging |
| See the environment loop | `BaseEnv.step` — the order of the 8 stages |
| Export ONNX | `deploy/export_onnx.py`; equivalence checking in `deploy/equivalence.py` |
| Quantize | `deploy/quantize.py`, dynamic and static |
| Measure latency | `deploy/benchmark.py`; read `utils/cpu_affinity.py` first, pinning is a precondition |
| Backtest a quantized policy | `deploy/evaluate.py` — how each of the three metrics is taken |
| Swap the inference backend | `deploy/engine.py`, `register_backend` |
| Add a new morphology | `docs/05_跨形态抽象.md` (Chinese), three steps, no environment or algorithm changes |

## Documentation

The main documentation set is in Chinese (`docs/`), with two tracks: a
from-zero path (chapters 00-07) and a module-level reference for readers who
already know RL. See [`docs/README.md`](docs/README.md). Chapter 07 lists the
known gaps and risks, ordered by severity.

## Benchmark results

Every number below is produced by the commands above (G1 velocity tracking,
15 actuated joints, 57-dim observation, 20 paired episodes). Reports are not
committed — `runs/` is large and regenerable — so run the commands to get them.

Before comparing against anything else, note what is being compared: the policy
was trained for 8448 steps and stands for 1.6 seconds. It cannot walk. So what
follows compares **two implementations of the same policy**, not two policies,
and the absolute values are not a performance claim.

### On-device size and latency

| | FP32 | INT8 dynamic | INT8 static |
| --- | --- | --- | --- |
| File size | 199.3 KB | 61.0 KB (**3.27×**) | 60.9 KB (**3.27×**) |
| min latency (1 thread / batch 1) | 0.0099 ms | 0.0116 ms | 0.0118 ms |
| Speedup by min | 1.00× | **0.85×** | **0.85×** |
| Backtest return (20 episodes) | 64.085 ± 16.665 | 64.032 (−0.08%) | 64.020 (−0.10%) |
| Per-joint action MAE | — | 0.00093 | 0.00127 |
| Max action deviation | — | 0.0070 | 0.0099 |
| Action distribution KL (mean / worst dim) | — | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

Measured on an Intel Core Ultra 7 255H, onnxruntime CPU backend, single thread,
batch 1, process pinned to CPU1.

The two INT8 columns carry one more precondition: **the runtime has to execute
int8 faithfully.** On some CPUs onnxruntime's int8 kernels compute the wrong
answer — same file, same inputs, a 0.196 maximum deviation from a pure-numpy
reference implementation, two orders of magnitude larger than the error
quantisation itself introduces, and the quantised artifact is fine (the
reference and another set of CPUs agree). So "what does quantisation cost" only
holds once you measure it on your target hardware, and the table above is how to
measure it. Criteria and the lab notes are in item 3 of
[07](docs/07_未决问题与风险.md) (Chinese).

**The honest result: on this CPU, INT8 does not make this small network
faster.** Both quantized models are about 15% slower at the minimum latency than
FP32. What quantization buys here is size — 3.27×, a real win for on-device
storage and memory bandwidth. The latency win is what an NPU backend would
deliver, which is exactly what `PolicyEngine.register_backend` leaves room for.
"Quantization is faster" is usually stated as a premise; here it is a
measurement, and the measurement disagrees.

Dynamic and static quantization are nearly indistinguishable on this model:
same size, same latency, backtest returns within 0.1% of each other. Either
works; this project defaults to dynamic because it needs no calibration data,
which is one fewer step that can go wrong.

### Why these latency numbers can be trusted

Microsecond-scale inference benchmarks produce false conclusions easily. This
project got it wrong once, so every point below has an implementation behind it
(`robotrl/deploy/benchmark.py` and `robotrl/utils/cpu_affinity.py`):

- **Pin the process to one core first.** This is the most easily overlooked step
  and the one with the largest effect. The CPU is a hybrid design: measuring the
  same model on the same input across all logical cores gives a minimum latency
  from 0.0100 ms to 0.1097 ms — **nearly 8×**. Unpinned, the scheduler migrates
  the process between cores, and the effect under test is only 15%, so it drowns
  completely. Before pinning was added, the same model produced speedups of 0.86×
  and 1.24× in different runs — opposite conclusions, which at the time looked
  like measurement noise but were the process changing cores.
- **Conclusions come from `min`, not the mean.** Interference only ever makes a
  measurement slower, never faster, so the minimum is the cleanest estimate of
  the actual compute time. Pinned, FP32's min holds at 0.0099–0.0100 ms while the
  p50 of the same runs swings between 0.011 and 0.030 ms.
- **Paired measurement.** FP32 and INT8 are measured alternately within each round
  rather than one after the other, so both see the same machine state.
- **Refuses to conclude when noise exceeds the effect.** If the max/min spread of
  the per-round ratios crosses a threshold, the report says "this machine cannot
  resolve the difference" instead of printing a two-decimal speedup. This machine
  runs a pile of endpoint-management software, so background load never settles,
  and **every report it produces carries that warning**. That is the tool working
  as intended: the warning covers the mean and p99 columns, and the conclusion
  here is taken from `min` only.

### Training throughput

The same G1 task on CPU: about 1.5 ms per environment-step, so stepping 64
environments serially costs 96 ms per control step. Dispatching the whole batch
to worker processes at once brings that to roughly 7 ms (**5.8×**).

PyTorch's thread count also defaults to 1 rather than the core count: with only
tens of thousands of parameters, the synchronization cost of splitting an
operator across cores exceeds the compute it saves. A 64×64 MLP backward pass
takes 258 ms at 16 threads and 0.92 ms at one thread (**280×**) on a 16-core
machine. See `robotrl/utils/torch_runtime.py`.

## Credits

What is valuable here is assembling mature pieces into a complete chain, not
reinventing them. Each dependency is the right tool for its job:

| Dependency | Used for | License |
| --- | --- | --- |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | physics simulation and rendering | Apache-2.0 |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | G1 / H1 / Go2 models and scenes | BSD-3-Clause |
| [PyTorch](https://github.com/pytorch/pytorch) | algorithm implementation and training | BSD-3-Clause |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | on-device inference backend | MIT |
| [Pillow](https://github.com/python-pillow/Pillow) | GIF recording | MIT-CMU |

The rest are ordinary Python libraries (NumPy, PyYAML, ONNX, pytest, ruff).

Algorithm choices were compared against
[CleanRL](https://github.com/vwxyzjn/cleanrl),
[rl_games](https://github.com/Denys88/rl_games) and
[legged_gym](https://github.com/leggedrobotics/legged_gym); hyperparameters come
from the papers, and the code is a rewrite.

## License

Apache-2.0. Robot models come from MuJoCo Menagerie (BSD-3-Clause) and are not
redistributed with this repository; `scripts/fetch_assets.py` fetches them.
