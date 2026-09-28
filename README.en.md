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

- **PPO and SAC written from scratch**, under 300 lines each including comments, no RL
  library. Hyperparameters follow the papers and mainstream implementations, with the
  implementation details commented one by one.
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

```bash
git clone https://github.com/Qxy661/robot-rl-lab.git && cd robot-rl-lab
pip install -e ".[deploy,dev]"

# Fetch robot models (sparse checkout of MuJoCo Menagerie)
python scripts/fetch_assets.py

# Run the full loop in minutes: train → export → quantize → benchmark → backtest
python examples/toy_train.py
```

The toy environment is a point-mass velocity tracking task that does not need
MuJoCo. It trains to a visible improvement in seconds, and its purpose is to let
you confirm the whole pipeline still works after touching any layer.

For real robot training:

```bash
python scripts/train.py --config robotrl/configs/train/g1_velocity.yaml
python scripts/play.py  --run-dir runs/g1_velocity
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

## Documentation

The main documentation set is in Chinese (`docs/`), with two tracks: a
from-zero path (chapters 00-06) and a module-level reference for readers who
already know RL. See [`docs/README.md`](docs/README.md).

## Benchmark results

Every number below is produced by the scripts in this repository; the JSON
reports live in `runs/<task>/exported/`.

Task: G1 velocity tracking, 15 actuated joints, 57-dim policy observation.
The checkpoint comes from 8192 training steps across 16 parallel environments —
enough to stand up, not enough to walk well. What is compared here is **two
implementations of the same policy**, so the absolute values are not a
performance claim.

### On-device size and latency

| | FP32 | INT8 dynamic | INT8 static |
| --- | --- | --- | --- |
| File size | 199.3 KB | 61.0 KB (**3.27×**) | 61.0 KB (**3.27×**) |
| min latency (1 thread / batch 1) | 0.0100 ms | 0.0118 ms | 0.0118 ms |
| Speedup by min | 1.00× | **0.91×** | **0.84×** |
| Backtest return (20 episodes) | 63.157 | 62.997 (−0.25%) | 63.141 (−0.02%) |
| Per-joint action MAE | — | 0.0013 | 0.0019 |

Measured on an Intel Core Ultra 7 255H, onnxruntime CPU backend, single
thread, batch 1.

**The honest result: on this CPU, INT8 does not make this small network
faster.** Both quantized models are 10–19% slower at the minimum latency than
FP32. What quantization buys here is size — 3.27×, a real win for on-device
storage and memory bandwidth. The latency win is what an NPU backend would
deliver, which is exactly what `PolicyEngine.register_backend` leaves room for.
"Quantization is faster" is usually stated as a premise; here it is a
measurement, and the measurement disagrees.

### Why these latency numbers can be trusted

Microsecond-scale inference benchmarks produce false conclusions easily, so
three things are handled explicitly (`robotrl/deploy/benchmark.py`):

- **Conclusions come from `min`, not the mean.** Interference only ever makes a
  measurement slower, never faster, so the minimum is the cleanest estimate of
  the actual compute time. On this machine the p50 of one fixed model swings
  between 0.010 and 0.032 ms, while `min` holds to four significant figures
  across six interleaved rounds.
- **Paired measurement.** FP32 and INT8 are measured alternately within each
  round rather than one after the other, so both see the same machine state;
  the reported ratio is the median of the per-round ratios.
- **Refuses to conclude when noise exceeds the effect.** If the max/min spread
  of the per-round ratios crosses a threshold, the report says "this machine
  cannot resolve the difference" instead of printing a two-decimal speedup.

### Training throughput

The same G1 task on CPU: about 1.5 ms per environment-step, so stepping 64
environments serially costs 96 ms per control step. Dispatching the whole batch
to worker processes at once brings that to roughly 7 ms (**5.8×**).

PyTorch's thread count also defaults to 1 rather than the core count: with only
tens of thousands of parameters, the synchronization cost of splitting an
operator across cores exceeds the compute it saves. A 64×64 MLP backward pass
takes 258 ms at 16 threads and 0.92 ms at one thread (**280×**) on a 16-core
machine. See `robotrl/utils/torch_runtime.py`.

## License

Apache-2.0. Robot models come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(BSD-3-Clause) and are not redistributed with this repository.
