# Isolated CUDA device energy

This benchmark measures a bounded Cadence patch: local neuron state, named sensor and
output populations, residual readback, synaptic update moves, and reproducible evidence.
The measurements concern three library operations, not task accuracy, biological
efficiency, or the energy of a complete training system.

The exact runtime package is the complete archive
`../frozen/sources-2bd9e9056d35055f.tar.gz`, SHA-256
`2bd9e9056d35055f25c455f2ccecf4d553ae9a77179b149bfedde0433e3575c7`.
Both arms import this frozen package. The baseline restores the four original runtime
functions from revision `9f859bfd8f4df52aed5460dbd84306da02a61e34`, as in the existing CPU
and MPS comparison. The current arm uses the archived implementations. Later actor,
skip-initialization and documentation changes cannot change these runs.

Each invocation archives its own executable, the complete runtime source archive and,
for measurement, the pilot receipt. A subprocess imports only the extracted frozen
source. The receipt binds `sources.zip` by SHA-256; compressed NumPy snapshots bind all
final potentials, activations, efficacies and biases. The verifier checks their actual
differences as well as the sensor arithmetic, schedule and declared numerical budgets.

## Fixed protocol

An otherwise idle, dedicated NVIDIA A10G is used. No player or composer GPU is used.
Both arms use float64, 128 inputs, 128 hidden neurons, 16 outputs, batch 64, effective
row mass 1, the same graph/data seeds 0–4 and identical initial conditions. Torch and
BLAS use one CPU thread. Three units warm each workload/arm before measurement.

- Fixed settling: 60 steps from rest.
- Centered learning: five updates, each with 40 free and two 20-step nudged phases,
  with no stopping tolerance. Every repetition restores the original device parameters
  and creates a fresh learner. That reset and learner construction are timed. Graph
  construction and snapshot readback are outside the interval.
- Cold residual solve: chunk 4, budget 128, potential/adaptation residual tolerance
  `1e-8`, with the independent float64 host residual checked afterward.

The seed-0 pilot measures three unit durations after three warmups. For each workload
it locks `ceil(15 / faster_arm_median_seconds)` repetitions for both arms and all five
seeds. The measured interval must last at least 10 seconds. The locked counts are
593 fixed solves, 69 five-update units and 725 residual solves. They are never adjusted
after observing the measurement results. There is one long interval per seed/arm/
workload, 30 total. Arm order alternates by seed. Both interval endpoints synchronize
CUDA. Consequently these timings have a different repetition boundary from the original
single-unit workstation benchmark; compare only the matched arms within this receipt.

## Device-energy scope

The primary reading is the difference of cumulative GPU energy counters, in millijoules,
converted to joules. This is the supported
[NVML total-energy API](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html),
which counts GPU energy since the driver was last loaded. Timestamped samples every
0.2 seconds retain cumulative energy, power, temperature and graphics/memory clocks.
Every interval records its endpoint process occupancy; any foreign compute or graphics
process causes failure. No other workload is launched on the dedicated instance.

Three-second idle intervals immediately precede and follow each work interval. The
mean of their observed wattages is an estimated idle rate. The receipt reports both
raw GPU joules and `raw_joules - mean_idle_watts * elapsed_seconds`, without clamping
or substituting one for the other. Adjacent idle windows can contain clock/temperature
transients; subtraction is an estimate, not a separate hardware energy counter.
Raw readings are the primary energy comparison. Negative corrections, short intervals,
counter errors, missing samples and competing-process errors are retained as failures.

No host CPU, memory, networking, power-supply or whole-system energy is measured. A
device result cannot establish overall energy superiority or undo historical training
cost gaps. Five deterministic fixture seeds are not five independent machines, and
one long interval per cell does not establish a hardware-wide confidence bound.

## Observed result

The dedicated A10G run used Torch 2.14.0+cu130, CUDA 13.0, driver 615.71.09,
Python 3.11.16 and NumPy 2.4.6. All 30 intervals passed their minimum duration:
14.730–19.694 seconds. The complete measured schedule took 687.701 seconds,
with 3,355 raw telemetry samples and no sensor or occupancy errors. Temperature
ranged from 28 to 37 °C. Source and numerical verification passed; the largest
baseline/current state or parameter discrepancy was 4.440892098500626e-16.
Every cold solve used 28 steps, with maximum host residual 8.980547819281703e-9.
The separate phase audit recorded 300 expected phase counters; both repeated
snapshots matched each other and the measured snapshots exactly.

The following are medians across seeds, per unit. Each entry is baseline/current.

| Unit | Time (ms) | Raw GPU joules | Idle-corrected GPU joules |
| --- | ---: | ---: | ---: |
| 60 fixed steps | 33.1163 / 27.0462 | 2.161187 / 1.778474 | 0.227934 / 0.197582 |
| Five centered updates | 256.0684 / 215.5023 | 16.714812 / 14.117855 | 1.758105 / 1.453895 |
| Cold residual solve | 22.4581 / 20.6133 | 1.448061 / 1.345068 | 0.121504 / 0.131333 |

The medians of the five **paired** baseline/current ratios are distinct summaries:

| Unit | Time ratio | Raw energy ratio | Idle-corrected energy ratio |
| --- | ---: | ---: | ---: |
| 60 fixed steps | 1.224000 | 1.214295 | 1.169266 |
| Five centered updates | 1.191827 | 1.190217 | 1.241673 |
| Cold residual solve | 1.090086 | 1.071501 | 0.916529 |

Raw GPU energy decreased in every pair. The idle-corrected cold-residual estimate
**increased in every seed**: its paired baseline/current ratios ranged from 0.683952
to 0.961507. This negative result remains part of the evidence. The resident residual
path transfers arithmetic previously performed on the CPU onto the GPU, so reduced
latency need not imply reduced incremental GPU work. This is a possible explanation,
not a separately isolated mechanism experiment. Adjacent idle estimates were
57.356–60.296 W, close to workload power; their subtraction is sensitive to idle
state and clock/temperature transients. Graphics clocks remained at 1710 MHz
through every work and post-idle window (1695–1710 MHz before work), so this is
an adjacent idle estimate with a resident CUDA context and high device clocks.
The primary raw measurements and this
secondary correction are both reported. Whole-system energy remains unmeasured.

`summary.json` contains every per-seed value and ratio, generated by `verify.py`.
The executable receipt tests reject corrupted counts, duplicate cells, altered
tolerances/source identity/numerical gaps, false energy or idle arithmetic, foreign
occupancy and short intervals. All 12 tests pass.

## Reproduction

Use a dedicated idle NVIDIA device with a supported cumulative energy counter, Python
3.11+, NumPy, CUDA-enabled Torch and `nvidia-ml-py`. From the repository root:

```sh
python benchmarks/device_energy/run.py --mode pilot \
  --runtime-sources benchmarks/frozen/sources-2bd9e9056d35055f.tar.gz --out /tmp/energy-pilot
python benchmarks/device_energy/verify.py /tmp/energy-pilot/receipt.json
python benchmarks/device_energy/run.py --mode measure \
  --runtime-sources benchmarks/frozen/sources-2bd9e9056d35055f.tar.gz \
  --pilot /tmp/energy-pilot/receipt.json --out /tmp/energy-results
python benchmarks/device_energy/verify.py /tmp/energy-results/receipt.json
python benchmarks/device_energy/check_phases.py --results /tmp/energy-results \
  --out /tmp/energy-phases
python benchmarks/device_energy/verify.py /tmp/energy-results/receipt.json \
  --phases /tmp/energy-phases/receipt.json
```

For the exact archived experiment, extract its `sources.zip` and use the enclosed
`run.py`, `runtime_sources.tar.gz` and (for measurement) `pilot.json`. The pilot must
refer to the same GPU UUID. A new GPU requires its own pilot and fresh output folder.
Replay results are new observations; they should not overwrite this receipt.

The saved `pilot/`, `results/` and their logs preserve the original source bundles,
warmups, timestamps, raw sensor samples, final snapshots and any error records. The
receipt verifier requires only NumPy and can run without CUDA or Git.

The separate `phases/` audit runs after energy measurement. It repeats each centered
learning unit twice on the same GPU, checks all 300 individual phase counters, checks
that both initial parameter tensors remain unchanged, and compares both resulting
states and parameter vectors with the measured snapshot. This untimed diagnostic does
not alter the measurement schedule or supply additional energy observations.
