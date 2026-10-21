# Backends, devices, precision

<!-- requires: torch -->

Every graph brain runs on the `"cpu"` backend in NumPy float64 unless you pick another. Start
from the NumPy-only [installation](../README.md#get-started) and add an extra:

```bash
python -m pip install "cadence-net[fast]==0.84.3"
```

`[fast]` installs Numba and SciPy, `[accel]` PyTorch and `[apple]` MLX; from a checkout, use
`python -m pip install -e ".[fast]"`. Installing a library never selects it: pass
`backend="torch"` or `backend="mlx"`. `available_backends()` lists the importable backends
with the device and default precision each would use.

```python
import cadence as cd

print(cd.available_backends())
# e.g. {'cpu': 'numpy float64', 'torch': 'mps float32', 'mlx': 'gpu float32'} on an M-series Mac
```

## Choose a backend

| backend | where it runs | precision | extra | use it for |
|---|---|---|---|---|
| `"cpu"` | NumPy, with Numba and SciPy when installed | float64 | none; `[fast]` to accelerate | first examples, receipts, numerical checks, sparse graphs, small continuing brains |
| `"torch"` on CUDA | NVIDIA GPU | float64, or float32 with `precision="float32"` | `[accel]` | large brains, large batches, long lives |
| `"torch"` on MPS | Apple silicon GPU through Metal | float32 only | `[accel]` | a Mac when torch is what you have |
| `"mlx"` | Apple silicon GPU through MLX, unified memory | float32 | `[apple]` | an alternative Apple path for blocked connectomes |
| `"torch"` on CPU | torch on the host | float64, or float32 with `precision="float32"` | `[accel]` | one code path on a machine without a GPU |

`backend="torch"` picks CUDA, then MPS, then CPU; `device=` overrides it. One brain uses one
device. MPS refuses float64, `"mlx"` refuses a connectome whose blocks exceed `dense_limit`,
and a float32 backend refuses a neuron rule whose constants float32 cannot represent.

## What runs where

`backend=` (default `"cpu"`) and `device=` (default `None`, the best device) are construction
arguments of `NeuralGraph`, `Brain.compose`, `PatchNet.create` and `PatchNet.recursive`;
`Brain.retune` refuses them. `precision=` (`"float64"` or `"float32"`, read by torch, default
by device) is an argument of `NeuralGraph`, `Brain.load`, `Learner.load` and `PatchNet.load`
only, so a composed brain runs at its device's default precision until it is saved and loaded
with another. Loads default to the saved precision; `Brain.load` and `PatchNet.load` default to
`backend="cpu"`, `Learner.load` to the saved backend. `Learner.load(path, backend="cpu")`
brings a checkpoint learned on any device back to the float64 receipt backend.

The composed brain's memory stores are read on the host inside a device brain too.

On a device backend `settle_batch` returns a state that stays resident (`BrainState.device`;
`None` on the CPU and from the one-stimulus `settle`), so a continuing phase starts there.
Blocked graphs compute the learning contrast there too (`NeuralGraph.contrast_on_device`,
`None` unless both states are resident). Blocked torch `Learner` updates, momentum and RMS
normalization stay on the device; MLX hands the contrast to a host optimizer, and sparse
contrasts use bounded host chunks. `ActorCritic` updates on the device only with its own
`momentum` and `normalize` both zero; its critic and dopamine use host arrays. Reading
parameters or saving a checkpoint copies arrays back. Parameter edits and the `learner.brain`
rule are in [contracts](contracts.md).

## Choosing a device

```python
connectome = cd.layered(4, 16, 2, seed=0)
neuron_model = cd.learning_neuron_model()
cd.NeuralGraph(connectome, neuron_model, backend="torch")                        # cuda, else mps, else cpu
cd.NeuralGraph(connectome, neuron_model, backend="torch", device="cpu")          # force a device
cd.NeuralGraph(connectome, neuron_model, backend="torch", precision="float32")   # float32 settling
if "mlx" in cd.available_backends():
    cd.NeuralGraph(connectome, neuron_model, backend="mlx")
brain = cd.Brain.compose(8, 3, backend="torch", device="cpu")                    # a composed brain
check = cd.conformance(cd.NeuralGraph(connectome, neuron_model, backend="torch"), {0: 1.0})
print(check["backend"], check["max_abs_deviation"])                              # vs the float64 reference
```

## Which hardware

Use CPU with Numba for small continuing brains, and measure an accelerator on the actual
changing-input workload before switching: small brains spend more time launching GPU
operations than doing arithmetic, while large batches and dense blocks gain from accelerator
matrix products. The [System 1 CUDA qualification](../benchmarks/system1_cuda/RESULTS.md)
compares CPU, CUDA float64 and CUDA float32 on complete decisions; on its small continuing
nursery CPU with Numba is faster. Cap BLAS threads for parallel workers. To compare two
implementations, use identical dtype, steps and tolerances over several seeds, warm up,
alternate the order, synchronize around each timed call, and compare residuals and states too.

## Precision matters

Near multiple attractors a rounding difference can change the selected state, and float32
learning need not track float64 within a fixed error; RMS normalization amplifies small
contrast errors. Every backend forms the phase contrast as
`A_plus.T @ (B_plus - B_minus) + (A_plus - A_minus).T @ B_minus`, so identical phases give
exactly zero. Check the deployed backend with `conformance` on representative drives, compare
residuals and task readouts, and declare precision in any receipt. Run gradient checks on
float64 CPU or CUDA with the nudge size and phase residuals declared.

## Residual checks, masks and conformance

`NeuralGraph.equilibrate` chooses its work from the potential and adaptation equation
residual: a warm state within tolerance takes zero steps and one residual check. `chunk` trades
check frequency against overshoot at the same final tolerance. Step count measures numerical
settling work, never surprise or task quality. `residual(drive, state, on_device=True)` checks
an unread float64 torch state on its device, one scalar per row back; float32 and MLX states, a
state read on the host and `on_device=False` use the float64 host reference. A positive
movement `tolerance` in `settle_batch` costs a scalar check every step; fixed step counts and
the inner chunks of `equilibrate` avoid it.

Masks have shape `(n,)` or `(1, n)` for every row, or `(batch, n)` per row, with entries in
[0, 1], on every backend. A mask scales potential and activation at each step; zero removes a
neuron's published activity while its adaptation decays. A warm start recomputes the published
activation under the current mask, and the fused kernel reads a shared mask without copying it.

`cd.conformance(graph, stimulus, steps=60)` settles a `NeuralGraph` (`brain.learner.brain`
for a composed brain) from rest and compares its trajectory with the neuron-by-neuron float64
reference. It returns `backend`, `steps`, `max_abs_deviation` over the trajectory, the
reference transmission `ledger` and `final_active` (neurons at activation 0.5 or above).

## Implementation notes

Transport is dense, block products between neuron ranges, when the block layout fits
`NeuralGraph(dense_limit=2048)` (at most `dense_limit ** 2` entries). The CPU paths, torch CPU
included, reuse the products of a source range (one that receives no synapses) whose
activation did not change. Torch on CUDA and MPS recomputes every product each step, and CUDA
accumulates each product directly into its destination. MLX checks each source range for
equality on the host at every step, one synchronization per range per step, and reuses the
products of an unchanged one.

Above `dense_limit` transport is sparse: a SciPy CSR multiply on the CPU when SciPy is
installed, a NumPy segmented sum otherwise, gather and scatter on torch; `"mlx"` needs the
blocks. CSR indices are cached on the connectome (about `4 * (E + N + 1)` bytes for E synapses
and N neurons), and each brain's matrix shares its weights. Summation order differs slightly
between kernels. `graph.to_dict()["transport"]` is `"dense"` or `"segmented"`; `"sparse_kernel"`
is `None` until a sparse CPU multiply runs, then `"scipy_csr"` or `"numpy_segmented"`.

With Numba installed, the CPU backend settles blocked connectomes in one compiled loop (block
transport, then each neuron's update, activation, adaptation and nudge in place) with the
NumPy loop's float64 arithmetic whenever no trajectory or recording is requested, and computes
`residual` the same way. `CADENCE_FUSED=0` in the environment before `cadence` is imported
forces the NumPy loop.

A device kernel implements five operations: block products between neuron ranges, the
elementwise neuron update, the softmax over the nudged group, an equality check on still
ranges and the largest movement for the tolerance. Check a new one against the
neuron-by-neuron reference through `conformance`; `tests/test_brain.py` holds the tests every
kernel passes.

## Timing a decision

`cadence.timing.latency(decide, repeats=1000, warmup=20)` runs `decide()` `warmup` times, then
times `repeats` calls (an integer of at least 1; `warmup` at least 0) and returns `repeats`,
`p50_us`, `p90_us`, `p99_us`, `max_us`, `mean_us`, `jitter` (`(p99 - p50) / p50`),
`voluntary_switches` and `involuntary_switches` from `getrusage`, and `environment`.
`cadence.timing.environment()` records machine, system, cores, Python version, BLAS thread
limits, CPU affinity on Linux, load average and the NumPy, Numba and torch versions. On
Windows the load and switch counts are `None`.

```python
from cadence.timing import latency
graph = cd.NeuralGraph(connectome, neuron_model)
inputs = [{int(i): 1.0} for i in connectome.populations["input"]]
life = {"t": 0, "state": None}
def decide():  # a new input each call, continuing from the previous state
    life["t"] += 1
    life["state"] = graph.settle(inputs[life["t"] % len(inputs)], steps=30, state=life["state"])
    return life["state"].activation
report = latency(decide, repeats=50, warmup=5)
print(round(report["p50_us"]), report["environment"]["cores"])
```

`decide()` must advance a changing input and carry or reset state as the deployment would
(one unchanged warm input times a stable-state fast path) and synchronize accelerator work.
Report cold and warm results, output quality and residuals with the time.
