# Backends, devices, precision

```python
import cadence as cd
cd.available_backends()
# {'cpu': 'numpy float64', 'torch': 'mps float32', 'mlx': 'gpu float32'}   # an M-series Mac with both installed
```

| backend | where it runs | precision | install | use it for |
|---|---|---|---|---|
| `"cpu"` | NumPy, and the fused numba kernel | float64 | `cadence-net[fast]` | receipts, conformance, anything you will cite; every readout |
| `"torch"` on CUDA | NVIDIA GPU | float64, or float32 with `precision="float32"` | `cadence-net[accel]` | training at scale; receipts too, in float64 |
| `"torch"` on MPS | Apple silicon GPU through Metal | float32 | `cadence-net[accel]` | training on a Mac when torch is what you have |
| `"mlx"` | Apple silicon GPU through MLX, unified memory | float32 | `cadence-net[apple]` | an alternative Apple backend; benchmark the actual workload |
| `"torch"` on CPU | torch CPU | float64 | `cadence-net[accel]` | one code path on a box without a GPU |

Every backend does the same arithmetic: the block transport of the wiring (dense blocks
between the owner ranges the named sets cut; a range that did not move keeps its product),
then one owner-local update, a nudge, adaptation, a mask, a tolerance, and `repair`. The
device backends keep the settled state on the device (`state.device`) so that a phase that
continues from it starts there, and the learning rule's contrast is read on the device
(`Settlement.contrast_on_device`). For blocked PyTorch learners, contrast, momentum,
RMS normalization and parameter updates remain on the device. Scalar step reports still
synchronize. Reading parameters or optimizer history, saving a checkpoint, or entering a
host-only path materializes the required arrays. Public optimizer attributes remain
mutable NumPy arrays; edits made through them are picked up by the next update. History
uses float32 on MPS and float64 on torch CPU/CUDA. MLX contrast currently returns arrays
to the host for the optimizer.
Large wirings whose blocks do not fit `dense_limit` use sparse transport. The CPU backend
uses SciPy CSR when installed, and the NumPy segmented sum otherwise. PyTorch uses its
gather/scatter path; `"mlx"` needs the blocks.

## Sparse CPU transport

A CSR row holds the overlaps heard by one owner. The matrix multiplies the published
activations directly, avoiding the NumPy fallback's temporary array with one value per
batch row and overlap. This changes transport only: local repair, nudges, masks,
adaptation and stopping conditions are the same. Floating-point summation order can
change slightly, so compare complete trajectories at the precision a task requires.

SciPy comes with `cadence-net[fast]`; the NumPy-only installation remains supported.
Sparse indices are cached on the wiring, and each engine's matrix shares its current
weights. Replacing parameters drops the old matrix wrapper; the topology can be reused.
For a batch of size B with E overlaps and N owners, one avoided float64 message array
occupies `8 * B * E` bytes, while the result occupies `8 * B * N` bytes. The CSR index
cache adds approximately `4 * (E + N + 1)` bytes when 32-bit indices suffice.

`engine.to_dict()["transport"]` retains `"segmented"` for sparse wiring compatibility.
`"sparse_kernel"` is `null` before a sparse CPU multiply runs, then `"scipy_csr"` or
`"numpy_segmented"`. Inspection does not instantiate a kernel.

## Which hardware

Choose using the actual changing-input workload. Small nets can spend more time
launching GPU operations than doing arithmetic; large batches and dense blocks can
benefit from accelerator matrix products. Sparse graphs require a separate measurement.
Warm up the backend and synchronize the GPU around wall-clock measurements.

The CPU backend uses NumPy float64, with optional numba and SciPy acceleration through
`cadence-net[fast]`. Cap BLAS threads when running independent experiment workers;
`cadence.timing.environment()` records thread settings. PyTorch supports CPU, CUDA and
Apple MPS, while MLX provides an additional Apple path. A single net currently uses one
device. Neither backend choice nor parameter count establishes efficiency by itself.

CUDA defaults to float64; `precision="float32"` selects float32 settlement. Hardware
throughput and the useful precision depend on the device and problem. Compare outputs,
residuals and learning curves with float64, especially near multiple equilibria. MPS
uses float32 for parameters and state because it does not support float64.

## Choosing a device

```python
cd.Settlement(w, rule, backend="torch")                          # cuda, else mps, else cpu
cd.Settlement(w, rule, backend="torch", device="cpu")             # force
cd.Settlement(w, rule, backend="torch", precision="float32")      # speed on a consumer GPU
cd.Settlement(w, rule, backend="mlx")                             # Apple silicon through MLX
```

A learner built on a device engine trains there; `Learner.load(path, backend="cpu")`
brings a checkpoint back to the receipt backend, whatever trained it.

## Precision matters

Owners can sit on knife edges, where a difference of 1e-7 in a drive flips a bistable
readout. Float32 summation order alone did that to a motor neuron in the fly brain. Two
habits keep this honest:

1. Make receipts on `"cpu"` or on CUDA float64.
2. When you use MPS float32 for a page or a demo, run `cd.conformance` on the same wiring
   and clamp, and show the deviation. It is usually around 1e-5; when it is not, a readout
   near threshold needs closer inspection.

## Extending to another device

The two device kernels, `settle._TorchKernel` and `settle._MlxKernel`, are each one class
with the same five operations: block products between owner ranges (a matrix product per
pair), the elementwise owner update, a softmax over the nudged group, an equality check on
the still ranges, and a max over the movement for the tolerance. Any array library with
those five can host a backend; the owner-by-owner reference and `conformance` are what
you check it against, and `tests/test_settle.py` has the test each kernel passes.


## The fused kernel

With numba installed (`pip install "cadence-net[fast]"`) the CPU backend settles blocked
wirings in one compiled loop: the block transport, then every owner's repair, activation,
adaptation and nudge in place, the same float64 arithmetic in the same order as the NumPy
loop (checked to 2e-16). Two more things it does are exact for the same reason: an owner
whose potential did not move keeps the activation it published, and a range that hears
nothing is skipped for good once it is still, because such an owner's update is a fixed
function of its own state. It is used automatically when the wiring is blocked and no
trajectory is requested; `CADENCE_FUSED=0` in the environment forces the NumPy loop. The
owner-by-owner reference and `conformance` are unchanged and remain what any kernel is
measured against.

The fused CPU path currently expects one shared mask with shape `(n,)`. For a
different ablation mask per batch row, use sparse CPU settlement
(`dense_limit=0`) or call `settle` separately for each row. The NumPy sparse path
accepts `(batch, n)` masks; this does not imply that the fused kernel supports them.

## Timing a decision

`cadence.timing.latency(decide)` calls `decide()` a thousand times and reports the median,
the 90th and 99th percentiles and the maximum in microseconds, with the number of voluntary
and involuntary context switches the scheduler made during the measurement (from
`getrusage`), and `environment()` records the thread limits, the cores the process is
pinned to when the platform can say (Linux), the load average and the library versions. A
wall-clock number in a receipt is a fact about a program on a machine on a day; this is the
machine's half of it. A tail far above the median with many involuntary switches is the
machine, not the net.

For a changing-input workload, `decide()` must advance an input sequence and carry or
reset state according to the deployment contract. Repeating an unchanged input measures
a stable-state fast path. Report cold and warm results, output quality, and residuals
separately; elapsed time on that fast path does not establish general inference speed.
