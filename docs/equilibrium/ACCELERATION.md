# Execution and cost

The device changes how repair proposals are computed. The energy, the learning
rule and the float64 admission test are the same on every device, so choose a
connected graph and a useful teaching stream first ([design guide](BRAIN_DESIGN.md))
and a device second.

## Select execution explicitly

`Cortex(device=..., dtype=...)` selects the engine. The default is the
standard-library float64 reference engine. Tensor devices need the optional
extra, `python -m pip install "cadence-net[accel]"`.

| `device` | Proposal arithmetic | `dtype` |
| --- | --- | --- |
| `"python"` (default) | Standard-library reference engine | `"float64"` only |
| `"cpu"` | PyTorch tensors on CPU | `"float64"` (default) or `"float32"` |
| `"mps"` | PyTorch tensors on an Apple GPU | `"float32"` only |
| `"cuda"` or `"cuda:N"` | PyTorch tensors on an NVIDIA GPU | `"float64"` (default) or `"float32"` |

`"cuda"` resolves to `"cuda:0"`; any other combination raises `ValueError` at
construction. Construction and checkpoint loading do not import PyTorch or
reserve a device; the first solve does, raising `ImportError` without PyTorch and
`ValueError` for unavailable hardware. Cadence never falls back to another
device. PyTorch supplies only tensor arithmetic: Cadence computes its own
analytic derivatives and repair steps, with no autograd or optimizer.

```python
from cadence.experimental.equilibrium import Brain, Cortex, bootstrap

layout = Cortex(seed=2, device="python")  # or "cpu", "mps", "cuda"
signal = layout.input("signal", shape=1)
base = layout.column("perception", patches=4, inputs=signal)
response = layout.column("response", patches=2, inputs=(signal, base))
layout.output("answer", shape=1, reads=response)
brain = layout.build()

examples = [({"signal": [x]}, {"answer": [x]}) for x in (-0.8, 0.8)]
checks = [({"signal": [x]}, {"answer": [x]}) for x in (-0.4, 0.4)]
report = bootstrap(brain, examples, checks=checks, max_error=0.2)
assert report["passed"], report
```

## Where coupled settlement spends time

`build()` raises `ValueError` for a population that reads only sensors and is
read by no other population, and for clusters that settle apart from the rest.
Such a patch has a closed-form answer, `clip(p / (1 + state_prior))` of its
sensory prediction `p`, with no repair between patches, so its speed says
nothing about a brain.

```python
lonely = Cortex(seed=0)
alone = lonely.column("alone", patches=2, inputs=lonely.input("signal", shape=1))
lonely.output("answer", shape=1, reads=alone)
try:
    lonely.build()
    raise AssertionError("an uncoupled population must be refused")
except ValueError as error:
    assert "settles with no other population" in str(error)
```

In a connected graph, state and error contacts couple the patches, and every
answer must pass the stationarity test over all eligible coordinates. Each
evaluation traverses patches and edges forward and backward, roughly linear in
graph size. The number of evaluations depends on conditioning, weights,
starting state, clamps and `tolerance`, and can vary substantially. In a
single-row query, predictions that depend only on fixed sensors are computed
once ([query cache](SPECIFICATION.md#query-derivatives-and-cache)). Tensor
execution groups work within error-dependency levels, so each extra
error-observation level adds a sequential stage. Count `work["evaluations"]`,
`work["proposals"]`, `work["backtracks"]`, `work["edge_visits"]`, `sweeps` and
wall time; a sweep budget does not bound elapsed seconds.
`python examples/equilibrium/layout_cost.py --out <new path>` compares the
query cost of equally wide untrained layouts at fixed parameters.

## Batch experience on one device

`brain.observe_batch(examples)` and `bootstrap(..., batch_size=N)` admit labeled
rows with shared parameters and a private state per row. Every row starts from
the same retained activity, and the rows form no temporal sequence. Tensor
devices vectorize the rows; the Python engine runs the same operation.
Parameters repair against the mean row objective with one fixed pre-batch
anchor, while each row's state residual meets the full tolerance. The batch
commits its parameters and one event identity only when every row qualifies,
and the live state is kept.

```python
batch = brain.observe_batch([({"signal": [x]}, {"answer": [x]}) for x in (-0.6, 0.6)])
assert batch["qualified"] and len(batch["outputs"]) == 2
```

Start with 4 or 8 rows. Temporary memory grows with the row count, and grouping
changes the learning path compared with ordered single admissions, so check
held-out accuracy, retention, presentations, `updates` and refusals along with
throughput. Admissions to one brain stay serial: there is no multi-GPU learning
or checkpoint averaging.
`Reinforcement` uses the same batch execution with `source="estimate"`
([live operation](LIVE.md)).

[`batch_bootstrap.py`](../../examples/equilibrium/batch_bootstrap.py) teaches a
small relation (8 sensors, populations of 12 and 4 patches, 32 teaching rows,
8 readiness checks, 16 fresh tests) on each listed device and prints JSON with
startup, bootstrap and fresh-test times, `passed`, errors and repair work:

```sh
python examples/equilibrium/batch_bootstrap.py --devices python cpu mps --batch-size 8 --repeats 3 --threads 1
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--devices` | `python` | Devices to run; an unavailable device fails |
| `--batch-size` | 8 | Rows per batch, 1 to 128 |
| `--repeats` | 3 | Seeds per device, 1 to 128 |
| `--epochs` | 20 | Bootstrap epoch limit, 1 to 128 |
| `--threads` | 1 | PyTorch CPU threads per process |

## Keep admission precise

Device arithmetic only proposes. The Python float64 engine rechecks every
candidate against the original inputs, clamps and anchors at the requested
`tolerance`, refines it or restarts from the original coordinates within the
remaining budget, and commits nothing it refuses
([rules](SPECIFICATION.md#optional-tensor-execution)). Float32 never loosens
admission. It gets at most `max(1, budget // 2)` device sweeps, can reach a
different stationary point than float64, and raises `ValueError` for values
that do not fit the dtype, without changing the continuation.

A tensor solve adds `result["execution"]` (`device`, `dtype`, `torch`,
`tensor_sweeps`, `reference_sweeps`, `reference_evaluations`,
`reference_restart`). `work`, `sweeps` and `energy_history` join both phases;
`energy`, `errors`, `stationarity` and `qualified` are final reference values.

## Move a continuation between devices

`Brain.from_snapshot(text, device=..., dtype=...)` validates the whole
checkpoint, then applies the override:

```python
saved = brain.snapshot()
reference = Brain.from_snapshot(saved, device="python")
assert reference.state == brain.state and reference.weights == brain.weights
assert reference.inspect()["admissions"] == brain.inspect()["admissions"]
```

<!-- not-run: Requires the optional PyTorch installation and available MPS hardware. -->
```python
gpu = Brain.from_snapshot(saved, device="mps")
assert gpu.settle({"signal": [0.4]})["qualified"]
```

Arrays and the admission cursor carry over exactly. A device without `dtype`
takes that device's default, a `dtype` alone keeps the saved device, and
neither keeps the saved settings. `restore` requires the exact configuration,
and checkpoints from different engine sources are refused. A `Reinforcement`
checkpoint also holds replay, RNG and pending-action state and has no device
override: choose the device when building its brain, because a brain-only
transfer is not a whole-life restore.

Tensor arithmetic varies with hardware and PyTorch version, and
[CUDA `index_add_` can be nondeterministic](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.index_add_.html),
so a seed does not guarantee bitwise GPU replay. Cadence leaves PyTorch's
determinism flags alone.

## Parallel simulated lives

Independent lives can run in separate processes. Each worker owns its brain,
because one admission changes the anchors used by the next; sharing a brain or
averaging checkpoints breaks that order. Collectors can instead send one
ordered witness stream to a single owner. Use the spawn start method for
workers that use PyTorch (a forked worker can inherit OpenMP state and
deadlock), bound each worker's tensor and BLAS threads, and do not start one
GPU process per CPU core.

[`parallel_bootstrap.py`](../../examples/equilibrium/parallel_bootstrap.py)
gives each life a point-body simulator on the Python engine: the brain learns
the next position from position and commanded velocity, then predicts fresh
outcomes before admitting them. It prints readiness, work, admissions, wall
time and checkpoint digests, and exits 1 if any life fails readiness.

```sh
python examples/equilibrium/parallel_bootstrap.py --lives 4 --workers 2
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--lives` | 4 | Independent brains, 1 to 64 |
| `--workers` | 1 | CPU processes, 1 to 64; 1 runs every life in this process, more use spawned workers |
| `--epochs` | 20 | Bootstrap epoch limit, 0 to 100 |
| `--seed` | 0 | First nonnegative seed; lives use consecutive seeds |

## Measure before choosing hardware

Try `device="cpu"` first: on graphs of a few patches, dispatch and synchronization cost more
than the arithmetic, and process startup can make a small parallel workload slower than a
serial one. Float32 can be slower than float64, because its trajectory differs and can need
more sweeps. Sweep counts near a budget cap vary by platform, so a cap that admits an example
on one machine can refuse it on another; repeat every timing.

Compare arms from the same starting arrays, inputs, witness order and `tolerance`. Record
device, dtype, hardware, library versions, refusals and work beside the time, and count
repeated bootstrap presentations apart from new experience. For a real-time application,
start from the smallest coupled baseline with the actual sensors and add levels only against
that control at matched capacity and time budget. Time complete commands (preprocessing,
queries, learning, qualification, queue delay, transfers, action delivery) after a warmup,
and report latency distributions and deadline misses. Every query qualifies the complete
connected brain, so a scheduler cannot accept a partly solved branch; the
[live controller](LIVE.md#keep-the-body-responsive) keeps the body responsive without making
an answer ready sooner.

To check CUDA correctness on your hardware, use a clean, committed checkout
installed with `python -m pip install -e ".[dev]"` and a CUDA build of PyTorch:

```sh
python -X utf8 examples/equilibrium/cuda_qualification.py --out data/cuda-qualification.json
```

[`cuda_qualification.py`](../../examples/equilibrium/cuda_qualification.py)
runs the CUDA cases of the tensor and batch tests at both precisions; `--full`
runs all of `tests/equilibrium`. `--out` must be a new path. It refuses a dirty
checkout, fails on skipped CUDA cases or sources changed during the run, and
records the commit, source hashes, hardware and library versions. Recorded
runs are in [examples/equilibrium/receipts](../../examples/equilibrium/receipts).
They cover correctness on small fixtures; speed and memory peaks need your own
measurement.
