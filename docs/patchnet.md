# PatchNet

`PatchNet` wraps one reciprocal `NeuralGraph` and its local `Learner` for
continuous observations. Inputs enter as soft drives, the whole graph settles,
outputs are read from that same state, and `observe` learns from the
free/nudged contrast on the output ports you mark as observed. It has no
associative stores and no action loop; [`Brain.compose`](brain.md) has those.
`create` builds a fully reciprocal input/hidden/output graph;
[`recursive`](recursive-settlement.md) adds observing populations.

```python
import numpy as np
import cadence as cd

net = cd.PatchNet.create(
    inputs=2, hidden=8, outputs=1, seed=7, density=1.0,
    config=cd.LearnerConfig(nudge="quadratic", eta=0.2, eta_bias=0.02),
)
drive = net.stimulus(np.array([[0.2, 0.8]]))
result = net.observe(drive, np.array([[0.6]]), source_id="sensor:1")
assert result.updated, result.reason
print(net.read(result.free))
```

| Argument | Default | Meaning |
|---|---|---|
| `create(inputs, hidden, outputs)` | | port counts of the factory graph |
| `seed`, `density`, `init` | `0`, `0.3`, `1.0` | graph seed, input-to-hidden density, initial weight scale |
| `config` | `LearnerConfig(nudge="quadratic")` | learning rule, see [learning](learning.md#7-every-knob) |
| `backend`, `device` | `"cpu"`, `None` | see [backends](backends.md) |
| `input_index` | the `"input"` population | neurons that `stimulus` drives |
| `steps`, `chunk`, `tolerance` | `512`, `32`, `1e-5` | sweeps per phase, sweeps between residual checks, equation tolerance |
| `source_capacity` | `256` | evidence-ID window; `0` turns it off |
| `context_strength`, `context_mask` | `0.0`, hidden neurons | overlap with the previous free activity |
| `solver`, `refinement_steps` | `"local"`, `64` | `"hybrid"` adds bounded Newton refinement |

Rows from `input_index` down are arguments of `PatchNet(learner, ...)`, which
wraps any compatible `Learner`; `create` forwards them as `**runtime_options`.

## What persists

Activity is carried between calls. Weights and biases change only through
`observe` or an explicit parameter edit. `settle(drive)` advances the live
activity without learning; `read()` returns the outputs of the live state or of
a given phase. `reset()` clears activity and keeps parameters, optimizer
history and the evidence-ID window. Batch rows are persistent streams: a drive
whose batch size differs from the carried state raises `ValueError`, and you
call `reset()` whenever the rows stop being the same streams.

With `context_strength=kappa > 0`, each solve adds the energy
`kappa/2 * sum(context_mask * (activity - c)**2)`, where `c` is the previous
event's free activity (zero after `reset`). `c` is fixed across the phases of
one event and then advances to the new free activity; the nudged state never
becomes `c`, and no gradient flows through earlier events. The default mask is
the `"hidden"` population; a graph without one needs an explicit nonempty
boolean mask. Large strengths can need a smaller integration step. This overlap
is experimental; durable memory lives in the [composed brain](memory.md).

## Observed ports and local detuning

`stimulus(inputs, amplitude=1.0)` writes `amplitude * inputs` on the input
neurons as a drive; the network also feeds back onto them.
`observe(drive, target, observed=None, weight=None, source_id=None)` takes
targets of shape `(batch, outputs)`, one boolean `observed` vector shared by
all rows (default all true) and an optional nonnegative gain per row.
Unobserved targets may be NaN.

`observe` settles a target-free free phase, then a `+beta` phase nudged toward
the targets and, only when `LearnerConfig.centered=True` (the default), a
`-beta` phase, both from the free state. With batch size `B`, outputs `s`,
targets `y`, mask `m` and gains `w`, the objective is

`L = (1 / (2 B)) * sum_b w[b] * sum_o m[o] * (s[b, o] - y[b, o])**2`.

`L` is divided neither by the observed-port count nor by the sum of gains; with
all ports observed and unit gains it is `outputs / 2` times the mean squared
error.

The update commits only when every required phase qualifies. A rejected attempt
keeps the finite free activity and leaves weights, optimizer and IDs unchanged.
`PatchObservation` holds `free`, `plus`, `minus` (each an `Equilibrium` or
`None`), `updated`, `reason` and the learner's `metrics`.

| `reason` | Meaning |
|---|---|
| `updated` | all phases qualified and the parameters moved |
| `duplicate` | this `source_id` was committed with the same payload; nothing ran |
| `no_observations` | empty mask or all gains zero; only the free activity advanced |
| `free_unconverged`, `free_unqualified` | the free phase missed the tolerance, or failed the hybrid checks |
| `nudge_unconverged`, `nudge_unqualified` | a nudged phase missed the tolerance, or failed the hybrid checks |

The constructor and each `observe` raise `ValueError` unless the learner is
quadratic with one slot, its effective weights are reciprocal without
adaptation, every contact has a reciprocal candidate contact, and reciprocal
contacts have equal contact/gain factors and plasticity masks. Both factories
meet this. Why the contrast follows the gradient of `L` is in
[learning](learning.md#5-why-the-contrast-is-a-gradient).

## Train on observed targets

Fit input statistics on training data only and freeze them; floor the standard
deviation and leave flags such as "value missing" unstandardized. Keep targets
inside the output range, about `(-0.1, 1)` for `create` and `(-1, 1)` for
`recursive`, fix their scale before evaluation and report clipping. Mark an
outcome you did not observe as unobserved; never write a zero for it.

Call `reset()` before every independently sampled minibatch and evaluation
batch. Since the mask is shared, group records that each observe one output by
that output, or teach them one at a time. Use `source_capacity=0` for
deliberate offline replay, and evaluate held-out data without calling
`observe`. A lesson's phases belong to the pre-update parameters, so settle
again before acting and act only on rows whose `qualified` is true. The bias
part of
`learner.contrast(free, plus, minus)` is the negative gradient of `L`:

```python
def quadratic_loss(prediction, target, observed):
    return float(0.5 * np.mean(np.sum((prediction - target)[:, observed] ** 2, axis=1)))

small = cd.PatchNet.recursive(
    3, [4, 2], 3, seed=83, steps=256, chunk=8, tolerance=1e-10, source_capacity=0,
    config=cd.LearnerConfig(nudge="quadratic", beta=0.01, eta=0.01, eta_bias=0.01),
)
drive = small.stimulus(np.array([[0.2, -0.1, 0.3], [-0.2, 0.1, 0.4]]), amplitude=2.0)
observed = np.array([False, True, False])
target = np.full((2, 3), np.nan)
target[:, 1] = [0.3, -0.2]
bias, port = small.brain.bias.copy(), int(small.output_index[1])

def loss_with_bias(shift):
    small.brain.bias = bias + shift * (np.arange(len(bias)) == port)
    small.reset()
    return quadratic_loss(small.read(small.settle(drive)), target, observed)

numeric = (loss_with_bias(1e-5) - loss_with_bias(-1e-5)) / 2e-5
small.brain.bias = bias
small.reset()
lesson = small.observe(drive, target, observed=observed)
assert lesson.updated, lesson.reason
_, neurons = small.learner.contrast(lesson.free.state, lesson.plus.state, lesson.minus.state)
assert np.isclose(-neurons[port], numeric, rtol=1e-3)
```

## The hybrid solver

`solver="hybrid"` runs the local iteration, then up to `refinement_steps` warm
Newton steps on the global energy for unresolved rows, with the inputs
eliminated. It needs the CPU float64 backend, the smooth `tanh(v/2)` neuron
(slope 1, threshold 0, leak 1) without adaptation, exact reciprocal effective
weights, zero weights among inputs, disjoint input and output indices, and no
context anchoring of inputs. `recursive` graphs qualify; `create` graphs (leak
0.1) are refused. The structure is rechecked before every solve, since
learning moves the weights.

A hybrid row qualifies when its full residual meets `tolerance` and the
smallest eigenvalue of the reduced energy Hessian is positive: a local
stability check, without a global-minimum or uniqueness guarantee.
`Equilibrium.converged` tests the residual alone, `qualified` adds the solver
checks, `steps` counts local sweeps, and `refinement` (`None` for local solves)
is a `RefinementReport` with per-row `local_residual` (before refinement),
accepted Newton `steps`, `min_curvature` and `status`.
`status` is `local` or `refined` on a qualified row; other values
(`step_cap`, `nonpositive_curvature`, `line_search_exhausted`, ...) name the
failure. The counters omit line-search trials and dense matrix work.

With `I` input and `C` other neurons, refinement stores `I x C` and `C x C`
blocks and costs about `O(I C^2)` per row for the curvature plus `O(C^3)`
dense algebra, so it suits small cores. Learning credit is the local phase
contrast in both modes; [settling certificates](certificate.md) bound the
local solver.

```python
hybrid = cd.PatchNet(small.learner, steps=256, chunk=8, tolerance=1e-10, solver="hybrid")
phase = hybrid.settle(drive)
assert phase.qualified.all(), phase.refinement.status
print(phase.refinement.min_curvature)
```

## Evidence and imagined continuations

Only `observe` accepts teaching evidence. A `source_id` (nonempty, at most 1024
characters) repeated with the same payload returns `duplicate` without
advancing live state; the same ID with a different payload raises
`ValueError`. IDs are recorded on `updated` only; the oldest expires once
`source_capacity` is exceeded and can then be accepted again. The window
suppresses repeats; distinct IDs are not thereby independent evidence.

`imagine(drives, state=None)` settles a sequence of drives on a private branch
from the live state (or `state`) and returns a tuple of `Equilibrium`. Pass
the last branch state to continue a rollout. Live activity, parameters,
optimizer and IDs are untouched, also when a drive raises. Feedback of
predictions and action choice are application code; nothing imagined becomes a
target.

```python
branch = net.imagine([net.stimulus(np.array([[0.3, 0.7]])),
                      net.stimulus(np.array([[0.4, 0.6]]))])
more = net.imagine([net.stimulus(np.array([[0.5, 0.5]]))], state=branch[-1].state)
print(net.read(more[-1]))
```

## Checkpoint and readback

`save(path, compressed=True)` atomically writes a `cadence-patch/3` NumPy
archive with learner parameters and optimizer, current activity, the ID window
and every constructor setting. `load(path, backend="cpu", device=None,
precision=None)` validates it and also reads `cadence-patch/1` and `/2`, which
load with the local solver (the [API reference](api.md) lists each format).
Exact continuation needs the same backend and precision; input statistics, RNG
state and data order belong to the application. `state` returns a detached
copy of the live activity and `snapshot()` the checkpoint as a dict of detached
arrays.

```python
path = net.save("patch-life.npz")
resumed = cd.PatchNet.load(path)
np.testing.assert_array_equal(resumed.state.v, net.state.v)
resumed.reset()  # an independent evaluation starts from fresh activity
```
