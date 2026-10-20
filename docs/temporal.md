# TemporalPatchNet

`TemporalPatchNet` learns transitions over short observed paths. Each moment of a path is a
patch with hidden activity and an output, overlapping the moment before it. Inputs enter
declared ports; targets are read only while learning. Learning repairs the whole path at
once, so an error late in the path changes the activity of earlier moments.

```python
import numpy as np
from cadence import TemporalPatchNet

net = TemporalPatchNet(inputs=2, hidden=4, outputs=1, seed=7)
observations = np.zeros((2, 6, 2))
observations[:, 0, 0] = [-1.0, 1.0]  # one initial cue per stream
observations[:, :2, 1] = 1.0        # a supplied observation-presence port
heard = np.array([0.0, 0.0, 0.2, -0.2, 0.2, -0.2])
target = np.stack([heard, -heard])[:, :, None]

result = net.observe(observations, target, beta=0.01, rate=0.1)
assert result.updated and result.reason == "updated"
assert result.free.converged and result.plus.converged and result.minus.converged
np.testing.assert_array_equal(net.state, result.free.final_state)
```

## A finite observed path

Inputs and targets have shape `(batch, time, ports)`; batch rows are independent streams
sharing one set of parameters. The live state has shape `(batch, hidden)`, so call `reset()`
before changing the batch size.

`observe(inputs, target, beta=0.01, rate=0.1, backtrack=False, symmetry_tolerance=0.15,
max_halvings=8)` solves three phases from one initial boundary: the free path and a positive
and a negative teaching path. `rate` must be finite and nonnegative. It returns a
`TemporalObservation` with the phases (`plus` and `minus` are `None` when the free phase
fails), `updated`, `reason`, the centered step `delta` and the `beta` used. Only a converged
free phase's final state becomes the live state. A failed phase changes no parameters and no
update count; invalid input changes nothing. `net.updates` counts committed parameter steps.
Measure accuracy on withheld paths.

| Constructor option | Default | Effect |
| --- | --- | --- |
| `seed` | `0` | Initial `A`, `B`, `C`. |
| `output_precision` | all ones | Per-output teaching weights ([below](#explicit-output-teaching-precision)). |
| `initial_radius` | `0.95` | `A` starts as this times a random orthogonal matrix. |
| `tolerance` | `1e-9` | Gradient bound a phase must meet to converge. |
| `max_iterations` | `80` | Newton iterations per teaching phase. |
| `max_backtracks` | `24` | Step sizes tried in each Newton line search (1, then halving). |
| `max_damping_trials` | `16` | Damping levels per iteration (0, then `1e-8` growing tenfold). |

## Context, private continuation and readback

The learned arrays `A`, `B`, `C` are kept apart from the live hidden state. `advance(inputs)`
runs the free recurrence without learning and carries its final state when it converges.
`imagine(inputs, state=None)` starts at the live state or an explicit `state` and changes
nothing; pass a zero state for a cold branch. `settle(inputs, target=None, beta=0.0,
state=None)` is the read-only detuned solve for diagnostics and ignores the target at
`beta=0`. `reset()` clears the live state and readback and keeps parameters and `updates`.

```python
net.reset()
primer = net.advance(observations[:, :2])
assert primer.converged
before = net.snapshot()
future_inputs = np.zeros((2, 8, 2))
branch = net.imagine(future_inputs)
assert branch.converged
for name, values in before.items():
    np.testing.assert_array_equal(values, net.snapshot()[name])

readback = net.readback()
assert readback.state.shape == (2, 4)
assert np.all(np.abs(readback.activity) <= 1.0)
assert readback.state_parameter_revision == readback.parameter_revision
```

`readback()` returns a `TemporalReadback` with `state`, `activity` (its tanh), the last
carried free `residual` and `energy`, `updates`, `parameter_revision` and
`state_parameter_revision`. After `observe` it labels the revision of the free phase, which
precedes the update. `state` and `parameters()` return copies; `set_parameters` replaces
`A`, `B` and `C` together after validation, keeps the live state and raises the revision.

The caller decides which ports carry observations, intentions or controls. A zero input is
zero drive; when an observed zero must differ from an absent observation, add a presence
port, as the first example does. The solver inserts no clock, future cue or desired output
into free inference.

`snapshot()` and `TemporalPatchNet.restore(snapshot)` copy the complete state.
`save(path, compressed=True)` writes an atomic `.npz` archive (`compressed=False` writes
faster) and `load(path)` restores it without pickle.

```python
path = net.save("temporal_brain.npz")
restored = TemporalPatchNet.load(path)
np.testing.assert_array_equal(
    restored.imagine(future_inputs).output,
    net.imagine(future_inputs).output,
)
```

## Energy and update

For inputs `u[t]` and a fixed initial boundary `h[-1]`:

```text
e[t] = h[t] - A tanh(h[t-1]) - B u[t]
r[t] = y[t] - C tanh(h[t])
E = mean_batch(0.5 sum_time(||e[t]||² + ||r[t]||²))
L = 0.5 mean_batch,time,outputs(output_precision * (y - target)²)
```

Free inference is the causal recurrence. The teaching phases minimize `E + beta L` and `E - beta L` from the free path. Outputs are eliminated
with `b[j] = beta * output_precision[j] / (time * outputs)`, and each negative-phase `b[j]`
must exceed `-1`. Adjacent hidden-width blocks exchange precision matrices and information
vectors in an exact block Hessian solve. A damped Newton step needs positive Schur pivots and
an Armijo decrease (rounding allowance `1e-15 * max(1, abs(previous_energy))` per stream). A
phase converges when the full state gradient meets `tolerance` and the final undamped Hessian
has positive pivots, which certifies a local minimum only.

The parameter step uses local products and no derivative through the solver:

```text
dE/dA = -mean_batch(sum_time(e[t] tanh(h[t-1])ᵀ))
dE/dB = -mean_batch(sum_time(e[t] u[t]ᵀ))
dE/dC = -mean_batch(sum_time(r[t] tanh(h[t])ᵀ))
delta = (dE_plus - dE_minus)/(2 beta)
parameter <- parameter - rate * delta
```

The boundary is held fixed, so learning does not reach into earlier calls. On a smooth
stable branch the contrast approaches the free-loss gradient as beta goes to zero;
`tests/test_temporal.py` checks it against independent derivatives. The free recurrence has
zero residual by construction even when its predictions are wrong, so measure surprise and
accuracy with the observation-minus-prediction error.

## Explicit output teaching precision

Teaching precision defaults to one per output. Supply finite, strictly positive weights
for a declared task metric or noise assumption, here or as `output_precision=` at
construction.

```python
precision = np.array([2.0])
net.set_output_precision(precision)
np.testing.assert_array_equal(net.output_precision, precision)
```

The change is atomic and touches only the teaching loss: parameters, live state, revisions
and free diagnostics are unchanged, and [`TemporalMemory`](temporal-memory.md) bindings stay
valid. Checkpoints store the vector. `observe` requires
`0 < beta * max(output_precision) < time * outputs`, so large weights need a smaller beta.
Weights computed from training statistics should be fixed before evaluation, with the
unweighted metric reported too.

## Checking that the contrast is centered

The contrast is a derivative only while the two teaching paths sit symmetrically around the
free path; a teaching solve can converge on another branch of the energy and report success.
`observe` and [`plan`](planning.md) compute `cadence.temporal.contrast_asymmetry(free, plus,
minus)`, the norm of `plus + minus - 2 * free` over the norm of `plus - minus` on the hidden
paths. The ratio grows in proportion to beta and the contrast error with its square.

Above `symmetry_tolerance` (default 0.15) both phases are solved again at half the beta, up
to `max_halvings` times (default 8). A pair that stays off center is refused with
`contrast_asymmetric`: parameters, revisions and `updates` do not change and the valid free
state is carried. The result reports the last `beta` and `contrast_halvings`.
`symmetry_tolerance=np.inf` disables the check. A small ratio does not bound the gradient error.

## Checking a learning step

A good direction can overshoot. `observe(..., backtrack=True)` tries the rate and up to
fifteen halvings. Each candidate predicts the path without targets from the original
boundary and is accepted when its precision-weighted half-MSE drops below the old loss by a
floating-point floor and `new_loss <= old_loss - 1e-4 * step * ||delta||²` for the tried
rate `step`. The result
reports `initial_loss`, `final_loss`, `accepted_rate`, `replay_losses` (`None` for a
nonfinite or unsolved candidate) and `replay_calls`. With no decreasing step the reason is
`no_decreasing_parameter_step`; parameters are kept and only the free state is carried. Rate
zero cannot commit. The check covers the current path only, and
[`TemporalMemory.observe`](temporal-memory.md) does not run it.

## Reasons

| `TemporalObservation.reason` | Meaning |
| --- | --- |
| `updated` | Parameters changed; `updates` and the revision rose by one. |
| `free_phase_failed` | The free path did not converge. |
| `phase_failed` | A teaching phase failed; see `plus.reason` and `minus.reason`. |
| `contrast_asymmetric` | The pair stayed off center after `max_halvings`. |
| `nonfinite_update` | The proposed parameters were not finite. |
| `no_decreasing_parameter_step` | With `backtrack=True`, no tried rate lowered the loss. |

| `TemporalPhase.reason` | Meaning |
| --- | --- |
| `exact_free`, `free_residual_failed` | Free path met or missed `tolerance`. |
| `nonfinite_free_state` | The recurrence overflowed. |
| `nonfinite_energy`, `nonfinite_residual` | Teaching energy or gradient became nonfinite. |
| `converged` | Teaching phase met `tolerance` with positive pivots. |
| `iteration_cap` | `max_iterations` reached. |
| `nonminimum_stationary_point` | Stationary, but the Hessian was not positive. |
| `no_decreasing_spd_step` | No damping level gave a descending step. |
| `full_residual_failed` | The residual including outputs missed `tolerance`. |

A `TemporalPhase` also exposes `hidden`, `output`, `activity`, `final_state`, `residual`,
`energy`, `energy_history`, `iterations`, `accepted_step_sizes`, `accepted_damping`,
`minimum_pivot_eigenvalue`, `block_chain_attempts` (failed solves and the final curvature
check included), `energy_evaluations` (line searches included) and `message_bytes`.

## Cost and memory

Parameters take `8*(H² + H*inputs + outputs*H)` bytes for hidden width `H`; precision adds
`8*outputs` and the live state `8*batch*H`. The net keeps no training paths, replay buffer or
protection against interference.

For batch `N` and path length `T`, one teaching chain factorization costs on the order of
`N*T*H³` operations. Its message storage, reported as `message_bytes`, is `8*N*T*(H² + H)`
bytes in float64; Hessian blocks, arrays and solver workspace cost more. Free prediction skips
the factorization and runs sequentially over time. The net runs in NumPy on the CPU: the
optional backends of the settling brain do not accelerate it, and
[connection masks](partitioned.md) do not make the solver sparse. Planning over action ports
is on the [planning page](planning.md); work accounting is on [evaluation](evaluation.md).
