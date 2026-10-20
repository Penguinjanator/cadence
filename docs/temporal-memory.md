# Protect a temporal response

`TemporalMemory` keeps a [`TemporalPatchNet`](temporal.md) from changing a
response the caller marks as important. It stores orthonormal bases of the
protected presynaptic directions of the A, B and C blocks and projects later
updates out of them; it keeps no replay phrase and no output table, and it
infers no importance. It is a `TemporalPatchNet` tool that `Brain.compose` does
not use; the composed brain's memories are on [memory.md](memory.md). The
projection is orthogonal weight modification ([Zeng et al., *Continual learning
of context-dependent processing in neural networks*](https://www.nature.com/articles/s42256-019-0080-x)),
applied to the temporal model's local EP update.

```python
import numpy as np
from cadence import TemporalPatchNet, TemporalMemory

net = TemporalPatchNet(2, 8, 1, seed=151)
memory = TemporalMemory()
old_input, new_input = np.array([[[1.0, 0.0]]]), np.array([[[0.0, 1.0]]])
cold = np.zeros((1, 8))
old_output = net.imagine(old_input, state=cold).output
assert memory.protect(net, old_input, state=cold).maximum_residual < 1e-12

for _ in range(8):
    net.reset()
    assert memory.observe(net, new_input, np.full((1, 1, 1), 0.25)).updated
np.testing.assert_allclose(
    net.imagine(old_input, state=cold).output, old_output, atol=1e-12
)
```

`protect(net, inputs, *, state=None)` leaves the network unchanged. It runs the
free path from `state` (the live state when omitted, zero when there is none)
and adds its inputs and activity to the bases. It protects the current
response, errors included, and takes no target. Other cues and boundaries are
unprotected, so test retention with the same ones. An unconverged path raises
`ValueError`.

## A local quadratic parameter constraint

Every protected presynaptic vector `x` must satisfy `(W_new - W_old) @ x = 0`.
With `Q` an orthonormal basis of those vectors, the closest admissible update to
`D` in Frobenius norm is `D_protected = D @ (I - Q @ Q.T)`, applied twice to
remove floating-point residue. Fixing A's response on previous activity, B's on
inputs and C's on current activity preserves the protected causal path by
induction over time, in exact arithmetic. `TemporalMemory(*,
relative_tolerance=1e-12)` sets the rank cut: a new direction is admitted when
its singular value exceeds `relative_tolerance` times the Frobenius norm of the
new vectors, so retention holds to that precision. Admitted directions are
always kept. A tolerance outside (0, 1) raises `ValueError`.

## A local metric for poorly conditioned readout updates

New activity close to the protected span has a small free component, and the
plain projected readout step then learns slowly. A positive `readout_damping`
value `mu` rescales the C update by the free-activity covariance in the
unprotected directions (A and B keep the plain projected update):

```text
X = flattened free tanh activity,  P = I - Q_C @ Q_C.T,  S = X.T @ X / (batch * time)
Delta_C = -rate * gradient_C @ P @ inverse(P @ S @ P + mu * I) @ P
```

The gradient is the same centered EP gradient; the metric reads no target and
keeps no optimizer state. The default `None` uses the plain update; a
non-positive or non-finite value raises `ValueError`.

```python
net.reset()
attempt = memory.observe(net, new_input, np.full((1, 1, 1), 0.35),
                         beta=0.001, rate=1.0, readout_damping=1e-4)
assert attempt.updated or attempt.reason == "metric_step_rejected"
assert np.allclose(net.imagine(old_input, state=cold).output, old_output, rtol=0, atol=1e-12)
```

The candidate is accepted when its change along each basis is at most
`1e-12 * max(1, max(abs(W_before @ Q)))` and a target-free replay from the
original boundary lowers the weighted half-MSE by more than
`64 * eps * max(abs(old_loss), abs(new_loss), tiny)`. Otherwise the call returns
`updated=False`, `reason="metric_step_rejected"`, carries only the free
activity and keeps weights, counters and binding; near-ties can be rejected.
The extra hidden-width solve and replay are not in the phase counters.

## Atomic learning and saved state

`observe(net, inputs, target, *, beta=0.01, rate=0.1, readout_damping=None,
symmetry_tolerance=0.15, max_halvings=8)` runs centered EP and the projection on
private copies and commits network and memory together. As in
`TemporalPatchNet.observe`, beta halves until the contrast passes
`symmetry_tolerance`, at most `max_halvings` times; there is no parameter line
search. A failed phase or an asymmetric contrast returns `updated=False`, carries
the free activity and keeps weights and binding. An exception changes neither
object. `result.delta` is the raw EP derivative before projection; compare
committed parameters to measure the applied step. `observe` is single-threaded
and raises `TypeError` for anything but the base class, including
[`RecordPatchNet`](record-patch.md) and the [partitioned net](partitioned.md).

A parameter fingerprint binds the memory to the network: a change made outside
`observe` or `project` makes the next call raise `ValueError`, while resetting
the active state is allowed. `snapshot()` returns detached arrays (the
tolerance, `basis_A`, `basis_B`, `basis_C` and the 32-byte binding);
`TemporalMemory.restore` validates them, loads no pickled objects and raises
`ValueError` on a malformed checkpoint. Save and restore both objects:

```python
np.savez("protected.npz", **{"net_" + k: v for k, v in net.snapshot().items()},
         **{"memory_" + k: v for k, v in memory.snapshot().items()})
with np.load("protected.npz", allow_pickle=False) as saved:
    def part(prefix):
        return {k[len(prefix):]: saved[k] for k in saved.files if k.startswith(prefix)}
    restored = TemporalPatchNet.restore(part("net_"))
    recalled = TemporalMemory.restore(part("memory_"))
restored.reset()
assert recalled.observe(restored, new_input, np.full((1, 1, 1), 0.25)).updated
```

`report()` returns a `ConstraintReport`: `ranks` per block, `bytes` (bases,
eight for the tolerance, 32 for the binding; Python objects and solver
workspace are extra) and `maximum_residual`, which only `protect` fills with
the fit of its new vectors; a plain `report()` gives 0.0, which certifies no
retention. `project(before, proposed)` returns a constrained parameter dict and
advances the binding before the caller applies it with `net.set_parameters`;
use `observe` for training.

## Limits

Capacity is finite: each protected path consumes basis dimensions, and a full
basis freezes its block. Before that, new activity near the protected span needs
large weight changes, so report ranks and weight norms with retention. Protected
responses stay protected, errors included, with no decay, relevance selection or
contradiction resolution. Test retention and new learning together.
