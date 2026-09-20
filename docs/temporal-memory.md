# Explicit response protection

`TemporalMemory` constrains how a `TemporalPatchNet` may change after a caller
marks a learned response as important. It retains orthonormal bases of the
protected presynaptic directions for A, B and C. It stores neither a raw replay
phrase nor a future-output lookup table. Importance is **supplied**, not inferred.

The method is related to orthogonal weight modification: projecting updates to
preserve earlier responses has established prior art in [Zeng et al., *Continual
learning of context-dependent processing in neural networks*](https://www.nature.com/articles/s42256-019-0080-x).
Cadence applies this constraint to its temporal residual model's local EP update;
it does not claim the projection principle is novel.

```python
import numpy as np
from cadence import TemporalPatchNet, TemporalMemory

net = TemporalPatchNet(2, 8, 1, seed=151)
memory = TemporalMemory()
old_input = np.array([[[1.0, 0.0]]])
new_input = np.array([[[0.0, 1.0]]])
cold = np.zeros((1, 8))
old_output = net.imagine(old_input, state=cold).output
report = memory.protect(net, old_input, state=cold)
assert report.maximum_residual < 1e-12

for _ in range(8):
    net.reset()
    result = memory.observe(net, new_input, np.full((1, 1, 1), 0.25))
    assert result.updated
np.testing.assert_allclose(
    net.imagine(old_input, state=cold).output, old_output, atol=1e-12
)
```

`protect` is read-only with respect to the network. It follows the free path
from the supplied initial boundary and gathers the actual inputs and neural
activity into bases. It protects the model's **current response**, including any
error that response already contains. Targets are not an argument. Preserve
the same initial boundary and query when testing retention; other queries,
nearby cues, or different live context are not automatically protected.

## A local quadratic parameter constraint

For every protected presynaptic vector `x`, require
`(W_new - W_old) @ x = 0`. If `Q` has orthonormal columns spanning the protected
vectors, the closest admissible update to `D` in Frobenius norm is

```text
D_protected = D @ (I - Q @ Q.T)
```

This is a quadratic overlap constraint on local parameter responses. It is
solved analytically; it is not an additional teacher or a gradient through the
temporal solver. A numerical second projection removes residual floating-point
components along the same basis. Previously admitted basis directions are
retained when later data have a different scale. Singular values below the
configured tolerance are discarded, so finite-precision retention is measured,
not presumed exact.

Preserving A's responses on previous activity, B's responses on inputs, and C's
responses on current activity preserves the selected causal path in exact
arithmetic, by induction. Three [Lean lemmas](../lean/CadenceMission/ProtectedTemporalPath.lean)
prove the conditional induction from exact map agreement. They do not certify
the numerical SVD or prove that a chosen response is true or useful.

As a basis fills its input space, that parameter block loses all plastic
directions. Protecting many paths can eventually prevent compatible new
learning. There is no automatic decay, contradiction resolution, relevance
selection or unlimited capacity claim. Retention and remaining new-task
acquisition must be tested together.

## Atomic learning and saved state

`memory.observe(net, inputs, target, beta=..., rate=...)` stages ordinary centered
EP and constrained projection on private snapshots. It commits the network and
constraint binding together after validation. An exception leaves both live
objects unchanged. Failed phases carry only valid free activity, just as
`TemporalPatchNet.observe` does, while keeping parameters and binding unchanged.
The returned `TemporalObservation.delta` is the raw EP derivative before
projection; inspect committed parameter differences to measure the applied step.
This transaction is single-threaded and accepts the base `TemporalPatchNet`;
custom subclasses need their own complete state transaction.

A parameter fingerprint detects changes made outside the admitted protected
update path. Resetting active state is allowed because the binding concerns
learned parameters. Restoring only the network or only the memory can produce a
binding mismatch. Save and restore both:

```python
net_state, memory_state = net.snapshot(), memory.snapshot()
np.savez("protected_brain.npz",
         **{"net_" + k: v for k, v in net_state.items()},
         **{"memory_" + k: v for k, v in memory_state.items()})
with np.load("protected_brain.npz", allow_pickle=False) as archive:
    restored = TemporalPatchNet.restore(
        {k[4:]: archive[k] for k in archive.files if k.startswith("net_")}
    )
    recalled = TemporalMemory.restore(
        {k[7:]: archive[k] for k in archive.files if k.startswith("memory_")}
    )
restored.reset()
assert recalled.observe(restored, new_input, np.full((1, 1, 1), 0.25)).updated
```

`report().ranks` states the consumed dimensions per parameter block.
`report().bytes` counts basis arrays, the eight-byte tolerance and the 32-byte
binding when present; Python objects, serialization and transient SVD/solver
workspace are extra. `maximum_residual` from `protect` measures how well its
new vectors fit the retained bases. A plain `report()` does not re-evaluate
protected paths and its default residual is not a retention certificate.

The lower-level `project(before, proposed)` remains available for explicit
transactions. It advances the memory binding before the caller applies the
result, so normal training should use `memory.observe` to avoid a partially
committed pair. No raw examples are replayed by either operation.
