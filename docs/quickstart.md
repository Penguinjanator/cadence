# Quickstart

Install Cadence using the [README](../README.md#install). The examples below need
only NumPy and Cadence. Each main example includes its imports and data, so it can
run in a Python session or a saved `.py` file.

## Settle a small circuit

Owner 0 receives input, owner 1 relays it to owner 2, and owners 1 and 2 feed back
to each other. An *owner* holds a potential and publishes an activation. An edge
carries one owner's activation into another owner's inbox.

```python
import numpy as np
import cadence as cd

wiring = cd.Wiring.from_edges(
    3,
    pre=[0, 1, 2],
    post=[1, 2, 1],
    sign=[1.0, 0.4, 0.4],
    sets={"input": [0], "output": [2]},
)
engine = cd.Settlement(wiring, cd.learning_rule(dt=1.0))
drive = np.array([[1.0, 0.0, 0.0]])  # one row, one drive value per owner
state = engine.settle_batch(drive, steps=100, tolerance=1e-10)

print(state.activation.round(4))
print(engine.residual(drive, state).max() < 1e-9)
```

Expected output:

```text
[[0.4621 0.236  0.0472]]
True
```

The wiring and weights are supplied; no training occurs. `learning_rule` gives a
responsive activation around zero and also works for inference. The engine starts
from rest and repeatedly updates each owner from its state, inbox, and external
drive. `steps` caps the work. `tolerance` stops on a small activation change;
`residual` separately checks the fixed-point equations. A small residual alone
does not prove a unique or stable equilibrium.

### Change the circuit

Continue in the same session. A binary mask removes owner 1's activity:

```python
mask = np.array([1.0, 0.0, 1.0])
cut = engine.settle_batch(drive, mask=mask, steps=100, tolerance=1e-10)
print(cut.activation.round(4))
assert engine.residual(drive, cut, mask=mask).max() < 1e-9
```

The output is `[[0.4621 0.     0.    ]]`: owner 2 receives no signal through
the cut relay. Always pass the same mask to settlement and its residual check.

### Carry state and check the implementation

```python
changed_drive = np.array([[0.9, 0.0, 0.0]])
continued = engine.settle_batch(changed_drive, state=state, steps=100, tolerance=1e-10)
assert engine.residual(changed_drive, continued).max() < 1e-9

check = cd.conformance(engine, drive[0], steps=30)
print(check["ledger"]["clean"], check["max_abs_deviation"] < 1e-12)
```

The final line prints `True True`. `conformance` compares this trajectory against
an owner-by-owner reference; it does not verify an entire learning algorithm.
Starting from an earlier state may save steps for small changes. Multiple
attractors can make the answer depend on that state, so compare cold and warm
starts. Omit `state=` to start an independent episode from rest.

## Learn a response

This separate example fits two labelled observations. It is a check of the
training API, not a held-out accuracy benchmark.

```python
import numpy as np
import cadence as cd

wiring = cd.layered(2, 8, 2, density=1.0, seed=0)
learner = cd.Learner(
    cd.Settlement(wiring, cd.learning_rule(dt=1.0)),
    wiring.sets["output"],
    cd.LearnerConfig(eta=2.0, eta_bias=0.02),
)
drive = np.zeros((2, wiring.n))
drive[:, list(wiring.sets["input"])] = np.eye(2)
labels = np.array([0, 1])  # class indices within the output group

for _ in range(50):
    phases, report = learner.step(drive, labels)

print(learner.predict(drive))
```

Expected output: `[0 1]`. Each `step` runs a free phase, two opposite nudged
phases, and a parameter update. `phases.free` contains the answer before that
update; `predict` runs a fresh free phase using the learned parameters.
For evaluation, train on one split, choose settings on validation data, and read
the test split after selection. [Digits](https://github.com/muellerberndt/cadence-examples/tree/7302f2af3dc0638bbafd1da1446ed96ffabaa9dd/01_digits)
shows this workflow with an MLP control and a receipt.

Save and reload this learner:

```python
learner.save("tiny_learner.npz")
restored = cd.Learner.load("tiny_learner.npz", backend="cpu")
assert np.array_equal(restored.predict(drive), learner.predict(drive))
```

The checkpoint includes learned parameters and optimizer history. Separate
`FastSeams` records and `Trace` state belong to the caller and are not included.
Use [memory](memory.md) when the task is storing observations, rather than fitting
a reusable input/output response. See [learning](learning.md) for the gradient
assumptions and parameter choices.

## Shapes and common mistakes

| Symptom or question | What to check |
|---|---|
| `ModuleNotFoundError: cadence` | Activate the environment used for installation. Run `python -m pip show cadence-net` with that same Python. |
| Missing `observe` or `recall` | Install the GitHub source shown in the README. Inspect `cadence.__file__` for an older install or a local file named `cadence.py`. |
| Drive shape error | Use `(batch, wiring.n)`, including zero columns for hidden and output owners. `clamp_levels` scales values; it does not pad missing columns. |
| Invalid classification labels | Use integer class indices `0` through `len(outputs)-1`, not owner indices or one-hot rows. Use `nudged` for explicit target patterns. |
| Confusing input amplitudes | `settle({owner: level})` scales levels by `rule.clamp_amplitude`. A dense NumPy vector and `settle_batch(drive)` carry drive values directly. |
| Small step count, poor answer | Inspect the residual and activations. Saturation can stop movement before the potential equilibrates. More steps cannot correct a wrong supplied model. |
| Rule rejects a resting sigmoid rounded to zero or one | Reduce the magnitude of `slope * threshold`; the rebased activation needs a representable resting value strictly between zero and one. |
| State batch mismatch | Continuing state must have the same batch size and row identities. Start independent rows from rest. |
| Slow first call with `[fast]` installed | Numba compiles the CPU kernel on first use. Record setup separately from warmed execution when timing. |

For held-out intervention predicates, see [protocols](protocols.md). To bind results
to source files and verify stored arithmetic, see [receipts](receipts.md).
