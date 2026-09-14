# Quickstart

Install Cadence using the [README](../README.md#install). The examples below need
only NumPy and Cadence. Each main example includes its imports and data, so it can
run in a Python session or a saved `.py` file.

## Settle a small circuit

Neuron 0 receives input, neuron 1 relays it to neuron 2, and neurons 1 and 2 feed
back to each other. A *neuron* holds a potential and publishes an activation. A
*synapse* carries one neuron's activation into another neuron's synaptic input.

```python
import numpy as np
import cadence as cd

connectome = cd.Connectome.from_synapses(
    3,
    pre=[0, 1, 2],
    post=[1, 2, 1],
    sign=[1.0, 0.4, 0.4],
    populations={"input": [0], "output": [2]},
)
brain = cd.Brain(connectome, cd.learning_neuron_model(dt=1.0))
drive = np.array([[1.0, 0.0, 0.0]])  # one row, one drive value per neuron
state = brain.settle_batch(drive, steps=100, tolerance=1e-10)

print(state.activation.round(4))
print(brain.residual(drive, state).max() < 1e-9)
```

Expected output:

```text
[[0.4621 0.236  0.0472]]
True
```

The connectome and weights are supplied; no training occurs.
`learning_neuron_model` gives a responsive activation around zero and also works for
inference. The brain starts from rest and repeatedly updates each neuron from its
state, synaptic input, and external drive. `steps` caps the work. `tolerance` stops on a small activation change;
`residual` separately checks the fixed-point equations. A small residual alone
does not prove a unique or stable equilibrium.

### Change the circuit

Continue in the same session. A binary mask removes neuron 1's activity:

```python
mask = np.array([1.0, 0.0, 1.0])
cut = brain.settle_batch(drive, mask=mask, steps=100, tolerance=1e-10)
print(cut.activation.round(4))
assert brain.residual(drive, cut, mask=mask).max() < 1e-9
```

The output is `[[0.4621 0.     0.    ]]`: neuron 2 receives no signal through
the cut relay. Always pass the same mask to settling and to its residual check.

### Carry state and check the implementation

```python
changed_drive = np.array([[0.9, 0.0, 0.0]])
continued = brain.settle_batch(changed_drive, state=state, steps=100, tolerance=1e-10)
assert brain.residual(changed_drive, continued).max() < 1e-9

check = cd.conformance(brain, drive[0], steps=30)
print(check["ledger"]["clean"], check["max_abs_deviation"] < 1e-12)
```

The final line prints `True True`. `conformance` compares this trajectory against
a neuron-by-neuron reference; it does not verify an entire learning algorithm.
Starting from an earlier state may save steps for small changes. Multiple
attractors can make the answer depend on that state, so compare cold and warm
starts. Omit `state=` to start an independent episode from rest.

## Learn a response

This separate example fits two labelled observations. It is a check of the
training API, not a held-out accuracy benchmark.

```python
import numpy as np
import cadence as cd

connectome = cd.layered(2, 8, 2, density=1.0, seed=0)
learner = cd.Learner(
    cd.Brain(connectome, cd.learning_neuron_model(dt=1.0)),
    connectome.populations["output"],
    cd.LearnerConfig(eta=2.0, eta_bias=0.02),
)
drive = np.zeros((2, connectome.n))
drive[:, list(connectome.populations["input"])] = np.eye(2)
labels = np.array([0, 1])  # class indices within the output group

for _ in range(50):
    phases, report = learner.step(drive, labels)

print(learner.predict(drive))
```

Expected output: `[0 1]`. Each `step` runs a free phase, two opposite nudged
phases, and a parameter update. `phases.free` contains the answer before that
update; `predict` runs a fresh free phase using the learned parameters.
For evaluation, train on one split, choose settings on validation data, and read
the test split after selection. The [design guide](design.md) covers that workflow,
component ablations and comparison with conventional models.

For a solve checked by its equations, use `learner.brain.equilibrate(drive)` and inspect
its per-row `converged` and `residual`. Its `budget` caps work exactly; use it when the
claim is a common equilibrium rather than a fixed-duration response.

Save and reload this learner:

```python
learner.save("tiny_learner.npz")
restored = cd.Learner.load("tiny_learner.npz", backend="cpu")
assert np.array_equal(restored.predict(drive), learner.predict(drive))
```

The checkpoint includes learned parameters and optimizer history. Separate
`FastSynapses` records and `Trace` state belong to the caller and are not included.
`GenericBrain.save/load` preserves those components when using the standard composition.
Use [memory](memory.md) when the task is storing observations, rather than fitting
a reusable input/output response. See [learning](learning.md) for the gradient
assumptions and parameter choices.

## Shapes and common mistakes

| Symptom or question | What to check |
|---|---|
| `ModuleNotFoundError: cadence` | Activate the environment used for installation. Run `python -m pip show cadence-net` with that same Python. |
| Missing `observe` or `recall` | Install the GitHub source shown in the README. Inspect `cadence.__file__` for an older install or a local file named `cadence.py`. |
| Drive shape error | Use `(batch, connectome.n)`, including zero columns for hidden and output neurons. `Brain.stimulus_levels` scales values; it does not pad missing columns. |
| Invalid classification labels | Use integer class indices `0` through `len(outputs)-1`, not neuron indices or one-hot rows. Use `nudged` for explicit target patterns. |
| Confusing input amplitudes | `settle({neuron: level})` scales levels by `brain.neuron_model.stimulus_amplitude`. A dense NumPy vector and `settle_batch(drive)` carry drive values directly. |
| Small step count, poor answer | Inspect the residual and activations. Saturation can stop movement before the potential equilibrates. More steps cannot correct a wrong supplied model. |
| `NeuronModel` rejects a resting sigmoid rounded to zero or one | Reduce the magnitude of `slope * threshold`; the rebased activation needs a representable resting value strictly between zero and one. |
| State batch mismatch | Continuing state must have the same batch size and row identities. Start independent rows from rest. |
| Slow first call with `[fast]` installed | Numba compiles the CPU kernel on first use. Record setup separately from warmed execution when timing. |

For held-out intervention predicates, see [protocols](protocols.md). To bind results
to source files and verify stored arithmetic, see [receipts](receipts.md).
