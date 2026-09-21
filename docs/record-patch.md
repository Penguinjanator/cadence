# RecordPatchNet

`RecordPatchNet` is a temporal patch whose context is linear and gated and
whose memory of particular readings is a record store inside the patch. The
nonlinearity sits at the ports. It is a development addition on `main` after
0.11.0; pin a commit when reproducing an experiment.

For inputs `u[t]`, context `h[t]` and outputs `y[t]`:

```text
l[t] = sigmoid(g + G u[t])               per-channel retention in (0, 1)
z[t] = tanh(B u[t] + b)                  the input port
h[t] = l[t] * h[t-1] + (1 - l[t]) * z[t]
m[t] = read(code([u[t], r * h[t]]))      the record read, a port value
y[t] = C h[t] + c + m[t]
```

The context is a convex combination, so activity stays bounded. The seam
residual `h[t] - l[t] h[t-1] - (1 - l[t]) z[t]` is linear in the context path,
which makes the energy quadratic: the free path is its unique zero-defect
normal form, and the two detuned equilibria of a teaching loss are unique.
Channels start at retention timescales log-spaced from two moments to
`slowest` moments; `r` reads each channel in the unit of its own fluctuations.

## One observed path

```python
import numpy as np
from cadence import RecordPatchNet

net = RecordPatchNet(inputs=17, hidden=8, outputs=6, seed=2, cells=4096, active=32)
rng = np.random.default_rng(7)
inputs = np.zeros((1, 16, 17))
inputs[0, 0, 0] = 1.0                            # one cue
inputs[0, np.arange(16), 1 + np.arange(16)] = 1.0  # sixteen distinct heard events
identities = rng.integers(6, size=16)
target = np.eye(6)[identities][None]

first = net.observe(inputs, target, rate=1.0, backtrack=True)
assert first.updated and first.writes == 16
recalled = net.imagine(inputs, state=np.zeros((1, 8)))
assert np.array_equal(recalled.output[0].argmax(axis=1), identities)
```

`observe` predicts the path with the records as they stood when the call
began, moves the slow parameters against the adjoint gradient of the
precision-weighted half mean squared error, and then writes what the slow
readout got wrong at each reading into that reading's records. With
`backtrack=True` a parameter step is admitted only after a target-free
replay from the original boundary lowers the loss, trying up to sixteen
halved rates. `write=False` learns without writing. The final free context
becomes the live state; `advance` carries context without learning or
writing, `imagine` is private, and `reset` clears context while keeping
parameters and records.

## Records hold what the slow model does not know

Each write target is `y_obs[t] - C h[t] - c`. A reading the slow parameters
have learned leaves a record near zero, so the store holds only the
residual the slow model has not absorbed. One write corrects its own
reading; readings whose codes overlap move by their overlap, as in the
record theorems. Keys of neighbouring moments share the slowly varying
context, so one write is not exact and the delta rule converges over
repeated observation. Readings that differ only by the bar they belong to,
under a periodic clock, are separated only partly by a leaky context; a
heard stream, in which the previous event is an input, supplies the rest.
Whether carried context alone can return a theme after many bars is an
open measurement, not a property of this class.

Records are read with the tables at the start of the call and written after
the path, so a prediction is a function of the parameters, the records and
the boundary alone, and a private branch reproduces it exactly. To let a
record written at one moment inform the next, observe in shorter paths.

## Detuning as the acceptance check

```python
net.reset()
adjoint = net.observe(inputs, target, rate=0.0, write=False).delta
net.reset()
contrast = net.detune(inputs, target, beta=1e-4)
assert contrast.converged
for name, value in adjoint.items():
    assert np.linalg.norm(contrast.contrast[name] - value) < 1e-6 * np.linalg.norm(value)
```

`detune` solves the two detuned equilibria of the quadratic energy by
conjugate gradients from the free path and returns the centered contrast of
the energy's parameter derivatives. For this patch that contrast equals the
adjoint gradient up to terms of order `beta^2`; `observe` computes the same
learning signal by one backward scan. Neither route differentiates through
the record code: the read is a port value.

## Cost and state

One update is one forward scan, one backward scan, one record read per
moment and one write per observed moment. Work per moment is linear in the
context width plus the record projection; state is the context, the
parameters and the record tables, constant in stream length. The dense
hidden-width block messages of `TemporalPatchNet` do not appear. The receipt
`cadence-mission/results/record_patch_g1.json` measures one update against
the dense kernel at the shapes of the scalable brain plan.

## Checkpoints

`snapshot`, `restore`, `save` and `load` carry the parameters, the record
tables, the running mean and counts, the output precision and the live
context. `readback` exposes the detached state, update and write counts, the
parameter revision the state was computed under and the record entry count.

## What this class does not do

Planning through action ports, protected responses through `TemporalMemory`
and coupled components are not yet available for this class. The record
rate, the number of active cells and the reading scale are supplied. There
is no automatic importance, forgetting, or learned relevance: a record is
written for every observed moment and moves only when its cells are read
again.
