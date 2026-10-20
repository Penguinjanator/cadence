# RecordPatchNet

`RecordPatchNet` learns observed streams (heard events, words, notes) with a gated linear
context and a store of one-shot records inside the patch. The nonlinearity sits at the
ports. For inputs `u[t]`, context `h[t]` and outputs `y[t]`:

```text
l[t] = sigmoid(g + G u[t])               per-channel retention in (0, 1)
z[t] = tanh(B u[t] + b)                  the input port
h[t] = l[t] * h[t-1] + (1 - l[t]) * z[t]
m[t] = read(code([u[t] * sqrt(n) / s, r * h[t]]))   the record read, a port value
y[t] = C h[t] + c + m[t]
```

The context is a convex combination, so it stays bounded; the seam residual
`h[t] - l[t] h[t-1] - (1 - l[t]) z[t]` is linear in the context path, so the energy with the
linear readout is quadratic and the free path is its unique zero-defect solution. Channels
start at retention timescales log-spaced from two moments to `slowest` (default 128, an
initial scale: the gates are learned); `r` is each channel's initial fluctuation unit.

## One observed path

```python
import numpy as np
from cadence import RecordPatchNet

net = RecordPatchNet(inputs=17, hidden=8, outputs=6, seed=2, cells=4096, active=32)
rng = np.random.default_rng(7)
inputs = np.zeros((1, 16, 17))
inputs[0, 0, 0] = 1.0                              # one cue
inputs[0, np.arange(16), 1 + np.arange(16)] = 1.0  # sixteen distinct heard events
identities = rng.integers(6, size=16)
target = np.eye(6)[identities][None]

first = net.observe(inputs, target, rate=1.0, backtrack=True)
assert first.updated and first.writes == 16
recalled = net.imagine(inputs, state=np.zeros((1, 8)))
assert np.array_equal(recalled.output[0].argmax(axis=1), identities)
```

`observe(inputs, target, rate=1.0, backtrack=False, write=True)` predicts with the records
as they stood when the call began, moves the slow parameters against the adjoint gradient of
the precision-weighted half mean squared error of the slow readout `C h + c`, then writes
that readout's error at each reading into the reading's records. With `backtrack=True` a
step is admitted only after a target-free replay lowers the slow loss, trying up to sixteen
halved rates. `write=False` learns without writing; `rate=0` writes without a step. A
failed record write restores the records and leaves the live context unchanged.

The final context becomes the live state. `advance(inputs)` carries it without learning,
`imagine(inputs, state=None)` changes nothing, and `reset()` clears it. `parameters()` and
`set_parameters(...)` cover `G, g, B, b, C, c`; `seed` draws them, the record projection and
offsets, and `R` below. `output_precision` (default ones) and `set_output_precision` hold
the teaching weights; `records` is the [`Records`](records.md) store.

| Result | Fields |
|---|---|
| `RecordObservation` | `updated`, `reason` (`updated`, `no_step`, `nonfinite_prediction`, `nonfinite_update`, `no_decreasing_parameter_step`), `prediction`, `delta`, `initial_loss`, `final_loss`, `accepted_rate`, `replay_losses`, `replay_calls`, `writes` |
| `RecordPath` | `hidden`, `output` (slow readout plus read), `gate`, `read`, `loss`, `slow_loss`; `slow_output`, `final_state` |
| `RecordReadback` from `readback()` | `state`, `updates`, `writes`, `parameter_revision`, `state_parameter_revision`, `record_entries` |

## The reading: two blocks of unit variance per unit

The record's key is the input block and the context block side by side. The context is
read in units of each channel's fluctuation (`r`); the input block is scaled by
`sqrt(n) / s`, with `n` the number of inputs and `s` a running average (rate 0.01) of
witnessed input norms. Both blocks then have unit variance per unit, a cell's drive has
unit scale, and the fixed offset (`record_bias`, default 0.3) is a small preference. The
store subtracts the running mean of witnessed readings, which settles to rate
`habituation` (default `1e-5`, 0 subtracts nothing); imagined readings never move it. The
expansion into `cells` (default 4096) with `active` winners (default 32) is a fixed random
sparse code ([records](records.md#records)).

## Records hold what the slow model does not know

Each write target is `y_obs[t] - C h[t] - c`. A reading the slow parameters have learned
leaves a record near zero, so the store holds only the residual the slow model has not
absorbed. One write at `record_rate=1` corrects its own reading; readings whose codes
overlap move by their overlap. Neighbouring moments share slowly varying context, so their keys overlap and the
delta rule converges over repeated observation. Records are read with the tables at the
start of the call and written after the path, so a prediction depends on the parameters,
the records and the boundary alone, and a private branch reproduces it exactly. To let a
record written at one moment inform the next, observe in shorter paths.

## Diagnosing a record store

A store can fail silently: prediction with records beats prediction without them because
the store always holds something. Replay a training stream's readings through
`net.records.code(..., valued=False)`, count each cell's activations, and check:

| Check | Healthy | Failure and fix |
|---|---|---|
| Cells in use; share of activations in the 200 most active | share near `200 / cells` | A quarter of the cells in use, most activations in the top 200: a block of larger scale owns the address. Give every unit of the reading unit variance and recount after any change. A reading normalised to unit norm lets the offsets decide, and the same cells win every time. |
| Code overlap when only one block changes | moves with the input; keeps most cells when only distant context moves | An address that does not move with a block stores nothing about it. |
| Held-out score with records frozen, zeroed, writing online, writing online from empty | frozen close to online | Frozen far below online: the store forgets its corpus ([write rate](#what-a-write-rate-stores)). Online equal to from-empty: trained records add nothing. |
| The read at readings no stream wrote (a rollout from silence) | near zero | A large read is interference from unrelated writers. |
| First event of a rollout from silence | the convention's event | A rare reading is poorly learned ([boundary readings](#boundary-readings)). |
| Overlap between moments that should recall each other | most cells shared | Low after a phase shift ([recall by content](#recall-is-by-content-not-by-position)). |

## What a write rate stores

`record_rate` (default 0.5, at most 2) is the delta-rule step. Near one half a cell holds
its last few writers: the read calibrates fast to a new stream, and frozen records recall
little of a training corpus. `record_averaging=True` (default off) sets a cell's rate to
one over its written mass, floored at `record_rate`, so a familiar cell averages its
writers: retention improves and adaptation to a new stream slows in proportion.
`record_homeostasis` (a rate in [0, 1], default 0) moves each cell's offset toward an equal
activation share, which spreads the code and also slows adaptation. Choose by which the
task reads: retention or adaptation.

For a store written once per reading, use enough active cells that neighbouring readings
overlap, `record_rate=1.0`, and a reading in which token, cue and context are in balance;
a lower rate leaves each record holding part of its residual.

To recite a sequence exactly from the store, give it more cells than associations to hold,
about seven cells per association, and tens of presentations, with slow weights that have
learned most of the material first; neighbouring moments share key cells. Closed-loop
recitation multiplies the per-word rate over the sentence, and teacher-forced word accuracy
is reached long before exact closed-loop recitation, so measure both.

<a id="recall-is-by-content-not-by-position"></a>

## Recall is by content

A record is keyed by what was heard and the context. Where a stream recurs, the key recurs
and the codes one loop apart share most of their cells. When the stream drifts in phase,
so that the same place in a bar hears another event, the codes barely overlap and nothing
written there is read back. For recall by position, put position in the reading as an input
of its own with enough ports to move the code; eight clock ports among eighty inputs do not.

## Boundary readings

A reading met once per stream, such as waking with nothing heard, gets a handful of
updates per epoch and shares its cells with common readings, so the first prediction of a
rollout from silence is poorly determined. Make the boundary common by convention: let the
wake hear a count-in, the last event of the loop, so the best-known rule gives the first event.

## Categorical ports

With `groups` the slow readout ends in one softmax per group of ports and learns by
cross-entropy (output error `softmax - target`). Use it for symbols: words, moves, notes.

```python
words = RecordPatchNet(inputs=17, hidden=8, outputs=6, seed=2, cells=4096, active=32,
                       record_rate=1.0, groups=(6,))
taught = words.observe(inputs, target, rate=0.0)
assert np.allclose(taught.prediction.slow_output.sum(axis=-1), 1.0)
recalled = words.imagine(inputs, state=np.zeros((1, 8)))
assert np.array_equal(recalled.output[0].argmax(axis=1), identities)
```

A record holds `onehot - softmax(C h + c)`. The prediction `softmax + read` can have
negative entries: its largest port classifies, and sampling or log loss needs a declared
normalisation. The loss clips entries at `1e-12` and renormalises each group, so a read
that pushes the true category below zero costs a very large loss; report mean held-out log
loss beside accuracy. A group's teaching weight is the mean `output_precision` of its ports.

## Writing a batch

`record_writes="batch"` writes all moments of a call at once through `Records.write_batch`:
every error is taken against the tables as they stood, and each cell moves by the mean of
its writers' moves. One writer reproduces a sequential write; agreeing writers move a shared
cell as far as one would. With the default `"sequential"` a later moment overwrites an
earlier one that shares its cells, so a repeated path needs more presentations.

## Do not record a choice

When a reading has several right outcomes (a message said in four ways), the slow readout
learns their distribution and its error there is whichever choice was taught. A record of
it pulls later predictions toward that choice and spends cells. Write records for what must
be reproduced (a sentence read, a phrase heard) and teach choice points with `write=False`.

## A store narrower than its port

With `record_width=w` the cells hold a fixed random sign code of the residual,
`residual @ R` with `R` of shape `(outputs, w)`, entries `+-1/sqrt(w)` drawn from `seed`,
and the read is decoded by `held @ R.T`. Crosstalk has standard deviation
`|residual| / sqrt(w)` per port, which the largest port survives, so a port of thousands of
categories can use a few hundred columns. The delta rule compares the coded residual with
what the cells hold; re-projecting a decoded error would multiply the step by `outputs / w`.

## Acquisition in two phases: records by day, weights by night

By day, `observe(..., rate=0)` writes each observation once and leaves the slow weights
alone: a fact is usable in the next sentence and a corpus is taken in one pass, but the
store generalises only by overlap, and new material written into a store that holds a
library degrades the library a little. Give the library a store of its own.

By night, `sleep(cues, passes=1, rate=1.0, backtrack=False, dawn_passes=2)` moves what the
store holds into the slow weights. Each cue is dreamed once with the store as it stands at
bedtime; the weights learn those fixed dreams for `passes` passes of
`observe(..., write=False)`; at dawn each dream is written back `dawn_passes` times, so the
store keeps only what the weights did not take. `sleep` resets the live context and returns
`cues`, `updates`, `dream_loss_before`, `dream_loss_after` and `dawn_writes`.
`dream(inputs)` is one dream: the free path from rest with the records, as a target (the
chosen category per group for categorical ports). It changes nothing.

```python
night = words.sleep([inputs], passes=50, rate=8.0, backtrack=True)
assert night["dawn_writes"] == 32 and night["dream_loss_after"] < night["dream_loss_before"]
slow = words.imagine(inputs, state=np.zeros((1, 8))).slow_output
assert np.array_equal(slow[0].argmax(axis=1), identities)   # the weights alone recall
```

Dreams are fixed at bedtime because a record is a residual against the readout at write
time; re-dreaming would teach the weights their own drift. Sleep restructures and corrects
nothing: dreams carry the store's errors, and dreams of recombined cues teach its guesses as
facts. A teacher with the corpus beats a night; sleep pays when the store is the only copy.

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

`detune(inputs, target, beta=1e-3, state=None, tolerance=1e-14, max_iterations=10_000)`
certifies each reduced quadratic of the slow patch's energy positive definite, solves the
two detuned equilibria by conjugate gradients from the free path (from boundary `state`,
default the live context, to relative residual `tolerance`, at most `max_iterations` steps
each) and returns the centered contrast of the energy's parameter derivatives over
`2 beta`. It equals the adjoint gradient of `observe` up to order `beta^2`; the record read
lies outside the energy, and the patch does not change. `RecordContrast` holds `contrast`,
`plus_hidden`, `minus_hidden`, `plus_energy`, `minus_energy`, `iterations`, `residuals` and
`converged`. It refuses `groups` and `beta * output_precision / (time * outputs) >= 1`.

## Cost, checkpoints and running outside Python

<a id="cost-and-state"></a>
<a id="checkpoints"></a>
<a id="running-a-trained-patch-outside-python"></a>

An update is one forward scan, one backward scan, one record read per moment and one write
per observed moment; work per moment is linear in the context width plus the projection
(reading width times `cells`), and state is constant in stream length.

`snapshot()`, `restore(snapshot)`, `save(path, compressed=True)` and `load(path)` carry the
parameters, record tables and counts, running mean, input norm, output precision, live
context and revisions; the projection is regenerated from `seed`. The format is
`cadence-record-patch/2`, or `/3` with `groups`, batch writes, `record_width` or a `port`.
An application that branches imagination owns its generator, action adapter and
normalisation, and keeps imagined targets apart from observations.

The forward pass needs no library. Export the six parameter arrays, the table, the running
mean, the input norm, the homeostatic offsets if `record_homeostasis` is set and `R` if
`record_width` is; `r` follows from `hidden` and `slowest`. Projection and offsets come
from the seed through `Mulberry32` (in `cadence.records`): `normals(n)` is Box-Muller over `2 * ceil(n / 2)` draws, radii from the
first half and angles from the second, cosines before sines; the projection is
`reading * cells` normals over `sqrt(reading)`, row by row, and the offsets the next `cells`
normals times `record_bias`. A moment is the context update, the reading minus the mean,
the `active` largest drives with positive parts normalised to unit length, the weighted
table rows and the readout: 1.7 million multiply-adds at 208 reading units and 8,192 cells.
Ship the table as float32 and keep a Python rollout as the parity test.

## Maps: a structured input port

`StructuredPort(inputs, blocks, broadcast=None, mask=None)` (in `cadence.ports`) replaces
the dense `B u` and `G u` with a declared layout. `MapBlock(start, channels_in, height,
width, channels_out, kernel, stride=1)` is a tied local kernel over a channel-major grid of
the inputs, giving retinotopic maps; `DenseBlock(start, inputs, outputs)` is a matrix over
a slice. The port supplies apply, transpose and parameter gradient, so learning is the same
rule; `hidden` must equal `port.outputs`. Dense blocks covering the inputs give the plain
patch, and the checkpoint carries the layout and the mask.

```python
from cadence.ports import DenseBlock, MapBlock, StructuredPort

fovea = MapBlock(start=0, channels_in=2, height=12, width=12, channels_out=4, kernel=3, stride=2)
rest = DenseBlock(start=fovea.inputs, inputs=8, outputs=16)
port = StructuredPort(fovea.inputs + 8, [fovea, rest])
eye = RecordPatchNet(port.inputs, port.outputs, 4, port=port, cells=512, active=8)
seen = eye.observe(rng.normal(size=(1, 5, port.inputs)), np.zeros((1, 5, 4)))
assert seen.writes == 5 and port.outputs == 4 * 5 * 5 + 16
```

`broadcast=(start, count)` is a slice every map block also reads as constant channels tiled
over its grid (its kernel gains `count` input channels), so what a body did reaches every
position. `mask` is one flag per input; a masked input is zero to every block in the forward
map, transpose and gradient, so which inputs a patch hears is a gene apart from the weights.

## Two patches in depth

The gate of one patch sees only the present input: it learns a conjunction of the last two
symbols and cannot tell a subject from a later noun that arrives as the same input.
`RecordPatchStack` puts a context patch below a complete `RecordPatchNet` that reads
`[u, r1 * h1]`, so the upper gate sees what came before.

```python
from cadence import RecordPatchStack

stack = RecordPatchStack(inputs=5, hidden=8, outputs=4, seed=5, cells=2048, active=16,
                         groups=(4,), record_rate=1.0)
symbols = np.random.default_rng(4).integers(5, size=(3, 8))
pairs = np.eye(4)[(symbols + np.roll(symbols, 1, axis=1)) % 4]
stacked = stack.observe(np.eye(5)[symbols], pairs, rate=2.0, backtrack=True)
assert stacked.updated and stacked.writes == 24
```

`RecordPatchStack(inputs, hidden, outputs, lower=None, seed=0, slowest=128.0, groups=None,
lower_port=None, **upper)`: `lower` is the lower width (default `hidden`), `lower_port` a
`StructuredPort` below, and `**upper` goes to `stack.upper`, which owns readout and records.
The upper adjoint hands its input gradient to the lower scan. `observe`,
`imagine(state=(lower, upper))`, `advance`, `reset`, `parameters`, `set_parameters`,
`snapshot` and `restore` (formats `cadence-record-stack/1`, `/2` with a lower port) follow
the single patch; `StackObservation` holds `updated`, `reason`, `prediction`, `delta`,
`initial_loss`, `final_loss`, `accepted_rate` and `writes`. The stack has no `detune`.

## Several patches joined by ports

`Port(source, target, start, width)` declares a seam: the band `[start, start + width)` of
the source's scaled context `r * h` is an input of the target in the same moment, read by
its gate and input port and keyed by its records.
`JointRecordPatches(cortices, own_inputs, ports, rounds=1, damping=1.0,
cross_adjoint=True)` iterates the coupled equations for `rounds` Jacobi rounds per moment
from the previous moment's contexts: round one is the delayed port, each further round
re-reads this moment's contexts. With `damping` below one a round moves each context that
fraction of the way to its new value. The joint energy adds `1/2 |p[t] - S r h_source[t]|^2`
per port.

```python
from cadence import JointRecordPatches, Port
from cadence.record_ports import build

see = RecordPatchNet(inputs=5 + 3, hidden=8, outputs=4, seed=1, cells=256, active=8, groups=(4,))
hear = RecordPatchNet(inputs=5 + 2, hidden=6, outputs=4, seed=2, cells=256, active=8, groups=(4,))
seams = [Port(1, 0, 0, 3), Port(0, 1, 2, 2)]
joined = JointRecordPatches([see, hear], own_inputs=[5, 5], ports=seams, rounds=2)
xs = [np.eye(5)[rng.integers(5, size=(3, 8))] for _ in range(2)]
ys = [np.eye(4)[rng.integers(4, size=(3, 8))] for _ in range(2)]
both = joined.observe(xs, ys, rate=2.0, backtrack=True)
assert both.settled.seam.shape == (3, 8, 2, 2)      # batch, time, rounds, ports
same = build([8, 6], [5, 5], [4, 4], seams, seed=1, rounds=2, groups=[(4,), (4,)], cells=256, active=8)
```

`cadence.record_ports.build(hidden, own_inputs, outputs, ports, *, seed, rounds=1,
damping=1.0, cross_adjoint=True, groups=None, **records)` sizes each patch's inputs from its
incoming ports, seeds patch `i` with `seed + 101 * i` and passes `**records` to every
`RecordPatchNet`. `JointRecordPatches` refuses cortices with a structured port.

For one one-way port from an independent source, two undamped rounds reach the fixed point;
longer chains need more rounds, and recurrent ports have no convergence guarantee. Inspect
`settled.seam` (port residual per moment and round) and `settled.settle` (context change per
round) before claiming a joint equilibrium. With `cross_adjoint=True` a target's loss
gradient reaches the source's context through the port; with `False` the port is a plain
input. With `backtrack=True` a step is admitted by backtracking on the joint slow loss.
Records are written after the path, transactionally across patches. There is no joint
`detune`.

`JointObservation` holds `updated`, `reason`, `settled`, `delta`, `initial_loss`,
`final_loss`, `accepted_rate`, `replay_calls` and `writes`. `imagine(xs, states=None)`,
`advance`, `reset`, `state`, `parameter_count()`, `record_entries()`, `snapshot`, `restore`,
`clone`, `save` and `load` (format `cadence.record-ports/1`) follow the single patch.
`cut = True` makes every port carry zeros: measure a port by cutting it. Source, target,
band and width are genes for `evolve(..., mutate=genes(space))` ([any
genome](evolution.md#any-genome)); channels are ordered by timescale, so the band picks the
source's current input (fast) or carried state (slow).

## Limits

The record rate, the number of active cells and the reading scale are supplied. There is
no automatic importance, forgetting or learned relevance: every observed moment is written,
and a record moves only when its cells are read again. This class has no planning through
action ports, and [`TemporalMemory`](temporal-memory.md) protects a `TemporalPatchNet` only.
