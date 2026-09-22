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
m[t] = read(code([u[t] * sqrt(n) / s, r * h[t]]))   the record read, a port value
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
precision-weighted half mean squared error of the slow readout `C h + c`,
and then writes what that readout got wrong at each reading into the
reading's records. The record read never enters the slow gradient: the slow
parameters learn the observation itself, and the record patches their
current error until they have. With `backtrack=True` a parameter step is
admitted only after a target-free replay from the original boundary lowers
the slow readout's loss, trying up to sixteen halved rates. `write=False`
learns without writing. The final free context becomes the live state;
`advance` carries context without learning or writing, `imagine` is
private, and `reset` clears context while keeping parameters and records.

## The reading: two blocks of unit variance per unit

The record's key is the input block and the context block side by side. The
context is read in units of each channel's fluctuation (`r`), which gives its
units unit variance. The input block is scaled by `sqrt(n) / s`, where `n` is
the number of input ports and `s` the running rms norm of witnessed inputs,
so its units have unit variance too and a cell's drive has unit scale, with
the fixed offset a small preference. Before this scaling the context block
outweighed the input block about five to one on the composer stream: the
address hardly moved with the event heard (48-cell code overlap 0.66 when
only the crop changed), a few hundred hub cells took three quarters of all
activations, and the store forgot the training corpus behind its last
writers. With the scaling the same stream uses 5,812 of 8,192 cells instead
of 2,225, the address follows the crop heard (overlap 0.15), and held-out
next-crop and bass-note accuracy both rise. Two further options exist and
are off by default: `record_averaging` makes a cell's write rate one over its
written mass with `record_rate` as the floor (a fresh cell takes its first
outcome whole, a familiar one averages), which on the composer stream
calibrated reads at unfamiliar readings and improved retention but slowed
adaptation to a new track; `record_homeostasis` moves each cell's offset
toward an equal activation share, which spread the code further but cost
adaptation as well. Both are measured in
`cadence-mission/results/record_addressing_v8.json`.

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

## Diagnosing a record store

A record store can fail silently: prediction with records still beats
prediction without them, because the store always holds something. Four
numbers show whether it holds the right thing. Replay the readings of a
training stream through `records.code(..., valued=False)` and count, per
cell, how often it is active.

- **Cells in use** and **the share of activations taken by the most active
  cells.** With `active` of `cells` winners the ideal share of any 200 cells
  is `200 / cells`. On the composer stream the defective reading used 2,225
  of 8,192 cells with 0.74 of all activations in the top 200; the repaired
  one 5,812 cells with 0.27.
- **Code overlap when only one block of the reading changes.** Hold the
  context, change the input (and the reverse), and compare the two codes.
  An address that does not move with a block cannot store anything about
  it. There the overlap was 0.62 to 0.71 when only the event heard changed
  and 0.92 when the context moved four bars; after the repair 0.15 and 0.88.
- **Held-out scores with records frozen, zeroed, writing online, and writing
  online from empty.** Frozen far below online means the store forgets its
  training stream; online equal to online-from-empty means the trained
  records contribute nothing to a new stream.
- **The read at readings no stream has written,** for instance a rollout
  from silence. A read far from zero there is interference from unrelated
  writers, not memory.

## What a write rate stores

The delta rule at a rate near one half makes a cell hold its last few
writers. That is the right memory for a new stream: its first moments
overwrite the cells they touch, and the read then calibrates the prediction
to that stream (on the composer, four observed bars of an unheard track
moved the change port's read from +0.13 to -0.22 and its rate of predicted
changes from 0.24 to 0.07 per moment). It is the wrong memory for a corpus:
with records frozen the same store recalled 0.03 to 0.11 of the training
basslines. `record_averaging` makes each cell average its writers instead,
with `record_rate` as the floor; retention and calibration at unfamiliar
readings improve, adaptation to a new stream slows in proportion. One table
cannot do both at full strength; choose by which the task reads.

## Recall is by content, not by position

A record is keyed by the reading: what was heard and the context. Where the
stream recurs, the key recurs. On a composer playing its loop from silence,
the code at one place and the code at the same place one loop later share
0.91 of their cells, against 0.04 for two unrelated moments. When the
playing drifts in phase, so that the same place in the bar hears another
event, the overlap at the same place is 0.16 and nothing written there is
read back. This decided an experiment: a composer that wrote its own first
four bars into its records, as it does for a heard track, did not bring
its opening back four bars later (the opening's departures returned at 0.09
to 0.20 of their places, against 0.18 without the writes), because by then
its playing had shifted in phase and it heard other events at those places.
A patch recalls a
moment when it meets the same event in a similar context. If a task needs
recall by position (the same bar of a phrase, the same step of an episode),
position has to be in the reading as an input of its own, with enough ports
to move the code; a clock of eight ports among eighty inputs does not.

## Boundary readings

A reading met once per stream, such as the wake moment with nothing heard,
receives a handful of updates per epoch. The slow parameters barely learn
it and its record cells are shared with common readings, so the first
prediction of a rollout from silence is poorly determined (on the composer
it started the loop out of phase with the clock). Make the boundary a
common reading by convention instead of hoping it is learned: there the
wake hears a count-in, the last event of the loop, and the rule the patch
knows best produces the first event.

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
the energy's parameter derivatives. The energy describes the slow patch: its
readout residual is `y - C h - c`, and the record read lies outside it. For
this patch the contrast equals the adjoint gradient up to terms of order
`beta^2`; `observe` computes the same learning signal by one backward scan.
Neither route sees the record: the read is a port value added to the
readout.

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

## Running a trained patch outside Python

The forward pass needs no library. Export the six parameter arrays, the
record table, the running mean of the reading, the input norm and the
record configuration; everything else is regenerated. The record projection
and offsets come from the seed: the `Mulberry32` generator, then
`normals(n)` by Box-Muller over `2 * ceil(n / 2)` draws with the radii from
the first half of the draws and the angles from the second half, cosines
before sines; the projection takes `reading * cells` normals divided by
`sqrt(reading)`, row by row over the reading, and the offsets take the next
`cells` normals times `bias`. A moment is then the gate and the context
update, the reading `[u * sqrt(n) / s, r * h]` minus the mean, the drives,
the `active` largest of them (a heap of that size is enough), their
positive parts normalised to unit length, the table rows weighted by that
code, and the readout. At 208 reading units and 8,192 cells a moment is
about 1.7 million multiply-adds. A JavaScript port built the projection in
0.1 s and ran 128 moments in 0.5 s; with the table shipped as float32 it
reproduced an archived Python rollout in every played event, with outputs
equal to 2e-8. Keep such an archived rollout as the port's parity test.

## Categorical ports

Symbols (words, moves, note names) are taught as distributions, not as levels.
With `groups` the slow readout ends in one softmax per group of ports and
learns by cross-entropy, whose output error is `softmax - target`:

```python
import numpy as np
from cadence import RecordPatchNet

net = RecordPatchNet(inputs=17, hidden=8, outputs=6, seed=2, cells=4096, active=32,
                     record_rate=1.0, groups=(6,))
inputs = np.zeros((1, 16, 17))
inputs[0, 0, 0] = 1.0
inputs[0, np.arange(16), 1 + np.arange(16)] = 1.0
identities = np.random.default_rng(7).integers(6, size=16)
first = net.observe(inputs, np.eye(6)[identities][None], rate=0.0)
assert np.allclose(first.prediction.slow_output.sum(axis=-1), 1.0)
recalled = net.imagine(inputs, state=np.zeros((1, 8)))
assert np.array_equal(recalled.output[0].argmax(axis=1), identities)
```

A record then holds `onehot - softmax(C h + c)`, a residual bounded by one in
every port, so the record algebra is the one above: a reading the slow weights
have learned leaves a record near zero. The prediction `softmax + read` is not
a distribution; decide by its largest port. One moment whose read pushes the
true port below zero costs the clipped cross-entropy 27 nats, so judge a
categorical prediction by accuracy or by the median loss, not the mean. Each
group's teaching weight is the mean `output_precision` of its ports. `detune`
refuses categorical ports: the energy is quadratic only with the linear
readout. Default nets keep checkpoint format 2; a net with `groups` or batch
writes saves format 3.

## Writing a batch

`record_writes="batch"` writes all moments of a call at once through
`Records.write_batch`: every error is taken against the tables as they stood,
and each cell moves by the mean of the moves its writers would have made alone.
One writer reproduces `Records.write` to rounding; writers that agree move a
shared cell as far as one of them would, so a batch does not overshoot. On a
path of four streams of ten symbols presented twelve times, batch writes left
34, 12, 7, 4, 1 wrong moments after the first five presentations and sequential
writes 34, 22, 18, 10, 6; both end at the one moment that is undecidable (two
streams open with the same symbol and different outcomes). Sequential writes
within a call let a later moment overwrite an earlier one that shares its
cells; the batch average does not.

## Do not record a choice

When a reading has several right outcomes (a message that can be said in four
ways), the slow readout learns their distribution and its error at that moment
is the choice that happened to be taught. A record of it pulls later
predictions toward one arbitrary choice and spends cells. Write records for
what was observed once and must be reproduced (a sentence read, a phrase
heard), and leave choice points to the slow weights.

## Sizing a store for recitation

A store recites a sequence exactly only when it has more cells than
associations to hold, and it needs several presentations, because keys of
neighbouring moments overlap. Closed-loop recitation of random sentences of 8
to 22 words from a 1,500-word vocabulary, keyed by a message and the word just
said, records only (untrained slow weights), 32 active cells, eight passes:

| sentences | associations | cells | words right, teacher-forced | sentences exact, closed loop |
| --- | --- | --- | --- | --- |
| 300 | 4,400 | 8,192 | 0.93 | 0.36 |
| 1,000 | 14,700 | 8,192 | 0.78 | 0.04 |
| 1,000 | 14,700 | 32,768 | 0.97 | 0.66 |

A closed loop multiplies the per-word rate over the sentence, so exact
recitation needs the per-word rate near one: more cells, more passes, and slow
weights that learn most of the material first.

## A store narrower than its port

A port of thousands of categories need not give the store one column per
category. With `record_width=w` the cells hold a fixed random sign code of the
residual, `residual @ R` with `R` of shape `(outputs, w)` and entries
`+-1/sqrt(w)`, and the read is decoded by the transpose, `held @ R.T`. A stored
residual comes back with crosstalk of standard deviation `|residual| /
sqrt(w)` per port, which the largest port survives. The delta rule has to
compare like with like: the coded residual with what the cells hold. Taking
the error after decoding and projecting it again multiplies the step by
`outputs / w` (10.7 at 5,481 words and 512 columns), and the store diverges:
in the language work this recited nothing until it was found. Closed-loop
recitation of 300 random sentences (4,353 associations, records only, a
1,500-word port): one column per word 0.44 exact after eight passes, a
256-column code 0.41, at a sixth of the memory. At write rate one a
16,384-cell store recited 0.39, 0.74, 0.93, 0.987 of the sentences exactly
after 4, 8, 16, 32 passes, and a 32,768-cell store 0.77, 0.957, 0.98, 0.997:
about seven cells per association and thirty passes for exact recitation
when the slow weights know nothing.

## A grammar from records alone

Records generalise by overlap, and that is enough for a small grammar. A speaker whose input is
a message (act, kind, number and agreement features, 124 ports) and the word it has just said,
with the slow weights left at their random initial values, was given one pass of delta-rule
writes over 10,672 (message, sentence) pairs of a grammar of 1,276 messages and 6,173 sentences,
then asked to say sentences for messages it had never met: a quarter of the (subgenre, era,
form) combinations and a tenth of the other pairs were held out. With 65,536 cells, 32 active, a
320-column output code and write rate one, the store alone produced a sentence inside the
grammar for 0.815 of the held-out combinations and 0.885 of the other held-out messages, and
preferred the grammatical member of every minimal pair (a/an, is/are, deal/deals, both/all); a
second pass gave 0.85 and 0.915. The writes took about a minute. A new message shares most of
its features with taught ones, so its reading touches their records, and the read averages
them: the record principle doing agreement and word order without a gradient. A two-layer patch
whose slow weights learned the same pairs by fifty epochs of cross-entropy reached 1.00 on the
same held-out sets, and so did a GRU and a transformer of the same width; at a twelfth of the
pairs and equal updates the patch had the lowest held-out loss of the three at every width
(0.32 against 0.34 and 0.36 nats per word at width 64). The grammar is finite and every learner
reaches its ceiling with enough data; what the store shows is what one pass of writes buys.
The receipts are `results/e5_records_speaker.json`, `results/e3_scaling.json` and
`results/baselines.json` of the language work.

## Two patches in depth

The gate of a record patch sees only the present input. That is enough for a
conjunction of the last two symbols: an arbitrary function of the ordered pair
is learned by the slow weights alone (0.9999 on a 16-symbol probe), because
the gate multiplies the carried context. It is not enough to tell a subject
from a later noun. On subject-verb agreement across prepositional phrases with
nouns of the opposite number, one patch chose the right verb number in 1.00,
1.00 and 0.03 of sentences with zero, one and two distracting nouns (0.51 with
records); a GRU and a two-layer transformer scored 1.00 throughout. A second
patch that reads the first patch's context has a gate that sees what came
before, and scored 1.00, 1.00, 1.00:

```python
import numpy as np
from cadence import RecordPatchStack

stack = RecordPatchStack(inputs=5, hidden=8, outputs=4, seed=5, cells=2048, active=16,
                         groups=(4,), record_rate=1.0)
symbols = np.random.default_rng(4).integers(5, size=(3, 8))
target = np.eye(4)[(symbols + np.roll(symbols, 1, axis=1)) % 4]
seen = stack.observe(np.eye(5)[symbols], target, rate=2.0, backtrack=True)
assert seen.updated and seen.writes == 24
```

`RecordPatchStack` is a context patch below a complete `RecordPatchNet` that
reads `[u, r1 * h1]`; the upper patch owns the readout and the records. The
upper adjoint scan hands its input gradient to the lower scan, and the
gradient matches finite differences in both patches. Work per moment stays
linear in the two widths.

## What this class does not do

Planning through action ports, protected responses through `TemporalMemory`
and coupled components beyond the two-patch stack are not yet available for
this class. The stack has no `detune`. The record
rate, the number of active cells and the reading scale are supplied. There
is no automatic importance, forgetting, or learned relevance: a record is
written for every observed moment and moves only when its cells are read
again.
