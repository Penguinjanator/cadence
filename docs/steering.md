# A brain that reads itself

A **steered cortex** is a [belief patch](belief.md) whose sensory gains a second belief patch
sets from the cortex's own diagnostics, one moment at a time. A **life** runs a patch under a
**governor** that reads the brain's surprise and picks habit, imagination or learning. Each
moment runs readback, steering and cortex in sequence; they do not settle one joint fixed point,
and the [contract guide](contracts.md) states that difference.

## The steered cortex

`Steered(cortex, steering, weighing)` assembles the cortex's readback into a vector, runs the
steering patch's finite repair on it, turns its outputs into one gain per block of the cortex's
port through the weighing, and lets the cortex assimilate the moment under those gains. The
readback's channels, in order (`brain.layout`, `brain.readback_names`, `brain.channels`):

| channels | what they carry | setting |
| --- | --- | --- |
| `probe:b` | the repair map's move from the expectation with only block `b` heard | always |
| `surprise:b` | block `b`'s reading against the reading the previous belief implies, in persistence units (zeros without an implied reading) | always |
| `residual` | the previous moment's repair residual | always |
| `output:i` | the previous moment's outputs | `reads_output=True` |
| `age:b` | moments since block `b`'s gain last reached `age_core` (default 0.5), divided by the block count | `reads_age=True` |
| `evidence:b:i` | block `b`'s encoded evidence before any gain | `evidence=[b, ...]` |
| `extra:i` | channels from `extra(cortex_readback, prev_output, prev_residual)`, the first `None` when lagged | `extra=`, `extra_channels=` |

`lagged=True` fills the probe, surprise and evidence channels from the previous moment's path,
masked where that moment's gain was zero, so under a sensing cost the steering patch reads only
what was heard (nothing at a fresh boundary). `relative=True` rolls every per-block group so
index zero is the block under a `Gaze` centre; it refuses any other weighing. `rate_scale`
scales the steering patch's step against the cortex's. Which channels the steering patch hears
is the mask of its port, a gene; probes nobody hears are not computed (`probes_on`). With a
steering patch and no weighing the weighing is `Softmax(blocks)`. The constructor refuses a
steering port that does not read exactly `channels`, a weighing whose `blocks` or outputs do not
match, a rule beside a steering patch, and the cortex as its own steering patch.

Give each sense its own block of the cortex's port, so each has a gain. Surprise needs the
cortex's implied reading and persistence units, each sense's mean squared change from one quiet
moment to the next on the training streams (`set_implied_reading`, in
[the belief patch](belief.md)). A steering patch can treat an event differently only if what
distinguishes it is in the readback, so route a distinguishing sense through `evidence=`. Give
the sense that should dominate in the quiet an order of magnitude less noise, or the day's
weighing rests on whichever is more predictive and nothing is left to capture.

```python
import numpy as np
from cadence import BeliefPatch, DenseBlock, Softmax, Steered, StructuredPort

# a dummy at a position in [0, 1]^2 reported by an eye and an ear; in one phase the ear is displaced
rng = np.random.default_rng(0)
def world(n, t, displaced):
    x = np.cumsum(rng.normal(0.0, 0.05, (n, t + 1, 2)), axis=1) % 1.0
    eye = x[:, :-1] + rng.normal(0.0, 0.01, (n, t, 2))
    ear = x[:, :-1] + (0.3 if displaced else 0.0) + rng.normal(0.0, 0.01, (n, t, 2))
    return np.concatenate([eye, ear], axis=-1), np.zeros((n, t, 1)), x[:, 1:]

eye, ear = DenseBlock(0, 2, 6), DenseBlock(2, 2, 6)
cortex = BeliefPatch(StructuredPort(4, [eye, ear]), actions=1, belief=12, outputs=2, cells=64, active=4, record_width=8, seed=1)
def implied(y):                       # what each sense should read if the belief is right
    r = np.full((len(y), 4), np.nan)
    r[:, :2] = y                      # the eye reads the position
    return r
cortex.set_implied_reading(implied, units=[0.0025, 0.0025])

# the steering patch hears the two surprises and the residual, not the probes
hears = np.array([False, False, True, True, True])
steering = BeliefPatch(StructuredPort(5, [DenseBlock(0, 5, 4)], mask=hears), actions=1, belief=6, outputs=2, cells=16, active=2, record_width=4, seed=2)
brain = Steered(cortex, steering, Softmax(2, span=2.0), rate_scale=0.5)
assert brain.channels == 5 and brain.readback_names == ["probe:0", "probe:1", "surprise:0", "surprise:1", "residual"]
assert not brain.probes_on
```

`run(observations, actions, target=None, *, rate=0.0, state=None, keep_live=False,
learn_cortex=True, backtrack=True)` lives a chunk `(batch, time, ...)`. With a target and a
rate it takes the cortex's adjoint, pulls the gradient into the gains back through the weighing
to the steering patch, and admits one joint step by a replay of the chunk on the recorded
readbacks. The step starts at `rate`, later at `min(rate, 2 * step_size)`, and halves while the
objective misses the Armijo margin; `backtrack=False` takes `rate` once with no replay.
`learn_cortex=False` lets the cortex sleep: only the steering patch steps, on the cortex's
gradient. A run without `state` continues the live boundary when its batch size matches and
starts fresh otherwise; `reset()` forgets the live boundary and keeps the step size and
counters.

```python
for phase in range(6):
    o, a, y = world(4, 16, displaced=bool(phase % 2))
    brain.reset()                                            # each batch from a fresh boundary
    taught = brain.run(o, a, y, rate=4.0)
    assert taught.updated and taught.steering_updated and taught.reason == "updated"
o, a, y = world(4, 16, displaced=True)
brain.reset()
path = brain.run(o, a, y)
assert np.allclose(path.gains.sum(axis=-1), 2.0)          # weight moves between the senses
assert path.gains.shape == (4, 16, 2) and path.readback.shape == (4, 16, 5)
assert brain.step_size is not None and brain.cost["replays"] >= 6
```

`SteeredPath` carries `output`, `gains`, `readback`, `residual`, `steering_output`, `loss`,
`price`, `objective` (loss plus price), the step (`updated`, `steering_updated`, `step`,
`halvings`, `replays`) and the last moment's two `BeliefPath`s (`last`, `last_steering`).
`reason` is `"updated"`, `"no_step"`, `"no_gradient"`, `"nonfinite_prediction"` or
`"no_decreasing_parameter_step"`; under a rule, fixed gains or an ablation only the cortex
steps and `reason` is its own `observe` reason. Invalid inputs or a failed replay leave the
boundary and parameters unchanged. `snapshot()`/`save(path, compressed=True)` keep both
patches, the settings, the step size, the ablation, the deafness and the live boundary;
`Steered.restore(snapshot, rule=, extra=, ablation=)` and `load(path, ...)` take the callables
back as arguments. The implied reading is not in a snapshot: declare it again.

## The weighing

The weighing declares what the steering patch may do to the cortex.

| weighing | gains | steering outputs |
| --- | --- | --- |
| `Softmax(blocks, span=1.0)` | `blocks * softmax(span * tanh(y))`: they sum to `blocks`, so weight moves between senses and none is amplified against the belief | `blocks` |
| `Gaze(blocks, sigma, cut=2.5, lamp=0.2, span=0.5, price=0.0, start=0.0)` | a Gaussian window over a ring of blocks at bearings `2*pi*k/blocks` (`sigma` in radians), cut beyond `cut` widths, at brightness `min(1, lamp / sigma)`; the centre starts at `start` and turns by `span * tanh(y)` each moment, priced `0.5 * price * mean(turn**2)` in the objective; the centre travels in the boundary | 1 |
| `Rule(fn, macs=0)` | a hand-written map from the readback `(batch, channels)` to gains `(batch, blocks)`, the control; no steering patch, the cortex learns by its own step | none |
| none | fixed gains of one: the brain below | none |

A new weighing subclasses `cadence.steering.Weighing` with `outputs`, `blocks`, `begin(n)` (its
state at a fresh boundary), `gains(y, state) -> (gains, state)`, `pull(ys, gains, dgains,
states)` (the Jacobian's action on the gradient into the gains), and optionally `price(ys,
states)`. `snapshot()` needs its `to_dict()`, and `restore` rebuilds only `Softmax` and `Gaze`.

```python
from cadence import Gaze, Rule

ring = BeliefPatch(StructuredPort(8, [DenseBlock(i, 1, 1) for i in range(8)]), actions=1, belief=8, outputs=8, cells=16, active=2, record_width=4, seed=3)
lookout = BeliefPatch(StructuredPort(17, [DenseBlock(0, 17, 4)]), actions=1, belief=4, outputs=1, cells=16, active=2, record_width=4, seed=4)
beam = Steered(ring, lookout, Gaze(8, sigma=0.5, cut=2.0, lamp=0.3, span=0.4, price=0.01))
lit = beam.run(rng.random((2, 6, 8)), np.zeros((2, 6, 1)))
assert lit.gains.shape == (2, 6, 8) and (lit.gains == 0.0).any()          # cut beyond two widths
assert beam.boundary().weighing.shape == (2,)                              # the centre per stream

def rule(readback):                                                        # the hand-designed control
    ear = np.where(readback[:, 3] > 2.0, 1.8, 0.4)
    return np.stack([2.0 - ear, ear], axis=-1)
control = Steered(cortex, weighing=Rule(rule, macs=6))
ruled = control.run(o, a, y, rate=4.0)
assert ruled.steering_output is None
assert control.macs_per_moment() == cortex.macs_per_moment() + cortex.readback_macs(probe=False) + 6
below = Steered(cortex)
assert np.all(below.run(o, a).gains == 1.0)
```

## The switch and the accounting

A mechanism is accepted when the task works with the switch on and fails with it off, at
matched information and compute. The three arms are constructions of one class over one
identical cortex: `Steered(cortex, steering, weighing)` with the step on; `Steered(cortex)`, or
a steering patch whose mask hides the tested channel, with it off; `Steered(cortex,
weighing=Rule(...))` as the control. Write the claim, the control and the falsifier before
anything runs.

Two ablations act on a trained brain at test time. `brain.ablation = "cut"` holds the gains at
one while the steering patch keeps running and is counted; a callable maps the gains (a shuffle,
`lambda g: g[:, ::-1]`). `brain.deaf = mask` multiplies the readback channels by the mask.

Every arm reads one accounting: `macs_per_moment()` (the cortex's moment, its readback with the
probes when heard, the steering patch's moment, a rule's declared `macs`),
`moments_per_decision()` (two with a steering patch, one otherwise) and `cost`, the two
patches' counters summed with the joint admission's replays and the executed readbacks and rule
operations. Their scope is in [cost counters](contracts.md#cost-counters); a zero gain does not
make the dense port kernel sparse. `reset_cost()` and `reset_step()` clear the counters and the
remembered step.

```python
brain.ablation = "cut"
brain.reset()
cut = brain.run(o, a)
assert np.all(cut.gains == 1.0) and cut.steering_output is not None
brain.ablation = None
brain.deaf = np.array([True, True, True, False, True])   # deaf to the ear's surprise
brain.reset()
assert np.all(brain.run(o, a).readback[:, :, 3] == 0.0)
brain.deaf = None
assert brain.macs_per_moment() == cortex.macs_per_moment() + cortex.readback_macs(probe=brain.probes_on) + steering.macs_per_moment()
```

## Day and night

- **Day.** Train every arm on batches of whole streams, each from a fresh boundary (`reset()`
  first), one joint step per chunk, held-out validation, the best `snapshot()` kept. A short day
  leaves the weighing on the wrong sense.
- **Night.** One long stream in short chunks with `state=None`, at a cap near a tenth of the
  day's (the day's cap wrecks online tracking), the cortex asleep (`learn_cortex=False`) or gated
  by a `Life`. An exception is learned only with the cortex asleep; a learning cortex absorbs it.
  A `rate_scale` above one pins a sense. Measure the night length the exception needs.
- **Data.** Keep training, calibration and test streams disjoint; fix governor baselines and
  thresholds on calibration data.

Select genomes with `evolve(fitness, hand_set, mutate=genes(space), ...)` by
[the selection protocol](evolution.md#the-selection-protocol), the hand-set genome first.

## A life lived online

`boundary()` returns a detached copy of where the brain is between two moments (`Boundary`: both
beliefs, the previous output and residual, the weighing's state, what was heard, the moments
seen). `run(state=boundary)` starts there and `keep_live=True` leaves the live boundary where it
was. To learn from a chunk a page has shown, keep the boundary before it, step through it, then
learn from the kept boundary: the replay makes the same moments the steps made.

```python
brain.reset()
for k in range(8):                                         # a stream lived one moment at a time
    brain.run(o[:1, k : k + 1], a[:1, k : k + 1])
kept = brain.boundary()
seen = np.concatenate([brain.run(o[:1, k : k + 1], a[:1, k : k + 1]).gains for k in range(8, 16)], axis=1)
learned = brain.run(o[:1, 8:], a[:1, 8:], y[:1, 8:], rate=0.5, state=kept, keep_live=True)
assert np.allclose(learned.gains, seen) and brain.boundary().moments == 16
```

## A life with a governor

`Life(patch, governor, *, habit, propose, advance, cost, target, baseline=1.0, residual0=1.0,
config=None, refit=None, keep_records=True)` is one continuing loop over a `BeliefPatch` or a
`Steered` cortex. It is separate from `Brain.compose`, whose arousal is in
[continuous interaction](continuous.md). The application supplies the body: `habit(reading)`
the cheap policy, `propose(reading)` the candidate actions, `advance(reading, output)` how a
predicted change moves the reading, `cost(readings, actions)` the task's price per candidate and
`target(reading, next)` what the brain should have predicted. `baseline` and `residual0` are
the routine surprise and residual, medians on a quiet stretch.

The surprise of a moment is the mean squared difference between the expected output and the
target. At each decision the life builds a seven-channel readback (the last surprise over its
baseline, log-compressed; its slow average; the last residual divided by `residual0`; the last
mode as three flags; a constant) and the governor names the mode:

- **Habit** executes `habit(reading)`.
- **Imagine** runs each candidate over `horizon` moments of the belief's private continuation
  and executes the best first action.
- **Learn**, when the signals allow it (`can_learn`), imagines the action and replays the last
  `window` moments of the current continuation with up to `passes` admitted steps at
  `learn_rate`; otherwise the decision imagines.

A learned window whose loss did not fall below `validity` times its start is undone (with
`rollback`) and the baseline habituates to `max(floor, min(habituate * baseline,
median(recent)))`. A kept window resets the baseline to `max(floor, 2 * loss_after)` and runs
`refit(life)`, whose return goes into the learn record. Either way the recent surprises and the
slow average clear and `cooldown` starts. At every outcome the baseline follows surprises up to
three baselines at `baseline_rate`; `floor` is in units of the constructor's `baseline`.
`decide(reading)` returns the action, `outcome(next_reading)` closes the moment and
`step(reading, world)` does both. `reset()` forgets the live state and the governor's and drops
an open decision; parameters, history and counters stay.

| `LifeConfig` field | default | meaning |
| --- | --- | --- |
| `window`, `min_window` | 96, 32 | moments a learn call replays; moments it needs |
| `passes`, `learn_rate` | 6, 0.1 | admitted steps per learn call, their rate |
| `validity`, `rollback` | 0.9, True | the fall a window needs to be kept; undo otherwise |
| `min_cooldown`, `cooldown` | 32, 0 | decisions after a learn call before the next |
| `horizon`, `hold` | 6, True | imagined moments; hold the candidate (False: habit after the first) |
| `baseline_rate`, `floor`, `habituate` | 0.02, 1.0, 2.0 | the surprise baseline's tracking, floor, habituation |
| `slow_rate`, `recent`, `keep` | 0.02, 96, 512 | slow average rate; surprises the governor sees; moments kept |
| `write` | False | write the patch's record store on a learn (refused for `Steered`) |
| `learn_cortex` | True | False: the cortex sleeps, and only a `Steered` steering patch learns |
| `extra` | `{}` | the application's own constants; the life does not read them |

| governor | what it is |
| --- | --- |
| `PatchGovernor(genome=None, *, cortex=4, model=GOVERNOR_MODEL, budget=200, chunk=10, tolerance=1e-3)` | a settling rate-neuron graph: seven readback units, `cortex` units and three motor units, fourteen neurons by default (`GOVERNOR_MODEL` lives in `cadence.life`). Every synapse and bias is a gene (`rc_i_j`, `rm_i_k`, `cm_j_k`, `cb_j`, `mb_k`, `lateral_cortex`, `lateral_motor`, `warm` to start each settle from the last). `PatchGovernor.hand_set(cortex)` is the founder wiring and `PatchGovernor.space(cortex)` the space for `genes`. The settled motor state is the mode; an unconverged settle returns habit; nothing in it learns within a life. |
| `ThresholdGovernor(genome=None)` | the hand-designed control: a surprise above `k_imagine` baselines starts `imagine_budget` decisions of imagination (renewed by a new spike when `renew`); a learn call needs the surprise above `k_learn` baselines on `persist_share` of the last `persist` decisions, then waits `cooldown`. `HAND_SET` holds the founder values (`k_imagine=8`, `imagine_budget=12`, `k_learn=8`, `persist=96`, `persist_share=0.25`, `renew=True`, `cooldown=64`), `SPACE` the gene space; a genome overrides any of them. |
| `AlwaysAwake`, `NeverWakes` | the two ends of the switch: imagine every decision (learning by the threshold rule); habit every decision |

A custom governor subclasses `Governor` with `settle(signals) -> (mode, steps)`, optional
`after(signals)` and `reset()`, and `synapses`, which prices its steps in the brain's moments.
`Signals` carries `readback`, `surprise`, `baseline`, `residual`, `slow`, `mode`, `recent` and
`can_learn` (at least `min_cooldown` decisions since the last learn call, `min_window` moments
in the current continuation, and `cooldown` run out).

```python
from cadence import Life, LifeConfig, PatchGovernor, ThresholdGovernor

class Sill:                      # a dot that drifts and a paw the action moves; the drift flips at t = 60
    def __init__(self, seed=0):
        self.rng, self.x, self.p, self.t = np.random.default_rng(seed), 0.5, 0.5, 0
    def step(self, a):
        self.t += 1
        self.x = float(np.clip(self.x + (0.01 if self.t < 60 else -0.03) + self.rng.normal(0.0, 0.001), 0.0, 1.0))
        self.p = float(np.clip(self.p + 0.1 * float(a[0]), 0.0, 1.0))
        return np.array([self.x, self.p])

def make_life(governor, seed=0):
    patch = BeliefPatch(StructuredPort(2, [DenseBlock(0, 2, 6)]), actions=1, belief=8, outputs=2, cells=32, active=4, record_width=4, seed=seed)
    return Life(patch, governor,
                habit=lambda r: np.array([0.5 * (0.5 - r[1])]),                  # the paw drifts home
                propose=lambda r: np.array([[-1.0], [0.0], [1.0]]),              # three pushes to imagine
                advance=lambda r, y: r + y,                                       # a predicted change moves the reading
                cost=lambda R, A: (R[:, 0] - R[:, 1]) ** 2 + 0.01 * A[:, 0] ** 2, # the task's price
                target=lambda r, r2: r2 - r,                                      # what the brain should have predicted
                baseline=1e-4, residual0=0.1,
                config=LifeConfig(window=16, min_window=8, min_cooldown=4, passes=3, learn_rate=2.0, horizon=3))

for governor in (PatchGovernor(), ThresholdGovernor({"k_imagine": 3.0, "k_learn": 1.5, "persist": 6, "persist_share": 0.5, "cooldown": 0})):
    sill, life = Sill(), make_life(governor)
    reading = np.array([0.5, 0.5])
    for _ in range(90):
        reading, decision = life.step(reading, sill.step)   # decide, act in the world, close the moment
    report = life.compute()
    assert sum(life.totals["decisions"].values()) == 90 and report["moments_per_decision"] >= 1.0
    assert report["learn_calls"] == report["kept"] + report["undone"]
assert PatchGovernor().n == 14
```

| what a life keeps | contents |
| --- | --- |
| `life.records` | a `Decision` per decision: `t`, `mode`, `action`, `expected`, `residual`, `readback`, `governor_steps`, `imagined`, `learned`, `surprise`, `baseline`, `target`; `keep_records=False` skips them for a long life |
| `life.learns` | each learn call: `t`, `window`, `loss_before`, `loss_after`, `valid`, `kept`, `passes_kept`, `refit` |
| `life.totals` | decisions by mode, `governor_steps`, `governor_moments`, `imagined`, `learn_calls`, `kept`, `undone` |
| `life.compute()` | the totals plus the patch's `moments`, `macs` and `replays`, `moment_macs`, `total_moments` (governor steps converted to moments by its synapse count) and `moments_per_decision` |

## The instruments of the orienting response

`orienting(gain, events, *, pre=4, post=12, quiet=None, bins=10)` measures a gain trace
`(frames,)` against events `(kind, start, length)`.

| result | what it is |
| --- | --- |
| `rows[i]` | per measured event: `kind`, `start`, `length`, `baseline` (mean of the quiet frames among the `pre` before onset), `peak`, `capture` (peak minus baseline), `latency` (first frame from onset reaching half the capture, `-1` without a positive capture), `return` (frames after the event until the gain is within a fifth of the capture of the baseline; `None` without a positive capture or a return), `return_censored` (a positive capture that did not return within `post` frames) |
| `kinds[k]` | per kind: `count`, `capture_first` and `capture_last` (means over the first and last `bins` events), `curve` (means over consecutive groups of `bins`), `capture_mean`, `latency_zero`/`_one`/`_none` shares, `latency_mean`, `return_mean`, `return_censored` (a count), `baseline_mean`, `peak_mean` |

The result also carries `pre` and `post`. `quiet` `(frames,)` marks frames that may serve as a
baseline, by default every frame outside every event; an event without a quiet frame among its
`pre`, or running past the trace, is skipped.

`dishabituation(rows, *, consequential, kind, window=60, count=2)` pools, around each
consequential event, the mean capture of the last `count` events of `kind` within `window`
frames before it (`before`) and of the first `count` within `window` after it (`after`), with
`events` the number pooled; a side with no event is `None`.

The sign of a capture depends on the baseline at dawn: where a sense rests open, habituation is
a closing of it. Report the gain at the consequential event beside the gain at the habituated
one, and the held-out error at the consequence.

```python
from cadence import dishabituation, orienting

gain = np.full(120, 0.5)
events = []
for i, start in enumerate([10, 25, 40, 55, 70, 85, 110]):   # a clock whose capture falls, then returns
    gain[start] = 0.5 + (0.4 * (1 - i / 6) if start < 100 else 0.4)
    events.append(("clock", start, 1))
gain[100:103] = [0.6, 0.9, 1.0]                              # a cry
events.append(("cry", 100, 3))
measured = orienting(gain, events, pre=4, post=12, bins=3)
clock, cry = measured["kinds"]["clock"], measured["kinds"]["cry"]
assert clock["capture_first"] > clock["capture_last"] and cry["capture_mean"] > 0.4
assert [r["latency"] for r in measured["rows"] if r["kind"] == "cry"] == [1]
assert not any(r["return_censored"] for r in measured["rows"])
around = dishabituation(measured["rows"], consequential="cry", kind="clock")
assert around["after"] > around["before"]
```

## Limits

The joint step's replay does not re-run the readback under the proposed cortex. A `Gaze` pull
is local in time: the effect of an earlier turn on later centres is not propagated. The surprise
of a stream's first moment compares against the readout of the boundary belief. `Steered.run`
does not write the cortex's record store. Imagination in a `Life` is the application's proposal
and cost; the library carries the loop, the governor and the accounting.
