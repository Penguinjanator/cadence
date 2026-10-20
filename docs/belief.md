# The belief patch

`BeliefPatch` keeps a belief `z` that a learned transition carries forward under the executed
action. Each moment's evidence repairs it through `K = iterations` damped steps of one map that
also reads a record store, and a linear readout, patched by the store, answers. Unlike
[`RecordPatchNet`](record-patch.md), whose context comes from the moment's input, it can
imagine with no input: the transition needs only the belief and an action.

```text
p[t]    = g * z[t-1] + (1 - g) * tanh(T [z[t-1]; a[t-1]] + t_b),   g = sigmoid(G [z; a] + g_b)
e[t]    = gain[block] * tanh(port(o[t]) + e_b)
z(0)    = p[t]
m(k)    = read(code([e[t], z(k)]))
z(k+1)  = (1 - alpha) z(k) + alpha tanh(F [z(k); e[t]; p[t]; m(k); 1] + f_b)      k < K
y[t]    = C z(K) + c + decode(m(K))
```

`alpha` is `damping`. The repair runs a fixed budget with no fixed-point threshold; see
[contracts](contracts.md#choose-the-computation) before reading results as equilibrium evidence.

```python
import numpy as np
from cadence import BeliefPatch, DenseBlock, MapBlock, StructuredPort

# a small retina with the action broadcast into its map, and a dense block of fields
port = StructuredPort(1 * 8 * 8 + 4, [MapBlock(0, 1, 8, 8, 4, 3, 2), DenseBlock(64, 4, 8)], broadcast=(64, 4))
brain = BeliefPatch(port, actions=3, belief=16, outputs=6, iterations=2, cells=256, active=8, record_width=16, seed=0)
rng = np.random.default_rng(0)
observations = rng.random((2, 8, port.inputs))         # (N, T, inputs)
actions = rng.random((2, 8, 3))                         # (N, T, actions)
targets = rng.normal(size=(2, 8, 6))                    # (N, T, outputs)
brain.reset()
seen = brain.assimilate(observations, actions)          # the belief follows the evidence
ahead = brain.imagine(actions[:, :4])                   # private, from the live belief
learned = brain.observe(observations, actions, targets, rate=1.0, write=True)
assert learned.updated and learned.writes == 16
```

## What each part is for

Three clocks stay apart: moments `t`, repair iterations `k` with the evidence held fixed, and
learning between chunks.

- **The transition** predicts what follows an action. Scene and action meet inside the repair
  map, so an action's effect can depend on what is where.
- **The repair** assimilates evidence. `BeliefPath.step` is the last move per belief unit and
  `residual` its norm, a diagnostic only.
- **The store** is a [`Records`](records.md#records) store keyed on its fixed sparse code of
  `[evidence, belief]`. `observe(..., write=True)`
  writes the slow readout's residual, coded to `record_width` signs, once per observed moment;
  no gradient reaches it. `write` defaults to `False`. Write it when the slow model misses
  something specific; the price is reads that habituate to stale residuals.
- **Imagination** (`imagine`) runs the transition alone with the store read at the
  expectation. It changes no parameter, record or live state, and the cost counters include
  it. It cannot make a continuation conditioned on recorded future inputs.

## Learning

`observe` computes the adjoint of the precision-weighted half mean squared error of the slow
readout through every iteration and the transition, with store reads taken as given, and
matches finite differences to 1e-5 (`tests/test_belief.py`). The loss is a mean over time and
outputs, so a useful `rate` scales with their product; a wide picture port needs a rate in the
thousands, or the torch backend.

`cadence.belief_torch.TorchBelief(port, actions, belief, outputs, *, iterations=2,
damping=0.5, record_width=64)` (extra `accel`) is the slow half as a torch module for GPU
batches. It starts at zero; `load(patch.parameters())` copies a patch, and `export()` returns
arrays for `set_parameters`, so training continues in NumPy with the records.
`forward(observations, actions, state=None, reads=None, gains=None, observed=None)` returns
`(beliefs, slow outputs)`: `observations=None` imagines, `reads` `(N, T, record_width)` are
given store reads (zero when absent), `observed` is an `(N, T)` mask, and gains that require
grad receive their gradient. Loss weights, admission and external gradients are the caller's
optimiser; the readback stays on the NumPy patch.

### Training the transition: the imagination loss

Trained on the one-step read alone, the repair takes what it needs from the next frame and
open-loop imagination decays below persistence. The imagination loss counters that teacher
forcing: from random moments `t0` of each chunk, imagine `H` steps under the recorded actions
from the belief at `t0 - 1` with no observation, and add the drift to the one-step loss:

```text
L = L_one_step + w * mean over starts of  1/2 * mean( W * [cumsum_k(y_hat[t0+k] - y[t0+k]) ; fields]^2 )
```

The cumulative sum runs over retina outputs, fields compare step by step, and `W` is the
one-step weighting. The gradient reaches `T`, `G`, `C` and the starting belief. Two starts per
chunk, `H = 8` and `w = 0.3` are a starting point on pixels. A seed can leave the transition at
persistence; track the open-loop curve and raise `w` first. The recipe is the application's,
usually on `TorchBelief`.

## A gain per block inside the repair

Each observation-port block (`block_count` of them) carries a gain on its encoded evidence,
applied before the repair map and the store code, so it changes what the belief is repaired
toward. `assimilate` and `observe` take `gains=` as `(blocks,)`, `(batch, blocks)` or
`(batch, time, blocks)` (`imagine` only carries them on the path, since it hears nothing);
`observe` returns `gain_gradient` per moment.

```python
import numpy as np
from cadence import BeliefPatch, DenseBlock, StructuredPort

eye, ear = DenseBlock(0, 6, 8), DenseBlock(6, 2, 4)                # two senses, two blocks
cortex = BeliefPatch(StructuredPort(8, [eye, ear]), actions=1, belief=8, outputs=2, cells=128, active=4, record_width=8, seed=1)
rng = np.random.default_rng(1)
o, a, y = rng.random((4, 6, 8)), np.zeros((4, 6, 1)), rng.normal(size=(4, 6, 2))
cortex.reset()
cortex.observe(o, a, y, rate=1.0, write=False)                    # a readout, so the loss reaches the evidence
gains = np.tile([1.8, 0.2], (4, 6, 1))                            # (N, T, blocks): eye 1.8, ear 0.2
cortex.reset()
taught = cortex.observe(o, a, y, rate=0.0, write=False, gains=gains)
assert taught.gain_gradient.shape == (4, 6, 2) and np.array_equal(taught.path.gains, gains)
```

Free gains amplify every sense; gains summing to the block count move weight between senses.
That map belongs to the application ([steering](steering.md#the-weighing)).

## The readback of a moment

`BeliefPath` holds per moment `belief`, `expectation`, `residual`, `step`, `output`, `read`
(`slow_output = output - read`), `final_state` and the chunk's `loss`, plus `evidence` after
the gains, `code` (the store code of the final reading), `gains`, `residual_alone` (with
`probe=True`, the map's move from the expectation with one block heard and the store at zero)
and `surprise` (each block's reading against the one the previous belief's readout implies,
an RMS in persistence units, so one means no better than expecting no change).

`set_implied_reading(implied, units)` declares the surprise: a map from outputs to the reading
each block should give (`NaN` where not implied) and each block's mean squared change between
moments on training data. `None` withdraws it; snapshots do not keep it.
`readback(observations, actions, *, state=None, probe=True)` takes `(batch, inputs)` and
`(batch, actions)` and returns a `BeliefReadback` (`expectation`, `residual_alone`, zeros with
`probe=False`, `surprise`, `None` without a declaration, and `evidence` before gains) for one
moment before its repair, changing nothing and counting nothing. `encode(observations)` gives
that evidence for any `(..., inputs)` reading.

```python
def implied(outputs):
    reading = np.full((len(outputs), 8), np.nan)   # channels left NaN are not compared
    reading[:, :2] = outputs                       # the eye's first two channels read the predicted position
    return reading

cortex.set_implied_reading(implied, units=[0.01, 0.01])
cortex.reset()
path = cortex.assimilate(o, a, gains=gains, probe=True)
assert path.surprise.shape == (4, 6, 2) and np.all(path.surprise[:, :, 1] == 0.0)   # the ear implies nothing
moment = cortex.readback(o[:, 0], a[:, 0], state=np.zeros((4, 8)))
assert np.allclose(moment.residual_alone, path.residual_alone[:, 0])
assert np.allclose(moment.evidence, cortex.encode(o[:, 0]))
```

The first surprise compares against the boundary belief (`state`, else the live belief, else
zero); an unobserved row has zero probe and surprise. Probes cost one map evaluation per block
and moment, surprise one call of the declared map.

## An admitted step

With a target, `observe(rate=r)` replays the chunk from the same boundary under each proposed
step, with the store as it stands, halving from the start (sixteen tries at most), and takes
the first step whose loss falls by the Armijo margin (`1e-4 * step * |gradient|^2`); the store
is written afterwards. The start is twice the last admitted step, capped at `rate`, so `rate`
is a ceiling. `step_size` holds that step (`None` before one), travels with the snapshot,
survives `reset()` and is cleared by `reset_step()`. `backtrack=False` takes the plain step,
as does `output_gradient=`. An accepted step never raises the chunk's loss.

`BeliefObservation` carries `updated`, `reason`, `path`, `delta`, `initial_loss`,
`final_loss`, `accepted_rate`, `replay_calls`, `writes` and `gain_gradient`. Reasons are
`updated`; `no_step` (`rate=0`); `nonfinite_prediction`; `nonfinite_update` (plain step);
`no_decreasing_parameter_step` (no try qualified, or a zero gradient). Only `updated` moves parameters.

```python
cortex.reset()
admitted = cortex.observe(o, a, y, rate=100.0, write=False)
assert admitted.updated and admitted.final_loss < admitted.initial_loss
assert 0 < admitted.accepted_rate < 100.0 and cortex.step_size == admitted.accepted_rate
```

## A life lived online

`assimilate` and `observe` take `state=`, a boundary in place of the live belief, and
`keep_live=True` leaves the live state as it was, so a life can learn from a window it stepped
through without losing its place. Whole streams from a fresh boundary suit a high `rate` cap;
one stream online in short chunks needs a low one ([steering](steering.md#a-life-lived-online)).

```python
cortex.reset()
cortex.assimilate(o[:, :3], a[:, :3])                    # the life steps through three moments
boundary, here = np.zeros((4, 8)), cortex.state
cortex.observe(o[:, :3], a[:, :3], y[:, :3], rate=1.0, state=boundary, keep_live=True)
assert np.array_equal(cortex.state, here)
```

## The cost of a moment

`macs_per_moment(probes=False, surprise=True)` counts one moment's multiply-accumulates: port
(a map block per output position), transition and gate, store reads and repair per iteration,
readout and decode; `probes=True` adds the per-block probes, and `surprise=True` the
declared comparison when an implied reading is set. `cost` holds `moments`,
`macs` and `replays` since `reset_cost()`, over every assimilated, observed, imagined or
replayed moment; replays omit the surprise. `readback_macs(probe=True)` estimates one readback,
which leaves the counters alone. [Contracts](contracts.md#cost-counters) lists what the
counters exclude.

```python
cortex.reset_cost()
cortex.reset()
taught = cortex.observe(o, a, y, rate=100.0)
expected = 4 * 6 * (cortex.macs_per_moment() + taught.replay_calls * cortex.macs_per_moment(surprise=False))
assert cortex.cost["moments"] == 4 * 6 * (1 + taught.replay_calls) and cortex.cost["macs"] == expected
```

## A weight per moment and a mask per row

`observe(loss_weight=)`, `(time,)` or `(batch, time)`, weighs each moment's error, normalized
by the weight sum; weights are nonnegative with a positive sum. A zero-weight moment is
neither taught nor written, which keeps a discontinuity out of the gradient and the store.
`observed=` in the same shapes masks rows in `assimilate` and `observe`; an unobserved row
keeps its expectation, so streams that go dark at different moments share a batch.

```python
weight, observed = np.ones((4, 6)), np.ones((4, 6), dtype=bool)
weight[:, 2] = 0.0                       # neither taught nor written
observed[1, 3:] = False                  # one stream goes dark at moment 3
cortex.reset()
masked = cortex.observe(o, a, y, rate=0.0, write=True, loss_weight=weight, observed=observed)
assert masked.writes == int(np.sum(observed & (weight > 0)))
assert np.allclose(masked.path.belief[1, 3], masked.path.expectation[1, 3])
```

## Teaching through a seam

A steering patch's outputs set another patch's gains, so it has no target of its own.
`observe(..., output_gradient=dy)` replaces `target` with a gradient on the outputs
`(batch, time, outputs)`: nothing is written, no loss is reported, and `loss_weight` and
`backtrack=True` are refused. The cortex's `gain_gradient` through the caller's map Jacobian is
that `dy`. A joint admission of both patches is the caller's, and a readback that depends on
the previous moment's prediction is taken as given in the steering patch's adjoint.

```python
steering = BeliefPatch(StructuredPort(4, [DenseBlock(0, 4, 6)]), actions=1, belief=6, outputs=2, cells=64, active=4, record_width=4, seed=2)
readbacks = np.concatenate([path.residual_alone, path.surprise], axis=-1)   # (N, T, 4)
steering.reset()
seam = steering.observe(readbacks, a, output_gradient=taught.gain_gradient, rate=0.5)   # identity map
assert seam.updated and seam.writes == 0 and seam.path.loss is None
```

## Settings and limits

| `BeliefPatch(observation, actions, belief, outputs, *, ...)` | Default | Meaning |
| --- | --- | --- |
| `iterations` | 2 | Repair iterations `K` per observed moment. |
| `damping` | 0.5 | `alpha` in `(0, 1]`, the share of the map's proposal taken per iteration. |
| `cells`, `active` | 4096, 32 | Store code cells and winners per reading. |
| `record_rate` | 0.5 | Delta-rule write rate of the store. |
| `record_width` | 64 | Width of the coded residual the store holds and the repair reads. |
| `habituation` | 1e-5 | Slowest rate of the store's running mean of readings. |
| `record_bias` | 0.3 | Scale of the store cells' fixed offsets. |
| `output_precision` | ones | Positive per-output loss weight; `set_output_precision` changes it. |
| `seed` | 0 | Initial weights, output code and store projection. |

```text
assimilate(observations, actions, observed=None, *, state=None, gains=None, probe=False, keep_live=False)
imagine(actions, *, state=None, gains=None)
observe(observations, actions, target=None, *, observed=None, rate=1.0, write=False, state=None,
        gains=None, loss_weight=None, output_gradient=None, backtrack=None, probe=False, keep_live=False)
```

Observations are `(N, T, inputs)`, actions `(N, T, actions)`, targets `(N, T, outputs)`.
`observe` takes a target or an `output_gradient`, exactly one, and refuses a negative or
nonfinite `rate`; `assimilate` and `imagine` raise `FloatingPointError` on a nonfinite belief.
`state` is a copy of the live belief, `None` after `reset()`, which keeps step size and
counters; `updates` counts taken steps. `parameters()` and `set_parameters()` carry the ten
parameter arrays; `snapshot()`, `BeliefPatch.restore(snapshot)`, `save(path, compressed=True)`
and `BeliefPatch.load(path)` also carry the store, output precision, step size, `updates` and
live state. The repair map's contraction is measured, never proved, so there is no
[certificate](certificate.md). The store does not partition by action. The readout is linear;
categorical outputs and the night belong to the record patch.
