# Memory: read before writing

A brain with bounded state needs records it can read back and correct. `FastSynapses`
stores a key-to-value matrix of fast synaptic weights per stream. Its storage is fixed by
the key and value widths; it does not grow with the number of observations. Slow weights
can learn representations around that store, but the store does not learn its own keys
or decide when an observation is trustworthy.

For repeated and salient experiences, [`SynapticMemory`](continuous.md) adds persistent
shared synapses and fading per-stream residuals. New generic brains with `episodic=True`
use this consolidation rule. The `FastSynapses` API below retains its original independent
stream behavior and immediate residual-write rule.

For retrieval from observed content, [competitive content memory](content_memory.md) learns
bounded prototypes and selects one by cue similarity. For retention of past supervised
tasks in a single output head, [explicit replay](replay.md) stores a declared reservoir of
past feature/label pairs. Both add counted memory and neither learns a general address policy.

## One correction

For a unit key `k`, value `v`, and matrix `M`:

```text
prediction = k @ M
error      = v - prediction
M         += rate * outer(k, error)
```

With `rate=1`, `amplitude=1` and a nonzero key, reading that key immediately after the write returns
`v`, to numerical precision. Repeating the same correct observation has zero residual
and does not accumulate strength. For an earlier key `q`, the change in its read is
`rate * dot(q, k) * error`: orthogonal records are preserved, correlated ones interfere.
This identity explains both the useful behaviour and its limit.

Cadence normalizes delta keys and queries to unit length. Normalization reads the whole
key vector; each fast synapse then updates from its presynaptic key coordinate and its
postsynaptic neuron's error. This is normalized delta learning, closely related to
normalized LMS and [delta-rule fast weights](https://arxiv.org/abs/2406.06484).
A read is one linear transport from key to value; it involves no softmax attention and
no settling.

## Reading and writing directly

```python
import numpy as np
import cadence as cd

memory = cd.FastSynapses(np.arange(4), np.arange(4, 6), rule="delta")
keys = np.eye(4)[:2]                   # two independent streams
values = np.array([[1., 0.], [0., 1.]])
before = memory.recall(keys)          # query before revealing the values
memory.observe(keys, values)          # one observation in each stream
assert np.allclose(memory.recall(keys), values)
memory.reset(2, rows=np.array([True, False]))  # only the first episode ended
```

`observe(key, value, write=None)` accepts two-dimensional arrays with matching batches;
a boolean `write` mask gates each stream. Every call decays all strengths by `decay`
before writing selected rows. Reads do not decay state. `rate` lies in `[0, 1]` in delta
mode. A zero key cannot write an association. `amplitude` scales reads as a drive,
without changing the target used by the write rule. A changed batch size starts fresh
streams; use `keep(rows)` when dropping streams while preserving their identities.
`reset` clears records; `writes` counts lifetime write events.

## With an existing brain

The same memory can read key neurons and drive value neurons:

```python
connectome = cd.Connectome.from_synapses(
    6, pre=[0, 1, 2, 3], post=[4, 4, 5, 5], count=[1] * 4,
    populations={"key": range(4), "value": (4, 5)},
)
brain = cd.Brain(connectome, cd.learning_neuron_model())
drive = np.zeros((2, connectome.n))
drive[:, :4] = np.eye(4)[:2]

# pre/post are declared neuron indices in this connectome.
fast = cd.FastSynapses(
    np.array(connectome.populations["key"]), np.array(connectome.populations["value"]), rule="delta"
)
drive_with_record = fast.stimulate(drive)
state = brain.settle_batch(drive_with_record)
# Later, after observing what actually followed:
observed_rows = np.array([True, False])
observed_values = np.array([[1.0, 0.0], [0.0, 0.0]])
fast.update(state, write=observed_rows, post=observed_values)
```

The key used by `update` is the key neurons' *activation*, while `read`/`stimulate` uses the
key range of the supplied *drive*. Supply compatible representations; nonlinear
activation can change a dense key's direction. Use direct `observe`/`recall` when the
same external key should be used for both operations. Avoid writing the brain's
own prediction as if it were new evidence.

`update(state, write=None)` only decays the stored strengths, while
`observe(..., write=None)` writes every row. The caller owns episodic fast state;
`Learner.save` does not save a separate `FastSynapses` or `Trace` object.

## Additive memory

The default `rule="hebb"` adds `rate * k vᵀ` on each write.
`normalize=True` unit-normalizes keys and divides a read by decayed write mass.
`replace=True` clears matrix rows whose key coordinates are positive before adding the
association. That is useful for one-hot slots, but erases unrelated records with
dense positive keys. Both flags are rejected in delta mode to avoid mixing incompatible
read and write semantics.

The [changing-memory example](https://github.com/muellerberndt/cadence-examples/tree/7302f2af3dc0638bbafd1da1446ed96ffabaa9dd/05_memory)
compares all-key retention and newest-key accuracy with additive writes, exact lookup,
and a trained transformer on identical streams. Its key/value parsing is supplied to
all methods. Count mutable matrix storage as well as trained parameters. Fixed storage
is a capacity tradeoff: contradictory associations, nearly parallel keys, and more
independent values than the key rank cannot all be represented exactly.

## Pattern separation

A write at key `k` moves the read at key `q` by the dot product `q · k` times the correction,
so records interfere exactly as much as their keys overlap, and keys with disjoint supports
do not interfere at all. `PatternSeparator` turns correlated keys into sparse codes before
the record sees them: a fixed random projection onto a wider range, then the strongest
`winners` entries kept and the rest set to zero. `center` subtracts a running mean of the
observed keys first, which removes what every key shares.

```python
import numpy as np
import cadence as cd

key_neurons, value_neurons = np.arange(32), np.arange(32, 40)
sep = cd.PatternSeparator(inputs=32, expansion=1024, winners=8, seed=0, center=0.99)
sep.habituate(np.random.default_rng(0).standard_normal((256, 32)))  # the environment's keys
memory = cd.FastSynapses(pre=key_neurons, post=value_neurons, rule="delta", separator=sep)
```

`habituate` sets the running mean from a sample of the environment's keys before anything
is stored; the slow `center` then tracks it. A fast running mean moves the codes between a
write and its read, which is the one way to lose a record.

The record then has `expansion` rows per stream instead of `inputs`, which is the price:
storage grows with the code, capacity grows with it too (exact storage is bounded by the
code width, not the key width). Sixteen keys at cosine 0.9 that a plain delta record
holds at a third are held exactly after separation; see `examples/certified_memory.py`
and the certificate guide. The dentate gyrus expands and sparsifies entorhinal input before
the hippocampus stores it for the same reason.

`SynapticMemory` supports the same expanded write/read coordinates, but requires
`separator.center=0`: moving the separator's mean would move the address of persistent
records. If centering is needed, apply one fixed training-fitted transform to both inputs
and queries. This protects a coordinate convention; it does not prevent interference
between overlapping keys or representation drift in a separately trained encoder.
