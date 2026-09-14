# Memory: read before writing

A bounded observer-like patch needs records it can read back and correct through its
ports. `FastSeams` stores a key-to-value matrix per stream. Its storage is fixed by the
key and value widths; it does not grow with the number of observations. Slow weights
can learn representations around that store, but the store does not learn its own keys
or decide when an observation is trustworthy.

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

Cadence normalizes delta keys and queries to unit length. Normalization reads the key
patch's whole vector; the seam update then reads its key coordinate and the receiving
owner's error. This is normalized delta learning, closely related to normalized LMS and
[delta-rule fast weights](https://arxiv.org/abs/2406.06484), not a new learning theorem.
A read is linear transport, not softmax attention or a recurrent settlement.

## Direct ports

```python
import numpy as np
import cadence as cd

memory = cd.FastSeams(np.arange(4), np.arange(4, 6), rule="delta")
keys = np.eye(4)[:2]                   # two independent streams
values = np.array([[1., 0.], [0., 1.]])
before = memory.recall(keys)          # query before revealing the new values
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

## With an existing patch net

The same memory can read key owners and drive value owners:

```python
# pre/post are declared owner indices in this wiring.
fast = cd.FastSeams(np.array(w.sets["key"]), np.array(w.sets["value"]), rule="delta")
drive_with_record = fast.clamp(drive)
state = engine.settle_batch(drive_with_record)
# Later, after observing what actually followed:
fast.update(state, write=observed_rows, post=observed_values)
```

The key used by `update` is the key owners' *activation*, while `read`/`clamp` uses the
key range of the supplied *drive*. Supply compatible representations; nonlinear
activation can change a dense key's direction. Use direct `observe`/`recall` when the
same external key port should be used for both operations. Avoid writing the model's
own prediction as if it were new evidence.

`update(state, write=None)` retains the older API's decay-only behaviour. This differs
from `observe(..., write=None)`, which writes every row. Episodic fast state belongs to
the caller: `Learner.save` does not save a separate `FastSeams` or `Trace` object.

## Existing additive memory

The default `rule="hebb"` preserves earlier experiments: each write adds `rate * k vᵀ`.
`normalize=True` unit-normalizes keys and divides a read by decayed write mass.
`replace=True` clears matrix rows whose key coordinates are positive before adding the
new association. That is useful for one-hot slots, but erases unrelated records with
dense positive keys. Both flags are rejected in delta mode to avoid mixing incompatible
read and write semantics.

The [changing-memory example](https://github.com/muellerberndt/cadence-examples/tree/7302f2af3dc0638bbafd1da1446ed96ffabaa9dd/05_memory)
compares all-key retention and newest-key accuracy with additive writes, exact lookup,
and a trained transformer on identical streams. Its key/value parsing is supplied to
all methods. Count mutable matrix storage as well as trained parameters. Fixed storage
is a capacity tradeoff: contradictory associations, nearly parallel keys, and more
independent values than the key rank cannot all be represented exactly.
