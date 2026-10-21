# Standalone memory stores

These stores work next to a `NeuralGraph` without `Brain.compose`: `cd.FastSynapses` with
its `cd.PatternSeparator`, `cd.SynapticMemory` used directly, and the fading traces of
`cadence.stream`. Each writes by one delta-rule correction or a fixed decay, and none
settles. The memories of a composed brain are in [Memory](memory.md); signatures are in the
[API reference](api.md#streams-cadencestream).

## One correction

For a unit key `k`, value `v` and matrix `M`, a write is `M += rate * outer(k, v - k @ M)`
(normalized LMS, close to [delta-rule fast weights](https://arxiv.org/abs/2406.06484)). At
`rate=1` the key reads back `v` at once, a repeated correct observation has zero residual,
and the read at an earlier key `q` moves by `dot(q, k)` times the error: orthogonal records
are kept, overlapping ones interfere. A read is one linear product, with no softmax and no
settling. `FastSynapses(rule="delta")` and `SynapticMemory` scale keys to unit length. A
fixed matrix cannot hold contradictory values, nearly parallel keys or more independent
values than its key rank exactly; measure retention on the streams the store will see.

```python
import numpy as np
import cadence as cd


def write(M, key, value, rate=1.0):
    error = value - key @ M
    M += rate * np.outer(key, error)
    return error


M, q, p, k = np.zeros((4, 2)), np.array([0.6, 0.8, 0, 0]), np.eye(4)[2], np.eye(4)[0]
write(M, q, np.array([1.0, 0.0]))
write(M, p, np.array([0.0, 1.0]))
old_q = q @ M
error = write(M, k, np.array([0.0, 2.0]))                 # k overlaps q, is orthogonal to p
assert np.allclose(k @ M, [0.0, 2.0])                     # read back exactly
assert np.allclose(q @ M - old_q, (q @ k) * error)        # q moved by the overlap
assert np.allclose(p @ M, [0.0, 1.0])                     # p kept its record
assert np.allclose(write(M, k, np.array([0.0, 2.0])), 0)  # a repeat writes nothing
```

## A store or a settling learner

A delta-rule store suffices when the outcome follows its key (the consequence of an action,
the reward of a choice, a fact seen at a place), the key holds what decides it, and keys
that should generalise overlap. Completing a partial reading, carrying context across a
delay and crediting late reward need a settled region ([write a cortex](cortex.md)); the
[certificate](certificate.md) covers that settling and not the stores. A dense learner
moves every prediction with each update, which rehearsal of retained real observations
counters. Give a baseline the same replay policy, and never replay obsolete evidence.

## Reading and writing directly

`FastSynapses(pre, post, decay=1.0, rate=1.0, amplitude=1.0, normalize=False, replace=False,
rule="hebb", separator=None)` holds one `(pre, post)` matrix per stream between declared
neuron indices.

```python
memory = cd.FastSynapses(np.arange(4), np.arange(4, 6), rule="delta")
keys, values = np.eye(4)[:2], np.array([[1.0, 0.0], [0.0, 1.0]])   # two streams
memory.observe(keys, values)
assert np.allclose(memory.recall(keys), values)
memory.reset(2, rows=np.array([True, False]))          # only the first stream's episode ended

connectome = cd.Connectome.from_synapses(
    6, pre=[0, 1, 2, 3], post=[4, 4, 5, 5], count=[1] * 4,
    populations={"key": range(4), "value": (4, 5)},
)
graph = cd.NeuralGraph(connectome, cd.learning_neuron_model())
drive = np.zeros((2, connectome.n))
drive[:, :4] = keys
state = graph.settle_batch(memory.stimulate(drive))   # the read is added to the value neurons
memory.update(state, write=np.array([True, False]), post=np.array([[1.0, 0.0], [0.0, 0.0]]))
```

`observe(key, value, write=None)` takes `(batch, width)` ports; a boolean `write` gates each
stream, and every observe first decays all strengths by `decay` (reads never do).
`recall(key)` reads keys, `read(drive)` the key columns of a drive, and `stimulate(drive,
inplace=False, key=None)` adds the read of the drive's key columns, or of `key`, into the
post columns. `update(state, write=None,
post=None)` writes from the settled key activation with `post` as the observed value, and
only decays when `write` is `None`. Because `update` keys on activation and `stimulate` on
drive, a nonlinear activation can write under a key direction the read never uses; use
`observe`/`recall` when one external key serves both. Never write the store's own prediction back as evidence.

`rule="delta"` needs `rate` in `[0, 1]`, refuses `normalize` and `replace`, and a zero key
writes nothing. Scaling a key to unit length (`rule="delta"` or `normalize=True`) divides by
the norm of the whole key vector, the one step of the store that reads more than a synapse. The default `rule="hebb"` adds `rate * outer(k, v)`: `normalize=True` uses
unit keys and divides a read by the decayed write mass, and `replace=True` clears the rows of
a key's positive coordinates first, which suits one-hot slots and erases records under dense
keys. `amplitude` scales the read without changing the write target. An overflowing write
raises `ValueError` and changes nothing. `strength` holds the `(batch, key_width, post)`
matrices (`key_width` is the separator's `expansion`, else the pre count) and `writes`
counts rows written. `Learner.save` saves neither these stores nor the traces below; the
caller saves them.

`SynapticMemory(pre, post, decay=0.9, consolidation=0.05, separator=None, ...)` adds a
persistent `consolidated` matrix shared by all streams under the fast residual. Its rule,
salience and `value_mask` are in [the associative store](memory.md#associative-store-the-hippocampus).

## Streams and independent episodes

Rows of a batch are independent streams. Score a prediction before revealing its label.
On `FastSynapses`, `SynapticMemory`, every `Trace` and `History`, `reset(batch, rows=mask)`
clears the streams that ended and `keep(rows)` drops streams while the rest keep their
identities; `reset(batch)` without rows, or at a new batch size, clears all. A read at
another batch size returns the empty-stream baseline and leaves live records alone; a write
at another batch size starts fresh streams. `SynapticMemory.reset` clears only the fast
residuals and keeps the consolidated matrix, which is also its baseline; `clear()` erases both.

## Pattern separation

`PatternSeparator(inputs, expansion, winners, seed=0, center=0.0, fan_in=0, cover=0)` codes
keys before a store sees them: a fixed random projection onto `expansion` units, of which the
`winners` strongest positive entries stay. Codes with disjoint winners do not interfere. On
keys of cosine 0.9 a separated delta store revises at least 95 percent of its records
correctly, 50 points above the plain store ([separation tests](../tests/test_separation.py)).

```python
key_neurons, value_neurons = np.arange(32), np.arange(32, 40)
sep = cd.PatternSeparator(inputs=32, expansion=1024, winners=8, seed=0, center=0.99)
sep.habituate(np.random.default_rng(0).standard_normal((256, 32)))   # the environment's keys
separated = cd.FastSynapses(key_neurons, value_neurons, rule="delta", separator=sep)
separated.observe(np.ones((1, 32)), np.eye(8)[:1])
assert separated.key_width == 1024 and np.allclose(separated.recall(np.ones((1, 32))), np.eye(8)[:1])
```

`code(key, learn=False)` returns the `(batch, expansion)` code. With `center` above zero the
running `mean` is subtracted first, writes move it by forgetting factor `center`, and
`habituate(keys)` sets it from a sample before anything is stored; a moving mean can shift a
stored key's code. `fan_in` gives each code unit that many inputs (`index` and `weight` in
place of `projection`); `cover` (needs `fan_in`) reads each input through an ON and an OFF
channel with positive weights, each channel read by `cover` units. The projection is fixed:
it does not learn which differences matter. Storage and capacity grow to `expansion` rows
per stream. `SynapticMemory` refuses `center` other than 0, since a moving mean would move
persistent addresses; apply one fixed centering to inputs and queries instead.

## Traces

Fading traces carry the moment before into the next settle, one row per stream. Each reads
a `source` population and writes `amplitude` times the trace into the paired `target`
population through `stimulate(drive)`; decays lie in `[0, 1)`.

| Name | Defaults | Behaviour |
|---|---|---|
| `Trace(connectome, decay=0.5, amplitude=1.0, source="hidden", target="context")` | | `update(state)` moves the trace toward the source activation, `t <- decay t + (1 - decay) h`. `ringing(floor=0.1)` gives each source neuron's share of the trace, a salience for `ActorCritic.salience`. |
| `Efference(...)` | `source="motor"`, `target="efference"` | `issue(command)` decays toward the `(batch, motor)` one-hot of the executed action; `update` raises. |
| `History(connectome, decays, amplitude=3.0, source="sensory", target="history")` | | One target block per decay; `hear(observations)` moves block `d` by `h = d h + (1 - d) x`. |
| `cadence.stream.columns(index)` | | A slice for a contiguous range, else the index array; slices avoid gather copies on wide batches. |

Inside a composed brain the working trace, efference copy and heard history are owned by
[Memory](memory.md#working-trace).
