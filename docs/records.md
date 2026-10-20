# Standalone memory stores

These stores work next to a `NeuralGraph` or a record patch without `Brain.compose`: the
records cortex `cd.Records`, `cd.FastSynapses` with its `cd.PatternSeparator`,
`cd.SynapticMemory` used directly, and the fading traces of `cadence.stream`. Each writes by
one delta-rule correction or a fixed decay, and none settles. The memories of a composed
brain are in [Memory](memory.md); signatures are in the [API reference](api.md#streams-cadencestream).

## One correction

For a unit key `k`, value `v` and matrix `M`, a write is `M += rate * outer(k, v - k @ M)`
(normalized LMS, close to [delta-rule fast weights](https://arxiv.org/abs/2406.06484)). At
`rate=1` the key reads back `v` at once, a repeated correct observation has zero residual,
and the read at an earlier key `q` moves by `dot(q, k)` times the error: orthogonal records
are kept, overlapping ones interfere. A read is one linear product, with no softmax and no
settling. `FastSynapses(rule="delta")` and `SynapticMemory` scale keys to unit length;
`Records` uses its unit-length sparse code as the key. A fixed matrix cannot hold
contradictory values, nearly parallel keys or more independent values than its key rank
exactly; measure retention on the streams the store will see.

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

## Records

`cd.Records(inputs, fields)` keeps one table per predicted field and one record per cell. A
reading is coded in three steps: each unit's running mean is subtracted, a fixed random
projection plus fixed offsets drives `cells` cells, and the `active` strongest keep their
rectified drive, scaled to unit length. This expansion is a fixed sparse code. A read sums the records of the
active cells weighted by their activity; a write moves exactly those records by the error.

```python
records = cd.Records(6, {"next": 3, "reward": 1}, valued=["reward"], cells=4000, active=40)
reading = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])    # one of three places, one of three actions
code = records.code(reading, adapt=True)[:, 0]        # a witnessed reading, shape (2, cells)
records.write(code, {"next": np.array([0.0, 1.0, 0.0]), "reward": np.array([1.0])})
after = records.read(code)
print(after["next"], after["reward"])                 # [0.  0.2 0. ] [1.]

other = records.code(np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]))[:, 0]   # another place
moved = 0.2 * (code[0] @ other[0]) * np.array([0.0, 1.0, 0.0])          # rate x overlap x error
assert np.allclose(records.read(other)["next"], moved)

twin = cd.Records(**records.to_dict())                # the same fixed cells, rebuilt from the seed
twin.load_state(records.state())                      # and the same learned state
assert np.allclose(twin.read(code)["next"], after["next"])
```

`code(readings, *, adapt=False, valued=True)` returns `(2, batch, cells)`: the plain code,
then the valued code. `write(code, targets, known=None)` takes one `(2, cells)` code, writes
the fields named in `targets` and returns their count; `known` masks the observed entries of
a target, and the others keep their records. `write_batch(codes, targets)` takes every error
against the tables as they stood at the call and moves each cell by the mean of its writers'
moves, so agreeing writers do not overshoot. Invalid input leaves every table unchanged.

| Setting | Default | What it does |
|---|---|---|
| `cells`, `active` | 8000, 40 | Code width and cells kept per reading. Fewer active cells mean less sharing between readings. |
| `rate` | 0.2 | Write rate of the consequence fields, in `[0, 2]`: a regularity is averaged over many outcomes. |
| `valued`, `valued_rate` | `()`, 1.0 | Fields read and written through the valued code; at 1.0 one exposure writes the target. Without `pathways` or `tasks` the valued code is the plain code. |
| `habituation` | `1e-5` | Slowest rate of the running mean. With `adapt=True` the mean moves at `max(habituation, 1 / seen)`; 0 codes raw readings. With the mean subtracted, a unit constant across readings adds nothing and a small cue moves the code. |
| `bias` | 0.3 | Scale of the fixed offsets, in drive units. A reading far below unit variance per unit lets the offsets pick the same winners every time. |
| `pathways`, `pathway_rate` | `()`, 0.002 | Index arrays into the reading. The valued code divides each pathway by its running norm (`pathway_norm`), so a one-unit goal has a say comparable to a dense observation. |
| `tasks` | `()` | Indices of one-hot task units. Cells split into one group per task (`task_of_cell`, each of at least `active` cells); the valued code draws winners from the group of the largest task unit, or from every cell when none is positive, so a value learned for one task survives the rewards of the next. |
| `fan_in` | 0 | Each cell reads that many of the pathways (needs `pathways`, at most their count) plus every input outside them, so a consequence learned at one place generalises across what the cell does not read. |
| `averaging` | `False` | Each consequence cell's rate becomes the larger of `rate` and one over the code mass written into it, at most 1: a fresh cell takes its first outcome whole, a used one averages. |
| `homeostasis` | 0.0 | Applied by `witness` only (`code(adapt=True)` skips it): tracks each cell's activation share (`usage`) and moves its offset (`boost`, within three times `drive_scale`) so the share approaches `active / cells` and no cell becomes a hub. Every code adds `boost`. |
| `seed` | 0 | Starts the `Mulberry32` generator that draws the projection, then the offsets, the task groups and each cell's pathways ([export of a trained store](record-patch.md)). |

Witnessed readings adapt; code an imagined one (a candidate action, a predicted next
observation) with `adapt=False` and the same missing flags as the witnessed readings it
stands for, or its read goes through cells nobody wrote. `valued=False` leaves the valued
code NaN when only consequence fields are read. `witness(readings)` applies the adaptation
without coding, plus the homeostasis step, for a caller that codes after the outcome is
known. Keep components that vary without deciding the outcome, such as a context trace, out
of the reading, and give each unit of a concatenated reading unit variance
([diagnosing a record store](record-patch.md#diagnosing-a-record-store)).

A read is one product of the code with each table and a write one outer-product step per
field, on NumPy. `tables` maps each field to its `(cells, width)` records and `parameters()`
counts their entries; `projection` and `offset` are fixed and rebuilt from `to_dict()`.
`state()` returns the learned arrays (`table_<field>`, `mean`, `seen`, `pathway_norm`,
`writes`, `count`, `usage`, `boost`, `drive_scale`);
`load_state` replaces them atomically and refuses any other key set or wrong shapes.

## Records or a settling learner

Records suffice when the outcome follows its reading (the consequence of an action, the
reward of a choice, a fact seen at a place), the reading holds what decides it, and readings
that should generalise share cells. They learn from one correlated stream in its own order
([records tests](../tests/test_records.py)). Completing a partial reading, carrying context
across a delay and crediting late reward need a settled region ([write a cortex](cortex.md));
the [certificate](certificate.md) covers that settling and not the records. A dense learner
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
| `Trace(connectome, decay=0.5, amplitude=1.0, focus=0.0, source="hidden", target="context")` | | `update(state)` moves the trace toward the source activation, `t <- decay t + (1 - decay) h`; `focus` above zero weights each neuron by its movement since the last moment. `ringing(floor=0.1)` gives each source neuron's share of the trace, a salience for `ActorCritic.salience`. |
| `Echo(...)` | as `Trace` | The trace of the hidden equilibria into `context`. |
| `Afterglow(...)` | `focus=1.0`, `target="afterglow"` | The focused trace; with `source="input"` an afterimage that reads a cue against a static background. |
| `Efference(...)` | `source="motor"`, `target="efference"` | `issue(command)` decays toward the `(batch, motor)` one-hot of the executed action; `update` raises. |
| `History(connectome, decays, amplitude=3.0, source="sensory", target="history")` | | One target block per decay; `hear(observations)` moves block `d` by `h = d h + (1 - d) x`. |
| `stateful(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0, context_init=1.0)` | | Returns `(connectome, tie_groups)`: an embedded network plus a `context` range for an `Echo`. |
| `cadence.stream.columns(index)` | | A slice for a contiguous range, else the index array; slices avoid gather copies on wide batches. |

Inside a composed brain the working trace, efference copy and heard history are owned by
[Memory](memory.md#working-trace).
