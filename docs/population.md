# Many brains on one device

<!-- requires: torch -->

`cadence.population.PopulationPatch` runs a population of [record patches](record-patch.md)
as batched torch tensors, for evolution or game populations on a GPU or MPS (torch setup in
[Backends](backends.md)). Each instance is a brain with its own parameters; each stream is a
world it lives in, with its own record store and statistics.

```python
import torch
from cadence import RecordPatchNet
from cadence.population import PopulationPatch

net = RecordPatchNet(20, 16, 3, groups=(3,), cells=512, active=8, slowest=2.0)
pop = PopulationPatch.from_patch(net, instances=8, streams=32)  # 8 brains, each in 32 worlds
x = torch.rand(8, 32, 20, device=pop.dev)                     # one reading per brain and world
out = pop.imagine(x)                                           # out["output"]: (8, 32, 3)
labels = torch.randint(0, 3, (8, 32), device=pop.dev)
target = torch.nn.functional.one_hot(labels, 3).to(pop.dev, pop.dtype)
pop.observe(x, target, rate=0.3)          # each brain its own slow step, each world its own write
fitness = -pop.loss(pop.imagine(x)["slow"], target)            # (8,), higher is better
pop.inherit(fitness.argsort(descending=True)[:4].repeat_interleave(2), sigma=0.1)
```

## Build a population

`PopulationPatch(inputs, hidden, outputs, *, instances, streams, seed=0, cells=4096, active=32, record_rate=0.5, habituation=1e-5, record_bias=0.3, slowest=2.0, groups=None, device=None, dtype=torch.float32)`
builds one fresh `RecordPatchNet` from the shared arguments (meanings in
[record patch](record-patch.md)) and copies it into every instance. `slowest` defaults to 2.0
(128.0 in the NumPy patch); use float64 on the cpu for parity checks. `groups` is `None` (a
linear readout) or `(outputs,)` (one softmax), else `ValueError`. `device=None` picks
`cadence.population.device_of()`: cuda, then mps, then cpu; a name such as `"cpu"` is used as given.

`PopulationPatch.from_patch(net, *, instances, streams, device=None, dtype=torch.float32)`
copies a NumPy patch (parameters, projection, offsets, statistics, tables) into every instance
and stream: a schooled patch becomes a population. It raises `ValueError` unless
`record_writes="sequential"`, `record_width=None` and `record_averaging=False`. The population
carries no structured `port`, `output_precision` or `record_homeostasis`, and its only grouping
is one softmax over every output; school the patch without them.

`inherit(parents, *, sigma=0.0, generator=None)` makes instance `p` a copy of instance
`parents[p]`, Adam moments included. With `sigma > 0` each slow tensor takes a normal step of
`sigma` times its spread (at least 1e-3), drawn on the cpu (`generator` is a cpu
`torch.Generator`). Stores and statistics restart. `clear_records()` empties tables and write
counts only.

## What batches, and what does not

Instances, streams and, through several `PopulationPatch` objects, the patches of one brain
batch, because nothing crosses them: each instance's slow step is the adjoint of its own
one-moment loss, each stream's records take their own residual, and no gradient reaches a
store or another patch. Every reading settles once from rest, as a decision or a school batch
reads a patch. The moments of one stream do not batch: this class carries no context between
moments, and a temporal patch settles them in order.

`imagine(inputs, *, stream_of=None)` reads `(instances, streams, inputs)` and returns `output`
(slow readout plus record read), `slow`, `read`, `hidden`, the dense `code` and `familiarity`,
with no learning. `observe(inputs, target, *, rate=1.0, write=True, mask=None, stream_of=None,
weight=None)` steps each instance by `rate` (a float or an `(instances,)` tensor), writes each
stream's residual when `write`, and returns the per-instance `loss`, `slow`, `hidden`,
`residual` and `code`. `stream_loss(slow, target, weight=None)` is the slow loss per stream
(half the mean squared error, or cross-entropy with `groups`); `loss(slow, target)` is its mean
per instance. `settles` and `writes` count calls. `parameters()` returns the slow tensors;
`state()` returns them with tables, means, `seen` and input norms as NumPy arrays, plus the
counters. `code(readings)` is the dense code, `read(code)` its record read, and
`familiarity(indices, values, stream_of=None)` takes a code as its active cells.

## Options for wide readings

A game population reads wide and lives through episodes of unequal length. The application
keeps the lesson queues; these options teach them in one call.

| Option | Shape | Effect |
| --- | --- | --- |
| `stream_of` on `imagine`/`observe` | `(moments,)` or `(instances, moments)` | Names each moment's stream, so several moments per stream settle or learn in one call. The writes of one stream land against the store as it stood, as the NumPy patch's `write_batch`. |
| `mask` on `observe` | `(instances, moments)` | Only masked moments learn, write and move statistics; an instance's slow loss is the mean over its masked moments. A masked-out target may be NaN. With `"sgd"` an instance with no masked moment keeps its parameters; under Adam its momentum moves it unless its `rate` is zero. |
| `weight` on `observe` | `(outputs,)` or `(instances, outputs)`, summing to `outputs` | Reweights the squared errors of a linear readout, so a few ports that matter count by their weight. |
| `record_weight` attribute | `(instances, outputs)` or `None` | Scales each output's record write; damp an output whose target is a sample rather than a fact (a payoff). |
| `familiarity` from `imagine` | `(instances, moments)` in [0, 1] | The share of a reading's code on cells this stream's store has written: whether this brain has been near this reading in this world. |
| `optimizer` attribute | `"sgd"` (default) or `"adam"` | Adam (betas 0.9 and 0.999) scales the slow step by the adjoint's running moments per instance and parameter, and `rate` is the learning rate. Use it when a fixed-rate step barely moves wide weights. |

```python
lin = PopulationPatch(12, 8, 4, instances=2, streams=3, cells=64, active=4)
lin.optimizer = "adam"
lin.record_weight = torch.tensor([[1, 1, 1, 0.2]] * 2, device=lin.dev)  # output 3: a payoff
moments = torch.rand(2, 5, 12, device=lin.dev)              # five queued moments per brain
stream_of = torch.tensor([0, 0, 1, 2, 2], device=lin.dev)   # the stream of each moment
mask = torch.tensor([[1, 1, 1, 0, 1], [1, 0, 0, 0, 0]], device=lin.dev)
seen = lin.imagine(moments, stream_of=stream_of)["familiarity"]  # (2, 5)
lin.observe(moments, torch.rand(2, 5, 4, device=lin.dev), rate=0.01, mask=mask,
            stream_of=stream_of, weight=torch.tensor([1.0, 1.0, 0.5, 1.5], device=lin.dev))
```

## Parity, atomic observe and memory size

`tests/test_population.py` holds the population to the NumPy patch on the cpu in float64 to
1e-9 (prediction, slow step, write and read); `tests/test_population_audit.py` matches
`stream_of` writes to the unfolded path.
[`benchmarks/population/throughput.py`](../benchmarks/population/README.md) times one imagined
and one observed moment per brain and stream, one row per instances x streams.

`observe` stages every change and commits none on a bad moment: nonfinite inputs, rate,
`weight`, `record_weight` or unmasked targets, or a write or slow step that would leave a
nonfinite table, statistic, parameter or Adam moment, raise `ValueError` naming the check. A
write whose rows could overflow is summed on a copy first.

The record read and write touch only the `active` rows of each stream's table, so the store
sets memory: `instances * streams * cells * outputs` table values, `instances * streams * cells`
write counts and `instances * streams * (inputs + hidden)` running means. The code is a product
with the `(inputs + hidden) x cells` projection, returned dense. `memory_bytes()` counts
parameters, tables and means. Keep stores small and brains many.

## Several devices

Instances share nothing, so a population larger than one device splits into one
`PopulationPatch` per device, with only parameters (from `state()`) moving between generations.
Size each device's share by `memory_bytes()`.
