# Build a brain, teach it a routine, keep it

This tutorial uses the experimental population solver
`cadence.experimental.equilibrium`. Its scope and its relation to the default
`Brain.compose` brain are described on the [solver landing page](index.md).
The Python blocks on this page share one namespace and run in order.

The solver needs Python 3.11 or later and runs on the standard library. The
package also installs NumPy; optional PyTorch execution needs the `[accel]`
extra ([execution and cost](ACCELERATION.md)).

```sh
python -m pip install "cadence-net==0.84.3"
# or, from a checkout of this version:
python -m pip install -e .
```

## 1. Build the smallest brain and teach it

The task is a toy calibration: an instrument moves by `movement = 0.6 * command`.
Training supplies measured pairs. Later queries supply only the command, so a
correct answer has to come from the learned relation.

A `Cortex` describes the wiring and `build()` creates the persistent `Brain`.
The smallest brain has two populations: `features` reads the sensor and
`response` reads the live states of `features`. Their patches settle against
each other, and the shared equilibrium is the answer.

```python
from cadence.experimental.equilibrium import Brain, Cortex, bootstrap

layout = Cortex(seed=2)
command = layout.input("command", shape=1)
features = layout.column("features", patches=4, inputs=command)
response = layout.column("response", patches=1, inputs=features)
layout.output("movement", shape=1, reads=response)
brain = layout.build()

def pairs(*xs):
    return [({"command": [x]}, {"movement": [0.6 * x]}) for x in xs]

examples, checks = pairs(-0.8, -0.4, 0.4, 0.8), pairs(-0.6, 0.6)
report = bootstrap(brain, examples, checks=checks, epochs=20, batch_size=4,
                   max_error=0.1, seed=2)
assert report["passed"], report
assert report["updates"] > 0
```

`shape` is the shape of the supplied data and `patches` counts processing
states. An output exposes selected patch states; there is no separate readout
network.

`bootstrap` replays the examples through the normal learning calls, for at
most `epochs` passes, until every example and check answers within `max_error`
(in output units) with its targets absent. `batch_size=4` groups each epoch
into `observe_batch` calls; the default `1` uses `observe`. The solver's
`tolerance` measures numerical settlement, a separate quantity: a settled brain
can answer wrongly. Readiness and retention checks are in
[design and teaching](BRAIN_DESIGN.md).

## 2. Test answers the teacher has not supplied

The readiness checks decide when training stops, so the final test uses other
values and no output targets:

```python
saved = brain.snapshot()
for x in (-0.7, -0.2, 0.2, 0.7):
    result = brain.settle({"command": [x]})
    assert result["qualified"], result["reason"]
    assert abs(result["outputs"]["movement"][0] - 0.6 * x) < 0.1
assert brain.snapshot() == saved
```

`settle` returns outputs with diagnostics such as `qualified`, `reason` and
`work`. Use an answer only when it qualified. `predict` returns just the outputs
and raises `SettlementError` when qualification fails. Neither call changes the
brain.

## 3. Save it and use it live

```python
resumed = Brain.from_snapshot(saved)
assert resumed.snapshot() == saved
assert resumed.predict({"command": [0.5]}) == brain.predict({"command": [0.5]})

activity = resumed.step({"command": [0.3]})
assert activity["accepted"]
assert abs(activity["outputs"]["movement"][0] - 0.18) < 0.1

# After the instrument executes the command, its measured outcome teaches.
update = resumed.observe({"command": [0.3]}, {"movement": [0.18]})
assert update["accepted"]
```

`snapshot()` returns JSON text holding the layout, configuration, parameters,
live activity, event history and hashes of the implementation sources. Store it
with the package version, units, preprocessing and any body or history state
kept outside the brain ([checkpoint reference](REFERENCE.md#checkpoints)).

| Operation | Retained on success |
| --- | --- |
| `settle` / `predict` | Nothing; a pure query |
| `step` | Qualified live activity; parameters stay fixed |
| `observe` | Live activity and parameters taught by the supplied targets |
| `observe_batch` | Shared parameters learned from the batch; live activity stays fixed |
| Any refused call | Nothing; live state, parameters and event history are unchanged |

An accepted update can worsen another skill, so recheck free answers after
learning. Exact result fields and retry identities are in the
[API reference](REFERENCE.md).

## 4. Add an intermediate population, and why an isolated population is refused

A harder relation can use more populations. They all settle together, so the
later relation influences earlier states through the shared energy. The named
inputs and outputs stay the same.

```python
cortex = Cortex(seed=2)
command = cortex.input("command", shape=1)
representation = cortex.column("representation", patches=4, inputs=command)
integration = cortex.column("integration", patches=2, inputs=representation)
response = cortex.column("response", patches=1, inputs=integration)
cortex.output("movement", shape=1, reads=response)
candidate = cortex.build()

learned = bootstrap(candidate, examples, checks=checks, epochs=20,
                    batch_size=4, max_error=0.1, seed=2)
assert learned["passed"], learned
assert abs(candidate.predict({"command": [0.5]})["movement"][0] - 0.3) < 0.1
```

Both layouts learn with the same patch rule. The two-population brain solves
this calibration; add a population when its measured quality gain justifies
the extra coupling work.

A population that reads only sensors and is read by no other population would
settle alone, outside the brain's equilibrium. `build()` refuses it:

```python
alone = Cortex(seed=2)
sensor = alone.input("command", shape=1)
patch = alone.column("response", patches=1, inputs=sensor)
alone.output("movement", shape=1, reads=patch)
try:
    alone.build()
except ValueError as error:
    assert "settles with no other population" in str(error)
else:
    raise AssertionError("an isolated population must be refused")
```

A cluster of populations that no read connects to the rest raises a
`ValueError` too. Wider branches, observer wiring and the cost of connectivity
are in [design and teaching](BRAIN_DESIGN.md#connectivity-and-cost).

## Continue with a real task

[Design and teaching](BRAIN_DESIGN.md) covers body information, wiring,
teaching rows, retention and diagnosis; [live use](LIVE.md) covers delayed
outcomes, `History` and reward learning with `Reinforcement`; the
[specification](SPECIFICATION.md) states the mathematics.

[layout_learning.py](../../examples/equilibrium/layout_learning.py) teaches,
queries, saves and resumes the relation `answer = 0.6 * signal` on several
layouts, runs one `step` and one `observe`, and prints a JSON list with one
summary per layout: patches, connections, updates, the largest error on four
fresh inputs and the work of each call.

```sh
python examples/equilibrium/layout_learning.py
python examples/equilibrium/layout_learning.py --layout all --seed 5
```

| Option | Values | Meaning |
| --- | --- | --- |
| `--layout` | `small` (default) | Two populations, as in section 1 |
| | `deep` | An added intermediate population, as in section 4 |
| | `recursive` | Observers that read the previous population's states and current prediction errors ([observer settlement](BRAIN_DESIGN.md#experimental-recursive-observer-settlement)) |
| | `all` | The three layouts in turn |
| `--seed` | integer, default `2` | Layout seed passed to `Cortex(seed=...)`; the teaching seed is fixed at `2` |
