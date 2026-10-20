# Start a continuing brain

Install Cadence as described in [the documentation's install section](README.md#install),
then build one **System 1** brain with `Brain.compose` and run it with `brain.live`.

```python
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, arousal=True)
action = brain.live([[1.0, 0.0, 0.0, 0.0]])
```

By default `compose` builds one association region (`modules=(64,)`) between the
sensory and motor neurons, a working trace that carries the last interpretation into
the next settle, and an associative memory that consolidates what it writes. Every
region settles together into one equilibrium, and that equilibrium is the answer.
The [world-model guide](world-model.md) describes the life this brain is meant to
live: bootstrap an interpretation, use it, repair witnessed failures and continue.

## Observe, act and learn

```python
import numpy as np

observation = np.array([[1.0, 0.0, 0.0, 0.0]])

# Execute the choice in a tiny environment: action 0 earns one unit.
reward = (action == 0).astype(float)
next_observation = np.array([[0.0, 1.0, 0.0, 0.0]])
action = brain.live(next_observation, reward=reward)
assert action.shape == (1,)
assert brain.learner.updates > 0
```

`live` follows one continuing stream. Pass one observation row, execute the returned
action index, and report that action's actual outcome as `reward` with the next
observation. A young brain (its first 100 moments, `arousal_youth`) or an aroused one
samples its action, learns from the outcome and writes memory. A calm brain answers
greedily from one settle and learns nothing.
Call `reset()` before starting a different stream. `brain.last_arousal` holds the
readings of the latest moment.

For batches, teaching labels or learning from every outcome, use
[`step`](continuous.md#explicit-learning-with-step). Named overrides such as
`actor_eta=0.1` change a rate at construction, and `brain.retune(...)` changes
selected settings without losing acquired state. `brain.describe()` and
[the defaults table](brain.md#defaults-and-expert-overrides) show what is running.

## Inspect the work of answering

```python
report = brain.last_settlement
assert report is not None and report["qualified"]
assert report["max_residual"] <= report["tolerance"]
print("free-answer sweeps:", report["steps"], "residual:", report["max_residual"])
```

`last_settlement` is a read-only report of the latest free-answer solve, refused
attempts included: sweeps, residual, residual checks and damping halvings. It leaves
out reward eligibility, teaching and memory work. A small residual means the network
agrees with itself; whether the action was good is measured by the task's outcomes.
The [continuing example](../examples/continuing_brain.py) tracks these counters
through bootstrap, unchanged conditions, disruption and correction.

## Imagine privately and resume

```python
phases = brain.imagine([observation, next_observation])
assert len(phases) == 2 and all(np.all(phase.qualified) for phase in phases)

brain.save("brain.npz")
resumed = Brain.load("brain.npz")

# Resume the same pending action with the same measured outcome.
reward = (action == 0).astype(float)
continued = brain.live(observation, reward=reward)
replayed = resumed.live(observation, reward=reward)
assert np.array_equal(continued, replayed)
assert brain.arousal.to_dict() == resumed.arousal.to_dict()
```

`imagine` settles the brain's responses to a supplied sequence of observations
along a private trace. Live memory, parameters, random state and pending feedback
are unchanged. A refused phase ends the branch and stays in the result, so check
each phase's `qualified` flags before using a response. `Brain.compose` builds no
transition model; learned consequences and planning are in [planning.md](planning.md).

`save` and `load` keep the brain's full continuation, pending feedback included.
Save the environment separately and resume its stream as well. If `live` refuses
an answer, the [qualification rules](continuous.md#qualification-and-refusal) say
how to retry without reporting an outcome twice.

## Add optional observers

**System 2** adds observer regions to the same graph:

```python
recursive = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), observers=(8,), seed=7,
    arousal=True,
)
assert recursive.live(observation).shape == (1,)
```

Observers read from and return influence to the processing regions, the motor
region and earlier observers, inside the same settlement and the same `live` and
`step` calls. They attach to any module count; this project adds them to a developed
two-module System 1 by convention. Any task advantage they bring has to be learned
and measured. [Recursive observers](recursive-settlement.md) covers depth and widths.

## Specialist guides

- Run the brain: [continuous interaction](continuous.md) and
  [compose a brain](brain.md) for custom regions, wiring and checkpoints.
- Memory: [every memory the brain has, and which to use when](memory.md).
- Learning and reward: [the free/nudged rule](learning.md) and
  [learning from reward](reward.md).
- Build and wire: [write a cortex](cortex.md) and [evolve a brain](evolution.md).
- Other model families: [temporal models](temporal.md), [planning](planning.md),
  [response protection](temporal-memory.md), [record patches](record-patch.md) and
  [reciprocal patches](patchnet.md).
- Evaluate: [evaluate a brain](evaluation.md).
- Reference: [API reference](api.md).
- Population solver: [exact state-and-error readback](equilibrium/index.md).
