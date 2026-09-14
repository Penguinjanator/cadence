# Task recipes

Start with the operation your task requires. These recipes describe interfaces;
the linked public examples carry their own controls, settings, and source-bound
results. Dataset encoding and the evaluation split belong to your application.

| Task | Input | Operation and output | Example or guide |
|---|---|---|---|
| Known interacting constraints | A drive and a declared wiring | Settle; read output activations and residual | [Circuit quickstart](quickstart.md), [interventions](https://github.com/muellerberndt/cadence-examples/tree/main/06_interventions) |
| Revise an addressed record | Key and observed value | `FastSeams.observe`, then `recall` | [Memory](memory.md) |
| Classification | Feature values on input owners | `Learner.step(drive, labels)`; `predict` returns class indices | [Digits](https://github.com/muellerberndt/cadence-examples/tree/main/01_digits) |
| Imitation | An observation and a teacher's action | Classification over actions, with legal-action masking at deployment | [Connect Four](https://github.com/muellerberndt/cadence-examples/tree/main/03_connect_four) |
| Regression or reconstruction | Features and an output pattern | Quadratic nudge; read continuous output activations | [Pattern targets below](#pattern-targets) |
| A continuing stream | Each observation before its label arrives | Predict, score, then update; retain history explicitly when needed | [Memory](memory.md), [embodiment](embodied.md) |
| Reward-driven action | Observation, chosen action, reward | Weight action-target nudges by advantage, or use eligibility traces | [Games](games.md), [reward](reward.md) |
| A measured edge list | Supplied topology and declared stimuli | Settle and score held-out predicates with controls | [Protocols](protocols.md) |

## Put features on input owners

A batched drive has one column per owner, including hidden and output owners.
Continue with a learner created in the [quickstart](quickstart.md#learn-a-response):

```python
x = np.array([[1.0, 0.0], [0.0, 1.0]])
levels = np.zeros((len(x), learner.engine.wiring.n))
levels[:, list(learner.engine.wiring.sets["input"])] = x
drive = learner.engine.clamp_levels(levels)
```

`clamp_levels` multiplies by the rule's clamp amplitude; it does not pad or
clip the input. Fit feature scaling, category dictionaries, and missing-value
handling on training data. For a scalar with a wide range, a few localized bumps
over training-fitted bin centers are one possible representation. Compare that
encoding against simpler scaling using validation data.

## Pattern targets

`step` accepts integer class indices. For continuous or multi-label patterns,
construct full-owner targets and use the phase operations directly:

```python
import numpy as np
import cadence as cd

wiring = cd.layered(2, 8, 2, density=1.0, seed=0)
learner = cd.Learner(
    cd.Settlement(wiring, cd.learning_rule()),
    wiring.sets["output"],
    cd.LearnerConfig(nudge="quadratic"),
)
drive = np.zeros((2, wiring.n))
drive[:, list(wiring.sets["input"])] = np.eye(2)
target = np.zeros_like(drive)
target[:, learner.output_index] = np.array([[0.2, 0.8], [0.8, 0.2]])

free = learner.free(drive)
plus = learner.nudged(drive, free, target)
minus = learner.nudged(drive, free, target, sign=-1.0)
learner.update(free, plus, minus)
prediction = learner.free(drive).activation[:, learner.output_index]
print(prediction.shape)  # (2, 2)
```

This performs one update; it does not establish fit quality. Choose target scaling
within the activation's useful range using training data. Select thresholds for
multi-label outputs or calibrate continuous readouts on validation data, then
hold those choices fixed on test data. `predict` is an argmax classifier; read
activations directly for these pattern tasks.

## Streams and independent episodes

Score a prediction before revealing its label. Keep the order of observations,
actions, and rewards explicit. `Learner.free(..., warm=state)` carries transient
state; `Trace` carries fading activity; `FastSeams` carries observations.
These have different memory contracts. A fixed-point solver alone does not
guarantee recall of earlier inputs.

Rows of a batch represent independent streams. If a stream ends, clear its
episodic state. For `FastSeams` and `Trace`, `reset(batch, rows=...)` resets
selected rows and `keep(rows)` retains a subset without changing identities.
Changing the batch size through an ordinary read or update can initialize fresh
fast state; use the explicit stream operations when records must survive.

## A game player's learning life

Start by designing the state ranges and connections for the task, with enough
capacity to fit a held-out portion of the teacher's demonstrations. A temporal
task should receive the current observation and carry its own `Trace` state;
a fully visible board need not carry an extra temporal copy.

Use this sequence for the game examples:

1. Fit teacher demonstrations with `Learner.step(drive, target)`.
2. Play and experiment. An action and its reward-derived advantage can use
   `Learner.step(drive, action, weight=advantage)`; `ActorCritic` supplies a
   continuing reward learner with eligibility and a critic.
3. Inspect recurring failures and obtain additional teacher demonstrations on
   those states. Mix them with earlier examples to retain prior skills.
4. Keep collecting experience in later games and save the learned parameters
   with `Learner.save`. Retain a reusable episode record and rehearsal examples.

Evaluate candidate updates on separate validation episodes, then measure the
selected checkpoint on untouched test episodes. Every game supplies experience;
not every update improves performance. Report rejected updates as well as gains.
During reward replay, use the observation and trace available when the action
was taken. Do not advance the live trace with shuffled training rows or targets.
Reset transient state between episodes while preserving learned parameters.
`Learner.save` does not save separately owned traces or experience buffers.

## Several learners in one net

Use separate `Learner` objects with `trainable_overlaps` and
`trainable_owners` masks to restrict updates and decay. Each learner owns its
optimizer history; manually swapping just its momentum arrays is insufficient
when normalization or step counters also matter.

A learner replaces its engine when updating parameters. If two learners should
operate on one evolving parameter set, explicitly pass the updated engine to
the other before its next phase, while preserving that learner's masks and
history. Check checkpoint and reset behaviour for the complete composition.
