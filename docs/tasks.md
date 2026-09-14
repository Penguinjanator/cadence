# Task recipes

Start with the operation your task requires. These recipes describe interfaces;
the linked public examples carry their own controls, settings, and source-bound
results. Dataset encoding and the evaluation split belong to your application.

| Task | Input | Operation and output | Example or guide |
|---|---|---|---|
| Known interacting constraints | A drive and a declared connectome | Settle; read output activations and residual | [Circuit quickstart](quickstart.md), [C. elegans circuit](https://github.com/muellerberndt/cadence-examples/tree/main/worm) |
| Revise an addressed record | Key and observed value | `FastSynapses.observe`, then `recall` | [Memory](memory.md) |
| Classification | Feature values on input neurons | `Learner.step(drive, labels)`; `predict` returns class indices | [Label-fitting quickstart](quickstart.md#learn-a-response) |
| Imitation | An observation and a teacher's action | Classification over actions, with legal-action masking at deployment | [Learning recipes](learning.md), [learning life](patterns.md#a-learning-life) |
| Regression or reconstruction | Features and an output pattern | Quadratic nudge; read continuous output activations | [Pattern targets below](#pattern-targets) |
| A continuing stream | Each observation before its label arrives | Predict, score, then update; retain history explicitly when needed | [Memory](memory.md), [body loop](patterns.md#sensor-opposing-motors-body) |
| Reward-driven action | Observation, chosen action, reward | Weight action-target nudges by advantage, or use eligibility traces | [Signed feedback](patterns.md#signed-feedback), [reward](reward.md) |
| A measured synapse list | Supplied topology and declared stimuli | Settle and score held-out predicates with controls | [Protocols](protocols.md) |

## Put features on input neurons

A batched drive has one column per neuron, including hidden and output neurons.
Continue with a learner created in the [quickstart](quickstart.md#learn-a-response):

```python
x = np.array([[1.0, 0.0], [0.0, 1.0]])
levels = np.zeros((len(x), learner.brain.connectome.n))
levels[:, list(learner.brain.connectome.populations["input"])] = x
drive = learner.brain.stimulus_levels(levels)
```

`stimulus_levels` multiplies by the neuron model's `stimulus_amplitude`; it does not pad
or clip the input. Fit feature scaling, category dictionaries, and missing-value
handling on training data. For a scalar with a wide range, a few localized bumps
over training-fitted bin centers are one possible representation. Compare that
encoding against simpler scaling using validation data.

## Pattern targets

`step` accepts integer class indices. For continuous or multi-label patterns,
construct targets with one column per neuron and use the phase operations directly:

```python
import numpy as np
import cadence as cd

connectome = cd.layered(2, 8, 2, density=1.0, seed=0)
learner = cd.Learner(
    cd.Brain(connectome, cd.learning_neuron_model()),
    connectome.populations["output"],
    cd.LearnerConfig(nudge="quadratic"),
)
drive = np.zeros((2, connectome.n))
drive[:, list(connectome.populations["input"])] = np.eye(2)
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
state; `Trace` carries fading activity; `FastSynapses` carries observations.
These have different memory contracts. Settling to a fixed point alone does not
guarantee recall of earlier inputs.

Rows of a batch represent independent streams. If a stream ends, clear its
episodic state. For `FastSynapses` and `Trace`, `reset(batch, rows=...)` resets
selected rows and `keep(rows)` retains a subset without changing identities.
Changing the batch size through an ordinary read or update can initialize fresh
fast state; use the explicit stream operations when records must survive.

## Several learners in one net

Use separate `Learner` objects on one brain with `plastic_synapses` and
`plastic_neurons` masks to restrict updates and decay. Each learner owns its
optimizer history; manually swapping just its momentum arrays is insufficient
when normalization or step counters also matter.

An update replaces `learner.brain` with a `Brain` carrying the updated parameters.
If two learners should operate on one evolving parameter set, assign the updated
brain to the other learner's `brain` before its next phase, while preserving that
learner's masks and history. Check checkpoint and reset behaviour for the complete
composition.
