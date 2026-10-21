# The learned code

`Brain.compose(..., learned_code=units)` adds a `code` region of that many neurons between the
sensory region and the first module. The region settles with the rest of the brain to a sparse
code of the observation, the cortex reads the code in place of the observation, the memories
key on it, and the region learns at every accepted moment by local rules. The founder
`learned_code=None` builds the brain without it and keys the memories on
[the observation code](memory.md#the-observation-code) as before; `code=units`, the fixed
random code, is the control at matched width.

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=53, actions=4, seed=0, learned_code=64)
token = np.eye(53)[[3]]                              # one token, its own coordinate
brain.act(token, greedy=True)                        # the code settles with the brain
report = brain.last_settlement
assert report["qualified"] and report["code_steps"] > 0    # the code phase, then the whole graph
key = brain.observation_code(token)                  # what the memories key on
assert key.shape == (1, 64) and (key >= 0).all() and np.isclose(np.linalg.norm(key), 1.0)
code = brain.learned_code                            # the tables the region learns
assert code.dictionary.shape == (53, 64) and code.work["updates"] == 1
```

## What the region is

The region's tables are the brain's own synapses and biases, read in every sweep as every
other synapse is:

| Table | Synapses | Role |
| --- | --- | --- |
| dictionary `Phi`, `inputs` by `units`, unit-length columns | one way from the sensory neurons to the code units | the bottom-up boundary: code unit `j` is driven by `Phi_j . s_sensory` |
| the Gram matrix of the dictionary | every ordered pair of code units, `-(Phi_j . Phi_k)` | lateral competition: units whose columns overlap inhibit each other |
| predictor `W`, `units` by the association width | one way from the prefrontal region, scaled by `learned_code_top_down` | the top-down boundary: the working trace's prediction of the code lowers the thresholds of the units it predicts |
| thresholds `t`, one per unit | the units' biases, `-t_j` | how much drive a unit needs before it publishes |

A code unit settles under the brain's neuron model to

```text
v_j = Phi_j . (s_sensory - Phi s) + s_j + top_down * p_j - t_j,    s_j = f(v_j),    p = W s_prefrontal
```

the sparse state that reconstructs the sensory activation through the dictionary as far as
the thresholds allow, drawn toward what the context predicts. The sensory neurons project to
the code region alone, and nothing leads back into it, so its fixed point depends on the
observation and the working trace alone.

## How a moment settles

A moment's settle has a code phase first: the region's sub-network (its units, its lateral
synapses, its biases, the same neuron model) settles under the dictionary's read of the
sensory activation and the predictor's read of the prefrontal activation, by the brain's
solver policy (the fast repair at the model's step, then halved steps within the same budget)
to the learner's tolerance within its free budget, warm from the stream's code. The memories
then address the moment on the rectified settled code at unit length, and the whole graph
settles warm from the stream's state with the sensory, prefrontal and code potentials at their
fixed points, qualified by the one residual over every neuron. The code units receive in the
whole settle the input they settled under in the phase, so they stay where the phase left
them. `last_settlement` reports the phase as `code_steps`, `code_residual` and
`code_halvings` beside the whole graph's `steps`. A code phase that does not qualify
refuses the moment with `RuntimeError` before anything changes, as a refused free solve does.

The code's fixed points are stable equilibria of the neuron equations, but the brain's
synchronous sweep at the model's step can oscillate around one when several units with
overlapping columns are active together (the sweep contracts only while `dt * (1 - mu_min)`
stays below 2, `mu_min` the smallest eigenvalue of the slope-weighted lateral coupling).
After each phase the region reports the halvings of the step its settled code needs
(`LearnedCode.stability`, keeping `dt * (1 - mu_min)` at or below 1.5), and the brain
integrates every phase of the moment at that step, the lessons' and the actor's phases
included, with the step budgets (`learning_free_steps`, `learning_nudged_steps`,
`actor_eligibility_steps`, an imagined branch's budget) scaled by the same power of two so
that every phase covers the same time; `describe()` reports the configurations as given, and
`learned_code["stepping"]` the level in effect. At the chamber of 53 tokens the level stays
at 0; a dictionary whose columns come to overlap strongly, as in a world of a few frames,
raises it to 1 or 2 and the settles cost that many times the sweeps.

The top-down boundary reads the working trace, the settled association activity of the moment
before, and never the association region of the moment: the recall memory addresses the moment
before its settle on the code, and a code that read the association region would depend on
what the moment recalls; a prediction read from the moment's own association activity would
copy the code it reads.

## How it learns

At every accepted moment (each `act`, `step` and `live` answer, and each moment `wait` senses)
the region takes one step from the settled state, each row of the batch one event, with `a`
the rectified code, `x'` the sensory activation, `z` the prefrontal activation and `p = W z`:

| Table | Rule | Gene (founder) |
| --- | --- | --- |
| dictionary | `Phi += rate * sum_rows (x' - Phi a) a^T`, each column scaled back to unit length | `learned_code_rate` (0.125) |
| lateral weights | recomputed from the dictionary | with the dictionary |
| predictor | `W += sum_rows r_row (a - p) z^T`, `r_row = min(predictor, 1 / \|z\|^2)` | `learned_code_predictor` (0.03) |
| thresholds | `t += threshold_rate * sum_rows ((a > 0) - target)`, floored at 0 | `learned_code_threshold_rate` (0.001), `learned_code_target` (0.1) |

The brain's graph takes the new tables at once. While the code fits, the residuals are small
and nothing moves. The lesson's contrast and the dopamine leave the region's synapses and
biases alone (the learner's plastic masks exclude them), and lessons, `fit`, `predict` and
`imagine` take no step. Under `live` a calm brain learns nothing from its outcome and writes
no record; the region still takes its step, as the traces advance at every moment. A rate of
zero freezes that part; `retune` changes every rate and `learned_code_top_down` in the same
life.

```python
teacher = Brain.compose(inputs=53, actions=4, seed=1, learned_code=64, recall=True)
sentence = [np.eye(53)[[k]] for k in (40, 3, 44)]   # a context, a token, a context
for x in sentence:
    teacher.act(x, greedy=True)                      # three moments, three steps
assert teacher.learned_code.work["updates"] == 3
assert teacher.recall.keys.shape == (1, 3, 64)      # the slots hold the moments' codes
teacher.lesson(sentence[1], [2])                     # a lesson moves the plastic synapses alone
assert teacher.learned_code.work["updates"] == 3
teacher.retune(learned_code_rate=0.0)                # the dictionary stays from here on
```

## Settings

| Setting | Founder | When | What it sets |
| --- | --- | --- | --- |
| `learned_code` | `None` | construction | the region, of that many units (at least 2); `None` builds nothing |
| `learned_code_top_down` | 4.0 | retune | weight of the prediction's drive on the code units |
| `learned_code_rate` | 0.125 | retune | the dictionary's Hebbian rate per row and moment |
| `learned_code_predictor` | 0.03 | retune | the predictor's delta rate per row, bounded by `1 / \|z\|^2` |
| `learned_code_threshold` | 0.1 | construction | every unit's initial threshold |
| `learned_code_threshold_rate` | 0.001 | retune | the homeostatic rate per row |
| `learned_code_target` | 0.1 | retune | the share of rows a unit should be active in |

Refused with `ValueError`: `learned_code` below 2, a `learned_code_*` gene without the region,
`code` beside `learned_code` (a key that joins the observation's own identity to its meaning
is not built in this version), an image brain, `memory_expansion` or `memory_fan_in` (the
memories key on the code as it is), a backend other than `cpu`, and a custom connectome
without a `code` population and the complete dictionary, lateral and predictor synapses.

## What the memories key on

With the region, `brain.code_width` is `units`, and the associative store, the episodic place
and context records and the recall memory key on the rectified settled code at unit length:
the code of the moment in the stream's context. A decision keeps the code it was keyed on
until its outcome is recorded. `brain.observation_code(x)` returns the code the observation
settles to on its own, without a context; `stimulus(..., memory=False)`, `fit`, `predict` and
`accuracy` settle that context-free code too. The heard history holds the code, in blocks of
`units` per decay, and the recall memory's event values are codes.

## Save, load, describe, cost

A brain with the region saves checkpoint format `cadence-generic/10`: the `learned_code`
metadata (units, widths, seed, genes, work counts) and the tables `learned_code/dictionary`,
`learned_code/weights` and `learned_code/thresholds`, with `moment/code` and `lived/code` for
a decision awaiting its outcome. `Brain.load` installs the saved tables and refuses a
checkpoint whose saved synapses disagree with them. `describe()` reports `observation_code`
with kind `"learned"`, the genes as `learned_code_*` and a `learned_code` section; `parameters()`
counts the three tables; `reset()` leaves them as it leaves the weights.

The region adds `units` neurons, `inputs * units` dictionary synapses, `units * (units - 1)`
lateral synapses and `width * units` predictor synapses, and replaces the `inputs * width`
sensory synapses of the first module by `units * width`: `compose(53, 4, learned_code=64)`
has 20,236 synapses against 8,716 without the region. Each moment costs the code phase's
sweeps (about 32 at rest; more under a strong prediction, where the halved step settles it)
and one learning step of `2 * inputs * units + units * width` multiply-adds. The dictionary
is dense, so at camera width it costs `inputs * units` values; a bounded fan-in per unit is
not built.
