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
key = brain.observation_code(token)                  # what the memories key on: the code
assert key.shape == (1, 64 + 53)                     # joined to the token's own identity
assert (key >= 0).all() and np.isclose(np.linalg.norm(key), 1.0)
code = brain.learned_code                            # the tables the region learns
assert code.dictionary.shape == (53, 64) and code.work["updates"] == 1
```

## What the region is

The region's tables are the brain's own synapses and biases, read in every sweep as every
other synapse is:

| Table | Synapses | Role |
| --- | --- | --- |
| dictionary `Phi`, `inputs` by `units`, unit-length columns | one way from the sensory neurons to the code units, at the read gain `1 / f(A)` (`f` the neuron's activation, `A` the model's stimulus amplitude): the region reads the sensory activation in units of one full input, `x' = s_sensory / f(A)` | the bottom-up boundary: code unit `j` is driven by `Phi_j . x'`; one input at amplitude 1 drives the unit whose column matches it by one |
| the Gram matrix of the dictionary | every ordered pair of code units, `-(Phi_j . Phi_k)` | lateral competition: units whose columns overlap inhibit each other |
| predictor `W`, `units` by the association width | one way from the prefrontal region, scaled by `learned_code_top_down` | the top-down boundary: the working trace's prediction of the code, `p = W s_prefrontal` |
| thresholds `t`, one per unit | the units' biases, `-t_j` | how much drive a unit needs before it publishes |
| the standing boundary | the solver's quadratic anchor toward zero at gain `learned_code_top_down` on the code units, carried by every phase (`brain.learner.standing`) | the restoring term `-top_down * s_j`; with the predictor's drive it is the pull `top_down * (p_j - s_j)` toward the prediction, the gradient of `top_down / 2 * |s - p|^2` |

A code unit settles under the brain's neuron model to

```text
v_j = Phi_j . (x' - Phi s) + s_j + top_down * (p_j - s_j) - t_j,    s_j = f(v_j),    x' = s_sensory / f(A),    p = W s_prefrontal
```

the sparse state that reconstructs the sensory activation, as read, through the dictionary
as far as the thresholds allow, drawn toward what the context predicts. The sensory neurons project to
the code region alone, and nothing leads back into it, so its fixed point depends on the
observation and the working trace alone. The boundary is held fixed through every solve and
is part of the residual that qualifies a state, as the temporal boundary of `PatchNet` is;
at `learned_code_top_down=0` there is none.

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
`code_halvings` beside the whole graph's `steps`. With the code held (`learned_code_held`,
the founder) the code units are a source in the whole settle: their synapses in the whole
graph carry nothing, the moment's drive is the phase's potential less their bias, and they
publish the phase's code exactly through the free settle and every lesson's phase, so the
rest of the brain integrates at the model's step with its budgets, its adaptation and its
finite learning phases as a brain without the region has them; the step halvings below then
apply to the region's own phase alone. A code phase that does not qualify at the
brain's step is settled once more at the step the dictionary's coupling allows over every
state the region can reach (`LearnedCode.rest_stability`: every unit active, every slope at
its largest), and refuses the moment with `RuntimeError` before anything changes when it
does not qualify there either, as a refused free solve does. A world of one input needs
this from its first moment: every column of the dictionary is then plus or minus the same
vector, every pair of units competes at full strength, and the code settles at a sixteenth
of the model's step.

The code's fixed points are stable equilibria of the neuron equations, but the brain's
synchronous sweep at the model's step can oscillate around one when several units with
overlapping columns are active together (the sweep contracts only while `dt * (1 - mu_min)`
stays below 2, `mu_min` the smallest eigenvalue of the slope-weighted lateral coupling).
After each phase the region reports the halvings of the step its settled code needs
(`LearnedCode.stability`, keeping `dt * (1 - mu_min)` at or below 1.5, the restoring term
on the coupling's diagonal), and the brain
integrates every phase of the moment at that step, the lessons' and the actor's phases
included, with the step budgets (`learning_free_steps`, `learning_nudged_steps`,
`actor_eligibility_steps`, an imagined branch's budget) scaled by the same power of two so
that every phase covers the same time; `describe()` reports the configurations as given, and
`learned_code["stepping"]` the level in effect. At `learned_code_top_down` 4 the restoring
term alone asks for the halved step wherever a unit sits near its threshold, so a life with
the region runs at level 1 for the most part; a dictionary whose columns come to overlap
strongly, as in a world of a few frames, raises it to 2 or 3, and the settles cost that
many times the sweeps.

The top-down boundary reads the working trace, the settled association activity of the moment
before, and never the association region of the moment: the recall memory addresses the moment
before its settle on the code, and a code that read the association region would depend on
what the moment recalls; a prediction read from the moment's own association activity would
copy the code it reads.

## How it learns

At every accepted moment (each `act`, `step` and `live` answer, and each moment `wait` senses)
the region takes one step from the moment's free settle, each row of the batch one event;
the lesson's nudged phases, the outcome's eligibility phases, the imagined branches and
the readings taken on copies enter no table, as the contrast and the dopamine enter no
synapse of the region. With `a` the rectified code, `a_free` the moment's context-free code
(the region settled on the observation alone, a second code phase, `context_free_steps`), `x'` the sensory activation as read (at the gain above), `z` the
prefrontal activation and `p = W z`:

| Table | Rule | Gene (founder) |
| --- | --- | --- |
| dictionary | `Phi += rate * sum_rows (x' - Phi a) a^T`, each column scaled back to unit length | `learned_code_rate` (0.125) |
| lateral weights | recomputed from the dictionary | with the dictionary |
| predictor | `W += sum_rows r_row (a - p) z^T`, `r_row = predictor / max(\|z\|^2, predictor^2)`: a row closes the share `predictor` of its own gap whatever its context's energy (the prefrontal activation is small in the brain's units); a context fainter than `\|z\| = predictor` is counted at that energy, so no row commands a synaptic step above its gap | `learned_code_predictor` (0.0625) |
| thresholds | `t += threshold_rate * sum_rows ((a_free > 0) - target)`: a unit's share is the share of the observations that drive it on their own, so what the context predicts may recruit a unit without raising its threshold and the context-free code keeps the target share by construction; floored at 0; the repair: a row whose code leaves at least the share `repair` of its input's energy unexplained (`\|x' - Phi a\|^2 >= repair * \|x'\|^2`; at the founder level 1 a row whose code is silent or worse than nothing), in context or context-free where the place key is settled, lowers the threshold of every unit silent in it by `threshold_rate * silence`; a unit that fires, and a row the code reconstructs, are left to the share rule; there is no ceiling | `learned_code_threshold_rate` (0.000625), `learned_code_target` (0.1), `learned_code_repair` (1.0), `learned_code_silence` (10) |

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
assert teacher.recall.keys.shape == (1, 3, 64 + 53) # the slots hold the moments' keys
teacher.lesson(sentence[1], [2])                     # a lesson moves the plastic synapses alone
assert teacher.learned_code.work["updates"] == 3
teacher.retune(learned_code_rate=0.0)                # the dictionary stays from here on
```

## Settings

| Setting | Founder | When | What it sets |
| --- | --- | --- | --- |
| `learned_code` | `None` | construction | the region, of that many units (at least 2); `None` builds nothing |
| `learned_code_top_down` | 2.0 | retune | weight of the prediction's drive on the code units and gain of the pull toward it; the smallest weight at which the context-free codes of a group's members come together in the embedding chamber's pilots |
| `learned_code_rate` | 0.125 | retune | the dictionary's Hebbian rate per row and moment |
| `learned_code_predictor` | 0.0625 | retune | the share of a row's prediction gap the predictor closes per event, in [0, 1], normalized by the context's energy |
| `learned_code_threshold` | 0.1 | construction | every unit's initial threshold |
| `learned_code_threshold_rate` | 0.000625 | retune | the homeostatic rate per row |
| `learned_code_target` | 0.1 | retune | the share of rows a unit should be active in |
| `learned_code_repair` | 1.0 | retune | the share of a row's input energy its code may leave unexplained, in (0, 1], before the row lowers the thresholds of its silent units: at 1 a row whose code is silent (or worse than nothing); lower, a row the code reconstructs poorly |
| `learned_code_silence` | 10 | retune | how many under-active rows a silent unit counts as in a row that repairs: its threshold falls by `threshold_rate * silence` per such row, so no observation is left without a unit; 0 leaves the thresholds to the share rule alone; `brain.learned_code.work["repairs"]` counts the unit-rows it lowered |
| `learned_code_held` | `True` | construction | the code units held at the code phase's fixed point through the whole settle, a source to the cortex as the sensory neurons are (nothing leads into them in the whole graph; the moment's drive holds them), so the rest of the brain integrates at the model's step with its learning phases as configured and the region's phase steps alone; `False` settles the code units with the whole graph and integrates the whole brain at the step the code's competition needs |
| `learned_code_fields` | `None` | construction | the widths of the observation's fields (a sequence summing to `inputs`; `None` is one field): the units are shared among the fields in proportion to their widths (at least two each), each unit's dictionary column reads its field alone through every learning step, and the competition among the units (the Gram) is block-diagonal, so a recombination of fields is a recombination of their codes and a second input in another field keeps its own code |
| `learned_code_active` | 0.05 | retune | the activation a unit must reach to count as active in the thresholds' share: a unit firing below it counts as under-active, so the homeostasis does not fade every unit of a frequent observation toward zero; 0 counts any firing |
| `learned_code_reads` | `"free"` | construction | what the cortex and the memories read of a held code: the context-free code (the founder; the embedding chamber reads held-out members at .97 and 5 of 6 development founders with it, .92 and 4 of 6 with the code in context), or with `"context"` the moment's code in its context, or with `"free"` the context-free code (the observation's code settled alone, the region's second phase), while the code in context, with the prediction's pull, is what the dictionary and the predictor learn from; `"free"` needs `learned_code_held` |
| `learned_code_identity` | 0.95 | construction | the weight of the observation's own identity in the memories' key, in [0, 1); the key width is `units + inputs`, or `units` at 0 |

Refused with `ValueError`: `learned_code` below 2, a repair level outside (0, 1], an
identity weight outside [0, 1), a `learned_code_*` gene without the region,
`code` beside `learned_code` (a key that joins the observation's own identity to its meaning
is not built in this version), an image brain, `memory_expansion` or `memory_fan_in` (the
memories key on the code as it is), a backend other than `cpu`, and a custom connectome
without a `code` population and the complete dictionary, lateral and predictor synapses,
or whose sensory neurons project anywhere but to the code region. A setting the readings
retired is refused by name with what the region does instead: `learned_code_cortex` (the
first module reads the code alone; the observation read beside the code lost the
embedding chamber's held-out members) and `learned_code_output` (the first module reads
the rectified code at unit length; the activations as they are lost the embedding
chamber, and a read at the observation's own loudness lost delayed cues) and
`learned_code_floor` (every code that is not silent is read at unit length; a floor of
one full input's activation lost the embedding chamber's held-out members) and
`learned_code_scale` (the memories' key takes the norm the observation itself has as a
key; a key at unit length lost the binding family's continuing task) and
`learned_code_place` (the associative store and the episodic place records key on the
context-free code).

## When to declare fields

With one field (the founder, `learned_code_fields=None`) every unit reads the whole
observation, and the region learns codes of whole moments: observations that look alike
as a whole share units, so a member of a group is read as its group's members are, and a
lesson on some members reaches the rest. That is what a world of single tokens or of
whole scenes wants.

When the observation is a combination of independent channels, declare one field per
channel: the coordinates of a sentence's word slots, of who acts, what is held and where,
of a pouch beside the cell a creature stands in, of the patches of an image. Give
`learned_code_fields` the channels' widths in the order of the observation's coordinates
(they sum to `inputs`). Each unit then reads one channel, the units of a channel compete
among themselves alone, and a moment's code is the union of its channels' codes. A
combination never seen before is then read as channels already known, and a second input
in another channel keeps its own code. With one field the same combination arrives as an
unfamiliar whole, and a second input is merged into the code of the first, distinguished
only by the code's strength.

```python
from cadence import Brain

# 8 coordinates of where, 4 of what is held: a held key keeps its own units
brain = Brain.compose(inputs=12, actions=4, seed=0, learned_code=32, learned_code_fields=(8, 4))
assert brain.learned_code.field_units == (21, 11)  # the units shared by the channels' widths
```

Declare fields for the channels the world fires. A field keeps its share of the units (by
width, at least two) for the whole life, and its units read its coordinates alone, so a
field whose coordinates never fire leaves its units silent: the repair lowers a silent
unit's threshold, which recruits nothing in a unit that receives no input. Coordinates the
world never fires belong in the field of a channel it fires.

## What the memories key on

With the region, every memory keys on the rectified settled code at unit length joined to
the observation's own identity, `unit([sqrt(1 - w) code, sqrt(w) unit(x)])` with `w` the
gene `learned_code_identity` (0.5; at 0 the code alone), so `brain.code_width` is
`units + inputs` (`units` at 0): the identity part keeps a binding readable while the code
part moves as the region learns. Two codes of a moment stand apart. The moment's key is its code in the
stream's context, the one the cortex reads: the recall slots (keys and event values) and the
episodic context records key on it, since their job is to tell apart what looks the same by
its history. The place key is the context-free code, the observation settled alone with a
zero prediction, what `brain.observation_code(x)` returns: the associative store and the
episodic place records key on it, since their job is to answer a known situation reached
along a new path, and a code in context would split one situation across the histories it
was reached by. The place key is a second code phase of the moment, taken only for brains
with a place store and reported as `context_free_steps`. A decision keeps the place key it
was taken under
until its outcome is recorded. `stimulus(..., memory=False)`, `fit`, `predict` and
`accuracy` settle the context-free code alone. The heard history holds the moment's code, in
blocks of `units` per decay.

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
