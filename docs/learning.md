# Learning: the free/nudged rule

A Cadence brain learns by settling twice: once on the observation alone (the free phase) and
once with its outputs pulled toward the shown answer (the nudged phases). Every synapse then
moves on the difference between its own endpoint activities in the two. In a composed brain,
teach with `Brain.lesson`, or with `Brain.step(teacher=...)`, which gives the same lesson and
then acts. A standalone `Learner` runs the rule on any `NeuralGraph`:

```python
import numpy as np
import cadence as cd

brain = cd.Brain.compose(inputs=4, actions=2, seed=0)
cue = np.array([[1.0, 0.0, 0.0, 0.0]])
lesson = brain.lesson(cue, np.array([1]))           # teach, without acting
action = brain.step(cue, teacher=np.array([1]))     # the same lesson, then an act
print(lesson["accepted_presentations"], brain.last_learning["demonstration_total_steps"])

connectome = cd.layered(4, 16, 2, seed=0)
learner = cd.Learner(
    cd.NeuralGraph(connectome, cd.learning_neuron_model()), connectome.populations["output"]
)
drive = np.zeros((2, connectome.n))
drive[:, :4] = np.eye(4)[:2]
for _ in range(20):
    state, report = learner.step(drive, np.array([0, 1]))
print(learner.predict(drive), report["total_row_sweeps"])
remaining = learner.brain.residual(drive, state.free)  # equation residual per row
```

`Brain.step` keeps the lesson report in `last_learning` with a `demonstration_` prefix. See
[reward](reward.md) for reward learning, [memory](memory.md) for memory writes, and
[explicit learning with `step`](continuous.md#explicit-learning-with-step) for teacher-acted moments.

## 1. What learning changes

Learning moves `efficacy` (one signed number per synapse) and `bias` (one per neuron); the
connectome and `log_gain` stay fixed. Synapse `e` drives `gain · count[e] · efficacy[e] ·
exp(log_gain[pre[e]])` per unit of presynaptic activation, which is `efficacy[e]` in a `layered`
graph with `learning_neuron_model()`. Call it `W[e]`.

## 2. What settling computes

One step of the neuron model, applied to every neuron at once:

    total[i] = Σ_{e: post[e]=i} W[e] · s[pre[e]] + stimulus[i] + bias[i]   (+ nudge[i])
    v[i]    <- v[i] + dt · (total[i] - v[i])
    s[i]     = act(v[i])

With `learning_neuron_model()`, `act(v) = tanh(v/2)` for `v >= 0` and `0.1 · tanh(v/2)` below:
rest publishes 0, drive saturates near `v ≈ 4`, and the leak keeps a small negative response the
rule can move. Settling repeats the step until no activation moves more than `tolerance` or the
cap is reached ([how the neural graph settles](concepts.md)). A small movement is no fixed-point
certificate, since saturation or a tiny `dt` can stall a neuron far from rest;
`brain.residual(drive, state)` (with the phase's `nudge` for a nudged state) checks the equations.

## 3. The two phases and the update

**Free phase.** Settle under the stimulus alone to `s⁰`. The outputs are the brain's answer.

**Nudged phases.** From `s⁰`, settle with an extra drive on the output neurons:

    nudge[i] = beta · weight · (target[i] - p[i]),   p = softmax(s[outputs] / T)

This is a cross-entropy nudge. `weight` is 1 for a label and the advantage for a reward. With
`centered=True` two phases run from `s⁰`, at `+beta` (`s⁺`) and `-beta` (`s⁻`). With several
output `slots`, each slot is its own softmax and carries `beta / slots`.

**Update.** Every synapse reads its two endpoints in the two phases:

    contrast[e]  = mean_batch( s⁺[pre]·s⁺[post] - s⁻[pre]·s⁻[post] ) / (2 beta)
    bias_term[i] = mean_batch( s⁺[i] - s⁻[i] ) / (2 beta)
    efficacy[e] += eta · contrast[e];   bias[i] += eta_bias · bias_term[i]

With `centered=False`, `s⁰` replaces `s⁻` and `beta` replaces `2 beta`. With `reciprocal=True` a
synapse and its reverse share one efficacy and move by their mean contrast. Plastic efficacies
are clipped to `±scale_cap`. No backward computation graph is kept.

## 4. One observation repairs a settled prediction

For a custom world model, save the free prediction (`Learner.free`) under the observation and
proposed action. Once the action has run, its observed consequence is the target, and both
nudged phases start from that saved state. The repair assigns no credit across delays; use
[reward eligibility](reward.md) for that. Score an outcome before teaching it. Where an outcome
follows its reading directly, [records](records.md) learn it without settling phases.

## 5. Why the contrast is a gradient

Let `F` be the recurrent neurons with the input activations held fixed. With no adaptation,
symmetric effective recurrent weights and a monotone differentiable activation, the
continuous-time dynamics descend

    E(v_F) = Σ_{i∈F} ∫₀^{v[i]} u·act'(u) du − ½ Σ_{i,j∈F} W[i→j] s[i] s[j] − Σ_{i∈F} d[i] s[i]
    d[i] = stimulus[i] + bias[i] + Σ_{k input} W[k→i] s[k]

On a smooth stable branch, converged phases obey the
[equilibrium-propagation identity](https://arxiv.org/abs/1602.05179): the limiting contrast is
minus the loss gradient for the one weight of a reciprocal pair. The quadratic nudge's loss is
half squared error; the cross-entropy drive omits `1/T`, so its loss is `T` times cross-entropy.
The [centered estimator](https://arxiv.org/abs/2006.03824) leaves an error of order `beta²`, and
a phase solved to sup-norm error `eps` adds at most `A·(eps⁺ + eps⁻)/beta` (`A` an activity
bound), so a smaller nudge needs a tighter solve. A nudge across the kink at zero or into
another attractor loses this accuracy; `leak=1` gives the smooth `tanh(v/2)`.

`Learner.contrast` returns the statistic. Where `c[e] = gain · count[e] · exp(log_gain[pre[e]])`
is not one, multiply by `c[e]` to get minus the gradient for a tied `efficacy`, and divide by `T`
for ordinary cross-entropy. Tie groups (`Learner(tie_groups=...)`) and reciprocal pairs combine
transitively; a group moves by its members' mean contrast and frozen members stay fixed. Tying
constrains increments, so tied members stay equal only if they start equal.
`tests/test_equilibrium.py` checks the conversion by finite differences;
[benchmarks/local_equilibrium](../benchmarks/local_equilibrium/README.md) compares the update
with an implicit-gradient learner on the same network, and the
[certificate](certificate.md#equilibrium-propagation-scope) says when phases contract.

### Qualified teaching budgets

Teaching is finite by default: the contrast is read after at most `free_steps` and
`nudged_steps` sweeps. `qualified=True` requires every phase to meet the full potential and
adaptation residual within `tolerance`, with up to `damping` step halvings inside its budget; a
failed phase raises `LearningPhaseError` (`phase`, `phases`, `report`, `hint`) and writes
nothing. Both budgets are then settle budgets, and a nudged phase needs about as many sweeps as
the free one: at the composed `free_steps=1024`, `nudged_steps=12` a qualified brain refuses
lessons on realistic input. `qualified=True` with `nudged_steps < free_steps` therefore warns,
and a refusal names the mismatch in its `hint`. Opt in with named overrides, which keep the
composed rates:

```python
checked = cd.Brain.compose(
    inputs=4, actions=2, seed=7, learning_qualified=True, learning_nudged_steps=1024,
)
qualified_lesson = checked.lesson(cue, np.array([0]))
print(qualified_lesson["qualified"], qualified_lesson["total_steps"])
```

Qualification certifies the equations at those states; the gradient reading also needs the
assumptions above. `contrast`, `update` and `apply` act on supplied states, and reward
eligibility keeps its finite phase. Answers follow
[qualification and refusal](continuous.md#qualification-and-refusal).

### Finite free phases that run out of budget

Finite teaching updates from whatever state the budgets reached. A free phase that used all of
`free_steps` usually never reached the movement stop. `Learner.step` then reports
`free_budget_exhausted` (`demonstration_free_budget_exhausted` through `Brain.step`) and warns;
`free_residual` holds the equation residual. Fixed-length phases (`tolerance=None`) stay silent.
Raise `free_steps`, lower the gain, or select `qualified=True` to refuse such lessons.

A lesson report holds `attempted_presentations`, `accepted_presentations`, `accepted`,
`qualified`, `qualification_required`, the mean absolute steps `scale_step` and `bias_step`,
and per phase (`free`, `nudged`, `opposite`) `_steps`, `_residual`, `_residual_checks`,
`_damping_halvings`, `_stagnation_checks` (complete-state comparisons that let a stalled
halving pass its unused sweeps on), `_row_sweeps` and `_row_residual_checks`, with `total_*`
sums. A refused lesson's work is in the error's `report`.

### Acquisition, working memory and durable retention

Acquisition, working-memory recall and durable retention are separate measurements: `predict`
and `accuracy` test the graph, `act` the graph with its memories
([check that a memory does the work](memory.md#check-that-a-memory-does-the-work),
[evaluate a brain](evaluation.md)).

### Lessons only where the answer differs

`LearnerConfig(surprise=True)` (`learning_surprise=True` when composing) teaches only the rows
whose answer differs from the label. A standalone learner judges by its free phase, which runs
on every row; agreeing rows get no nudged phase and no update. A caller holding the answer passes
`step(drive, labels, answer=...)`, and agreeing rows are dropped before the free phase. The
composed brain passes the act it would issue, warm from the stream; with an
[episodic vote](episodic.md#one-choice-for-acting-teaching-and-imagining) that act includes the
vote, so rows the memory answers are never taught to the cortex. The report adds
`agreed_presentations`, `LearnedState.taught` marks the taught rows, and an episodic memory
records only those demonstrations.

Select it where a readout saturates under lessons it answers correctly, or where streams are
shown row by row for many passes. The founder is `False`;
[benchmarks/stream_words](../benchmarks/stream_words/README.md) uses it with every-row lessons as
the control.

## Feedback and the reach of a nudge

A contrast is zero where neither endpoint moves under the nudge, so hidden neurons need a
feedback path from the outputs; `layered` supplies tied output-to-hidden feedback. Asymmetry
among free neurons breaks the energy argument; one-way projections from fixed inputs keep it.
`ep_structure(brain, fixed_inputs=...)` checks this structure and nothing about convergence.
Adding reverse synapses to a measured connectome changes the model ([connectomes](connectomes.md)).

## 6. Features on input neurons

A batched drive has one column per neuron. `NeuralGraph.stimulus_levels(levels)` multiplies
dense, finite, possibly signed per-neuron levels by the neuron model's `stimulus_amplitude`; it
neither pads nor clips. `layered` input neurons have no inbound synapses, so with zero bias each
settles at `v = stimulus`:

```python
pattern = cd.layered(2, 8, 2, density=1.0, seed=0)
graph = cd.NeuralGraph(pattern, cd.learning_neuron_model())
levels = np.zeros((2, pattern.n))
levels[:, list(pattern.populations["input"])] = [[1.0, 0.0], [0.0, 1.0]]  # one row per case
pattern_drive = graph.stimulus_levels(levels)
```

Fit scaling and encodings on training data. The `Learner` operations:

| call | what it does |
|---|---|
| `step(drive, labels, warm=None, weight=None, *, answer=None)` | free phase, nudged phase(s), one update; returns `(LearnedState, report)`. `labels` holds one output index per row, `(batch, slots)` when slotted; `weight` scales each row's pull (an advantage) |
| `free(drive, warm=None)`, `nudged(drive, free, target, sign=1.0, weight=None)` | one phase; `sign=-1.0` is the opposite phase; `targets(labels)` builds the per-neuron target |
| `contrast(free, nudged, opposite=None)`, `contrast_rows(...)` | the per-synapse and per-neuron terms divided by `beta` or `2 beta`, batch mean or per row |
| `update(free, nudged, opposite=None)` | momentum, normalization and the step from supplied states |
| `apply(delta_scale, delta_bias)` | a precomputed step through masks, synapse rates, tying, decay and the cap |
| `predict(drive)`, `accuracy(drive, labels, batch=256)` | most active output per slot after a free phase; fraction correct over rows and slots |

`LearnedState` holds the `free`, `nudged` and `opposite` states (`opposite` is `None` when not
centered) and `taught`. A `warm` state starts the free phase from an earlier state: within one
basin this saves steps, across attractors it can change the answer. A centered update costs
three settles. A finite warm phase that stops at its step cap keeps its transients and need not
be an equilibrium; reset at independent episode boundaries and compare warm and cold settles on
changing inputs.

### Pattern targets

For continuous or multi-label targets, build one target column per neuron and run the phases
directly with the quadratic nudge (`beta · (target − s)`):

```python
regressor = cd.Learner(graph, pattern.populations["output"], cd.LearnerConfig(nudge="quadratic"))
target = np.zeros_like(pattern_drive)
target[:, regressor.output_index] = [[0.2, 0.8], [0.8, 0.2]]
free = regressor.free(pattern_drive)
plus = regressor.nudged(pattern_drive, free, target)
minus = regressor.nudged(pattern_drive, free, target, sign=-1.0)
regressor.update(free, plus, minus)
reading = regressor.free(pattern_drive).activation[:, regressor.output_index]
```

Keep targets inside the activation's range and read activations directly.

### Two learners on one graph

An update replaces a learner's graph. Build a second learner with `Learner.sharing` so both read
one `GraphHolder`:

```python
shared = cd.layered(4, 8, 4, seed=1)
outputs = list(shared.populations["output"])
first = cd.Learner(cd.NeuralGraph(shared, cd.learning_neuron_model()), outputs[:2])
second = cd.Learner.sharing(first, outputs[2:])  # other constructor fields as keywords
shared_drive = np.zeros((1, shared.n))
shared_drive[:, :4] = 1.0
first.step(shared_drive, np.array([1]))
assert second.brain is first.brain and second.holder is first.holder
```

An update replaces `holder.brain` once every check passes; assigning `learner.brain` replaces
it for every learner on the holder. Each learner keeps its own optimizer history. Disjoint
`plastic_synapses` and `plastic_neurons` masks keep one's decay off the other's
([write a cortex](cortex.md#which-synapses-a-head-owns)). `Learner.load` and
`dataclasses.replace` give a holder of its own; `copy.copy` shares it.

### Calibrating the operating point

`learner.calibrate(drive, level=None, grid=None)` picks and installs the global synaptic gain
whose free outputs sit nearest a target; it teaches nothing. The default target puts the mean,
over rows and slots, of the highest output activation at 0.5. An explicit `level` targets the
mean over every output, which pushes all choices up together and can saturate the readout. The
default grid is the current gain, then its multiples `2**k` for `k = -8..8`, `k ≠ 0`; ties keep
the first. Under `qualified=True` candidates must meet the equations. `last_calibration` holds
every candidate's gap, residuals and work; with no usable candidate it raises `RuntimeError`.
A refused candidate has `gap` `None`; its `admitted`, `finite`, `activation_consistent` and
residuals give the reason. The sampled gains may never reach the target, and reaching it does
not show that teaching will work.

`calibrate_bias(brain, drives, targets)` bisects biases toward a mean activation in `[0, 1]`
per population (a name or neuron indices in `targets`), or per member with `per_neuron=True`,
and returns the bias array, leaving the graph unchanged. Each bias is bisected `iterations=16`
times within `span=(-6.0, 6.0)`, every population in turn, for `rounds=3`; each solve settles
for at most `steps=100` sweeps to `tolerance=1e-4`. An unreachable target ends at a span
endpoint, so check `final_means` and `target_gaps` in the dict passed as `report`. With
`qualified=True` (and up to `damping=3` halvings) every solve must meet the equations; a
refused or nonfinite solve raises `RuntimeError`. Unlike the gain, which scales inhibitory
synapses too, a bias moves one population.

```python
operating = cd.Brain.compose(4, 3, modules=(8,), lateral=0.0, seed=9)
training = operating.stimulus(np.array([[1.0, 0.2, 0.0, 0.1], [0.2, 1.0, 0.1, 0.0]]), memory=False)
calibration = {}
bias = cd.calibrate_bias(
    operating.brain, training, {"motor": 0.25},
    qualified=True, damping=3, steps=256, tolerance=1e-7, report=calibration,
)
candidate = operating.brain.with_parameters(bias=bias)
assert np.all(candidate.equilibrate(training, budget=256, tolerance=1e-7, damping=3).qualified)
operating.learner.brain = candidate
print(calibration["final_means"], calibration["target_gaps"])
```

Calibrate on training inputs and freeze the result before evaluation.

## 7. Every knob

`LearnerConfig()` holds the standalone defaults below. Each field is the `learning_<field>`
override of `compose` and `retune`, with the composed founders in
[defaults and expert overrides](brain.md#defaults-and-expert-overrides);
`learning=LearnerConfig(...)` replaces the composed config whole. A standalone learner's frozen
config changes between updates with `learner.config = dataclasses.replace(learner.config,
eta=...)`. `eta_bias=None` derives `eta / 10`; a resolved value is kept when `eta` changes
unless set again or passed as `None`. Select rates on development data and freeze them.

| knob | where | what it does | default |
|---|---|---|---|
| `beta` | `LearnerConfig` | nudge strength; smaller is closer to the gradient | 0.1 |
| `eta`, `eta_bias` | `LearnerConfig` | efficacy and bias steps; an `eta_bias` above `eta` warns | 0.2, `None` (`eta / 10`) |
| `centered` | `LearnerConfig` | `+beta` against `-beta` instead of against the free state | `True` |
| `free_steps`, `nudged_steps` | `LearnerConfig` | sweep caps of the free and nudged phases; [settle budgets](#qualified-teaching-budgets) when qualified | 100, 50 |
| `tolerance` | `LearnerConfig` | movement stop for finite phases, equation residual when qualified; `None` fixes the length | 1e-4 |
| `nudge` | `LearnerConfig` | `"cross_entropy"` or `"quadratic"` | `"cross_entropy"` |
| `temperature` | `LearnerConfig` | softmax temperature `T` of the cross-entropy nudge; in a composed brain also the sampled policy's base temperature | 0.2 |
| `momentum` | `LearnerConfig` | in `[0, 1)`: step on a bias-corrected running mean of each contrast | 0.0 |
| `normalize`, `normalize_floor` | `LearnerConfig` | forgetting factor in `[0, 1)` of a running RMS that divides each step, plus a floor ([below](#rates-under-normalization)) | 0.0 (off), 1e-3 |
| `decay` | `LearnerConfig` | in `[0, 1)`: shrink every plastic efficacy and bias by this fraction per update | 0.0 |
| `scale_cap` | `LearnerConfig` | largest plastic efficacy magnitude; smaller keeps readouts out of saturation | 8.0 |
| `qualified`, `damping` | `LearnerConfig` | require full-equation phases; most `dt` halvings per phase | `False`, 3 |
| `surprise` | `LearnerConfig` | [teach only differing rows](#lessons-only-where-the-answer-differs) | `False` |
| `outputs`, `config` | `Learner` | output neuron indices; the `LearnerConfig` | required, `LearnerConfig()` |
| `plastic_synapses`, `plastic_neurons` | `Learner` | bool masks of moving efficacies and biases | all |
| `reciprocal` | `Learner` | tie each synapse and its reverse into one efficacy | `True` |
| `tie_groups` | `Learner` | int per synapse, `-1` untied; a group shares one step | `None` |
| `synapse_rate` | `Learner` | nonnegative step multiplier per synapse, before tying | `None` (1) |
| `slots` | `Learner` | outputs split into softmax groups: a count of equal groups or a size per group | 1 |
| `gain`, `slope`, `leak`, `dt`, `stimulus_amplitude` | `learning_neuron_model` | synaptic gain, activation slope, sub-rest response, integration step, input scale | 1.0, 1.0, 0.1, 0.5 (`Brain.compose` uses 1.0), 1.0 |
| `density`, `feedback`, `lateral`, `seed` | `layered` | input-to-hidden density, feedback scale, output-to-output weight, wiring seed | 0.3, 1.0, 0.0, 0 |
| `count`, `init`, `excitatory_forward` | `layered` | contacts per synapse, initial magnitude scale, positive forward signs only | 1.0, 1.0, `False` |
| `skip`, `skip_init` | `layered` | direct input-to-output synapses and their own init scale (`None` uses `init`) | `False`, `None` |

With `reciprocal=True` the learner refuses (`ValueError`) a connectome with two synapses between
the same pair of neurons; merge them with `Connectome.from_synapses` (counts add) or pass
`reciprocal=False`. The learner keeps `plastic_synapses` and `plastic_neurons` by reference, so
editing those arrays changes later updates. Tie groups are fixed at construction: rebuild the
learner to change them.

`embedded(vocabulary, positions, dim, hidden, outputs, seed=0, init=1.0)` returns a one-hot
token window through a shared embedding and its `tie_groups`. `Learner.parameters()` counts
plastic efficacies (a pair or group once) plus plastic biases. `updates` counts applied updates,
`contrast_updates` those from contrasts; `save` and `load` are in the [API reference](api.md).

## Rates under normalization

With `normalize > 0` each plastic efficacy and bias keeps a running second moment of its raw
signal. Before masks, synapse rates, tying, decay and clipping, the increment is

```text
increment = rate * signal / (sqrt(bias_corrected_second_moment) + floor)
```

with `rate` = `eta` or `eta_bias`, `signal` the raw contrast (its bias-corrected mean under
`momentum`; the second moment always reads the raw contrast) and `floor` = `normalize_floor`. A
first update is `rate * g / (abs(g) + floor)`, so a consistent signal well above the floor steps
by about the rate whatever its size; changing signals can step above it. Momentum with RMS is an
Adam-style update.

`Brain.compose` teaches under normalization: its teaching founder is `normalize=0.99`,
`normalize_floor=1e-4`, `eta=0.003`, `eta_bias=0.0003` (`eta / 10`) and `momentum=0.9`. The
composed actor keeps unnormalized rates (`eta=1.0`, `eta_bias=0.05`). Under normalization,
rates of that size give steps that dwarf the raw signals and saturate the outputs. Retune `eta`
and `eta_bias` together; a development sweep might start at `eta` 0.001 to 0.003, checked
against acquisition, retention and actual step sizes. `LearnerConfig` warns when
`normalize > 0` and `eta` or `eta_bias` exceeds 0.05; smaller rates can also fail. A named
`learning_eta` keeps the current bias rate, and `learning_normalize=0` with `learning_eta` and
`learning_eta_bias` returns a composed brain to raw steps. `learning=LearnerConfig(...)` replaces the whole teaching configuration.

The teaching founder comes from a copy measurement. A composed brain with one region of 256
neurons (`Brain.compose(260, 3, modules=(256,))`) learned the moves of a three-move look-ahead
demonstrator in a light-cycle game on a 24 x 24 grid: 260 inputs of an egocentric view, three
actions, about 124,000 labelled moments from 1,400 games, and 100 further games held out. The
moments came in random order through `brain.learner.step(brain.stimulus(x, memory=False),
labels)` and were scored cold with `predict`. The reference is a network with one hidden layer of
256 trained by backpropagation with Adam (rate 1e-3) on the same samples in the same order and
batch size. Each arm ran two seeds; the table gives the agreement with the demonstrator on the
held-out moments after 240,000 samples.

| Learner | All held-out moves | Moves where the demonstrator turned |
| --- | --- | --- |
| backpropagation with Adam, batch 8 | 0.874 / 0.888 | 0.690 / 0.804 |
| previous composed founder (`eta=0.5`, unnormalized), batch 8 | 0.81 at 80,000 samples | 0.55 at 80,000 samples |
| `normalize=0.99`, `eta=0.003`, batch 32 | 0.868 / 0.866 | 0.740 / 0.722 |
| `normalize=0.99`, `eta=0.003`, 48 nudged steps, batch 32 | 0.869 / 0.863 | 0.760 / 0.678 |
| `normalize=0.99`, `eta=0.001`, batch 8 | 0.865 / 0.869 | 0.701 / 0.701 |

At 720,000 samples backpropagation reached 0.900 / 0.903 on all moves and the normalized brain
0.886. Training agreement equalled held-out agreement in every arm. Lessons of about 32 rows or
more reduce the noise of each normalized update; with lessons of 8 rows the rate 0.001 reached
the same agreement. Normalization at `eta=0.01` overshot.

The working trace lowers teaching fidelity at its composed amplitude. In a continuing stream
school of the same task (32 streams watching the demonstrator, a greedy `act` and then a `lesson`
at each moment, the working trace on), the same teaching founder agreed with the demonstrator on
0.86 of all moves at 240,000 samples with a `working_memory_amplitude` of 0 or 0.3, and on 0.80 at
the composed amplitude of 3.0. Applications that bootstrap by imitation may lower
`working_memory_amplitude`.

A normalized founder for both composed learners was measured and declined in 0.73.1
([issue 131](https://github.com/muellerberndt/cadence/issues/131)). At `eta=0.003`,
`normalize=0.99` and `momentum=0.9` for teaching and the actor together, supervised acquisition
contracts passed and the reward stream's re-adaptation after a contingency change reached 0.486
against its 0.9 contract: absolute per-synapse steps of that size do not overturn an established
policy quickly. The normalized founder therefore covers teaching alone, and the actor keeps its
unnormalized rates. The actor's case is in [reward](reward.md#rates-under-normalization).

## Teach on the brain's own path

A brain that only watches a demonstrator learns the states the demonstrator visits. Once it
acts, its own small errors lead it to states no lesson covered, and the errors compound over an
episode. Labels on the states the brain itself reaches cover exactly those. The recipe is DAgger
(Ross, Gordon and Bagnell 2011). After a bootstrap by watching, the brain acts and its own move
is executed with `brain.act(observation, greedy=True)`, the demonstrator labels that same
observation, and `brain.lesson(observation, label)` teaches it. A greedy act holds nothing for
reward, so the lesson is legal; `lesson` refuses while a sampled action awaits its outcome. Each
batch row below is one stream:

```python
def observe(position):  # a one-hot position on a line of five cells
    return np.eye(5)[position]

def demonstrator(position):  # the policy the brain watches: 1 steps right, 0 steps left
    return (position < 2).astype(np.int64)

def move(position, action):
    return np.clip(position + np.where(action == 1, 1, -1), 0, 4)

pupil = cd.Brain.compose(inputs=5, actions=2, seed=0)
position = np.full(32, 2)  # 32 streams, one per batch row, starting in the middle
for moment in range(32):  # bootstrap by watching: the demonstrator's moves are executed
    label = demonstrator(position)
    pupil.lesson(observe(position), label)
    position = move(position, label)
cells = np.arange(5)
assert (pupil.predict(observe(cells)) != demonstrator(cells)).any()  # cells it never saw

pupil.reset()  # new streams, starting anywhere
position = np.random.default_rng(0).integers(0, 5, 32)
for moment in range(32):  # then the brain's own path
    shown = observe(position)
    action = pupil.act(shown, greedy=True)  # its own move is executed; nothing awaits reward
    pupil.lesson(shown, demonstrator(position))  # the demonstrator labels the state reached
    position = move(position, action)
assert (pupil.predict(observe(cells)) == demonstrator(cells)).all()
```

Watching alone, the pupil copies the two cells the demonstrator visits and answers the others
wrongly; on its own path it reaches those cells and learns them. The labels must come from the
demonstrator the brain watched: in the measurement of
[issue 215](https://github.com/muellerberndt/cadence/issues/215), corrections from a different
policy after a bootstrap by watching lowered the agreement with the watched demonstrator. Teach
many rows at once with the normalized teaching founder [above](#rates-under-normalization).
