# Compose a brain

`Brain.compose` builds **System 1**: one continuing neural graph with senses,
reciprocal processing regions, a motor choice, a working trace and an associative
memory, all settling together. Optional observer regions add **System 2** feedback
inside the same graph. Keep one composed brain through acquisition, use and
correction; the [world-model guide](world-model.md) explains that life.

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, arousal=True)
observation = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.live(observation)
assert action.shape == (1,)
```

`live` takes one stream's observation of shape `(1, inputs)` and returns one action
index ([continuous interaction](continuous.md) covers `live`, `step` and saving).

## Brain

The composed graph, in the order its regions are developed:

| Region | Neurons | Wiring | Home |
| --- | --- | --- | --- |
| `sensory`, or `visual` for an image | `inputs`, the `code=units` code, or the visual cortex | Projects to the first module; initial scale `sensory_scale` | [image inputs](#image-inputs) |
| `module_0` ... | `modules[:-1]` | A chain of reciprocal projections | this page |
| `association` | `modules[-1]` | Reciprocal with `motor` | this page |
| `prefrontal` | one per association neuron | The working trace of the settled association, held at `working_memory_amplitude` 3.0, read through a plastic projection of initial scale `working_memory_scale` 1.2 | [memory](memory.md#working-trace) |
| `motor` | `actions`, in `slots` | Lateral inhibition within a slot | [motor competition](#operating-point-and-motor-competition) |
| `observer_0` ... | `observers` | Reciprocal with every module, `motor` and earlier observers | below |
| `efference` | `actions` | With `efference_amplitude` above 0 | [memory](memory.md#efference-copy) |
| `history` | sensory width per decay | With a nonempty `history_decays` | [memory](memory.md#heard-history) |
| `recall`, `next` | observation-code width, or `modules[-1]` with `recall_value="cortex"` | With `recall=True`; `next` unless `recall_next=False` | [recall memory](recall.md) |

The hippocampus is a `SynapticMemory` from the observation code to the motor drive,
with `memory_choice=True` the [episodic memory](episodic.md), and with
`episodic=False` absent. The default `modules=(64,)` is one region settling
recurrently with the others; in `modules=(16, 8)` the last entry is the association
region and sets the working trace's width. An action is issued only from a state,
observers included, that meets the full-equation tolerance
([qualification and refusal](continuous.md#qualification-and-refusal)).

```python
recursive = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), observers=(8, 4), seed=7, arousal=True,
)
assert recursive.live(observation).shape == (1,)
```

Observers settle with the base and never override a finished answer. The wiring
permits recursive correction without teaching it; exact state-and-error readback
belongs to the [population solver](equilibrium/index.md).

## Image inputs

`inputs=(height, width)` or `(height, width, channels)` makes the sensory region a
`visual_cortex`: `features` maps (8) of `field` by `field` receptive fields (3) feed
the first module. Observations may be flat `(1, height * width * channels)` or keep
the image shape.

```python
eyes = Brain.compose(inputs=(8, 8), actions=3, features=4, seed=1)
assert eyes.act(np.random.default_rng(0).random((1, 8, 8))).shape == (1,)
```

The memories key on the pixels. With `code=units` they key on a fixed sparse code
and the cortex reads the pixels ([the observation code](memory.md#the-observation-code)).

## Defaults and expert overrides

Set only the values you intend to change; the rest keep their composed founders, which
are the control for any change you measure.

```python
tuned = Brain.compose(
    inputs=4, actions=2, arousal=True,
    sensory_scale=2.0, actor_eta=0.1, actor_eta_bias=0.01, arousal_youth=30,
)
settings = tuned.describe()
assert settings["genes"]["actor_eta"] == 0.1
assert settings["initialization"]["composition"]["sensory_scale"] == 2.0
```

`describe()` returns JSON-safe `layout`, the effective `genes`, the resolved
configurations and `initialization` (construction choices, not current weights).

| Setting | Composed founder | Changed at | Home |
| --- | --- | --- | --- |
| `modules`, `observers` | `(64,)`, `()` | construction | [map](#brain) |
| `lateral`, `slots`, `resting_bias` | -0.5 up to 8 actions and 0.0 above, `1`, `0.0` | construction | [motor competition](#operating-point-and-motor-competition) |
| `sensory_scale`, `features`, `field`, `seed` | `1.0`, `8`, `3`, `0` | construction | [image inputs](#image-inputs) |
| `learning_beta`, `learning_eta`, `learning_eta_bias`, `temperature` | `0.1`, `0.5`, `0.05`, `0.2` | retune | [learning](learning.md#7-every-knob) |
| `learning_free_steps`, `learning_nudged_steps`, `learning_tolerance`, `learning_qualified`, `learning_damping` | `1024`, `12`, `0.003`, `False`, `3` | retune | [learning](learning.md#qualified-teaching-budgets) |
| `learning_centered`, `learning_nudge`, `learning_decay`, `learning_scale_cap`, `learning_surprise` | `True`, `"cross_entropy"`, `0.0`, `8.0`, `False` | retune | [learning](learning.md#7-every-knob) |
| `learning_normalize`, `learning_normalize_floor`, `learning_momentum` | `0.0`, `0.001`, `0.9` | retune | [learning](learning.md#rates-under-normalization) |
| `actor_gamma`, `actor_lam`, `actor_eligibility_steps`, `actor_eta`, `actor_eta_bias`, `actor_eta_critic` | `0.9`, `0.8`, `12`, `1.0`, `0.05`, `0.3` | retune | [reward](reward.md#eligibility-traces-and-dopamine) |
| `actor_normalize`, `actor_momentum` | `0.0`, `0.0` | retune | [reward](reward.md#rates-under-normalization) |
| `actor_dopamine_cap`, `actor_dopamine_center`, `actor_dopamine_floor`, `actor_center_scale`, `actor_critic_normalize`, `actor_critic_signal` | `1.0`, `0.0`, `0.0`, `True`, `True`, `"auto"` | retune | [reward](reward.md#centered-dopamine-and-selective-activity) |
| `arousal` | off; `True` is `ArousalConfig()` | construction (switch), retune (fields) | [continuous](continuous.md#routine-and-repair-live) |
| `arousal_threshold`, `arousal_decay`, `arousal_tolerance`, `arousal_floor`, `arousal_fast`, `arousal_slow` | `0.2`, `0.9`, `2.0`, `0.1`, `0.05`, `0.005` | retune | [continuous](continuous.md#routine-and-repair-live) |
| `arousal_heat`, `arousal_youth`, `arousal_value_surprise`, `arousal_record_surprise`, `arousal_need`, `arousal_need_scale` | `2.0`, `100`, `1.0`, `0.0`, `0.0`, `0.0` | retune | [continuous](continuous.md#routine-and-repair-live) |
| `working_memory_*` | scale `1.2`, amplitude `3.0`, decay `0.0`, focus `0.0` | scale at construction | [memory](memory.md#working-trace) |
| `efference_*` | amplitude `0.0` (absent), decay `0.2` | a positive amplitude at construction adds it | [memory](memory.md#efference-copy) |
| `history_*` | decays `()` (absent), amplitude `3.0`, scale `1.2` | decays and scale at construction | [memory](memory.md#heard-history) |
| `episodic`, `memory_rate`, `memory_decay`, `memory_amplitude`, `consolidation` | `True`, `1.0` (`0.5` with `memory_choice`), `0.9`, `1.0`, `0.05` | `episodic` at construction | [memory](memory.md#associative-store-the-hippocampus) |
| `code` | unset: the observation itself | construction | [memory](memory.md#the-observation-code) |
| `memory_choice` and the other `memory_*` episodic settings | `False` | construction | [episodic memory](episodic.md) |
| `recall`, `recall_scale` and every `recall_*` | `False` | see the page | [recall memory](recall.md) |
| `backend`, `device` | `"cpu"`, `None` | construction | [backends](backends.md) |

Override rules:

- Every field of `LearnerConfig`, `ActorCriticConfig` and `ArousalConfig` is a
  `learning_<field>`, `actor_<field>` or `arousal_<field>` name. `temperature` aliases
  `learning_temperature` and sets both the teaching softmax and the sampled policy.
- Unknown names and the ambiguous bare `eta` or `eta_bias` raise `TypeError`.
- `arousal=True` uses the founders, a mapping such as `arousal={"heat": 1.0}` patches
  them, an `ArousalConfig` replaces them, and `arousal_*` names need one of the three.
  Without `arousal`, `live` raises `ValueError`.
- A changed rate keeps the existing bias rate; `actor_eta_bias=None` (or
  `learning_eta_bias=None`) derives one tenth of the new rate.
- A setting for an absent component, such as `memory_rate` with `episodic=False`, has
  no effect; `describe()` omits it and `retune` refuses it.
- `learning=LearnerConfig(...)` and `reward=ActorCriticConfig(...)` replace the whole
  configuration, and a fresh config replaces every founder: `LearnerConfig()` has
  `eta=0.2`, `momentum=0`, `normalize=0` (raw steps), budgets `100/50` and tolerance
  `1e-4`; `ActorCriticConfig()` has `gamma=0.99`, `lam=0.9` and rates `0.5/0.05/0.05`.
  Use named overrides for a one-field edit.

## Retune the same life

`retune` changes named settings on a living brain, for example from practice to
ordinary interaction:

```python
tuned.retune(
    arousal={"need": 0.0, "heat": 0.0},
    temperature=0.2,
    actor_eta=0.03, actor_eta_bias=0.003,
)
```

It takes the same names, validates the whole request before installing anything,
runs no moment and returns the same brain. Parameters, optimizer history, activity,
traces, records, random state and pending feedback are kept, and new rates apply from
the next learned outcome. It raises `ValueError` for a construction setting (those
marked construction above), for a missing component (arousal included), and for a
change of `learning_beta` while a sampled action awaits feedback, whose contrast
belongs to the old scale; learn that outcome first. With `recall_address_reward`
above zero the same holds for `recall_address`, `recall_address_step` and
`recall_address_reward` ([recall memory](recall.md)).

`reset_arousal=True` resets the arousal level and reward references and keeps its age
and work counters. An adult brain at the founder `need=0` is then roused only by
surprise at its own forecasts, and a positive `need_scale` wants nothing until outcomes
enter the new stream's reward scale; where the world may stop paying, leave it off or
set a positive `need` ([reset and save](continuous.md#reset-and-save)).

## Choose settings for the application

| Application | Interface |
| --- | --- |
| Vector senses, several motors | One motor slot per motor, `live` and `retune`, each real action paired with its outcome across stage changes, as in the [Robot Arena demo](https://github.com/muellerberndt/cadence-demos/tree/main/robot-arena). |
| Pixels and a teacher | Image `inputs`, teaching through `step`; teaching rates are `learning_*`, reward rates `actor_*`. See the [Atari demo](https://github.com/muellerberndt/cadence-demos/tree/main/atari-arcade). |
| Retained events | The separate [`RecordPatchNet`](record-patch.md) with its own rates and guarantees, as in the [Amen demo](https://github.com/muellerberndt/cadence-demos/tree/main/amen). |

Gain, bias and solver steps are distinct levers: gain scales synaptic drive in the
whole graph, a population bias moves selected neurons' operating points, and steps and
damping change only how the equations are solved. Check candidate biases with the
[calibration guide](learning.md#calibrating-the-operating-point), choose on development
inputs and compare against the founders on your own task ([evaluation](evaluation.md)).

## Operating point and motor competition

`lateral` is the signed weight between each pair of motor neurons in a slot; `0.0`
removes those synapses and keeps the association and motor feedback. Left unset it is
-0.5 up to 8 actions and 0.0 above, because from about 12 actions the undamped free
solve stops settling under -0.5. The `cortex`, `motor_cortex` and `layered` factories
default to `0.0`.

`slots` groups the motor neurons into readouts that settle together, one softmax each:
a count of equal groups or one size per group. An action with parts (kind, pitch,
octave) is one slot per part. `act` and `step` return one index per slot, the unset
`lateral` follows the largest slot, and `act(temperature=...)` takes one value or one
per slot. `save` and `load` carry the grouping.

```python
chord = Brain.compose(inputs=4, actions=5, slots=(3, 2), seed=3)
chosen = chord.act(observation, temperature=[0.5, 0.1])
assert chosen.shape == (1, 2)
```

`resting_bias` (finite, nonnegative, default `0.0`) initializes the bias of the
modules, association, observers and other named populations. The `sensory`, `visual`,
`prefrontal`, `efference`, `history` and `motor` families stay at zero, as do `recall`
and `next` with `recall=True`, even where another population aliases them. A positive
bias can reduce silence under some random drives; compare it against zero. Biases stay
plastic: `brain.resting_bias` records the choice, `brain.brain.bias` holds the learned
values, and loading restores them. A signed neuron below rest responds to input and
contributes to teaching, so the share of nonpositive emissions is an activity
diagnostic (`tests/test_operating_range_function.py`).

## One experience step by hand

`step` learns from the previous outcome and chooses the next action. Use `act` and
`learn` when the body manages that timing:

```python
body = Brain.compose(4, 2, modules=(16, 8), seed=7)
chosen = body.act(observation)
reward = (chosen == 0).astype(float)  # action 0 earns one unit
following = np.array([[0.0, 1.0, 0.0, 0.0]])
body.learn(reward, np.array([False]), following)
next_action = body.act(following)
```

Reward and termination describe the preceding executed action; a demonstration labels
the current observation. Keep batch-row identities fixed until `reset()`. The
[continuing brain example](../examples/continuing_brain.py) runs the full reward,
teacher and checkpoint loop.

## Genome, development, brain

Use a `Genome` for named regions, custom projections or evolved wiring:

```python
import cadence as cd
from cadence.regions import cortex, motor_cortex

genome = cd.Genome(
    regions=(cd.Region("senses", 5), cortex(16), motor_cortex(3)),
    projections=(
        cd.Projection("senses", "association", reciprocal=False),
        cd.Projection("association", "motor"),
    ),
)
network = cd.NeuralGraph(cd.develop(genome, seed=0), cd.learning_neuron_model())
```

This graph has no motor competition (`motor_cortex` defaults to `lateral=0.0`) and no
Brain loop. Three entry points give one:

- `Brain.genome(inputs, actions, hidden=64, density=1.0, lateral=None,
  working_memory=False, memory_scale=12.0, features=8, field=3, seed=0, slots=1)`
  returns the single-region default layout as a `Genome` to edit or evolve.
- `Brain.build(...)` develops it and wraps it, adding `episodic=True` and constructor
  options; it has no trace unless `working_memory=True` ([memory](memory.md#working-trace)).
- `Brain(connectome, ...)` wraps any connectome with `association`, `motor` and
  `sensory` (or `visual/input`) populations and takes the table's constructor options;
  `inputs=` is the observation width when the sensory population is a `code=units` code.

Both constructors keep the table's learning, actor, arousal and memory founders.
`compose` alone fills in the episodic founders: with `Brain.build` or `Brain(...)`,
`memory_choice=True` starts its context, replay, demonstrations and separation at
zero unless you pass them ([episodic memory](episodic.md)).

[Write a cortex](cortex.md) covers regions, ports and projections;
[evolution](evolution.md) covers genomes and selection.

## The brain in a browser page

The library ships no browser page, environment or body. For an application display,
`record_settlements(callback, label=)` sends every actual settlement inside the
context to the callback, including those inside learning and imagination, with the
potential, activation and adaptation at every iteration:

```python
records = []
with cd.record_settlements(records.append, label="display"):
    brain.live(observation)
assert records and records[0].label == "display"
```

Recording copies every iteration and costs time, so use it for a display or an audit
([record every settling step](api.md#record-every-settling-step)). The public
[cadence-demos](https://github.com/muellerberndt/cadence-demos) repository holds
application pages that run or draw a brain.
