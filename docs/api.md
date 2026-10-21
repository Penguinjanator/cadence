# API reference

One entry per public name: signature, defaults, behaviour in a few lines, refusals, and
the guide that explains it. `Brain.compose` is the entry point; pass optional arguments by
keyword.

| Section | Names |
| --- | --- |
| [Brain](#brain-cadence) | `Brain`, its methods, attributes and checkpoint formats |
| [Arousal](#arousal-cadencearousal) | `ArousalConfig`, `Arousal` |
| [Streams](#streams-cadencestream) | `Trace`, `Efference`, `History`, `Recall`, `FastSynapses`, `SynapticMemory`, `PatternSeparator` |
| [Learning](#learning-cadencelearning), [checkpoints](#checkpoints-cadencecheckpoint), [reward](#the-agent-cadenceplasticity) | `LearnerConfig`, `Learner`, `GraphHolder`, `LearningPhaseError`, graph factories and diagnostics, `save`, `load`, `ActorCritic`, `ActorCriticConfig` |
| [The graph](#neuralgraph-cadence) | `NeuralGraph`, `BrainState`, `Equilibrium`, `Nudge`, `Connectome`, `NeuronModel`, `Adaptation`, blocks, `Certificate`, the neuron-by-neuron reference |
| [Wiring and instruments](#regions-and-genome-cadenceregions-cadencegenome) | `Region`, `regions`, `Projection`, `Genome`, `develop`, `evolve`, `genes`, `Receipt` |
| [Recording](#record-every-settling-step) | `record_settlements` |

`cadence.__version__` is the installed release string.

## Brain (`cadence`)

### Construction

```text
Brain.compose(inputs, actions, *, connectome=None, modules=(64,), observers=(),
              lateral=None, sensory_scale=1.0, working_memory_scale=1.2,
              history_scale=1.2, recall_scale=1.2, features=8, field=3, code=None,
              learned_code=None, seed=0, slots=1, **options) -> Brain
```

The composed System 1 brain, with optional System 2 `observers` in the same graph.
`inputs` is a vector length or an image shape `(height, width[, channels])`; `features` and
`field` size the visual cortex of an image. `modules` are the widths of the reciprocal
processing chain, the last the association region; each `observers` width adds a region
connected both ways to every module, the motor region and earlier observers. `lateral` is
the weight between motor neurons of one slot (unset: -0.5 up to 8 actions in the largest
slot, else 0.0); `slots` is a count of equal motor groups or one size per group, each read
as its own softmax. `code=units` keys the memories on a fixed sparse
code of that many units and, for a vector brain, makes the sensory region that code
([the observation code](memory.md#the-observation-code)). `learned_code=units` (a vector
brain, without `code`) adds a `code` region of that many neurons that settles with the brain
to a learned sparse code of the observation, read by the cortex and keyed on by the
memories, with the genes `learned_code_top_down`, `learned_code_rate`,
`learned_code_predictor`, `learned_code_threshold`, `learned_code_threshold_rate`,
`learned_code_target`, `learned_code_repair`, `learned_code_silence`,
`learned_code_place`, `learned_code_output` and `learned_code_identity` ([the learned code](code.md)). The `*_scale` arguments are initial
projection magnitudes, fixed at construction. `options` takes every constructor keyword below
and every `learning_<field>`, `actor_<field>` and `arousal_<field>` name; with
`memory_choice=True` it fills in the episodic founders. The region map and the founder table
are in [compose a brain](brain.md#defaults-and-expert-overrides); the memory settings are owned
by [memory](memory.md), [episodic memory](episodic.md) and [recall memory](recall.md).

Refusals: unknown names, the bare `eta` or `eta_bias`, and duplicate aliases raise
`TypeError`; a `recall_` name without `recall=True`, a setting outside its range, and
`arousal_*` names without `arousal` raise `ValueError`.

`connectome` wraps a custom anatomy with the same founders, the episodic founders filled in
under `memory_choice=True`: its `sensory` (or `visual/input`) population holds the inputs (the
code's units with `code=units`) and its `motor` population the actions. It brings its own
layout, so `modules`, `observers`, `lateral`, the `*_scale` arguments, `features` and `field`
stay at their founders (`ValueError` otherwise). `Genome.composed` returns the layout
`compose` develops, to edit or evolve ([genome, development,
brain](brain.md#genome-development-brain)).

```text
Brain(connectome, *, episodic=True, consolidation=0.05, memory_decay=0.9,
      memory_rate=1.0, memory_amplitude=1.0, memory_choice=False,
      memory_context=0.0, memory_replay=0, memory_demonstrations=0.0,
      memory_expansion=0, memory_winners=32, memory_fan_in=None,
      working_memory_decay=0.0, working_memory_amplitude=3.0,
      efference_decay=0.2, efference_amplitude=0.0,
      history_decays=(), history_amplitude=3.0, recall=False, learning=None,
      reward=None, slots=1, arousal=None, seed=0,
      backend="cpu", device=None, inputs=None, code=None, learned_code=None, **genes)
```

Wraps any connectome with populations `association`, `motor` and `sensory` (or
`visual/input`). It uses `prefrontal` for a working trace, `efference` (one neuron per motor
neuron) for an efference copy, `history` (one sensory-width block per entry of
`history_decays`; one block of `learned_code` units with a learned code) for a heard
history, with `recall=True` the `recall` and `next` regions, and with `learned_code=units` a
`code` population of that many neurons with its complete dictionary, lateral and predictor
synapses ([the learned code](code.md)).
`inputs` names the observation width when the sensory region is a `code` of that width.
`genes` carries the `recall_*` and `learned_code_*` settings. `learning` and `reward` default to the composed
`LearnerConfig` and `ActorCriticConfig`; `arousal` is `True`, a field mapping or an
`ArousalConfig`. With `memory_choice=True`, `compose` fills in `memory_context=0.5`,
`memory_rate=0.5`, `memory_replay=8`, `memory_demonstrations=1.0`, `memory_expansion=1024`
(0 with a code) and `memory_winners=24`; the constructor keeps its own defaults above, so
pass these here or use `compose` ([episodic settings](episodic.md#settings)).

| Rule | Raised |
| --- | --- |
| `memory_choice` needs `episodic=True` | `ValueError` |
| `memory_context`, `memory_replay`, `memory_demonstrations`, `memory_expansion` need `memory_choice` | `ValueError` |
| `memory_fan_in` needs `memory_expansion` (so `memory_choice` too) and at most the key width; a coded brain (`code` or `learned_code`) refuses both | `ValueError` |
| `learned_code` at least 2; a `learned_code_*` gene without it, `code` beside it, an image brain, a backend other than `cpu`, or a connectome without the region's populations and complete synapses | `ValueError` |
| `memory_context` in `[0, 1)`; `working_memory_decay`, `efference_decay` in `[0, 1)` | `ValueError` |
| a missing population that a switched-on component needs | `ValueError` |

The efference projection into `association` has the fixed initial scale 12.0; the working
trace's `working_memory_scale` founder is 1.2.

### Methods

| Method | Behaviour | Guide |
| --- | --- | --- |
| `live(observations, *, reward=None, done=None, decision_id=None) -> actions` | One moment of one stream; needs `arousal`. Calm: one greedy qualified settle, no learning. Aroused: samples, learns from the outcome, writes memory. Omitted `reward` with a pending action is a zero outcome. A wrong `decision_id` raises `ValueError` before any change. | [routine and repair](continuous.md#routine-and-repair-live) |
| `wait(observations) -> None` | Senses a moment while a `live` action awaits its outcome: settles, advances the traces, writes the recall slot; issues nothing and learns nothing. Raises `ValueError` without arousal or with several streams, `RuntimeError` without an awaited action or when the settle cannot qualify. | [outcomes that arrive later](continuous.md#outcomes-that-arrive-later-wait) |
| `step(observations, *, reward=None, done=None, teacher=None, salience=None, bootstrap=None) -> actions` | Learns from the preceding outcome, teaches `teacher` labels of the present observation, samples the next action. Omitted `reward` is a real zero. | [explicit learning](continuous.md#explicit-learning-with-step) |
| `lesson(observations, labels) -> dict` | The teaching part of `step` without the act; records the demonstration in an episodic memory and runs the recall address lesson. Refused with `RuntimeError` while a sampled action awaits its outcome; a refused phase raises `LearningPhaseError` and changes nothing. | [explicit learning](continuous.md#explicit-learning-with-step) |
| `act(observations, *, greedy=False, temperature=None) -> actions` | One qualified free settle and a sampled (or greedy) choice, one index per slot; advances the working trace. `temperature` is one value or one per slot. Exhaustion raises `RuntimeError` before any change. | [one experience step by hand](brain.md#one-experience-step-by-hand) |
| `learn(reward, done, next_observations, *, bootstrap=None, salience=None) -> dict` | Feedback for the pending sampled action: hippocampal record, actor and critic update; `done` rows reset their traces. A refused feedback solve rolls everything back and keeps the action pending. | [reward](reward.md#decision-and-reset-contract) |
| `imagine(observations, *, budget=1024, tolerance=1e-6) -> tuple[Equilibrium, ...]` | A private branch over a sequence of batches: memories read, nothing written, live state unchanged. A refused phase ends the branch and is the last entry. | [private imagination](continuous.md#private-imagination) |
| `fit(observations, labels, *, epochs=30, batch=32) -> list[float]`, `predict(observations)`, `accuracy(observations, labels)` | Independent samples, without traces or recall; `fit` resets the streams first, arousal included, and returns training accuracy per epoch. Noninteger labels and mismatched batches raise before any update. | [independent-sample controls](continuous.md#independent-sample-controls) |
| `retune(*, reset_arousal=False, **genes) -> Brain` | Changes mutable settings in the same life, atomically. Refuses construction settings, absent components, and `learning_beta` (or the recall address genes under `recall_address_reward`) while a sampled action awaits feedback. | [retune the same life](brain.md#retune-the-same-life) |
| `reset(*, keep_memory=False, keep_arousal=False)` | Clears per-stream state, eligibility and reward centering; keeps records and weights. Arousal begins the new stream calm, its reward scale cleared (a `need_scale` want returns once outcomes enter it), and keeps its age; `keep_arousal=True` keeps the whole arousal state, its level included, as saving `brain.arousal.to_dict()` before the reset and restoring it with `Arousal.from_dict` after it does. `keep_memory=True` keeps the recall slots and ends each stream's episode. Both keywords must be booleans, checked before anything changes. | [reset and save](continuous.md#reset-and-save) |
| `describe() -> dict` | JSON-safe `layout`, effective `genes`, resolved configurations, memory settings, `initialization` and `pending_feedback`. | [defaults](brain.md#defaults-and-expert-overrides) |
| `observation_code(observations) -> (batch, code_width)` | The unit-length key every memory reads and writes; with a learned code, the code the observations settle to without a context. | [the observation code](memory.md#the-observation-code), [the learned code](code.md) |
| `stimulus(observations, *, memory=True) -> drive` | The drive of a batch; with `memory` the trace, copy, history, recall and hippocampal reads are added, and the recall address is fixed for the moment. | [memory](memory.md) |
| `parameters() -> int` | Actor and critic parameters, the consolidated memory matrix, a recall address map and a learned code's tables. | |
| `copy.deepcopy(brain)` | An independent copy: every array, store, state and random generator copied, the immutable connectomes shared. |
| `save(path) -> Path`, `Brain.load(path, *, backend="cpu", device=None, precision=None)` | The whole continuation, written atomically ([the format](#checkpoint-format)). `Brain.load` refuses a learner-only file; `Learner.load` reads either. | [reset and save](continuous.md#reset-and-save) |

Observations are a nonempty finite batch `(streams, inputs)`, images flattened per row or in
their shape. Keep row identities fixed until `reset()`.

### Attributes

| Attribute | Holds |
| --- | --- |
| `connectome`, `brain`, `learner` | the developed graph, the current `NeuralGraph` (`learner.brain`) and the `Learner` |
| `basal_ganglia` | the `ActorCritic` reading the association region |
| `memories` | the present memories by name, in the order the brain reads them ([`Memories`](#the-memories-cadencememories)) |
| `working_memory`, `efference`, `history`, `recall` | the `Trace`, `Efference`, `History` and `Recall`, each `None` when absent |
| `hippocampus` | the associative `SynapticMemory` (with `memory_choice` the episodic place records), or `None` with `episodic=False` |
| `context_memory` | the episodic context records with `memory_choice` and a positive `memory_context`, else `None` |
| `sparse_code`, `code_width`, `coded_cortex`, `inputs` | the `PatternSeparator` of `code=units` or `None`; the key width; whether the sensory region is the code; the observation width |
| `learned_code`, `code_index`, `prefrontal_index` | the `LearnedCode` of `learned_code=units` or `None`; neuron indices of the `code` and `prefrontal` populations (empty without them) |
| `sensory_index`, `association_index`, `motor_index` | neuron indices of those populations |
| `memory_choice`, `memory_context`, `memory_replay`, `memory_demonstrations` | the resolved construction settings |
| `arousal` | the stream's `Arousal`, or `None` |
| `pending_feedback` | read-only: the issued action awaits its outcome (`wait` keeps it) |
| `decision_id` | read-only: the `arousal.age` at which `live` issued the awaited action, or `None`; saved and restored |
| `last_arousal` | immutable snapshot of the latest `live` moment; `None` after construction, reset or load ([fields](continuous.md#routine-and-repair-live)) |
| `last_settlement` | immutable report of the latest free-answer solve, refused ones included: `operation`, `scope`, `qualified`, `row_qualified`, `residual`, `max_residual`, `steps`, `budget`, `tolerance`, `residual_checks`, `damping_halvings`, `stagnation_checks`, and with a learned code `code_steps`, `code_residual`, `code_halvings` and, for a brain with a place store, `context_free_steps` ([inspect the work](quickstart.md#inspect-the-work-of-answering)) |
| `last_learning` | the latest transition's learning report, teaching under `demonstration_*` keys |
| `rng` | the brain's `numpy.random.Generator` |

### Checkpoint format

`Brain.save` writes `cadence-brain/1`, one format whatever the brain holds:

| Members | What they hold |
| --- | --- |
| `meta` and the unprefixed arrays | the learner's checkpoint ([checkpoints](#checkpoints-cadencecheckpoint)) |
| `brain` | the metadata: the actor's config, both random states, the free phase and a pending action, the arousal and the action `live` issued with its forecast, the state a waiting stream sensed, a `code=units` code or a learned code, the composition, and under `memories` the settings of each memory the brain holds |
| `actor/`, `free/`, `pending/`, `moment/`, `prepared`, `lived/`, `awaiting/` | the critic and eligibility, the free phase, a pending action's phases and decision, the action `live` issued, the sensed state |
| `code/`, `learned_code/` | a `code=units` code's tables; a learned code's `dictionary`, `weights` and `thresholds` |
| `memories/working_memory/`, `memories/efference/` | each trace's `trace`, `last` and `cold` |
| `memories/history/` | the heard history's `trace` |
| `memories/recall/` | the slots, `held`, `previous`, the address map, and where set the actor's `trace` and `pending` contrast of the map, `ended` and the window's `past` |
| `memories/hippocampus/` | the associative store's ports, strengths, masses, consolidated records and separator |
| `memories/episodic/` | the episodic choice (`memory_choice`): the context store (`context_memory/`), the stream's `context` and `decision`, and the replay `path/` |

`Brain.load` reads it and, through 0.85.x, the formats of 0.84.3 (`cadence-generic/1` to `/9`);
a setting of 0.84.3 that this version removed loads at the value whose law this version keeps
and is refused otherwise with a message naming 0.84.3. Record and recall key widths are checked
against the code; code tables that do not read every ON and OFF channel twice with positive
weights are refused, as is a code whose metadata lacks the `cover` field; a recall window of
another length is refused; learned-code tables that disagree with the saved synapses and
biases of the code region are refused.

## Arousal (`cadence.arousal`)

```text
ArousalConfig(threshold=0.2, decay=0.9, tolerance=2.0, floor=0.1, fast=0.05,
              slow=0.005, heat=2.0, youth=100, value_surprise=1.0, need=0.0,
              need_scale=0.0)
Arousal(config=None)
```

`ArousalConfig` holds the genes of the law; `to_dict()` returns them and
`ArousalConfig.space()` declares their mutation space for [`genes`](evolution.md#any-genome),
without `need_scale`, which a space of your own can add
(`{**ArousalConfig.space(), "need_scale": ("linear", 0.02, 0.0, 1.0)}`).
Construction refuses negative values, `decay` outside `[0, 1)`, `fast` or `slow` outside
`(0, 1]`, `fast < slow` and a `youth` that is not a nonnegative integer. The law and the
meaning of each gene are in [routine and repair](continuous.md#routine-and-repair-live).

`Arousal` is one stream's state; `Brain.live` runs it, and it can be used alone.

| Member | Behaviour |
| --- | --- |
| `outcome(error, reward, *, own=True, learned=True) -> (surprise, want)` | Applies one outcome. Only `own` outcomes can surprise and enter `usual`; every outcome moves `recent` and `longrun`; `learned` outcomes, and any in youth or that leave the stream aroused, enter `scale`. An outcome whose statistics cannot stay finite raises `ValueError` and changes nothing. |
| `lived(sweeps, learning_sweeps=0)` | Counts one moment in `moments`, `sweeps` (per mode), `learning_sweeps` and `age`. |
| `reset()` | Begins another stream calm; clears the averages and their counts, the reward scale among them (so a scaled need is zero until outcomes enter it again), keeps `age` and the work counts. `Brain.reset(keep_arousal=True)` leaves the state as it is. |
| `to_dict()`, `Arousal.from_dict(values)` | The complete state with every gene, marked `cadence-arousal/4`. |
| `level`, `age`, `moments`, `sweeps`, `learning_sweeps`, `outcomes`, `rewards`, `spreads`, `config` | State and counters: `outcomes` counts own outcomes, `rewards` all outcomes. |
| `aroused`, `mode`, `want`, `heat`, `usual`, `recent`, `longrun`, `scale` | Read-only: `aroused` while `level >= threshold` or within `youth`; `heat` is `1 + config.heat * want`. |

`from_dict` reads `cadence-arousal/4` and the states of 0.84.3: `cadence-arousal/3`,
`cadence-arousal/2`, `cadence-arousal/1` and unversioned states. It requires every gene and
continuation field, except that an unversioned state with the gene set before `need` loads
with `need=0` and a state under a format before `cadence-arousal/3` loads with
`need_scale=0`, left out or written as zero; a positive `need_scale` under an earlier format
is refused. A `cadence-arousal/1` or unversioned state without `rewards` takes it from
`outcomes`. The states of 0.84.3 carry the record channel, a second surprise at the outcome
the action's record forecast, which this version removed: one weighed at the founder zero
loads without its readings, and one weighed above zero is refused with a message naming
0.84.3.

## The memories (`cadence.memories`)

```text
Memories(members)
Moment(observations, code, place, stores=True)
EpisodicRecords(place, temporal, *, width, offsets, share, steps, demonstrations)
```

`brain.memories` is a `Memories`: a read-only mapping from a memory's name to the memory, the
present ones only, in the order `working_memory`, `efference`, `history`, `recall`,
`hippocampus`, `episodic`. Every member keeps the contract `Memory`:

| Member | Behaviour |
| --- | --- |
| `stimulus(drive, moment) -> drive` | the drive of a moment before the settle with what the memory adds: the trace or history read, the recall memory's address of `moment.code`, the store's read under `moment.place` (of the drive without one; nothing when `moment.stores` is false, as under an episodic vote) |
| `branch()` | a private copy for an imagined branch, which reads the live state and writes none |
| `begin(keep=False)` | the streams begin afresh as `Brain.reset` begins them: the stores keep their records, and with `keep` the recall memory its slots |
| `reset(batch, rows=None)`, `keep(rows)` | fresh streams, all or `rows`; the state of `rows` kept |
| `to_dict()`, `arrays()` | the settings and counts, and the arrays a checkpoint keeps under `memories/<name>/` |

`Memories.stimulus(drive, moment)` applies every member in order.

With `memory_choice=True` the member `episodic` is an `EpisodicRecords`. `place` is the
associative store (`brain.hippocampus`) and `temporal` the context records
(`brain.context_memory`, None at `memory_context=0`); `share`, `steps` and `demonstrations`
are `memory_context`, `memory_replay` and `memory_demonstrations`. Per stream it holds
`context` (the temporal context after the last decision), `decision` (that decision's key)
and `paths` (the recent decisions, one per situation left). `vote(code, key)` is the mean of
the place read and the context read that joins the motor readout; `key(code)` is the key of a
decision taken now; `outcome(...)` records a known outcome and replays a reward along the
path; `remember(...)` writes both kinds of record; `reset(batch, rows)` begins the next
attempt of `rows` with no context and no path. Its `stimulus` returns the drive unchanged:
the vote joins after the settle.

## Streams (`cadence.stream`)

The per-stream memories of the composed brain, usable alone on any graph. The composed
brain's memories are explained in [memory](memory.md); the standalone stores in
[standalone memory stores](records.md).

```text
Trace(connectome, decay=0.5, amplitude=1.0, source="hidden", target="context")
Efference(connectome, decay=0.5, amplitude=1.0, source="motor", target="efference")
```

Methods: `reset(batch, rows=None)`, `keep(rows)`, `stimulate(drive)`, `update(state)`,
`ringing(floor=0.1)`, `to_dict()`; `Efference.issue(command)` (its `update` raises).
Behaviour: [traces](records.md#traces); in the brain: [working trace](memory.md#working-trace),
[efference copy](memory.md#efference-copy).

```text
History(connectome, decays, amplitude=3.0, source="sensory", target="history")
```

One block of `source` width per decay in `target`; `hear(observations)` moves block `d` by
`h <- d h + (1 - d) x`, and `stimulate(drive)` writes `amplitude * h` into the `target`
columns of a copy (another batch size reads silence). `reset(batch, rows=None)`, `keep(rows)`, `trace` `(batch, blocks * source)`,
`to_dict()`. In the brain: [heard history](memory.md#heard-history).

<a id="recall-memory-genes"></a>

```text
Recall(connectome, amplitude=3.0, temperature=0.25, heard=1.0, recency=0.25,
       capacity=64, next=True, address=0.01,
       address_temperature=0.1,
       address_step="lms", null=None, address_reward=0.0,
       window=1, window_decay=1.0,
       source="sensory", target="recall", inputs=0, writes=0)
```

The held-input index memory of each stream's recent moments. Each constructor argument is the
brain's `recall_<name>` gene; their meanings, founders and retune rules are the
[recall settings table](recall.md#settings), the mechanism is
[how a moment is addressed](recall.md#how-a-moment-is-addressed). `inputs` is the key width
when it differs from the `source` population (a coded brain passes the code width). The
settings of 0.84.3 that left in this version, each with its reading, are
[listed on the recall page](recall.md#removed-in-this-version); release 0.84.3 takes them.

| Member | Behaviour |
| --- | --- |
| `stimulate(drive, observations)` | Addresses the moment, fixes the softmax over the slots (`attended`) and writes the read into `recall` and, with `next`, `next`. |
| `witness(observations, state)` | Writes the moment as a slot after its settle. |
| `learn(graph, learned) -> dict` | The address lesson after a teaching step; returns `address_rows`, `address_error`, `address_refused`. A nonfinite step is refused per row. |
| `contrast_rows(graph, plus, minus)`, `propose(step)`, `accept(proposal)` | Reward training of the address map at `address_reward > 0`; `ActorCritic` calls them. `contrast_rows` is `None` at 0. |
| `end(batch, rows=None)` | Ends episodes and keeps the slots, as `Brain.reset(keep_memory=True)` does; the latest slot learns no successor. |
| `weights(address)`, `ages()`, `readback(graph, region, movement)`, `activity(state)` | The read's slot shares, slot ages, and the association movement read back through the recall synapses. |
| `reset(batch, rows=None)`, `keep(rows)`, `snapshot()`, `restore(snapshot)`, `reading()`, `hold(reading)`, `imagined()` | Stream bookkeeping; `imagined()` is a copy that reads the live slots and writes none. |
| `Recall.founders()`, `Recall.options()`, `Recall.setting(name, value)`, `Recall.null_logit(null, temperature)`, `Recall.removed(name)` | The always-described genes, the five options saved only where they differ, validation of one gene, the null slot's logit, the refusal of a removed setting. |
| `RUNTIME`, `CONSTRUCTION`, `OPTIONS`, `CONSTRUCTION_OPTIONS`, `RETIRED`, `FORMAT`, `STEPS` | Class tuples: retunable genes, construction genes, the options and those fixed at construction, the settings this version removed, the keys the 0.84.3 format wrote for three of them with the one value at which such a file loads, and the allowed values of `address_step`. |
| `keys`, `values`, `nexts`, `held`, `previous`, `ended`, `past`, `w_x`, `w_c`, `b`, `address_width`, `neurons`, `streams`, `slots`, `attended`, `to_dict()` | State arrays and sizes. |

Construction and `Brain.retune` refuse a `null` whose logit is not finite.

```text
FastSynapses(pre, post, decay=1.0, rate=1.0, amplitude=1.0, normalize=False,
             replace=False, rule="hebb", separator=None, writes=0)
SynapticMemory(pre, post, decay=0.9, rate=1.0, amplitude=1.0, normalize=False,
               replace=False, rule="delta", separator=None, writes=0,
               consolidation=0.05)
PatternSeparator(inputs, expansion, winners, seed=0, center=0.0, fan_in=0, cover=0)
```

`FastSynapses` is one mutable `(pre, post)` matrix per stream: `observe(key, value,
write=None)`, `recall(key)`, `update(state, write=None, post=None)`, `read(drive)`,
`stimulate(drive, inplace=False, key=None)`, `reset(batch, rows=None)`, `keep(rows)`,
`key_width`, `to_dict()`. `rule="delta"` refuses `normalize`, `replace` and rates outside
`[0, 1]`. `SynapticMemory` (in `cadence.memory`) adds a persistent `consolidated` matrix
shared across streams: `observe(..., *, salience=None, value_mask=None)`, `reset` keeps the
consolidated synapses, `clear()` erases both, `salience_vector(salience, batch)` validates a
salience; it refuses any `rule` but `"delta"`, a `consolidation` outside `[0, 1]` and a
`separator` with nonzero `center`.
`PatternSeparator` codes keys by a fixed random projection keeping `winners` of `expansion`:
`code(key, learn=False)`, `habituate(keys)`, `arrays()`, `channels`, `to_dict()`; it raises
`ValueError` for nonpositive sizes, `winners > expansion`, a `fan_in` or `cover` without room,
and `center` outside `[0, 1)`. All three: [standalone memory stores](records.md); the
associative store: [memory](memory.md#associative-store-the-hippocampus).

```text
cadence.stream.columns(index) -> slice | ndarray
```

`columns` returns a slice for a contiguous index.

## The learned code (`cadence.code`)

```text
LearnedCode(inputs, units, context, model, seed=0, top_down=2.0, rate=0.125, predictor=0.0625,
            threshold=0.1, threshold_rate=0.000625, target=0.1, repair=1.0, silence=10.0,
            place="free", output="unit", floor=0.0, identity=0.9, cortex="code",
            scale="observation", held=True, reads="free")
```

The learned code region's tables and rules, built by `Brain.compose(..., learned_code=units)`
and held as `brain.learned_code`: `dictionary` `(inputs, units)` with unit-length columns,
`weights` `(units, context)` the predictor, `thresholds` `(units,)`, `read` (the gain the
sensory activation is read at, one over the activation one input at amplitude 1 gives, from
the model), the lateral `connectome`
and `graph` of the units (the sub-network a moment's code phase settles). `lateral()`,
`lateral_efficacy()`, `gram()`; `stimulus(sensory, context)`, `prediction(context)`,
`settle(stimulus, warm, *, budget, tolerance, damping=3) -> Equilibrium`,
`code_of(activation)` (the rectified code at unit length), `output_of(activation)` (what the
module reads with the unit read: the rectified code over its norm or over the floor), `residuals(sensory, activation,
context)`; `standing() -> Nudge | None` (the region's quadratic boundary toward zero at gain
`top_down` over its units); `stability(v) -> int` (the halvings of the step the settled
code's competition is stable under), `rest_stability() -> int` (the same over every state
the region can reach, from the dictionary alone) and `restep(halvings)`, `halvings`; `learn(sensory,
activation, context, free=None) -> bool` (one step of every table from a settled moment,
each row an event; `free` the context-free code, whose activity the thresholds' share reads
and whose residual the repair reads), `install()`,
`take(dictionary, weights, thresholds)`;
`arrays()`, `genes()`, `to_dict()`, `parameters()`, `nbytes()`, `work` (settles, rows, sweeps,
capped settles, updates, and `repairs`, the unit-rows the repair lowered), `last` (the latest phase's sensory activation, context, activation,
prediction and, at a learned moment, the context-free code `free`). `LearnedCode.founders()` and `LearnedCode.setting(name, value)` give and
validate the genes; `GENES`, `RUNTIME` and `CONSTRUCTION` name them. Guide:
[the learned code](code.md).

## Learning (`cadence.learning`)

```text
LearnerConfig(beta=0.1, eta=0.2, eta_bias=None, free_steps=100, nudged_steps=50,
              tolerance=1e-4, nudge="cross_entropy", temperature=0.2, normalize=0.0,
              normalize_floor=1e-3, momentum=0.0, scale_cap=8.0, qualified=False,
              damping=3, surprise=False)
Learner(brain, outputs, config=LearnerConfig(), plastic_synapses=None,
        plastic_neurons=None, reciprocal=True, tie_groups=None,
        synapse_rate=None, slots=1, updates=0, contrast_updates=0, standing=None)
```

Every field is explained in [every knob](learning.md#7-every-knob). `eta_bias=None` derives
`eta / 10`. Construction warns (`RuntimeWarning`) when `eta_bias` exceeds a positive `eta`,
when `normalize > 0` with a rate above 0.05
([rates under normalization](learning.md#rates-under-normalization)), and when
`qualified=True` with `nudged_steps < free_steps`
([qualified budgets](learning.md#qualified-teaching-budgets)).

| `Learner` member | Behaviour |
| --- | --- |
| `step(drive, labels, warm=None, weight=None, *, answer=None) -> (LearnedState, report)` | Free phase, nudged phase(s), one update. A refused qualified phase raises `LearningPhaseError` and changes nothing. |
| `free(drive, warm=None)`, `nudged(drive, free, target, sign=1.0, weight=None)`, `targets(labels)`, `nudge_for(target, beta, weight=None)` | One phase; the per-neuron target; the `Nudge`. |
| `standing`, `bounded(nudge)` | A standing boundary every phase carries (a `Nudge` whose anchor terms are held fixed through each solve; a composed brain's learned code sets it), and a phase's nudge with those terms joined; `None` without one. |
| `contrast(free, nudged, opposite=None)`, `contrast_rows(...)`, `update(free, nudged, opposite=None)`, `apply(delta_scale, delta_bias)` | The contrast, per row, and an update through masks, rates, tying and the cap. `update` does not check qualification. |
| `calibrate(drive, *, level=None, grid=None) -> float` | Picks the global gain by the free output operating point; reports in `last_calibration`; raises `RuntimeError` without a usable candidate and changes nothing ([calibration](learning.md#calibrating-the-operating-point)). |
| `predict(drive)`, `accuracy(drive, labels, batch=256)`, `parameters()`, `to_dict()` | Readout, score, plastic parameter count. |
| `save(path, *, compressed=True)`, `Learner.load(path, ...)` | [Checkpoints](#checkpoints-cadencecheckpoint). |
| `Learner.sharing(other, outputs, **options)`, `holder`, `brain` | A second learner on the same `GraphHolder` ([two learners](learning.md#two-learners-on-one-graph)). |
| `reverse`, `velocity`, `velocity_bias`, `second_moment`, `second_moment_bias`, `output_index`, `updates`, `contrast_updates` | Reverse-synapse index (-1 for none), optimizer state, counters. |

Construction refuses duplicate reciprocal pairs (`ValueError`; merge with
`Connectome.from_synapses` or pass `reciprocal=False`).

`GraphHolder(brain)` holds the graph that learners on it share. `LearningPhaseError(phase,
phases, hint=None)` is a `RuntimeError` carrying `phase`, `phases`, `report` and `hint`.
`cadence.learning.LearnedState(free, nudged, opposite=None, taught=None)` is what `step`
returns; under `surprise`, `taught` marks the taught rows.

| Function | Behaviour |
| --- | --- |
| `layered(inputs, hidden, outputs, *, density=0.3, feedback=1.0, lateral=0.0, seed=0, count=1.0, init=1.0, skip=False, skip_init=None, excitatory_forward=False) -> Connectome` | Populations `input`, `hidden`, `output` ([every knob](learning.md#7-every-knob)). |
| `embedded(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0) -> (Connectome, tie_groups)` | A one-hot token window through one shared embedding. |
| `learning_neuron_model(gain=1.0, *, slope=1.0, leak=0.1, dt=0.5, stimulus_amplitude=1.0) -> NeuronModel` | Responsive starting settings for local learning. |
| `calibrate_bias(brain, drives, targets, *, per_neuron=False, rounds=3, span=(-6.0, 6.0), iterations=16, steps=100, tolerance=1e-4, qualified=False, damping=3, report=None) -> ndarray` | Candidate biases by coordinate bisection toward target mean activations; the graph is unchanged. `qualified=True` raises `RuntimeError` on a refused solve; nonfinite states always raise. |
| `naive_efficacy(connectome, plastic) -> ndarray` | Efficacies giving every plastic synapse class the same weight. |
| `preflight(brain, outputs, plastic, drives, *, level=0.5, saturation=0.02, shared_max=0.25, eligibility_min=10, steps=100, tolerance=1e-4) -> dict` | Pre-lesson checks: saturated readouts, shared sender codes, too few eligible synapses, indistinct drives; `warnings` names the repair ([check before the first lesson](reward.md#check-before-the-first-lesson)). |
| `seam_report(connectome, pre, post) -> dict` | `classes`, `synapses`, `coverage_pre`, `classes_per_post`, `median_count` of a plastic seam ([connectomes](connectomes.md)). |

## Checkpoints (`cadence.checkpoint`)

```text
save(learner, path, *, compressed=True) -> Path
load(path, *, backend=None, device=None, config=None, precision=None) -> Learner
```

Also `Learner.save` and `Learner.load`. One `.npz` with the connectome, efficacies, gains,
biases, neuron model, configuration, outputs and slots, plasticity masks, ties, synapse rates,
optimizer state and update counts, replaced atomically. `config` replaces the saved
configuration; `backend`, `device` and `precision` may differ from the saved ones. Separate
`FastSynapses`, `Trace` and `ActorCritic` objects are not saved; use `Brain.save` for a
composed brain. The format is `cadence-checkpoint/2`; `/1` loads, and a `cadence-checkpoint/3`
file is refused with a message naming the release that reads it.

## The agent (`cadence.plasticity`)

```text
ActorCritic(learner, critic, config=None, seed=0)
ActorCriticConfig(gamma=0.99, lam=0.9, eta=0.5, eta_bias=None, eta_critic=0.05,
                  dopamine_cap=1.0, eligibility_steps=None)
```

The reward learner of a stream: a `Learner` whose outputs are the actions, a learned linear
critic on the `critic` neurons, eligibility traces and a dopamine signal. Fields, the report
and the decision contract are in [learning from reward](reward.md#settings).

| Member | Behaviour |
| --- | --- |
| `act(drive, greedy=False, temperature=None, *, bias=None) -> action` | One free phase and a draw per slot. `bias` adds a finite `(batch, motor)` vote before each softmax; eligibility scores the readout's own policy. A greedy act drops the pending action, so `learn` refuses until a sampled act; the decaying traces are kept. |
| `learn(reward, done, next_drive, bootstrap=None, *, observed=None, warm=None) -> dict` | The TD error made into dopamine and written through the eligibility. At least one row must be `observed`. `bootstrap` applies only to `done` rows: a truncated row passes `done=True` with the value of its last observation (`value_of`) as `bootstrap`; other `done` rows bootstrap from zero. |
| `fade(done=None)` | One moment without a new eligibility: every trace decays by `gamma * lam`. |
| `probabilities(state, temperature=None, *, bias=None)`, `value(state)`, `value_of(drive)`, `settle(drive)` | Policy, critic value, a cold settle. |
| `reset()`, `parameters()`, `to_dict()` | Clears cache, eligibility and salience; keeps learned parameters. |
| `state`, `salience`, `memory`, `trace`, `trace_memory` | The latest free phase; per-sender salience weights; a memory with learned parameters (`Recall`) and its eligibility. |

Construction warns when `eta_bias` exceeds a positive `eta`.

## NeuralGraph (`cadence`)

```text
NeuralGraph(connectome, neuron_model, *, backend="cpu", efficacy=None,
            log_gain=None, bias=None, device=None, dense_limit=2048,
            layout=None, precision=None)
```

The runnable graph of graded neurons ([how the graph settles](concepts.md)). `efficacy`
defaults to the connectome's signs, `log_gain` and `bias` to zero. Graphs whose dense blocks
fit in `dense_limit` squared entries use the [block transport](#blocks-cadenceblocks);
`layout` passes a precomputed cut. `precision` (torch only) is `"float32"` or `"float64"`
([backends](backends.md)). Parameter arrays are read-only; assign whole validated arrays to
`efficacy`, `bias` or `log_gain`, or use `with_parameters`. Nonfinite values are refused
before any change.

| Member | Behaviour |
| --- | --- |
| `settle(stimulus=None, *, steps=60, state=None, mask=None, trajectory=False, nudge=None, tolerance=None) -> BrainState` | A list of neurons at full amplitude, a `{neuron: level}` map or a dense drive vector. A positive `tolerance` stops once no activation moves more. |
| `settle_batch(drive, ...) -> BrainState` | The same for `(batch, n)` drives. `mask` is `(n,)`, `(1, n)` or `(batch, n)` in `[0, 1]`. |
| `equilibrate(drive, *, budget=512, chunk=32, tolerance=1e-5, state=None, mask=None, nudge=None, damping=0) -> Equilibrium` | Settles until the equation residual is below `tolerance`, checked per chunk; `damping` halvings share the budget. |
| `residual(drive, state, *, nudge=None, mask=None, on_device=True) -> ndarray` | Per-row fixed-point discrepancy; no state change. |
| `stimulus_vector(stimulus)`, `stimulus_levels(levels)`, `readings(state, names, i=0, level=0.5)` | Drive construction ([pattern targets](learning.md#pattern-targets)); population readings. |
| `with_parameters(*, efficacy=None, log_gain=None, bias=None)`, `weights`, `dense()`, `to_dict()` | A new graph; effective drive per synapse; the `W[pre, post]` matrix. |
| `contrast_on_device(plus, minus)` | The learning contrast on the accelerator, or `None`. |
| `connectome`, `neuron_model`, `layout` | The topology, the model and the block cut in use. |

Nonfinite drives and warm states are refused. Convergence of a solve says nothing about
stability, uniqueness or task quality ([the certificate](certificate.md)).

| Type | Fields |
| --- | --- |
| `BrainState(v, activation, adaptation, steps, trajectory=None, activity_change=None, device=None)` | `(n,)` or `(batch, n)` arrays; `row(i)`, `mean(members, i)`, `fraction_active(members, level, i)`, `active(level, i)`, `batched`. |
| `Equilibrium(state, residual, steps, tolerance, residual_checks=0, damping_halvings=0, stagnation_checks=0)` | Per-row `converged` and `qualified` (the residual within `tolerance`); `steps` counts local sweeps. |
| `Nudge(target, mask, beta, softmax_temperature=None, weight=None, groups=None, anchor=None, anchor_gain=None)` | Extra drive `beta * mask * (target - s)`, or against `softmax(s / T)` per group; `weight` scales rows. `anchor` and `anchor_gain` (supplied together, finite, gain nonnegative per neuron) add `anchor_gain * (anchor - s)`, unscaled by `beta`, mask or weight. `drive(s)`. |

`available_backends() -> dict` names what is installed, such as
`{"cpu": "numpy float64", "torch": "mps float32"}`. `ep_structure` and the settling
certificate are in [Certificate](#certificate-cadencecertificate).

## Connectome (`cadence.connectome`)

```text
Connectome.from_synapses(n, *, pre, post, count=None, sign=None, populations=None,
                         label="connectome", min_count=0.0) -> Connectome
```

Parallel synapses merge (counts add), autapses and synapses below `min_count` drop, and the
arrays are sorted by `(post, pre)`. Fields `n`, `pre`, `post`, `count`, `sign`, `populations`
and `label` are read-only. `members(*names)`, `with_populations(**populations)`,
`in_degree()`, `out_degree()`, `synapses`, `summary()`, `digest()` (SHA-256 of the arrays).
Guide: [brains from a connectome](connectomes.md).

## Neuron model (`cadence.neuron`)

```text
NeuronModel(dt=0.2, slope=4.0, threshold=1.5, gain=0.02, stimulus_amplitude=3.0,
            adaptation=None, leak=0.0)
Adaptation(tau_steps=50.0, strength=1.0)
```

The graded neuron: `activation(v)` is a rectified re-based sigmoid that publishes zero at
rest, `slope_at(v)` its derivative, `rest_emission` the raw value subtracted at rest;
`replace(**changes)`, `to_dict()`. `Adaptation` is a slow per-neuron variable
([activation and adaptation](concepts.md#activation-and-adaptation)).

## Blocks (`cadence.blocks`)

```text
layout(connectome, *, max_pairs=256) -> Layout
BlockTransport(layout, flat)
block_contrast(layout, s_plus, s_minus) -> ndarray
```

`layout` cuts the neurons at the boundaries of contiguous populations and pairs the ranges
that carry synapses; with no contiguous populations or more than `max_pairs` blocks it is one
full block. `Layout` has `starts`, `pair_pre`, `pair_post`, `offset`, `edge_index`, `ranges`,
`pairs`, `size`, `bounds(k)`, `sources()` (ranges receiving no synapses),
`parallel_synapses` (distinct synapses sharing a matrix entry), `flat(weights)`,
`blocks(flat)` and `to_dict()`. `BlockTransport.synaptic_input(s)` reuses the product of
every source range that did not move since the previous call. `block_contrast` is the
learning contrast as one Gram product per block.

## Certificate (`cadence.certificate`)

```text
certificate(brain) -> Certificate
Certificate(row_mass, lipschitz, dt, adaptation)
row_mass(brain) -> float
lipschitz_constant(model) -> float
ep_structure(brain, *, fixed_inputs=(), tolerance=1e-12) -> EPStructure
```

`Certificate` (frozen) has `rate` (`1 - dt * (1 - lipschitz * row_mass)`), `certified`
(`rate < 1` and no adaptation), `mass_limit`, `error_bound(movement)`,
`apriori_bound(first_movement, steps)`, `steps_for(change, tolerance)` (raises `ValueError`
when uncertified) and `to_dict()`; construction requires finite values, `row_mass >= 0`,
`lipschitz > 0` and `dt` in `(0, 1]`. `EPStructure` (frozen) reports `fixed_inputs`,
`free_neurons`, `free_asymmetry`, `fixed_incoming_mass`, `adaptation`, `tolerance` and
`compatible`, which covers structure only. Guide: [the settling certificate](certificate.md),
[equilibrium-propagation scope](certificate.md#equilibrium-propagation-scope).

## Neuron-by-neuron reference (`cadence.reference`)

```text
cadence.reference.settle_neuron_by_neuron(connectome, neuron_model, stimulus, *, steps,
    log_gain=None, bias=None, efficacy=None) -> (trajectory, Ledger)
conformance(brain, stimulus, *, steps=60) -> dict
```

The reference settles one neuron at a time, each reading only its own state and synaptic
input, and counts one transmission per declared synapse per step. `conformance` runs a brain
against it on the same stimulus and returns `backend`, `steps`, `max_abs_deviation`, `ledger`
and `final_active` (neurons at or above 0.5).
`Ledger(declared_synapses, steps=0, transmissions=0, undeclared=0)` has `clean` and
`to_dict()` ([conformance](backends.md#residual-checks-masks-and-conformance)).

## Regions and genome (`cadence.regions`, `cadence.genome`)

```text
Region(name, size=0, circuit=None, inputs=None, outputs=None)
cadence.regions.visual_cortex(height, width, *, channels=1, features=8, field=3,
    stride=1, init=1.0, seed=0, name="visual")
cadence.regions.cortex(size, *, lateral=0.0, name="association")
cadence.regions.motor_cortex(actions, *, lateral=0.0, name="motor", slots=1)
cadence.regions.prefrontal_cortex(holds, *, name="prefrontal")
cadence.regions.slot_sizes(actions, slots) -> list[int]
```

A blank region has a `size`; a designed one a `circuit` whose `inputs` and `outputs`
populations receive and send projections. `designed`, `neurons(population=None)`,
`to_dict()`. `visual_cortex` has populations `input` (pixels, row by row) and `output`
(`features` maps of `field`-square windows). A nonzero `lateral` connects every pair of
neurons with that weight (negative inhibits): all of a `cortex`, fixing its size, or each
slot of a `motor_cortex`, whose population is `actions`. `slot_sizes` refuses slots that do
not divide or add up to `actions`. `prefrontal_cortex` has one neuron per neuron it holds. Guide:
[write a cortex](cortex.md).

```text
Projection(pre, post, density=1.0, sign=0.0, scale=1.0, count=1.0, reciprocal=True)
Genome(regions, projections, label="genome")
Genome.composed(inputs, actions, **settings) -> Genome
Genome.observation_width(inputs, code=None, learned_code=None,
                         learned_code_identity=None) -> int
develop(genome, seed=0) -> Connectome
cadence.genome.mutate(genome, rng, *, size_step=0.25, fixed=(), tied=()) -> Genome
genes(space, *, rate=1.0) -> mutation
evolve(fitness, genome, *, generations=10, population=8, keep=2, seed=0, mapper=map,
       report=None, mutate=None, grow=None, **mutation) -> Lineage
```

A projection end is a region name or `region/population`; `sign` is the mean sign and
`reciprocal` adds the reverse synapses with the same weights. `Genome` checks unique names and
valid ends; `region(name)`, `to_dict()`, `Genome.from_dict(d, designed=None)`.
`Genome.composed` is the layout `Brain.compose(inputs, actions, **settings)` develops; the
settings that do not shape it change nothing. `Genome.observation_width` is the key width
the memories of such a brain will use, the size of a custom anatomy's `recall` and `next`
regions. `develop` is
deterministic in the seed. `evolve` scores `fitness(grown, seed)` per life, requires finite
scores and returns `cadence.genome.Lineage` (`generations`, `best`, `best_fitness`). Guide:
[evolve a brain](evolution.md).

## Receipts (`cadence.receipts`)

```text
Receipt.build(kind, body, sources=()) -> Receipt
Receipt.read(path) -> Receipt
Receipt.verify(path, *, sources=None, check=None) -> (ok, message)
canonical_json(value) -> str
cadence.receipts.canonical_sha256(value) -> str
cadence.receipts.source_manifest(files) -> dict
```

`Receipt(kind, body, source, digest)` with `write(path)` and `to_dict()`. Guide:
[receipts](receipts.md).

## Record every settling step

```text
record_settlements(callback, *, label="")
```

A context manager: every settle inside it, including those inside `Learner`, `ActorCritic`
and imagination, calls `callback` with a `SettlementRecord(label, connectome, neuron_model,
drive, mask, weights, bias, nudge, potential, activation, adaptation, steps)`. Histories have
shape `(steps + 1, batch, neurons)`: the initial state, then one row per actual iteration, so
successive differences of `potential` are the signed local repairs.

```python
import numpy as np
import cadence as cd

brain = cd.NeuralGraph(cd.layered(4, 16, 2, seed=0), cd.learning_neuron_model())
records = []
with cd.record_settlements(records.append, label="observe and act"):
    result = brain.settle(stimulus={0: 1.0}, steps=32)
repairs = np.diff(records[0].potential, axis=0)
assert records[0].potential.shape == (33, 1, brain.connectome.n)
```

Recording copies every neuron's state after every step; write bounded chunks from the
callback for long runs. CPU recording uses the NumPy kernel instead of the fused path, so
round-off and wall time can differ. A callback that is not callable raises `TypeError`; it
runs synchronously with recording suspended. Nested contexts restore the previous callback on
exit, callback failures propagate, and the recorded arrays own their storage. Treat the shared
connectome as read-only. For a display see
[the brain in a browser page](brain.md#the-brain-in-a-browser-page).
