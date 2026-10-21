# API reference

One entry per public name: signature, defaults, behaviour in a few lines, refusals, and
the guide that explains it. `Brain.compose` is the entry point; pass optional arguments by
keyword.

| Section | Names |
| --- | --- |
| [Brain](#brain-cadence) | `Brain`, its methods, attributes and checkpoint formats |
| [Arousal](#arousal-cadencearousal) | `ArousalConfig`, `Arousal` |
| [Streams](#streams-cadencestream) and [records](#records-cadencerecords) | `Trace`, `Echo`, `Afterglow`, `Efference`, `History`, `Recall`, `FastSynapses`, `SynapticMemory`, `PatternSeparator`, `stateful`, `Records`, `Mulberry32` |
| [Learning](#learning-cadencelearning), [checkpoints](#checkpoints-cadencecheckpoint), [reward](#the-agent-and-the-valence-cadenceplasticity) | `LearnerConfig`, `Learner`, `GraphHolder`, `LearningPhaseError`, graph factories and diagnostics, `save`, `load`, `ActorCritic`, `ActorCriticConfig`, `Valence`, `Bins` |
| [The graph](#neuralgraph-cadence) | `NeuralGraph`, `BrainState`, `Equilibrium`, `RefinementReport`, `Nudge`, `Connectome`, `NeuronModel`, `Adaptation`, blocks, `Certificate`, the neuron-by-neuron reference |
| [Wiring and instruments](#regions-and-genome-cadenceregions-cadencegenome) | `Region`, `regions`, `Projection`, `Genome`, `develop`, `evolve`, `genes`, `Protocol`, `Row`, `Receipt`, timing |
| [Temporal patch](#temporalpatchnet-cadencetemporal) | `TemporalPatchNet`, `TemporalPlan`, `TemporalMemory`, `PartitionedTemporalPatchNet` |
| [Record patch](#recordpatchnet-cadencerecord_patch) | `RecordPatchNet`, `RecordPatchStack`, `JointRecordPatches`, `Port`, structured ports |
| [Belief and steering](#beliefpatch-cadencebelief) | `BeliefPatch`, `Steered`, weighings, `Life`, governors, instruments |
| [PatchNet](#patchnet-cadencepatch), [compositions](#optional-task-compositions-cadencecircuits), [recording](#record-every-settling-step), [population](#cadencepopulation) | `PatchNet`, `cadence.circuits`, `record_settlements`, `PopulationPatch` |

`cadence.__version__` is the installed release string.

## Brain (`cadence`)

### Construction

```text
Brain.compose(inputs, actions, *, modules=(64,), observers=(), lateral=None,
              sensory_scale=1.0, working_memory_scale=1.2, history_scale=1.2,
              recall_scale=1.2, features=8, field=3, code=None, learned_code=None,
              seed=0, slots=1, **options) -> Brain
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
`learned_code_place` and `learned_code_identity` ([the learned code](code.md)). The `*_scale` arguments are initial
projection magnitudes, fixed at construction. `options` takes every constructor keyword below
and every `learning_<field>`, `actor_<field>` and `arousal_<field>` name; with
`memory_choice=True` it fills in the episodic founders. The region map and the founder table
are in [compose a brain](brain.md#defaults-and-expert-overrides); the memory settings are owned
by [memory](memory.md), [episodic memory](episodic.md) and [recall memory](recall.md).

Refusals: unknown names, the bare `eta` or `eta_bias`, and duplicate aliases raise
`TypeError`; a `recall_` name without `recall=True`, a setting outside its range, and
`arousal_*` names without `arousal` raise `ValueError`.

```text
Brain.build(inputs, actions, *, hidden=64, density=1.0, lateral=None,
            working_memory=False, memory_scale=12.0, episodic=True, features=8,
            field=3, seed=0, slots=1, **options) -> Brain
Brain.genome(inputs, actions, *, hidden=64, density=1.0, lateral=None,
             working_memory=False, memory_scale=12.0, features=8, field=3,
             seed=0, slots=1) -> Genome
Brain.observation_width(inputs, code=None, learned_code=None,
                        learned_code_identity=None) -> int
```

`genome` returns the single-region layout (`sensory` or `visual`, `association`, `motor`, and
with `working_memory` a `prefrontal` region projecting at `memory_scale`) for editing or
evolution; `build` develops and wraps it, passing `options` to the constructor.
`observation_width` is the key width the memories will use, the size of a custom anatomy's
`recall` and `next` regions ([genome, development, brain](brain.md#genome-development-brain)).

```text
Brain(connectome, *, episodic=True, consolidation=0.05, memory_decay=0.9,
      memory_rate=1.0, memory_amplitude=1.0, memory_choice=False,
      memory_context=0.0, memory_replay=0, memory_demonstrations=0.0,
      memory_expansion=0, memory_winners=32, memory_fan_in=None,
      working_memory_decay=0.0, working_memory_amplitude=3.0,
      working_memory_focus=0.0, efference_decay=0.2, efference_amplitude=0.0,
      history_decays=(), history_amplitude=3.0, recall=False, learning=None,
      reward=None, resting_bias=0.0, slots=1, arousal=None, seed=0,
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
| `resting_bias` a finite nonnegative real scalar; booleans and arrays refused | `ValueError` |
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
| `reset(*, keep_memory=False, keep_arousal=False)` | Clears per-stream state, eligibility and reward centering; keeps records and weights. Arousal begins the new stream calm and keeps its age; `keep_arousal=True` keeps the whole arousal state, its level included, as saving `brain.arousal.to_dict()` before the reset and restoring it with `Arousal.from_dict` after it does. `keep_memory=True` keeps the recall slots and ends each stream's episode. Both keywords must be booleans, checked before anything changes. | [reset and save](continuous.md#reset-and-save) |
| `describe() -> dict` | JSON-safe `layout`, effective `genes`, resolved configurations, memory settings, `initialization` and `pending_feedback`. | [defaults](brain.md#defaults-and-expert-overrides) |
| `observation_code(observations) -> (batch, code_width)` | The unit-length key every memory reads and writes; with a learned code, the code the observations settle to without a context. | [the observation code](memory.md#the-observation-code), [the learned code](code.md) |
| `stimulus(observations, *, memory=True) -> drive` | The drive of a batch; with `memory` the trace, copy, history, recall and hippocampal reads are added, and the recall address is fixed for the moment. | [memory](memory.md) |
| `parameters() -> int` | Actor and critic parameters, the consolidated memory matrix, a recall address map and a learned code's tables. | |
| `copy.deepcopy(brain)` | An independent copy: every array, store, state and random generator copied, the immutable connectomes shared. |
| `save(path) -> Path`, `Brain.load(path, *, backend="cpu", device=None, precision=None)` | The whole continuation, written atomically ([formats](#checkpoint-formats)). `Brain.load` refuses a learner-only file; `Learner.load` reads either. | [reset and save](continuous.md#reset-and-save) |

Observations are a nonempty finite batch `(streams, inputs)`, images flattened per row or in
their shape. Keep row identities fixed until `reset()`.

### Attributes

| Attribute | Holds |
| --- | --- |
| `connectome`, `brain`, `learner` | the developed graph, the current `NeuralGraph` (`learner.brain`) and the `Learner` |
| `basal_ganglia` | the `ActorCritic` reading the association region |
| `working_memory`, `efference`, `history`, `recall` | the `Trace`, `Efference`, `History` and `Recall`, each `None` when absent |
| `hippocampus` | the associative `SynapticMemory` (with `memory_choice` the episodic place records), or `None` with `episodic=False` |
| `context_memory` | the episodic context records with `memory_choice` and a positive `memory_context`, else `None` |
| `sparse_code`, `code_width`, `coded_cortex`, `inputs` | the `PatternSeparator` of `code=units` or `None`; the key width; whether the sensory region is the code; the observation width |
| `learned_code`, `code_index`, `prefrontal_index` | the `LearnedCode` of `learned_code=units` or `None`; neuron indices of the `code` and `prefrontal` populations (empty without them) |
| `sensory_index`, `association_index`, `motor_index` | neuron indices of those populations |
| `memory_choice`, `memory_context`, `memory_replay`, `memory_demonstrations`, `resting_bias` | the resolved construction settings |
| `arousal` | the stream's `Arousal`, or `None` |
| `pending_feedback` | read-only: the issued action awaits its outcome (`wait` keeps it) |
| `decision_id` | read-only: the `arousal.age` at which `live` issued the awaited action, or `None`; saved and restored ([feedback ownership in the population solver](equilibrium/LIVE.md#preserve-execution-and-feedback-ownership) is the stricter contract) |
| `last_arousal` | immutable snapshot of the latest `live` moment; `None` after construction, reset or load ([fields](continuous.md#routine-and-repair-live)) |
| `last_settlement` | immutable report of the latest free-answer solve, refused ones included: `operation`, `scope`, `qualified`, `row_qualified`, `residual`, `max_residual`, `steps`, `budget`, `tolerance`, `residual_checks`, `damping_halvings`, `stagnation_checks`, and with a learned code `code_steps`, `code_residual`, `code_halvings` and, for a brain with a place store, `context_free_steps` ([inspect the work](quickstart.md#inspect-the-work-of-answering)) |
| `last_learning` | the latest transition's learning report, teaching under `demonstration_*` keys |
| `rng` | the brain's `numpy.random.Generator` |

### Checkpoint formats

`Brain.save` names the format of the highest component the brain holds; each format may also
carry every component of the lower ones. `load` refuses a component under a lower format name
than its own and a format name without its component; it also reads `cadence-generic/1`.

| Format | Introduces |
| --- | --- |
| `cadence-generic/2` | the composition: learner, critic, optimizers, random state, traces, hippocampus, a pending action |
| `cadence-generic/3` | arousal, with the action `live` issued and its forecast |
| `cadence-generic/4` | an efference copy |
| `cadence-generic/5` | the state a waiting stream sensed (needs arousal) |
| `cadence-generic/6` | the episodic choice (`memory_choice`): context records, temporal context, replay paths |
| `cadence-generic/7` | a heard history |
| `cadence-generic/8` | a recall memory: slots, `held`, `previous`, `history`, the address map, and where set `recall/trace`, `recall/pending`, `recall/ended`, `recall/before`, `recall/command`, `recall/past`; a window of another length is refused |
| `cadence-generic/9` | a `code=units` code (`code/index`, `code/weight`); record and recall key widths are checked against it, and tables that do not read every ON and OFF channel twice with positive weights are refused; a code whose metadata lacks the `cover` field is refused with a message naming that field |
| `cadence-generic/10` | a learned code (`learned_code/dictionary`, `learned_code/weights`, `learned_code/thresholds`, and `moment/code`, `lived/code` for a decision awaiting its outcome); the key widths are checked against its units, and tables that disagree with the saved synapses and biases of the code region are refused |

## Arousal (`cadence.arousal`)

```text
ArousalConfig(threshold=0.2, decay=0.9, tolerance=2.0, floor=0.1, fast=0.05,
              slow=0.005, heat=2.0, youth=100, value_surprise=1.0,
              record_surprise=0.0, need=0.0)
Arousal(config=None)
```

`ArousalConfig` holds the genes of the law; `to_dict()` returns them and
`ArousalConfig.space()` declares their mutation space for [`genes`](evolution.md#any-genome).
Construction refuses negative values, `decay` outside `[0, 1)`, `fast` or `slow` outside
`(0, 1]`, `fast < slow` and a `youth` that is not a nonnegative integer. The law and the
meaning of each gene are in [routine and repair](continuous.md#routine-and-repair-live).

`Arousal` is one stream's state; `Brain.live` runs it, and it can be used alone.

| Member | Behaviour |
| --- | --- |
| `outcome(error, reward, *, own=True, learned=True, record_error=None) -> (surprise, want)` | Applies one outcome. Only `own` outcomes can surprise and enter `usual`; every outcome moves `recent` and `longrun`; `learned` outcomes, and any in youth or that leave the stream aroused, enter `scale`. An outcome whose statistics cannot stay finite raises `ValueError` and changes nothing. |
| `lived(sweeps, learning_sweeps=0)` | Counts one moment in `moments`, `sweeps` (per mode), `learning_sweeps` and `age`. |
| `reset()` | Begins another stream calm; clears the averages and their counts, keeps `age` and the work counts. `Brain.reset(keep_arousal=True)` leaves the state as it is. |
| `to_dict()`, `Arousal.from_dict(values)` | The complete state, marked `cadence-arousal/2`. |
| `level`, `age`, `moments`, `sweeps`, `learning_sweeps`, `outcomes`, `rewards`, `records`, `spreads`, `config` | State and counters: `outcomes` counts own outcomes, `rewards` all outcomes, `records` record-channel outcomes. |
| `aroused`, `mode`, `want`, `heat`, `usual`, `usual_record`, `recent`, `longrun`, `scale` | Read-only: `aroused` while `level >= threshold` or within `youth`; `heat` is `1 + config.heat * want`. |

`from_dict` reads `cadence-arousal/2`, `cadence-arousal/1` and unversioned states. It requires
every gene and continuation field, except that an unversioned state with the gene set before
`need` loads with `need=0`. A `cadence-arousal/1` or unversioned state without `rewards`
takes it from `outcomes`.

## Streams (`cadence.stream`)

The per-stream memories of the composed brain, usable alone on any graph. The composed
brain's memories are explained in [memory](memory.md); the standalone stores in
[standalone memory stores](records.md).

```text
Trace(connectome, decay=0.5, amplitude=1.0, focus=0.0, source="hidden", target="context")
Echo(connectome, decay=0.5, amplitude=1.0, focus=0.0, source="hidden", target="context")
Afterglow(connectome, decay=0.5, amplitude=1.0, focus=1.0, source="hidden", target="afterglow")
Efference(connectome, decay=0.5, amplitude=1.0, focus=0.0, source="motor", target="efference")
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
       capacity=64, value="event", next=True, address=0.01,
       address_temperature=0.1, context=0.0, context_decay=0.5,
       address_step="lms", write="always", null=None, address_reward=0.0,
       episodes="forget", key="event", window=1, window_decay=1.0,
       source="sensory", target="recall", inputs=0, writes=0)
```

The held-input index memory of each stream's recent moments. Each constructor argument is the
brain's `recall_<name>` gene; their meanings, founders and retune rules are the
[recall settings table](recall.md#settings), the mechanism is
[how a moment is addressed](recall.md#how-a-moment-is-addressed). `inputs` is the key width
when it differs from the `source` population (a coded brain passes the code width).

| Member | Behaviour |
| --- | --- |
| `stimulate(drive, observations)` | Addresses the moment, fixes the softmax over the slots (`attended`) and writes the read into `recall` and, with `next`, `next`. |
| `witness(observations, state, *, aroused=True, command=None)` | Writes the moment as a slot after its settle; with `write="aroused"` only for the streams `aroused` marks. `command` feeds `key="predecessor"`. |
| `learn(graph, learned) -> dict` | The address lesson after a teaching step; returns `address_rows`, `address_error`, `address_refused`. A nonfinite step is refused per row. |
| `contrast_rows(graph, plus, minus)`, `propose(step)`, `accept(proposal)` | Reward training of the address map at `address_reward > 0`; `ActorCritic` calls them. `contrast_rows` is `None` at 0. |
| `end(batch, rows=None)` | Ends episodes and keeps the slots; the latest slot learns no successor. |
| `weights(address)`, `ages()`, `charges()`, `readback(graph, region, movement)`, `activity(state)` | The read's slot shares, slot ages, age charges, and the association movement read back through the recall synapses. |
| `reset(batch, rows=None)`, `keep(rows)`, `snapshot()`, `restore(snapshot)`, `reading()`, `hold(reading)`, `imagined()` | Stream bookkeeping; `imagined()` is a copy that reads the live slots and writes none. |
| `Recall.founders()`, `Recall.options()`, `Recall.setting(name, value)`, `Recall.null_logit(null, temperature)` | The always-described genes, the eight options saved only where they differ, validation of one gene, the null slot's logit. |
| `RUNTIME`, `CONSTRUCTION`, `OPTIONS`, `CONSTRUCTION_OPTIONS`, `VALUES`, `STEPS`, `WRITES`, `EPISODES`, `KEYS` | Class tuples: retunable genes, construction genes, the options and those fixed at construction, and the allowed values of `value`, `address_step`, `write`, `episodes` and `key`. |
| `keys`, `values`, `nexts`, `held`, `previous`, `history`, `ended`, `before`, `command`, `past`, `w_x`, `w_c`, `b`, `address_width`, `neurons`, `streams`, `slots`, `attended`, `to_dict()` | State arrays and sizes. |

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
stateful(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0,
         context_init=1.0) -> (Connectome, tie_groups)
cadence.stream.columns(index) -> slice | ndarray
```

`stateful` is `embedded` plus a `context` range for a `Trace`; `columns` returns a slice for a
contiguous index.

## The learned code (`cadence.code`)

```text
LearnedCode(inputs, units, context, model, seed=0, top_down=2.0, rate=0.125, predictor=0.0625,
            threshold=0.1, threshold_rate=0.000625, target=0.1, repair=1.0, silence=10.0,
            place="free", floor=0.0, identity=0.95, scale="observation", held=True,
            reads="free", fields=None, active=0.05)
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
validate the genes; `GENES`, `RUNTIME` and `CONSTRUCTION` name them, and `REMOVED` the
settings the readings retired with what the region does instead. Guide:
[the learned code](code.md).

## Records (`cadence.records`)

```text
Records(inputs, fields, *, cells=8000, active=40, rate=0.2, valued=(),
        valued_rate=1.0, habituation=1e-5, bias=0.3, pathways=(),
        pathway_rate=0.002, tasks=(), fan_in=0, seed=0, averaging=False,
        homeostasis=0.0)
Mulberry32(seed)
```

A delta-rule store over sparse codes of readings; the options table and the read/write
recipe are in [records](records.md#records).

| Member | Behaviour |
| --- | --- |
| `code(readings, *, adapt=False, valued=True) -> (2, batch, cells)` | Plain and valued codes; `adapt=True` first moves the running mean and pathway norms. |
| `witness(readings)` | The adaptation of `code(adapt=True)` without coding, plus the `homeostasis` step. |
| `read(code) -> dict` | Each field's read. |
| `write(code, targets, known=None) -> int`, `write_batch(codes, targets) -> int` | One outcome, or many against the tables as they stood; invalid or nonfinite updates change nothing. |
| `parameters() -> int`, `state()`, `load_state(state)`, `to_dict()` | Record entries; the learned arrays; their atomic replacement; the configuration including `averaging` and `homeostasis`, from which `Records(**to_dict())` rebuilds the same cells. |

Construction raises `ValueError` for nonpositive sizes, `active > cells`, no fields, a
`valued` name outside `fields`, rates outside their ranges, a negative `bias`, an index outside
the reading, fewer than `active` cells per task group, and a `fan_in` without `pathways` or
above their count. `Mulberry32` is the 32-bit generator
the store draws from, so a page implementing the same generator reproduces its cells:
`random()`, `batch(n)`, `normals(n)`, `state`.

## Learning (`cadence.learning`)

```text
LearnerConfig(beta=0.1, eta=0.2, eta_bias=None, centered=True, free_steps=100,
              nudged_steps=50, tolerance=1e-4, nudge="cross_entropy",
              temperature=0.2, normalize=0.0, normalize_floor=1e-3,
              momentum=0.0, decay=0.0, scale_cap=8.0, qualified=False,
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
| `contrast(free, nudged, opposite=None)`, `contrast_rows(...)`, `update(free, nudged, opposite=None)`, `apply(delta_scale, delta_bias)` | The contrast, per row, and an update through masks, rates, tying, decay and the cap. `update` does not check qualification. |
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

## The agent and the valence (`cadence.plasticity`)

```text
ActorCritic(learner, critic, config=None, seed=0, population=None)
ActorCriticConfig(gamma=0.99, lam=0.9, eta=0.5, eta_bias=None, eta_critic=0.05,
                  normalize=0.0, momentum=0.0, dopamine_cap=1.0,
                  dopamine_center=0.0, dopamine_floor=0.0, center_scale=True,
                  critic_normalize=True, critic_signal="auto",
                  eligibility_steps=None)
Valence(level=0.0, floor=0.0, cap=1.0, units=True, per_stream=True, mean=0.0, var=1.0)
Bins(dims, size=9)
```

The reward learner of a stream: a `Learner` whose outputs are the actions, a learned linear
critic on the `critic` neurons, eligibility traces and a dopamine signal. Fields, the report
and the decision contract are in [learning from reward](reward.md#settings).

| Member | Behaviour |
| --- | --- |
| `act(drive, greedy=False, temperature=None, *, bias=None) -> action` | One free phase and a draw per slot (per dimension with `Bins`). `bias` adds a finite `(batch, motor)` vote before each softmax; eligibility scores the readout's own policy. Bins refuse a bias. A greedy act drops the pending action, so `learn` refuses until a sampled act; the decaying traces are kept. |
| `learn(reward, done, next_drive, bootstrap=None, *, observed=None, warm=None) -> dict` | The TD error made into dopamine and written through the eligibility. At least one row must be `observed`. `bootstrap` applies only to `done` rows: a truncated row passes `done=True` with the value of its last observation (`value_of`) as `bootstrap`; other `done` rows bootstrap from zero. |
| `fade(done=None)` | One moment without a new eligibility: every trace decays by `gamma * lam`. |
| `probabilities(state, temperature=None, *, bias=None)`, `value(state)`, `value_of(drive)`, `settle(drive)` | Policy, critic value, a cold settle. |
| `reset()`, `parameters()`, `to_dict()` | Clears cache, eligibility, salience and centering; keeps learned parameters. |
| `state`, `salience`, `memory`, `trace`, `trace_memory`, `valence`, `delta_mean`, `delta_var` | The latest free phase; per-sender salience weights; a memory with learned parameters (`Recall`) and its eligibility; the valence and its statistics. |

`ActorCriticConfig.critic_target` is the resolved `critic_signal`; construction warns when
`eta_bias` exceeds a positive `eta` and when `normalize > 0` with an actor rate above 0.05.
`Valence` turns a prediction error into dopamine: `valence(delta, observed=None)` and
`reset()`. `Bins` is the population code of a continuous action: `centres`, `groups`, `read`,
`size`; it needs at least two levels per dimension.

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
| `Equilibrium(state, residual, steps, tolerance, refinement=None, residual_checks=0, damping_halvings=0, stagnation_checks=0)` | Per-row `converged` and `qualified`; `steps` counts local sweeps. |
| `RefinementReport(local_residual, steps, min_curvature, status)` | Newton refinement of a hybrid `PatchNet` solve; `status` per row is `local`, `refined` or a failure reason. |
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
develop(genome, seed=0) -> Connectome
cadence.genome.mutate(genome, rng, *, size_step=0.25, fixed=(), tied=()) -> Genome
genes(space, *, rate=1.0) -> mutation
evolve(fitness, genome, *, generations=10, population=8, keep=2, seed=0, mapper=map,
       report=None, mutate=None, grow=None, **mutation) -> Lineage
```

A projection end is a region name or `region/population`; `sign` is the mean sign and
`reciprocal` adds the reverse synapses with the same weights. `Genome` checks unique names and
valid ends; `region(name)`, `to_dict()`, `Genome.from_dict(d, designed=None)`. `develop` is
deterministic in the seed. `evolve` scores `fitness(grown, seed)` per life, requires finite
scores and returns `cadence.genome.Lineage` (`generations`, `best`, `best_fitness`). Guide:
[evolve a brain](evolution.md).

## Protocols (`cadence.protocol`)

```text
Row(id, stimulus, readout, predicate, reference="", ablate=(), relative_to="",
    tier="experiment", versus="")
Protocol(stimuli, rows, training=(), levels=Levels(), steps=60)
cadence.protocol.Levels(active=0.5, inactive=0.2, margin=0.15, sparse_min=0.005,
    sparse_max=0.2, densify_margin=0.05, specific_max=0.25, code_level=0.5)
evaluate_predicate(predicate, value, reference, levels=None) -> bool
shuffled(connectome, seed, *, keep=None) -> Connectome
select_gain(make_brain, protocol, grid, *, sparsity_cap=0.05) -> (gain, table)
cadence.protocol.shared_code(activation, other, members, level) -> float
```

`Protocol.score(brain)`, `neurons_for(connectome, stimulus)`, `to_dict()`; `stimuli` maps a
name to the populations it drives. `versus` names the second stimulus a `specific` row holds
apart. `PREDICATES` maps each predicate to its definition. `shuffled` is the control.
`select_gain` maximizes training facts passed among admissible candidates, smallest on a tie;
an empty grid or no admissible candidate raises `ValueError`. Guide: [protocols](protocols.md).

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

## Timing (`cadence.timing`)

```text
cadence.timing.latency(decide, *, repeats=1000, warmup=20) -> dict
cadence.timing.environment() -> dict
```

`latency` times `decide()` and reports the median, 90th and 99th percentiles and maximum in
microseconds, the mean, `jitter` (p99 over p50 minus one) and context switches.
`environment` reports machine, cores, Python, thread limits, pinned cores, load and library
versions ([timing a decision](backends.md#timing-a-decision)).

## TemporalPatchNet (`cadence.temporal`)

```text
TemporalPatchNet(inputs, hidden, outputs, *, seed=0, output_precision=None,
                 initial_radius=0.95, tolerance=1e-9, max_iterations=80,
                 max_backtracks=24, max_damping_trials=16)
```

A residual temporal model with shared `A`, `B`, `C` maps over paths `(batch, time, ports)`;
the constructor options are tabled in [TemporalPatchNet](temporal.md).

| Member | Behaviour |
| --- | --- |
| `observe(inputs, target, *, beta=0.01, rate=0.1, backtrack=False, symmetry_tolerance=0.15, max_halvings=8) -> TemporalObservation` | Repairs the free and detuned paths and carries only the free state. `beta` halves until the detuned pair is centred; an uncentred pair is refused as `contrast_asymmetric` ([centering](temporal.md#checking-that-the-contrast-is-centered)). `backtrack` admits parameters only after a decreasing causal replay. |
| `advance(inputs)`, `imagine(inputs, *, state=None)`, `settle(inputs, *, target=None, beta=0.0, state=None)` | Free inference carried into context; a private path; a diagnostic phase. |
| `plan(inputs, *, goal, controls, bounds=None, state=None, beta=0.01, rate=1.0, max_steps=32, max_backtracks=16, tolerance=1e-6, goal_tolerance=1e-6, symmetry_tolerance=0.15, max_halvings=8, method="steepest") -> TemporalPlan` | Repairs the selected input ports toward `goal`; executes nothing and changes no live state. `method="bfgs"` uses a quasi-Newton direction ([planning](planning.md)). |
| `readback()`, `state`, `parameters()`, `set_parameters(mapping)`, `output_precision`, `set_output_precision(precision)` | Detached values; atomic, validated replacements. |
| `reset()`, `snapshot()`, `TemporalPatchNet.restore(snapshot)`, `save(path, *, compressed=True)`, `TemporalPatchNet.load(path)` | `reset` keeps parameters; saves are `cadence-temporal/1`. |

`TemporalPhase` holds `hidden`, `output`, `residual`, `converged`, `reason`, `iterations`,
`minimum_pivot_eigenvalue`, `message_bytes`, `energy_history`, `accepted_step_sizes`,
`accepted_damping`, `block_chain_attempts` and `energy_evaluations`, with `activity`,
`energy` and `final_state`. `TemporalObservation` holds `updated`, `reason`, the `free`,
`plus` and `minus` phases, `delta`, losses, `accepted_rate`, replays, `beta` and
`contrast_halvings` ([reasons](temporal.md#reasons)). `TemporalReadback` binds the activity to
its parameter revision. `cadence.temporal.contrast_asymmetry(free, plus, minus)` is
`||plus + minus - 2 free|| / ||plus - minus||` over the hidden paths.

## TemporalPlan (`cadence.planning`)

The detached result of `plan`: `inputs`, `prediction`, `initial_prediction`, `boundary`,
`losses`, `step_sizes`, `projected_residual`, `converged`, `predicted_goal_met`, `reason`,
`phase_calls`, `block_chain_attempts`, `energy_evaluations`, `peak_message_bytes`,
`parameter_revision`, `beta`, `contrast_halvings`, `method`, and the properties `cost`,
`initial_cost`, `improved`, `iterations`. `predicted_goal_met` checks modelled cost only
([reading the result](planning.md#reading-the-result)).

## TemporalMemory (`cadence.temporal_memory`)

```text
TemporalMemory(*, relative_tolerance=1e-12)
```

`protect(net, inputs, *, state=None) -> ConstraintReport` admits the current free response
of a query as a local constraint. `observe(net, inputs, target, *, beta=0.01, rate=0.1,
readout_damping=None, symmetry_tolerance=0.15, max_halvings=8)` stages protected learning
atomically. `report(maximum_residual=0.0)`, `snapshot()`, `TemporalMemory.restore(snapshot)`, and the low-level
`project(before, proposed)`. `ConstraintReport(ranks, bytes, maximum_residual)`. Save the net
and its memory together. Guide: [protect a temporal response](temporal-memory.md).

## PartitionedTemporalPatchNet (`cadence.experimental`)

```text
cadence.experimental.PartitionedTemporalPatchNet(inputs, hidden, outputs, *,
    masks=None, **options)
cadence.experimental.two_group_masks(context_hidden, motor_hidden, inputs, outputs, *,
    context_inputs, motor_inputs, cross_coupling=True) -> dict
```

The temporal model with boolean `A`/`B`/`C` masks (omitted masks allow every entry);
`masks` returns copies and `trainable_parameter_count` counts permitted entries. Restore
through the subclass. `TemporalMemory.observe` refuses it before any change. Guide:
[experimental fixed connectivity](partitioned.md). The population solver,
`cadence.experimental.equilibrium`, has its own [reference](equilibrium/REFERENCE.md).

## RecordPatchNet (`cadence.record_patch`)

```text
RecordPatchNet(inputs, hidden, outputs, *, seed=0, output_precision=None,
               cells=4096, active=32, record_rate=0.5, habituation=1e-5,
               record_bias=0.3, slowest=128.0, record_averaging=False,
               record_homeostasis=0.0, groups=None, record_writes="sequential",
               record_width=None, port=None)
```

A gated linear context whose `Records` store patches the slow readout. `record_averaging` and
`record_homeostasis` pass to `Records`; `groups` makes the ports categorical;
`record_writes="batch"` writes a call's moments through `write_batch`; `record_width` stores a
fixed random sign code of the residual; `port` is a `StructuredPort`. Guide:
[RecordPatchNet](record-patch.md).

| Member | Behaviour |
| --- | --- |
| `observe(inputs, target, *, rate=1.0, backtrack=False, write=True) -> RecordObservation` | Predicts with the records at the start of the call, steps the slow readout, writes the residual. At `rate=0` it returns `updated=False`, `reason="no_step"`. |
| `imagine(inputs, *, state=None)`, `advance(inputs)`, `reset()` | Private; carries context; clears context and keeps parameters and records. |
| `dream(inputs)`, `sleep(cues, *, passes=1, rate=1.0, backtrack=False, dawn_passes=2) -> dict` | The store's completion of a cue; the night that moves dreams into the slow weights ([records by day, weights by night](record-patch.md#acquisition-in-two-phases-records-by-day-weights-by-night)). |
| `detune(inputs, target, *, beta=1e-3, state=None, tolerance=1e-14, max_iterations=10000) -> RecordContrast` | Both detuned equilibria by conjugate gradients; changes nothing. |
| `parameters()`, `set_parameters(mapping)`, `records`, `readback()`, `output_precision`, `set_output_precision(precision)` | `G`, `g`, `B`, `b`, `C`, `c`; the store; a `RecordReadback`. |
| `snapshot()`, `RecordPatchNet.restore(snapshot)`, `save(path, *, compressed=True)`, `RecordPatchNet.load(path)` | `cadence-record-patch/2`, or `/3` with groups, batch writes, a record width or a port. |

`RecordPath(hidden, output, gate, read, loss, slow_loss)` with `slow_output` and
`final_state`; `RecordObservation(updated, reason, prediction, delta, initial_loss,
final_loss, accepted_rate, replay_losses, replay_calls, writes)`; `RecordContrast(contrast,
plus_hidden, minus_hidden, plus_energy, minus_energy, iterations, residuals, converged)`;
`RecordReadback(state, updates, writes, parameter_revision, state_parameter_revision,
record_entries)`.

```text
RecordPatchStack(inputs, hidden, outputs, *, lower=None, seed=0, slowest=128.0,
                 groups=None, lower_port=None, **upper)
JointRecordPatches(cortices, own_inputs, ports, *, rounds=1, damping=1.0,
                   cross_adjoint=True)
Port(source, target, start, width)
cadence.record_ports.build(hidden, own_inputs, outputs, ports, *, seed, rounds=1,
    damping=1.0, cross_adjoint=True, groups=None, **records) -> JointRecordPatches
```

`RecordPatchStack` puts a context patch of width `lower` below a `RecordPatchNet`;
`observe`, `imagine` (`state` is the pair of contexts), `advance`, `reset`, `parameters`,
`set_parameters`, `snapshot` and `restore`; `observe` returns `StackObservation(updated,
reason, prediction, delta, initial_loss, final_loss, accepted_rate, writes)`.
`JointRecordPatches` couples patches through ports over `rounds` Jacobi rounds: `observe(xs,
targets, *, rate=1.0, backtrack=False, write=True) -> JointObservation(updated, reason,
settled, delta, initial_loss, final_loss, accepted_rate, replay_calls, writes)`, where
`settled` carries the paths with `seam` and `settle` per moment and round, `imagine(xs, *,
states=None)`, `advance(xs)`, `reset`, `parameters`, `parameter_count`, `record_entries`,
`state`, `snapshot`, `restore`, `clone`, `save`, `load`, and `cut = True` zeroes every port.
Lists must match the patch count; nonfinite paths are refused; a failed joint write restores
earlier writes. `build` grows the patches at the widths the ports need
([joined patches](record-patch.md#several-patches-joined-by-ports)).

## Ports (`cadence.ports`)

```text
MapBlock(start, channels_in, height, width, channels_out, kernel, stride=1)
DenseBlock(start, inputs, outputs)
StructuredPort(inputs, blocks, broadcast=None, mask=None)
```

A `MapBlock` is a tied local kernel over a grid of the inputs (`inputs`, `outputs`,
`out_height`, `out_width`, `weights`), a `DenseBlock` a matrix over a slice. `StructuredPort` has `apply(u, weights)`, `transpose(v, weights)`, `gradient(v, u)`,
`initial(rng, scale)`, `weight_shape(block)`, `dense_matrix(weights)`, `to_dict()` and
`from_dict(d)`; `broadcast=(start, count)` tiles a slice into every map block, and `mask`
names the inputs the port hears. `RecordPatchNet(port=)`, `RecordPatchStack(lower_port=)` and
`BeliefPatch` read through it ([maps](record-patch.md#maps-a-structured-input-port)).

## BeliefPatch (`cadence.belief`)

```text
BeliefPatch(observation, actions, belief, outputs, *, iterations=2, damping=0.5,
            cells=4096, active=32, record_rate=0.5, record_width=64,
            habituation=1e-5, record_bias=0.3, output_precision=None, seed=0)
```

A belief carried by a learned transition and repaired by evidence through a
`StructuredPort`, with a record store read inside the repair. Guide:
[the belief patch](belief.md).

| Member | Behaviour |
| --- | --- |
| `assimilate(observations, actions, observed=None, *, state=None, gains=None, probe=False, keep_live=False) -> BeliefPath` | Advances the belief through observed moments; learns and writes nothing. |
| `imagine(actions, *, state=None, gains=None) -> BeliefPath` | The transition alone, private. |
| `observe(observations, actions, target=None, *, observed=None, rate=1.0, write=False, state=None, gains=None, loss_weight=None, output_gradient=None, backtrack=None, probe=False, keep_live=False) -> BeliefObservation` | One backward scan and an admitted step ([an admitted step](belief.md#an-admitted-step)); `output_gradient` replaces `target` and writes nothing. |
| `readback(observations, actions, *, state=None, probe=True) -> BeliefReadback`, `encode(observations)` | One moment's expectation, probes and surprise; the encoded evidence. Both change nothing. |
| `set_implied_reading(implied, units=None)` | Declares the reading each block should give, enabling `surprise`; not part of a snapshot. |
| `step_size`, `reset_step()`, `macs_per_moment(*, probes=False, surprise=True)`, `readback_macs(*, probe=True)`, `cost`, `reset_cost()` | Admission memory and cost ([the cost of a moment](belief.md#the-cost-of-a-moment)). |
| `reset()`, `state`, `parameters()`, `set_parameters()`, `set_output_precision()`, `records`, `block_count`, `snapshot()`, `restore()`, `save()`, `load()` | `reset` keeps the step size and counters; saves are `cadence-belief/1`. |

`BeliefPath(belief, expectation, residual, output, read, loss, step, evidence, code, gains,
residual_alone, surprise)` with `slow_output` and `final_state`.
`BeliefObservation(updated, reason, path, delta, initial_loss, writes, final_loss,
accepted_rate, replay_calls, gain_gradient)`. `BeliefReadback(expectation, residual_alone,
surprise, evidence)`. With torch, `cadence.belief_torch.TorchBelief(port, actions, belief,
outputs, *, iterations=2, damping=0.5, record_width=64)` is the slow half as a module:
`forward(observations, actions, state=None, reads=None, gains=None, observed=None)`,
`export()`, `load(params)`.

## Steered (`cadence.steering`)

```text
Steered(cortex, steering=None, weighing=None, *, reads_output=False, evidence=(),
        extra=None, extra_channels=0, rate_scale=1.0, lagged=False, relative=False,
        reads_age=False, age_core=0.5)
Softmax(blocks, *, span=1.0)
Gaze(blocks, *, sigma, cut=2.5, lamp=0.2, span=0.5, price=0.0, start=0.0)
Rule(fn, *, macs=0)
```

A `BeliefPatch` cortex whose block gains a steering `BeliefPatch` sets through a weighing, or
a hand-written `Rule`, or nothing. The readback channels and switches are tabled in
[a brain that reads itself](steering.md#the-steered-cortex).

| Member | Behaviour |
| --- | --- |
| `run(observations, actions, target=None, *, rate=0.0, state=None, keep_live=False, learn_cortex=True, backtrack=True) -> SteeredPath` | The moments of a chunk; with a target and a rate one joint admitted step. `learn_cortex=False` lets the cortex sleep. |
| `boundary() -> Boundary`, `reset()` | A detached copy of where the life is, for `run(state=...)`; `reset` forgets it. |
| `ablation`, `deaf` | `None`, `"cut"` or a callable over the gains; a mask of readback channels zeroed at run time. |
| `readback_names`, `layout`, `channels`, `probes_on`, `step_size`, `reset_step()`, `moments_per_decision()`, `macs_per_moment()`, `cost`, `reset_cost()`, `parameter_count()`, `parameters()`, `set_parameters()` | Layout and cost. |
| `snapshot()`, `Steered.restore(snapshot, *, rule=None, extra=None, ablation=None)`, `save()`, `Steered.load(path, **kw)` | Saves the live boundary and run state as `cadence-steered/1`; callables are supplied again, and the cortex's implied reading is declared again. |

`SteeredPath(output, gains, readback, residual, steering_output, loss, price, updated,
steering_updated, step, halvings, replays, reason, last, last_steering)` with `objective`.
`Boundary(cortex, steering, output, residual, weighing, moments, heard)`. A weighing
(`cadence.steering.Weighing`, with `Softmax` and `Gaze`; not exported at the top level) has
`begin(n)`, `gains(y, state) -> (gains, state)`, `pull(ys, gains, dgains, states)`,
`price(ys, states)` and `to_dict()`; `Gaze` adds `profile(centre)` and `turn(y)`
([the weighing](steering.md#the-weighing)).

## Life (`cadence.life`)

```text
Life(patch, governor, *, habit, propose, advance, cost, target, baseline=1.0,
     residual0=1.0, config=None, refit=None, keep_records=True)
LifeConfig(window=96, passes=6, learn_rate=0.1, rollback=True, validity=0.9,
           min_window=32, min_cooldown=32, cooldown=0, keep=512, horizon=6,
           hold=True, baseline_rate=0.02, floor=1.0, habituate=2.0,
           slow_rate=0.02, recent=96, write=False, learn_cortex=True, extra={})
```

The decision loop over a `BeliefPatch` or `Steered`, with a governor choosing habit,
imagination or learning. The application supplies `habit(reading) -> action`,
`propose(reading) -> (candidates, actions)`, `advance(reading, output) -> reading`,
`cost(readings, actions) -> (candidates,)` and `target(reading, next) -> (outputs,)`.
`baseline` and `residual0` (finite, positive) are the routine surprise and residual that
scale the readback; `refit(life)` runs after each kept learning window and its result is
stored in that `learns` entry; `keep_records=False` keeps no `Decision` copies. The
`LifeConfig` fields, `learn_cortex` and `extra` included, are tabled in
[a life with a governor](steering.md#a-life-with-a-governor); construction refuses
non-integer counts, negative rates and `baseline_rate` or `slow_rate` outside `[0, 1]`.

| Member | Behaviour |
| --- | --- |
| `decide(reading) -> action`, `outcome(next_reading) -> Decision`, `step(reading, world) -> (next_reading, Decision)` | One decision, closed by what the world returned. |
| `readback()`, `signals() -> Signals` | The seven-channel readback (`cadence.life.READBACK`); what a governor reads. |
| `records`, `learns`, `totals`, `compute()`, `reset()` | Every decision and learn call; the cost per decision. `reset` drops unfinished decisions and replay windows. A failed learning step restores parameters, records and counts. |

`Decision(t, mode, action, expected, residual, readback, governor_steps, imagined, learned,
surprise, baseline, target)`. `Signals(readback, surprise, baseline, residual, slow, mode,
recent, can_learn)`.

A governor implements `Governor`: `settle(signals) -> (mode, steps)` returns `"habit"`,
`"imagine"` or `"learn"` and the steps it took, `after(signals)` sees the outcome, `reset()`,
and `synapses` prices the steps. Provided: `PatchGovernor(genome=None, *, cortex=4,
model=GOVERNOR_MODEL, budget=200, chunk=10, tolerance=1e-3)` with `hand_set(cortex)`,
`space(cortex)`, `converged` and `residual` (a capped solve selects habit);
`ThresholdGovernor(genome=None)` with `HAND_SET`, `SPACE` and `persist`;
`AlwaysAwake(genome=None)`; `NeverWakes()`.

## Instruments (`cadence.instruments`)

```text
orienting(gain, events, *, pre=4, post=12, quiet=None, bins=10) -> dict
dishabituation(rows, *, consequential, kind, window=60, count=2) -> dict
```

`orienting` returns `rows` (per event baseline, peak, capture, latency, return and
`return_censored`), `kinds` (count, first and last `bins` captures, curve, mean capture,
latency shares, mean return over uncensored events), `pre` and `post`. `dishabituation`
returns `before`, `after` and `events`. Guide:
[the instruments of the orienting response](steering.md#the-instruments-of-the-orienting-response).

## PatchNet (`cadence.patch`)

```text
PatchNet(learner, *, input_index=None, steps=512, chunk=32, tolerance=1e-5,
         source_capacity=256, context_strength=0.0, context_mask=None,
         solver="local", refinement_steps=64)
PatchNet.create(inputs, hidden, outputs, *, seed=0, density=0.3, init=1.0,
                config=None, backend="cpu", device=None, **runtime_options)
PatchNet.recursive(inputs, layers, outputs, *, seed=0, coupling=1.0, config=None,
                   backend="cpu", device=None, **runtime_options)
```

A reciprocal graph and `Learner` for continuing observations on declared ports.
`runtime_options` are the constructor keywords after `learner`: phase budget `steps`, `chunk`
and `tolerance`, the evidence-ID window `source_capacity`, temporal overlap
`context_strength` and `context_mask`, and `solver`. `solver="hybrid"` follows the local
budget with up to `refinement_steps` Newton steps per unresolved row and refuses unsupported
configurations ([the hybrid solver](patchnet.md#the-hybrid-solver)). `recursive` adds observer
layers connected both ways to all earlier neurons; `coupling` sets the initial maximum
absolute row sum ([recursive observers](recursive-settlement.md)).

| Member | Behaviour |
| --- | --- |
| `stimulus(inputs, *, amplitude=1.0)`, `settle(drive) -> Equilibrium`, `read(phase_or_state=None)` | Drive construction; live free settle; output ports. |
| `observe(drive, target, *, observed=None, weight=None, source_id=None) -> PatchObservation` | Free and nudged phases and an update (`reason="updated"`). No commit for `duplicate` (a `source_id` seen with the same evidence; different evidence raises), `no_observations`, `free_unconverged`, `free_unqualified`, `nudge_unconverged`, `nudge_unqualified`. |
| `imagine(drives, *, state=None) -> tuple[Equilibrium, ...]` | Consecutive private free equilibria. |
| `reset()`, `state`, `snapshot()`, `brain`, `output_index` | `reset` keeps parameters, optimizer history and the evidence window. |
| `save(path, *, compressed=True)`, `PatchNet.load(path, *, backend="cpu", device=None, precision=None)` | `cadence-patch/3` with the solver and refinement budget; `/2` loads with the local solver, `/1` also without temporal overlap. |

`PatchObservation(free, plus, minus, updated, reason, metrics)`. Guide: [PatchNet](patchnet.md).

## Optional task compositions (`cadence.circuits`)

```text
from cadence.circuits import (assemble, imagine, Deliberator, Deliberation, Future,
                              ActivityMonitor, Readback, reflex_arc)
```

Small compositions of ordinary neuron dynamics with host-side orchestration; budgets and
scheduling are in [defaults and the thinking clock](continuous.md#defaults-and-the-thinking-clock).

| Name | Behaviour |
| --- | --- |
| `assemble(regions, synapses=()) -> Connectome` | Merges an ordered `{name: Connectome}` mapping; each synapse is `(source_region, neuron, target_region, neuron, weight)`. Regions and their populations stay addressable. Unknown regions, out-of-range neurons, autapses and nonfinite weights are refused. |
| `imagine(live, actions, transition, evaluate, terminal, *, depth=2, max_nodes=10000, adversarial=False, prune=False, clone=deepcopy) -> Deliberation` | Compares copied futures from the root actor's view; `adversarial` alternates min and max, `prune` uses alpha-beta. Exhaustion raises `ValueError` before the next transition and returns no partial ranking; invalid budgets and nonfinite scores raise. |
| `Deliberator(actions, transition, evaluate, terminal, *, depth=6, max_nodes=10000, adversarial=False, prune=False, clone=deepcopy)` | Resumable iterative deepening: `start(live)`, `tick(nodes=128)` (the last completed `Deliberation` or `None`), `pending`, `pause()`, `resume()`, `cancel()`. `nodes` counts the positions visited since `start`; `budget_exhausted` is true once `max_nodes` stopped the search; `result` holds the last completed `Deliberation`, which an incomplete depth never replaces. A callback error cancels the search and re-raises. Stores no weights and writes no feedback. |
| `Deliberation(futures, nodes, depth)`, `Future(action, score, sequence, state)` | Futures sorted by score. |
| `ActivityMonitor().read(activity, scores, *, pressure=0) -> Readback`, `reset()` | A heuristic reading of finite 1D vectors: `Readback(activity_change, ambiguity, pressure, uncertainty, request_more, state)`. |
| `reflex_arc(axes=2) -> Connectome` | Sensory error ports and opposing motor pairs; the application supplies the body. |

`clone` must isolate all mutable branch state, the other callbacks must not mutate outside objects, and the callbacks stay fixed for one search. Node limits do not interrupt a running callback.

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

## `cadence.population`

Needs torch ([many brains on one device](population.md)).

```text
PopulationPatch(inputs, hidden, outputs, *, instances, streams, seed=0, cells=4096,
                active=32, record_rate=0.5, habituation=1e-5, record_bias=0.3,
                slowest=2.0, groups=None, device=None, dtype=torch.float32)
PopulationPatch.from_patch(net, *, instances, streams, device=None, dtype=torch.float32)
device_of(name=None) -> torch.device
```

Record patches in lockstep on one device. `groups` is `None` or `(outputs,)`.

| Member | Behaviour |
| --- | --- |
| `imagine(inputs, *, stream_of=None) -> dict` | `output`, `slow`, `read`, `hidden`, `code`, `familiarity` for `(instances, N, inputs)`; private. `stream_of` `(N,)` or `(instances, N)` assigns each moment to a stream's store. |
| `observe(inputs, target, *, rate=1.0, write=True, mask=None, stream_of=None, weight=None) -> dict` | One moment per stream: slow step and record writes. `mask` `(instances, streams)` names the streams with a moment; `stream_of` teaches several queued moments of one stream at once; `weight` `(outputs,)` or `(instances, outputs)` reweights a linear readout's errors. Every change is staged; a nonfinite result raises `ValueError` and changes nothing. |
| `code(readings)`, `read(code)`, `familiarity(indices, values, stream_of=None)`, `stream_loss(slow, target, weight=None)`, `loss(slow, target)` | The store's code and read; how much of a code falls on written cells; losses. |
| `clear_records()`, `inherit(parents, *, sigma=0.0, generator=None)`, `parameters()`, `state()`, `memory_bytes()` | Bookkeeping and selection. |

`device_of` picks cuda, then mps, then cpu.
