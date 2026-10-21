# Memory

A memory is state that outlives one settle and changes a later answer. Everything else a
brain computes is settled afresh at each moment from the observation and its memories.

Each memory reaches the answer through the settle. The working trace, efference copy, heard
history and recall memory drive regions of their own that the association region reads
through plastic synapses; the associative store drives the motor neurons; the episodic memory
adds a vote to the motor readout. The answer is one settle, which has to
[qualify](continuous.md#qualification-and-refusal).

Memories have three lifetimes. Per stream, cleared by `reset()` and by an ended episode: the
working trace, efference copy, heard history, recall slots and actor eligibility. Across
resets: the associative and episodic records and the recall address map. Permanent until
learning changes them: the learned weights. The per-stream traces advance at every moment;
under [`live`](continuous.md#routine-and-repair-live) a calm brain reads its records and
weights and changes neither, and an aroused one learns and writes. `reset()` also clears
arousal's income and error references, which decide when a `live` brain wakes, and
`reset(keep_arousal=True)` keeps the arousal state as it is
([reset and save](continuous.md#reset-and-save)).

## Memory in the composed brain

<a id="three-kinds-of-memory-in-brain"></a>

| Memory | On by default | Switch | Holds | Keyed by | Written | Survives `reset()` | Home |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Working trace | yes | `working_memory_amplitude=0.0` silences it | the settled association activity of the moment before | none | after each act and each moment `wait` senses | no | [below](#working-trace) |
| Efference copy | no | `efference_amplitude=1.0` | a fading one-hot of the commands issued | none | after each issued act | no | [below](#efference-copy) |
| Heard history | no | `history_decays=(0.2, 0.8)` | the observations as heard, one fading block per decay | none | after each act and each moment `wait` senses | no | [below](#heard-history) |
| Associative store | yes | `episodic=False` removes it | one value per action per situation, in reward units | observation code | each outcome of the chosen action | yes | [below](#associative-store-the-hippocampus) |
| Episodic memory | no | `memory_choice=True` | place and context records that vote in the choice | observation code, and code with the temporal context | known outcomes, demonstrations, replay of the path to a reward | records yes, context no | [episodic.md](episodic.md) |
| Recall memory | no | `recall=True` | the stream's recent moments as slots, and a learned address map | observation code | a slot per moment; the map at lessons | map yes, slots no | [recall.md](recall.md) |
| Learned weights | yes | learning rates of zero freeze them | synapse scales and biases, actor and critic | none | lessons, and outcomes (aroused ones under `live`) | yes | [learning.md](learning.md) |
| Actor eligibility | yes | `actor_lam=0.0` credits the last action only | each synapse's credit for recent sampled acts | none | each sampled act | no | [reward.md](reward.md#eligibility-traces-and-dopamine) |

`predict`, `accuracy` and `fit` answer independent samples and read none of these except the
learned weights, and `fit` starts with a `reset()`. Test a memory through `act`, `step` or
`live` on a continuing stream. `save` keeps every memory in the table, per-stream traces
included, and `Brain.load` continues the stream where it stopped.

Stores used directly, outside a composed brain (`FastSynapses`, `PatternSeparator`,
`SynapticMemory`, the `Trace` family), are listed
[at the end](#stores-outside-the-composed-brain).

## Which memory for which task

The figures come from this repository's benchmarks and tests; each row names its operating
point where it differs from the founders.

| Task | Use | Evidence | Where it fails |
| --- | --- | --- | --- |
| Carry a cue across a short delay to its reward | working trace (on) | delayed response: seeds 8 to 13 solve at 0.96 to 1.00 ([test](../tests/test_context_defaults.py)) | a history that carries nothing useful, next row |
| Answer from the present when the history does not matter | the founder; try `working_memory_amplitude=0.0` as the control | continuing contextual bandit with `actor_gamma=0.0, actor_lam=0.0`, seeds 0 to 5: 0.23 to 1.00, mean 0.60, chance 0.25 ([test](../tests/test_context_defaults.py)) | the actor has to learn the trace away |
| Hold one cue and replace it with a later one over a delay | no measured memory; the working trace fails it | [recall benchmark](../benchmarks/recall/README.md): a tuned working-trace recipe without the associative store fails on fresh founders; recall memory and heard history are not measured there | a loop strong enough to hold a cue resists the next write |
| Alternate or sequence its own actions under identical input | efference copy at `efference_amplitude=3.0`, `efference_decay=0.0` | [rhythm](../benchmarks/rhythm/README.md), `modules=(32,)`, `episodic=False`, `working_memory_decay=0.1, working_memory_scale=12.0`: alternation 0.97 and 0.93 against 0.50 and 0.40 with the trace alone; uniform random alternates at 0.50 | delayed-credit foraging ([key-door](../benchmarks/keydoor/README.md)): a copy at amplitude 0.3 feeds less at every delay, and at 3.0 the creature does not feed |
| Learn which action pays in a situation, and reverse quickly | associative store (on) | [reversal](../benchmarks/reversal/README.md), `modules=(32,)`, `working_memory_amplitude=0.3, working_memory_scale=12.0, working_memory_decay=0.2, consolidation=0.25`, actor rate 0.1 (bias 0.01): memory alone 1.00 from 300 exposures against the graph alone 0.60 to 0.80 under rule A | at the composed founders the chamber stays at chance (`defaults` 0.48 to 0.53); noisy pay overwrites a one-shot record, next row |
| Noisy or signed pay | `memory_rate=0.2` | [competing](../benchmarks/competing/README.md): acquires as at 1.0 and halves the loss | half the loss is left |
| Situations whose keys overlap strongly | a standalone `FastSynapses` behind a centered, habituated `PatternSeparator` ([records](records.md)); not available on the founder store | 64 keys of cosine 0.9, expansion 1024, 8 winners: revision accuracy at least 0.95, at least 0.5 above the plain store ([test](../tests/test_separation.py)) | the founder store has no separator; the episodic memory's fixed separator is uncentered and not measured by this test |
| Credit an action paid a few moments later | actor eligibility (on) | [key-door](../benchmarks/keydoor/README.md), `modules=(32,)` at the same operating point (scale 12.0, decay 0.2): fed 0.92 at delays 2 and 5; `actor_lam=0` falls to 0.53 at delay 5 | delay 10: 0.49 against a tabular learner's 0.83 to 0.90 |
| Learn a multi-step chain whose stage is observed | the founder (weights and store together) | [chain](../benchmarks/chain/README.md), pay at each correct step: 204 greedy completions per thousand moments against tabular 250; pay at the end only: 42, uniform random's rate | slow one-stream reward learning, see reversal |
| A state the observation does not show | no memory; supply the state as an observation | [chain](../benchmarks/chain/README.md) hidden: no arm beats the memoryless 62 | see [what breaks a chain](sequential-tasks.md#what-breaks-a-chain) |
| A moment heard once, the past at several timescales, what happened here before | recall memory, heard history, episodic memory | contract tests only: [recall](../tests/test_recall_memory.py), [history](../tests/test_heard_history.py), [episodic](../tests/test_episodic_memory.py) | no benchmark measurement in this repository |

## Working trace

After each moment the trace copies the settled activity of the association region (the last
entry of `modules`) into the `prefrontal` region. The next settle holds those neurons at
`working_memory_amplitude` times the copy, and the association region reads them through
plastic synapses that start quiet, so a newborn answers mostly from the present. The copy is
fixed; learning strengthens the synapses where the carried interpretation predicts the
outcome.

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=3, actions=2, seed=0)        # the trace is on by default
brain.act(np.array([[1.0, 0.0, 0.0]]), greedy=True)       # a cue
carried = brain.working_memory.trace.copy()               # the settled association activity
drive = brain.stimulus(np.array([[0.0, 0.0, 1.0]]))       # the next moment's drive
prefrontal = brain.connectome.populations["prefrontal"]
assert np.allclose(drive[0, prefrontal], 3.0 * carried[0])  # held at the amplitude
brain.reset()                                             # a new stream starts without it
assert not brain.working_memory.trace.any()
```

| Setting | Founder | Changed by | Raise it when | Lower it when |
| --- | --- | --- | --- | --- |
| `working_memory_amplitude` | 3.0 | `retune` | the copy should weigh more in the settle and in learning | the carried context should fade (0 silences it) |
| `working_memory_decay` | 0.0, in [0, 1) | `retune` | the context changes slowly and should be averaged over moments | 0 carries the last interpretation only |
| `working_memory_focus` | 0.0 | `retune` | only the neurons that just changed should be carried | 0 carries every neuron |
| `working_memory_scale` | 1.2 | construction (`compose` argument) | the past must matter from the first moment, before learning | the brain repeats itself and ignores what it sees |
| `modules` | `(64,)` | construction | more items or distractors: `modules=(128,)` | a narrow last module such as `(48, 24)` carries little |

`Brain.build` has no trace unless `working_memory=True` (projection scale `memory_scale`
12.0); `Brain(connectome, ...)` has one when the connectome has a `prefrontal` region.
`done=True` clears a stream's trace, so mark episodes only where the world restarts, and
teach with `learning_surprise=True` so lessons touch only the moments answered wrongly
([learning](learning.md#lessons-only-where-the-answer-differs)). The synapses that read the
trace learn only where the outcome or the lesson depends on something the brain saw earlier,
so the task has to make the past matter; distinct cues, such as one marking new information
and one marking noise, help. The trace costs 64 neurons
and 4,096 synapses at the founder. Two pasts that settled alike are one context to it, and a
distractor that looks like evidence can overwrite the context of a small brain.

## Efference copy

A positive `efference_amplitude` at construction adds an `efference` region of one neuron per
motor neuron, holding `c <- d c + (1 - d) onehot(action)` for the command each stream issued
(`d = efference_decay`), read through a plastic projection of fixed initial scale 12.0. The
working trace carries the executed action only through the margin the motor competition
left, and nothing of a sampled choice; the copy holds the command itself.

```python
brain = Brain.compose(inputs=3, actions=2, seed=0, efference_amplitude=1.0)  # opt-in
action = brain.act(np.array([[0.0, 1.0, 0.0]]))           # a sampled act issues a command
copy = brain.efference.trace                              # fading one-hot, decay 0.2
assert np.isclose(copy[0, action[0]], 0.8)
```

`efference_amplitude` (0.0) adds the region only at construction; `retune` changes it
afterwards and refuses it on a brain built without the copy. `efference_decay` (0.2, in
[0, 1)) can be retuned. The copy costs one neuron per action and as many synapses into each
association neuron.

## Heard history

<a id="hear-the-past-at-several-timescales"></a>

The heard history keeps the observations themselves, where the trace keeps what the brain
made of them. `history_decays` adds a `history` region of one block of sensory width per
decay. After each act and each moment `wait` senses, the block of decay `d` moves
`h <- d h + (1 - d) x`, `x` the values the sensory neurons were driven with, so at a decision
it holds the past and the sensory drive the present. A plastic projection reads it. Add it
when the same observation needs different answers after different earlier observations.

```python
brain = Brain.compose(inputs=3, actions=2, seed=0, arousal=True, history_decays=(0.2, 0.8))
brain.live(np.array([[1.0, 0.0, 0.0]]))                   # a cue
brain.live(np.array([[0.0, 0.0, 1.0]]), reward=[0.0])     # the cue is gone
trace = brain.history.trace                               # two blocks of three heard values
assert trace.shape == (1, 6) and trace[0, 3] > 0          # the slow block holds the cue
```

`history_decays` (`()`, each in [0, 1)) and `history_scale` (1.2, the projection's initial
magnitude, a `compose` argument) are construction settings; `history_amplitude` (3.0) can be
retuned. On a coded vector brain the blocks hold the code. Three decays at 256 inputs and
four actions add 768 neurons and 49,152 synapses to a founder of 21,004 synapses.

## Associative store (the hippocampus)

`brain.hippocampus` is a `SynapticMemory` from the observation code to the motor neurons, one
value per action in reward units: a persistent matrix `C` shared by every stream plus a fast
residual `F` per stream. Before each settle the read `C + F` drives the motor neurons at
`memory_amplitude`. For a unit-length key `k` and the observed value `v` of the chosen action:

```text
F     *= memory_decay
alpha  = min(1, consolidation * (1 + salience))
C     += alpha * outer(k, v - k @ C)
F     += memory_rate * outer(k, v - k @ (C + F))
```

Salience is `abs(reward)` unless `salience=` supplies a nonnegative vector; the signed value
decides what is stored. A value mask leaves the actions not taken untouched, and a batch
averages the change to `C` over its writing rows. The store writes every outcome of a
sampled act given to `step` and `learn`, zeros included (an all-zero key writes nothing);
under `live`, the outcomes of sampled acts and the routine outcomes that woke a calm brain.
`fit` and lessons write nothing to this store. Reads change nothing.

```python
brain = Brain.compose(inputs=4, actions=2, seed=7)        # the store is on by default
x = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.step(x)                                    # a sampled act
brain.step(x, reward=[1.0])                               # its reward is recorded under x
memory = brain.hippocampus
assert np.isclose(memory.recall(x)[0, action[0]], 1.0)    # memory_rate 1.0: the last outcome
brain.reset()                                             # a new stream keeps F and C
memory.reset(1)                                           # clear the fast residual F only
assert 0.0 < memory.recall(x)[0, action[0]] < 0.2         # what consolidation kept in C
memory.clear()                                            # erase F and C
assert not memory.recall(x).any()
```

| Setting | Founder | Changed by | What it does |
| --- | --- | --- | --- |
| `episodic` | `True` | construction | `False` builds the brain without a store |
| `memory_rate` | 1.0, in [0, 1] | `retune` | how far a write moves `F` toward the outcome; 1.0 keeps the last one |
| `memory_decay` | 0.9, in [0, 1] | `retune` | how much of `F` each later write keeps |
| `memory_amplitude` | 1.0 | `retune` | the strength of the read as drive |
| `consolidation` | 0.05, in [0, 1] | `retune` | the slow rate into the shared `C` |

`brain.hippocampus.writes` counts the writes. The store costs key width times actions numbers
for `C` and as many per stream for `F`. It has no separator: a write at key `k` moves the read
at `q` by `q . k` times the correction, so overlapping situations interfere. Under
`memory_choice=True` the same names set both episodic stores, and `compose` starts
`memory_rate` at 0.5.

## Episodic memory

`memory_choice=True` makes the hippocampus an episodic memory. Each record is kept twice,
keyed by the observation code (place) and by the code with a fading trace of what the stream
saw and did (`brain.context_memory`), each behind a fixed separator (none on a coded brain),
and the mean of the two reads joins the motor readout in every choice. It records known
outcomes and demonstrations and replays a reward back along the recent path. Its settings,
evidence and limits are on [episodic.md](episodic.md).

```python
frozen = {"actor_eta": 0.0, "actor_eta_bias": 0.0, "actor_eta_critic": 0.0}  # records alone
chooser = Brain.compose(inputs=3, actions=2, seed=0, memory_choice=True, **frozen)
place = np.array([[1.0, 0.0, 0.0]])
assert chooser.act(place, greedy=True)[0] == 0    # the newborn's answer
action = chooser.step(place)
for _ in range(6):                                # only action 1 pays
    action = chooser.step(place, reward=[1.0 if action[0] == 1 else 0.0])
assert chooser.act(place, greedy=True)[0] == 1    # the records vote for the action that paid
```

## Recall memory

<a id="recall-the-streams-own-moments"></a>

`recall=True` keeps each stream's recent moments as slots of key (the event heard), value (the
event) and next (the moment that followed). Before each settle an address, the key plus a
learned map that teacher lessons move, weighs the slots by a softmax, and the recalled value
and next drive a `recall` and a `next` region through the settle. `reset(keep_memory=True)`
keeps the slots. Every `recall_*` setting, the cost and the limits are on
[recall.md](recall.md), with the settings of 0.84.3 that left in this version and their
readings ([removed in this version](recall.md#removed-in-this-version)).

```python
brain = Brain.compose(inputs=4, actions=2, modules=(16,), seed=1, recall=True)
brain.act(np.array([[1.0, 0.0, 0.0, 1.0]]), greedy=True)  # a name (0) stated with a value (3)
brain.act(np.array([[0.0, 1.0, 0.0, 0.0]]), greedy=True)  # another moment
name = np.array([[1.0, 0.0, 0.0, 0.0]])
recalled = brain.stimulus(name)[0, brain.connectome.populations["recall"]]
assert recalled[3] > recalled[1]                  # the moment that shares the name comes back
assert brain.lesson(name, [1])["address_rows"] == 1.0   # the lesson steps the address map
brain.reset()                                     # slots emptied, the address map kept
assert brain.recall.slots == 0 and brain.recall.w_x.any()
```

## Learned weights and eligibility

The learned weights (synapse scales and biases, the actor's readout and the critic) are the
permanent memory. Lessons move them by the free/nudged contrast ([learning](learning.md)),
outcomes by dopamine times each synapse's eligibility ([reward](reward.md)). The eligibility
trace is the per-stream memory of which synapses produced recent sampled acts,
`e <- actor_gamma * actor_lam * e + contrast` at each outcome (0.9 and 0.8 at the founders),
so a reward a few moments later credits the act that earned it. Greedy acts keep no
eligibility, and a calm `live` moment fades it.

```python
brain = Brain.compose(inputs=4, actions=2, seed=0, episodic=False)  # weights only
x, labels = np.eye(4)[:2], np.array([1, 0])
for _ in range(50):
    if (brain.predict(x) == labels).all():
        break
    brain.lesson(x, labels)                       # the contrast moves the synapses
brain.reset()
assert (brain.predict(x) == labels).all()         # the graph alone answers, after a reset

actor = Brain.compose(inputs=3, actions=2, seed=0)
actor.step(np.array([[1.0, 0.0, 0.0]]))                # a sampled act holds its contrast
actor.step(np.array([[0.0, 1.0, 0.0]]), reward=[0.0])  # its outcome: the contrast joins the trace
assert np.abs(actor.basal_ganglia.trace).sum() > 0
actor.reset()
assert actor.basal_ganglia.trace is None               # a new stream begins without eligibility
```

## The observation code

`brain.observation_code(x)` returns the key every memory reads and writes: by default the
observation itself. `code=units` (construction; at least 32 inputs and 24 units) replaces it
with a fixed random sparse code of `units` width and unit length, one unit in ten active,
saved with the brain as `brain.sparse_code`. In `compose` a vector brain's sensory region
reads the code too; on an image brain and in `Brain(connectome, code=units)` only the
memories use it. Set it when wide observations make the episodic separators and the recall
address map too large: at 12,824 inputs they take 1,532 MB uncoded with `memory_choice` and
`recall`, and 9.8 MB with `code=1024`. A coded brain refuses `memory_expansion` and
`memory_fan_in`, and keys stored under one code do not match another.
`Brain.observation_width(inputs, code)` gives the key width before building, the size of a
custom anatomy's `recall` and `next` regions. The code is fixed and learns nothing; a learned
code is work in progress in [#211](https://github.com/muellerberndt/cadence/issues/211).

```python
plain = Brain.compose(inputs=2048, actions=3, seed=0)
coded = Brain.compose(inputs=2048, actions=3, seed=0, code=1024)
x = np.random.default_rng(0).random((1, 2048))
assert plain.observation_code(x).shape == (1, 2048)     # the observation itself
key = coded.observation_code(x)                         # what every memory keys on
assert key.shape == (1, 1024) and (key > 0).sum() == 102 and np.isclose(np.linalg.norm(key), 1.0)
assert Brain.observation_width(2048, 1024) == 1024      # the key width before building
assert len(coded.connectome.populations["sensory"]) == 1024  # the cortex reads the code
```

## Check that a memory does the work

Save the acquired brain, load two copies, erase one memory in one copy and compare their
answers on the same query. If they never differ, the brain answers without that memory.

```python
import os
import tempfile

# chooser and place from the episodic example above
path = os.path.join(tempfile.mkdtemp(), "brain.npz")
chooser.save(path)
remembering, forgetting = Brain.load(path), Brain.load(path)
forgetting.hippocampus.clear()                    # erase the place records
forgetting.context_memory.clear()                 # and the context records
assert remembering.act(place, greedy=True)[0] == 1  # the records vote for what paid
assert forgetting.act(place, greedy=True)[0] == 0   # without them the newborn answer returns
```

| Memory | Erase it on a copy |
| --- | --- |
| Working trace | `brain.working_memory.reset(1)` |
| Efference copy | `brain.efference.reset(1)` |
| Heard history | `brain.history.reset(1)` |
| Associative store | `brain.hippocampus.reset(1)` clears `F`; `brain.hippocampus.clear()` erases `C` too |
| Episodic memory | `brain.hippocampus.clear()` and `brain.context_memory.clear()` ([episodic](episodic.md#check-that-the-memory-does-the-work)) |
| Recall memory | `brain.recall.reset(1)` empties the slots |
| Any memory read | `brain.retune(<memory>_amplitude=0.0)` silences it and keeps what it holds |

## Stores outside the composed brain

<a id="records"></a>
<a id="reading-and-writing-directly"></a>

- [Standalone memory stores](records.md): `FastSynapses`, `PatternSeparator`, `SynapticMemory` and the `Trace` family used directly.
