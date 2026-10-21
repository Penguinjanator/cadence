# Memory

A memory is state that outlives one settle and changes a later answer. Everything else a
brain computes is settled afresh at each moment from the observation and its memories.

Each memory reaches the answer through the settle. The working trace, efference copy, heard
history and recall memory drive regions of their own that the association region reads
through plastic synapses; the associative store drives the motor neurons; the episodic memory
adds a vote to the motor readout. The answer is one settle, which has to
[qualify](start.md#qualification-and-refusal).

Memories have three lifetimes. Per stream, cleared by `reset()` and by an ended episode: the
working trace, efference copy, heard history, recall slots and actor eligibility. Across
resets: the associative and episodic records and the recall address map. Permanent until
learning changes them: the learned weights. The per-stream traces advance at every moment;
under [`live`](start.md#routine-and-repair-live) a calm brain reads its records and
weights and changes neither, and an aroused one learns and writes. `reset()` also clears
arousal's income and error references, which decide when a `live` brain wakes, and
`reset(keep=("arousal",))` keeps the arousal state as it is
([reset and save](start.md#reset-and-save)).

## Memory in the composed brain

<a id="three-kinds-of-memory-in-brain"></a>

| Memory | On by default | Switch | Holds | Keyed by | Written | Survives `reset()` | Home |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Working trace | yes | `working_memory_amplitude=0.0` silences it | the settled association activity of the moment before | none | after each act and each moment `wait` senses | no | [below](#working-trace) |
| Efference copy | no | `efference_amplitude=1.0` | a fading one-hot of the commands issued | none | after each issued act | no | [below](#efference-copy) |
| Heard history | no | `history_decays=(0.2, 0.8)` | the observations as heard, one fading block per decay | none | after each act and each moment `wait` senses | no | [below](#heard-history) |
| Associative store | yes | `episodic=False` removes it | one value per action per situation, in reward units | observation code | each outcome of the chosen action | yes | [below](#associative-store-the-hippocampus) |
| Episodic memory | no | `memory_choice=True` | place and context records that vote in the choice | observation code, and code with the temporal context | known outcomes, demonstrations, replay of the path to a reward | records yes, context no | [below](#episodic-memory-what-happened-in-this-situation-before) |
| Recall memory | no | `recall=True` | the stream's recent moments as slots, and a learned address map | observation code | a slot per moment; the map at lessons | map yes, slots no | [below](#the-recall-memory-in-full) |
| Learned code | no | `learned_code=units` | a dictionary, a predictor and thresholds: the code region's synapses and biases | none | every accepted moment, by local rules | yes | [the learned code](senses.md#the-learned-code) |
| Learned weights | yes | learning rates of zero freeze them | synapse scales and biases, actor and critic | none | lessons, and outcomes (aroused ones under `live`) | yes | [the free/nudged rule](settling.md#learning-the-freenudged-rule) |
| Actor eligibility | yes | `actor_lam=0.0` credits the last action only | each synapse's credit for recent sampled acts | none | each sampled act | no | [reward.md](reward.md#eligibility-traces-and-dopamine) |

`predict`, `accuracy` and `fit` answer independent samples and read none of these except the
learned weights, and `fit` starts with a `reset()`. Test a memory through `act`, `step` or
`live` on a continuing stream. `save` keeps every memory in the table, per-stream traces
included, and `Brain.load` continues the stream where it stopped.

`brain.memories` holds the present memories by name, in the order the brain reads them before
a settle and writes them after it: `working_memory`, `efference`, `history`, `recall`,
`hippocampus` and, with `memory_choice=True`, `episodic` (the episodic records: their vote,
the temporal context and the paths a reward is replayed along); `brain.memories["recall"]` is
`brain.recall`. Every member keeps one contract
(`cadence.memories`): `stimulus(drive, moment)` adds what it holds to the drive of a moment,
`branch()` gives an imagined branch a private copy, `begin(keep=False)` starts its streams
afresh as `reset` does, `reset(batch, rows=None)` and `keep(rows)` act on streams, and
`to_dict()` and `arrays()` are what a checkpoint keeps of it, under `memories/<name>/`.

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
| Alternate or sequence its own actions under identical input | efference copy at `efference_amplitude=3.0`, `efference_decay=0.0` | [rhythm](../benchmarks/rhythm/README.md), `modules=(32,)`, `episodic=False`, `working_memory_decay=0.1, scales={"trace": 12.0}`: alternation 0.97 and 0.93 against 0.50 and 0.40 with the trace alone; uniform random alternates at 0.50 | delayed-credit foraging ([key-door](../benchmarks/keydoor/README.md)): a copy at amplitude 0.3 feeds less at every delay, and at 3.0 the creature does not feed |
| Learn which action pays in a situation, and reverse quickly | associative store (on) | [reversal](../benchmarks/reversal/README.md), `modules=(32,)`, `working_memory_amplitude=0.3, scales={"trace": 12.0}, working_memory_decay=0.2, consolidation=0.25`, actor rate 0.1 (bias 0.01): memory alone 1.00 from 300 exposures against the graph alone 0.60 to 0.80 under rule A | at the composed founders the chamber stays at chance (`defaults` 0.48 to 0.53); noisy pay overwrites a one-shot record, next row |
| Noisy or signed pay | `memory_rate=0.2` | [competing](../benchmarks/competing/README.md): acquires as at 1.0 and halves the loss | half the loss is left |
| Situations whose keys overlap strongly | a standalone `FastSynapses` behind a centered, habituated `PatternSeparator` ([records](#standalone-memory-stores)); not available on the founder store | 64 keys of cosine 0.9, expansion 1024, 8 winners: revision accuracy at least 0.95, at least 0.5 above the plain store ([test](../tests/test_separation.py)) | the founder store has no separator; the episodic memory's fixed separator is uncentered and not measured by this test |
| Credit an action paid a few moments later | actor eligibility (on) | [key-door](../benchmarks/keydoor/README.md), `modules=(32,)` at the same operating point (scale 12.0, decay 0.2): fed 0.92 at delays 2 and 5; `actor_lam=0` falls to 0.53 at delay 5 | delay 10: 0.49 against a tabular learner's 0.83 to 0.90 |
| Learn a multi-step chain whose stage is observed | the founder (weights and store together) | [chain](../benchmarks/chain/README.md), pay at each correct step: 204 greedy completions per thousand moments against tabular 250; pay at the end only: 42, uniform random's rate | slow one-stream reward learning, see reversal |
| A state the observation does not show | no memory; supply the state as an observation | [chain](../benchmarks/chain/README.md) hidden: no arm beats the memoryless 62 | see [what breaks a chain](reward.md#what-breaks-a-chain) |
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
| `scales["trace"]` | 1.2 | construction (`compose` argument) | the past must matter from the first moment, before learning | the brain repeats itself and ignores what it sees |
| `modules` | `(64,)` | construction | more items or distractors: `modules=(128,)` | a narrow last module such as `(48, 24)` carries little |

`Brain(connectome, ...)` has a trace when the connectome has a `prefrontal` region.
`done=True` clears a stream's trace, so mark episodes only where the world restarts, and
teach with `learning_surprise=True` so lessons touch only the moments answered wrongly
([learning](teach.md#lessons-only-where-the-answer-differs)). The synapses that read the
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

`history_decays` (`()`, each in [0, 1)) and `scales["history"]` (1.2, the projection's initial
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
evidence and limits are in [episodic memory](#episodic-memory-what-happened-in-this-situation-before),
below.

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
and next drive a `recall` and a `next` region through the settle. `reset(keep=("slots",))`
keeps the slots. Every `recall_*` setting, the cost and the limits are in
[the recall memory in full](#the-recall-memory-in-full), with the settings of 0.84.3 that left in this version and their
readings ([removed in this version](#removed-in-this-version)).

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
permanent memory. Lessons move them by the free/nudged contrast ([learning](settling.md#learning-the-freenudged-rule)),
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
| Any member of `brain.memories` | `brain.memories[name].reset(1)` empties its stream: the trace, the history, the recall memory's slots, the associative store's `F`, the episodic context and path |
| Associative store | `brain.hippocampus.clear()` erases `C` too |
| Episodic memory | `brain.hippocampus.clear()` and `brain.context_memory.clear()` ([episodic](#check-that-the-memory-does-the-work)) |
| Any memory read | `brain.retune(<memory>_amplitude=0.0)` silences it and keeps what it holds |

## Stores outside the composed brain

- [Standalone memory stores](#standalone-memory-stores): `FastSynapses`, `PatternSeparator`, `SynapticMemory` and the `Trace` family used directly.

## Episodic memory: what happened in this situation before

Synapses learn slowly; an animal also learns in one go, from a place where food was found once
or a route a parent showed once. `Brain.compose(..., memory_choice=True)` adds an episodic
memory: records of what happened in a situation, which vote in every choice beside the slow
cortex. It is opt-in ([when to turn it on](#when-to-turn-it-on)); [memory](#episodic-memory)
places it among the brain's other memories.

### How it works

```text
observation now ─────────────────────► place records ───┐
seen and done before ─┐                                  ├─► mean ─► vote ─┐
observation now ──────┴─► temporal context ─► context records ─┘           ├─► softmax ─► action
observation now ─► settle ─► motor readout ─────────────────────────────────┘
```

- **Two keys.** Each record is kept twice. `brain.hippocampus` holds the place records, keyed
  by the present [observation code](senses.md#the-observation-code). `brain.context_memory`
  holds the context records, keyed by the temporal context: the present code with a fixed
  fading trace of past codes and of the brain's own issued actions, where `memory_context` is
  the share the past keeps from one moment to the next. Two moments that look alike after
  different histories get different context keys, so a cue that is gone can decide the vote.
  The place records keep a situation together when a new path reaches it.
- **Records** hold one value per action in reward units. An uncoded key passes a fixed random
  separation first (`memory_expansion` cells, the `memory_winners` most active kept), so
  different situations write different synapses. A coded brain keys on its code as it is.
- **What is written.** The outcome of the chosen action, once it is known: a reward or a
  punishment, the end of an attempt (`done`), a surprise that woke a calm `live` brain, or an
  outcome marked with a positive `salience=`. A zero outcome in the middle of an attempt
  writes nothing. A demonstration (`teacher=`) is recorded for its situation as if the shown
  action had paid `memory_demonstrations`. `done=True` makes the last outcome known and starts
  the next attempt with a fresh temporal context and path.
- **Replay.** A reward replays the stream's recent path backwards, up to `memory_replay`
  decisions, each credited up to the reward discounted by the actor's `gamma` per step.
  Replay only adds credit. Its rules:
  - the path keeps one decision per situation, the one that left it; a decision whose own
    outcome was a reward or a punishment stays on the path even in an unchanged situation;
  - replay runs through decisions whose outcome was nothing and stops at the last known
    outcome, so nothing has to mark where an attempt began;
  - a reward with `done=True` is replayed whole into both kinds of record;
  - in a continuing life a reward is replayed in proportion to its surprise (the reward less
    what the memory expected of the paid decision): into each decision's context record, and
    into its place record only as far as the decision was an exploration rather than the
    brain's best guess. An expected reward replays nothing.
- **The vote.** At every choice, greedy or sampled, the mean of the place and the context reads
  is added to the motor readout before the softmax. The vote never enters the drive, so the
  settle is unchanged. The cortex learns from the actions taken, so what the memory chose and
  what paid consolidates into its synapses over time.
- **Two speeds.** Each store is a [`SynapticMemory`](#associative-store-the-hippocampus).
  A write corrects the stream's record halfway at once (`memory_rate` 0.5), a correction that
  fades by `memory_decay` per later write. The record all streams share, which `reset` keeps,
  moves by `consolidation`, faster with a supplied `salience=`. One outcome is remembered at
  once, repetition makes it lasting, and one misleading outcome cannot erase it.

### One choice for acting, teaching and imagining

The vote joins after the settle, so the most active motor neuron is not always the act. Every
place that asks what the brain would do reads the same complete choice: readout plus vote, with
the context records read under that decision's own temporal context.

- **Lessons only where the answer differs** (`learning_surprise=True`,
  [learning](teach.md#lessons-only-where-the-answer-differs)) judge a row by the act the
  brain would issue. A demonstration corrects a brain whose memory turns a right cortical
  answer into a wrong act. An answer the memory makes right gets no lesson: no cortex update
  and no recorded demonstration. The cortex then learns those rows from the outcomes of its
  own acts.
- **Surprise in `live`** counts only the outcome of the brain's own best guess, the act it
  would issue greedily. A greedy act the vote decided is its own, and its outcome counts.
- **Imagination** (`imagine`) issues privately the act the brain would issue, to its efference
  copy, with the vote read under the branch's own temporal context. A returned state holds
  the cortex's response before the vote; act greedily
  on a [loaded copy](#check-that-the-memory-does-the-work) to read the complete act.

### Turning it on

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=3, actions=2, arousal=True, memory_choice=True, seed=0)
x = np.array([[1.0, 0.0, 0.0]])
assert brain.act(x, greedy=True)[0] == 0   # the newborn's answer
brain.step(x, teacher=[1])                 # shown once: recorded at memory_demonstrations
assert brain.live(x)[0] == 1               # the record votes for the shown action
```

`memory_choice=True` in `Brain.compose` brings the founder values in the table below; each can
be passed beside it, also with `connectome=` for a custom anatomy. `Brain(...)` takes the
same names, but its defaults are 0 (and `memory_winners` 32, `memory_rate` 1.0), so name
every setting there.
`memory_choice` needs `episodic=True`.

### Episodic settings

All of these are construction settings: pass them at construction; `retune` refuses them.

| Setting | With `memory_choice=True` | Change it when |
| --- | --- | --- |
| `memory_choice` | `True` (founder `False`) | `False` keeps the [associative store](#associative-store-the-hippocampus), keyed by the senses and read into the motor drive |
| `memory_context` | `0.5` | Raise toward 1 (exclusive) when the answer depends on what came long before; lower when history does not matter; 0 keeps the place records alone and builds no `context_memory` |
| `memory_replay` | `8` | Raise when rewards come at the end of long chains; 0 replays nothing |
| `memory_demonstrations` | `1.0` | Raise when shown actions should outweigh the brain's own trials; lower for an unreliable teacher; 0 records none |
| `memory_expansion` | `1024` (0 on a coded brain) | Raise when many situations must be told apart; 0 uses the key unseparated. With `code=units` it is set to 0 and a nonzero value is refused |
| `memory_winners` | `24` | How many separation cells stay active per key; at most `memory_expansion` |
| `memory_fan_in` | `0` (dense) | Above 0 each separation cell reads that many random key values; use 32 for keys wider than 256 values. Needs `memory_expansion` and at most the key width; refused with `code=units` |

A dense separation holds `width x 1,024` floats per store (105 MB at 12,824 inputs); with
`memory_fan_in=32` it holds `1,024 x 32` indices and weights at any width. A brain with
[`code=units`](senses.md#the-observation-code) has no separation of its own.

`memory_rate` (0.5 here, 1.0 without `memory_choice`), `memory_decay`, `memory_amplitude` and
`consolidation` belong to the [associative store](#associative-store-the-hippocampus);
`retune` changes them for both stores together.

### When to turn it on

- A skill is taught by demonstration or by correcting executed mistakes.
- A chain of actions is paid only at its end ([sequential tasks](reward.md#what-breaks-a-chain)).
- The right answer depends on the situation's history: a cue that is gone, the next step of a
  sequence of the brain's own actions.

Leave it off where the history does not matter, and compare against the composition without it
on the task. The mechanism contracts are in
[`tests/test_episodic_memory.py`](../tests/test_episodic_memory.py) (writes, replay, the
temporal context recalling a cue that is gone) and
[`tests/test_episodic_policy.py`](../tests/test_episodic_policy.py) (the one-choice rule).

### Check that the memory does the work

Compare the same brain with and without its records, on loaded copies. Here the actor is frozen,
so only the records can change the answer:

```python
import os
import tempfile

frozen = {"actor_eta": 0.0, "actor_eta_bias": 0.0, "actor_eta_critic": 0.0}
paid = Brain.compose(inputs=3, actions=2, memory_choice=True, seed=0, **frozen)
assert paid.act(x, greedy=True)[0] == 0          # the newborn's answer
action = paid.step(x)
for _ in range(6):                               # only action 1 pays
    action = paid.step(x, reward=[1.0 if action[0] == 1 else 0.0])
path = os.path.join(tempfile.mkdtemp(), "brain.npz")
paid.save(path)
remembering, forgetting = Brain.load(path), Brain.load(path)
forgetting.hippocampus.clear()                   # erase the place records
forgetting.context_memory.clear()                # and the context records
assert remembering.act(x, greedy=True)[0] == 1   # the records choose the action that paid
assert forgetting.act(x, greedy=True)[0] == 0
```

If the answers never differ, the cortex has consolidated what the records held, or they hold
nothing for that key. [Memory](#check-that-a-memory-does-the-work) lists the erase call
for every memory.

### Episodic limits

- Storage is fixed by the key width and the separation and does not grow with experience;
  correlated keys interfere, as in every delta-rule store.
- The keys are not learned: the memory recalls the same place and the same context, and does
  not discover which part of a history matters.
- Where history matters, the place records of aliased moments average over histories and
  halve the context records' share of the vote.
- Records are in reward units: rewards of a very different size weigh the vote more or less
  against the readout. `memory_amplitude` scales the read.
- A record that paid is not extinguished by failures that pay nothing: a zero outcome in the
  middle of an attempt writes nothing, and replay only adds credit. In a continuing life where
  a once-paid action stops paying and its failures pay 0, its record keeps voting for it and
  the new action is often not learned: in a two-armed bandit through `live` whose paying arm
  moved, 4 of 12 lives learned the new arm, against 12 of 12 when failures cost 1. Where a
  learned answer can stop paying, let failures cost a little, mark them with a positive
  `salience=`, which makes them known outcomes that the records write, or leave
  `memory_choice` off.
- `predict` and `fit` answer independent samples and read no memory.

## The recall memory in full

`Brain.compose(..., recall=True)` keeps each stream's recent moments as slots and gives one
back when the present asks for it. Use it when a moment heard once has to return by its
content (a name stated with a value, asked for later), or when the brain needs what followed
an earlier moment that matches the present. The [memory overview](memory.md) compares it
with the other memories; the founder `recall=False` is the control. Each slot holds a key
(the event as heard), a value (the event) and a next (the value of the moment that
followed). Two source regions, `recall` and `next`, carry the recalled value and next into
the association region.

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, modules=(16,), seed=1, recall=True)
fact = np.array([[1.0, 0.0, 0.0, 1.0]])          # a name (0) stated with a value (3)
brain.act(fact, greedy=True)                     # each act writes one slot per stream
brain.act(np.array([[0.0, 1.0, 0.0, 0.0]]), greedy=True)
name = np.array([[1.0, 0.0, 0.0, 0.0]])
drive = brain.stimulus(name)                     # the name addresses the slots before its settle
recalled = drive[0, brain.connectome.populations["recall"]]
assert recalled[3] > recalled[1]                 # the moment that shares the name comes back
report = brain.lesson(name, [1])                 # a lesson also steps the address map
assert report["address_rows"] == 1.0
brain.reset()                                    # slots emptied, the learned address map kept
assert brain.recall.slots == 0 and brain.recall.w_x.any()
```

### How a moment is addressed

Before each settle the present moment computes its address

```text
q = heard * x + x W_x + c W_c + b
p = softmax(q . K / temperature - recency * age)
```

`x` is the observation as the memory hears it, `c` the settled association activity of
the moment before, `K` the stored keys, and `W_x`,
`W_c`, `b` a learned map that starts at zero, so before any lesson the address is the key
and a name finds the moment it was stated in. `p` is kept in `brain.recall.attended`. The
`recall` region is driven at
`amplitude * sum p V` and the `next` region at `amplitude * sum p N`, held through the settle as a sensory drive is: the settle answers
on the mixture the address chose and cannot revise the choice. Both regions reach the
association region through one-way plastic synapses of initial magnitude `scales["recall"]`.

A number `recall_null` adds a slot of logit `null / temperature` that reads nothing, so a
moment that matches no slot recalls little.

### How it writes and how the address learns

After each act, and after each moment `wait` senses, every stream writes one slot and the
slot before learns the new value as its next. Beyond `recall_capacity` the oldest slot is
dropped. Imagination writes none.

The address lesson runs after each teacher lesson (`step(..., teacher=...)` or
`brain.lesson`). It reads the association region's movement between the nudged phases back
through the recall and next synapses, scores each slot's value and next against it, weighs
the scores by `softmax(score / address_temperature + log p)` and steps the map toward the
keys so weighted (`K*`) at the taught rows, each at the rate `min(recall_address, 1 / |z|^2)`
on `z = [x, c, 1]`. The report gives `address_rows`, `address_error` and `address_refused`
(rows whose non-finite step was refused). The founder step `"lms"` takes the error `K* - q`;
`recall_address_step="toward"` takes `K* - sum p K`, raising the named slots against the
attended ones whatever the keys' norms.

At the founder `recall_address_reward=0` rewards leave the map as it is. Above zero, each
sampled act keeps the step the address lesson would take, were its action a lesson's label,
as the map's eligibility. The actor holds it in a trace of its own (decaying by
`gamma * lam`, see [reward](reward.md)), and each outcome steps the map by
`recall_address_reward` times the dopamine times that trace; `learn` reports it as
`address_reward_step`. At 1.0 a unit of dopamine moves the map by one lesson's step.
`retune` refuses a change of `recall_address_step`, `recall_address` or
`recall_address_reward` while a sampled action's address contrast is pending.

### Episodes, keys and windows

Each stream has its own slots. A plain `reset()` empties them and keeps the address map. When
an episode ends, that stream's slots are emptied with its working trace and heard history,
and the cortex of the moment before, the history and the window begin afresh.
`brain.reset(keep=("slots",))` keeps the slots across a reset: each stream ends an episode
there (`Recall.end`), and the ended episode's last slot learns no successor from the next
stream's first moment.

The founder key is the present event alone, so where an event recurs with different
continuations the read returns a mixture of every occurrence's successor. One construction
setting widens the key:

| Key | Holds | Address width |
| --- | --- | --- |
| the founder key | `heard * x_t` | `code` |
| `recall_window=k` | the ordered last k events, the event j moments back weighted `recall_window_decay^j` | `k * code` |

`code` is the width of the [observation code](senses.md#the-observation-code), and
`brain.recall.address_width` reports the result. Keys, address, map outputs and a reward
eligibility take that width; values, nexts and the two regions keep the value width.

### Recall settings

Each gene is passed as `recall_<name>` beside `recall=True`; a `recall_` name without it is
refused. Retune genes change in a life through `brain.retune`; construction genes are fixed
when the brain is built. The options from `recall_address_step` on are described and saved
only where they differ from their founder. A saved brain keeps its genes, slots and address
map.

| Setting | Founder | When | What it sets |
| --- | --- | --- | --- |
| `recall` | `False` | construction | Adds the memory and its `recall` and `next` regions |
| `recall_amplitude` | `3.0` | retune | Drive of a slot that holds the whole weight |
| `recall_temperature` | `0.25` | retune | Softmax temperature over the slots |
| `recall_heard` | `1.0` | retune | Weight of the present event in its key and address |
| `recall_recency` | `0.25` | retune | Logit cost per moment of age |
| `recall_address` | `0.01` | retune | Address lesson rate, bounded per row by `1 / \|z\|^2`; `0.0` freezes the map |
| `recall_address_temperature` | `0.1` | retune | Temperature of the slots' scores in the address lesson |
| `recall_capacity` | `64` | construction | Slots per stream, oldest dropped first |
| `recall_next` | `True` | construction | The `next` region and each slot's next |
| `scales["recall"]` | `1.2` | construction | Initial magnitude of the two projections (a `compose` argument) |
| `recall_address_step` | `"lms"` | retune | `"toward"`: the named keys less the attended mixture |
| `recall_null` | `None` | retune | A number adds a null slot; a non-finite logit is refused |
| `recall_address_reward` | `0.0` | retune | Rate at which reward trains the address map |
| `recall_window`, `recall_window_decay` | `1`, `1.0` | construction | Window of the last k events, decay in (0, 1] |

The [API reference](api.md#recall-memory-genes) gives the `Recall` class signature.

### All options in one example

```python
optioned = Brain.compose(
    inputs=4, actions=2, modules=(16,), seed=1, recall=True,
    recall_address_step="toward", recall_null=0.0, recall_address_reward=0.1,
)
other = np.array([[0.0, 1.0, 0.0, 0.0]])
optioned.act(fact)                               # every act writes a slot, sampled or greedy
optioned.act(other, greedy=True)
assert optioned.recall.slots == 2
optioned.stimulus(np.array([[0.0, 0.0, 1.0, 0.0]]))  # matches no slot
assert optioned.recall.attended.sum() < 1.0      # the null slot takes the rest
optioned.act(name)                               # sampled: the map's eligibility is kept
report = optioned.learn(np.array([1.0]), np.array([True]), fact)  # the outcome ends the episode
assert report["address_reward_step"] > 0.0
assert optioned.recall.slots == 0                # an ended episode empties the slots
windowed = Brain.compose(inputs=4, actions=2, modules=(16,), seed=1, recall=True, recall_window=2)
assert windowed.recall.address_width == 2 * 4
```

### Removed in this version

Settings of 0.84.3 that left in this version, each refused by name with a message naming
0.84.3 as the release that takes it, with the reading that retired it:

- `recall_write="aroused"`, a moment written only when its act was sampled: neutral on every
  chamber it was read on; every accepted moment is written.
- `recall_key="predecessor"`, the previous event and the last command in every key: neutral
  where it was read, and the window key and the efference copy carry the same information.
- `recall_episodes="keep"`, slots kept across an ended episode: it lowered success by .108 on
  a crafting chain whose observation names the stage; an ended episode empties the slots, and
  `reset(keep=("slots",))` keeps them across a reset as before.
- `recall_recency=None`, the power-law age charge `r * log(1 + age)`: it failed its
  confirmation on the camera task's founders; a number keeps the linear charge
  `recency * age`.
- `recall_value="cortex"`, the settled association activity as a slot's value: it failed all
  four binding tasks (token .06, succession .07, fresh bindings .55, camera .00); the values
  are the events.
- `recall_context` and `recall_context_decay`, a fading history of the earlier events added
  to every key: never read above zero in a chamber.

A checkpoint of 0.84.3 carries `value="event"`, `context=0.0` and `context_decay=0.5`, the one
value each had, with an empty history array; it loads, and neither `describe()` nor a
checkpoint of this version carries them.

### Recall cost

The two regions add one neuron per value coordinate each: `compose(250, 2)` grows from 380 to
880 neurons and from 20,354 to 52,354 synapses. The address map is `code` by `address_width`
plus the association width by `address_width`; with compose's 64 association neurons it is
0.66 MB at 256 inputs, 1,322 MB at 12,824 inputs on the uncoded observation and 8.9 MB with
`code=1024`. One stream's slots hold `capacity` keys, values and nexts, 0.39 MB at 256
inputs. The map is learned: a reset keeps it, `parameters()` counts it and a checkpoint saves it. The keys are the observation code, so
changing the code makes the stored keys inconsistent with new addresses.

### In a custom anatomy

`Brain(connectome, recall=True, ...)` needs a `recall` and a `next` region (or `recall` alone
with `recall_next=False`), each with one neuron per value coordinate,
`Genome.observation_width(inputs, code)`. Without `recall=True` populations of those names
are ordinary ones.
`tests/test_recall_memory.py` checks that the library reproduces the reference memory
`benchmarks/binding/index_read.py` moment for moment. To check that the memory does the work,
erase a copy's slots with `brain.recall.reset(batch)` before the query
([ablation](#check-that-a-memory-does-the-work)).

### Recall limits

- The address is fixed before the settle, so a second recall takes a second moment.
- Chained recall, where the address depends on a recalled token and the question together,
  reaches the linear map only through conjunctions held in the cortex of the moment before.
- The slots are a record within one stream. Nothing consolidates across streams except
  through the plastic synapses that read them.

## Standalone memory stores

These stores work next to a `NeuralGraph` without `Brain.compose`: `cd.FastSynapses` with
its `cd.PatternSeparator`, `cd.SynapticMemory` used directly, and the fading traces of
`cadence.stream`. Each writes by one delta-rule correction or a fixed decay, and none
settles. The memories of a composed brain are in [Memory](memory.md); signatures are in the
[API reference](api.md#streams-cadencestream).

### One correction

For a unit key `k`, value `v` and matrix `M`, a write is `M += rate * outer(k, v - k @ M)`
(normalized LMS, close to [delta-rule fast weights](https://arxiv.org/abs/2406.06484)). At
`rate=1` the key reads back `v` at once, a repeated correct observation has zero residual,
and the read at an earlier key `q` moves by `dot(q, k)` times the error: orthogonal records
are kept, overlapping ones interfere. A read is one linear product, with no softmax and no
settling. `FastSynapses(rule="delta")` and `SynapticMemory` scale keys to unit length. A
fixed matrix cannot hold contradictory values, nearly parallel keys or more independent
values than its key rank exactly; measure retention on the streams the store will see.

```python
import numpy as np
import cadence as cd

def write(M, key, value, rate=1.0):
    error = value - key @ M
    M += rate * np.outer(key, error)
    return error

M, q, p, k = np.zeros((4, 2)), np.array([0.6, 0.8, 0, 0]), np.eye(4)[2], np.eye(4)[0]
write(M, q, np.array([1.0, 0.0]))
write(M, p, np.array([0.0, 1.0]))
old_q = q @ M
error = write(M, k, np.array([0.0, 2.0]))                 # k overlaps q, is orthogonal to p
assert np.allclose(k @ M, [0.0, 2.0])                     # read back exactly
assert np.allclose(q @ M - old_q, (q @ k) * error)        # q moved by the overlap
assert np.allclose(p @ M, [0.0, 1.0])                     # p kept its record
assert np.allclose(write(M, k, np.array([0.0, 2.0])), 0)  # a repeat writes nothing
```

### A store or a settling learner

A delta-rule store suffices when the outcome follows its key (the consequence of an action,
the reward of a choice, a fact seen at a place), the key holds what decides it, and keys
that should generalise overlap. Completing a partial reading, carrying context across a
delay and crediting late reward need a settled region ([write a cortex](anatomy.md#write-a-cortex)); the
[certificate](settling.md#the-settling-certificate) covers that settling and not the stores. A dense learner
moves every prediction with each update, which rehearsal of retained real observations
counters. Give a baseline the same replay policy, and never replay obsolete evidence.

### Reading and writing directly

`FastSynapses(pre, post, decay=1.0, rate=1.0, amplitude=1.0, normalize=False, replace=False,
rule="hebb", separator=None)` holds one `(pre, post)` matrix per stream between declared
neuron indices.

```python
memory = cd.FastSynapses(np.arange(4), np.arange(4, 6), rule="delta")
keys, values = np.eye(4)[:2], np.array([[1.0, 0.0], [0.0, 1.0]])   # two streams
memory.observe(keys, values)
assert np.allclose(memory.recall(keys), values)
memory.reset(2, rows=np.array([True, False]))          # only the first stream's episode ended

connectome = cd.Connectome.from_synapses(
    6, pre=[0, 1, 2, 3], post=[4, 4, 5, 5], count=[1] * 4,
    populations={"key": range(4), "value": (4, 5)},
)
graph = cd.NeuralGraph(connectome, cd.learning_neuron_model())
drive = np.zeros((2, connectome.n))
drive[:, :4] = keys
state = graph.settle_batch(memory.stimulate(drive))   # the read is added to the value neurons
memory.update(state, write=np.array([True, False]), post=np.array([[1.0, 0.0], [0.0, 0.0]]))
```

`observe(key, value, write=None)` takes `(batch, width)` ports; a boolean `write` gates each
stream, and every observe first decays all strengths by `decay` (reads never do).
`recall(key)` reads keys, `read(drive)` the key columns of a drive, and `stimulate(drive,
inplace=False, key=None)` adds the read of the drive's key columns, or of `key`, into the
post columns. `update(state, write=None,
post=None)` writes from the settled key activation with `post` as the observed value, and
only decays when `write` is `None`. Because `update` keys on activation and `stimulate` on
drive, a nonlinear activation can write under a key direction the read never uses; use
`observe`/`recall` when one external key serves both. Never write the store's own prediction back as evidence.

`rule="delta"` needs `rate` in `[0, 1]`, refuses `normalize` and `replace`, and a zero key
writes nothing. Scaling a key to unit length (`rule="delta"` or `normalize=True`) divides by
the norm of the whole key vector, the one step of the store that reads more than a synapse. The default `rule="hebb"` adds `rate * outer(k, v)`: `normalize=True` uses
unit keys and divides a read by the decayed write mass, and `replace=True` clears the rows of
a key's positive coordinates first, which suits one-hot slots and erases records under dense
keys. `amplitude` scales the read without changing the write target. An overflowing write
raises `ValueError` and changes nothing. `strength` holds the `(batch, key_width, post)`
matrices (`key_width` is the separator's `expansion`, else the pre count) and `writes`
counts rows written. `Learner.save` saves neither these stores nor the traces below; the
caller saves them.

`SynapticMemory(pre, post, decay=0.9, consolidation=0.05, separator=None, ...)` adds a
persistent `consolidated` matrix shared by all streams under the fast residual. Its rule,
salience and `value_mask` are in [the associative store](#associative-store-the-hippocampus).

### Streams and independent episodes

Rows of a batch are independent streams. Score a prediction before revealing its label.
On `FastSynapses`, `SynapticMemory`, every `Trace` and `History`, `reset(batch, rows=mask)`
clears the streams that ended and `keep(rows)` drops streams while the rest keep their
identities; `reset(batch)` without rows, or at a new batch size, clears all. A read at
another batch size returns the empty-stream baseline and leaves live records alone; a write
at another batch size starts fresh streams. `SynapticMemory.reset` clears only the fast
residuals and keeps the consolidated matrix, which is also its baseline; `clear()` erases both.

### Pattern separation

`PatternSeparator(inputs, expansion, winners, seed=0, center=0.0, fan_in=0, cover=0)` codes
keys before a store sees them: a fixed random projection onto `expansion` units, of which the
`winners` strongest positive entries stay. Codes with disjoint winners do not interfere. On
keys of cosine 0.9 a separated delta store revises at least 95 percent of its records
correctly, 50 points above the plain store ([separation tests](../tests/test_separation.py)).

```python
key_neurons, value_neurons = np.arange(32), np.arange(32, 40)
sep = cd.PatternSeparator(inputs=32, expansion=1024, winners=8, seed=0, center=0.99)
sep.habituate(np.random.default_rng(0).standard_normal((256, 32)))   # the environment's keys
separated = cd.FastSynapses(key_neurons, value_neurons, rule="delta", separator=sep)
separated.observe(np.ones((1, 32)), np.eye(8)[:1])
assert separated.key_width == 1024 and np.allclose(separated.recall(np.ones((1, 32))), np.eye(8)[:1])
```

`code(key, learn=False)` returns the `(batch, expansion)` code. With `center` above zero the
running `mean` is subtracted first, writes move it by forgetting factor `center`, and
`habituate(keys)` sets it from a sample before anything is stored; a moving mean can shift a
stored key's code. `fan_in` gives each code unit that many inputs (`index` and `weight` in
place of `projection`); `cover` (needs `fan_in`) reads each input through an ON and an OFF
channel with positive weights, each channel read by `cover` units. The projection is fixed:
it does not learn which differences matter. Storage and capacity grow to `expansion` rows
per stream. `SynapticMemory` refuses `center` other than 0, since a moving mean would move
persistent addresses; apply one fixed centering to inputs and queries instead.

### Traces

Fading traces carry the moment before into the next settle, one row per stream. Each reads
a `source` population and writes `amplitude` times the trace into the paired `target`
population through `stimulate(drive)`; decays lie in `[0, 1)`.

| Name | Defaults | Behaviour |
|---|---|---|
| `Trace(connectome, decay=0.5, amplitude=1.0, source="hidden", target="context")` | | `update(state)` moves the trace toward the source activation, `t <- decay t + (1 - decay) h`. `ringing(floor=0.1)` gives each source neuron's share of the trace, a salience for `ActorCritic.salience`. |
| `Efference(...)` | `source="motor"`, `target="efference"` | `issue(command)` decays toward the `(batch, motor)` one-hot of the executed action; `update` raises. |
| `History(connectome, decays, amplitude=3.0, source="sensory", target="history")` | | One target block per decay; `hear(observations)` moves block `d` by `h = d h + (1 - d) x`. |
| `cadence.stream.columns(index)` | | A slice for a contiguous range, else the index array; slices avoid gather copies on wide batches. |

Inside a composed brain the working trace, efference copy and heard history are owned by
[Memory](#working-trace).
