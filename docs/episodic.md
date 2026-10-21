# Episodic memory: what happened in this situation before

Synapses learn slowly; an animal also learns in one go, from a place where food was found once
or a route a parent showed once. `Brain.compose(..., memory_choice=True)` adds an episodic
memory: records of what happened in a situation, which vote in every choice beside the slow
cortex. It is opt-in ([when to turn it on](#when-to-turn-it-on)); [memory](memory.md#episodic-memory)
places it among the brain's other memories.

## How it works

```text
observation now ─────────────────────► place records ───┐
seen and done before ─┐                                  ├─► mean ─► vote ─┐
observation now ──────┴─► temporal context ─► context records ─┘           ├─► softmax ─► action
observation now ─► settle ─► motor readout ─────────────────────────────────┘
```

- **Two keys.** Each record is kept twice. `brain.hippocampus` holds the place records, keyed
  by the present [observation code](memory.md#the-observation-code). `brain.context_memory`
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
- **Two speeds.** Each store is a [`SynapticMemory`](memory.md#associative-store-the-hippocampus).
  A write corrects the stream's record halfway at once (`memory_rate` 0.5), a correction that
  fades by `memory_decay` per later write. The record all streams share, which `reset` keeps,
  moves by `consolidation`, faster with a supplied `salience=`. One outcome is remembered at
  once, repetition makes it lasting, and one misleading outcome cannot erase it.

## One choice for acting, teaching and imagining

The vote joins after the settle, so the most active motor neuron is not always the act. Every
place that asks what the brain would do reads the same complete choice: readout plus vote, with
the context records read under that decision's own temporal context.

- **Lessons only where the answer differs** (`learning_surprise=True`,
  [learning](learning.md#lessons-only-where-the-answer-differs)) judge a row by the act the
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

## Turning it on

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
be passed beside it. `Brain(...)` and `Brain.build(...)` take the same names, but their
defaults are 0 (and `memory_winners` 32, `memory_rate` 1.0), so name every setting there.
`memory_choice` needs `episodic=True`.

## Settings

All of these are construction settings: pass them at construction; `retune` refuses them.

| Setting | With `memory_choice=True` | Change it when |
| --- | --- | --- |
| `memory_choice` | `True` (founder `False`) | `False` keeps the [associative store](memory.md#associative-store-the-hippocampus), keyed by the senses and read into the motor drive |
| `memory_context` | `0.5` | Raise toward 1 (exclusive) when the answer depends on what came long before; lower when history does not matter; 0 keeps the place records alone and builds no `context_memory` |
| `memory_replay` | `8` | Raise when rewards come at the end of long chains; 0 replays nothing |
| `memory_demonstrations` | `1.0` | Raise when shown actions should outweigh the brain's own trials; lower for an unreliable teacher; 0 records none |
| `memory_expansion` | `1024` (0 on a coded brain) | Raise when many situations must be told apart; 0 uses the key unseparated. With `code=units` it is set to 0 and a nonzero value is refused |
| `memory_winners` | `24` | How many separation cells stay active per key; at most `memory_expansion` |
| `memory_fan_in` | `0` (dense) | Above 0 each separation cell reads that many random key values; use 32 for keys wider than 256 values. Needs `memory_expansion` and at most the key width; refused with `code=units` |

A dense separation holds `width x 1,024` floats per store (105 MB at 12,824 inputs); with
`memory_fan_in=32` it holds `1,024 x 32` indices and weights at any width. A brain with
[`code=units`](memory.md#the-observation-code) has no separation of its own.

`memory_rate` (0.5 here, 1.0 without `memory_choice`), `memory_decay`, `memory_amplitude` and
`consolidation` belong to the [associative store](memory.md#associative-store-the-hippocampus);
`retune` changes them for both stores together.

## When to turn it on

- A skill is taught by demonstration or by correcting executed mistakes.
- A chain of actions is paid only at its end ([sequential tasks](sequential-tasks.md#what-breaks-a-chain)).
- The right answer depends on the situation's history: a cue that is gone, the next step of a
  sequence of the brain's own actions.

Leave it off where the history does not matter, and compare against the composition without it
on the task. The mechanism contracts are in
[`tests/test_episodic_memory.py`](../tests/test_episodic_memory.py) (writes, replay, the
temporal context recalling a cue that is gone) and
[`tests/test_episodic_policy.py`](../tests/test_episodic_policy.py) (the one-choice rule).

## Check that the memory does the work

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
nothing for that key. [Memory](memory.md#check-that-a-memory-does-the-work) lists the erase call
for every memory.

## Limits

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
