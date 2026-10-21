# Recall memory

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

## How a moment is addressed

Before each settle the present moment computes its address

```text
q = heard * x + context * h + x W_x + c W_c + b
p = softmax(q . K / temperature - recency * age)
```

`x` is the observation as the memory hears it, `h` the fading history of earlier events,
`c` the settled association activity of the moment before, `K` the stored keys, and `W_x`,
`W_c`, `b` a learned map that starts at zero, so before any lesson the address is the key
and a name finds the moment it was stated in. `p` is kept in `brain.recall.attended`;
`brain.recall.charges()` gives each slot's age charge. The `recall` region is driven at
`amplitude * sum p V` and the `next` region at `amplitude * sum p N`, held through the settle as a sensory drive is: the settle answers
on the mixture the address chose and cannot revise the choice. Both regions reach the
association region through one-way plastic synapses of initial magnitude `recall_scale`.

`recall_recency=None` charges age as `r * log(1 + age)`, `r = heard^2 / temperature /
log(1 + capacity)`, so an unmatched old slot's weight falls as `(1 + age)^-r` and a full
match of a unit-length key outweighs every age the capacity holds. The linear charge suits
codes whose queries match in part; the power law suits keys that match fully over long
spans. A number `recall_null` adds a slot of logit `null / temperature` that reads nothing,
so a moment that matches no slot recalls little.

## How it writes and how the address learns

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

## Episodes, keys and windows

Each stream has its own slots. A plain `reset()` empties them and keeps the address map. When
an episode ends, that stream's slots are emptied with its working trace and heard history,
and the cortex of the moment before, the history and the window begin afresh.
`brain.reset(keep_memory=True)` keeps the slots across a reset: each stream ends an episode
there (`Recall.end`), and the ended episode's last slot learns no successor from the next
stream's first moment.

The founder key is the present event alone, so where an event recurs with different
continuations the read returns a mixture of every occurrence's successor. One construction
setting widens the key:

| Key | Holds | Address width |
| --- | --- | --- |
| the founder key | `heard * x_t`, plus `context * h` | `code` |
| `recall_window=k` | the ordered last k events, the event j moments back weighted `recall_window_decay^j` | `k * code` |

`code` is the width of the [observation code](memory.md#the-observation-code), and
`brain.recall.address_width` reports the result. Keys, address, map outputs and a reward
eligibility take that width; values, nexts and the two regions keep the value width.

## Settings

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
| `recall_recency` | `0.25` | retune | Logit cost per moment of age; `None` gives the power-law charge |
| `recall_address` | `0.01` | retune | Address lesson rate, bounded per row by `1 / \|z\|^2`; `0.0` freezes the map |
| `recall_address_temperature` | `0.1` | retune | Temperature of the slots' scores in the address lesson |
| `recall_context`, `recall_context_decay` | `0.0`, `0.5` | retune | Fading history of earlier events in a key, and its retention per moment in [0, 1) |
| `recall_capacity` | `64` | construction | Slots per stream, oldest dropped first |
| `recall_value` | `"event"` | construction | `"cortex"` stores the settled association activity and sizes the regions by the association width |
| `recall_next` | `True` | construction | The `next` region and each slot's next |
| `recall_scale` | `1.2` | construction | Initial magnitude of the two projections (a `compose` argument) |
| `recall_address_step` | `"lms"` | retune | `"toward"`: the named keys less the attended mixture |
| `recall_null` | `None` | retune | A number adds a null slot; a non-finite logit is refused |
| `recall_address_reward` | `0.0` | retune | Rate at which reward trains the address map |
| `recall_window`, `recall_window_decay` | `1`, `1.0` | construction | Window of the last k events, decay in (0, 1] |

The [API reference](api.md#recall-memory-genes) gives the `Recall` class signature.

## All options in one example

```python
optioned = Brain.compose(
    inputs=4, actions=2, modules=(16,), seed=1, recall=True,
    recall_address_step="toward", recall_recency=None, recall_null=0.0,
    recall_address_reward=0.1,
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

## Removed in this version

Settings of 0.84.3 that left in this version, each refused by name with a message naming
0.84.3 as the release that takes it, with the reading that retired it:

- `recall_write="aroused"`, a moment written only when its act was sampled: neutral on every
  chamber it was read on; every accepted moment is written.
- `recall_key="predecessor"`, the previous event and the last command in every key: neutral
  where it was read, and the window key and the efference copy carry the same information.
- `recall_episodes="keep"`, slots kept across an ended episode: it lowered success by .108 on
  a crafting chain whose observation names the stage; an ended episode empties the slots, and
  `reset(keep_memory=True)` keeps them across a reset as before.

## Cost

The two regions add one neuron per value coordinate each: `compose(250, 2)` grows from 380 to
880 neurons and from 20,354 to 52,354 synapses. The address map is `code` by `address_width`
plus the association width by `address_width`; with compose's 64 association neurons it is
0.66 MB at 256 inputs, 1,322 MB at 12,824 inputs on the uncoded observation and 8.9 MB with
`code=1024`. One stream's slots hold `capacity` keys, values and nexts, 0.39 MB at 256
inputs. The map is learned: a reset keeps it, `parameters()` counts it and a checkpoint saves it. The keys are the observation code, so
changing the code makes the stored keys inconsistent with new addresses.

## In a custom anatomy

`Brain(connectome, recall=True, ...)` needs a `recall` and a `next` region (or `recall` alone
with `recall_next=False`), each with one neuron per value coordinate:
`Brain.observation_width(inputs, code)` with event values, the association width with
`recall_value="cortex"`. Without `recall=True` populations of those names are ordinary ones.
`tests/test_recall_memory.py` checks that the library reproduces the reference memory
`benchmarks/binding/index_read.py` moment for moment. To check that the memory does the work,
erase a copy's slots with `brain.recall.reset(batch)` before the query
([ablation](memory.md#check-that-a-memory-does-the-work)).

## Limits

- The address is fixed before the settle, so a second recall takes a second moment.
- Chained recall, where the address depends on a recalled token and the question together,
  reaches the linear map only through conjunctions held in the cortex of the moment before.
- `recall_value="cortex"` carries no content the next settle can read back; the founder
  stores events.
- The slots are a record within one stream. Nothing consolidates across streams except
  through the plastic synapses that read them.
