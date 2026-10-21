# Continuous interaction with Brain

A `Brain.compose` brain is one continuing life with no training/inference switch.
Each call is one moment: the body reports the outcome of the preceding action, the
brain settles on the present observation and issues the next action. Observations
have shape `(streams, inputs)`, the other vectors one entry per stream, and each row
stays the same stream until `reset()`. Use `live` for one creature whose arousal
decides when to learn, and `step` for batches, teacher labels or learning from every
outcome. The [quickstart](quickstart.md) is the first run, [world model](world-model.md)
the design.

## Routine and repair: `live`

`live(observations, *, reward=None, done=None, decision_id=None)` follows one stream
of a brain composed with `arousal=True`. A calm brain answers with the greedy choice
of one qualified settle and learns nothing: no eligibility phases, no parameter
change, no associative write; earlier eligibility fades one step. Two things rouse it:

- **Surprise**: the outcome contradicts the forecast made before acting, which is
  the critic's value of the situation and, weighed by `record_surprise`, the
  associative record held for the chosen action. Only outcomes of the brain's own
  greedy choices can surprise it or move its usual error.
- **Want**: the recent reward falls below what this life usually pays, below
  the body's `need`, or below `need_scale` times the spread of the rewards the
  brain has learned from. The need's want is the unmet share of the need; it never
  habituates and a rare reward's small mean cannot dilute it. The scaled need is a
  need in units of that spread (opt-in, founder 0): it applies after youth, once an
  outcome has entered the scale, and follows what the life's rewards spread over.

An aroused brain samples its policy, learns from every outcome as `step` does and
writes the outcome that woke it to memory. Want heats one uniformly chosen motor slot
to `learning.temperature * (1 + heat * want)`. Every actual reward updates the income
averages, so a brain habituates to a poorer life unless a positive `need` keeps it
wanting.

```python
import numpy as np
from cadence import Brain

cues = np.eye(4)
observation, following = cues[[0]], cues[[1]]
brain = Brain.compose(
    4, 2, modules=(16,), seed=0,
    consolidation=0.25,                  # lasting memory takes half of a witnessed unit outcome
    arousal=True, arousal_youth=30,      # a short youth for a short life
    actor_eta=0.1, actor_eta_bias=0.01,  # one stream's operating point on the odour nursery
)
rng = np.random.default_rng(0)
cue = 0
action = brain.live(cues[[cue]])
modes, hits = [], []
for moment in range(600):
    rewarded = cue % 2 if moment < 300 else 1 - cue % 2   # the rule turns over halfway
    hits.append(action[0] == rewarded)
    cue = int(rng.integers(4))
    action = brain.live(cues[[cue]], reward=[1.0 if hits[-1] else -1.0])
    modes.append(brain.last_arousal["mode"])

assert modes[250:300].count("routine") >= 45 and sum(hits[250:300]) >= 45  # calm and right
assert modes[300:340].count("aroused") >= 20                               # the change wakes it
assert modes[-50:].count("routine") >= 45 and sum(hits[-50:]) >= 45        # repaired, calm again
assert brain.arousal.moments["routine"] > 4 * brain.arousal.moments["aroused"]
```

`consolidation` and the actor rates are one stream's operating point on the
[odour nursery](../benchmarks/reversal/README.md), where the composed actor rate locks
a single stream's policy. Select them again per task; [`retune`](brain.md#retune-the-same-life)
changes them in the same life.

The law, per outcome with unsigned temporal-difference error `error`:

```text
surprise = log(error / (tolerance * usual + floor * scale))   when positive, else 0
want     = max(clip((longrun - recent) / scale, 0, 1), clip((need - recent) / need, 0, 1),
               clip((need_scale * scale - recent) / (need_scale * scale), 0, 1))
level    = decay * level + (1 - decay) * (surprise + want)
```

`usual` is the running own-choice error, `recent` and `longrun` the reward at the
`fast` and `slow` rates, `scale` the spread of learned outcomes around `longrun`; a
want term with a zero denominator is zero. The `need_scale` term reads the scale as it
stood before the outcome entered it, applies after `youth` and is zero while no outcome
has entered the scale; the want and heat read afterwards use the scale as it stands. The
record channel has its own `usual`, and surprise is the larger weighted channel. The
brain is aroused while `level >= threshold` and for its first `youth` moments. Set genes
with `arousal_<field>` at `compose` or `retune`
([composition](brain.md#defaults-and-expert-overrides)); construction refuses values
outside their ranges, a negative `need` or `need_scale` and `fast < slow`.

| Gene | Founder | Effect |
| --- | --- | --- |
| `threshold` | 0.2 | level at or above which the brain leaves routine |
| `decay` | 0.9 | share of the level one moment hands to the next, in `[0, 1)` |
| `tolerance` | 2.0 | errors within this many usual errors are no surprise; large removes surprise |
| `floor` | 0.1 | nor are errors within this many reward scales |
| `fast` | 0.05 | rate of the recent reward; at `slow` it removes the long-run want |
| `slow` | 0.005 | rate of the long-run reward, the usual error and the scale |
| `heat` | 2.0 | temperature gain per unit of want on the heated slot; 0 removes it |
| `youth` | 100 | first moments of a life spent aroused |
| `value_surprise` | 1.0 | weight of surprise at the critic's forecast |
| `record_surprise` | 0.0 | weight of surprise at the action's record; on the [odour nursery](../benchmarks/reversal/README.md#how-the-operating-point-and-the-founders-were-selected) it wakes lives sooner and leaves more of them searching too briefly |
| `need` | 0.0 | reward per moment the body requires; 0 removes the need's want |
| `need_scale` | 0.0 | the need as a share of the reward scale, `need_scale * scale`, after youth; 0 removes it |

`brain.last_arousal` snapshots the latest moment: `mode`, `level`, `error`,
`record_error`, `surprise`, `want`, `temperature`, per-slot `temperatures`,
`heated_slot`, `learned`, `recorded`, `sweeps` and `learning_sweeps`; it is `None`
after construction, reset or load. `brain.arousal` counts `moments` and `sweeps` per
mode, `learning_sweeps` and `age` ([class methods](api.md#arousal-cadencearousal)).
An action that `step` or `act` sampled is adopted, so a bootstrapped brain continues
through `live` without a reset. Under `memory_choice=True` the greedy choice includes
the [episodic vote](episodic.md#one-choice-for-acting-teaching-and-imagining).

At `need=0` an adult whose life always paid poorly is not roused; bootstrap it with
`step` or a longer `youth`. In the nursery the associative memory carries the
adaptation, and a minority of lives never find the moved reward. In an unchanging
world, reward noise rouses the founders for a small share of moments. A routine moment
pays one full settle.

`need_scale` is opt-in: at the founder 0 every brain, checkpoint and composed life is as
without it, and a checkpoint carries the gene (`cadence-arousal/3`) only above zero. It is
outside `ArousalConfig.space()`, so seeded mutations at the founder draw as before; add it to
a space to select on it. A positive `need_scale` rouses an adult whose income falls below
that share of its learned reward spread, whatever the unit of its rewards, and leaves a life
that never paid at all calm (its scale is zero). At `need_scale=0.1`, with the actor step at
the composed founder, against the founder on fresh seeds the composed brain kept every
chamber of the preservation panel: in the [key-door chamber](../benchmarks/keydoor/README.md)
it acquired rule A on 10 of 16 lives against 8 of 16, with fed rule-ends calm on 20 of 21
and unfed ones aroused (late aroused share .10 at fed rule-ends against .84 at unfed ones,
where the founder is calm at both, .01 and .00, with 17 of 17 fed rule-ends calm and 16 of
16 lives calm in every rule); in the [prerequisite chain](../benchmarks/chain/README.md) 3 of
12 lives collapsed onto one action against 12 of 12 and 7 of 12 learned the chain against 1
of 12; the [odour nursery](../benchmarks/reversal/README.md) passed 8 of 12 lives in both;
and a two-armed bandit with reversal won 30 of 30 blocks in both. The cost follows from the
law: in a world whose income stays below the share, the want stays at the unmet share and
the level settles above the threshold, so the brain stays aroused and pays learning sweeps
at every moment. With pay at the end of the chain, 5 of 6 lives stayed aroused for .60 to
.67 of the life on an income of .017 to .022 per moment, the sixth, earning .15 per moment,
for .06, and the founder's 6 lives, all collapsed, for .02. Through `step` the gene is
inert.

<a id="observations-actions-and-reward"></a>

## Explicit learning with `step`

`step(observations, *, reward=None, done=None, teacher=None, salience=None,
bootstrap=None)` learns from the preceding outcome, teaches the label of the present
observation, then samples and returns the next action.

```python
stepped = Brain.compose(4, 2, modules=(16, 8), seed=7)
action = stepped.step(observation)
reward = (action == 0).astype(float)   # the body pays action 0
action = stepped.step(following, reward=reward, done=np.array([False]))
stepped.act(following, greedy=True)              # a teacher acted instead; nothing to credit
shown = stepped.lesson(following, np.array([1]))  # teach the shown action
assert stepped.last_learning["demonstrations"] == 1.0
```

| Argument | Describes | Notes |
| --- | --- | --- |
| `reward` | the preceding action | omitted is a real zero-reward transition, never an unknown outcome |
| `done` | the preceding action | for an ended row pass the next episode's reset observation |
| `bootstrap` | an ended row | per-row value of a truncated episode's last observation; omitted, an ended row bootstraps from zero |
| `salience` | the preceding outcome | nonnegative per row; defaults to `abs(reward)` for consolidation |
| `teacher` | the present observation | changes graph parameters |

Call `step` once per real transition. `act(observations, *, greedy=False,
temperature=None)` and `learn(reward, done, next_observations, *, bootstrap=None,
salience=None)` split the moment when the body needs that timing
([by hand](brain.md#one-experience-step-by-hand)). Teach from actual labels; the
brain's own guesses can reinforce mistakes. A `teacher` label writes no associative
record, except under `memory_choice=True`, where it is recorded and zero outcomes in
the middle of an attempt write nothing ([episodic memory](episodic.md#how-it-works)).

When a teacher's action ran in place of the issued one, as at the end of the example,
replace the issued action with a greedy `act`, which learns nothing from it, and teach
the shown action with `lesson(observations, labels)`; the next `step` takes no reward.
A reward reported for the shown action would credit an action that did not run.
`lesson` reads every memory the act reads, keeps its report in `last_learning` under
`demonstration_*` keys and refuses while a sampled action awaits its outcome.

## A supervised-only stream

`step(teacher=...)` issues a reward-eligible action, so the next `step` without a
reward teaches the actor and critic a zero outcome. For demonstrations with no reward
protocol, read context with greedy actions and teach the learner on the memory-bearing
drive of `stimulus(observations, *, memory=True)`:

```python
supervised = Brain.compose(4, 2, modules=(16, 8), seed=7)
supervised.act(observation, greedy=True)   # advance context without reward eligibility
_, lesson_work = supervised.learner.step(supervised.stimulus(following), np.array([1]))
assert supervised.act(following, greedy=True).shape == (1,)
```

Direct learner calls leave `Brain.last_learning` untouched; keep their report. Finish
any outstanding real outcome before a reward-driven life switches to greedy actions.

## Qualification and refusal

Every action, wait and independent prediction must satisfy the full
potential/adaptation equations, observers included, to `learning.tolerance` (`3e-3`)
within `learning.free_steps` (1024). A hard solve may fall back to half-step
numerical damping inside the same budget (a `LearnerConfig(qualified=True)` learner
uses its `damping` instead); the final residual is checked against the original
model, and the teaching rule is unchanged. `tolerance=None` is refused (`ValueError`).
Other operations keep their own tolerances: `imagine` uses `1e-6`.

A refused answer raises `RuntimeError` before changing activity, memory, random state
or pending feedback. The retry depends on what was accepted first:

| What refused | State afterwards | Retry |
| --- | --- | --- |
| the forecast settle of `live`, before feedback | unchanged | the same call |
| the answer, after the outcome was accepted (`live`, `step`) | outcome learned and counted | `live(observations)` or `act(observations)` without the outcome |
| learning from the outcome (`learn` feedback solve) | memory writes, trace resets and actor changes rolled back; action pending | the same outcome |

`brain.pending_feedback` says whether the current action awaits its outcome, including
a routine choice; read it before resubmitting.

```python
answer_work = brain.last_settlement
assert answer_work["qualified"] and answer_work["steps"] <= answer_work["budget"]
assert answer_work["max_residual"] <= answer_work["tolerance"]
```

`last_settlement` snapshots the latest completed free-answer solve, refused ones
included: `operation` (`"act"`, `"wait"` or `"predict"`; `step` reports `act`,
`accuracy` and `fit` their last `predict` batch), `scope` (`"free_answer"`), `qualified`,
per-row `row_qualified` and `residual`, `max_residual`, `steps`, `budget`,
`tolerance`, `residual_checks`, `damping_halvings` and `stagnation_checks`.
Validation errors, `learn` and `imagine` leave it; reset and load clear it. It counts
answer work only: teaching and feedback work, refused teaching included, is in
`last_learning`, and [settlement records](api.md#record-every-settling-step) capture
every solver call. Reward eligibility has its own budget, `actor_eligibility_steps=12`;
qualified teaching is opt-in ([budgets](learning.md#qualified-teaching-budgets)).

## Private imagination

```python
phases = brain.imagine([observation, following], budget=1024, tolerance=1e-6)
assert all(phase.qualified for phase in phases)
```

`imagine(observations, *, budget=1024, tolerance=1e-6)` settles supplied
observations (each a batch of the live streams) on a private trace and reads durable
memory without writing it. The branch's efference copy and recall predecessor key
take in the act the brain would issue at each phase. A refused phase ends the branch
and is returned. Live state, parameters, records, random state and pending outcomes are
untouched. A learned model of the world's consequences is
[planning](planning.md#learn-consequences-then-act).

## Outcomes that arrive later: `wait`

When a body reports an outcome only after more frames, sense those frames with
`wait(observations)`. It settles the frame, advancing activity, working trace, heard
history and recall slots, and issues no action and takes no outcome. The awaited
action keeps its forecasts, eligibility and situation, so its outcome is credited as
an immediate one would be.

```python
creature = Brain.compose(4, 2, modules=(16,), seed=0, arousal=True)
creature.live(cues[[0]])
owner = creature.decision_id                 # the action that owns the next outcome
waited = 0
for frame in (cues[[1]], cues[[2]]):         # the reach is under way
    creature.wait(frame)
    waited += creature.last_settlement["steps"]
assert creature.pending_feedback and creature.decision_id == owner
creature.live(cues[[3]], reward=[1.0], decision_id=owner)
assert creature.decision_id == owner + 1     # the outcome was taken once
work = sum(creature.arousal.sweeps.values()) + creature.arousal.learning_sweeps + waited
```

Waits are outside arousal's counts; add their `steps` to total a stream's work. One
outcome is one temporal-difference step however many frames were waited. `decision_id`
is the arousal age of the awaiting `live` action (the first is 1), or `None`; an
outcome under another identity, or one reported twice, raises `ValueError` and changes
nothing. Give the awaited outcome an explicit `reward`, since an omitted one is a zero
outcome. There is no deadline: for an outcome that will never come, `act` replaces the
action without learning and `reset` starts a new stream. `wait` refuses without
arousal or with several streams (`ValueError`) and with no awaited action
(`RuntimeError`).

The caller decides which frames are waited and links the late outcome to its action.
Where the brain is to learn that link, as in a delayed key-door reward, report the
moments in between through `live` and leave the credit to eligibility.

## Reset and save

`reset(*, keep_memory=False, keep_arousal=False)` starts every stream afresh. It clears
activity, eligibility, any pending action, reward centering, the working trace, the
efference copy, the heard history and the episodic context, plus `last_learning`,
`last_settlement` and `last_arousal`. It keeps parameters and associative records;
recall slots survive only with `keep_memory=True` ([recall](recall.md)). Arousal
starts calm with its references cleared and its age kept, as after
`retune(reset_arousal=True)`, so an adult at `need=0` is roused only by surprise and
can stay calm in a world that stopped paying; its reward scale is cleared too, so a
positive `need_scale` wants nothing until outcomes enter the new stream's scale.
`keep_arousal=True` keeps the arousal state as it is: its level, its recent and long-run
reward, its usual errors, its reward scale (and with it the scaled need), their counts,
its age and its work counts carry into the new stream, as when the state saved before
the reset with `brain.arousal.to_dict()` is restored after it with `Arousal.from_dict`,
and a brain aroused when it was reset begins the new stream aroused. Where the world may
stop paying, reset with `keep_arousal=True`, set a positive `need`, or keep living
without the reset.

In a two-armed bandit with one constant observation whose paying arm moved after 300
moments (`Brain.compose(1, 2, modules=(16,), arousal=True)`, seeds 0 to 4), adults reset
by default chose the newly paying arm at none of the next 120 moments. Reset with
`keep_arousal=True`, they fell short of the income they were used to, woke and chose as
they did living on without the reset at every one of the 120 moments, the newly paying
arm at each of the last 40; uniform random chooses it at half the moments. Reset into a
stream that paid as before, they kept their arm, calm, at all 120 moments; into one
whose arm paid 2.0 in place of 1.0, they were aroused at 9 of 600 moments and at none of
the last 40 of any seed, and the adults living on without the reset at 6 of 600. Either
reset starts the neural state afresh: in the stream that paid as before, the first 6 to
14 moments after a reset settled from rest at 32 sweeps each, and the continuing stream
needed no sweep.

```python
state = brain.arousal.to_dict()
brain.reset(keep_arousal=True)                   # the arousal state carries on as it is
assert brain.arousal.to_dict() == state
brain.save("continuing-brain.npz")
resumed = Brain.load("continuing-brain.npz")
```

A save carries parameters, critic, optimizers, random state, every trace and memory,
arousal, an action awaiting feedback and what a waiting stream has sensed; load
validates them ([formats](api.md#brain-cadence)). Resume the same rows, supply the
pending outcome once and save the environment separately.

## Independent-sample controls

`predict(observations)`, `accuracy(observations, labels)` and `fit(observations,
labels, *, epochs=30, batch=32)` treat rows as independent samples: qualified cold
solves that read no trace or associative memory. `fit` resets stream state, arousal
included, so a live adult taught with `fit` starts its next stream with no income or
error references; for a brain that keeps living, save `brain.arousal.to_dict()` before
`fit` and restore it with `Arousal.from_dict` after, the state `reset(keep_arousal=True)`
keeps. `fit` returns training accuracy per epoch; a refused score keeps accepted lessons.
Use them as controls on a separate instance; greedy `act` on a loaded copy measures the
memory-aware response.

```python
control = Brain.compose(4, 2, modules=(16,), seed=1)
control.fit(cues, np.array([0, 1, 0, 1]), epochs=2)
assert control.predict(cues).shape == (4,) and 0 <= control.accuracy(cues, [0, 1, 0, 1]) <= 1
```

## Defaults and the thinking clock

| Mechanism | Default in `Brain.compose` | Advances on |
| --- | --- | --- |
| Neural activity | retained | each accepted action and each `wait` |
| Memories | working trace and associative store on, the rest opt-in | as listed in [memory](memory.md#memory-in-the-composed-brain); `wait` advances the working trace, heard history and recall slots |
| Reward plasticity and demonstrations | `step`, `learn`, `lesson`, aroused `live` | actual outcomes and supplied labels |
| Arousal | off unless `arousal=True` | each outcome `live` receives |
| Observers | none unless requested | the same neural solve |
| Imagination | explicit call | supplied hypothetical observations |

No hidden thread drives the brain; the application owns scheduling. A `step` consumes
a real transition, so never call it per UI frame; frames before a `live` outcome go to
`wait`. [Memory](memory.md) describes each store, and the bounded `Deliberator` search
in [the API](api.md) runs on a fixed model across `tick` calls.
