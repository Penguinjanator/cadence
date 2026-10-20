# Use a learned brain with an actual body

An acquired population brain forecasts, the body executes, and the brain
learns from the measured outcome: the [continuing world model](../world-model.md)
on this solver. Start from a skill that passes free assessment
([teaching a skill](BRAIN_DESIGN.md)). Signatures live in the
[reference](REFERENCE.md), limits under [scope and limits](index.md#scope-and-limits).

## Predict first, execute, then learn from the outcome

The brain learns to predict a body's next position from actual evaluations;
the body equation supplies data and is never wired into the brain.

```python
from math import tanh
from cadence.experimental.equilibrium import Cortex, bootstrap

def body(position, command):
    return tanh(0.7 * position + 0.3 * command)

layout = Cortex(seed=2)
senses = layout.input("senses", shape=2)  # current position, executed command
features = layout.column("features", patches=4, inputs=senses)
prediction = layout.column("prediction", patches=1, inputs=features)
layout.output("next_position", shape=1, reads=prediction)
brain = layout.build()

grid = [(x, u) for x in (-0.5, 0.0, 0.5) for u in (-0.5, 0.0, 0.5)]
row = lambda x, u: ({"senses": [x, u]}, {"next_position": [body(x, u)]})
report = bootstrap(brain, [row(x, u) for x, u in grid],
                   checks=[row(-0.3, 0.2), row(0.3, -0.2)],
                   epochs=30, batch_size=9, max_error=0.08, seed=2)
assert report["passed"], report

position = 0.25
for command in (0.4, 0.0, -0.4):
    inputs = {"senses": [position, command]}
    proposal = brain.step(inputs)               # forecast before execution
    forecast = proposal["outputs"]["next_position"][0]
    position = body(position, command)          # the body executes
    update = brain.observe(inputs, {"next_position": [position]})
    assert proposal["accepted"] and update["accepted"], update["reason"]
```

The commands are supplied, so this acquires a body predictor and no policy.
Keep each forecast with its context, the executed command and the measured
consequence (internal patch errors change during repair and record no earlier
forecast). Record what the actuator did, since a movement can be blocked.
`step(inputs, targets=...)` or `interventions=...` conditions retained
activity for that call without learning. The [quickstart](QUICKSTART.md#3-save-it-and-use-it-live)
table lists what each call retains.

## Supply recent context explicitly

`History(size, steps=4)` is a bounded window of `steps` observations of
`size` values, fed by the caller. A brain that uses it declares an input with
`shape=history.shape` and learns from examples with the same encoding.

```python
from cadence.experimental.equilibrium import History

history = History(2, steps=3)  # position and executed command per moment
history.push([0.25, 0.4])
history.push([0.20, 0.0])
encoded = history.push([0.14, -0.4])
assert len(encoded) == history.shape[0] == 3 * (2 + 1)
```

Each frame holds its values and a presence mask, oldest to newest, so padding
(mask zero) differs from a real zero frame. `preview(values)` returns the next
encoding without consuming a frame; `reset()` clears the window at an episode
boundary. A cue that has left the window is gone.

## Learn a discrete choice from reward

Use `Reinforcement` when an executed action earns a reward instead of a
correct-output witness. It records transitions, builds estimated Q targets and
fits them through the same patch repair with `source="estimate"`. For a small
action set, pass `action_input=None` and one scalar output name per action as
`value_output`: one query answers every action. The default
(`action_input="action"`, `value_output="value"`) is a one-hot action sensor
and one value output, queried once per action
([reference](REFERENCE.md#reinforcement-discrete-reward-driven-choices)).

```python
from cadence.experimental.equilibrium import Reinforcement

choices = Cortex(seed=2)
position_sensor = choices.input("position", shape=1)
situation = choices.column("situation", patches=2, inputs=position_sensor)
values = choices.column("values", patches=2, inputs=situation)
for i, name in enumerate(("left", "right")):
    choices.output(name, shape=(), reads=values, indices=(i,))
learner = Reinforcement(choices.build(), actions=2, action_input=None,
                        value_output=("left", "right"), seed=2)

position = 0.6
decision = learner.act({"position": [position]})
executed = decision["action"]                # the body reports what it did
next_position = position + (-0.1, 0.1)[executed]
reward = abs(position) - abs(next_position)  # moving toward zero pays
feedback = learner.feedback(reward, {"position": [next_position]},
                            decision_id=decision["decision_id"],
                            executed_action=executed)
assert feedback["stored"] and feedback["accepted"]
```

One transition shows the protocol; acquisition needs repeated practice and an
assessment of free decisions over complete episodes. The one-step target is
`(1-discount)*value_scale*reward/reward_scale + discount*clip(max_a Q(next, a), ±value_scale)`,
without the future term on terminal transitions; the
[specification](SPECIFICATION.md) gives the multi-step form.

| Setting | Default | Use |
| --- | --- | --- |
| `discount` | 0.95 | In [0, 1). Zero skips future-value queries; higher values shrink the immediate term. |
| `exploration` | 0.1 | Epsilon in [0, 1]. `act(..., explore=False)` disables it for one call; exact ties break at random. |
| `reward_scale`, `value_scale` | 1.0, 0.9 | A reward beyond ±`reward_scale` raises; `value_scale` must sit below the attainable output bound. |
| `capacity`, `batch_size` | 1024, 16 | FIFO replay; each fit samples without replacement and includes the latest transition. |
| `credit_horizon` | 1 | Transitions per target, at most `capacity`; a longer return stops at a non-greedy action, gap, `reset`, terminal or the newest record. |
| `seed` | 0 | Exploration, tie-breaking and replay sampling; saved in `snapshot()`. |

Every candidate query counts as work. Replay computes all targets from the
pre-update brain and fits one batch, clamping only each executed action's
value. Nothing replays in the background: call `replay(budget=...)` between
decisions for more learning. Replay reduces forgetting without preventing it,
so measure old skills after new learning.

## Preserve execution and feedback ownership

Call `act` only when no action is pending; a refused `act` (`accepted=False`)
consumes no decision ID and changes nothing. Pass its `decision_id` and the
action the body actually executed to `feedback`; credit goes to the executed
action. Abandon a discarded or expired command with `reset()`, which marks a
credit boundary and keeps parameters, activity and replay (a fresh life needs
a new learner). Invalid feedback changes nothing. Valid feedback stores the
transition and consumes the pending action even when the fit then refuses or
raises, so `stored` does not mean learning succeeded; `replay()` tries again.
An identical retry of the latest feedback returns `duplicate=True` and learns
nothing; a changed or older acknowledgment raises.

Use `terminal=True` (with `next_inputs=None`) only when the reward horizon
ends; a collection pause is not terminal. For delayed rewards, record every
intervening transition with its actual reward, often zero.
`feedback(..., learn=False)` stores experience without touching parameters,
for budgeted `replay()` calls scheduled separately. Assess on a
`Reinforcement.from_snapshot` copy and keep one serial owner per learner. Fit
targets derived from a simulator with `observe_batch(..., source="estimate")`
on a separate candidate.

Readiness checks exercise the deployed action path; an imitation output does
not qualify a separate value output used after takeover. To protect a working
skill, train a candidate restored from the actor's snapshot, compare both on
held-out episodes and keep the actor for rollback.

## Decode discrete actions from settled scores

Decoding belongs to the application, since button scores, preferences over
legal compound actions and action values mean different things. Choose a
threshold, mean-relative or argmax decoder on separate calibration data (a
zero threshold can miss rare buttons), check complete legal actions on
held-out episodes, and recheck after learning shifts the scores.

## Drives and curiosity

Energy, fatigue, contact and strain are measured by the body adapter. Supply
needs as sensors and their consequences as bounded rewards, for example
reduced energy deficit; that sets a preference without scripting a route.

```python
from cadence.experimental.equilibrium import LearningProgress

curiosity = LearningProgress(rate=0.1)
assert curiosity.update("body_prediction", 0.5) == 0.0  # new context
bonus = curiosity.update("body_prediction", 0.2)
assert 0.0 < bonus <= 1.0
```

`LearningProgress(rate=0.1, capacity=128)` keeps a smoothed error `m` per
string key (`rate` in (0, 1]; up to 4096 keys, least recently updated
evicted) and returns `max(0, m - m_new) / max(1, m)`, zero for a new key, so
error units matter. Feed it the forecast error against the observed outcome, measured before
teaching that outcome. Noisy decreases also earn a bonus, so compare with
constant-error, random-noise and bonus-disabled controls. One bounded mix is
`reward = (1-w)*external + w*bonus` with external in [-1, 1] and `w` in
[0, 1]; the weight and the keys are adapter choices.

## Keep the body responsive

`LiveController` runs one callback on a serial worker while the physics loop
submits observations and reads commands without waiting. Here the host tries
three commands on the body predictor and keeps the one predicted closest to
zero; all three queries count as decision work.

```python
from time import monotonic, sleep
from cadence.experimental.equilibrium import LiveController, slew

def decide(observation):
    candidates = []
    for command in (-0.4, 0.0, 0.4):
        result = brain.settle({"senses": [observation["position"], command]})
        if not result["qualified"]:
            return {"qualified": False}
        candidates.append((abs(result["outputs"]["next_position"][0]), command))
    return {"qualified": True, "command": (min(candidates)[1],)}

assert decide({"position": 0.25})["command"] == (-0.4,)  # check before handing over
controller = LiveController(decide, fallback=(0.0,), max_age=0.25)
try:
    controller.submit({"position": 0.25})
    command = controller.read()          # never blocks; fallback until fresh
    actuator = slew((0.0,), command, rate=2.0, dt=1/60)
    deadline = monotonic() + 2.0         # demo only: wait for one completion
    while not controller.inspect()["completed"] and monotonic() < deadline:
        sleep(0.001)
    assert controller.inspect()["qualified"] == 1
finally:
    exited = controller.close(timeout=1.0)
assert exited  # the brain is free for another owner
```

The callback returns a mapping with a boolean `qualified` and, when
qualified, a `command` as long as `fallback`; an exception or malformed result
counts in `errors` and the worker carries on. Command age runs from `submit`
and expires after `max_age` seconds (default 0.25, on `clock`, default
`time.monotonic`). Before the first qualified result and after a refusal,
error, expiry or `close`, `read` returns the fallback.

The worker owns the callback and every brain call it makes, learning
included, until `close(wait=True, timeout=1.0)` returns true; it drops
pending work and lets a running callback finish. A newer observation replaces
the one waiting (`dropped` counts these), so keep rewards and executed actions
in the serial owner or a lossless queue; with `Reinforcement.act` in the
callback, `reset` an expired command from that owner. No call cancels a
running solve and a Python thread gives no hard real-time guarantee, so
measure `last_latency_seconds` tails, `dropped`, `result_age` and
fresh-command rate apart from rendering rate.
`slew(current, target, rate=..., dt=...)` moves each coordinate toward the
target by at most `rate * dt`, with `rate` a scalar or one value per
coordinate.

## Save a whole life and run the small gates

A brain-only snapshot holds no history, replay or body state. Save every
owner at one paused boundary, after draining or persisting outcome messages:

| Owner | What to save |
| --- | --- |
| `Reinforcement` | `snapshot()`: brain, replay, pending action and RNG |
| `History`, `LearningProgress` | Their own `snapshot()` |
| Body and environment | Physical state, episode position, observation encoding, RNG, simulation clock |
| Execution adapter | Command and outcome IDs, whether the pending action executed, unprocessed consequences |

On restore, deliver saved consequences only for an action that executed,
`reset` a discarded one, and never re-execute an acknowledged action.
`LiveController` has no snapshot: close it and start a new one. Checkpoints
bind their implementation sources, so keep the package version that wrote a
life. `Brain.from_snapshot` accepts `device` and `dtype`;
`Reinforcement.from_snapshot` restores on the saved execution
([checkpoints](REFERENCE.md#checkpoints)).

```sh
python examples/equilibrium/live_learning.py --seeds 0 2 7
python examples/equilibrium/live_control.py --decisions 20 --seed 0
```

`live_learning.py` gates hidden-cue recall through `History`, retention with
old-example replay, delayed reward with reversal and saved continuation for
each of `--seeds` (default `0 2 7`). `live_control.py` learns a
one-dimensional body and chooses commands through `LiveController` by
predicted need reduction, a host rule outside the brain; `--decisions` (1 to
100, default 20) and `--seed` (default 0) set the run. All three examples on
this page exit nonzero when a gate fails.

## Qualify temporal context and delayed credit

`examples/equilibrium/temporal_credit.py` runs two separate fixtures. The
memory fixture recalls a cue across 2, 4 and 8 ticks from a `History` window,
with removed-history and reset-activity controls. The credit fixture rewards
the final transition by whether the first choice matched the case's preferred
action, with each of the two actions taking that role, and compares one-step
TD with replay against discount-zero learning and frozen parameters.

```sh
python examples/equilibrium/temporal_credit.py --seeds 2 7 --out /tmp/temporal-credit.json
```

`--seeds` takes one to eight nonnegative seeds (default `2 7`); the required
`--out` receives every case, outcome, refusal, work count and the
saved-versus-resumed comparison. The schedule and pass criteria are in the
[examples guide](../../examples/equilibrium/README.md).
