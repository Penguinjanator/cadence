# Architectural patterns

A Cadence brain is a composition of a few operations: settlement over declared
seams, fast records, traces, local learning, and application code that owns the
body and the world. This page lists the compositions that work, with the wiring,
a minimal implementation and the check that tells you whether it did its job.
The [function map](biology.md) points from nervous-system functions to these
patterns.

| Pattern | Use it when | Public example |
|---|---|---|
| [Several regions, one equilibrium](#several-regions-one-equilibrium) | Functions must influence each other within one decision | All six websites |
| [Sensor, opposing motors, body](#sensor-opposing-motors-body) | A body moves toward a target or away from an error | Eye & arm, mouse, forager, worm |
| [Records in the loop](#records-in-the-loop) | One observation must set or revise an association | Mouse, forager, changing memory |
| [Records addressed by time](#records-addressed-by-time) | The answer is what happened n steps ago or at a position | |
| [Fading context](#fading-context) | Recent inputs matter after they disappear | |
| [Holding an item](#holding-an-item) | A report must ignite all-or-none and persist | |
| [Rhythm](#rhythm) | Output must alternate without a clock | [`half_center.py`](../examples/half_center.py) |
| [Restlessness](#restlessness) | A controller repeats choices without progress | |
| [Imagined futures](#imagined-futures) | Consequences can be simulated before acting | Connect Four |
| [Reading its own activity](#reading-its-own-activity) | Compute or abstention should depend on confidence | Connect Four |
| [Rehearse, select, revise](#rehearse-select-revise) | Several outputs are plausible and a critic can rank them | |
| [Signed feedback](#signed-feedback) | Reward or preference should change learned seams | |
| [A learning life](#a-learning-life) | A skill is taught, practiced and kept | |

The public examples are at [floatingpragma.io/cadence-examples](https://floatingpragma.io/cadence-examples/),
with code in [cadence-examples](https://github.com/muellerberndt/cadence-examples).

## Several regions, one equilibrium

Build each function as its own `Wiring` with named port sets. `couple` merges them
and adds directed bridges between ports. One `Settlement` then repairs every owner
from the previous joint state, so vision, memory and movement reach one fixed point
for the current observation. Settling regions separately and combining their
outputs does not implement this interaction.

```python
import numpy as np
import cadence as cd
from cadence.brains import couple, sensor_motor

vision = cd.Wiring.from_edges(1, pre=[], post=[], sets={"error": [0]})
brain = couple(
    {"vision": vision, "movement": sensor_motor(1)},
    [
        ("vision", 0, "movement", 0, 0.5),  # (region, owner, region, owner, weight)
        ("movement", 1, "vision", 0, -0.1),  # motor activity feeds back into vision
        ("movement", 2, "vision", 0, 0.1),
    ],
)
rule = cd.GradedRule(gain=1, slope=2, threshold=0, leak=1, dt=0.25, clamp_amplitude=1)
engine = cd.Settlement(brain, rule)
drive = np.zeros(brain.n)
drive[list(brain.sets["vision"])] = 0.6
state = engine.settle(drive, steps=400, tolerance=0)
print(engine.residual(drive, state)[0] < 1e-10, brain.sets["movement/motor"])
```

Bridges use region-local owner indices. Region sets keep their names as
`region/set`. Drives and states are concatenated in region insertion order.

**Check:** the residual of the combined wiring, a cut of each bridge (the
behavior that depends on it must change), and the task outcome. Feedback can
destabilize parts that settle alone. A small residual certifies self-consistency
at this boundary; it says nothing about uniqueness or the best action.

To add one-trial content to a trained graph, couple a cue region and a
recall region to it. Bridge each recall owner both ways with the intention owners
it concerns, leave the trained seams unchanged, and write the recall seams only
after an output is committed. The trained graph and the new record then settle
together. [`COUPLED_BRAINS.md`](https://github.com/muellerberndt/cadence-examples/blob/main/COUPLED_BRAINS.md)
lists the regions and bridges of all six websites.

## Sensor, opposing motors, body

`sensor_motor(axes)` gives each axis one sensory error owner that excites a
positive motor owner and inhibits a negative one. The body reads the rectified
difference, moves, and returns the new error as the next drive. Settlement state
carries from tick to tick.

```python
import numpy as np
import cadence as cd
from cadence.brains import sensor_motor

wiring = sensor_motor(2)
rule = cd.GradedRule(gain=1, slope=2, threshold=0, leak=1, dt=0.25, clamp_amplitude=1)
engine = cd.Settlement(wiring, rule)
sensory, motor = list(wiring.sets["sensory"]), list(wiring.sets["motor"])
position, target, state = np.zeros(2), np.array([0.5, -0.3]), None
for tick in range(200):
    drive = np.zeros(wiring.n)
    drive[sensory] = target - position  # the body's error enters the sensory ports
    state = engine.settle(drive, steps=24, state=state)
    rates = np.maximum(0, state.activation[motor]).reshape(2, 2)
    position += 0.1 * (rates[:, 0] - rates[:, 1])  # the body moves
print(np.abs(target - position).max() < 0.05)
```

Couple the sensory ports to perception regions and the motor ports to other
controllers when the body has more to do. The eye & arm feeds retinal samples and
proprioception into visual-error and joint-coordination regions that drive six
motor units for shoulder, elbow and pencil lift.

**Check:** mask one motor population and confirm its axis stops; freeze the
readback and confirm a disturbance goes uncorrected. For a learned
controller, compare cold and warm starts on changing observations, and reset
state, traces and records at episode boundaries.

## Records in the loop

A `FastSeams` store maps key owners to value owners. Before settlement, its recall
enters the value ports as drive; after the real outcome, `observe` writes the
prediction error. Recall of a key immediately after its write returns the written
value. Orthogonal keys keep their records; correlated keys interfere.

```python
import numpy as np
import cadence as cd

wiring = cd.Wiring.from_edges(
    5, pre=[2, 3], post=[4, 4], sign=[1.0, -1.0],
    sets={"cue": [0, 1], "value": [2, 3], "approach": [4]},
)
rule = cd.GradedRule(gain=1, slope=2, threshold=0, leak=1, dt=0.25, clamp_amplitude=1)
engine = cd.Settlement(wiring, rule)
nectar = cd.FastSeams(np.array(wiring.sets["cue"]), np.array(wiring.sets["value"]), rule="delta")

def approach(cue):
    drive = np.zeros((1, wiring.n))
    drive[:, list(wiring.sets["cue"])] = cue
    state = engine.settle_batch(nectar.clamp(drive), steps=80)  # read before acting
    return float(state.activation[0, wiring.sets["approach"][0]])

flower = np.array([[1.0, 0.0]])
first = approach(flower)
nectar.observe(flower, np.array([[1.0, 0.0]]))  # contact: sweet
sweet = approach(flower)
nectar.observe(flower, np.array([[0.0, 1.0]]))  # the nectar changed
print(first == 0.0, sweet > 0.5, approach(flower) < -0.5)
```

Write only observed outcomes. A prediction or an imagined result written as
evidence confirms itself. `reset(batch, rows=...)` clears ended episodes; a
checkpoint does not include the store. [Memory](memory.md) gives the update and
its interference limits.

**Check:** retention of earlier distinct keys after new writes, interference as
keys become similar, and a dictionary or exact-lookup control. The mouse stores
task cue to destination, the forager flower to nectar.

## Records addressed by time

Use clock owners as keys: one owner per beat of a period, or one per sequence
position. Each step reads the record at the current beat before writing the new
item there, so the read returns the item from one period earlier.

```python
import numpy as np
import cadence as cd

period, items = 3, np.eye(4)
memory = cd.FastSeams(np.arange(period), np.arange(period, period + 4), rule="delta")
sequence = [0, 2, 1, 3, 3, 0, 2, 1]
hits = []
for t, item in enumerate(sequence):
    beat = np.eye(period)[[t % period]]
    if t >= period:
        hits.append(int(memory.recall(beat).argmax()) == sequence[t - period])
    memory.observe(beat, items[[item]])
print(all(hits))
```

The same store with position keys holds a span: write each item at its position
during presentation, read positions in order during report. `decay` below one
gives a graded loss with length. The clock is supplied by the environment or by
a rhythm region. Recall can enter a memory range of a larger net as drive.

**Check:** accuracy against lag or span length, with a context-only control that
has no clock-keyed store.

## Fading context

A `Trace` keeps a decaying copy of a source range and adds it as drive to a target
range with one owner per source owner. `Afterglow(source="input")` traces the
input itself, which lets a net compare the current frame with the moment before.

```python
import numpy as np
import cadence as cd

wiring = cd.Wiring.from_edges(
    6, pre=[0, 2, 1, 3], post=[4, 4, 5, 5], sign=[1.0, -1.0, 1.0, -1.0],
    sets={"input": [0, 1], "afterglow": [2, 3], "output": [4, 5]},
)
engine = cd.Settlement(wiring, cd.learning_rule(dt=1.0))
glow = cd.Afterglow(wiring, decay=0.5, amplitude=2.0, focus=0.0, source="input")
glow.reset(1)
for frame in ([1.0, 0.0], [1.0, 0.0], [0.0, 1.0]):
    drive = np.zeros((1, wiring.n))
    drive[0, :2] = frame
    state = engine.settle_batch(glow.clamp(drive), steps=100, tolerance=1e-9)
    glow.update(state)  # once per real observation
    print(frame, state.activation[0, 4:].round(3))
```

Each output reads its input minus that input's afterglow, so it responds to change. Update the trace once per
real observation and reset it per episode. When replaying stored samples for
training, use the trace recorded with each sample; shuffled replay must not
advance the live trace. `focus` above zero weights owners by how much they changed.

**Check:** the task with the trace, without it, and with a warm-started state
only. A warm start retains history only while the settlement is capped or has
several attractors.

## Holding an item

For an all-or-none report that persists after its input ends, add one pair of
owners per item. The two owners of a pair excite each other, every pair inhibits
the others, and each pair is coupled both ways to its answer owner. An answer that
crosses threshold lights its pair, the pair holds the answer, and inhibition lets
one pair hold at a time.

```python
import numpy as np
import cadence as cd

answers, pairs = [0, 1], [(2, 3), (4, 5)]
pre, post, sign = [], [], []

def seam(a, b, weight):
    pre.append(a)
    post.append(b)
    sign.append(weight)

for c, (a, b) in enumerate(pairs):
    seam(a, b, 1.5)
    seam(b, a, 1.5)  # the pair excites itself
    for owner in (a, b):
        seam(answers[c], owner, 1.0)
        seam(owner, answers[c], 1.0)  # answer and pair, both ways
    for d, (e, f) in enumerate(pairs):
        if d != c:
            seam(a, e, -1.0)
            seam(b, f, -1.0)  # pairs compete

wiring = cd.Wiring.from_edges(
    6, pre=pre, post=post, sign=sign, sets={"answer": answers, "reverberation": [2, 3, 4, 5]}
)
engine = cd.Settlement(wiring, cd.learning_rule(dt=1.0))
cue = np.zeros((1, 6))
cue[0, 0] = 1.0
lit = engine.settle_batch(cue, steps=30)
held = engine.settle_batch(np.zeros((1, 6)), state=lit, steps=300)
print(held.activation[0, answers].round(2))
```

Keep these seams fixed and train the rest of a net around them:
`cd.Learner(engine, outputs, trainable_overlaps=mask, trainable_owners=owner_mask)`
with the reverberation seams and owners masked out. The held state resists the
next input, which costs accuracy on tasks that need a fresh answer every step.

**Check:** report against exposure length (the all-or-none signature is a steep
rise over a few steps), persistence after the input ends, and the same net
without the pairs.

## Rhythm

Two owners that inhibit each other settle to a fixed point. With `Adaptation`,
the active owner tires and the other takes over. Run
[`examples/half_center.py`](../examples/half_center.py). `Adaptation` in the rule
applies to every owner; for selected owners use an application-owned trace, as in
restlessness below. A body oscillation needs its mechanics and sensory feedback
as well as the rhythm.

## Restlessness

A learned controller can fall into fixed points or two-step cycles: walking back
and forth, staring at one place. Fatigue on the owners of the choices that can
loop breaks this. Keep a slow trace of their activity and subtract it from their
drive. Let its strength grow with the number of steps since the task mismatch last
shrank, and reset it per episode. Leave it off for choices where repetition is
correct.

```python
import numpy as np

class Restlessness:
    def __init__(self, owners, tau=24.0, strength=2.0, growth=0.05):
        self.owners, self.tau, self.strength, self.growth = np.asarray(owners), tau, strength, growth
        self.reset()

    def reset(self):
        self.fatigue, self.idle, self.best = 0.0, 0, np.inf

    def apply(self, drive, mismatch):
        self.idle = 0 if mismatch < self.best else self.idle + 1
        self.best = min(self.best, mismatch)
        drive = drive.copy()
        drive[..., self.owners] -= self.strength * (1 + self.growth * self.idle) * self.fatigue
        return drive

    def update(self, activation):
        active = np.maximum(0, activation[..., self.owners])
        self.fatigue = self.fatigue + (active - self.fatigue) / self.tau

restless = Restlessness(owners=[4, 5])
drive = restless.apply(np.zeros(8), mismatch=10.0)
restless.update(np.ones(8))
print(restless.apply(np.zeros(8), mismatch=10.0)[4:6].round(3))
```

**Check:** finished episodes and steps to finish with and without it, on the
controller's own states. Tune `tau`, `strength` and `growth` at deployment; the
learned seams are unchanged.

## Imagined futures

`imagine` copies the live state into isolated branches, applies a supplied
transition, and scores terminal states with a read-only evaluator. The controller
executes the first action of the best sequence. `adversarial=True` alternates
maximizing and minimizing layers. A spent node budget raises.

```python
from cadence.brains import imagine

def move(state, action):
    state.append(action)
    return state

def value(state):
    if len(state) < 2:
        return 0.0
    return (10.0 if state[1] == 0 else -10.0) if state[0] == 0 else 2.0

live = []
plan = imagine(live, lambda s: (0, 1), move, value, lambda s: len(s) == 2, depth=2, adversarial=True)
print(plan.futures[0].action, live == [])
```

Imagined rewards are predictions. Learn from the real outcome after acting, and
keep every branch's traces inside that branch. In Connect Four, the scores of
completed search depths drive seven candidate owners, a value owner and a monitor
that settle as one decision circuit; the settled candidates select the move.
[`examples/deliberation.py`](../examples/deliberation.py) adds a learned terminal
evaluator.

**Check:** the evaluator alone, fixed-depth search, and search with a wrong
transition model. The transition model supplies information a learner may not
have; count it in comparisons.

## Reading its own activity

`ActivityMonitor` is a six-owner circuit that reads how much the controller's
activity changed, how close the best options are, and how much budget is spent.
Its request owner asks for more work. It has an effect only when that request
actually gates computation.

```python
import numpy as np
from cadence.brains import ActivityMonitor

monitor = ActivityMonitor()
close = monitor.read(np.array([0.50, 0.48]), np.array([0.50, 0.48]), pressure=0.2)
spent = monitor.read(np.array([0.50, 0.48]), np.array([0.50, 0.48]), pressure=1.0)
print(close.request_more, spent.request_more)
```

A learned monitor is a second range in the same net. It reads the hidden owners
through one-way seams and drives one confidence owner. A second `Learner`, masked
to the monitor's seams and owners, nudges confidence toward one when the
classifier's own free answer was right and toward zero when it was wrong. The net
abstains when confidence rests below a validation-chosen threshold.

**Check:** reward with abstention against never abstaining, the confidence's
AUROC for predicting errors, and a frozen monitor. Compare with the softmax margin
of a conventional classifier.

## Rehearse, select, revise

For generative output, settle several candidates from the same context in separate
batch rows, each with bounded random drive on latent owners. Score the candidates
with an explicit critic and commit the best. Records are written only from the
committed output.

```python
import numpy as np
import cadence as cd

wiring = cd.layered(4, 16, 4, density=1.0, seed=0)
engine = cd.Settlement(wiring, cd.learning_rule())
hidden, output = list(wiring.sets["hidden"]), list(wiring.sets["output"])
rng = np.random.default_rng(0)

def rehearse(context, critic, candidates=6, amplitude=0.3):
    drive = np.repeat(context[None], candidates, axis=0)
    drive[:, hidden] += amplitude * rng.standard_normal((candidates, len(hidden)))
    state = engine.settle_batch(drive, steps=100, tolerance=1e-6)
    scores = np.array([critic(row) for row in state.activation[:, output]])
    return state.activation[scores.argmax(), output], scores

context = np.zeros(wiring.n)
context[list(wiring.sets["input"])] = [1.0, 0.0, 0.0, 1.0]
choice, scores = rehearse(context, critic=lambda out: -float(np.abs(out - 0.2).sum()))
print(scores.round(3))
```

To revise a draft, re-imagine its weakest segment with more candidates and accept
the replacement only when the score of the whole draft improves. A winner's
advantage over the other candidates can narrow the next exploration amplitude.

**Check:** the critic's score and an independent quality measure, against a single
unperturbed settlement and against a conventional sampler.

## Signed feedback

For reward over time, `ActorCritic` combines eligibility traces of settlement
contrasts with a critic's prediction error; see [reward](reward.md). For a signed
preference on one output, pass the valence as the nudge weight, rehearse real
data, and roll back when validation loss rises:

```python
learner.save("before.npz")
learner.step(drive, chosen, weight=valence)  # +1: more like this, -1: less
learner.step(replay_drive, replay_labels)  # rehearse real data
if validation_loss(learner) > 1.01 * baseline:
    learner = cd.Learner.load("before.npz")
```

`drive`, `chosen`, `valence`, the replay set and `validation_loss` come from the
application. Keep feedback at zero when nothing happened: a negative signal on
every quiet moment erodes seams.

**Check:** retention on held-out data after each accepted update, and the count of
rejected updates.

## A learning life

1. **Imitate.** Fit teacher demonstrations with `Learner.step(drive, labels)`.
   Choose capacity by validation fit.
2. **Practice.** Act in the environment, sampling the output softmax, and learn
   from outcomes with advantage-weighted steps or `ActorCritic`.
3. **Correct on its own states.** Run the pupil, ask the teacher what it would do
   in the states the pupil reaches, and add those demonstrations. Start episodes
   from diverse states the teacher never visits by itself.
4. **Rehearse.** Mix earlier examples into every later update.
5. **Keep.** Save parameters with `Learner.save`; traces, records and episode
   stores need their own save and reset policy.

Two learners can share one net with `trainable_overlaps` and `trainable_owners`
masks; see [task recipes](tasks.md#several-learners-in-one-net). Select checkpoints
on validation episodes and read the test episodes once.

Fuse senses in the same way: give each sense its own input set, project them into
shared owners, and train one learner. A pipeline that decides per sense and then
combines the decisions is the control.

## Every pattern

- Check the residual of the whole wiring when the claim is a joint equilibrium.
- Ablate the component: a bridge, a region, a record, a trace, a monitor.
- Keep imagined, predicted and observed values in separate state.
- State the reset contract for settlement state, traces, records and learned seams.
- Compare with the conventional solution for the task, under the same inputs and budget.
