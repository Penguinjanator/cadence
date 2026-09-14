# Architectural patterns

A Cadence brain is a composition of a few operations: settling over declared
synapses, fast records, traces, local learning, and application code that owns the
body and the world. This page lists the compositions that work, with the connectome,
a minimal implementation and the check that tells you whether it did its job.
The [function map](biology.md) points from nervous-system functions to these
patterns.

| Pattern | Use it when | Public example |
|---|---|---|
| [Several regions, one equilibrium](#several-regions-one-equilibrium) | Functions must influence each other within one decision | All five examples |
| [Settle until the equations hold](#settle-until-the-equations-hold) | A result must be a checked equilibrium | |
| [Sensor, opposing motors, body](#sensor-opposing-motors-body) | A body moves toward a target or away from an error | Eye & arm, forager, worm |
| [Records in the loop](#records-in-the-loop) | One observation must set or revise an association | Forager, composer |
| [Records addressed by time](#records-addressed-by-time) | The answer is what happened n steps ago or at a position | Composer |
| [Fading context](#fading-context) | Recent inputs matter after they disappear | Composer |
| [Holding an item](#holding-an-item) | A report must ignite all-or-none and persist | |
| [Rhythm](#rhythm) | Output must alternate without a clock | [`half_center.py`](../examples/half_center.py) |
| [Restlessness](#restlessness) | A controller repeats choices without progress | |
| [Future simulation](#future-simulation) | The consequences of an action should be assessed before acting | Connect Four, composer |
| [Reading its own activity](#reading-its-own-activity) | Compute or abstention should depend on confidence | Connect Four |
| [Expectations as synapses](#expectations-as-synapses) | Statistics learned over many experiences should shape what comes next | Composer |
| [Signed feedback](#signed-feedback) | Reward or preference should change learned synapses | |
| [A learning life](#a-learning-life) | A skill is taught, practiced and kept | |

The public examples are at [floatingpragma.io/cadence-examples](https://floatingpragma.io/cadence-examples/),
with code in [cadence-examples](https://github.com/muellerberndt/cadence-examples).

## Several regions, one equilibrium

Build each function as its own `Connectome` with named port populations. `assemble`
merges them into one connectome and adds directed synapses between ports of different
regions. One `Brain` then updates every neuron from the synaptic input of the previous
joint state, so vision, memory inputs and movement seek one fixed point for the current
observation. Settling regions separately and combining their outputs does not
implement this interaction.

```python
import numpy as np
import cadence as cd
from cadence.circuits import assemble, reflex_arc

vision = cd.Connectome.from_synapses(1, pre=[], post=[], populations={"error": [0]})
connectome = assemble(
    {"vision": vision, "movement": reflex_arc(1)},
    [
        ("vision", 0, "movement", 0, 0.5),  # (region, neuron, region, neuron, weight)
        ("movement", 1, "vision", 0, -0.1),  # motor activity feeds back into vision
        ("movement", 2, "vision", 0, 0.1),
    ],
)
neuron_model = cd.NeuronModel(gain=1, slope=2, threshold=0, leak=1, dt=0.25, stimulus_amplitude=1)
brain = cd.Brain(connectome, neuron_model)
drive = np.zeros(connectome.n)
drive[list(connectome.populations["vision"])] = 0.6
state = brain.settle(drive, steps=400, tolerance=0)
print(brain.residual(drive, state)[0] < 1e-10, connectome.populations["movement/motor"])
```

Cross-region synapses use region-local neuron indices. Region populations keep
their names as `region/population`. Drives and states are concatenated in region
insertion order.

**Check:** the residual of the assembled connectome, a cut of each cross-region
synapse (the behavior that depends on it must change), and the task outcome.
Feedback can destabilize parts that settle alone. A small residual certifies
self-consistency at this boundary; it says nothing about uniqueness or the best
action.

To add one-trial content to a trained graph, assemble a cue region and a recall
region with it. Connect each recall neuron both ways to the intention neurons it
concerns, leave the trained synapses unchanged, and write the recall synapses only
after an output is committed. Drive the cue of each position while that position
is produced and release it afterwards so the output can develop. The trained graph
and the new record then settle together. [`COUPLED_BRAINS.md`](https://github.com/muellerberndt/cadence-examples/blob/main/COUPLED_BRAINS.md)
lists the regions and cross-region synapses of all five websites.

## Designed and evolved regions

A `Region` is blank, only a number of neurons, or designed, with its own circuit and
named input and output populations. A `Genome` lists regions and the projections between
them; `develop` lays every region into one connectome, keeps each designed circuit, and
draws each projection from the outputs of one region to the inputs of another. The
standard regions in `cadence.regions` are `visual_cortex`, `cortex`, `motor_cortex` and
`prefrontal_cortex`.

```python
import numpy as np
import cadence as cd
from cadence.regions import cortex, motor_cortex, visual_cortex

genome = cd.Genome(
    regions=(visual_cortex(8, 8, features=4), cortex(32), motor_cortex(3, lateral=-0.5)),
    projections=(cd.Projection("visual", "association"), cd.Projection("association", "motor")),
)
connectome = cd.develop(genome, seed=0)
p = connectome.populations
print(connectome.n, len(p["visual/input"]), len(p["visual/output"]), p["motor/actions"])
```

This prints `243 64 144 (240, 241, 242)`: 64 retinotopic input neurons, four maps of 36
feature neurons, the association cortex and three action neurons. A projection end can
also name one population, `visual/input` for instance.

`evolve` selects genomes by a fitness you supply. Mutation changes the sizes of blank
regions and the density, sign and scale of projections; designed regions keep their
circuits.

```python
import numpy as np
import cadence as cd

def halves(n, seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    x = np.where(labels[:, None] == 0, [1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1])
    return np.clip(x + 0.4 * rng.standard_normal((n, 8)), 0, 1), labels

train, held = halves(60, seed=0), halves(200, seed=1)

def fitness(connectome, seed):
    brain = cd.GenericBrain(connectome, seed=seed)
    brain.fit(*train, epochs=5)
    return brain.accuracy(*held) - 0.001 * len(connectome.populations["association"])

genome = cd.GenericBrain.genome(8, 2, hidden=4)
lineage = cd.evolve(fitness, genome, generations=4, population=4, fixed=("sensory",))
print(lineage.best.region("association").size, round(lineage.best_fitness, 2))
```

These held-out samples are validation data because selection reads them.
Score fitness on data the brain did not learn from, and pass `mapper=pool.map` to develop
and score a generation in parallel.

**Check:** the evolved genome against the starting genome and against a designed layout of
the same size, each scored on a separate test set.

## Settle until the equations hold

A step cap stops work without certifying an equilibrium. Continue settling in
chunks until the residual of every row is below a tolerance, and record whether the
budget ran out.

```python
import numpy as np
import cadence as cd

connectome = cd.layered(3, 12, 2, density=1.0, seed=0)
brain = cd.Brain(connectome, cd.learning_neuron_model())
drive = np.zeros((2, connectome.n))
drive[:, list(connectome.populations["input"])] = [[1, 0, 1], [0, 1, 0]]
result = brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-5)
print(result.steps, result.residual, result.converged)
```

Report rows above the residual tolerance as unsettled. The final chunk never exceeds
the remaining budget, and a zero budget checks the starting state. Passing `state=`
can save work within the same attracting basin; measure it on changing inputs.

## A generic brain

`GenericBrain` assembles the standard regions into one connectome: a sensory region, or
a visual cortex for pictures, an association cortex, and a motor cortex with one neuron
per action and lateral inhibition. `basal_ganglia` names its `ActorCritic` helper:
a linear critic reads association activity, and reward prediction error modulates local
eligibility traces. `episodic=True` (the default) supplies `SynapticMemory` under the name
`hippocampus`, recording cue-to-action rewards and consolidating repeated or salient
observations into shared persistent synapses. These two helpers keep state outside
the connectome; they are functional analogues, not anatomical simulations. Their
readbacks and recalled drives influence the next joint solve.

For a continuously interacting agent use [`step`](continuous.md), which incorporates
the preceding outcome and chooses the next action. Demonstrations enter through
`teacher=` in that same loop. The lower-level operations below are useful for controlled
experiments and independent-sample benchmarks.

It learns from labels with `fit`:

```python
import numpy as np
import cadence as cd

def bars(n, seed):
    rng = np.random.default_rng(seed)
    pictures, labels = np.zeros((n, 6, 6)), rng.integers(0, 2, size=n)
    for k, at in enumerate(rng.integers(1, 5, size=n)):
        if labels[k] == 0:
            pictures[k, :, at] = 1.0  # a vertical bar
        else:
            pictures[k, at, :] = 1.0  # a horizontal bar
    return np.clip(pictures + 0.2 * rng.standard_normal(pictures.shape), 0, 1), labels

brain = cd.GenericBrain.build((6, 6), 2, seed=0)
pictures, labels = bars(80, seed=0)
brain.fit(pictures, labels, epochs=30, batch=20)
print(brain.accuracy(*bars(200, seed=1)))
```

The printed value is accuracy on fresh pictures. `tests/test_generic.py` also checks
visual learning on a held-out bar task; evaluate new data and multiple seeds for your design.

It learns from reward with `act` and `learn`, one row per stream:

```python
import numpy as np
import cadence as cd

rng = np.random.default_rng(0)
brain = cd.GenericBrain.build(4, 4, seed=0)

def contexts(k):
    c = rng.integers(0, 4, size=k)
    return np.eye(4)[c], c

x, c = contexts(32)
for _ in range(400):
    action = brain.act(x)
    reward = (action == c).astype(float)
    x, c = contexts(32)
    brain.learn(reward, np.ones(32, dtype=bool), x)
test, answer = contexts(400)
brain.reset()
print((brain.act(test, greedy=True) == answer).mean())
```

The printed value measures held-out greedy action accuracy after practice. A fast
hippocampal record can also bias the next choice after a single reward:

```python
import numpy as np
import cadence as cd

brain = cd.GenericBrain.build(4, 4, episodic=True, seed=0)
cue = np.eye(4)[[1]]
choice = brain.act(cue)  # explore
brain.learn([1.0], [True], cue)  # the choice was rewarded, once
brain.reset()
print(brain.act(cue, greedy=True) == choice)
```

`working_memory=True` adds a prefrontal cortex: one neuron per association neuron, driven
by a `Trace` of the association cortex, projecting back into it. The brain then carries
a cue across a delay. In this task a cue is shown, then a go signal with the cue gone,
and only the action that names the cue is rewarded:

```python
import numpy as np
import cadence as cd

brain = cd.GenericBrain.build(3, 2, working_memory=True, seed=8)
rng = np.random.default_rng(8)
cue_neuron, go = np.eye(3)[:2], np.eye(3)[[2] * 32]
for _ in range(1500):
    cue = rng.integers(0, 2, size=32)
    brain.reset()
    brain.act(cue_neuron[cue])
    brain.learn(np.zeros(32), np.zeros(32, dtype=bool), go)
    action = brain.act(go)
    brain.learn((action == cue).astype(float), np.ones(32, dtype=bool), go)
cue = rng.integers(0, 2, size=400)
brain.reset()
brain.act(cue_neuron[cue], greedy=True)
print((brain.act(np.eye(3)[[2] * 400], greedy=True) == cue).mean())
```

This is a delayed-response task with an explicit memory population. The regression
in `tests/test_generic.py` compares it with the same controller without that memory;
success is not guaranteed for every seed or task.

`brain.brain`, `brain.learner`, `brain.basal_ganglia`, `brain.hippocampus` and
`brain.working_memory` expose the parts for everything the other patterns do. `GenericBrain.genome(...)` returns the layout
as a genome to change or evolve, and `GenericBrain(connectome)` wraps any connectome with
populations `sensory` (or `visual/input`), `association` and `motor`.

**Check:** the same task with a conventional model of similar size, and the generic brain
with the hippocampus or a region removed.

## Sensor, opposing motors, body

`reflex_arc(axes)` gives each axis one sensory error neuron that excites a
positive motor neuron and inhibits a negative one. The body reads the rectified
difference, moves, and returns the new error as the next drive. The brain state
carries from tick to tick.

```python
import numpy as np
import cadence as cd
from cadence.circuits import reflex_arc

connectome = reflex_arc(2)
neuron_model = cd.NeuronModel(gain=1, slope=2, threshold=0, leak=1, dt=0.25, stimulus_amplitude=1)
brain = cd.Brain(connectome, neuron_model)
sensory, motor = list(connectome.populations["sensory"]), list(connectome.populations["motor"])
position, target, state = np.zeros(2), np.array([0.5, -0.3]), None
for tick in range(200):
    drive = np.zeros(connectome.n)
    drive[sensory] = target - position  # the body's error enters the sensory ports
    state = brain.settle(drive, steps=24, state=state)
    rates = np.maximum(0, state.activation[motor]).reshape(2, 2)
    position += 0.1 * (rates[:, 0] - rates[:, 1])  # the body moves
print(np.abs(target - position).max() < 0.05)
```

Assemble the sensory ports with perception regions and the motor ports with other
controllers when the body has more to do. The eye & arm feeds retinal samples and
proprioception into visual-error and joint-coordination regions that drive six
motor units for shoulder, elbow and pencil lift.

**Check:** mask one motor population and confirm its axis stops; freeze the
readback and confirm a disturbance goes uncorrected. For a learned
controller, compare cold and warm starts on changing observations, and reset
state, traces and records at episode boundaries.

## Records in the loop

A `FastSynapses` store maps key neurons to value neurons. Before settling, its
recall enters the value ports as drive; after the real outcome, `observe` writes
the prediction error. Recall of a key immediately after its write returns the
written value. Orthogonal keys keep their records; correlated keys interfere.

```python
import numpy as np
import cadence as cd

connectome = cd.Connectome.from_synapses(
    5, pre=[2, 3], post=[4, 4], sign=[1.0, -1.0],
    populations={"cue": [0, 1], "value": [2, 3], "approach": [4]},
)
neuron_model = cd.NeuronModel(gain=1, slope=2, threshold=0, leak=1, dt=0.25, stimulus_amplitude=1)
brain = cd.Brain(connectome, neuron_model)
nectar = cd.FastSynapses(
    np.array(connectome.populations["cue"]), np.array(connectome.populations["value"]), rule="delta"
)

def approach(cue):
    drive = np.zeros((1, connectome.n))
    drive[:, list(connectome.populations["cue"])] = cue
    state = brain.settle_batch(nectar.stimulate(drive), steps=80)  # read before acting
    return float(state.activation[0, connectome.populations["approach"][0]])

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
keys become similar, and a dictionary or exact-lookup control. The forager stores
flower to nectar.

## Records addressed by time

Use clock neurons as keys: one neuron per beat of a period, or one per sequence
position. Each step reads the record at the current beat before writing the new
item there, so the read returns the item from one period earlier.

```python
import numpy as np
import cadence as cd

period, items = 3, np.eye(4)
memory = cd.FastSynapses(np.arange(period), np.arange(period, period + 4), rule="delta")
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
a rhythm region. Recall can enter a memory range of a larger brain as drive.

**Check:** accuracy against lag or span length, with a context-only control that
has no clock-keyed store.

## Records addressed by sparse codes

A write at one key moves the read at another by their dot product, so a record addressed by
correlated cues drifts with every revision. Pass the cues through a `PatternSeparator`
first: a fixed random expansion onto a wider range, the strongest few entries kept, the
running mean of the environment's cues removed. Codes that share no winners do not interfere
at all, and exact storage is bounded by the code width rather than the cue width.

```python
import numpy as np
import cadence as cd

rng = np.random.default_rng(0)
shared = rng.standard_normal(32)
cues = 0.95 * shared + 0.3 * rng.standard_normal((64, 32))       # 64 cues, strongly correlated
background = 0.95 * shared + 0.3 * rng.standard_normal((256, 32))  # what the environment looks like
values = np.eye(8)[rng.integers(0, 8, size=64)]

sep = cd.PatternSeparator(inputs=32, expansion=1024, winners=8, seed=0, center=0.99)
sep.habituate(background)
record = cd.FastSynapses(np.arange(32), np.arange(32, 40), rule="delta", separator=sep)
for t in rng.permutation(64):
    record.observe(cues[t : t + 1], values[t : t + 1])
reads = np.concatenate([record.recall(cues[t : t + 1]) for t in range(64)])
print(float(np.mean(reads.argmax(axis=1) == values.argmax(axis=1))))  # 1.0
```

Check: the same stream through a plain delta record reads back a fraction of the cues; the
separated record reads them all. Habituate before storing; a fast running mean moves a code
between its write and its read.

## Fading context

A `Trace` keeps a decaying copy of a source range and adds it as drive to a target
range with one neuron per source neuron. `Afterglow(source="input")` traces the
input itself, which lets a brain compare the current frame with the moment before.

```python
import numpy as np
import cadence as cd

connectome = cd.Connectome.from_synapses(
    6, pre=[0, 2, 1, 3], post=[4, 4, 5, 5], sign=[1.0, -1.0, 1.0, -1.0],
    populations={"input": [0, 1], "afterglow": [2, 3], "output": [4, 5]},
)
brain = cd.Brain(connectome, cd.learning_neuron_model(dt=1.0))
glow = cd.Afterglow(connectome, decay=0.5, amplitude=2.0, focus=0.0, source="input")
glow.reset(1)
for frame in ([1.0, 0.0], [1.0, 0.0], [0.0, 1.0]):
    drive = np.zeros((1, connectome.n))
    drive[0, :2] = frame
    state = brain.settle_batch(glow.stimulate(drive), steps=100, tolerance=1e-9)
    glow.update(state)  # once per real observation
    print(frame, state.activation[0, 4:].round(3))
```

Each output reads its input minus that input's afterglow, so it responds to change.
Update the trace once per real observation and reset it per episode. When replaying
stored samples for training, use the trace recorded with each sample; shuffled
replay must not advance the live trace. `focus` above zero weights neurons by how
much they changed.

**Check:** the task with the trace, without it, and with a warm-started state
only. A warm start retains history only while settling is capped by the step
budget or the brain has several attractors.

## Holding an item

For an all-or-none report that persists after its input ends, add one pair of
neurons per item. The two neurons of a pair excite each other, every pair inhibits
the others, and each pair is connected both ways to its answer neuron. An answer
that crosses threshold lights its pair, the pair holds the answer, and inhibition
lets one pair hold at a time.

```python
import numpy as np
import cadence as cd

answers, pairs = [0, 1], [(2, 3), (4, 5)]
pre, post, sign = [], [], []

def synapse(a, b, weight):
    pre.append(a)
    post.append(b)
    sign.append(weight)

for c, (a, b) in enumerate(pairs):
    synapse(a, b, 1.5)
    synapse(b, a, 1.5)  # the pair excites itself
    for neuron in (a, b):
        synapse(answers[c], neuron, 1.0)
        synapse(neuron, answers[c], 1.0)  # answer and pair, both ways
    for d, (e, f) in enumerate(pairs):
        if d != c:
            synapse(a, e, -1.0)
            synapse(b, f, -1.0)  # pairs compete

connectome = cd.Connectome.from_synapses(
    6, pre=pre, post=post, sign=sign, populations={"answer": answers, "reverberation": [2, 3, 4, 5]}
)
brain = cd.Brain(connectome, cd.learning_neuron_model(dt=1.0))
cue = np.zeros((1, 6))
cue[0, 0] = 1.0
lit = brain.settle_batch(cue, steps=30)
held = brain.settle_batch(np.zeros((1, 6)), state=lit, steps=300)
print(held.activation[0, answers].round(2))
```

Keep these synapses fixed and train the rest of a brain around them:
`cd.Learner(brain, outputs, plastic_synapses=synapse_mask, plastic_neurons=neuron_mask)`
with the reverberation synapses and neurons masked out. The held state resists the
next input, which costs accuracy on tasks that need a fresh answer every step.

**Check:** report against exposure length (the all-or-none signature is a steep
rise over a few steps), persistence after the input ends, and the same brain
without the pairs.

## Rhythm

Two neurons that inhibit each other settle to a fixed point. With `Adaptation`,
the active neuron tires and the other takes over. Run
[`examples/half_center.py`](../examples/half_center.py). `Adaptation` in the neuron
model applies to every neuron; for selected neurons use an application-owned trace,
as in restlessness below. A body oscillation needs its mechanics and sensory
feedback as well as the rhythm.

## Restlessness

A learned controller can fall into fixed points or two-step cycles, such as walking
back and forth or staring at one place. Fatigue on the neurons of the choices that
can loop breaks these cycles. Keep a slow trace of their activity and subtract it
from their drive. Let its strength grow with the number of steps since the task
mismatch last shrank, and reset it per episode. Leave it off for choices where
repetition is correct.

```python
import numpy as np

class Restlessness:
    def __init__(self, neurons, tau=24.0, strength=2.0, growth=0.05):
        self.neurons, self.tau, self.strength, self.growth = np.asarray(neurons), tau, strength, growth
        self.reset()

    def reset(self):
        self.fatigue, self.idle, self.best = 0.0, 0, np.inf

    def apply(self, drive, mismatch):
        self.idle = 0 if mismatch < self.best else self.idle + 1
        self.best = min(self.best, mismatch)
        drive = drive.copy()
        drive[..., self.neurons] -= self.strength * (1 + self.growth * self.idle) * self.fatigue
        return drive

    def update(self, activation):
        active = np.maximum(0, activation[..., self.neurons])
        self.fatigue = self.fatigue + (active - self.fatigue) / self.tau

restless = Restlessness(neurons=[4, 5])
drive = restless.apply(np.zeros(8), mismatch=10.0)
restless.update(np.ones(8))
print(restless.apply(np.zeros(8), mismatch=10.0)[4:6].round(3))
```

**Check:** finished episodes and steps to finish with and without it, on the
controller's own states. Tune `tau`, `strength` and `growth` at deployment; the
learned synapses are unchanged.

## Future simulation

Before acting, the brain simulates where its candidate actions lead and compares
the outcomes. Every future runs in its own state: its own batch row, trace and
records. The patterns here keep imagined outcomes from writing live memory or synapses;
real outcomes supply the teaching evidence. The application must enforce that separation.

### With the brain's own predictions

When the brain predicts its next observation, it rolls futures forward itself.
Each batch row hears its own previous event, settles, samples the next event and
continues. Bounded random drive on latent neurons (detuning) makes the futures
differ. A critic scores each finished future and the controller commits the best.

```python
import numpy as np
import cadence as cd

tokens, steps, futures = 4, 6, 8
connectome = cd.layered(tokens, 16, tokens, density=1.0, seed=1)
brain = cd.Brain(connectome, cd.learning_neuron_model())
inputs, outputs = list(connectome.populations["input"]), list(connectome.populations["output"])
rng = np.random.default_rng(0)

def simulate(current, critic, detune=0.1):
    last = np.full(futures, current)
    paths = [[] for _ in range(futures)]
    for _ in range(steps):
        drive = np.zeros((futures, connectome.n))
        drive[np.arange(futures), last] = 1.0  # each future hears its own last event
        drive[:, list(connectome.populations["hidden"])] += rng.normal(0, detune, (futures, 16))
        state = brain.settle_batch(drive, steps=100, tolerance=1e-6)
        p = np.exp(state.activation[:, outputs] / 0.1)
        p /= p.sum(axis=1, keepdims=True)
        last = np.array([rng.choice(tokens, p=row) for row in p])
        for path, token in zip(paths, last):
            path.append(int(token))
    scores = np.array([critic(path) for path in paths])
    return paths[int(scores.argmax())], scores

best, scores = simulate(0, critic=lambda path: len(set(path)))  # prefer varied futures
print(best, scores)
```

The snippet uses random untrained weights and a supplied diversity score to demonstrate
branching mechanics. For useful predictions, first train on real sequences, then validate
multi-step predictions and the critic independently. Detuning samples alternatives;
it does not by itself teach creativity.

Commit the first action of the best future, or a whole segment when the output is
a sequence. The winner's advantage over the other futures is a valence: pass it
through `Valence`, and let a large advantage narrow the next detuning. Keep every
candidate and its score for inspection.

### With a supplied world model

For work between actions, use [`Deliberator`](continuous.md#defaults-and-the-thinking-clock):
it retains unfinished hypotheses across bounded ticks while actual feedback keeps its own clock.

`imagine` copies the live state into isolated branches, applies a supplied
transition, and scores terminal states with a read-only evaluator. The controller
executes the first action of the best sequence. `adversarial=True` alternates
maximizing and minimizing layers. A spent node budget raises before another transition
runs, and no partial ranking is returned. `prune=True` enables alpha-beta pruning for
adversarial search; every root action still receives its exact depth-limited score.
`clone=` can replace `deepcopy` with a smaller isolated state snapshot. It must not share
mutable branch state or live memories. This is conventional search around the neural
evaluator, not search emerging from an unwired cortex.

```python
from cadence.circuits import imagine

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

In Connect Four, the scores of completed search depths drive seven candidate
neurons, a value neuron and a monitor that settle as one decision circuit; the
settled candidates select the move. [`examples/deliberation.py`](../examples/deliberation.py)
adds a learned terminal evaluator.

### Review and revise

Review a finished draft the way the world receives it: render it and measure the
rendering. Re-simulate the weakest segment with more futures, splice the best one
into the draft, and keep the change only when the whole draft scores higher and
the rendered check holds.

**Check:** the critic alone, a single unperturbed settling run, fixed-depth search,
a wrong transition model and an independent quality measure. A supplied transition
model carries information a learner may not have; count it in comparisons.

## Reading its own activity

`ActivityMonitor` is a six-neuron circuit that reads how much the controller's
activity changed, how close the best options are, and how much budget is spent.
Its request neuron asks for more work. It has an effect only when that request
actually gates computation.

```python
import numpy as np
from cadence.circuits import ActivityMonitor

monitor = ActivityMonitor()
close = monitor.read(np.array([0.50, 0.48]), np.array([0.50, 0.48]), pressure=0.2)
spent = monitor.read(np.array([0.50, 0.48]), np.array([0.50, 0.48]), pressure=1.0)
print(close.request_more, spent.request_more)
```

A learned monitor is a second range in the same brain. It reads the hidden neurons
through one-way synapses and drives one confidence neuron. A second `Learner`,
masked to the monitor's synapses and neurons, nudges confidence toward one when the
classifier's own free-phase answer was right and toward zero when it was wrong. The
brain abstains when confidence rests below a validation-chosen threshold.

**Check:** reward with abstention against never abstaining, the confidence's
AUROC for predicting errors, and a frozen monitor. Compare with the softmax margin
of a conventional classifier.

## Expectations as synapses

Statistics learned over many experiences, such as which chord follows which, can
become synapses. Count transitions, turn each row into centered log probabilities,
and use them as fixed synapses from context-cue neurons to expectation neurons.
Connect the expectation neurons to the intention neurons, with weaker synapses back.
The current context drives its cue neuron, and the expected continuations bias the
joint equilibrium.

```python
import numpy as np
import cadence as cd
from cadence.circuits import assemble

counts = np.array([[1, 8, 1], [1, 1, 8], [8, 1, 1]], float)  # observed transitions
logp = np.log((counts + 0.5) / (counts + 0.5).sum(axis=1, keepdims=True))
weights = np.clip(0.25 * (logp - logp.mean(axis=1, keepdims=True)), -1.2, 1.2)
expectation = cd.Connectome.from_synapses(
    6, pre=np.repeat(np.arange(3), 3), post=np.tile(np.arange(3, 6), 3), sign=weights.ravel(),
    populations={"cue": [0, 1, 2], "expected": [3, 4, 5]},
)
intention = cd.Connectome.from_synapses(3, pre=[], post=[], populations={"next": [0, 1, 2]})
connectome = assemble(
    {"expectation": expectation, "intention": intention},
    [("expectation", 3 + k, "intention", k, 0.35) for k in range(3)]
    + [("intention", k, "expectation", 3 + k, 0.04) for k in range(3)],
)
brain = cd.Brain(connectome, cd.learning_neuron_model(dt=1.0))
drive = np.zeros(connectome.n)
drive[connectome.populations["expectation/cue"][0]] = 2.5  # the current state is 0
state = brain.settle(drive, steps=200, tolerance=1e-9)
print(state.activation[list(connectome.populations["intention/next"])].round(4))
```

Encode relationships relative to the current context, such as intervals or
key-relative chords, so one expectation serves every key or position. The same
tables measure surprise, the negative log probability of what happened, which a
critic or a valence can read. The tables keep no complete episode.

**Check:** cutting the synapses from the expectation neurons to the intention
neurons changes the intention neurons, and held-out statistics are retained after
every update.

## Signed feedback

For reward over time, `ActorCritic` combines eligibility traces of the contrasts
between nudged settling runs toward and away from the chosen action with a critic's
prediction error; see [reward](reward.md). For a signed preference on one output,
pass the valence as the nudge weight, rehearse real data, and roll back when
validation loss rises.

```python
learner.save("before.npz")
learner.step(drive, chosen, weight=valence)  # +1: more like this, -1: less
learner.step(replay_drive, replay_labels)  # rehearse real data
if validation_loss(learner) > 1.01 * baseline:
    learner = cd.Learner.load("before.npz")
```

`drive`, `chosen`, `valence`, the replay set and `validation_loss` come from the
application. A preference can instead teach [expectations](#expectations-as-synapses):
move the statistics touched by the rated output toward or away from what it
contained, check held-out retention, and restore the previous tables when it
drops. Exact outputs never enter that memory, and the trained synapses stay
unchanged. Keep feedback at zero when nothing happened; a negative signal on
every quiet moment erodes synapses.

**Check:** retention on held-out data after each accepted update, and the count of
rejected updates.

## A learning life

0. **Design.** Choose enough capacity, define memory lifetimes, and verify feedback paths.
1. **Imitate.** Fit teacher demonstrations with `Learner.step(drive, labels)`.
   Choose capacity by validation fit.
2. **Practice.** Act in the environment, sampling the output softmax, and learn
   from outcomes with advantage-weighted steps or `ActorCritic`.
3. **Correct on its own states.** Run the pupil, ask the teacher what it would do
   in the states the pupil reaches, and add those demonstrations. Start episodes
   from diverse states the teacher never visits by itself.
4. **Rehearse.** Mix earlier examples into every later update.
5. **Keep.** `GenericBrain.save` preserves the complete standard composition between
   decisions. `Learner.save` preserves only its learner; custom traces, records and body
   state need their own save/reset policy. See the [design guide](design.md).

Two learners can share one brain with `plastic_synapses` and `plastic_neurons`
masks; see [task recipes](tasks.md#several-learners-in-one-net). Select checkpoints
on validation episodes and read the test episodes once.

Fuse senses in the same way: give each sense its own input population, project
them into shared neurons, and train one learner. A pipeline that decides per sense
and then combines the decisions is the control.

## Every pattern

- Check the residual of the whole connectome when the claim is a joint equilibrium.
- Ablate the component, whether a cross-region synapse, a region, a record, a trace or a monitor.
- Keep imagined, predicted and observed values in separate state.
- State the reset contract for brain state, traces, records and learned synapses.
- Compare with the conventional solution for the task, under the same inputs and budget.
