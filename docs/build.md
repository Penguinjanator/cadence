# Build your own brain

Two component recipes on your own data: a supervised neural graph and a temporal patch.
For a continuing brain, start with the [world-model lifecycle](world-model.md) and the
[Brain quickstart](quickstart.md). The blocks of one section run together, in order, on
NumPy after you [install](README.md#install) the package; [orientation](orientation.md)
explains the terms.

## A supervised graph-learning control

This independent-row classifier tests custom wiring and graph plasticity. A genome
names regions and projections, `develop` creates one connectome, `NeuralGraph` settles
it with reciprocal feedback, and `Learner` updates synapses from free and nudged phases.
For continuing context, memory and recovery use [Brain.compose](brain.md).

### 1. Encode and wire

A drive has shape `(batch, n)`, one column per neuron; the features, scaled to roughly
unit range, go on the input region's neurons. Labels are integer indices into the
output group.

```python
import numpy as np
import cadence as cd
from cadence.regions import cortex, motor_cortex

FEATURES, CLASSES = 8, 3
rng = np.random.default_rng(0)
centres = rng.standard_normal((CLASSES, FEATURES)) * 1.5

def make(count):
    labels = rng.integers(CLASSES, size=count)
    return centres[labels] + 0.6 * rng.standard_normal((count, FEATURES)), labels

x_train, y_train = make(240)
x_test, y_test = make(120)
```

```python
genome = cd.Genome(
    regions=(cd.Region("features", FEATURES), cortex(24), motor_cortex(CLASSES, lateral=-0.5)),
    projections=(
        cd.Projection("features", "association", reciprocal=False),   # senses drive the cortex one way
        cd.Projection("association", "motor"),                          # reciprocal: the nudge travels back
    ),
)
connectome = cd.develop(genome, seed=0)
brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
features = list(connectome.populations["features"])
learner = cd.Learner(brain, connectome.populations["motor/actions"], cd.LearnerConfig(eta=1.0, beta=0.1, free_steps=300))

def drive(x):
    d = np.zeros((len(x), connectome.n))
    d[:, features] = x
    return d
```

Hidden neurons learn only where a nudge on the outputs reaches them, so the projection
from the cortex to the motor region is reciprocal. `lateral=-0.5` makes the actions
compete. [Write a cortex](cortex.md) has the catalogue of regions, projections and heads.

### 2. Calibrate, learn, evaluate

`calibrate` picks the sampled gain whose free output operating point is nearest the
target; the default target puts the mean of the highest output activation at 0.5, and
the default grid tries the current gain, then its powers-of-two multiples from 1/256 to
256. This example declares a small-gain `grid` for its densely connected motor region.
`learner.last_calibration` keeps every candidate and its work; the chosen gain need not
make every output responsive. Calibrate on training
inputs before scoring held-out data
([calibrating the operating point](learning.md#calibrating-the-operating-point)). Each
`step` runs a free phase and two nudged phases and moves the synapses once.

```python
learner.calibrate(drive(x_train[:32]), grid=[0.02 * 1.3**k for k in range(16)])
before = learner.accuracy(drive(x_test), y_test)
for epoch in range(3):
    order = rng.permutation(len(x_train))
    for start in range(0, len(x_train), 24):
        rows = order[start:start + 24]
        learner.step(drive(x_train[rows]), y_train[rows])
    print(epoch, "held-out accuracy", round(learner.accuracy(drive(x_test), y_test), 3))
assert learner.accuracy(drive(x_test), y_test) >= 0.95
```

### 3. Check the settling

A converged phase satisfies the neuron equations at its state, and the residual is the
check. The [certificate](certificate.md) bounds the distance to the free equilibrium
only while the largest incoming weight mass of any neuron (`row_mass`) is below
`mass_limit` and the neurons have no adaptation; a learned brain often crosses it.

```python
free = learner.free(drive(x_test[:8]))
print("largest equation residual per row", learner.brain.residual(drive(x_test[:8]), free).round(4))
cert = cd.certificate(learner.brain)
print("certified", cert.certified, "row mass", round(cert.row_mass, 2), "limit", cert.mass_limit)
```

### 4. Save and load

```python
learner.save("decision_brain.npz")
again = cd.Learner.load("decision_brain.npz")
assert again.accuracy(drive(x_test), y_test) == learner.accuracy(drive(x_test), y_test)
```

To draw it, record the settlings of a solve and hand them to a page of your own
([the brain in a browser page](brain.md#the-brain-in-a-browser-page)).

### 5. Grow it

- Reward instead of labels: `ActorCritic` over the same learner ([reward](reward.md));
  `Brain` is the ready composition with a critic, a working trace and associative
  memory ([compose a brain](brain.md#brain)).
- Selection sizes the regions: `evolve` over the genome across lives
  ([evolve a brain](evolution.md)); any hand-set constant can be a gene.
- A measured connectome: `Connectome.from_synapses`, then a protocol with a shuffled
  control ([protocols](protocols.md)).
- Speed: `NeuralGraph(..., backend="torch")` for a large blocked connectome
  ([backends](backends.md)).

## A temporal patch on continuous observations and actions

The temporal patch models a body: continuous observations and actions on paths of
shape `(batch, time, ports)`, a learned dynamics model, and planning that repairs the
action ports under it. [The temporal patch](temporal.md) owns the model;
[learn consequences, then act](planning.md#learn-consequences-then-act) is the runnable
walkthrough. The call order is the recipe for any body:

1. Declare observation and action ports, with a presence flag where an observed zero
   must differ from no observation.
2. `net.reset()` between independent episodes; `observe(inputs, measured, beta=0.01, rate=...)`
   on executed inputs and measured outcomes only.
3. `imagine(inputs, state=...)` on held-out paths; report the forecast error beside persistence.
4. `plan(inputs, goal=..., controls=..., bounds=..., state=...)`; execute the first
   action through the real body, admit what was measured and plan again.
5. Protect a response with `memory = TemporalMemory()` and `memory.protect(net, inputs)`,
   then learn through `memory.observe(net, inputs, measured)`
   ([response protection](temporal-memory.md)).

Keep the hidden width and path length modest and read
[cost and memory](temporal.md#cost-and-memory) before a long run.

## Before you trust a number

Score on data the brain never saw, split before any tuning, against a stated control.
Read every `reason` other than an update, and bind seed, version and data to a
[receipt](receipts.md). [Evaluate a brain](evaluation.md) has the rules and the [missteps](evaluation.md#common-missteps).
