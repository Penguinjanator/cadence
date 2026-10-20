# Build your own brain

Three component recipes on your own data: a record patch on a stream, a supervised
neural graph and a temporal patch. For a continuing brain, start with the
[world-model lifecycle](world-model.md) and the [Brain quickstart](quickstart.md).
The blocks of one section run together, in order, on NumPy after you
[install](README.md#install) the package; [orientation](orientation.md) explains the terms.

## A record patch on a stream

The record patch hears something at each moment of a stream and predicts an outcome.
Use it when single facts must be kept after one exposure, when a regularity should move
into the weights, and when the data will be gone before the brain has learned them.

### 1. Encode

Paths have shape `(batch, time, ports)`; batch rows are independent streams sharing
parameters. Encode a categorical input as one-hot ports and give a continuous input
unit variance. A categorical outcome is one-hot and declared in `groups`, so the slow
readout ends in a softmax per group; a continuous outcome is a plain port. Here the
tokens come from a vocabulary of eight and the outcome is a rule of the token heard.
Replace `make_streams` with a loader that returns the same shapes.

```python
import numpy as np
from cadence import RecordPatchNet

VOCAB, OUTCOMES, HIDDEN = 8, 4, 16
rng = np.random.default_rng(3)

def make_streams(count, length):
    tokens = rng.integers(VOCAB, size=(count, length))
    return np.eye(VOCAB)[tokens], np.eye(OUTCOMES)[tokens % OUTCOMES]

heard, outcome = make_streams(32, 16)             # 32 training streams of 16 moments
heard_test, outcome_test = make_streams(8, 16)    # 8 held-out streams
assert heard.shape == (32, 16, VOCAB) and outcome.shape == (32, 16, OUTCOMES)
```

### 2. Size it

```python
net = RecordPatchNet(
    VOCAB, HIDDEN, OUTCOMES, seed=1,
    cells=4096, active=16,        # the store: 4,096 cells, 16 light per reading
    record_rate=1.0,              # a reading written once takes its outcome whole
    groups=(OUTCOMES,),           # one categorical port of four outcomes
    slowest=8.0,                  # context channels from two to eight moments at birth
)
```

Starting values for a first build. The table is a subset; every knob, its default
and its cost are in [the record patch](record-patch.md).

| Knob | What it sets | Where to start |
| --- | --- | --- |
| `hidden` | context channels; the width of the slow patch | 16 to 128 |
| `cells`, `active` | the store's size and how many cells a reading touches | 4,096 and 16 for a toy; 65,536 and 32 to 64 for a corpus; enough active cells that neighbouring readings overlap |
| `record_rate` | how much of the residual a write stores | 1.0 when each reading is written once; `record_averaging=True` for a store written hundreds of times per cell |
| `slowest` | the longest initial retention timescale, in moments; the gates learn from there | a few times the longest dependency you expect |
| `groups` | categorical ports, one softmax per group | every symbol output |
| `record_width` | a sign code narrower than a wide categorical port | hundreds of columns for thousands of categories |
| `record_writes` | `"sequential"` or `"batch"` writes within one call | `"batch"` when a call holds many moments that share cells |
| `rate` (in `observe`) | the slow step; the loss is a mean over time and outputs, so the useful rate scales with their product | 1 to 10 here; thousands for a wide picture port |

### 3. Evaluate before, during and after

Score held-out streams with the records and with the slow weights alone, against the
simplest control the task admits (here the most frequent held-out outcome).

```python
def accuracy(net, heard, outcome, records=True):
    probe = net if records else RecordPatchNet.restore(net.snapshot())
    if not records:
        probe.records.tables["y"][:] = 0.0                 # the same weights, an empty store
    said = probe.imagine(heard, state=np.zeros((len(heard), HIDDEN))).output.argmax(-1)
    return float((said == outcome.argmax(-1)).mean())

chance = np.bincount(outcome_test.argmax(-1).ravel()).max() / outcome_test[..., 0].size
before = accuracy(net, heard_test, outcome_test, records=False)
```

### 4. Learn by day

With the world present, learn at a slow rate above zero; every observed moment also
writes its residual into the store. `backtrack=True` admits a step only when a
target-free replay lowers the slow loss, halving the rate up to sixteen times;
otherwise `reason` is `"no_decreasing_parameter_step"`. Reset the context before
every independent pass.

```python
for day in range(20):
    net.reset()
    seen = net.observe(heard, outcome, rate=8.0, backtrack=True)
    assert seen.updated or seen.reason == "no_decreasing_parameter_step"

with_records = accuracy(net, heard_test, outcome_test)
slow_only = accuracy(net, heard_test, outcome_test, records=False)
print(f"held-out: chance {chance:.2f}, before {before:.2f}, records {with_records:.2f}, slow weights alone {slow_only:.2f}")
assert slow_only >= 0.95 and with_records >= 0.95
```

The records score held-out streams from the first passes, because a held-out reading
touches the cells of training readings that heard the same token. The slow weights learn
the rule more slowly and then need no store. A conjunction of this token and the last
needs a second patch ([two patches in depth](record-patch.md#two-patches-in-depth)).

### 5. Keep a fact after one exposure

Hear the held-out streams once at slow rate zero. The slow weights stay put
(`reason` is `"no_step"`), and the store recalls every moment:

```python
net.reset()
first = net.observe(heard_test, outcome_test, rate=0.0)
assert first.writes == heard_test.shape[0] * heard_test.shape[1]
net.reset()
assert accuracy(net, heard_test, outcome_test) == 1.0
```

### 6. Sleep when the data are gone

When the store is the only copy, a night moves what it holds into the slow weights.
`sleep(cues, passes=1, rate=1.0, backtrack=False, dawn_passes=2)` dreams every cue
once (the store's completion of it), teaches the slow weights those fixed dreams for
`passes` passes of `observe(write=False)`, and at dawn writes them back `dawn_passes`
times so the store holds only what the weights did not take. It returns the cue count,
admitted updates, the dream loss before and after, and the dawn writes.

```python
sleeper = RecordPatchNet(VOCAB, HIDDEN, OUTCOMES, seed=1, cells=4096, active=16,
                         record_rate=1.0, groups=(OUTCOMES,), slowest=8.0)
for _ in range(2):                                     # the day: records only
    sleeper.reset()
    sleeper.observe(heard, outcome, rate=0.0)
bedtime = accuracy(sleeper, heard_test, outcome_test, records=False)
night = sleeper.sleep([heard], passes=40, rate=8.0, backtrack=True)
dawn = accuracy(sleeper, heard_test, outcome_test, records=False)
print(f"slow weights alone, held-out: bedtime {bedtime:.2f}, dawn {dawn:.2f}; {night}")
assert dawn >= 0.95
```

Dreams carry the store's errors as faithfully as its regularities, so what the store
reads at bedtime bounds the night
([acquisition in two phases](record-patch.md#acquisition-in-two-phases-records-by-day-weights-by-night)).

### 7. Save, load, read back

```python
path = net.save("stream_brain.npz")
restored = RecordPatchNet.load(path)
np.testing.assert_array_equal(
    restored.imagine(heard_test, state=np.zeros((8, HIDDEN))).output,
    net.imagine(heard_test, state=np.zeros((8, HIDDEN))).output,
)
readback = net.readback()
print(readback.updates, readback.writes, readback.record_entries, readback.parameter_revision)
```

The checkpoint holds the parameters, the record tables, the reading statistics and the
live context. `readback` is the detached state with its counters; an application may
feed it to another patch through a port, never as a teaching signal by itself.

### 8. Diagnose the store

A store always holds something, so it can fail silently. Count how often each cell is
active over the training readings and read the three numbers in
[diagnosing a record store](record-patch.md#diagnosing-a-record-store).

### 9. Grow it

- Pictures: `RecordPatchNet(port=StructuredPort(...))`, a tied local kernel at the port
  ([maps](record-patch.md#maps-a-structured-input-port)).
- Depth: `RecordPatchStack` for a conjunction of what came before.
- Several senses: `JointRecordPatches` settles patches as one equilibrium through
  declared ports ([joined by ports](record-patch.md#several-patches-joined-by-ports)).
- A belief carried under action, with imagination: [the belief patch](belief.md).

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
