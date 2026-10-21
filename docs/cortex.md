# Write a cortex

A cortex is a named group of neurons with declared projections to other cortices. In a
`Genome` it is a `Region`; `develop` lays every region into one connectome, where each
becomes a named population, a port. A learning head is a `Learner` that owns the synapses
between some ports and changes them from settled phases. All regions settle in one
equilibrium. [Compose a brain](brain.md) builds a ready brain from these parts; this page
wires one by hand.

## Blank and designed regions

`Region(name, size)` is a blank region: a group of neurons with no synapses of its own. The
genome's projections connect it, and `mutate` may change its size.
`Region(name, circuit=connectome, inputs=None, outputs=None)` is a designed region: it
carries its own synapses and named populations, takes its size from the circuit and keeps
that size under mutation. `inputs` names the circuit population that receives projections
and `outputs` the population that sends them; each defaults to the whole region. A region
name is nonempty and holds no `/`; a blank region refuses `inputs` and `outputs`.

```python
import numpy as np
import cadence as cd

ring = cd.Connectome.from_synapses(
    6, pre=[0, 1, 2, 3, 4, 5], post=[1, 2, 3, 4, 5, 0], sign=[0.5] * 6,
    populations={"cue": [0], "readout": [3]}, label="ring",
)
loop = cd.Region("loop", circuit=ring, inputs="cue", outputs="readout")
print(loop.designed, loop.size, loop.neurons("readout"))   # True 6 (3,)
```

`designed` is `True` for a region with a circuit. `neurons(population=None)` returns
region-local indices, of one circuit population or of every neuron. `to_dict()` gives the
name and size, plus a designed region's circuit label, digest, `inputs` and `outputs`.

## The catalogue

`cadence.regions` builds the standard cortices, engineering analogues named for their
intended functions. A learner over a slotted motor port takes the same `slots`, one softmax
choice per slot.

| Builder | Region | Populations and synapses |
|---|---|---|
| `visual_cortex(height, width, *, channels=1, features=8, field=3, stride=1, init=1.0, seed=0, name="visual")` | designed | `input`: one neuron per pixel and channel, in the order of a `(height, width, channels)` image flattened row by row. `output`: `features` maps; each neuron reads one `field` by `field` window of the input, the windows stepping by `stride`. `field` fits within the image. Efficacies start with random signs and fan-scaled magnitudes times `init` (nonnegative). |
| `cortex(size, *, lateral=0.0, name="association")` | blank; designed for a nonzero `lateral` | With a nonzero `lateral`, every ordered pair of its neurons carries a synapse of that weight; a negative weight makes the neurons compete. |
| `motor_cortex(actions, *, lateral=0.0, name="motor", slots=1)` | designed | `actions`: one neuron per action, with a synapse of weight `lateral` between every ordered pair when `lateral` is nonzero. `slots` splits the actions into separate competitions that settle together: a count that divides `actions`, or one size per slot. The lateral synapses stay within each slot. |
| `prefrontal_cortex(holds, *, name="prefrontal")` | blank | One neuron per neuron of `holds` (a `Region` or a size), for a `Trace` of the held region that drives it as a stimulus. Keep the sizes tied under evolution with `mutate(..., tied=(("association", "prefrontal"),))`, held region first. |

```python
from cadence.regions import cortex, motor_cortex, prefrontal_cortex, visual_cortex

retina = visual_cortex(8, 8, features=4)          # 64 pixels, then 4 maps of 6 by 6
association = cortex(24)
motor = motor_cortex(3, lateral=-0.5)
held = prefrontal_cortex(association)
print(retina.size, retina.inputs, retina.outputs)                  # 208 input output
print(association.designed, motor.designed, held.name, held.size)  # False True prefrontal 24
print(motor_cortex(4, lateral=-0.5, slots=2).circuit.synapses)     # 4: two pairs per slot
```

## Projections

`Projection(pre, post, density=1.0, sign=0.0, scale=1.0, count=1.0, reciprocal=True)`
declares synapses from the neurons of `pre` to the neurons of `post`. An end names a
region, meaning its `outputs` population when it sends and its `inputs` population when it
receives (every neuron of a blank region), or one population as `region/population`.

| Field | Meaning |
|---|---|
| `density` | Probability in `[0, 1]` that a pair of neurons, one from each end, carries a synapse. |
| `sign` | Mean sign in `[-1, 1]`: each synapse is excitatory with probability `(1 + sign) / 2`. |
| `scale` | Nonnegative; multiplies the magnitudes, drawn uniformly from `[0, sqrt(6 / (pre neurons + post neurons))]`. |
| `count` | Synaptic contacts per synapse, nonnegative. |
| `reciprocal` | `True` adds the reverse of every synapse with the same signed weight, so a learner ties the two into one reciprocal synapse and a nudge travels back. `False` declares a one-way projection. |

Development draws every magnitude and sign from its seed. Synapses that two projections
place on the same ordered pair of neurons merge into one, whose count is the sum and whose
sign is the count-weighted mean; a synapse from a neuron onto itself is dropped. `Genome(regions, projections, label="genome")`
checks every projection end when it is built and refuses an empty or duplicated region name list.

## Ports after development

`develop(genome, seed=0)` lays the regions out in order as contiguous populations named
after them, adds each designed region's synapses and its populations as
`region/population`, and draws every projection. The result is one `Connectome` for one
`NeuralGraph`, the same for the same seed. A port is a named population of that connectome.

```python
genome = cd.Genome(
    regions=(cd.Region("senses", 6), association, motor, cd.Region("value", 1), loop),
    projections=(
        cd.Projection("senses", "association", density=0.5, reciprocal=False),
        cd.Projection("association", "motor"),
        cd.Projection("association", "value", scale=0.5),
        cd.Projection("loop", "association", sign=1.0, reciprocal=False),
        cd.Projection("association", "loop/cue", sign=-1.0, density=0.25, reciprocal=False),
    ),
    label="two-heads",
)
connectome = cd.develop(genome, seed=0)
print(list(connectome.populations))   # eight ports, motor/actions and loop/cue among them


def between(connectome, pre, post):
    """The synapses from the neurons of port ``pre`` to the neurons of port ``post``."""
    return np.isin(connectome.pre, connectome.populations[pre]) & np.isin(
        connectome.post, connectome.populations[post]
    )


print(between(connectome, "senses", "association").sum())   # 64 of 144 pairs at density 0.5
print(between(connectome, "association", "senses").sum())   # 0: a one-way projection
print(between(connectome, "motor", "association").sum())    # 72: the reverse synapses
print((connectome.sign[between(connectome, "loop", "association")] > 0).all())  # True
```

<a id="genome-records"></a>
`genome.region(name)` returns one region. `Genome.to_dict()` is a plain record of the
regions, projections and label, the form a lineage keeps. `Genome.from_dict(record,
designed=None)` reads it back. The record holds a designed region's circuit label and digest
only, so `designed` supplies each designed region by name, and its digest, size, `inputs` and
`outputs` must match the record.

```python
record = genome.to_dict()
again = cd.Genome.from_dict(record, designed={"motor": motor, "loop": loop})
assert cd.develop(again, seed=0).digest() == connectome.digest()
```

## Which synapses a head owns

A learning head is a `Learner` with two boolean masks: `plastic_synapses`, one entry per
synapse, and `plastic_neurons`, one entry per neuron whose bias moves. Synapses and biases
outside the masks keep their values through the head's updates, decay and clipping. The
helpers below build the masks from port pairs.

A reciprocal pair shares one efficacy. The learner zeroes the steps of frozen synapses,
averages the two directions of every pair and zeroes the frozen steps again, so a synapse
whose reverse is frozen moves by half its own step. A head owns both directions of every
reciprocal projection it learns. `Learner(synapse_rate=...)` multiplies each synapse's step
before that averaging, so one head learns its projections at different rates.

```python
def head(connectome, *pairs):
    """Both directions of the synapses between each pair of ports."""
    owned = np.zeros(connectome.synapses, bool)
    for pre, post in pairs:
        owned |= between(connectome, pre, post) | between(connectome, post, pre)
    return owned


def ports(connectome, *names):
    """A neuron mask of the named ports."""
    mask = np.zeros(connectome.n, bool)
    for name in names:
        mask[list(connectome.populations[name])] = True
    return mask
```

## A learner over one head

Heads with disjoint masks share one graph. Build the first head on the `NeuralGraph` and each
further head with `Learner.sharing(first, outputs, **options)`, where `options` are the
`Learner` fields after `outputs`. The heads share one holder, so each update replaces the graph
every head reads ([two learners on one graph](learning.md#two-learners-on-one-graph)). Disjoint
masks also keep one head's decay off the other's synapses.

`Learner.step(drive, labels)` settles a free phase, the nudged phases toward the labelled
outputs and one update. The phase operations `free`, `nudged` and `update` take an explicit
target pattern, here a value for the quadratic head.

```python
brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
policy = cd.Learner(
    brain,
    connectome.populations["motor/actions"],
    plastic_synapses=head(connectome, ("association", "motor")),
    plastic_neurons=ports(connectome, "motor"),
)
value = cd.Learner.sharing(
    policy,
    connectome.populations["value"],
    config=cd.LearnerConfig(nudge="quadratic"),
    plastic_synapses=head(connectome, ("association", "value")),
    plastic_neurons=ports(connectome, "value"),
)
assert not (policy.plastic_synapses & value.plastic_synapses).any()

drive = np.zeros((2, connectome.n))
drive[:, list(connectome.populations["senses"])] = np.eye(6)[:2]
before = brain.efficacy.copy()

policy.step(drive, np.array([0, 2]))          # one action index per row
assert value.brain is policy.brain            # the value head settles the updated graph
target = np.zeros((2, connectome.n))
target[:, list(connectome.populations["value"])] = [[1.0], [0.0]]
free = value.free(drive)
plus = value.nudged(drive, free, target)
minus = value.nudged(drive, free, target, sign=-1.0)
value.update(free, plus, minus)

moved = policy.brain.efficacy != before
owned = policy.plastic_synapses | value.plastic_synapses
assert moved[owned].any() and not moved[~owned].any()
```

## Settle or record

A cortex needs settling when its answer depends on a joint state of many neurons: completing
a missing part of the reading, carrying context across a delay through a recurrent state or a
`prefrontal_cortex`, or learning a policy under late reward through `ActorCritic`
([reward](reward.md)). When the outcome follows the reading it belongs to and the reading
holds what decides it, a delta-rule store outside the connectome (`FastSynapses`,
`SynapticMemory`) is enough, read and written without a settling step and without touching
any `NeuralGraph` parameter
([a store or a settling learner](records.md#a-store-or-a-settling-learner)). A store's key
can include the activation of a port, so a store and a settled policy combine in one brain;
[Compose a brain](brain.md) builds both.
