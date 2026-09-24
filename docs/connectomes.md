# Brains from a connectome

A measured wiring diagram becomes one `Brain`: every neuron a unit of the graded rate model,
every synapse class an entry with its count and its sign, one global gain. This page is the
recipe as two examples ran it, the worm (302 neurons) and the fruit fly (150,802 neurons, brain
and nerve cord), with the numbers that decide each step and the limits that were measured.
The fruit fly's code is in
[cadence-examples/fly-matrix](https://github.com/muellerberndt/cadence-examples/tree/main/fly-matrix).

## 1. Custody

Pin the release files by URL and SHA-256, fetch them once, and build a fixture the checks can
rebuild from the same bytes. Record every decision of the dictionary in one place:

- which objects are neurons (the fly: proofread objects of every super class except glia,
  trachea and non-neuronal, 150,802 of 188,508);
- the synapse floor (classes at five or more synapses: 1,861,418 classes carrying 23,104,740
  synapses; the floor decides the sub-net a page can settle later);
- the sign of every class from the presynaptic transmitter (acetylcholine +1; GABA, glutamate
  and histamine -1; modulators 0), which is a declaration, since the predictions are
  per neuron and imperfect;
- what the release lacks (the fly's release holds chemical synapses only, so the giant fibre's
  electrical path to the jump muscles is absent and no fact about it can be asked).

The fixture's manifest (counts, hashes, the dictionary) is committed; the fixture itself is
rebuilt by one command and is not.

## 2. The connectome and its populations

```python
from cadence import Brain, Connectome, NeuronModel

connectome = Connectome.from_synapses(pre, post, count=count, sign=sign, n=n, populations=sets)
brain = Brain(connectome, NeuronModel(gain=0.02), backend="cpu")
```

Name every population the gates and the page will read (`populations` is a dict of name to
neuron indices): the afferents by modality and side, the motor neurons by muscle and side, the
descending neurons by type, the memory site's cells by type. Names are the interface of every
later tool; the fly's `mbon:MBON11:right` is one neuron, `orn:decaying_fruit:left` 130.

The whole brain settles at about 4 ms per step on a laptop CPU at this size; rest is an exact
fixed point of the rectified activation, and activity stayed below one percent of neurons under
every stimulus up to a gain of 0.04. Never call `dense()` on a brain of this size.

## 3. The gain by protocol

One global gain is selected by two or three training facts from the physiology literature and
held out against the rest, with shuffled wirings (the same neurons and counts, postsynaptic
endpoints permuted) each selecting their own gain ([protocols](protocols.md)):

- a fact is a row: a stimulus on a named population at a level, a readout population, a
  predicate; the fly's steering circuit passed 7 of 13 held-out facts at the selected gain
  against 4, 3 and 0 on three shuffled wirings, its instincts 7 of 14 against 2 each;
- readouts over large populations need the `sparse` predicate (a mean over 1,312 descending
  neurons cannot reach 0.5) and left-right sets the `lateralized` one (a steering set holds
  motor neurons with no haltere input, so `active` on the set's mean fails).

Write the facts with their sources before the run, and keep the shuffled wirings' own gains in
the receipt; a control at the measured wiring's gain is a weaker control.

## 4. What a rate model cannot carry

Declare it, supply it, and put the gate on record. The fly's equilibrium reflex at the wingbeat
timescale rests on the spike timing of the haltere afferents; in the closed loop the measured
cord damped roll and yaw and did not damp pitch, so the reflex is the body's own inner loop,
the way the worm's undulation is the body's. Timing-coded pathways, electrical synapses and
neuromodulation are outside the model, and a page that lets the brain override a hand-written
layer keeps the boundary visible.

## 5. One gain, many circuits

At one global gain the fly's antennal lobe is bistable: below a receptor level near 0.15 nothing
reaches the Kenyon cells, above it the lobe ignites through its cholinergic local neurons (1.2
input units per projection neuron from local neurons against 0.5 from the receptors) and 22 to
57 percent of the Kenyon cells answer, almost all of them to every odour: the codes of the fruit
and yeast odours have a cosine of 0.98 at every gain from 0.005 to 0.03, where the animal's code
is sparse and specific. Neuron adaptation did not sparsen it. Measure the code of the site you
want to learn at (`tools/odour_code.py` in the example scans gain and level and writes the
receipt) before designing a lesson on it; what survives (a graded difference on about 180 cells,
a naive preference of one output neuron) bounds what plasticity there can express. A gain per
cell class, selected by the same protocol, is the open experiment.

## 6. The sub-net a page settles

A browser settles a sub-net: breadth-first from the populations the page reads, strongest
classes first, under a budget of neurons, a number of hops and a synapse floor. Keep a closure
receipt: the sub-net's readouts against the whole brain's under the page's stimuli (the fly:
60,000 neurons, 1,193,847 classes, every readout below 1e-3; 24,000 failed on the antennal
afferents, 36,000 and 48,000 on odour). Report member-level deviations separately; an odour
lights over a thousand neurons whose populations agree while their members deviate. Export the
payload in the library's order of synapses (by receiving neuron, then sender) so the browser
engine can be held to the library by a parity test
([the engine and its guide](https://github.com/muellerberndt/cadence-examples/tree/main/engine)).

## 7. Learning on it

The rule is the same actor-critic as on any brain ([learning from reward](reward.md)); the
traps it met on a real wiring are listed there with their readings. Two decisions come first:
which neurons read the action (the fly reads approach and avoidance from two mushroom body
output neurons of known valence, one cell each) and which synapses are plastic (the seam the
animal is known to change, `plastic_synapses` on the learner). Then the assay: every action the
readout can take must have an outcome, and the outcomes the value can balance.
