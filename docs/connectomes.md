# Brains from a connectome

A measured wiring diagram becomes one `NeuralGraph`: every neuron a unit of the graded rate
model, every synapse class one entry with its contact count and sign, one global gain. This
page is the import recipe. Biological dynamics and behaviour need their own validation
([numerical and learning contracts](contracts.md)). The
[Connectome demo](https://github.com/muellerberndt/cadence-demos/tree/main/connectome) runs
compiled fly wiring with plastic connections in the browser.

## 1. Custody: manifest and dictionary decisions

Pin the release files by URL and SHA-256, fetch them once, and build a fixture that the checks
rebuild from the same bytes. Commit a manifest (counts, hashes, `Connectome.digest()`, the
dictionary) and rebuild the fixture itself with one command. The dictionary records every
import decision in one place:

- **Which objects are neurons.** For the fly, the proofread objects of every super class except
  glia, trachea and non-neuronal: 150,802 of 188,508.
- **The synapse floor** (`min_count`, section 2), and the seams it must leave untouched.
- **The sign of every class**, from the presynaptic transmitter: acetylcholine +1; GABA,
  glutamate and histamine -1; modulators 0. Transmitter predictions are per neuron and
  imperfect, so the sign map is a declaration.
- **What the release lacks.** A release of chemical synapses only has no electrical paths, and
  no fact about them can be asked.

## 2. The connectome and its populations

`Connectome.from_synapses(n, pre=, post=, count=, sign=, populations=, label=, min_count=)`
drops self-synapses and every input row with fewer than `min_count` contacts (the floor applies
before merging), merges parallel rows between the same pair by summing counts (the sign becomes
the count-weighted mean), and sorts synapses by receiving neuron, then sender. `count` defaults
to 1, `sign` to +1, `label` to `"connectome"`, `min_count` to 0. It refuses indices outside
`[0, n)`, negative counts (inhibition goes in `sign`), non-finite values and a negative or
non-finite `min_count`. `populations` maps a name to neuron indices (stored sorted and
deduplicated). Names are the interface of every later tool (protocols, `seam_report`,
`calibrate_bias`), so name every set a gate or a page will read: afferents by modality and
side, motor neurons by muscle and side, the memory site's cells by type.

The floor removes noise and also removes distributed memories, which are many weak synapses.
`seam_report(connectome, pre, post)` says what custody left of a seam before a lesson is
designed on it, and the seam a lesson will move is kept at every count:

```python
import numpy as np
from cadence import Connectome, NeuralGraph, NeuronModel, seam_report

rng = np.random.default_rng(0)
groups = {"orn:fruit": range(0, 20), "orn:yeast": range(20, 40), "ln": range(40, 70),
          "pn:fruit": range(70, 90), "pn:yeast": range(90, 110), "kc": range(110, 290),
          "mbon": range(290, 292)}

def wire(a, b, p, mean):  # random classes from population a to b, geometric counts
    a, b = np.array(groups[a]), np.array(groups[b])
    i, j = np.nonzero(rng.random((len(a), len(b))) < p)
    return a[i], b[j], rng.geometric(1 / mean, len(i))

parts = [wire("orn:fruit", "pn:fruit", 0.4, 6), wire("orn:yeast", "pn:yeast", 0.4, 6),
         wire("orn:fruit", "ln", 0.3, 4), wire("orn:yeast", "ln", 0.3, 4),
         wire("ln", "ln", 0.5, 6), wire("ln", "pn:fruit", 0.5, 4), wire("ln", "pn:yeast", 0.5, 4),
         wire("pn:fruit", "kc", 0.03, 6), wire("pn:yeast", "kc", 0.03, 6),
         wire("kc", "mbon", 0.5, 1.5)]
pre, post, count = (np.concatenate(x) for x in zip(*parts))
sign = np.ones(len(pre))  # all cholinergic in this toy

floored = Connectome.from_synapses(292, pre=pre, post=post, count=count, sign=sign,
                                   populations=groups, min_count=2)
print(seam_report(floored, "kc", "mbon")["coverage_pre"])  # about 0.28: the floor thinned it

seam = np.isin(pre, groups["kc"]) & np.isin(post, groups["mbon"])
keep = (count >= 2) | seam  # the floor everywhere except the seam
C = Connectome.from_synapses(292, pre=pre[keep], post=post[keep], count=count[keep],
                             sign=sign[keep], populations=groups, label="toy-lobe")
report = seam_report(C, "kc", "mbon")
assert report["coverage_pre"] > 0.7
```

`seam_report` returns `classes`, `synapses`, `coverage_pre` (the share of senders with a class
onto some receiver), `classes_per_post` and `median_count`. Reading it, with `preflight` and
`naive_efficacy`, is described in [learning from reward](reward.md#check-before-the-first-lesson).
The default `cpu` backend keeps a large connectome sparse; never call `brain.dense()` on one.

## 3. The gain by protocol

Select the global gain (`NeuronModel(gain=)`) with `select_gain` from two or three training
facts from the physiology literature, and score the held-out rest. Each shuffled wiring (same
neurons and counts, receiving endpoints permuted) selects its own gain as the control. A mean
over a large readout cannot reach 0.5, so large readouts need the `sparse` predicate, and
left-right sets need `lateralized`. Write the facts with their sources before the run. [Protocols](protocols.md) owns `Protocol`, the predicates,
[the shuffled control](protocols.md#the-shuffled-control) and
[gain selection under a sparsity cap](protocols.md#gain-selection-under-a-sparsity-cap).

## 4. A gain per cell class

A loop of excitatory local neurons that is supercritical at the steering circuit's gain lights
every odour into the same code, and raising their threshold leaves a multiplicative loop
ignited. `NeuralGraph(log_gain=)` holds one log gain per neuron on its
outgoing synapses (weight = `gain * count * efficacy * exp(log_gain[pre])`), and `select_gain`
picks a class's value the same way as the global gain:

```python
from cadence import Protocol, select_gain
from cadence.protocol import Levels

def make(attenuation):  # the candidate: minus the log gain of the local neurons
    log_gain = np.zeros(C.n)
    log_gain[list(C.populations["ln"])] = -attenuation
    return NeuralGraph(C, NeuronModel(gain=0.05), log_gain=log_gain)

protocol = Protocol(
    stimuli={"fruit": ("orn:fruit",), "yeast": ("orn:yeast",)}, rows=[],
    training=[("fruit", "kc", "sparse"), ("yeast", "kc", "sparse"),
              ("fruit", "kc", "specific", "yeast")],
    levels=Levels(sparse_min=0.02, sparse_max=0.15, specific_max=0.25, code_level=0.05))
attenuation, table = select_gain(make, protocol, [0, 0.7, 1.4, 2.1, 3.0], sparsity_cap=0.3)
print(attenuation, [(row["gain"], row["facts_passed"]) for row in table])
assert attenuation == 0.7 and not table[0]["admissible"]
brain = make(attenuation)
```

`select_gain` keeps the candidates that are admissible (at most `sparsity_cap` of all neurons
active under every training stimulus; the default cap is 0.05 and `None` turns it off), takes
the one that passes the most training facts, and breaks ties by the smallest value. It raises
`ValueError` on an empty grid or when no candidate is admissible. Here the unattenuated lobe
activates 0.32 of all neurons, over the cap, and shares 0.8 of its Kenyon cell code between the
odours; 0.7 and 1.4 pass all three facts and 0.7 is chosen. Select on the whole brain, because a
sub-net's own stimulated receptors can exceed the cap, and declare the value in the dictionary
next to the global gain. Measure the code at the learning site before designing a lesson: a code shared by every stimulus carries nothing
a lesson can attach to.

## 5. What a rate model cannot carry

Timing-coded pathways, electrical synapses, neuromodulation, hormones and peptides are outside
the model, and transmitter signs are a fast-interaction map only. Declare such a pathway, supply
it from outside (a hand-written layer or the body's own loop), and keep the boundary visible
on any page that lets the brain override it. The fly's wingbeat-timescale equilibrium reflex
rests on haltere spike timing, so a closed loop supplies it.

## 6. Graph coverage in a browser

Export the full graph the dictionary selected. A reduced graph is an approximation that needs
its own evidence: equal population means do not give equal member-level behaviour, so compare
individual activities, readouts and task outcomes under the same stimuli. Export synapses in
`Connectome`'s order (receiving neuron, then sender) so a member-level parity test can hold a
browser engine to the library.

## 7. Before the first lesson

The learning rule is the same actor-critic as on any brain ([learning from reward](reward.md)).
Decide which neurons read the action and which synapses are plastic (the seam the animal is known
to change, passed as the learner's `plastic_synapses` mask). Two operating points are not in the connectome, and both are declared: the readout's
threshold (a readout cell on a rail gives a nudge little to act on;
[`calibrate_bias`](learning.md#calibrating-the-operating-point) searches its bias into
`NeuralGraph(bias=)`), and the seam's starting strength (`naive_efficacy` gives plastic classes
equal magnitude into `NeuralGraph(efficacy=)`, a controlled initialization that changes the
measured relative strengths and is reported as such). Run `preflight` before the first decision
([check before the first lesson](reward.md#check-before-the-first-lesson)); an empty warning list
does not certify equilibrium or useful acquisition. The assay then gives every action an
outcome, one decision per episode credited to that decision.
