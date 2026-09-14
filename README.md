<p align="center">
  <img src="docs/assets/patchnet.svg" alt="Owners hold state and exchange messages over declared seams" width="100%">
</p>

# Cadence

**Small, stateful networks that read, repair, and learn locally.**

Cadence builds observer-like software patches: bounded local state, declared input
and output ports, readback, records, and feedback that repairs a discrepancy.
A patch net connects those patches with weighted seams. Use it to explore recurrent
inference, learning by equilibrium detuning, and memory that can correct its records.

## Install

Use Python 3.11 or later and Git. Install the source for the API documented here:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@main"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
The package name is `cadence-net`; the Python import is `cadence`. Only NumPy is
required. From a checkout, use `python -m pip install -e .`.
See [backends](docs/backends.md) for optional CPU and GPU acceleration.

## Remember, then correct

Copy this into Python or save it as `remember.py` and run `python remember.py`:

```python
import numpy as np
import cadence as cd

# Three key coordinates, two value coordinates, one independent memory per batch row.
memory = cd.FastSeams(np.arange(3), np.arange(3, 5), rule="delta")
key = np.array([[0., 1., 0.]])
memory.observe(key, np.array([[1., 0.]]))
print(memory.recall(key))
memory.observe(key, np.array([[0., 1.]]))
print(memory.recall(key))
```

Expected output:

```text
[[1. 0.]]
[[0. 1.]]
```

For a unit key `k`, the write is `M += outer(k, value - k @ M)`. A correct
prediction causes no change. Orthogonal keys retain their own values; overlapping
keys can interfere. This is an established delta-rule associative memory, with
explicit ports and stream resets. It needs no recurrent solve or offline training.

## Choose the operation

| Your task | Start with | Runnable guide |
|---|---|---|
| Infer a response when parts of a circuit influence each other | `Wiring` and `Settlement` | [Quickstart: a three-owner circuit](docs/quickstart.md) |
| Store observations and revise them as values change | `FastSeams(rule="delta")` | [Memory](docs/memory.md) |
| Learn reusable responses from labelled observations | `Learner` | [Quickstart: fit two labels](docs/quickstart.md#learn-a-response) |
| Carry recent activity or learn from delayed reward | `Trace`, `ActorCritic` | [Task recipes](docs/tasks.md), [reward](docs/reward.md) |

Settlement repeatedly repairs interacting state. A learner settles freely, then
nudges its output ports toward and away from a target and updates each seam from
its endpoint changes. It retains phase states and optimizer history without a
backward computation graph. The equilibrium-gradient interpretation requires
symmetric effective recurrent weights, converged phases, and a smooth stable
equilibrium branch. [Learning](docs/learning.md) gives the equations and limits.

## Examples

**Feed a worm, teach a mouse, or watch an arm draw.** The five
[interactive websites](https://github.com/muellerberndt/cadence-examples#launch-any-demo)
run locally in your browser with bundled assets. No GPU, account or training run
is needed to start. Every demo shows its actual patch circuit beside the body on
desktop, grouped by function: activity, automatic repair cascades, transient
input-release probes and memory writes. The layout stacks on mobile.

| Main example | What it does | Cadence circuit size | Biological size reference |
|---|---|---|---|
| [C. elegans habitat](https://github.com/muellerberndt/cadence-examples/tree/main/showcase#food-and-walls-in-the-worm-habitat) | Detects local food cues, navigates around walls and consumes food patches on contact. Draw barriers, place food, erase a passage, or inspect and perturb the chemical circuit. | **297 patches · 3,604 seams**; supplied weights, no learned entries | Drawn from C. elegans anatomy; the adult hermaphrodite has **302 neurons** in its complete nervous system |
| [Teachable mouse](https://github.com/muellerberndt/cadence-examples/tree/main/showcase#ready-to-run-and-watch-learning) | Starts with cheese, home and water tasks. Teach a new destination, revise it without erasing other distinct cues, and carry the lesson into another maze. | **126 active ports / 259 allocated slots · 274 seams** in the starting maze; **32 adaptive entries** | Hundreds of software slots; C. elegans' **302-neuron** nervous system is a count reference |
| [Eye & arm](https://github.com/muellerberndt/cadence-examples/tree/main/showcase#why-cadence-fits-each-task) | Sketches built-in outlines or edges from your uploaded image. Disturb a joint and watch visual/motor feedback correct the movement. | **4 patches · 8 directed seams**; geometry-derived weights, no learned entries | Small subcircuit scale; fewer units than the **20-neuron C. elegans feeding network** |
| [Fly-inspired forager](https://github.com/muellerberndt/cadence-examples/tree/main/showcase#ready-to-run-and-watch-learning) | Visits flowers, learns their nectar value on contact and updates its preferences when you change the nectar. Move flowers to change its world. | **12 memory ports · 32 seams**; **32 adaptive entries** | Small subcircuit scale; fewer units than the **20-neuron C. elegans feeding network** |
| [Changing memory](https://github.com/muellerberndt/cadence-examples/tree/main/showcase#results-and-comparison-contract) | Learns a key/value association in one write, replaces outdated records and shows retention or interference as keys become more similar. Compare online MLP updates on the same observations. | **12 memory ports · 32 seams**; **32 adaptive entries** | Small subcircuit scale; fewer units than the **20-neuron C. elegans feeding network** |

**Read the counts:** patches/ports hold state; seams carry weighted messages;
adaptive entries are the values changed by learning. Active means unmasked,
not necessarily nonzero at that instant. These are circuit counts,
not total simulator parameters. The mouse allocates 247 spatial slots plus 12
memory ports; its starting maze masks 133 wall slots and has 242 spatial seams
plus 32 memory seams. Editing or regenerating the maze changes the active counts.
Memory seams count all matrix entries, including zeros. Body rules, environmental
fields and comparison MLPs are outside these circuit counts.

The biological references count neurons, not equivalent computing power. An adult
C. elegans hermaphrodite has 302 neurons, including 20 in its pharyngeal feeding
network ([WormAtlas](https://www.wormatlas.org/hermaphrodite/nervous/mainframe.htm)).
A software patch is not a biological neuron; the smaller examples have no claimed
whole-animal brain equivalent. The worm habitat uses supplied diffusion, heading,
body movement and contact-consumption rules, rather than validated animal
locomotion or digestion. The arm uses supplied geometry and image edge extraction.

Launch any website from the examples checkout:

```bash
python serve.py worm
python serve.py mouse
python serve.py eye-arm
python serve.py fly
python serve.py memory
```

Each command opens the browser on an available local port. The
[demo guide](https://github.com/muellerberndt/cadence-examples/tree/main/showcase)
explains why Cadence fits each task and the measured comparison: dictionary and
MLP memory controls, a circuit surrogate and tied recurrence, maze replanning,
and drawing with readback enabled or disabled. These are specific task results,
not a general efficiency theorem.

## Check your result

`Settlement.residual(drive, state)` checks the remaining fixed-point discrepancy.
`conformance` compares a trajectory with an owner-by-owner reference. A `Receipt`
can bind outcomes to sources and a caller-supplied arithmetic check. These checks
answer different questions; [the quickstart](docs/quickstart.md) shows the first
two, and [receipts](docs/receipts.md) explains reproducible comparisons.

[Quickstart](docs/quickstart.md) · [Concepts](docs/concepts.md) ·
[Biology-to-Cadence map](docs/biology.md) ·
[Learning](docs/learning.md) · [Memory](docs/memory.md) ·
[API](docs/api.md) · [All docs](docs/index.md)

Cadence grew from connectome and observer-patch experiments. Related methods include
[equilibrium propagation](https://arxiv.org/abs/1602.05179) and
[delta-rule linear transformers](https://arxiv.org/abs/2406.06484).

MIT licensed.

Compare predicted outcomes before acting with the tested
[deliberation pattern](docs/deliberation.md): independent imagined branches,
a learned evaluator, and feedback from real outcomes.
