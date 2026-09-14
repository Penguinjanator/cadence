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

## Examples with controls

Open the [live composite brains](https://github.com/muellerberndt/cadence-examples#live-composite-brains).
All five demos run locally in your browser with bundled assets; no GPU, account,
or training run is needed to start. Each explains what is supplied, what learns,
and what its comparison measures. A shared MRI-inspired circuit view shows real
activity, slowed repair iterations, recurrent decay probes and memory writes.

| Demo | What to try | What the control tests |
|---|---|---|
| [Teachable mouse](https://github.com/muellerberndt/cadence-examples/tree/main/showcase#ready-to-run-and-watch-learning) | Start with three taught tasks, teach a fourth, and transfer it to a new maze | One-write task revision and retention; navigation against a frozen route and a BFS replanner |
| [Eye & arm](https://github.com/muellerberndt/cadence-examples/tree/main/showcase#why-cadence-fits-each-task) | Draw an outline, upload an image, and disturb a joint | Joint visual/motor settlement with pose feedback enabled versus disabled |
| [Fly-inspired forager](https://github.com/muellerberndt/cadence-examples/tree/main/showcase#ready-to-run-and-watch-learning) | Watch learning on contact, then change the nectar | Residual memory versus online MLP updates; live agents collect different experiences |
| [C. elegans circuit](https://github.com/muellerberndt/cadence-examples/tree/main/showcase#results-and-comparison-contract) | Paint food and walls, then inspect the public chemical graph | Reusing a supplied recurrent rule versus a bundled trained MLP surrogate and fixed unrolling |
| [Changing memory](https://github.com/muellerberndt/cadence-examples/tree/main/showcase#results-and-comparison-contract) | Teach and revise records, run a stream, and increase key similarity | Identical observation streams for residual memory, MLP update budgets, and dictionary lookup |

The [demo guide](https://github.com/muellerberndt/cadence-examples/tree/main/showcase)
explains the advantages and boundaries: immediate record revision, reuse of a
known circuit under interventions, and continuous body feedback. These are
specific task comparisons; classical lookup, graph solvers and feedback
controllers are relevant alternatives too. The worm uses anatomical connectivity
with imposed dynamics, not a validated simulation of the complete animal.

Launch a website from the examples checkout with `python serve.py mouse`,
`python serve.py eye-arm`, `python serve.py fly`, `python serve.py worm`, or
`python serve.py memory`. Each command opens the browser on an available local
port. The worm habitat lets you place food, draw walls and inspect the circuit;
its body, sensory adapter and contact consumption are supplied rules.

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
