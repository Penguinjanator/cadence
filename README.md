<p align="center">
  <img src="docs/assets/cadence-logo.png" alt="Cadence: a mesh of stateful neural patches, feedback loops and synaptic signals" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/) · [Cadence page](https://floatingpragma.io/cadence/) · [Live brains](https://floatingpragma.io/cadence-examples/) · [Examples repository](https://github.com/muellerberndt/cadence-examples) · [PyPI](https://pypi.org/project/cadence-net/)

**An experimental neural library for learning through local overlap repair and equilibrium detuning.**

Cadence explores an animal-inspired hypothesis: the same network that interprets
an observation should carry context, change its learned relationships through
experience, and explore possible continuations. It does not claim to reproduce
an animal or human brain.

> **Under active development.** Pin an exact commit for reproducible work.
> The revised `PatchNet` interface is in this main checkout; earlier applications
> use other library compositions and their results do not validate it automatically.

## Start with PatchNet

`PatchNet` is the common starting point for new experiments. It uses the existing
nonlinear neural dynamics and local free/nudged learning rule, with a fully
reciprocal graph by default. Its observer-like patches have bounded activity,
declared ports, local readback and feedback/repair; experiments expose their
observations, residuals and learned changes through reproducible evidence.

- **Current context lives in neural activity.** Activity continues between
  observations. Optional temporal overlap holds each solve against the previous
  free activity. Whether a particular graph retains a cue through a delay must
  be measured; a converged network can also forget its previous input.
- **Acquired relationships live in continuous synapses and biases.** Resetting
  activity leaves learned parameters intact. No external fact store or replay
  buffer is required by this interface. Interference during further learning
  remains a separate test.
- **Real observations detune the network.** Continuous targets nudge only
  declared observed ports. Each synapse changes from its endpoints' free/nudged
  activity contrast. The implementation does not construct a backward graph.
- **Convergence is checked.** Every required phase must satisfy the neural
  equations within the declared residual tolerance before learning commits.
  A capped solve is reported as unfinished. A small residual does not establish
  a unique, stable or correct answer.
- **Imagined continuations are isolated.** Branches use the same learned net
  without changing live activity, parameters or evidence bookkeeping. Branch
  isolation is implemented; useful planning and musical improvisation need
  empirical validation.
- **A complete checkpoint resumes the learner.** It includes parameters,
  optimizer state, current activity and the optional bounded source-ID window.
  Repeated IDs are suppressed within that window; IDs do not prove that two
  environmental reports are independent.

Follow the [PatchNet guide](docs/patchnet.md) for the running example, memory
semantics, continuous targets and rehearsal. The earlier `Brain`, `Learner`,
`GenericBrain`, `Records` and circuit APIs remain available for existing
applications. Their separate associative memories are optional compositions,
not required components of `PatchNet`.

The research target is fewer local mechanisms supporting acquisition, selective
forgetting, retention and correction together. Learned importance, robust
lifelong memory, autonomous specialization and animal-level capability remain
open. In particular, the revised core does not freeze each weight into a binary
state: that candidate prevented compatible learning through shared connections.

## Install

Python 3.11+, with NumPy as the only required dependency:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run these commands from this checkout. The revision is not yet a PyPI release.
On Windows activate with `.venv\Scripts\Activate.ps1`.
[Optional backends](docs/backends.md) support Numba, PyTorch and MLX.

## Examples

Four earlier applications built with other Cadence compositions, each with its page, its acceptance receipt and the
verifier that recomputes the receipt from the event logs, are in the
[examples gallery](https://github.com/muellerberndt/cadence-examples).

- **Arm** ([page](https://floatingpragma.io/cadence-examples/arm/),
  [receipt](https://github.com/muellerberndt/cadence-examples/blob/main/arm/receipt.json)):
  a two-link arm learns its own body from motor babbling, copies what a visitor draws
  through a search over the consequences it has recorded, adapts to a longer link without
  a reset and returns to its original body. The whole brain runs beside the arm.
- **World** ([page](https://floatingpragma.io/cadence-examples/world/),
  [receipt](https://github.com/muellerberndt/cadence-examples/blob/main/world/receipt.json)):
  in a 6 by 6 world seen one cell at a time, one life learns the consequences of its
  actions, remembers where it saw an object, corrects that memory when the object moves,
  holds a cue across a delay and grounds words in objects.
- **Connect Four** ([page](https://floatingpragma.io/cadence-examples/connect_four/),
  [receipt](https://github.com/muellerberndt/cadence-examples/blob/main/connect_four/receipt.json)):
  one life learns what a dropped stone does and which windows are completed lines from the
  games it plays, and searches over what it learned; a visitor plays against it while it
  keeps learning.
- **Artist** ([page](https://floatingpragma.io/cadence-examples/artist/),
  [receipt](https://github.com/muellerberndt/cadence-examples/blob/main/artist/receipt.json)):
  the arm holds a pen over a canvas, learns what its strokes leave and draws figures it has
  never seen, replanning from its own canvas after every stroke; a visitor draws a figure
  and watches it drawn.

Each learns from one stream with no replay ring: the world model is a records cortex and
the settled regions complete partial readings, carry context and hold the policy. The
gallery's comparison runners put an online MLP and an online transformer in the same
lives, with and without a replay ring; the receipts hold the numbers.

## Reference

[Experience](docs/experience.md) · [Quickstart](docs/quickstart.md) ·
[Continuous interaction](docs/continuous.md) · [Records](docs/memory.md#records) ·
[Write a cortex](docs/cortex.md) · [Compose a brain](docs/brain.md) ·
[Evolve a brain](docs/evolution.md) · [Local learning](docs/learning.md) ·
[Reward](docs/reward.md) · [API](docs/api.md) · [All docs](docs/index.md) ·
[Examples gallery](https://github.com/muellerberndt/cadence-examples)

Check equation residuals before claiming equilibrium. Measure task quality and
learning cost; local updates alone guarantee neither capability nor speed.
[Concepts and limits](docs/concepts.md). MIT licensed.
