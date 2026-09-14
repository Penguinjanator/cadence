<p align="center">
  <img src="docs/assets/patchnet.svg" alt="Owners hold state and exchange messages over declared seams" width="100%">
</p>

# Cadence

**Stateful distributed networks that read, repair and learn locally.**

Cadence builds virtual brains, from a worm's circuit toward animal and human
brains. A brain is a network of owners. Each owner holds local state, reads its
neighbours through weighted seams and repairs the difference. Regions for seeing,
remembering, deciding and moving settle together into one equilibrium, the body
acts on it, and the changed world starts the next settlement.

## Install

Use Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@main"
```

The package is `cadence-net`, the import is `cadence`, and NumPy is the only
requirement. On Windows activate with `.venv\Scripts\Activate.ps1`.
[Backends](docs/backends.md) add Numba, PyTorch (CUDA, MPS) and MLX.

## Remember, then correct

```python
import numpy as np
import cadence as cd

memory = cd.FastSeams(np.arange(3), np.arange(3, 5), rule="delta")
key = np.array([[0., 1., 0.]])
memory.observe(key, np.array([[1., 0.]]))
print(memory.recall(key))  # [[1. 0.]]
memory.observe(key, np.array([[0., 1.]]))
print(memory.recall(key))  # [[0. 1.]]
```

One observation writes the record, and a changed observation corrects it. The
write stores only the prediction error, so a correct prediction changes nothing.

## What a Cadence brain has

| Feature | How it works | Docs |
|---|---|---|
| **Owners and seams** | An owner holds a potential and publishes an activation. Seams carry weighted activity between owners. | [Concepts](docs/concepts.md) |
| **Settlement** | Every owner repairs itself from its inbox until the network reaches a fixed point. Settlement continues until `residual` confirms the equations hold. | [Quickstart](docs/quickstart.md), [checked settlement](docs/patterns.md#settle-until-the-equations-hold) |
| **One brain from many regions** | `couple` joins vision, memory, planning and motor regions into one settlement. | [Regions](docs/patterns.md#several-regions-one-equilibrium) |
| **Senses and movement** | Encoders drive sensory owners. Opposing motor pairs move a body, and the body's new state is the next input. | [Body loop](docs/patterns.md#sensor-opposing-motors-body) |
| **Short-term memory** | The percept reverberates. `Echo` and `Afterglow` feed a fading trace of the last moments into the next settlement, brightest where something changed. A self-exciting owner pair holds an item after its input is gone. | [Fading context](docs/patterns.md#fading-context), [holding an item](docs/patterns.md#holding-an-item) |
| **Episodic memory** | Fast seams store what followed a cue after one observation and correct the record when the world changes. Clock owners address records by time or position. | [Memory](docs/memory.md), [time](docs/patterns.md#records-addressed-by-time) |
| **Long-term memory** | Plasticity. A seam strengthens or weakens from the difference between a free and a nudged settlement at its two ends. There is no backward pass. | [Learning](docs/learning.md) |
| **Expectation and surprise** | Statistics learned over many experiences become seams that bias what the brain expects next. Surprise measures how unexpected an outcome was. | [Expectations](docs/patterns.md#expectations-as-seams) |
| **Imitation** | A teacher's action nudges the output owners, and the seams learn the response. | [Learning life](docs/patterns.md#a-learning-life) |
| **Reward and valence** | Eligibility traces keep which seams were active. The prediction error, better or worse than expected, writes through them. A winning future's advantage narrows exploration, and a signed preference teaches learned relationships with a retention guard. | [Reward](docs/reward.md), [signed feedback](docs/patterns.md#signed-feedback) |
| **Future simulation** | Before acting, the brain rolls candidate futures forward through its own predictions or a world model, and a critic compares their consequences. Imagined outcomes never write memory; only real outcomes teach. | [Future simulation](docs/patterns.md#future-simulation) |
| **Self-reading** | Monitor regions read the brain's own activity, request more thought when options are close, and learn when the brain tends to be wrong. | [Self-reading](docs/patterns.md#reading-its-own-activity) |
| **Rhythm and restlessness** | Adaptation turns mutual inhibition into a rhythm. Fatigue on repeated choices moves a stuck controller out of a loop. | [Rhythm](docs/patterns.md#rhythm), [restlessness](docs/patterns.md#restlessness) |
| **Review and revision** | Detuned futures vary the output. A finished draft is rendered and measured, and its weakest part is re-simulated and kept only when the whole improves. | [Review and revise](docs/patterns.md#review-and-revise) |
| **Growth and evolution** | `grow` develops a wiring from regions and projections; `evolve` selects wirings by fitness. | [API](docs/api.md#constitution-cadenceconstitution) |
| **Evidence** | `conformance` checks an engine owner by owner. `Receipt` binds results to their sources. | [Receipts](docs/receipts.md) |

The [function map](docs/biology.md) lists nervous-system functions with their
Cadence wiring, and [patterns](docs/patterns.md) shows how to build each one.

## Examples

Six brains run in the browser with nothing to install. Each shows its live circuit
beside the body.

| Example | What it does | Circuit |
|---|---|---|
| **Eye & arm**<br>[Web demo](https://floatingpragma.io/cadence-examples/eye-arm/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/eye-arm) | A retina reads your drawing, and shoulder, elbow and pencil motors copy it. Push a joint or disable a motor population. | 593 owners, 28 seams |
| **Teachable mouse**<br>[Web demo](https://floatingpragma.io/cadence-examples/mouse/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/mouse) | Teach it a new destination, revise the lesson, and carry it into a new maze. | 265 owners, 285 seams, 32 learned |
| **C. elegans habitat**<br>[Web demo](https://floatingpragma.io/cadence-examples/worm/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/worm) | The worm's chemical connectome follows food cues around walls you draw. Stimulate or lesion its neurons. | 309 owners, 4,108 seams |
| **Fly-inspired forager**<br>[Web demo](https://floatingpragma.io/cadence-examples/fly/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/fly) | Learns nectar values on contact and changes its preferences when you change the flowers. | 18 owners, 44 seams, 32 learned |
| **Changing memory**<br>[Web demo](https://floatingpragma.io/cadence-examples/memory/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/memory) | One write stores an association and a later write replaces it. Compare with an MLP on the same stream. | 12 owners, 32 seams, 32 learned |
| **Connect Four**<br>[Web demo](https://floatingpragma.io/cadence-examples/connect-four/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/connect-four) | Imagines replies before moving, and a self-monitor asks for deeper search when the choice is close. | 19 owners, 39 seams |

Owners are software units and the counts cover the circuit only. For scale, an
adult C. elegans has 302 neurons. The
[methods](https://github.com/muellerberndt/cadence-examples/blob/main/METHODS.md)
give the supplied parts and the comparisons for each example. From a clone of
cadence-examples, `python serve.py <example>` runs one offline.

## Docs

[Quickstart](docs/quickstart.md) · [Concepts](docs/concepts.md) ·
[Patterns](docs/patterns.md) · [Function map](docs/biology.md) ·
[Memory](docs/memory.md) · [Learning](docs/learning.md) ·
[Reward](docs/reward.md) · [API](docs/api.md) · [All docs](docs/index.md)

Related methods: [equilibrium propagation](https://arxiv.org/abs/1602.05179) and
[delta-rule fast weights](https://arxiv.org/abs/2406.06484). MIT licensed.
