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

| Feature | In short | Docs |
|---|---|---|
| **Owners and seams** | Local state connected by weighted seams | [Concepts](docs/concepts.md) |
| **Settlement** | Owners repair their state until the network holds a checked equilibrium | [Quickstart](docs/quickstart.md) |
| **Regions** | Vision, memory, planning and movement settle as one brain | [Regions](docs/patterns.md#several-regions-one-equilibrium) |
| **Senses and movement** | Sensory owners drive motor owners that move a body | [Body loop](docs/patterns.md#sensor-opposing-motors-body) |
| **Short-term memory** | The percept reverberates as a fading trace | [Fading context](docs/patterns.md#fading-context) |
| **Episodic memory** | Fast seams record an event after one observation | [Memory](docs/memory.md) |
| **Long-term memory** | Seam plasticity from local free and nudged settlements | [Learning](docs/learning.md) |
| **Expectation** | Learned statistics shape what the brain expects next | [Expectations](docs/patterns.md#expectations-as-seams) |
| **Imitation** | Learning from a teacher's actions | [Learning life](docs/patterns.md#a-learning-life) |
| **Reward and valence** | Outcomes better or worse than expected change what was active | [Reward](docs/reward.md) |
| **Future simulation** | Candidate actions are played forward and their consequences compared | [Future simulation](docs/patterns.md#future-simulation) |
| **Self-reading** | The brain reads its own activity to decide when to think longer | [Self-reading](docs/patterns.md#reading-its-own-activity) |
| **Review** | A finished result is checked and its weakest part revised | [Review and revise](docs/patterns.md#review-and-revise) |
| **Rhythm and restlessness** | Adaptation produces rhythm and breaks loops | [Rhythm](docs/patterns.md#rhythm), [restlessness](docs/patterns.md#restlessness) |
| **Growth and evolution** | Wirings develop from regions and evolve by fitness | [API](docs/api.md#constitution-cadenceconstitution) |
| **Evidence** | Owner-by-owner conformance and source-bound receipts | [Receipts](docs/receipts.md) |

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
