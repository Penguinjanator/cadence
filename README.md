<p align="center">
  <img src="docs/assets/patchnet.svg" alt="Neurons hold state and exchange activity over declared synapses" width="100%">
</p>

# Cadence

**Build recurrent brains that settle together and learn locally.**

Cadence is a Python library for neurons, synapses and named functional regions.
Each neuron reads its own state and incoming activity, then relaxes toward their
combined drive. Feedback carries changes between regions as they seek a common
equilibrium. Local free/nudged contrasts teach responses; traces, fast memory and
reward learning let a controller carry experience into its next decision.

## Why use it?

- **Stateful interaction:** perception, memory inputs and motor intentions can influence
  one joint solve. Inspect the activity, intervene on a neuron, and measure the effect.
- **No backpropagation:** no backward pass, no autograd and no stored computation graph.
  Each synapse changes from the activities of its own two neurons in free and nudged
  phases. Under the [stated conditions](docs/learning.md#5-why-the-contrast-is-a-gradient)
  that local change equals a gradient step, without differentiating the network.
- **Continuous interaction:** `GenericBrain.step(observation, reward=...)` incorporates
  feedback and chooses the next action in one ongoing loop. Demonstrations enter the
  same loop through `teacher=`. Activity settles quickly; synapses change more slowly.
  Count all internal free/nudged phases when comparing computation.
- **Long-term memory through plasticity:** learned synaptic efficacies hold skills and
  representations until later learning changes them. Fast synapses store an association
  after one observation. `SynapticMemory` consolidates repeated or salient observations
  into persistent synapses that survive transient-memory resets. Later evidence can
  revise them. [Continuous learning and memory](docs/continuous.md).

These are useful design choices, not a universal speed or capability advantage.
Recurrent networks and memory-augmented transformers can also carry state and plan.
Cadence makes the state, feedback and update rules explicit; performance needs a
matched task, quality target and compute budget. [Comparison and limits](docs/concepts.md#compared-with-backprop-networks).

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

## Quickstart: teach a small brain

```python
import numpy as np
import cadence as cd

circuit = cd.layered(2, 8, 2, density=1.0, seed=0)
learner = cd.Learner(
    cd.Brain(circuit, cd.learning_neuron_model(dt=1.0)),
    circuit.populations["output"],
    cd.LearnerConfig(eta=2.0),
)
drive = np.zeros((2, circuit.n))
drive[:, list(circuit.populations["input"])] = np.eye(2)
for _ in range(50):
    learner.step(drive, np.array([0, 1]))
print(learner.predict(drive))  # [0 1]
learner.save("tiny_brain.npz")
```

This fits two responses, as an API demonstration. The [quickstart](docs/quickstart.md)
explains the phases, checks an equilibrium, intervenes on a circuit and reloads a model.
`GenericBrain` includes lasting synaptic memory by default. For thought between actions,
connect a `Deliberator` to your task loop; it keeps unfinished futures without inventing
new rewards. See [defaults, scheduling and tests](docs/continuous.md#defaults-and-the-thinking-clock).

For pictures, working memory and reward, start with [`GenericBrain`](docs/patterns.md#a-generic-brain).

## Build your brain

| Need | Start here |
|---|---|
| Neurons, synapses and a shared equilibrium | [Concepts](docs/concepts.md), [`equilibrate`](docs/api.md#brain-cadencebrain) |
| Standard visual, association, motor and working-memory regions | [Brain design guide](docs/design.md) |
| Imitation, practice and corrective teaching | [Learning lifecycle](docs/design.md#teach-practice-correct-and-retain) |
| Short- and long-term memory | [Function map](docs/biology.md), [memory](docs/memory.md) |
| Imagined futures, review and self-monitoring | [Patterns](docs/patterns.md#future-simulation) |
| Biological counterparts and what is still missing | [Biology mapping](docs/biology.md) |

## Examples

Six brains run in the browser with nothing to install. Each shows its live circuit
beside the body.

| Example | What it does | Circuit |
|---|---|---|
| **Eye & arm**<br>[Web demo](https://floatingpragma.io/cadence-examples/eye-arm/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/eye-arm) | A retina reads your drawing, and shoulder, elbow and pencil motors copy it. Push a joint or disable a motor population. | 3N + 17 neurons, up to 2N + 28 synapses (N sampled dark pixels) |
| **Teachable mouse**<br>[Web demo](https://floatingpragma.io/cadence-examples/mouse/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/mouse) | Teach it a new destination, revise the lesson, and carry it into a new maze. | 265 neurons, 285 synapses, 32 persistent + 32 transient weights |
| **C. elegans habitat**<br>[Web demo](https://floatingpragma.io/cadence-examples/worm/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/worm) | The worm's chemical connectome follows food cues around walls you draw. Stimulate or lesion its neurons. | 309 neurons, 4,108 synapses |
| **Fly-inspired forager**<br>[Web demo](https://floatingpragma.io/cadence-examples/fly/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/fly) | Learns nectar values on contact and changes its preferences when you change the flowers. | 18 neurons, 44 synapses, 32 persistent + 32 transient weights |
| **Connect Four**<br>[Web demo](https://floatingpragma.io/cadence-examples/connect-four/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/connect-four) | Imagines replies before moving, and a self-monitor asks for deeper search when the choice is close. | 19 neurons, 39 synapses |

Neurons are graded software units and the counts cover the circuit only. For scale, an
adult C. elegans has 302 neurons. The
[methods](https://github.com/muellerberndt/cadence-examples/blob/main/METHODS.md)
give the supplied parts and the comparisons for each example. From a clone of
cadence-examples, `python serve.py <example>` runs one offline.

## Docs

[Quickstart](docs/quickstart.md) · [Design a brain](docs/design.md) ·
[Concepts](docs/concepts.md) · [Patterns](docs/patterns.md) ·
[Function map](docs/biology.md) · [API](docs/api.md) · [All docs](docs/index.md)

Related methods: [equilibrium propagation](https://arxiv.org/abs/1602.05179) and
[delta-rule fast weights](https://arxiv.org/abs/2406.06484). MIT licensed.
