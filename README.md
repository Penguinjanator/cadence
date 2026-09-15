<p align="center">
  <img src="docs/assets/patchnet.svg" alt="Neurons hold state and exchange activity over declared synapses" width="100%">
</p>

# Cadence

**Decentralized neural networks that learn through symmetry breaking.**

Human brains do not learn by gradient descent and backpropagation, and they do not
freeze their weights after pretraining. Neither does Cadence. A Cadence network is made
of cortices that settle together into equilibria, where a transformer stacks feedforward
layers with attention. Each synapse changes from the activity of its own two neurons, so
there is no backward pass, and the network keeps learning while it runs.

Cadence is a Python library for neurons, synapses and named functional regions.
Each neuron reads its own state and incoming activity, then relaxes toward their
combined drive. Feedback carries changes between regions as they seek a common
equilibrium. Local free/nudged contrasts teach responses; traces, fast memory and
reward learning let a controller carry experience into its next decision.

## Why Cadence

- **No backpropagation.** Each synapse learns from its own two neurons. No neuron reads a
  global error, and no computation graph is stored.
- **No frozen weights.** Learning happens during use. A brain keeps adapting to new
  observations and rewards for as long as it runs.
- **Short-term memory arises naturally.** Activity fades slowly and feeds back into the
  next settle, so thoughts linger in the brain.
- **Long-term memory arises naturally.** Plasticity stores salient and repeated facts in
  persistent synapses. A fast synapse holds a new association after a single observation.
  [Memory](docs/memory.md).
- **A continuous stream of thought.** A Cadence brain does not run in shots or discrete
  invocations. It is one ongoing loop, and each observation and reward arrives while the
  brain is still thinking. [Continuous learning](docs/continuous.md).
- **Imagined futures.** Small random drive breaks the symmetry of a settled state and
  pushes the brain toward nearby alternatives. The brain settles each version of the
  future, compares them and acts on the best one.
- **Built like biology.** Neurons, synapses, cortices, basal ganglia and hippocampus are
  the building blocks, and real connectomes load as plain data.

These are useful design choices, not a universal speed or capability advantage.
Recurrent networks and memory-augmented transformers can also carry state and plan.
Cadence makes the state, feedback and update rules explicit; performance needs a
matched task, quality target and compute budget. [Measured comparisons](docs/comparisons.md)
and [architectural limits](docs/concepts.md#compared-with-backprop-networks).

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
| Short- and long-term memory | [Function map](docs/biology.md), [memory](docs/memory.md), [content retrieval](docs/content_memory.md), [rehearsal](docs/replay.md) |
| Imagined futures, review and self-monitoring | [Patterns](docs/patterns.md#future-simulation) |
| Biological counterparts and what is still missing | [Biology mapping](docs/biology.md) |

## Examples

Five official examples, each with its live circuit beside the task. Four run in the
browser with nothing to install. The composer runs locally with its pretrained model.

| Example | What it does | Circuit |
|---|---|---|
| **Eye & arm**<br>[Web demo](https://floatingpragma.io/cadence-examples/eye-arm/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/eye-arm) | A retina reads your drawing, and shoulder, elbow and pencil motors copy it. Push a joint or disable a motor population. | 3N + 17 neurons, up to 2N + 28 synapses (N sampled dark pixels) |
| **C. elegans habitat**<br>[Web demo](https://floatingpragma.io/cadence-examples/worm/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/worm) | The worm's chemical connectome follows food cues around walls you draw. Stimulate or lesion its neurons. | 309 neurons, 4,108 synapses |
| **Composer**<br>[Studio page](https://floatingpragma.io/cadence-examples/composer/)<br>[Model card](https://github.com/muellerberndt/cadence-examples/blob/main/composer/MODEL_CARD.md)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/composer) | maestro-1, a pretrained musician of wired cortices, composes from a mood. It imagines continuations, listens to the whole draft and edits its weakest passage, with every neuron in view. | 17,855 neurons, 68,570,458 synapses |
| **Fly-inspired forager**<br>[Web demo](https://floatingpragma.io/cadence-examples/fly/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/fly) | Learns nectar values on contact and changes its preferences when you change the flowers. | 18 neurons, 44 synapses, 32 persistent + 32 transient weights |
| **Connect Four**<br>[Web demo](https://floatingpragma.io/cadence-examples/connect-four/)<br>[Code](https://github.com/muellerberndt/cadence-examples/tree/main/connect-four) | Imagines replies before moving, and a self-monitor asks for deeper search when the choice is close. | 19 neurons, 39 synapses |

Neurons are graded software units and the counts cover the circuit only. For scale, an
adult C. elegans has 302 neurons. The
[methods](https://github.com/muellerberndt/cadence-examples/blob/main/METHODS.md)
give the supplied parts and the comparisons for each example. From a clone of
cadence-examples, `python serve.py <example>` runs one offline.
The [cortex catalogue](https://github.com/muellerberndt/cadence-examples/tree/main/cortices) in the same repository builds basic
regions (senses, association, motor, working memory, clocked record, conditioning) and
assembles them into one brain.

## Docs

[Quickstart](docs/quickstart.md) · [Design a brain](docs/design.md) ·
[Concepts](docs/concepts.md) · [Patterns](docs/patterns.md) ·
[Function map](docs/biology.md) · [API](docs/api.md) · [All docs](docs/index.md)

Related methods: [equilibrium propagation](https://arxiv.org/abs/1602.05179) and
[delta-rule fast weights](https://arxiv.org/abs/2406.06484). MIT licensed.
