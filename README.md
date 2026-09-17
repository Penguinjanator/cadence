<p align="center">
  <img src="docs/assets/cadence-logo.png" alt="Cadence: a mesh of stateful neural patches, feedback loops and synaptic signals" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/) · [Cadence page](https://floatingpragma.io/cadence/) · [Live brains](https://floatingpragma.io/cadence-examples/) · [Examples repository](https://github.com/muellerberndt/cadence-examples) · [PyPI](https://pypi.org/project/cadence-net/)

**Decentralized networks of neurons that learn through symmetry breaking, from an
ongoing stream of experience.**

> **Status: under heavy development.** Interfaces, defaults and training methods change
> rapidly between commits and releases as the research settles; pin an exact commit or a
> release for anything that has to keep working, and expect the documentation to lag the code
> in places.

Human brains do not learn by gradient descent and backpropagation, and they do not
freeze their weights after pretraining. Neither does Cadence. A Cadence brain is made of
cortices that settle together into equilibria, where a transformer stacks feedforward
layers with attention. Each synapse changes from the activity of its own two neurons, and
the same brain acts and learns throughout its life. Build a brain together with its
learning life: what it observes, remembers, predicts, wants and does.

## Why Cadence

- **No backpropagation.** Each synapse learns from its own two neurons and one broadcast
  error. No neuron reads a global gradient, and no computation graph is stored.
- **No training and inference steps.** There is one mode. Each `GenericBrain.step`
  takes in an observation and the previous action's outcome, updates memories and
  synapses, and chooses the next action. There is no switch between learning and use, no
  separate deployment model and no point at which learning must stop.
- **No frozen weights.** Learning happens during use. A brain keeps adapting to new
  observations, rewards and corrections for as long as it runs.
- **Short-term memory arises naturally.** Settled regions carry context across a delay
  and complete a partial reading, so thoughts linger in the brain.
- **Long-term memory arises naturally.** Salient and repeated facts are written into the
  records the current reading touches, and plasticity keeps them in persistent synapses.
  One stream, no replay ring.
- **A continuous stream of thought.** A Cadence brain does not run in shots or discrete
  invocations. It is one ongoing loop, and each observation and reward arrives while the
  brain is still thinking.
- **Imagined futures.** Small random drive breaks the symmetry of a settled state and
  pushes the brain toward nearby alternatives. The brain settles each imagined future in
  isolation, compares the recorded consequences and acts on the best one. Imagined
  outcomes never become witnessed facts.
- **Built like biology.** Neurons, synapses, cortices, a critic and an associative
  reward memory are the building blocks, and real connectomes load as plain data.

## How the brain works

**Observe → remember → predict → act or communicate → learn from the outcome.**

Thinking reads the current state; actual observations, rewards and corrections change
what is learned. The application controls when each event arrives. Keep the issued action
pending until its real outcome arrives; a clock tick alone is not new feedback. Start with
the [single-loop quickstart](docs/quickstart.md).

- **Local neural state:** neurons exchange activity over declared synapses; the settled
  regions reach a joint fixed point that completes a partial reading, carries context
  across a delay and holds the policy.
- **Records:** a mean-free reading passes through a fixed sparse expansion with
  winner-take-all inhibition; each active cell keeps a record of what followed; the
  prediction is the activity-weighted sum of the records the reading touches; the
  witnessed outcome is written into exactly those, at a slow rate for consequences and a
  fast rate for reward.
- **Local learning in the settled regions:** free and nudged phases and eligibility
  traces change synapses from their own two neurons' activity and one broadcast error,
  with no backward computation graph.
- **Prediction and goals:** imagined consequences are record reads; a supplied search
  over them and isolated imagined futures guide a choice.

These are observer-like, self-reading software patches: bounded local state, declared
ports, readback, records and feedback/repair, with checkable evidence. The
[experience guide](docs/experience.md) connects the functions and curriculum.
`GenericBrain` is the ready composition of settled regions: a recurrent policy, a critic
and an associative reward memory. The records cortex is `cd.Records`
([records](docs/memory.md#records)). Learned goals and language are not supplied.
Cadence does not claim to reproduce human learning or supply a pretrained chatbot.

## Install

Python 3.11+, with NumPy as the only required dependency:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@main"
```

On Windows activate with `.venv\Scripts\Activate.ps1`.
[Optional backends](docs/backends.md) support Numba, PyTorch and MLX.

## Examples

The [cadence-examples](https://github.com/muellerberndt/cadence-examples) repository has
complete brains to try out. Each one runs in the browser with the brain drawn beside it, and
the arm, world, Connect Four and artist come with acceptance receipts and a verifier that
recomputes them from the event logs. Clone it, play with the pages, then change the wiring or
the world and watch what the brain learns. Comparison runners put an online MLP and an online
transformer through the same lives.

[Patch World](https://github.com/muellerberndt/cadence-world) is the best place to start
tinkering. Creatures on a small world under a moving sun inherit their brain wiring and learn
within one life, and energy and death decide which brains survive. The whole simulation is one
JavaScript file and the page opens in a browser. Change the prices, add a rule and see whether
a bigger brain becomes worth carrying. It is MIT licensed and made to be forked.

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
