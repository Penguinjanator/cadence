<p align="center">
  <img src="docs/assets/cadence-logo.png" alt="Cadence: a mesh of stateful neural patches, feedback loops and synaptic signals" width="100%">
</p>

# Cadence

**Neural systems that learn through an ongoing stream of experience.**

Build a brain together with its learning life: what it observes, remembers,
predicts, wants and does. Its experience shapes the decisions it makes next.

## One mode: ongoing experience

**The same brain acts and learns throughout its life.** Each `GenericBrain.step`
incorporates an observation and the previous action's outcome, updates its
memories and synapses, and chooses the next action. There is no training/inference
switch, separate deployment model or point at which learning must stop.

**Observe → remember → predict → act or communicate → learn from the outcome.**

Thinking reads the current state; actual observations, rewards and corrections
change what is learned. Imagined outcomes never become witnessed facts. The
application controls when each event arrives. Keep the issued action pending until
its real outcome arrives; a clock tick alone is not new feedback.
Start with the [single-loop quickstart](docs/quickstart.md).

## What the brain is made of

- **Local neural state:** neurons exchange activity over declared synapses; the settled
  regions reach a joint fixed point that completes a partial reading, carries context
  across a delay and holds the policy.
- **Records:** a mean-free reading passes through a fixed sparse expansion with
  winner-take-all inhibition; each active cell keeps a record of what followed; the
  prediction is the activity-weighted sum of the records the reading touches; the
  witnessed outcome is written into exactly those, at a slow rate for consequences and a
  fast rate for reward. One stream, no replay ring.
- **Local learning in the settled regions:** free and nudged phases and eligibility
  traces change synapses from their own two neurons' activity and one broadcast error,
  with no backward computation graph.
- **Prediction and goals:** imagined consequences are record reads; a supplied search
  over them and isolated imagined futures guide a choice.

These are observer-like, self-reading software patches: bounded local state,
declared ports, readback, records and feedback/repair, with checkable evidence.
The [experience guide](docs/experience.md) connects the functions and curriculum.
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

Two applications built on this loop, each with its page, its acceptance receipt and the
verifier that recomputes the receipt from the event logs.

- [Arm](https://floatingpragma.io/cadence-examples/arm/): a two-link arm learns its own
  body from motor babbling, reaches targets through a search over the consequences it has
  recorded, adapts to a longer link without a reset and returns to its original body.
  [Receipt](https://github.com/muellerberndt/cadence-examples/blob/main/arm/receipt.json).
- [World](https://floatingpragma.io/cadence-examples/world/): in a 6 by 6 world seen one
  cell at a time, one life learns the consequences of its actions, remembers where it saw
  an object, corrects that memory when the object moves, holds a cue across a delay and
  grounds words in objects.
  [Receipt](https://github.com/muellerberndt/cadence-examples/blob/main/world/receipt.json).

Both learn from one stream with no replay ring: the world model is a records cortex and
the settled regions complete partial readings, carry context and hold the policy.
[Gallery](https://github.com/muellerberndt/cadence-examples).

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
