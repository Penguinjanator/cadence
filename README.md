<p align="center">
  <img src="docs/assets/cadence-logo.png" alt="Cadence — an interconnected mesh of stateful neural patches, feedback loops and synaptic signals" width="100%">
</p>

# Cadence

**Neural systems that learn through an ongoing stream of experience.**

Build a brain together with its learning life: what it observes, remembers,
predicts, wants and does. The same persistent system makes decisions and learns
from their consequences. There is no separate training/inference mode in the
`GenericBrain.step` loop.

**Observe → remember → predict → act or communicate → learn from the outcome.**

Thinking reads the current state; actual observations, rewards and corrections
change what is learned. Imagined outcomes never become witnessed facts. The
application controls when each event arrives, including while the world waits.

## What the brain is made of

- **Local neural state:** neurons exchange activity over declared synapses;
  recurrent regions influence one another as they settle.
- **Local learning:** observed targets and reward drive synaptic updates from
  neuronal activity and eligibility, with no backward computation graph.
- **Memory:** traces carry recent activity; fast synapses retain associations;
  slow synapses consolidate repeated or salient evidence.
- **Prediction and goals:** connect learned consequences, remembered events and
  desired outcomes to action. Isolated imagined futures can guide a choice.
- **Grounded communication:** learn words in context, retain an intention, express
  it and learn from a partner's response. Planning and speaking can interleave.

These are observer-like, self-reading software patches: bounded local state,
declared ports, readback, records and feedback/repair, with checkable evidence.
The [experience guide](docs/experience.md) connects the functions and curriculum.
World models, learned goals and language must be built and tested; `GenericBrain`
currently supplies a recurrent policy, critic and associative reward memory.
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

## Reference

[Experience](docs/experience.md) · [Quickstart](docs/quickstart.md) ·
[Continuous interaction](docs/continuous.md) · [Memory](docs/memory.md) ·
[Local learning](docs/learning.md) · [Reward](docs/reward.md) ·
[API](docs/api.md) · [All docs](docs/index.md)

Check equation residuals before claiming equilibrium. Measure task quality and
learning cost; local updates alone guarantee neither capability nor speed.
[Concepts and limits](docs/concepts.md). MIT licensed.
