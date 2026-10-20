<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: a continuing brain with local state, memory and repair" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/cadence/) · [Documentation](https://github.com/muellerberndt/cadence/blob/main/docs/README.md) · [Application demos](https://github.com/muellerberndt/cadence-demos) · [Paper](https://philpapers.org/rec/MUECAP-2) · [PyPI](https://pypi.org/project/cadence-net/)

[![CI](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/muellerberndt/cadence/blob/main/LICENSE)

**One continuing equilibrium brain: learn a world, act in it, repair what fails.**

```python
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, arousal=True)
action = brain.live([[1.0, 0.0, 0.0, 0.0]])
```

Execute the returned action, then pass its measured reward with the next
observation: `brain.live(next_observation, reward=reward)`. Tune named settings
with `compose` or `brain.retune(...)`; inspect them with `brain.describe()`.
The [composition guide](https://github.com/muellerberndt/cadence/blob/main/docs/brain.md#defaults-and-expert-overrides)
explains the defaults and which learning rates each operation uses.

Cadence builds a simulated animal-like brain from simplified biological
mechanisms. Its default **System 1** is one continuing brain with perception,
plastic connections, memory and action; optional **System 2** adds observing
cortical regions that feed back into the same neural graph. Biological names
describe functional software roles.

[One continuing equilibrium brain](https://github.com/muellerberndt/cadence/blob/main/docs/world-model.md)
is the guide to this lifecycle and its boundaries. `compose` constructs
the brain; `live` runs its life. Integrated learned world prediction, repair
localized to what failed and cheap stable operation are work in progress.
Cadence is alpha research software.

## How a Cadence brain differs from a feed-forward network

`Brain.compose(inputs=4, actions=2, modules=(64, 32, 16))` builds reciprocal
regions whose activity settles together, so later regions shape earlier ones
while an answer forms, and the action comes from the qualified neural state.
The learner compares local activities in a free and a nudged phase, without a
backward differentiation pass. The relevant comparison with a recurrent
backprop network is acquired behavior and total work on the same task, which
[the orientation page](https://github.com/muellerberndt/cadence/blob/main/docs/orientation.md#compared-with-a-backprop-network)
sets out; other model families have
[their own learning contracts](https://github.com/muellerberndt/cadence/blob/main/docs/contracts.md).

## Why Cadence

The library puts continuing state, reciprocal interpretation, local plasticity
and memory behind one small interface. A saved brain resumes an action that
awaits its actual outcome, and private imagination inspects responses without
rewriting live experience. These are mechanisms for testing acquisition,
retention and recovery through a life.

The target includes full-sentence language and reusable world understanding;
a small fixed-label task is a control.

<a id="get-started"></a>

## Start with System 1

Python 3.11+ and NumPy are required. Install the published release, whose
[released documentation](https://github.com/muellerberndt/cadence/blob/v0.84.3/docs/README.md)
describes the APIs in that package:

```sh
python -m pip install cadence-net==0.84.3
```

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, arousal=True)
observation = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.live(observation)

# A tiny environment rewards action 0 and supplies the next observation.
reward = (action == 0).astype(float)
next_observation = np.array([[0.0, 1.0, 0.0, 0.0]])
action = brain.live(next_observation, reward=reward)
assert action.shape == (1,)
```

`live` takes the **preceding action's** measured reward, then chooses the next
action for one continuing stream; supply one observation row per call. It needs
`arousal=True` (the founder arousal settings). A calm brain acts greedily without
learning; youth (the first 100 moments) or arousal enables exploration, learning
and writes to the associative memory. There is no training/inference mode switch.
The working trace advances at every accepted moment; with the opt-in recall
memory (`recall=True`) its slots advance at every accepted moment too, or only at
aroused moments with `recall_write="aroused"`.
[Continuous interaction](https://github.com/muellerberndt/cadence/blob/main/docs/continuous.md)
covers arousal, resets and saved continuation.

For explicit teaching, batched streams or learning from every outcome, use the
lower-level `step` loop, which works with or without `arousal=True`; its
`teacher=` labels the current observation. The
[continuing brain example](https://github.com/muellerberndt/cadence/blob/main/examples/continuing_brain.py)
runs that loop through changed conditions and saved continuation.

A teacher label changes graph parameters. The default associative memory
ignores labels; the opt-in episodic memory (`memory_choice=True`) records them.
[Memory](https://github.com/muellerberndt/cadence/blob/main/docs/memory.md)
lists every memory of the composed brain and which one suits which task.

```python
phases = brain.imagine([observation, next_observation])
assert phases  # Inspect phase.qualified before using an imagined response.
```

Imagination carries a private trace without changing live memory, random state
or pending feedback. Learned consequences and action planning belong to the
[temporal model](https://github.com/muellerberndt/cadence/blob/main/docs/planning.md#learn-consequences-then-act).

Each action solve reports its residual, qualification, sweeps and check counts
in read-only `brain.last_settlement`, as the
[quickstart](https://github.com/muellerberndt/cadence/blob/main/docs/quickstart.md#inspect-the-work-of-answering)
shows. An action whose settle exhausts its budget is refused without changing
live state; [qualification and refusal](https://github.com/muellerberndt/cadence/blob/main/docs/continuous.md#qualification-and-refusal)
says when to retry `live(observation)` without resubmitting the reward.

<a id="development-checkout"></a>

To work from source, install the library repository root as an editable package:

```sh
python -m pip install -e .
```

## Add optional System 2

```python
recursive = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), observers=(8,), seed=7,
    arousal=True,
)
```

Observer regions read and return influence to the base, motor regions and earlier
observers. They join the same settlement and use the same interaction interface.
Whether recursive feedback helps on a task is a question for experience and
evaluation.

## Go further

[MANIFESTO.md](https://github.com/muellerberndt/cadence/blob/main/MANIFESTO.md)
states the design points, including the language goal of whole consistent
sentences from the settled state.
[Build a brain](https://github.com/muellerberndt/cadence/blob/main/docs/brain.md)
for custom wiring, and [the memory/planning example](https://github.com/muellerberndt/cadence/blob/main/examples/memory_imagination.py)
for a bounded demonstration with actual toy-body outcomes.
[Record patches](https://github.com/muellerberndt/cadence/blob/main/docs/record-patch.md)
provide event records and consolidation. The advanced
[population solver](https://github.com/muellerberndt/cadence/blob/main/docs/equilibrium/index.md)
provides exact state-and-error readback under its own numerical contract.
[cadence-demos](https://github.com/muellerberndt/cadence-demos) contains the
application demos.

[Documentation](https://github.com/muellerberndt/cadence/blob/main/docs/README.md) ·
[API](https://github.com/muellerberndt/cadence/blob/main/docs/api.md) ·
[Contributing](https://github.com/muellerberndt/cadence/blob/main/CONTRIBUTING.md) ·
[Changelog](https://github.com/muellerberndt/cadence/blob/main/CHANGELOG.md) ·
[Research tasks](https://github.com/muellerberndt/cadence/issues)

Licensed under [GPL-3.0](https://github.com/muellerberndt/cadence/blob/main/LICENSE).

<details>
<summary>Related projects and funding</summary>

[Observer Patch Holography](https://github.com/FloatingPragma/observer-patch-holography)
(OPH) is the related physics project. Its physics evidence is separate from
Cadence's capability evidence.
[Pragma Research](https://floatingpragma.io/) brings the projects together.

Cadence has an official memecoin on Solana, deployed by the Solana community.
Its token address is `46doJPsSjEVMsNZm4b32UbbvbUXjFTM3TDPBWMM3pump`
([chart on Dexscreener](https://dexscreener.com/solana/4tjiqzqocttntmdqqmev2spoegzipuerfyramnfxv9su)).
Its trading fees are routed to Bernhard Mueller to fund the project. Pragma
Research holds none of the token. Cadence itself does not use it, and no
integration into the core library is planned. We encourage the crypto community
to build interesting projects that integrate it. Any other token using the
Cadence name is unrelated to this project. Nothing here is financial advice.

</details>
