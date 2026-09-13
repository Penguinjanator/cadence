<p align="center">
  <img src="docs/assets/patchnet.svg" alt="Owners hold state and exchange messages over declared seams" width="100%">
</p>

# Cadence

**Small, stateful networks that read, repair, and learn locally.**

Cadence builds OPH-style observer-like software patches: bounded local state, declared
input and output ports, readback, records, and feedback that repairs a discrepancy.
A patch net joins those patches with weighted seams. Public examples include executable
comparisons and evidence receipts.

The useful core is small:

| need | operation | API |
|---|---|---|
| infer under interacting constraints | owners exchange messages and repair their state | `Wiring`, `Settlement` |
| learn a reusable response | settle free and nudged, then contrast each seam's endpoints | `Learner` |
| remember an observation | read the association, then write its prediction error | `FastSeams(rule="delta")` |
| retain recent activity or delayed credit | decay a trace across moments | `Trace`, `ActorCritic` |

Use the operation the task needs. An addressed memory read needs one matrix product;
adding settlement to that read does not automatically improve it. A recurrent inference
problem may need several repairs. Fast records and slowly learned weights serve different
purposes, even when both are stored on seams.

## Install

```bash
pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@main"
```

Python 3.11+. This installs the current source used by the examples below. Add `[fast]`,
`[accel]` or `[apple]` after `cadence-net` for optional compiled CPU/SciPy, PyTorch or
MLX support. From an existing checkout, use `pip install -e ".[fast]"`.
The published PyPI version is still 0.8.1 and does not include the new memory API.

## Remember, then correct

```python
import numpy as np
import cadence as cd

memory = cd.FastSeams(np.arange(3), np.arange(3, 5), rule="delta")
key = np.array([[0., 1., 0.]])
memory.observe(key, np.array([[1., 0.]]))
print(memory.recall(key))             # [[1. 0.]]
memory.observe(key, np.array([[0., 1.]]))
print(memory.recall(key))             # [[0. 1.]]: the revised observation
```

For a unit key `k`, the entire write is `M += outer(k, value - k @ M)`.
It changes nothing when the prediction is already correct. Orthogonal keys retain their
own values; overlapping keys can interfere. This is a delta-rule associative memory,
an established online learning rule, with explicit ports and stream resets.
[Memory](docs/memory.md) gives the contract, limits, and composition with a settlement.

## Learn by detuning an equilibrium

```python
w = cd.layered(2, 8, 2, density=1.0, seed=0)
learner = cd.Learner(cd.Settlement(w, cd.learning_rule()), w.sets["output"])
drive = np.zeros((2, w.n))
drive[:, :2] = np.eye(2)
phases, report = learner.step(drive, np.array([0, 1]))
prediction = learner.predict(drive)
```

The free phase supplies an answer. Nudging the output ports toward and away from a
label changes the neighbouring owners through the same seams. Each seam updates from
the difference of its endpoint products. No backward computation graph is retained;
the phase states and any optimizer history are retained.

For symmetric effective recurrent weights, a smooth stable equilibrium branch, and
converged phases, the small-nudge limit gives an equilibrium-propagation gradient.
Finite steps, saturation, adaptation, or asymmetric feedback can invalidate that
interpretation. [Learning](docs/learning.md) states the equations and assumptions.

## Examples with controls

| example | what it demonstrates |
|---|---|
| [01 digits](https://github.com/muellerberndt/cadence-examples/tree/main/01_digits) | supervised classification, validation selection, and an MLP comparison |
| [02 recall](https://github.com/muellerberndt/cadence-examples/tree/main/02_recall) | one-hot associative storage; a direct read also solves it |
| [03 Connect Four](https://github.com/muellerberndt/cadence-examples/tree/main/03_connect_four) | imitation of a search policy, with positions grouped before reflection augmentation |
| [04 Pong](https://github.com/muellerberndt/cadence-examples/tree/main/04_pong) | reward-weighted learning and a REINFORCE comparison |
| [05 changing memory](https://github.com/muellerberndt/cadence-examples/tree/main/05_memory) | revisable records, additive-memory ablation, a trained transformer, exact lookup controls, and interference |
| [06 interventions](https://github.com/muellerberndt/cadence-examples/tree/main/06_interventions) | a supplied feedback circuit handles new wiring, stimulation and ablations; trained MLP, fixed unrolling and Newton controls |

In the memory example, eight orthogonal keys retain their latest values across 128 writes
with **100% accuracy**, versus **26.0% additive memory and 14.5% for the tested
transformer**. This is length extrapolation: the transformer trains through 32 writes
and reaches 98.2% there. Exact lookup also scores 100%; strongly overlapping keys can
make residual memory much worse than the transformer. See the complete result table.

The older supervised examples reach broadly similar accuracy to their MLP baselines,
usually with greater training time. Their historical results are bound to their original
sources. The new memory example isolates an advantage of an explicit, bounded store;
it also includes conditions where the trained transformer is better. It does not establish
general transformer superiority, solved continual learning, or orders-of-magnitude
improvements in general machine learning.

The intervention example tests a different saving: reuse a known local rule when the
circuit changes. Cadence can solve each new configuration without fitting an input/output
surrogate. A conventional graph recurrence or its feedforward unroll can reuse the same
rule too. The benchmark measures residual, prediction error and execution cost against
a trained MLP; it distinguishes this supplied structure from learned physics.

## Check what happened

`Settlement.residual(drive, state)` measures the remaining fixed-point discrepancy;
an activation stopping tolerance alone can be fooled by saturation. `conformance`
compares a settlement trajectory with an owner-by-owner reference. It checks the tested
transport and update, not every learning, memory, normalization, or reward operation.

`Receipt` binds a result to source digests and recorded outcomes. A useful comparison
also needs shared data, separate validation, complete seeds and controls, and timings
that include the work actually needed for a decision. Digests establish provenance;
they do not establish a fair benchmark by themselves.

[Quickstart](docs/quickstart.md) · [Concepts](docs/concepts.md) ·
[Learning](docs/learning.md) · [Memory](docs/memory.md) ·
[Backends](docs/backends.md) · [API](docs/api.md) · [All docs](docs/index.md)

Cadence grew from connectome and observer-patch experiments. Its slow learning rule is
[equilibrium propagation](https://arxiv.org/abs/1602.05179); residual fast weights are
also used in [delta-rule linear transformers](https://arxiv.org/abs/2406.06484).
These connections are useful prior art, not evidence that one architecture wins every task.

MIT licensed.
