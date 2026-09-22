<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: connected patches with local state, readback and repair" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/cadence/) · [Paper](https://floatingpragma.io/cadence/paper.pdf) · [PyPI](https://pypi.org/project/cadence-net/) · [Documentation](https://github.com/muellerberndt/cadence/blob/main/docs/index.md)

**Research toward general intelligence through overlap consensus, equilibrium detuning and self-reflection.**

Cadence's goal is a continuing learning system with the flexibility of animal
and human problem solving: acquiring skills from experience, retaining useful
knowledge, imagining alternatives and creating solutions across domains.
The mission is to find the smallest persistent state and local update rule
that can support these abilities. General intelligence is the research goal;
the current library establishes bounded learning, memory and control results.

The organizing idea comes from Observer Patch Holography: bounded,
observer-like patches with local state, ports, records, readback and repair.
A disturbance exposes disagreement. The network can explore a possible
response, test it against actual consequences and settle into a revised
organization. We seek fewer mechanisms that solve more problems.

## Three shared principles

- **Overlap consensus:** patches repair disagreement across their shared
  boundaries. The resulting equilibrium is an internally consistent model;
  its predictions still have to agree with experience.
- **Equilibrium detuning:** observed outcomes perturb that equilibrium.
  Local positive/negative contrasts change learned relationships; the same
  operation can adjust proposed actions while holding the model fixed.
- **Functional self-reflection:** patches can read internal state,
  predictions, proposed actions and unresolved mismatches through ordinary ports. Learning which
  internal summaries to read, how to feed them back and how to grow useful
  recursive organization is a central research direction.

The current implementation provides detached self-readback and private proposal
revision. Automatically learned recursive hierarchies, curiosity and reliable
creativity remain to be demonstrated. [Creativity and self-reflection](https://github.com/muellerberndt/cadence/blob/main/docs/creativity.md)
defines these goals and their behavioral tests.

## Current library

Version **0.11.0** brings learning and continuous action planning into the same
temporal model. It requires Python 3.11+ and NumPy:

```bash
python -m pip install cadence-net==0.11.0
```

| Component | What it supplies |
| --- | --- |
| [TemporalPatchNet](https://github.com/muellerberndt/cadence/blob/main/docs/temporal.md) | Local repair of observed paths, centered detuning, persistent context, private imagination and complete checkpoints. |
| [Private planning](https://github.com/muellerberndt/cadence/blob/main/docs/planning.md) | Bounded continuous action proposals under the learned model; accepted steps must improve predictions replayed without goal nudging. |
| [RecordPatchNet](https://github.com/muellerberndt/cadence/blob/main/docs/record-patch.md) | Development addition on `main`: a gated linear context with one-shot records inside the patch. Its detuning contrast equals its adjoint gradient, at linear cost per moment. Categorical ports, batched record writes, a two-patch stack (`RecordPatchStack`) and a store narrower than its port came from the language work; one pass of record writes, with no gradient, gives a small grammar for 0.8 of its never-taught combinations. |
| [TemporalMemory](https://github.com/muellerberndt/cadence/blob/main/docs/temporal-memory.md) | Explicit protection of selected local responses; an optional local readout metric improves conditioning without retaining a replay corpus. |
| [EquilibriumActor](https://github.com/muellerberndt/cadence/blob/main/docs/actor.md) | A separate fixed linear-body component with exact Gaussian history compression and factual readback. |

Development checkouts also expose [experimental fixed connectivity](https://github.com/muellerberndt/cadence/blob/main/docs/partitioned.md)
under `cadence.experimental`. It supplies routing constraints for controlled
comparisons, not learned specialization, and is not part of the 0.11.0 release.

Start with the executable [learn, act and observe guide](https://github.com/muellerberndt/cadence/blob/main/docs/interaction.md).
It acquires an action/consequence relation, plans privately, executes bounded
actions and uses actual readback to repair its next proposal. The
[architecture guide](https://github.com/muellerberndt/cadence/blob/main/docs/architecture.md) maps each capability to its API and current scope.

**Imagined continuations are isolated.** Branches use the learned network
without changing live activity, parameters or factual bookkeeping. Controlled
experiments demonstrate useful planning. Creativity requires additional
evidence that novel proposals satisfy meaningful constraints and survive
actual evaluation; musical improvisation is one possible example.

**Retained experience and new learning are tested together.** Protected-path
memory is conditional and finite. Importance is currently supplied; automatic
relevance, selective forgetting, specialization and broad skill transfer remain
research requirements. The [task-design guide](https://github.com/muellerberndt/cadence/blob/main/docs/task-design.md)
and [common missteps](https://github.com/muellerberndt/cadence/blob/main/docs/missteps.md) explain how to measure them.

## General mechanisms, different applications

Games, language, multimodal perception, embodied control and creative work
should use the same learning and memory mechanisms with declared observation
and action ports. The [examples repository](https://github.com/muellerberndt/cadence-examples)
holds four worked applications: a worm that learns during its life, a composer
that starts from silence, soft bodies that evolve together with their brains,
and Connect Four. They are application tests, not definitions of the
architecture. A result in one does not establish transfer to the others.

The scaling goal is better learned behavior from more experience and training,
with as little manual design as possible. Measure unique experience, repeated
training and model capacity separately while keeping port meanings and task
evaluation fixed. The [scaling guide](https://github.com/muellerberndt/cadence/blob/main/docs/scaling.md)
defines these comparisons and the current computational limits.

Application demonstrations are published only when they establish their
claimed behavior. Recall and interpolation are useful development tests;
original creation requires stronger evidence. Every example states what is
supplied, what is learned, what was measured and what it does not show, and
carries a check that recomputes its numbers. Research receipts remain
available with the paper without presenting those tests as finished products.

## Proofs and compatibility

The [bundled Lean library](https://github.com/muellerberndt/cadence/blob/main/lean/README.md)
contains 169 checked conditional theorems about the mathematical components
and their limits. It does not certify the complete Python implementation or
prove intelligence. The paper identifies assumptions and reproducible evidence.

The existing [PatchNet graph interface](https://github.com/muellerberndt/cadence/blob/main/docs/patchnet.md),
`Brain`, `Learner`, `GenericBrain`, `Records` and circuit APIs remain supported
for existing applications. Their optional memories and construction helpers
are distinct compositions, not required parts of the temporal core.
[API reference](https://github.com/muellerberndt/cadence/blob/main/docs/api.md).

Development installs use `python -m pip install -e .`.
[Optional backends](https://github.com/muellerberndt/cadence/blob/main/docs/backends.md)
apply to their documented graph APIs; the temporal implementation is NumPy.
Pin a release or exact commit for reproducible work. MIT licensed.
