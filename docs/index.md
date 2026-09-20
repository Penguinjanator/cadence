# Cadence documentation

Cadence researches general intelligence through overlap consensus, equilibrium
detuning and functional self-reflection. One continuing system should acquire
skills, retain useful experience, imagine alternatives and repair its behavior
across applications. The guides distinguish implemented operations from that
broader research goal.

## Start with the current architecture

1. [Architecture and integration](architecture.md): state, ports, repair and the scope of each component.
2. [Temporal learning](temporal.md): observed paths, persistent context, local detuning and isolated imagination.
3. [Response protection](temporal-memory.md): conditional retention and well-conditioned learning in remaining directions.
4. [Private planning](planning.md): repair continuous controls under the same learned model.
5. [Learn, act and observe](interaction.md): an executable end-to-end interaction example.
6. [Creativity and self-reflection](creativity.md): novel proposal evaluation, recursive readback and transfer as research requirements.

[Task design](task-design.md) and [common missteps](missteps.md) explain how to
validate observations, actions, learning and retained behavior before scaling.
The [API reference](api.md) lists current interfaces and supported compatibility
components. These guides target version 0.11.0.

[EquilibriumActor](actor.md) provides a separate fixed linear-model example of
factual inference, compressed past context and joint future-state/action repair.
Its exact compression assumptions do not extend to arbitrary changing models.

## Existing graph and composition APIs

These supported interfaces have distinct state and learning contracts; they
are not automatically wired into the temporal architecture.

- [PatchNet](patchnet.md): continuous observations and isolated graph dynamics.
- [Quickstart](quickstart.md), [experience](experience.md) and [continuous interaction](continuous.md): the earlier `GenericBrain` composition.
- [Write a cortex](cortex.md), [compose a brain](brain.md) and [evolve a brain](evolution.md): graph construction and supplied-fitness search.
- [Concepts](concepts.md), [local learning](learning.md), [records and memory](memory.md) and [reward](reward.md): the corresponding graph components.
- [Task recipes](tasks.md), [rehearsal](replay.md), [content memory](content_memory.md) and [sequence readback](sequence.md): optional application compositions.

## Measurement and mathematical scope

[Convergence certificates](certificate.md) · [Protocols](protocols.md) ·
[Receipts](receipts.md) · [Brain viewer](pages.md) · [Backends](backends.md) ·
[Conditional Lean proofs](../lean/README.md).

Examples illustrate a general mechanism. Their scores are evidence for the
named task, not proof that the architecture solves arbitrary problems.
