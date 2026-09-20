# Cadence documentation

Start with one ongoing learning life: observations, memories, predictions,
actions and their outcomes. Cadence makes the local state, ports, readback,
records and repair explicit.

[Architecture and integration](architecture.md) maps each requested function to
the implemented base API and its measured scope.

[Task design](task-design.md) and [common missteps](missteps.md) explain how to
validate what a task exposes, what its actions do, and what a learning result
actually establishes before scaling the system.

1. [PatchNet](patchnet.md): the revised common core, continuous observations, memory boundaries and isolated rehearsal.
2. [Quickstart](quickstart.md): the earlier `GenericBrain` composition, retained for compatibility.
3. [Experience](experience.md): connect a brain to its curriculum and test what it learns.
4. [Continuous interaction](continuous.md): feedback timing, clocks, memories and checkpoints.
5. [Records](memory.md#records): the records cortex, and what the settled regions keep.

[TemporalPatchNet](temporal.md) is the experimental temporal core in this checkout:
time-varying input paths, local overlap repair, centered detuning and isolated
continuations with persistent hidden context.

[EquilibriumActor](actor.md) provides fixed-model past inference, minimal bound
context and joint future-state/action repair through the public library.

## Build

- [Write a cortex](cortex.md): regions, projections, ports, the synapses a learning head
  owns, and when a cortex settles or records.
- [Compose a brain](brain.md): a genome developed into one brain, checked settling, one
  experience step by hand, `GenericBrain`, checkpoints and the browser page.
- [Evolve a brain](evolution.md): mutation, selection over short lives, parallel lives and
  equilibrium detuning.

## Mechanism reference

[Concepts and limits](concepts.md) · [Local learning](learning.md) ·
[Records and memory](memory.md) · [Reward](reward.md) · [Task recipes](tasks.md) ·
[Rehearsal](replay.md) · [Content memory](content_memory.md) ·
[Sequence readback](sequence.md) · [API](api.md)

## Measurement and implementation

[Convergence certificates](certificate.md) · [Protocols](protocols.md) ·
[Receipts](receipts.md) · [Brain viewer](pages.md) · [Backends](backends.md)

Formal results: [the bundled Lean library](../lean/README.md), with explicit hypotheses and implementation boundaries.
