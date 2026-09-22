# Cadence documentation

Cadence researches general intelligence through overlap consensus, equilibrium
detuning and functional self-reflection. One continuing system should acquire
skills, retain useful experience, imagine alternatives and repair its behavior
across applications. The guides distinguish implemented operations from that
broader research goal.

## Two primitives

- The **settling patch**: bounded local state, ports, an equilibrium under
  constraints, learned by the contrast of a free and a nudged settle. Built from a
  genome of regions (`Genome`, `develop`, `Brain`, `Learner`) or as one temporal
  patch (`TemporalPatchNet`).
- The **record patch** (`RecordPatchNet`): a gated linear context with a record
  store inside the patch. By day an observation is written once; by night the
  slow weights learn from the store's own dreams (`sleep`).

Everything else composes these two through ports. The [belief patch](belief.md)
is the composition toward a learned world model: a transition under action, a
repair of the belief by iteration with the store read inside it, and private
imagination; its input port reads grids through [maps](record-patch.md#maps-a-structured-input-port).

## Start here

1. [Quickstarts: three kinds of brains](quickstart.md): a record patch that learns a stream and sleeps, a settling brain that decides, a temporal patch that plans.
2. [Architecture and integration](architecture.md): state, ports, repair and the scope of each component.
3. [The record patch](record-patch.md): one-shot records, categorical ports, a store narrower than its port, maps at the port, two patches in depth, acquisition in two phases.
4. [The belief patch](belief.md): a transition under action, evidence repair by iteration, the store inside the repair, imagination that consumes no observation.
5. [Temporal learning](temporal.md): observed paths, persistent context, local detuning and isolated imagination.
6. [Private planning](planning.md) and [learn, act and observe](interaction.md): repair continuous controls under the learned model, end to end.
7. [Response protection](temporal-memory.md): conditional retention and well-conditioned learning in remaining directions.
8. [Creativity and self-reflection](creativity.md): novel proposal evaluation, recursive readback and transfer as research requirements.

[Task design](task-design.md), [common missteps](missteps.md) and
[scaling](scaling.md) say how to validate observations, actions, learning and
retained behavior before scaling. The [API reference](api.md) lists every
interface. [Experimental fixed connectivity](partitioned.md) supplies routing
constraints for comparisons, not learned specialization.

## Building settling brains from regions

[Write a cortex](cortex.md), [compose a brain](brain.md), [evolve a brain](evolution.md),
[local learning](learning.md), [records and memory](memory.md), [reward](reward.md)
and [concepts](concepts.md): regions, projections, the neuron model, the learner, the
records cortex and reward-weighted learning.

[EquilibriumActor](actor.md) is a separate fixed linear-model example of factual
inference, compressed past context and joint future-state/action repair; its exact
compression assumptions do not extend to arbitrary changing models.

## Kept for existing experiments

These compositions have their own state and learning contracts and are kept for
the experiments that used them: [PatchNet](patchnet.md), the `GenericBrain` loop in
[continuous interaction](continuous.md) and [experience](experience.md),
[task recipes](tasks.md), [rehearsal](replay.md), [content memory](content_memory.md)
and [sequence readback](sequence.md).

## Measurement and mathematical scope

[Convergence certificates](certificate.md) · [Protocols](protocols.md) ·
[Receipts](receipts.md) · [Brain viewer](pages.md) · [Backends](backends.md) ·
[Conditional Lean proofs](../lean/README.md).

Worked applications with their receipts and checks live in the [examples repository](https://github.com/muellerberndt/cadence-examples).
Examples illustrate a general mechanism. Their scores are evidence for the
named task, not proof that the architecture solves arbitrary problems.
