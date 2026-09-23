<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: connected patches with local state, readback and repair" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/cadence/) · [Examples](https://github.com/muellerberndt/cadence-examples) · [Paper](https://github.com/muellerberndt/cadence/blob/main/cadence-paper.pdf) · [PyPI](https://pypi.org/project/cadence-net/) · [Documentation](https://github.com/muellerberndt/cadence/blob/main/docs/index.md) · [Changelog](https://github.com/muellerberndt/cadence/blob/main/CHANGELOG.md)

**Research toward general intelligence through overlap consensus, equilibrium detuning and self-reflection.**

Cadence's goal is a continuing learning system with the flexibility of animal
and human problem solving: acquiring skills from experience, retaining useful
knowledge, imagining alternatives and creating solutions across domains.
The mission is to find the smallest persistent state and local update rule
that can support these abilities. The building block stays as simple as possible,
like in nature; every part of a brain answers with a settled state of that one
rule; and where a choice appears, evolution across lives is preferred to design. General intelligence is the research goal;
the current library establishes bounded learning, memory and control results.

The building block is a bounded, observer-like patch with local state,
ports, records, readback and repair. A disturbance exposes disagreement. The
network can explore a possible response, test it against actual consequences
and settle into a revised organization. We seek fewer mechanisms that solve
more problems.

## Four shared principles

- **Overlap consensus:** patches repair disagreement across their shared
  boundaries. The resulting equilibrium is an internally consistent model;
  its predictions still have to agree with experience.
- **Equilibrium detuning:** observed outcomes perturb that equilibrium.
  Local positive/negative contrasts change learned relationships; the same
  operation can adjust proposed actions while holding the model fixed.
- **Self-readback:** a patch's proposed actions and predicted consequences are
  available to the same brain and tested by its next observation. Physical
  readback is evidence; an imagined outcome never is.
- **Metacognition as recursive self-observation:** a patch of the same kind reads
  the beliefs, residuals and surprises of the rest of the brain through ordinary
  ports, and its settled state steers them: which evidence counts, where a sense
  samples, what a habit holds, how long a repair runs. Because it is bounded it
  cannot steer everything at once, so it must select what matters for what it is
  computing; that selection is attention. Because it is a patch it can be read in
  turn, and the levels form a ladder from reflex to a robot that acts and speaks
  among people. The genome decides which readbacks exist, and a rung is earned
  only by a task the brain below it fails at matched information and compute.
  The [metacognition ladder](docs/METACOGNITION_LADDER.md) states the rungs, their
  experiments and their issues.

The current implementation provides detached self-readback and private proposal
revision. Learned steering patches, curiosity and reliable
creativity are hypotheses with their tests on the ladder. [Creativity and self-reflection](https://github.com/muellerberndt/cadence/blob/main/docs/creativity.md)
defines these goals and their behavioral tests.

## Equilibrium world models

Cadence is being developed toward **evolving equilibrium world models**: brains
whose internal representation of an actor in its world grows more accurate and
expressive through experience. The representation should carry what is happening,
what persists out of sight, what the actor controls and what its actions could
cause. It need not describe those relationships in words to use them.

An equilibrium here need not mean motionless activity. A skilled actor can
follow a coherent, changing trajectory of perceptions, expectations and actions.
When events unfold as expected, the carried state should already be close to a
useful interpretation of the next moment. Familiar danger can prompt a learned
response immediately. Extra inference is needed when competing interpretations
or consequential choices warrant it.

The proposed mechanism is recursive composition through bounded, observer-like,
self-reading patches. Scene, body, candidate action and retrieved experience
meet through learned nonlinear ports. One interpretation can become input to
another, allowing the system to revise its understanding before acting. Actual
evidence anchors that revision; changing the interpretation and learning new
relationships are distinct operations. A random outcome can require a new
response while remaining consistent with a correctly learned probability
distribution. Teaching detuning uses a specified target to compute a learning
signal; it is not synonymous with surprise.

Skilled demonstrations and the actor's own actions supply complementary
experience. Watching reveals useful behavior and situations; acting tests what
controls actually cause. The same learned relationships should support private
branches that explore possible futures without altering factual memory. Plans
must be judged by subsequent real outcomes. Grounded replay and sleep should
then consolidate reliable relationships and successful decisions into cheaper
habits, while unfamiliar situations can reopen deliberation.

This is the architectural vision. Current APIs provide components and bounded
demonstrations, not the complete evolving world model. Each successive release
is intended to take a concrete step toward this goal. Progress must be shown in
prediction, retention, useful internal planning and behavior at declared
resource budgets; a new version alone does not establish it. The
[equilibrium world-model guide](docs/equilibrium-world-models.md) explains the
mechanism, the distinction between evidence repair and learning, and the current
implementation boundaries.

## Current library

Two primitives, composed through ports, and the belief patch that composes them toward a world model. The library requires Python 3.11+ and NumPy:

```bash
python -m pip install cadence-net
```

| Primitive | What it supplies |
| --- | --- |
| The settling patch: [TemporalPatchNet](https://github.com/muellerberndt/cadence/blob/main/docs/temporal.md), or a [brain of regions](https://github.com/muellerberndt/cadence/blob/main/docs/brain.md) | Local repair of observed paths, persistent context, private imagination, [continuous planning](https://github.com/muellerberndt/cadence/blob/main/docs/planning.md) and [protected responses](https://github.com/muellerberndt/cadence/blob/main/docs/temporal-memory.md); learning by the contrast of a free and a nudged settle. |
| The record patch: [RecordPatchNet](https://github.com/muellerberndt/cadence/blob/main/docs/record-patch.md) | A gated linear context with a record store inside the patch: an observation is written once by day, and by night the slow weights learn from the store's own dreams (`sleep`), with nothing outside the patch consulted. Categorical ports, batched writes, a store narrower than its port and a two-patch stack. One pass of writes, with no gradient, gives a small grammar for 0.8 of its never-taught combinations; one night lifts the slow weights alone to 1.0. |
| The belief patch: [BeliefPatch](https://github.com/muellerberndt/cadence/blob/main/docs/belief.md) | A belief carried by a learned transition under the executed action and repaired by a few iterations of one nonlinear map with the record store read inside it; imagination that consumes no observation. The composition toward a learned world model, trained with the [imagination loss](https://github.com/muellerberndt/cadence/blob/main/docs/belief.md#training-the-transition-the-imagination-loss) so the transition carries the belief. |

Start with the [quickstarts](https://github.com/muellerberndt/cadence/blob/main/docs/quickstart.md):
a record patch that learns a stream and sleeps, a settling brain that decides, and a
temporal patch that learns a consequence and plans. Each of them also runs in your browser,
trained in front of you in seconds from fixed seeds, with every neuron and synapse animated,
the distance from equilibrium as a heat on the neurons, the last change on the synapses, the
learning plotted as it is measured, and a line of text for each phase:

```bash
python -m pip install cadence-net
cadence-demo stream     # a record patch learns a stream, remembers in one shot, and sleeps
cadence-demo decide     # a settling brain decides
cadence-demo body       # a temporal patch learns a consequence and plans
```

Nothing is hosted and there is no checkpoint; [the quickstarts in your browser](https://github.com/muellerberndt/cadence/blob/main/docs/demos.md)
says what equilibrium means in each brain and what detuning it buys. The
[architecture guide](https://github.com/muellerberndt/cadence/blob/main/docs/architecture.md) maps each capability to its API and current scope;
[EquilibriumActor](https://github.com/muellerberndt/cadence/blob/main/docs/actor.md) is a separate fixed linear-body component with exact Gaussian history compression.

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

The [PatchNet graph interface](https://github.com/muellerberndt/cadence/blob/main/docs/patchnet.md),
`GenericBrain`, content memory, rehearsal and sequence readback are kept for the
experiments that used them; they are distinct compositions, not parts of the two
primitives. [API reference](https://github.com/muellerberndt/cadence/blob/main/docs/api.md).

Development installs use `python -m pip install -e .`.
[Optional backends](https://github.com/muellerberndt/cadence/blob/main/docs/backends.md)
apply to their documented graph APIs; the temporal implementation is NumPy.
Pin a release or exact commit for reproducible work. MIT licensed.

## Origins

Cadence started as an offshoot of a physics theory,
[Observer Patch Holography](https://github.com/FloatingPragma/observer-patch-holography),
which models reality itself as a metaphorical brain: a distributed network of
observers that finds global equilibria by repairing local conflict. The two
projects complement each other and share many of their theorems.
