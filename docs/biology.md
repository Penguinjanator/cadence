# From nervous systems to patch nets

Use this map to choose a Cadence mechanism for a nervous-system function. It
covers local circuits, sensing and movement, memory, learning, and imagining
futures. It is a functional engineering map, not an exhaustive account of biology
or a claim that Cadence reproduces a brain.

A Cadence patch is bounded software state with input/output ports, readback,
records, and feedback. An **owner** can represent a neuron, a population, or a
task variable; choose that resolution explicitly. A **seam** transports activity
between owners. Neither correspondence makes the software unit a biological cell.

The tables distinguish three levels:

- **Primitive:** an implemented core operation, with the limits stated in its row.
- **Composition:** an architecture using core operations and an encoder, body,
  controller, or experience store. A proposed composition still needs testing.
- **Not modeled:** the biological mechanism has no direct current implementation.

## Local circuits: start with C. elegans

The worm is a useful starting point for sensory, interneuron, motor, and body
connections. Its connectome distinguishes chemical synapses and gap junctions;
an anatomical graph alone does not specify Cadence's effective weights or update
rule. See [Cook et al., whole-animal connectomes (2019)](https://www.nature.com/articles/s41586-019-1352-7).
Loading that graph is a circuit model, not evidence that every worm behavior has
been reproduced.

| Biological structure or function | Cadence counterpart | Level | What the mapping supports and where it stops |
|---|---|---|---|
| Integrating signals in a neuron | Owner potential `v`, published activation `s`, `GradedRule` | Primitive | Leaky integration and nonlinear response. Values and steps have no biological units unless calibrated. |
| Excitatory and inhibitory transmission | Signed directed edges in `Wiring` | Primitive | Positive and negative contributions to a recipient's inbox. Signs and strengths must be supplied or learned; anatomy alone does not establish them. |
| Electrical coupling through gap junctions | No dedicated junction rule | Not modeled | Two reciprocal edges transport activity but do not automatically implement the voltage-difference current of an electrical junction. |
| Sensory transduction: touch, chemicals, temperature, light | An encoder drives named input owners | Composition | The environment adapter converts a stimulus into numbers. Cadence does not implement the molecular receptor. |
| Interneurons combining sensory signals | Recurrent paths and named populations in `Wiring` | Primitive | Local integration and feedback; their behavioral meaning depends on the supplied circuit and input encoding. |
| Rapid withdrawal or approach | Sensory drive → settlement → motor readout | Composition | A small sensorimotor loop can express a reaction. A worm-specific reflex needs stimulus, lesion, and behavioral validation. |
| Comparing recent chemical or thermal conditions while moving | Input `Trace`/`Afterglow` plus a body/environment loop | Composition | A candidate architecture for temporal comparison during navigation, not a demonstrated model of worm chemotaxis or thermotaxis. |
| Proprioception: movement feeds back into the nervous system | Body position or stretch encoded through input ports | Composition | The simulated body must return its state after an action. Recurrent neural connections alone do not provide body feedback. |
| Alternating or rhythmic motor activity | Mutual inhibition plus `Adaptation` | Composition | The [half-center example](../examples/half_center.py) demonstrates an oscillator. A locomotion or feeding rhythm additionally needs the appropriate muscles, mechanics, and validation. |
| Motor output and muscle recruitment | Output activations, optionally decoded with `Bins` | Composition | Produces commands for an actuator adapter. Muscle force, fatigue, and body mechanics are external. |
| Reduced response during sustained stimulation | `Adaptation` subtracts a slow activity-dependent term | Primitive | A transient adaptation mechanism. This alone does not establish biological habituation or its learning mechanisms. |
| Testing circuit necessity with lesions | Wiring interventions, `Protocol`, `Row`, readouts and controls | Composition | Measures the supplied model's responses to perturbations. [Protocols](protocols.md) explain the experimental interface. |

Body feedback matters: experiments locate a proprioceptive role in the worm's
motor circuit. This motivates a closed body–brain loop rather than equating a
neural oscillator with locomotion. [Wen et al. (2012)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3508473/)

Light sensing is also different from image vision. The worm's LITE-1-mediated
light response is not the pixel-processing task used in Pong.
[Gong et al. (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5388352/)

## Perception, reaction, and coordinated action

| Biological function | Cadence counterpart | Level | What the mapping supports and where it stops |
|---|---|---|---|
| Visual input | Pixel encoder feeding input owners | Composition | The historical Pong experiment uses one current 12 × 16 frame. This is an observation interface, not a model of the retina or visual cortex. |
| Motion-sensitive visual/motor coordination | Current pixels plus input `Afterglow`, connected to action owners | Composition | Pong learns paddle actions with recent sensory history. Afterglow is a decaying trace, not an explicit velocity sensor. |
| Combining several senses | Separate named input sets projecting into shared owners | Composition | Supports a multisensory architecture; alignment, normalization, and task learning remain the application's responsibility. |
| Selecting an action | Output readout; legal-action mask; softmax sampling or `Bins` | Composition | Turns preferences into discrete or population-coded commands. Masks and actuator constraints belong to the environment adapter. |
| Orienting toward changing input | Change-sensitive weighting through `Trace.focus`; `Afterglow` defaults to nonzero focus | Primitive | A simple salience heuristic. It does not supply full selective attention, object tracking, or a learned allocation of computation. That Pong run sets `focus=0`. |
| Trying different actions | Stochastic action sampling in `ActorCritic` | Primitive | Enables exploration under a reward objective. Curiosity, information-seeking goals, and exploration curricula require additional design. |

See the runnable [Pong example](https://github.com/muellerberndt/cadence-examples/tree/7302f2af3dc0638bbafd1da1446ed96ffabaa9dd/04_pong)
and the [embodiment recipe](embodied.md). The same vocabulary can describe a fly
controller, but that does not supply fly-specific anatomy, sensory coding, or
flight mechanics automatically.

## Memory and learning

Biological memory has several mechanisms and timescales. A synaptic working-memory
theory is one example of why “short-term memory” should not be assigned to a
single computational component. [Mongillo, Barak and Tsodyks (2008)](https://barak.net.technion.ac.il/files/2012/11/synapticmemory.pdf)
Long-lasting changes in synaptic efficacy motivate a separate analogy with learned
parameters. [Bliss and Lømo (1973)](https://pubmed.ncbi.nlm.nih.gov/4727084/)
Neither correspondence identifies Cadence's update rule with biological plasticity.

| Biological function | Cadence counterpart | Level | What the mapping supports and where it stops |
|---|---|---|---|
| Reverberating recent activity | Carry a `SettledState` into the next settlement (`Learner.free(..., warm=state)`) | Primitive | Reuses recurrent activity. A unique attracting equilibrium under fixed input eventually erases initial-state differences; warm starts alone do not guarantee memory. |
| A sensory afterimage or short-term activity trace | `Trace`, `Echo`, `Afterglow` | Primitive | Retains a decaying history and injects it through context ports. Update once per real observation and reset independent episodes. |
| Keeping and manipulating a task-relevant item | Trace or `FastSeams` plus read/write gates and a controller | Composition | A working-memory architecture. Maintenance, selection, and manipulation must be designed and evaluated; a passive trace alone does not do all three. |
| Rapid association between a cue and an outcome | `FastSeams(rule="delta")` | Primitive | Stores residual-corrected key/value associations. Keys, values, and write gates are explicit; correlated keys interfere and capacity is finite. See [memory](memory.md). |
| Long-term skills and learned representations | Edge scales and biases changed by `Learner` | Primitive | Reuses learned responses across episodes. It does not automatically yield semantic understanding or immunity to forgetting. |
| Remembering particular experiences | An external episode store plus encoding and retrieval | Composition | The historical game experiments save experience for rehearsal. A log is not an automatic autobiographical memory system. |
| Learning by imitation | `Learner.step(drive, teacher_labels)` | Primitive | Nudged phases teach a response from a demonstration. The teacher, coverage, and sufficient circuit capacity must be supplied. |
| Assigning delayed reward to earlier actions | `ActorCritic` eligibility traces | Primitive | Retains local settlement contrasts, then applies a reward prediction error. This is a computational three-factor learning analogue. |
| Better or worse outcomes than expected | Critic prediction error; optional `Valence` normalization and quiet band | Primitive | Signed feedback can strengthen or weaken eligible responses. It is not a simulation of dopamine chemistry, pleasure, or pain. |
| Rehearsal and consolidation | Replay stored observations through `Learner` while retaining old examples | Composition | Can reinforce or repair learned responses. The historical games implement rehearsal; no generic sleep or consolidation controller is included. |
| Forgetting and interference | Trace decay, finite fast-memory capacity, later parameter updates | Primitive | Different mechanisms lose information differently. Stable lifelong learning needs retention tests and an explicit rehearsal or protection strategy. |

The reward analogy concerns **prediction error**: an expected reward can produce
a smaller error than a surprising reward. A negative error means worse than
predicted, not necessarily a painful event. See
[Schultz, Dayan and Montague (1997)](https://pubmed.ncbi.nlm.nih.gov/9054347/)
and Cadence's [reward guide](reward.md).

Replay of behavioral sequences has been observed in hippocampal recordings.
That motivates testing rehearsal; saving and replaying a software episode does
not reproduce the biological process.
[Foster and Wilson (2006)](https://www.nature.com/articles/nature04587)

### Which memory survives what?

| Record | Lifetime and persistence |
|---|---|
| `SettledState` | Lives while the caller retains it; pass it explicitly to continue activity. |
| `Trace` / `Afterglow` | Lives in the stream object, decays when updated, and clears on reset. Separate from a learner checkpoint. |
| `FastSeams` | Lives in the fast-memory object; its contents are separate from a learner checkpoint. |
| Learned parameters | `Learner.save` / `load` preserve the learned circuit and its training state. |
| Episode history or imagined branches | Owned by the application; persistence, replay policy, and isolation must be explicit. |

For `focus=0`, the trace update is `trace = decay * trace + (1 - decay) * activity`.
Its timescale is measured in **updates**, not automatically biological seconds.
Warm recurrent state, an explicit trace, a fast association, and a learned weight
are distinct records; use the [small-core guide](condense.md) to choose among them.

## Imagining futures and organizing a brain

Prospective neural sequences at a choice point motivate comparing possible
paths before acting. This evidence comes from rat recordings; it is not evidence
that a worm, every neural circuit, or Cadence's components perform the same kind
of planning. [Johnson and Redish (2007)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6673267/)

| Biological function or organizing idea | Cadence counterpart | Level | What the mapping supports and where it stops |
|---|---|---|---|
| Predicting what an action will cause | A supplied transition model, or an application-trained world model | Composition | Cadence currently supplies no general learned world model. A game's exact rules provide information that a learner may not have inferred. |
| Imagining several possible futures | Isolated state copies and bounded rollouts in the [deliberation pattern](deliberation.md) | Composition | `compare_futures` is example code, not a core API. Compare branches without changing the live state or treating imagined events as real experience. |
| Comparing anticipated good and bad outcomes | A learned `Evaluator`, rewards, and an action-selection rule | Composition | The deliberation example trains a terminal evaluator. Ranking quality depends on both the predictor and evaluator; a wrong transition model can defeat it. |
| Revising expectations after acting | Execute the first action, observe the real outcome, update the evaluator or policy | Composition | The deliberation tests include learning from an unexpectedly bad outcome. Imagined reward and measured reward must remain distinguishable. |
| Specialized regions and communication paths | `Region`, `Projection`, `Constitution`, `grow` | Primitive | Declares populations and their wiring. Functional specialization requires appropriate inputs, learning, and evaluation. |
| Selecting a brain's size and layout | Explicit architecture choices, optionally external fitness search with `evolve` | Composition | Supports testing capacity and connectivity. This is not a model of biological development, synaptic pruning, or evolution within an individual. |

The historical Connect Four experiment combines supplied game-rule search with a learned policy
for breaking ties. Measure the raw policy, search alone, and their combination:
those controls show what each component contributes. The small deliberation
example separately tests branch isolation, a learned evaluator, and failure
under a wrong world model. Neither example establishes a general planning
advantage over other architectures.

## Biological mechanisms without direct counterparts

| Feature | Current boundary |
|---|---|
| Action potentials, ion channels, conduction delays, spike-timing-dependent plasticity | `GradedRule` is a graded activity rule, not a spiking or compartmental neuron model. |
| Receptors, transmitter release, multiple neuromodulators | Signed weights, adaptation, and reward errors are computational abstractions, not biochemical simulations. |
| Glia, myelin, vascular supply, metabolism | No direct model; arithmetic counts and timings do not establish biological energy efficiency. |
| Sleep stages and autonomous dream generation | Rehearsal and deliberation are application patterns; there is no current generic sleep/dream controller. |
| Language, social cognition, self-modeling, consciousness | No established full counterpart. A mapping of local operations does not demonstrate these capabilities. |

## Brain colors and chemical signals

The public circuit viewer can color actual activity, input drive and repair, or
highlight learned weight changes. These are dimensionless computational values,
not neurotransmitter concentrations. A signed reward prediction error can serve
as a modulatory learning signal without simulating dopamine chemistry. The current
showcase does not expose dopamine, serotonin, glutamate or GABA concentration fields.

The viewer groups actual owners by computational function. Behavior badges label
observed events such as food consumption or reaching a goal, not feelings.
Its fading repair trails are display history; input-release probes show recurrent
decay in an isolated copy. The moving demos also carry graded motor potentials between control ticks. This
is transient neural state; it does not establish learned working-memory behavior.
Real fMRI commonly measures blood-oxygen-related changes; PET investigates
selected molecular targets with specific tracers. Neither is a generic live map of all neurotransmitters. See
[the viewer's imaging explanation and sources](https://github.com/muellerberndt/cadence-examples/blob/main/METHODS.md#brain-colors-and-neurotransmitters).

## Build a task from the map

Choose the observation and action ports, body loop, memory timescale, and enough
capacity first. Teach useful responses through imitation, then let the agent
try actions and learn from their outcomes. Add demonstrations for recurring
failures and rehearse retained experience as new situations arrive. Add future
comparison when a usable transition model and evaluator are available.

Keep these as compositions of simple operations: biological inspiration does
not require a new core class for every biological function. The
[task recipes](tasks.md), [games guide](games.md), and
[deliberation example](../examples/deliberation.py) show where to begin.

## Reusable task wiring and self-reading

The optional [`couple`](brains.md#different-regions-one-equilibrium) helper puts
named regions in one connected wiring: local readback and repair seek a joint
fixed point for the current sensory boundary. This is an engineered composition,
not evidence that biological nervous systems always settle or optimize globally.

The optional [task brains](brains.md) compose motor populations, isolated future
branches and an activity monitor. The monitor reads the controller's own changing
candidate values and can request deeper reasoning. This is a testable
metacognitive feedback loop, not evidence of consciousness or a claim that humans,
apes and insects divide according to one particular recursive module.

The public viewer now exposes signed population activity and equation-mismatch
traces. They are observations of the software model across settlement iterations;
they are not EEG, inferred transmitter concentrations or evidence of consciousness.
A measured oscillation and a declining residual are different possible dynamics.
The viewer shows either honestly, and labels time expansion.
