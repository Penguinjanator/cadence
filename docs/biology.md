# From nervous systems to patch nets

This map points from a nervous-system function to the Cadence wiring that
implements it and to the [pattern](patterns.md) that shows how to build it. It is
a functional engineering map. An **owner** can stand for a neuron, a population or
a task variable, and a **seam** transports activity between owners; choose that
resolution explicitly. Neither makes a software unit a biological cell.

Examples refer to the [public websites](https://floatingpragma.io/cadence-examples/)
([code](https://github.com/muellerberndt/cadence-examples)).

## Sensing and moving

| Function | Cadence wiring | Pattern | Example |
|---|---|---|---|
| Integrating input in a neuron | Owner potential and activation under `GradedRule` | [Concepts](concepts.md#owners-and-seams) | All |
| Excitation and inhibition | Signed seams in `Wiring` | [Concepts](concepts.md#owners-and-seams) | C. elegans habitat |
| Sensory transduction | An application encoder drives named input owners | [Sensor, motors, body](patterns.md#sensor-opposing-motors-body) | Eye & arm retina, worm odor |
| Proprioception | Body state returns to input ports every tick | [Sensor, motors, body](patterns.md#sensor-opposing-motors-body) | Eye & arm |
| Approach and withdrawal | Sensory error drives an opposing motor pair that moves the body | [Sensor, motors, body](patterns.md#sensor-opposing-motors-body) | Worm, forager |
| Antagonist muscles, motor recruitment | Positive and negative motor owners read as a rectified difference; `Bins` for population codes | [Sensor, motors, body](patterns.md#sensor-opposing-motors-body) | Eye & arm, mouse |
| Combining senses | Separate input sets projecting into shared owners, trained as one learner | [A learning life](patterns.md#a-learning-life) | Eye & arm |
| Central pattern generator | Mutual inhibition plus `Adaptation` | [Rhythm](patterns.md#rhythm) | [`half_center.py`](../examples/half_center.py) |
| Adaptation to a sustained stimulus | `Adaptation` subtracts a slow activity trace | [Rhythm](patterns.md#rhythm) | |
| Lesions and stimulation | Masks, `Protocol` rows and `shuffled` controls | [Protocols](protocols.md) | C. elegans circuit |

## Integration and decision

| Function | Cadence wiring | Pattern | Example |
|---|---|---|---|
| Specialized regions exchanging signals | Named wirings merged by `couple`, or `Region`, `Projection` and `grow` | [Several regions, one equilibrium](patterns.md#several-regions-one-equilibrium) | All six |
| Spatial map and route finding | A spatial field region gated by a visual map, position error to directional motors | [Several regions, one equilibrium](patterns.md#several-regions-one-equilibrium) | Mouse |
| Choosing among options | Candidate owners settling with a value owner and a monitor | [Imagined futures](patterns.md#imagined-futures) | Connect Four |
| Imagining consequences | `imagine` over isolated copies with a supplied transition model | [Imagined futures](patterns.md#imagined-futures) | Connect Four |
| Confidence, knowing when to think longer | `ActivityMonitor`, or a learned confidence owner | [Reading its own activity](patterns.md#reading-its-own-activity) | Connect Four |
| Attention and salience | Supplied gaze or attention selection; `Trace(focus=...)` weights owners that changed | [Fading context](patterns.md#fading-context) | Eye & arm |
| Leaving a rut | Fatigue on looping choices, growing with idleness | [Restlessness](patterns.md#restlessness) | |
| Exploration and creative variation | Softmax sampling; detuned candidate settlements ranked by a critic | [Rehearse, select, revise](patterns.md#rehearse-select-revise) | |

## Memory

| Function | Cadence wiring | Pattern | Example |
|---|---|---|---|
| Recent activity | A carried `SettledState`; it keeps history only while capped or with several attractors | [Fading context](patterns.md#fading-context) | Motor potentials in the moving demos |
| Afterimage, recent history | `Trace`, `Echo` or `Afterglow` into context owners | [Fading context](patterns.md#fading-context) | |
| Holding an item, all-or-none report | Self-exciting owner pairs with mutual inhibition, coupled to answer owners | [Holding an item](patterns.md#holding-an-item) | |
| One-trial association of cue and outcome | `FastSeams(rule="delta")`, read before acting and written after the outcome | [Records in the loop](patterns.md#records-in-the-loop) | Mouse, forager, changing memory |
| Order and time since an event | Clock or position owners as record keys | [Records addressed by time](patterns.md#records-addressed-by-time) | |
| Skills and learned representations | Seam scales and biases trained by `Learner` | [Learning](learning.md) | |
| Rehearsal and consolidation | Stored real observations mixed into later updates | [A learning life](patterns.md#a-learning-life) | |
| Forgetting and interference | Trace decay, finite record capacity, later updates | [Memory](memory.md) | Changing memory |

## Learning

| Function | Cadence wiring | Pattern | Example |
|---|---|---|---|
| Imitation | `Learner.step(drive, teacher_labels)` | [A learning life](patterns.md#a-learning-life) | |
| Delayed credit | `ActorCritic` eligibility traces of settlement contrasts | [Reward](reward.md) | |
| Reward prediction error | Critic error shaped by `Valence` | [Signed feedback](patterns.md#signed-feedback) | |
| Preference | Signed nudge weight with replay and rollback | [Signed feedback](patterns.md#signed-feedback) | |
| Learning about its own errors | A second learner on a monitor range | [Reading its own activity](patterns.md#reading-its-own-activity) | |

## Without a counterpart

| Feature | Boundary |
|---|---|
| Spikes, ion channels, conduction delays, spike-timing plasticity | `GradedRule` is a graded activity rule |
| Transmitter release, neuromodulator chemistry | Signed weights, adaptation and prediction errors are computed values |
| Glia, metabolism, blood flow | Not modeled; step counts and timings are not energy |
| Sleep | Rehearsal is application code; there is no sleep controller |
| Language, social cognition, consciousness | A map of local operations demonstrates none of these |

## Colors in the circuit viewer

The public viewer colors activity, input drive, repair and learned weight changes.
These are dimensionless computed values. Behavior badges label observed events such
as reaching food. Population traces plot model state over settlement iterations;
they are not EEG, imaging or transmitter concentrations. See the
[viewer guide](https://github.com/muellerberndt/cadence-examples/blob/main/METHODS.md#brain-colors-and-neurotransmitters).
