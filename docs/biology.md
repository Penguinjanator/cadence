# From nervous systems to brains

This is a functional engineering map, not an anatomical equivalence claim or an
exhaustive model of a human brain. **Implemented** means a reusable library operation;
**pattern** means a documented composition needing application code; **partial** means
that only a restricted analogue is available. Missing functions are listed explicitly.
Shared synapses let regions seek a common equilibrium; convergence must be checked. A
**neuron** is a graded (rate) unit that can stand for one biological neuron, a
population or a task variable, and a **synapse** transports activity between
neurons; choose that resolution explicitly. Neither makes a software unit a
biological cell.

Examples refer to the [public websites](https://floatingpragma.io/cadence-examples/)
([code](https://github.com/muellerberndt/cadence-examples)).

## Objects and their biological counterparts

| Cadence object | Biological inspiration | What it actually implements |
|---|---|---|
| `Connectome` | Neural connectivity | Directed weighted contacts and named populations; topology is supplied or generated |
| `Brain` / `BrainState` | A recurrent circuit's ongoing state | Graded potentials, published activity and optional adaptation; no action potentials |
| `NeuronModel` | Input integration and firing-rate response | A leaky potential update and rebased sigmoid; the negative leak is a modeling convention |
| `Adaptation` | Activity-dependent fatigue | A slow local activity variable subtracting from drive |
| `Region` / `Projection` | Specialized areas and pathways | Named populations, ports, connection density and feedback |
| `Genome` / `develop` / `evolve` | Development and selection | Seeded graph generation and caller-scored parameter search; no gene expression model |
| `Learner` / `Nudge` | Synaptic plasticity and instructive feedback | Local free/nudged endpoint contrasts under documented assumptions |
| `Trace` / `Afterglow` / `Echo` | Fading recent activity | One leaky stored value per source neuron and stream; it enters later as drive |
| `FastSynapses` | Rapid associative memory | A bounded key/value matrix with Hebbian or delta writes; not hippocampal anatomy |
| `ActorCritic` | Reward-guided action and eligibility | Sampled actions, per-stream traces and a linear value readout outside the connectome |
| `Valence` | Signed reward modulation | Centering, scaling and capping a computed prediction-error signal; no emotion or chemistry |
| `imagine` | Prospective evaluation | Isolated bounded search using a supplied transition and evaluator |
| `ActivityMonitor` | Monitoring uncertainty and one's own activity | A six-neuron heuristic readback circuit; the caller must act on its request |
| `equilibrate` | Checking a circuit's settled state | A numerical equation-residual check with an exact work budget |

Neural prospective sequences have been observed before navigation in rats;
Cadence's search is an engineering analogy to that function, not the same mechanism.
[Pfeiffer and Foster, 2013](https://www.nature.com/articles/nature12112).
The free/nudged rule is related to [equilibrium propagation](https://arxiv.org/abs/1602.05179),
whose gradient identity depends on its model assumptions; it is not evidence that
all animal learning follows Cadence's equations.

## Sensing and moving

| Function | Support | Cadence connectome | Pattern | Example |
|---|---|---|---|---|
| Integrating input in a neuron | Implemented | Neuron potential and activation under `NeuronModel` | [Concepts](concepts.md#neurons-and-synapses) | All |
| Excitation and inhibition | Implemented | Signed synapses in `Connectome` | [Concepts](concepts.md#neurons-and-synapses) | C. elegans habitat |
| Sensory transduction | Pattern | An application encoder drives named input neurons | [Sensor, motors, body](patterns.md#sensor-opposing-motors-body) | Eye & arm retina, worm odor |
| Proprioception | Pattern | Body state returns to input ports every tick | [Sensor, motors, body](patterns.md#sensor-opposing-motors-body) | Eye & arm |
| Approach and withdrawal | Pattern | Sensory error drives an opposing motor pair that moves the body | [Sensor, motors, body](patterns.md#sensor-opposing-motors-body) | Worm, forager |
| Antagonist muscles, motor recruitment | Pattern | Positive and negative motor neurons read as a rectified difference; `Bins` for population codes | [Sensor, motors, body](patterns.md#sensor-opposing-motors-body) | Eye & arm, mouse |
| Combining senses | Pattern | Separate input populations projecting into shared neurons, trained as one learner | [A learning life](patterns.md#a-learning-life) | Eye & arm |
| Central pattern generator | Pattern | Mutual inhibition plus `Adaptation` | [Rhythm](patterns.md#rhythm) | [`half_center.py`](../examples/half_center.py) |
| Adaptation to a sustained stimulus | Implemented | `Adaptation` subtracts a slow activity trace | [Rhythm](patterns.md#rhythm) | |
| Lesions and stimulation | Implemented | Masks, `Protocol` rows and `shuffled` controls | [Protocols](protocols.md) | C. elegans circuit |

## Integration and decision

| Function | Support | Cadence connectome | Pattern | Example |
|---|---|---|---|---|
| Specialized regions exchanging signals | Implemented | Named connectomes merged by `assemble`, or `Region`, `Projection` and `develop` | [Several regions, one equilibrium](patterns.md#several-regions-one-equilibrium) | All six |
| Retinotopy and receptive fields | Implemented | `visual_cortex`: feature neurons each reading one local window of the picture | [Designed and evolved regions](patterns.md#designed-and-evolved-regions) | |
| Cortical areas and competition | Implemented | `cortex` with optional lateral inhibition; `motor_cortex` with one neuron per action | [A generic brain](patterns.md#a-generic-brain) | |
| Action selection by the basal ganglia | Partial | `GenericBrain.basal_ganglia`: an `ActorCritic` whose critic reads the association cortex and whose dopamine error drives plasticity | [A generic brain](patterns.md#a-generic-brain) | |
| Regions laid down by the genome | Implemented | `Genome` of blank and designed regions, `develop`, `mutate` and `evolve` | [Designed and evolved regions](patterns.md#designed-and-evolved-regions) | |
| Spatial map and route finding | Pattern | A spatial field region gated by a visual map, position error to directional motors | [Several regions, one equilibrium](patterns.md#several-regions-one-equilibrium) | Mouse |
| Choosing among options | Pattern | Candidate neurons settling with a value neuron and a monitor | [Future simulation](patterns.md#future-simulation) | Connect Four |
| Simulating consequences before acting | Pattern | Futures rolled forward through the brain's own predictions in isolated batch rows, scored by a critic | [Future simulation](patterns.md#future-simulation) | |
| Planning with a world model | Implemented | `imagine` over isolated copies with a supplied transition model | [Future simulation](patterns.md#with-a-supplied-world-model) | Connect Four |
| Expectation and surprise | Pattern | Transition statistics as synapses from context cues to expectation neurons | [Expectations as synapses](patterns.md#expectations-as-synapses) | |
| Self-review | Pattern | Render the draft, re-simulate its weakest part, keep only whole-draft improvements | [Review and revise](patterns.md#review-and-revise) | |
| Confidence, knowing when to think longer | Partial | `ActivityMonitor`, or a learned confidence neuron | [Reading its own activity](patterns.md#reading-its-own-activity) | Connect Four |
| Attention and salience | Partial | Supplied gaze or attention selection; `Trace(focus=...)` weights neurons that changed | [Fading context](patterns.md#fading-context) | Eye & arm |
| Leaving a rut | Pattern | Fatigue on looping choices, growing with idleness | [Restlessness](patterns.md#restlessness) | |
| Exploration and creative variation | Pattern | Softmax sampling; candidates settled under bounded random drive and ranked by a critic | [Future simulation](patterns.md#with-the-brains-own-predictions) | |

## Memory

| Function | Support | Cadence connectome | Pattern | Example |
|---|---|---|---|---|
| Recent activity | Partial | A carried `BrainState`; it keeps history only while capped or with several attractors | [Fading context](patterns.md#fading-context) | Motor potentials in the moving demos |
| Afterimage, recent history | Implemented | `Trace`, `Echo` or `Afterglow` into context neurons | [Fading context](patterns.md#fading-context) | |
| Working memory in prefrontal cortex | Partial | `prefrontal_cortex` driven by a `Trace` of the association cortex; `GenericBrain.build(..., working_memory=True)` | [A generic brain](patterns.md#a-generic-brain) | |
| Holding an item, all-or-none report | Pattern | Self-exciting neuron pairs with mutual inhibition, coupled to answer neurons | [Holding an item](patterns.md#holding-an-item) | |
| One-trial association of cue and outcome | Partial | `FastSynapses(rule="delta")`, read before acting and written after the outcome; `GenericBrain(episodic=True)` as a hippocampus | [Records in the loop](patterns.md#records-in-the-loop) | Mouse, forager, changing memory |
| Order and time since an event | Pattern | Clock or position neurons as record keys | [Records addressed by time](patterns.md#records-addressed-by-time) | |
| Skills and learned representations | Implemented | Synaptic efficacies and biases trained by `Learner` | [Learning](learning.md) | |
| Rehearsal and consolidation | Pattern | Stored real observations mixed into later updates | [A learning life](patterns.md#a-learning-life) | |
| Forgetting and interference | Pattern | Trace decay, finite record capacity, later updates | [Memory](memory.md) | Changing memory |

## Learning

| Function | Support | Cadence connectome | Pattern | Example |
|---|---|---|---|---|
| Imitation | Implemented | `Learner.step(drive, teacher_labels)` | [A learning life](patterns.md#a-learning-life) | |
| Delayed credit | Implemented | `ActorCritic` eligibility traces of free and nudged contrasts | [Reward](reward.md) | |
| Reward prediction error | Partial | Critic error shaped by `Valence` and broadcast as the dopamine signal | [Signed feedback](patterns.md#signed-feedback) | |
| Preference | Pattern | Signed nudge weight with replay and rollback, or a signed update of expectation tables with a retention guard | [Signed feedback](patterns.md#signed-feedback) | |
| Learning about its own errors | Pattern | A second learner on a monitor range | [Reading its own activity](patterns.md#reading-its-own-activity) | |

## Broader functions and missing mechanisms

These entries prevent a role name from implying a completed cognitive system.

| Function | Available analogue | What still needs implementation or evidence |
|---|---|---|
| Hearing, touch, balance, smell and interoception | Encoders driving sensory populations | Modality-specific transduction and learned representations; no ready auditory/vestibular cortex |
| Object recognition and visual scenes | Local receptive fields and coupled regions | Useful trained representations, object binding, invariance and scene benchmarks |
| Cerebellar prediction and motor calibration | Prediction learning and sensorimotor error feedback | No standard cerebellar microcircuit or general motor-learning guarantee |
| Selective attention and gating | Supplied drive/masks, lateral competition and salience | Learned attention control; a mask is not a learned thalamic circuit |
| Spatial navigation and body schema | Task-coded map, body feedback and prediction | General learned maps and transfer across unfamiliar bodies are application work |
| Episodic recollection and semantic memory | Fast records and slow learned representations | Learned addressing, autobiographical organization and consolidation that preserves earlier skills |
| Timing and sequence organization | State, traces, positional cues and adaptation | Long-horizon temporal credit and hierarchical sequence control need task-specific design |
| Motivation, emotion and homeostasis | Reward, internal-state inputs and `Valence` | No integrated affective system; a positive reward signal does not establish happiness |
| Reasoning and imagination | Learned/supplied transitions, bounded search and review | General reasoning and reliable learned simulators do not emerge automatically |
| Metacognition and an integrated self | Activity readback, learned confidence, persistent records | No validated self-model, subjective experience or consciousness measure |
| Language and social understanding | Token encodings and trainable output groups | No pretrained language comprehension, theory of mind or social cognition module |
| Sleep, replay and consolidation | Rehearsal of saved real observations | Scheduling, retention tests and replay are application code; no sleep controller |

## Biological processes outside the model

| Feature | Boundary |
|---|---|
| Spikes, ion channels, conduction delays, spike-timing plasticity | `NeuronModel` is a graded activity rule |
| Transmitter release, neuromodulator chemistry | Signed weights, adaptation and the dopamine signal are computed values |
| Glia, metabolism, blood flow | Not modeled; step counts and timings are not energy |
| Sleep | Rehearsal is application code; there is no sleep controller |
| Language, social cognition, consciousness | A map of local operations demonstrates none of these |

## Colors in the circuit viewer

The public viewer maps every declared neuron and directed synapse, with pan, zoom
and a full-screen view. Region colors identify connectome functions; they do not prove
exclusive biological specialization. Separate overlays show activity, input drive,
activity change and learned weight changes.
These are dimensionless computed values. Behavior badges label observed events such
as reaching food. Population traces plot model state over settling steps;
they are not EEG, imaging or transmitter concentrations. See the
[viewer guide](https://github.com/muellerberndt/cadence-examples/blob/main/METHODS.md#brain-colors-and-neurotransmitters).
