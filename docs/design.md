# Design a brain

Start with an observable task: what the brain senses, what it can change, what an
improvement means, and how long it may think. Cadence supplies local state,
synaptic transport, feedback, records and plasticity. Your application supplies
the body, observations, targets, rewards and any model used to imagine outcomes.
The [biology map](biology.md) distinguishes those responsibilities.

## Choose the parts and their ports

Use a small number of named regions. Size each for the information it must carry;
a larger brain adds capacity and settling cost, and still needs useful inputs and feedback.

| Function | Starting construction | Capacity and contract |
|---|---|---|
| Vector sensing | `Region("sensory", inputs)` | One neuron per supplied feature; scale on training data |
| Image sensing | `visual_cortex(height, width, features=..., field=...)` | Pixels plus local receptive-field neurons; independently learned fields, no pretrained visual knowledge |
| Association | `cortex(hidden)` | A population that becomes useful through its projections and training |
| Action selection | `motor_cortex(actions)` | One action neuron per discrete choice; `Bins` represents multiple continuous axes |
| Fading working memory | `prefrontal_cortex(hidden)` and `Trace` | One trace value per source neuron per stream; wire its projection back into association |
| Addressed event memory | `FastSynapses(rule="delta")` | A key-width × value-width matrix per stream; correlated keys interfere |
| Lasting associations | `SynapticMemory` | Shared persistent synapses plus fading per-stream residuals; repetition/salience consolidate actual observed values |
| Prediction/value | A learner's output population; `ActorCritic` critic | Predict observed transitions or returns; validate on held-out episodes |
| Deliberation | `imagine` with a transition model and evaluator | Explicit branch state, horizon and node budget |
| Self-monitoring | `ActivityMonitor`, or a learned readback region | Read activity and option scores; connect its request to an actual decision budget |

Keep the **standard cortex modules as optional topology builders**. They return ordinary
`Region` objects and introduce no new neuron dynamics. Their biological names describe
a role, not a faithful cortical microcircuit. A `cortex` with no projections is only a
population; `prefrontal_cortex` alone has no memory. Replace a standard region with your
own circuit and keep its ports explicit. Avoid a mandatory catalogue of anatomical parts
that every task must pay for.

## Couple the parts

`Genome`/`develop` join regions through named projections; `assemble` joins existing
connectomes through explicit synapses. Use one `Brain` on that combined graph. During
a settling step all neurons read the previous joint activity. A motor discrepancy can
therefore affect association and perception through feedback, and their next activity
can affect the motor intention in turn. This repeated local correction seeks a common
self-consistent state. It does not guarantee a global optimum.

A direct sensory-to-motor route can shorten credit assignment for a simple reflex
while the association population learns a richer response. For a layered design,
`layered(..., skip=True, skip_init=0.0)` adds trainable direct synapses without
changing its initial predictions or existing effective weights. This improved
delayed binary choices across five seeds in the
[wiring experiment](../benchmarks/sensorimotor_skip/README.md). The conventional
score-rule control remained stronger on long-delay stochastic accuracy; test the
route on your own task instead of adding it to every design automatically.

A visual pathway with feedback to its feature population:

```python
import numpy as np
import cadence as cd
from cadence.regions import cortex, motor_cortex, visual_cortex

genome = cd.Genome(
    (visual_cortex(6, 6, features=2), cortex(12), motor_cortex(2)),
    (cd.Projection("visual", "association"), cd.Projection("association", "motor")),
)
circuit = cd.develop(genome, seed=0)
brain = cd.Brain(circuit, cd.learning_neuron_model(dt=0.5))
drive = np.zeros((1, circuit.n))
drive[:, list(circuit.populations["visual/input"])] = np.eye(6).reshape(1, -1)
result = brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-5)
print(circuit.n, circuit.synapses, result.steps, result.converged.tolist())
```

This demonstrates a supplied circuit, before training. Inspect `result.residual` and
`result.state.activation`. A reached step budget is a valid result, but it is not a
converged equilibrium. A rhythmic motor circuit may intentionally never reach a fixed
point; run it for a fixed physical tick with `settle` instead.

The auxiliary operations have different clocks. `Trace` feeds previous activity into
this solve, `FastSynapses` recalls before it and writes after a real observation, and
`ActorCritic` changes weights after reward. Their stored arrays are outside the joint
neuron equations. To put a memory population inside the joint solve, drive its recall
ports and connect those neurons to the other regions. Draw every auxiliary store and
readout in the architecture as well as the neurons; a combined picture alone is not coupling.

For a task that needs ordered history, first test whether its state carrier preserves
the relevant observations. A leaky average can erase their order. In the
[repeat-card experiment](../experiments/temporal_address/README.md), a three-stage
sensory register reads its oldest entry before shifting in the current observation.
Both patch policies and a smaller linear policy learn perfectly with that history;
masking it leaves both near chance. This register is controller state outside the
joint solve, with a task-specific lag. It is a diagnostic design pattern, not a
learned temporal address or a default memory module. Count its storage and give
comparison models the same history.

## Equilibrium, error and surprise

These quantities answer different questions:

| Quantity | Meaning | Use |
|---|---|---|
| Equation residual | Potential/adaptation disagree with the current drive and synaptic input | Decide whether the current joint solve has converged |
| Activation movement | Activity changed during a step or across observations | Display propagation and transients; it can be small under saturation |
| Teaching error | The response differs from a demonstrated target | Nudge outputs; let feedback carry a change to hidden neurons |
| Prediction surprise | An observation was unlikely under a learned prediction | An application can prioritize prediction errors or new observations |
| Reward prediction error | Reward plus discounted next value, minus current value | `ActorCritic` writes through eligibility; `Valence` can center or cap the signal |

A brain can be confidently wrong at a perfectly valid equilibrium. Teaching changes its
future response, not just the stopping tolerance. Conversely, interesting oscillations
can be task dynamics rather than a learning failure. Plot both task quality and residuals.

## Teach, practice, correct and retain

These are kinds of experience in one ongoing loop. Use
[`GenericBrain.step`](continuous.md) to receive feedback and choose the next action
without switching modes; demonstrations can enter any moment through `teacher=`.
The lower-level sequence below also supports controlled batch experiments.

1. **Design capacity and feedback.** Declare senses, actions, memory lifetimes and regions.
   Confirm a target nudge reaches the hidden regions that should learn. Verify the coupled
   dynamics before scaling capacity or training time.
2. **Imitate.** Learn teacher actions with `Learner.step` or `GenericBrain.fit`. Split data
   by episode, and evaluate imitation on new situations rather than training accuracy alone.
3. **Experiment.** Use `act`, take that action in the real environment, then `learn` from
   its reward and next observation. Each row is the same stream throughout a batch.
4. **Correct on encountered failures.** Gather demonstrations for situations the learner
   actually reaches. Finish or reset a pending reward transition before further imitation.
5. **Retain and reassess.** Rehearse earlier real episodes and evaluate both old and new
   tasks. Every game can contribute evidence; an update is not guaranteed to improve play.
   Select checkpoints by validation quality and keep a separate final test set.

A complete, small example of imitation followed by interaction:

```python
import numpy as np
import cadence as cd

rng = np.random.default_rng(0)
x = np.repeat(np.array([[1, 1, 0, 0], [0, 0, 1, 1]], float), 40, axis=0)
y = np.repeat([0, 1], 40)
x = np.clip(x + rng.normal(0, 0.1, x.shape), 0, 1)
agent = cd.GenericBrain.build(4, 2, hidden=16, episodic=True, seed=0)
agent.fit(x, y, epochs=10)
observation = x[:1]
action = agent.act(observation)
reward = (action == y[:1]).astype(float)  # replace with the actual environment outcome
agent.learn(reward, np.array([True]), x[40:41])
path = agent.save("agent.npz")
resumed = cd.GenericBrain.load(path)
assert resumed.parameters() == agent.parameters()
```

`fit`, `predict` and `accuracy` treat samples independently and bypass episodic/working
memory. `act` and `learn` use that memory. Greedy `act` updates the live working state but
creates no learning eligibility: use a separate instance for evaluation, or reset after
it. Do not reward a greedy read as though it were a sampled training decision.

`done` is a boolean vector. True rows start their next episode from rest; supply the reset
observation in `next_observations`. For a time-limit truncation, provide the old episode's
last-state value in `bootstrap`; other rows ignore that entry. Reward is a finite vector,
one entry per action. Observations always have a batch axis, even for one image or vector.

## Imagine, evaluate and revise

Train a predictive region from real transitions, or supply a known simulator. A future
needs its own state, memory, action history and random state if stochastic. It can then
simulate several actions, score consequences, and compare alternatives. `imagine` provides
bounded search, with optional adversarial pruning. It does not learn a simulator from
an empty graph. The [future-simulation patterns](patterns.md#future-simulation) show both
supplied and learned-model integration.

Keep live learned parameters frozen during candidate evaluation. `clone` can snapshot
small branch state efficiently while sharing explicitly immutable model parameters.
Replaying a predicted outcome as a real observation creates self-confirming evidence.
Use actual outcomes for correction, and measure model error as the planning horizon grows.

For review, identify a weak segment, propose replacements, evaluate the entire revised
output and accept only an improvement on the declared criterion. Human preference can
supply feedback; the critic's score is not an independent proof of beauty or correctness.

## State, checkpoints and cost

`GenericBrain.save/load` preserves the full standard composition during interaction,
including the critic, both optimizers, random generators, stream eligibility, next free
phase, working memory, consolidated/transient synapses and pending-action states.
Loading defaults to CPU and can select a different backend.
Maintain saved row identities to continue the same streams. `reset()` clears working
state and eligibility but keeps slow weights and hippocampal records. For new brains,
`agent.hippocampus.reset(batch)` clears transient residuals; `clear()` also erases
consolidated synapses. Changing batch size keeps shared consolidated knowledge and
starts fresh transient streams. `Learner.save/load` saves only its own
learned response and optimizer, and cannot save an application body or environment.

Report neuron count, directed synapse count, `parameters()`, all additional mutable arrays,
and work per decision. Fast memory consumes `batch × key_width × value_width` numbers;
eligibility consumes roughly `batch × (synapses + neurons)`, besides the critic trace.
No array size is a neuron-count equivalent for an animal brain or a measurement of energy.

Check a new design with a component ablation, shuffled cues, both warm and cold starts,
held-out task quality, and matched conventional/algorithmic controls. Start with NumPy,
then select hardware using the actual workload. A single `Brain` uses one device;
multi-GPU orchestration is application code. [Backends](backends.md) explains precision,
profiling and transport choices.
