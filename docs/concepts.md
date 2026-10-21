# How the neural graph settles

The mechanics below [`Brain.compose`](brain.md), for building or debugging a `NeuralGraph`
or `Learner` directly. Learning rules live in [learning](learning.md) and
[reward](reward.md); [orientation](orientation.md#where-the-learning-signal-comes-from) says
where each family's teaching signal comes from.

## Neurons and synapses

A neuron holds a potential `v` and publishes an activation `s`: a graded (rate) unit for
one neuron or, at coarse resolution, a population. A directed synapse carries one neuron's
activation to another. A `Connectome` declares the synapses and named populations:
`input` neurons receive the stimulus, `output` neurons are read and taught, `hidden`
neurons carry the rest (`cd.layered` creates all three). With `Learner(reciprocal=True)`,
the default, a synapse and its reverse share one learned weight.

On each step every neuron collects its synaptic input and then updates its own state:

```text
synaptic_input[i] = sum(weights[e] * s[pre[e]] for synapses e ending at i)
v[i] += dt * (synaptic_input[i] + drive[i] + bias[i] - strength * a[i] - v[i])
s[i]  = activation(v[i])
```

The effective weight of synapse `e` is `gain * count[e] * efficacy[e] * exp(log_gain[pre[e]])`.

| `NeuralGraph` parameter | Shape | Meaning |
|---|---|---|
| `efficacy` | one per synapse | Signed strength; starts from the connectome's signs and is what learners change |
| `log_gain` | one per neuron | Scales every outgoing synapse of that neuron; set it per cell class to declare a region's gain (`g = np.zeros(C.n); g[C.populations["ln"]] = -1.0`, then `log_gain=g`); learners leave it alone |
| `bias` | one per neuron | Constant drive; starts at zero |
| `count` | on the connectome | Number of synaptic contacts |
| `gain` | on the `NeuronModel` | Global scale |

The graph's parameter arrays are read-only views; replace one by assigning a whole array
(`graph.bias = b`) or with `graph.with_parameters(...)`, which rebuilds the effective
weights. Backend, device, precision and `dense_limit`: see [backends](backends.md).

The drive is added to a neuron's input on every step. It does not clamp the potential:
synaptic input, bias and adaptation move a stimulated neuron too.
`NeuralGraph.settle(stimulus)` accepts a dense vector, a list of neurons (each driven at
`stimulus_amplitude`) or a `{neuron: level}` map (`level * stimulus_amplitude`).

A `Nudge` adds teaching drive `beta * mask * (target - s)`. With `softmax_temperature` set,
the masked neurons compete and the drive is `beta * (target - softmax(s / T))`, a cross-entropy nudge; `groups` (one integer
per neuron, `-1` for none) splits them into separate softmax groups, and a temperature
vector must be constant within each group. `weight`, one number per batch row, scales the
nudge per row; a negative weight pushes away from the target, which is how an advantage
enters reward learning. `anchor` and `anchor_gain` add an independent boundary drive
`anchor_gain * (anchor - s)` that `beta`, `mask` and `weight` do not scale, so positive and
negative phases share the same boundary. They must be supplied together; `anchor_gain` is a
nonnegative vector over neurons. In the graph update a neuron reads only its own state, its
synaptic input and its drive. Two operations read a declared group: a softmax nudge reads its
group, and key normalisation in `FastSynapses` (`normalize=True` or `rule="delta"`) reads the
whole key vector (see [standalone memory stores](records.md#reading-and-writing-directly)).

## Settling and equilibrium

`settle(stimulus, *, steps=60, state=None, mask=None, trajectory=False, nudge=None,
tolerance=None)` and `settle_batch(drive, ...)` (dense `(batch, n)` drive, same keywords)
run the update from rest or from a supplied `state`. `steps` caps the work; `tolerance`
stops early once the largest activation change in a step falls below it (`None` or zero
disables this). `mask` holds fractions in `[0, 1]`, shape `(n,)` or `(batch, n)`, and zeros
ablated neurons; pass the same `mask` and `nudge` to a settle and to its check. The
returned `BrainState` holds `v`, `activation`, `adaptation`, `steps` taken,
`activity_change` (total movement per row) and, with `trajectory=True`, the activation
after every step. It can be a transient, a fixed point or part of an oscillation.

`residual(drive, state)` returns the largest fixed-point equation error per row. Small
movement does not imply a small residual: a saturated neuron moves little while its
equation is far from satisfied. `equilibrate(drive, *, budget=512, chunk=32,
tolerance=1e-5, state=None, mask=None, nudge=None, damping=0)` continues until every row's
residual is below `tolerance` or the budget is spent, at one extra transport per chunk.
All rows advance together. It returns an `Equilibrium` with `state`, per-row `residual`,
`steps` (sweeps in this call), `converged` (residual within tolerance) and `qualified`
(converged plus any refinement checks). `damping=k` splits the budget over up to `k + 1`
attempts, each at half the previous `dt`, all checked against the original equations. A
spent budget is a result to report and act on.

Regions that share one connectome settle toward one joint fixed point under the current
drive; independent solves merged afterwards do not couple them. An equilibrium need not be
unique or stable. Carrying a state between inputs (warm start) can save work or select a
different attractor than starting from rest (cold start); test both. A unique attracting
equilibrium forgets its initial condition, so history needs a trace, a memory or a drive
that carries it.

When every neuron's absolute incoming effective weight sum is below `1 / L` (`L` the
largest activation slope) and adaptation is off, one free-phase step (no nudge) is a
contraction and [`cd.certificate(brain)`](certificate.md) bounds the distance to the unique
equilibrium.

## Activation and adaptation

`NeuronModel(dt=0.2, slope=4.0, threshold=1.5, gain=0.02, stimulus_amplitude=3.0,
adaptation=None, leak=0.0)` rebases a sigmoid so zero potential emits zero; with zero
drive, bias and adaptation the all-zero state is an exact fixed point. `dt` is the
relaxation fraction in `(0, 1]` and `steps` counts updates of the map; neither is elapsed
time. `slope` and `gain` must be positive and `leak` lies in `[0, 1]`. With `leak` above
zero a neuron below rest emits a small negative activation, no lower than `-leak`, so a
learning rule can tell a silent neuron from a dead one.

`learning_neuron_model(gain=1.0, *, slope=1.0, leak=0.1, dt=0.5, stimulus_amplitude=1.0)`
sets threshold zero for nets that learn: unit slope keeps a fan-scaled layer responsive, a
unit stimulus reads input levels roughly linearly, and the larger `dt` converges in tens of
steps. Builders start with zero bias, so many hidden neurons can sit near rest under signed
inputs; measure activation distributions on your stream and shift `bias` if needed.

`Adaptation(tau_steps=50.0, strength=1.0)` adds one variable per neuron,
`a <- a + (s - a) / tau_steps`, and subtracts `strength * a` from its drive. In mutually
inhibitory circuits it produces rhythm instead of a fixed point. It is off by default;
leave it off for equilibrium learning, and note that the certificate excludes it.

## Three state lifetimes

| State | What changes it | How to manage it |
|---|---|---|
| Potential, activation, adaptation | Settling steps | Pass `state=` to continue; omit it to start from rest |
| Trace or fast-memory matrix | Explicit activity or observation updates | Keep it through a continuing stream; reset only at a declared terminal |
| Weights, biases, consolidated associations | Learner updates, reward, `SynapticMemory.observe` | Save each owning component; evaluate on a separate snapshot |

A raw `NeuralGraph` changes no weights by settling. `Brain.step` schedules learning in a
composed brain; a custom composition owns its update calls. See
[traces](api.md#streams-cadencestream), [memory](memory.md) and
[standalone stores](records.md).

## Three clocks

| Clock | What changes | What stays fixed within the operation |
|---|---|---|
| Environment step | Observations, executed actions and carried state | Previously admitted evidence |
| Inference iteration | The interpretation of the same evidence | Evidence, parameters and the memory snapshot |
| Learning update | Parameters, memory contents or a learned address map | Which targets were observed, taught or imagined |

An environment step has the duration its simulator or body declares; solver sweeps and
`dt` have none, so measure wall time separately. An imagined future has its own step index
inside a private branch; its steps simulate environment time and count neither as
inference on the present nor as new observations. [Continuous interaction](continuous.md#defaults-and-the-thinking-clock)
states which events advance trace, eligibility, arousal and learning in `Brain`.

Assign every state update to one clock. A previous-state mixing term adds no inference
depth to a one-step read from a zero initial state, because that term is zero; more
internal steps or a deeper port are a separate computation with a separate cost.

## Prediction error is not model error

A correct probabilistic model cannot predict which random outcome occurs. A fair-coin
model assigns one half to heads; observing heads resolves this toss and gives no reason to
switch to an always-heads rule. An observation can change the current belief and the best
action with the learned dynamics unchanged: state estimation and model learning are
separate operations. The cross-entropy logit gradient `p - y` is nonzero on each toss and
has zero expectation under a calibrated prediction, so sample updates average out;
repeated systematic errors justify changing the model. Judge probabilistic predictions by
mean log loss and calibration against the declared uncertainty, and do not widen
uncertainty to make every error look acceptable.

An unexpected observation supplies evidence and possibly a teaching target; the `beta` of a
learning contrast, inference damping and the random drive below are separate quantities.

<a id="detuning-inside-a-life"></a>

## Sample candidate equilibria by drive perturbation

A settled graph gives one answer under one drive. To sample alternatives inside one
decision, repeat the drive in a batch, add a bounded random drive to the latent neurons of
each row, settle with `equilibrate`, keep the converged rows and rank their actions by a
validated read such as a table of witnessed outcomes, a [delta-rule store](records.md) or a
critic. Parameters, memories and the genome stay as they are; only this decision's drive
changes. Elsewhere in these docs "detuning" means the learning nudge (`beta`), which is a
different operation.

```python
import numpy as np
import cadence as cd

connectome = cd.layered(2, 16, 2, density=1.0, seed=1)    # cue in, action out
brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
cue_units, latent, action_units = (list(connectome.populations[k]) for k in ("input", "hidden", "output"))

witnessed = np.zeros((2, 2))             # witnessed: the action equal to the cue pays
for cue in range(2):
    for action in range(2):
        witnessed[cue, action] = float(action == cue)

cue = np.eye(2)[1]
drive = np.zeros((8, connectome.n))      # eight candidate rows, the same cue in each
drive[:, cue_units] = cue
drive[:, latent] += np.random.default_rng(0).uniform(-0.5, 0.5, (8, len(latent)))
result = brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-6)
candidates = result.state.activation[:, action_units].argmax(axis=1)
score = witnessed[1, candidates]         # the outcome each candidate was seen to bring
score[~result.converged] = -np.inf       # a row without an equilibrium offers no candidate
choice = int(candidates[np.argmax(score)])
print(candidates, score.round(2), choice)
```

A uniform draw in `[-a, a]` keeps each added drive within `a`; a Gaussian draw has no such
bound. In a composed brain build the repeated drive with `brain.stimulus([observation])`,
settle with `brain.brain.equilibrate`, perturb `brain.association_index` and read actions
at `brain.motor_index`; `brain.basal_ganglia.value(result.state)` ranks the same rows by the
critic. Perturbation proposes; the ranking only helps as far as
the table, store or critic has learned consequences.

## Controls and when to add machinery

A [protocol](protocols.md) declares stimuli, readouts, interventions and predicates. A
[shuffled connectome](protocols.md#the-shuffled-control) tests whether a response depends on
the particular connections; gain selection can
[cap the active fraction](protocols.md#gain-selection-under-a-sparsity-cap) so high gains do
not saturate an excitatory circuit.
`cd.conformance(brain, stimulus)` compares a trajectory with the neuron-by-neuron
reference. A [receipt](receipts.md) binds stored results to their sources.

Settling, memory and learning are separate operations because their state lifetimes
differ. A new mechanism needs a failure it fixes and an ablation on the same inputs,
budget and seeds, with its state and cost counted.
