# Concepts

A Cadence brain is software state organised into neurons, synapses and regions,
with readouts, records and a learning update. It is a design for computation. It
does not ascribe experience to the software or establish a biological brain model.

In OPH terms this is an observer-like, self-reading design: local state, declared
synaptic boundaries, readback, records and feedback, with protocols and source-bound
receipts for evidence. The biological names describe computational roles.

The [biology-to-Cadence map](biology.md) connects nervous-system functions to
these operations, from sensory reactions to memory and imagining futures.

## Neurons and synapses

A *neuron* holds a potential `v` and publishes an activation `s`. It is a graded
(rate) unit and can stand for one biological neuron or, at coarse resolution, a
population. A directed *synapse* carries one neuron's activation to another. A
*reciprocal synapse pair* is a synapse and its reverse sharing one learned weight
(`Learner(reciprocal=True)`, the default). `Connectome` declares the synapses and
named populations of neurons, such as `input` and `output`.

`Brain` holds the connectome, the neuron model and the parameter arrays. On each
step it collects every neuron's synaptic input, then each neuron updates its own
state:

```text
synaptic_input[i] = sum(weights[e] * s[pre[e]] for synapses e ending at i)
v[i] += dt * (synaptic_input[i] + drive[i] + bias[i] - strength * a[i] - v[i])
s[i]  = activation(v[i])
```

The adaptation term is absent unless enabled. The effective weight of synapse `e`
is `gain * count[e] * efficacy[e] * exp(log_gain[pre[e]])`, where `count` is the
number of synaptic contacts. The synaptic `efficacy` starts from the connectome's
signs and can be learned. Transport follows the declared synapses; the neuron update
reads its own state and synaptic input. Softmax and key normalization additionally
read their declared groups.

A *stimulus* supplies the external drive, which is added to the neuron's input on
every step. The stimulus does not clamp the membrane potential, so synaptic input,
bias and adaptation still move a stimulated neuron.

## Settling and equilibrium

Named functional regions can share one connectome. Neuron-local updates cross the
synapses between regions and seek a common fixed point of the whole brain under the
current drive. Settle the brain once for that joint state; independent solves
followed by a merged visualization do not couple the regions. See [patterns](patterns.md#several-regions-one-equilibrium).

Settling runs the update from rest or from a supplied state. A returned
`BrainState` can be a transient, a fixed point, or part of an oscillation.
`steps` limits work and `tolerance` stops on small activation movement;
`BrainState.activity_change` reports the total movement per row.
`Brain.residual` separately measures the remaining fixed-point equation
error. Saturation can produce small movement with a large residual.

An equilibrium need not be unique or stable. Carrying a state between inputs can
save work or select a different attractor. Test both cold and warm starts, and
reset state at independent episode boundaries. A unique attracting equilibrium
under a fixed drive erases its initial condition; keeping history then requires
an explicit record, a trace, or a drive that carries history.


When every neuron's absolute incoming effective weight sum is below `1 / L`, with `L` the largest slope of the activation, one settling step is a contraction and the [certificate](certificate.md) bounds the remaining distance to the unique equilibrium from the last step's movement. `cd.certificate(brain)` reports it.

## Activation and adaptation

`NeuronModel` rebases a sigmoid so that zero potential emits zero. With zero
drive, zero bias, and zero adaptation, the all-zero state is an exact fixed
point. `learning_neuron_model()` uses a gentler slope and a small negative leak to
give the learner a responsive starting point. Neither setting guarantees convergence.

`Adaptation` adds one variable per neuron that follows its activation and
subtracts from its drive. In suitable mutually inhibitory circuits this can
produce an oscillation. The [half-center example](../examples/half_center.py)
demonstrates one such circuit. Leave adaptation off when beginning with
equilibrium learning; its gradient interpretation needs additional assumptions.

## Three state lifetimes

| State | What changes it | How to manage it |
|---|---|---|
| Potential, activation, adaptation | Settling steps | Pass `state=` to continue; omit it to start from rest |
| A trace or fast-memory matrix | Explicit activity or observation updates | Reset at episode boundaries and preserve batch row identities |
| Learned weights and biases | `Learner.step` or `Learner.update` | Save with `Learner.save`; evaluate with `predict` or `free` |

A [trace](api.md#streams-cadencestream) retains fading activity.
[Fast memory](memory.md) retains associations between supplied keys and values.
[Learning](learning.md) changes a reusable response through free/nudged endpoint
contrasts. The centered learner uses three phases: free, positive nudge, and
negative nudge. Under its equilibrium assumptions, the small-nudge contrast
corresponds to a loss gradient with the stated parameter scaling.

## Evidence and controls

A [protocol](protocols.md) declares stimuli, readouts, interventions, and predicates.
A shuffled connectome tests whether a response depends on the particular connections
under the same neuron model. It is one control, not proof of a biological mechanism.
High gains can saturate an excitatory circuit, so protocols can limit the active
fraction during gain selection.

`conformance` compares a trajectory with the neuron-by-neuron reference.
It checks the tested transport and update. A [receipt](receipts.md) binds stored
results to sources when those files are included and checked; its caller supplies
the arithmetic verifier. Neither check establishes benchmark fairness.

## Compared with backprop networks

| | Feed-forward model trained by backprop | Cadence brain |
|---|---|---|
| Inference | Evaluate layers | Repeated neuron updates with a bounded solve and an optional residual check |
| State between inputs | A cache or a separate memory | A settled state, trace or record, each explicit |
| Credit | Reverse-mode differentiation | Free and nudged endpoint contrasts |
| Exact gradient conditions | Differentiable computation | Stable smooth equilibrium, symmetric effective weights, converged phases, vanishing nudge |
| Work | Forward and backward passes | Every step of the free and nudged phases plus the update |

Measure inference, learning and record maintenance separately on the same held-out
data, and include the simple algorithmic solver when a task has one.

## When to add machinery

Inference, memory and learning are separate operations because their state
lifetimes differ. A new mechanism needs a failure it fixes and an ablation on the
same inputs, budget and seeds, with its state and cost counted. Brain functions
are compositions of the existing operations; [patterns](patterns.md) lists them.
