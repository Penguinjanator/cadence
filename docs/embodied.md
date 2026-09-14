# Deploying in a body

A controller reads sensors, puts their values on input ports, settles, and reads
an action from output ports. Learning is a separate choice: update when a target
or reward arrives. The [games guide](games.md) describes historical imitation and reward
examples; [reward](reward.md) describes the eligibility-trace composition.

## One shared control state

The public task brains connect their named sensory, memory, spatial and motor
regions before settling. Every local repair contributes to one joint equilibrium
for the current observation. Actions change the next observation; lessons update
records between phases. The display reports the combined equation residual and
can replay repairs across region boundaries. A converged circuit still needs a
behavior test: self-consistency alone does not establish an optimal controller.
See [coupling regions](brains.md#different-regions-one-equilibrium) for the small
`couple` wiring helper and the complete working example.

## Try the composite brains

The [live showcase](https://github.com/muellerberndt/cadence-examples#live-composite-brains)
includes a teachable mouse, a drawing arm, and a fly-inspired forager. Run
`python serve.py` in cadence-examples to open them. Their simple bodies and
sensor adapters are supplied; their retained records and feedback are visible.

The mouse learns explicit task-cue/destination associations with `FastSeams`.
A visual map gates the recurrent spatial field, and the body's position feeds
the next action selection. Teaching a new association and learning to infer an
unknown map are different tasks; this example demonstrates the former.

The arm reads a drawing pad through 576 retinal units. Target and proprioceptive
ports feed three error units, two joint-coordination units and six antagonistic
motor units. Shoulder, elbow and pencil lift move from those motor outputs; ink
appears through paper contact. The retina receives pixels rather than stroke
coordinates. Geometry, attention selection and weights are supplied. Disabling
joint or pencil motor populations tests their causal roles; disabling pose
readback supplies a disturbance control.

The other moving demos also expose their sensory/motor path: the mouse uses
position-error and directional motor owners, the forager uses bearing/approach
and turn/propulsion owners, and the worm adds directional odor readback and motor
owners around its public chemical graph. The memory example isolates associative
storage and has no body. These are complete controllers for simplified tasks,
not complete biological brains. Each example lives in its own folder and page.

These examples use existing operations. The browser numerical kernels are
checked against Python Cadence, and the showcase records navigation, task
retention, drawing coverage and the relevant conventional controls.

## The loop

This application sketch assumes a saved learner and application-defined
`sense`, `act`, and episode handling. Start with the [quickstart](quickstart.md)
to create the checkpoint.

```python
import numpy as np
import cadence as cd

learner = cd.Learner.load("brain.npz", backend="cpu")
inputs = list(learner.engine.wiring.sets["input"])
state = None
while alive:  # supplied by the application
    features = np.asarray(sense(), dtype=float)  # one value per input owner
    levels = np.zeros((1, learner.engine.wiring.n))
    levels[0, inputs] = features
    drive = learner.engine.clamp_levels(levels)
    state = learner.free(drive, warm=state)
    action = int(np.argmax(state.activation[0, learner.output_index]))
    act(action)
    if episode_ended:
        state = None
```

The full drive includes hidden and output columns. `clamp_levels` scales it by
the rule's clamp amplitude; it does not pad or clip. Normalize sensor readings
using training-fitted scales or an explicit physical encoding.

Warm starts may save work when observations change little. With multiple
attractors they can change the answer, and capped phases can retain transients.
Measure cold and warm control quality on changing observations. A reset starts
an independent episode from rest; separately reset any traces and fast memory.

For stochastic exploration, sample the declared output softmax rather than
always taking the argmax. Mask illegal actions before selecting one. Continuous
actions can use a group of output owners as a population code; see `Bins` in
the [API reference](api.md).

## Learning and checkpoints

Use `learner.step(drive, labels)` for integer action labels from a teacher.
For reward, the target is the action taken and the nudge weight is an estimate
of its advantage. The [reward guide](reward.md) explains delayed credit and
terminal versus truncated episodes. Avoid substituting an immediate reward
for delayed credit without testing that approximation.

`learner.save` includes learned parameters and optimizer history. Loading a
checkpoint and calling `free` or `predict` performs inference without updating
weights. A separate `FastSeams`, `Trace`, environment, or controller state
needs its own save/reset policy. For multiple learners sharing parameters, see
[task recipes](tasks.md#several-learners-in-one-net).

## Measure the deployed loop

Check representative trajectories with `conformance`, inspect residuals when
an equilibrium is intended, and evaluate the actual control objective.
Time sensing, encoding, settlement, readout, and action delivery when reporting
decision latency. Include initialization and compilation separately, and
synchronize accelerator work. A cached response to one unchanged input does
not measure a changing environment.

Keep a [receipt](receipts.md) for the trained model and its evaluation.
Continuing to learn changes the deployed parameters; that checkpoint and
receipt describe the model that was tested.

## A food-and-wall habitat

Launch `python serve.py worm` in the examples checkout. Paint food or walls,
erase a passage, pause, change playback speed, or disable smell. The habitat
wraps the public C. elegans chemical circuit in a supplied diffusion field, local
directional sensory/motor owners, body steps and contact consumption. It is an
inspectable body/environment loop, not learned locomotion or validated digestion.
Switch to **Circuit** for neural stimulation, lesions and the trained MLP
comparison. That comparison measures circuit responses, not foraging ability.

## Inspect the circuit

All six websites place the circuit beside the live body on desktop, with a
stacked mobile layout. Actual owners and seams are grouped by function. Automatic
replay reconstructs sampled settlements from captured inputs and couplings,
slowing early repair steps and compressing later convergence. A changed motor input can trigger a replay even when the spatial field remains
unchanged. Observed associative
weight changes have a separate plasticity view. Behavior badges label seeking,
correction and positive outcomes; they do not measure feelings or chemicals.
Violet repair trails fade as display history, not as controller memory. An
input-release probe exposes transient recurrent decay in an isolated copy. This diagnostic does not add behavioral short-term
memory to an application; use `Trace` or deliberately carried state where the
task needs history. Supplied sensing, steering and geometry adapters are labeled
separately from circuit owners.
The integrated display also plots each population's signed mean activation and
RMS owner-equation mismatch through the captured settlement. Changed messages
appear on their actual seams. Oscillations are displayed when measured; the
viewer never invents ongoing activity at equilibrium. These model traces use
simulation iterations, not biological EEG frequencies.

**Slow thought** holds a prepared motor command until its captured iterations
finish playing. This is a presentation gate around the existing controller,
without another neural rule. Edits invalidate the pending command. Search in
Connect Four has its own explicit thinking/preview phase. A settlement's step
count measures numerical work, not semantic difficulty.

See the [viewer guide](https://github.com/muellerberndt/cadence-examples/tree/main/showcase#read-the-brain-view).

The graded motor potentials persist between control ticks. They provide transient
neural state during behavior; the one-second visual repair trail is separately
labeled display history. No new core primitive is required: ordinary `Settlement`
state continuation and explicit input/output ports implement this pattern.
