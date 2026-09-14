# Deploying in a body

A controller reads sensors, puts their values on input ports, settles, and reads
an action from output ports. Learning is a separate choice: update when a target
or reward arrives. The [games guide](games.md) shows public imitation and reward
examples; [reward](reward.md) describes the eligibility-trace composition.

## Try the composite brains

The [live showcase](https://github.com/muellerberndt/cadence-examples#live-composite-brains)
includes a teachable mouse, a drawing arm, and a fly-inspired forager. Run
`python serve.py` in cadence-examples to open them. Their simple bodies and
sensor adapters are supplied; their retained records and feedback are visible.

The mouse learns explicit task-cue/destination associations with `FastSeams`.
A visual map gates the recurrent spatial field, and the body's position feeds
the next action selection. Teaching a new association and learning to infer an
unknown map are different tasks; this example demonstrates the former.

The arm places two visual-error owners and two motor-correction owners in one
`Wiring`. The visual owners receive negative feedback from predicted movement;
the motor owners receive visual residuals through the supplied arm Jacobian.
They settle together, the joints move, and actual pose readback supplies the
next drive. Disabling readback provides an open-loop disturbance control.
The image adapter extracts outlines; this is not learned image understanding.

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
