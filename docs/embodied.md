# Deploying in a body

A controller reads sensors, puts their values on input ports, settles, and reads
an action from output ports. Learning is a separate choice: update when a target
or reward arrives. The [games guide](games.md) shows public imitation and reward
examples; [reward](reward.md) describes the eligibility-trace composition.

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
