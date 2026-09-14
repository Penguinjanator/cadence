# Reusable task brains

Task-specific wiring is useful when its ports and causal role are explicit.
Cadence's optional `cadence.brains` module supplies three small compositions over
the existing owner rule. They are engineered designs, not claims that nature has
one universal circuit for each capability.

```python
from cadence.brains import ActivityMonitor, imagine, sensor_motor
```

| Component | Inputs and outputs | Role |
|---|---|---|
| `sensor_motor(axes)` | One sensory error and an opposing motor pair per axis | A body reads positive-rate minus negative-rate activity to actuate a joint or direction. Continue settlement state between ticks. |
| `imagine(...)` | Live state, legal actions, transition model, evaluator and terminal predicate | Copy state into bounded candidate futures; return action scores and predicted sequences. Alternate max/min for adversarial tasks. |
| `ActivityMonitor` | Controller activity, candidate scores and budget pressure | Read activity change and ambiguity into a six-owner circuit; its output can request more computation. |

Bodies, sensors, transition models, action semantics and evaluators are supplied
by the application. A monitor has an effect only when its readout actually gates
computation; merely drawing it is insufficient. Imagined outcomes are predictions,
not observed rewards or training examples.

## Motor control

```python
import numpy as np
import cadence as cd
from cadence.brains import sensor_motor

engine = cd.Settlement(
    sensor_motor(2),
    cd.GradedRule(gain=1, slope=2, threshold=0, leak=1, dt=0.25, clamp_amplitude=1),
)
state = engine.settle([0.2, -0.1, 0, 0, 0, 0], steps=24)
rates = np.maximum(0, state.activation[2:]).reshape(2, 2)
command = rates[:, 0] - rates[:, 1]
# body.apply(command); then read new sensory error and continue from state
```

The wiring names its `sensory`, `motor`, `positive` and `negative` ports so a
larger controller can connect or mask them.

The browser eye/arm extends this design: retinal samples and target/proprioceptive
ports feed coupled visual-error and joint-coordination regions, then six motor
units drive shoulder, elbow and pencil height. Masking a motor population removes
its actuation. This is a causal computational motor unit, not a model of all
biophysics in a motor neuron.

## Comparing futures

`imagine` deep-copies every branch, including array state. The evaluator returns
utility from the root actor's perspective. `adversarial=True` alternates maximizing
and minimizing layers. The finite node budget raises on exhaustion, so partially
searched alternatives cannot silently masquerade as a completed comparison.
Transitions must not mutate external objects; evaluators must remain read-only.

The Connect Four website uses a specialized alpha-beta search with iterative
deepening and a node budget. It retains only completed depths. Its threat readout
feeds a six-owner graded evaluator; candidate values then feed the activity
monitor. An ambiguous decision can extend four-ply search to six plies. The UI
shows the hypothetical continuation separately from the real board.

This is explicit architectural deliberation. Recurrence alone does not guarantee
a useful transition model, planning algorithm or evaluator. Transformers and
conventional controllers can also be coupled to search. The relevant controls
are evaluator-only, fixed-depth search, monitor-disabled search and exact
terminal verification.

## Self-reading and consciousness

`ActivityMonitor` observes the controller's own changing predictions rather than
only its environment. It retains its state, combines repair/ambiguity/pressure
signals, and returns a request that the controller can act on. Tests check that
ambiguity requests more work, a strong preference needs less, and a spent budget
prevents further work. These signals are heuristics, not calibrated confidence.

This is metacognitive control. It does not establish subjective experience, an
integrated self, or a species boundary between humans, apes and insects. Scientific
accounts of consciousness remain contested; a recent adversarial experiment
challenged central predictions of two major theories
([Cogitate Consortium, 2025](https://www.nature.com/articles/s41586-025-08888-1)).
A claim that recursive monitoring is necessary or sufficient for consciousness
would require a separate theory and evidence.

The APIs add no new learning law or large framework. They compose bounded patches,
local readback, records and feedback into testable task controllers.
