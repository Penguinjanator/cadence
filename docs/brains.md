# Reusable task brains

Task-specific wiring is useful when its ports and causal role are explicit.
Cadence's optional `cadence.brains` module supplies small compositions over
the existing owner rule. They are engineered designs, not claims that nature has
one universal circuit for each capability.

```python
from cadence.brains import ActivityMonitor, couple, imagine, sensor_motor
```

| Component | Inputs and outputs | Role |
|---|---|---|
| `couple(regions, bridges)` | Named wirings and directed connections between their ports | Assemble one graph for one settlement; keep functional region labels. |
| `sensor_motor(axes)` | One sensory error and an opposing motor pair per axis | A body reads positive-rate minus negative-rate activity to actuate a joint or direction. Continue settlement state between ticks. |
| `imagine(...)` | Live state, legal actions, transition model, evaluator and terminal predicate | Copy state into bounded candidate futures; return action scores and predicted sequences. Alternate max/min for adversarial tasks. |
| `ActivityMonitor` | Controller activity, candidate scores and budget pressure | Read activity change and ambiguity into a six-owner circuit; its output can request more computation. |

Bodies, sensors, transition models, action semantics and evaluators are supplied
by the application. A monitor has an effect only when its readout actually gates
computation; merely drawing it is insufficient. Imagined outcomes are predictions,
not observed rewards or training examples.

## Different regions, one equilibrium

A composite brain is one connected graph with named regions. Each owner reads
messages from its incoming seams and repairs its local potential; those changes
become its neighbors' next readback. Vision, memory, planning and movement have
different functions, but participate in the **same joint state**. Concatenating
pictures of independently settled regions would not implement this interaction.

For the simple graded rule used by the browser task brains:

```text
activation[i] = tanh(potential[i])
error[i] = drive[i] + sum(weight[j,i] * activation[j]) - potential[i]
potential[i] += dt * error[i]
```

All owners read the previous joint state in a synchronous iteration. A masked
owner emits zero activation and is projected to zero potential. Geometry,
observations and weights stay fixed during a control phase. The body consumes
the settled motor readout, changes the sensory boundary, and starts another
phase. Observed lessons can change memory weights between phases. This uses the
existing `GradedRule`; it introduces no additional learning law.

```python
from cadence.brains import couple, sensor_motor
import cadence as cd

vision = cd.Wiring.from_edges(1, pre=[], post=[])
brain = couple(
    {"vision": vision, "movement": sensor_motor(1)},
    [
        ("vision", 0, "movement", 0, 0.5),
        ("movement", 1, "vision", 0, -0.1),
        ("movement", 2, "vision", 0, 0.1),
    ],
)
# One Settlement(brain, rule), with drives/state in region insertion order.
# brain.sets["movement/motor"] identifies the resulting motor ports.
```

Run `python examples/coupled_brain.py` for the complete executable example.
Bridges use region-local owner indices; original contacts, signs and named sets
are preserved. All regions currently share the settlement's rule. `couple` does
not automatically choose useful feedback gains, sensory encodings or a body.

Check `engine.residual(drive, state)` on the **combined** wiring. It measures the
potential equations, independently of how small the integration step is. A small
residual establishes self-consistency for this boundary, not stability, a unique
solution or a globally best action. Feedback can destabilize otherwise stable
parts. Tune and test the combined system, including interventions and useful
behavior. For a contraction under fixed input, full settlement eventually erases
differences due only to initial state; durable lessons need retained records.

Each public demo labels its functional regions and reports a global and regional
equation error. Its replay reconstructs one captured joint trajectory. External
world models and mutually exclusive imagined branches remain separate; the
strategy brain integrates their candidate scores in a shared decision circuit.
See the [six task designs and tests](https://github.com/muellerberndt/cadence-examples/blob/main/COUPLED_BRAINS.md).

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
feeds a six-owner graded evaluator. For each completed depth, the current value,
seven candidate owners and six monitor owners settle in one coupled decision
circuit; that state selects a legal move and the further-work request. An ambiguous decision can extend four-ply search to six plies. The UI
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
