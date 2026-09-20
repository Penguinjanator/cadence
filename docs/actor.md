# EquilibriumActor

`cadence.actor.EquilibriumActor` brings the verified fixed-body observer and
joint future/action repair into the base library. It uses supplied or previously
learned two-coordinate dynamics. Its past and future use the same family of
squared local residuals, with a causal boundary between factual inference and
private preference-driven planning.

This is a transparent linear Gaussian/quadratic component. It does not turn
`TemporalPatchNet` into a general nonlinear action planner, train its body model
online or provide a musical composition policy.

## Observe, plan, execute and observe again

The inherited body coordinates are position and one-tick displacement. The
sensor exposes position only. The model predicts
`state_next = F @ state + G * action`. Sensor variance and process covariance
are supplied assumptions. The default process covariance is
`0.04 * outer(G, G) + 1e-6 * eye(2)`, with initial mean zero and covariance
`100 * eye(2)`. These values reproduce the tested stress model; they are not
learned reliability estimates.

```python
import numpy as np
from cadence.actor import BodyModel, EquilibriumActor

F = np.array([[1.0, 1.0], [0.0, 1.0]])
G = np.array([1.0, 1.0])
actor = EquilibriumActor(BodyModel(F, G, sensor_variance=0.0), goal=0.6)

# Two actual initial readings, joined by an actual zero action.
actor.admit(0.0, identifier=0)
actor.admit(0.0, identifier=1, executed_action=0.0)
plan = actor.plan()
assert plan.residual < 1e-8

# A tiny simulated body; an application calls its real actuator/sensor here.
physical_state = np.zeros(2)
executed_action = plan.action
physical_state = F @ physical_state + G * executed_action
actor.admit(physical_state[0], identifier=2, executed_action=executed_action)
next_plan = actor.plan(horizon=2)  # explicitly decrease this task's deadline
```

`goal` and default `horizon` are retained supplied task information. Planning
without an override uses that horizon again; no hidden clock counts down a
deadline. The application executes the first proposed action and then admits
what actually happened. Proposals are **unconstrained**: this solver does not
impose actuator bounds. If a body clips an action, report the executed value.
Admission IDs enforce order and exactly-once processing within this actor; they
do not authenticate a sensor or prove that an arbitrary caller supplied reality.

## Minimal fixed-model memory

The actor retains one immutable position record and a Gaussian incoming prefix
message. Before replacing that record, it conditions the old reading exactly
once, propagates through the known executed action, and marginalizes the old
latent state. The incoming message excludes the retained record's likelihood.
This prevents counting the same measurement twice.

A zero sensor variance fixes position exactly. Only displacement remains
uncertain, so the current posterior covariance can be singular. Propagating it
with positive process covariance produces a valid next prior without inverting
the singular posterior. A goal never participates in this past calculation.

The compressed prefix is bound to the model values, shapes, dtypes, noise and
fixed initial prior. Body arrays are copied and read-only. If that protection is
bypassed and a model changes, subsequent admission, inference, planning and
snapshotting refuse to reinterpret the old prefix. Relearning a model after
throwing away its original observations requires an explicit new inference
policy; this API does not claim to solve that problem.

## Private future repair and self-readback

Each future patch owns `(position, displacement, action)`. With the factual
current estimate held fixed, neighboring patches minimize

```text
0.5*q*sum(||state_next - F state - G action||²)
+ 0.5*||terminal_state - [goal, 0]||²
+ 0.5*effort*sum(action²)
```

Defaults are `q=1000` and `effort=0.02`. Adjacent three-coordinate blocks exchange
precision matrices and information vectors. Every Schur pivot must be positive,
and the final equation residual must be below `1e-8`. Failure raises an error
without changing evidence. The model is fixed during planning. Its soft dynamics
can bend hypothetical trajectories, so inspect seam defects and evaluate actual
executed cost. Internal goal satisfaction is not physical success.

```python
before = actor.snapshot()
positive = actor.plan(goal=0.6)
negative = actor.plan(goal=-0.6)
assert actor.snapshot() == before
np.testing.assert_array_equal(positive.boundary, negative.boundary)
np.testing.assert_array_equal(positive.covariance, negative.covariance)
assert positive.action != negative.action

readback = actor.readback()
assert readback.record.identifier == 2
assert readback.residual < 1e-7
```

`readback()` exposes the detached factual mean, covariance, current raw record,
model binding and admission/marginalization counts. `plan()` exposes hypothetical
states, proposed actions/readings, seam defects, energy terms and solver work.
That is inspectable self-readback and private imagination, not an automatically
learned higher-level self-model. Covariance is reported but is not used as a
chance constraint or uncertainty-dependent policy in future planning.

## Checkpoints and cost

```python
path = actor.save("acting_brain.npz")
restored = EquilibriumActor.load(path)
assert restored.snapshot() == actor.snapshot()
np.testing.assert_array_equal(restored.plan().actions, actor.plan().actions)
```

Snapshots use plain data; atomic NumPy archives load without pickle. They include
the fixed model, bound incoming prefix, record ID, counters, goal, horizon and
planning settings. Checkpoint validation detects a mismatched model binding and
inconsistent counters. It is structural validation, not cryptographic
attestation of arbitrary edited evidence.

The retained numerical payload after admission is 232 bytes: model arrays,
prefix coefficients, one record, seven scalar/counter values and a 32-byte model
binding. Python objects and serialized archives cost more. Inference allocates a
current mean/covariance; each prefix update performs two 2-by-2 solves and one
scalar measurement conditioning. Noisy inference performs one 2-by-2 solve;
exact-position inference solves one scalar. Future planning factors one
3-by-3 block per horizon step. `message_bytes` and `coefficient_bytes` count
explicit planner arrays, excluding numerical-library workspace. These figures
are component accounting, not a hardware benchmark.

## What is implemented

| Function | Current library mechanism | Limit |
| --- | --- | --- |
| Learned temporal relationships | `TemporalPatchNet.observe` and centered detuning | Finite supplied paths; no protected lifelong retention guarantee. |
| Continuing context | `TemporalPatchNet` hidden boundary | Retention and useful continuation require behavioral tests. |
| Fixed-model factual memory | `EquilibriumActor` bound Gaussian prefix and one real record | Exact sufficiency applies to its fixed linear Gaussian assumptions. |
| Imagination | Detached temporal rollouts and actor future-state/action repair | Prediction quality depends on acquired relationships. |
| Planning and correction | Joint quadratic state/action solve, actual executed-action/readback admission | Supplied goal, coordinates, model and deadline; unconstrained actions. |
| Self-readback | Detached state, uncertainty, residuals and proposal diagnostics | No learned hierarchy or consciousness claim. |

The base package now exposes these components directly. Their APIs make the
boundary between real observation and private imagination explicit. Coupling a
nonlinear learned temporal model to free action ports, autonomous intention,
reliable specialization, continued model revision and Amen composition remain
separate experimental requirements.
