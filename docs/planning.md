# Learn consequences, then plan

`TemporalPatchNet.plan` proposes values for declared input ports so that a
learned [temporal model](temporal.md) predicts a supplied goal path. It uses the
same temporal energy and centered detuning as learning, on a private copy of
the model with the initial hidden boundary held fixed. Every accepted proposal
improves an ordinary free prediction of the goal. The call executes no action
and admits no observation: the application runs the action, measures the
result and teaches that record through `observe`. A useful plan needs a model
that has learned actual consequences, as in the
[worked example](#learn-consequences-then-act) below.

## A bounded proposal

This supplied toy model responds only to its second input port. The first port
is a fixed observation.

```python
import numpy as np
from cadence import TemporalPatchNet

toy = TemporalPatchNet(inputs=2, hidden=1, outputs=1, tolerance=1e-12)
toy.set_parameters({
    "A": np.zeros((1, 1)),
    "B": np.array([[0.0, 1.0]]),
    "C": np.ones((1, 1)),
})
toy_inputs = np.array([[[5.0, 0.0], [-3.0, 0.0]]])
toy_before = toy.snapshot()
toy_plan = toy.plan(
    toy_inputs,
    goal=np.full((1, 2, 1), 0.8),
    controls=np.array([False, True]),
    bounds=(-0.2, 0.2),
    rate=8.0,
    max_steps=8,
)

assert toy_plan.improved
np.testing.assert_array_equal(toy_plan.inputs[:, :, 0], toy_inputs[:, :, 0])
np.testing.assert_allclose(toy_plan.inputs[:, :, 1], 0.2)
assert toy_plan.converged                # No feasible local descent is left.
assert not toy_plan.predicted_goal_met   # The bound prevents reaching 0.8.
for name, values in toy_before.items():
    np.testing.assert_array_equal(values, toy.snapshot()[name])
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `inputs` | required | Starting input path, `(batch, time, inputs)`. |
| `goal` | required | Desired output path, `(batch, time, outputs)`; batch and time must match `inputs`. |
| `controls` | required | Boolean mask broadcast to the input shape; selects at least one entry. Only selected entries change. |
| `bounds` | `None` | `(lower, upper)` tuple broadcast to the input shape. Selected starting values must lie inside; fixed ports may lie outside. |
| `state` | `None` | Initial hidden boundary `(batch, hidden)`. Otherwise the live boundary is copied, or zero for a net with no active state. |
| `beta` | `0.01` | Goal detuning. Requires `0 < beta * max(output_precision) < time * outputs`. |
| `rate` | `1.0` | Scale of a steepest step before backtracking. |
| `max_steps` | `32` | Cap on accepted steps. `0` evaluates stationarity only. |
| `max_backtracks` | `16` | Step sizes tried per direction, each half the last; at least 1. |
| `tolerance` | `1e-6` | Projected residual that counts as stationary. |
| `goal_tolerance` | `1e-6` | Cost at or below which `predicted_goal_met` is true. |
| `symmetry_tolerance` | `0.15` | Largest contrast asymmetry accepted; `inf` disables the check. |
| `max_halvings` | `8` | How often beta may halve over the whole call to center contrasts. |
| `method` | `"steepest"` | `"steepest"` or `"bfgs"`. |

Invalid arguments raise `ValueError` without touching the network. An initial free
prediction that fails or has a nonfinite cost raises `ArithmeticError`.

## What is optimized

The cost is half the weighted mean squared error of a free prediction, with
the network's [output precision](temporal.md#explicit-output-teaching-precision)
as the weights:

```text
cost = 0.5 * mean_batch,time,outputs(output_precision * (free_output - goal)²)
```

At each proposal the solver settles positive and negative goal detunings from
the same fixed boundary. With `e_plus` and `e_minus` their transition defects
and `B` the input projection, the input contrast is

```text
gradient = -((e_plus - e_minus) @ B) / (2 * beta * batch)
```

and it is zero on fixed ports. Each contrast must pass the same centering check
as `observe` ([checking that the contrast is centered](temporal.md#checking-that-the-contrast-is-centered)).
An off-center pair halves beta and is solved again. Halvings count over the
whole call and later contrasts keep the halved beta; a pair off center after
`max_halvings` halvings ends the search. A projected step clips the controlled values
to their bounds. Backtracking accepts it only after a converged, target-free
replay gives a strict cost decrease and the Armijo decrease. Detuned outputs
never stand in for the returned prediction.

On a smooth stable branch the contrast approaches the gradient of the free
cost as beta tends to zero; finite beta adds error, and a stationary result is
local.

## A quasi-Newton direction

An early action moves every later output and a late one moves almost none, so
the cost over an action path is badly conditioned and steepest steps zigzag.
`method="bfgs"` keeps an estimate of the inverse curvature over the controlled
entries, updated by BFGS from accepted steps, and steps along
`-curvature @ contrast` with the same line search and replay. Its first step,
and the first step after the estimate is dropped, is the steepest step scaled
by `rate`; later `step_sizes` are multiples of the curved direction, with 1 the
full step.

When an accepted pair fails the curvature condition, that update is skipped and
the estimate is kept. The estimate is dropped after a step clipped by the
bounds, when its direction does not descend, or when no step along a curved
direction decreases the cost; the search then continues from the contrast.
`tests/test_planning.py` checks that `bfgs` reaches the same stationary cost as
`steepest` with fewer iterations and less than half the phase calls.

## Reading the result

`TemporalPlan` holds the proposed `inputs` and the fixed `boundary` as
detached arrays, the free `prediction` and `initial_prediction` as
`TemporalPhase` results, and the frozen model's `parameter_revision`.

| Field | Meaning |
| --- | --- |
| `cost`, `initial_cost`, `losses` | Costs of the initial and every accepted free prediction; each accepted cost strictly decreases. |
| `improved`, `iterations`, `step_sizes` | Whether cost decreased, accepted step count, and the step scales used. |
| `projected_residual` | Largest change under a unit projected contrast step at the last accepted proposal; `None` when its contrast failed or was nonfinite. |
| `converged` | The projected residual met `tolerance`; the goal may be out of reach. |
| `predicted_goal_met` | The returned prediction's cost is at most `goal_tolerance`. It says nothing about a real outcome. |
| `reason` | Why the search stopped (below). |
| `beta`, `contrast_halvings` | Detuning of the last contrast and how often the supplied beta was halved. |
| `method` | `"steepest"` or `"bfgs"`. |
| `phase_calls`, `block_chain_attempts`, `energy_evaluations`, `peak_message_bytes` | Work over all attempted phases, rejected ones included. Checkpoint copies and Python overhead are excluded. |

| `reason` | Meaning |
| --- | --- |
| `projected_stationary` | The projected residual met `tolerance`; `converged` is true. |
| `step_cap` | `max_steps` accepted steps were taken; the cap is not convergence. |
| `plus_<phase reason>`, `minus_<phase reason>` | A detuned phase failed; the suffix is its [phase reason](temporal.md). |
| `contrast_asymmetric` | The contrast stayed off center after `max_halvings` halvings. |
| `nonfinite_input_gradient` | The contrast had a nonfinite entry. |
| `no_decreasing_causal_step` | No backtracked step gave a converged replay with lower cost. |

On every stop the last accepted proposal and its free prediction are returned;
only `projected_stationary` claims stationarity. Planning runs on a checkpoint copy: it changes no learned arrays,
active state, readback or revision of the original network, and no caller array.

## Closing the loop

Execute a chosen control in the real application, measure its consequence, and
teach that actual record through `observe`, never the goal or the predicted
future. If the live state or parameters change before execution, plan again
from the current boundary. Choosing goals is the application's job.

## Learn consequences, then act

This example learns how a small nonlinear body moves, plans an action path,
executes its first action and measures the result before planning again. Only
executed actions and measured positions reach the learner.

| Input port | Meaning | May planning change it? |
| --- | --- | --- |
| 0 | Measured position at the start of this path | No |
| 1 | Initial measurement is present | No |
| 2 | Executed action during learning; proposed action during planning | Yes, within `[-1, 1]` |

Only the first time step carries the position and its presence flag; later
steps carry actions and predict later positions. Cadence does not read a zero
presence flag as missing data; the encoding is the application's. The body is
fully observed, so each replan starts from a zero hidden boundary; a partially
observed world needs a tested state estimate instead.

```python
def actual_step(position, action):
    # The environment's transition, used to execute actions and read sensors.
    return 0.85 * position + 0.2 * np.tanh(2 * action) - 0.05 * position**3


def collect_actual_paths(rng, batch, horizon=6):
    position = rng.uniform(-0.6, 0.6, batch)
    actions = rng.uniform(-1.0, 1.0, (batch, horizon))
    paths = np.zeros((batch, horizon, 3))
    paths[:, 0, 0] = position
    paths[:, 0, 1] = 1.0
    paths[:, :, 2] = actions
    measured = np.empty((batch, horizon, 1))
    for t in range(horizon):
        position = actual_step(position, actions[:, t])
        measured[:, t, 0] = position
    return paths, measured


net = TemporalPatchNet(inputs=3, hidden=16, outputs=1, seed=709, initial_radius=0.8)
training_rng = np.random.default_rng(10709)
for _ in range(256):
    executed, measured = collect_actual_paths(training_rng, batch=8)
    net.reset()  # Independent body episodes, not one continuous path.
    assert net.observe(executed, measured, beta=0.01, rate=1.0).updated

# Held-out states and action paths: no update uses these outcomes.
test_inputs, test_measured = collect_actual_paths(np.random.default_rng(20709), batch=128)
test_prediction = net.imagine(test_inputs, state=np.zeros((128, 16)))
assert test_prediction.converged
forecast_mse = float(np.mean((test_prediction.output - test_measured) ** 2))
assert forecast_mse < 0.004
```

During the control trial the parameters stay fixed. Each step plans six actions,
executes only the first, and replans from the measured position. At step 6 an
unannounced push moves the body; the controller sees it only through the next
measurement. A zero-action body with the same push is the comparison.

```python
frozen = net.snapshot()
position = zero_action_position = -0.45
goal_position = 0.35
measured_positions, zero_action_positions, plan_reasons = [], [], []

for t in range(12):
    if t == 6:
        position -= 0.25
        zero_action_position -= 0.25
    start = np.zeros((1, 6, 3))
    start[0, 0, :2] = [position, 1.0]
    proposal = net.plan(
        start,
        goal=np.full((1, 6, 1), goal_position),
        controls=np.array([False, False, True]),
        bounds=(-1.0, 1.0),
        state=np.zeros((1, 16)),
        beta=0.01,
        rate=8.0,
        max_steps=32,
    )
    assert proposal.prediction.converged
    assert proposal.reason in {"projected_stationary", "step_cap", "no_decreasing_causal_step"}
    np.testing.assert_array_equal(proposal.inputs[:, :, :2], start[:, :, :2])
    assert np.all(np.abs(proposal.inputs[:, :, 2]) <= 1.0)

    position = float(actual_step(position, proposal.inputs[0, 0, 2]))
    measured_positions.append(position)
    zero_action_position = float(actual_step(zero_action_position, 0.0))
    zero_action_positions.append(zero_action_position)
    plan_reasons.append(proposal.reason)
    for name, value in frozen.items():  # Planning changed no learned array.
        np.testing.assert_array_equal(value, net.snapshot()[name])

late_error = float(np.mean((np.array(measured_positions[-3:]) - goal_position) ** 2))
zero_action_error = float(np.mean((np.array(zero_action_positions[-3:]) - goal_position) ** 2))
assert late_error < 0.01 and late_error < 0.25 * zero_action_error
print(f"Held-out forecast MSE: {forecast_mse:.6f}; late goal MSE: {late_error:.6f}")
```

The measurement, never the prediction, supplies the next planning boundary. To
adapt later, train on executed-input and measured-output records through
`observe` and plan again; that learning needs its own
[retention checks](temporal-memory.md). The proposals may stop at their step
cap while control improves, so read `plan_reasons`, `converged` and the
actual outcomes separately. The assertions hold for this seed and body. A
one-step controller also solves this monotone constant-goal task.
