# Runtime and mathematical specification

This page specifies the model and numerical contract of the population solver
in `cadence.experimental.equilibrium`. Scope and limits are on the
[landing page](index.md#scope-and-limits); every argument, result field and
refusal is in the [API reference](REFERENCE.md).

## Layout and state

- `Cortex` is a declaration builder. `build()` requires at least one output
  and at least two populations joined into one connected system: every
  population reads another population's states or errors or is read by one.
  After wiring is resolved, the patches must also form one component under
  state and error contacts; shared fixed inputs do not connect them, and a
  sparse layout that splits is refused instead of gaining edges.
- `fan_in=None` reads every source coordinate. A positive `fan_in` samples
  sparse wiring, raised where needed to cover every source coordinate.
  Exceeding `max_connections` raises instead of dropping inputs. A contact is
  potential influence; a zero weight, clamp or saturation can suppress it.
- `inputs=` reads sensor samples or live population states; `observes=` also
  reads their exact current prediction errors. Declarations read only earlier
  populations, so error readback is acyclic. Returning influence comes from
  energy derivatives at the same contacts, with no learned reverse edge. The
  private kernel accepts recurrent state contacts; the builder does not.
- `Input`, `Population` and `Output` are immutable layout handles. `Brain`
  owns live state, weights and biases; callers serialize access to it.
- `state_bound` (default `1.0`) boxes live states and clamps;
  `parameter_bound` (default `4.0`) boxes weights, biases and anchors. A start,
  clamp or anchor outside its box raises `ValueError`. Sensor samples need
  only be finite; the application chooses their normalization.
- Live states and biases start at zero. Each weight starts uniform in
  `[-initial_scale, initial_scale]` (default `0.3`) divided by the square root
  of its target's incoming connection count.

## Prediction and disagreement

Each processing patch `i` owns a live scalar state `x_i`, one weight per
incoming signal and one bias. A population groups patches; an observer is a
population that reads other populations' states and prediction errors.

```text
signal_j = clamped sensor sample, live patch state, or live prediction error
p_i      = tanh(b_i + sum_j w_ij * signal_j)
e_i      = x_i - p_i
E        = 1/2 sum_i e_i² + state_prior/2 sum_i x_i²
```

Agreement means reducing each patch's disagreement with its incoming relation;
patches are not pushed toward a common value. Sensor values stay fixed during
a solve and patch states are repaired together. Predictions and errors are
recomputed on every evaluation, so an error read by an observer is always the
current mismatch, and the observer's energy term reaches the observed states
through exact derivatives, including transitive error dependencies. That error
is no stored forecast-versus-outcome comparison; applications that measure
surprise keep forecasts and outcomes at the body boundary.

Learning adds a proximal prior that anchors relation parameters to their values
immediately before the experience:

```text
E_learning = E + parameter_prior/2 * ||parameters - anchor||²
```

The anchor stays fixed for the whole solve. Both `state_prior` (default
`0.01`) and `parameter_prior` (default `0.1`) must be strictly positive.
Retained parameters are the learned memory used by later queries and stay
plastic: the anchor limits one update and gives no protected consolidation.
Retained live activity is a warm start for the next solve.

## Zero-penalty limit; recursive depth and future time

The positive `state_prior` is part of the model. In the idealized limit
`state_prior = 0`, an unclamped query on a layered graph with fixed parameters
and `state_bound >= 1` has a unique stationary solution: forward substitution,
with every prediction error zero. Zero penalty is not a supported setting. At
positive penalty, or with output or intervention clamps, returning constraints
can move earlier states away from forward substitution. That joint settlement
is a single solve; it does not by itself form recurrent working memory.

An observer reads the current joint state and prediction errors. Nesting
another observer increases observation depth; it does not add a future time
step. To predict outcomes, first teach relations between a situation, an action
and its consequence, then query those relations with alternative actions.
`settle` accepts hypothetical clamps without changing the brain, so such queries
admit no imagined events as experience.

Planning also needs a goal and a way to compare candidate action sequences;
a loop that searches alternatives outside the brain is an application planner.
Planning inside one equilibrium would need actions and future states
represented in that equilibrium, with observers reading and influencing those
states, and behavioral tests of the result. Comparisons keep horizon,
recursive depth, width and total repair work apart.

## Batch experience objective

`observe_batch` groups `B` labeled input/target pairs under one shared
parameter vector `theta`. Row `b` has private activity `x_b`, its own fixed
sensor values and its own output clamps. All rows start from the same retained
live state. With `E_b` the residual-plus-state-prior energy of row `b`, batch
repair minimizes

```text
F_batch = (1/B) * sum_b E_b(x_b, theta)
          + parameter_prior/2 * ||theta - theta_before_batch||²
```

The anchor is fixed for the complete solve and charged once. Shared parameters
couple the rows; rows have no temporal connections to each other. A batch
learns a different trajectory from serial `observe` calls, which retain each
solved state and reanchor after every row.

## Repair and qualification

The solver minimizes the energy by projected analytic-gradient steps with
sufficient-decrease backtracking. The reference engine is synchronized: every
sweep moves all eligible coordinates at once under one global energy check,
with no asynchronous local-message protocol. Eligible coordinates are the free
(unclamped) states, plus all weights and biases when learning. For eligible
coordinates `z` with box projection `P`:

```text
stationarity = max_abs(z - P(z - grad(E)))
```

computed with a unit step, independent of the line-search `step`. A solve
qualifies when `stationarity <= tolerance` (default `1e-6`). For a batch,
the state part uses each row's own unaveraged gradient and the parameter part
uses the gradient of `F_batch`:

```text
max(max_b ||x_b - clip(x_b - grad_x E_b)||_infinity,
    ||theta - clip(theta - grad_theta F_batch)||_infinity)
```

Averaging therefore cannot admit an unresolved row, and the tolerance does not
depend on batch size. Repair moves each row's states by the same unaveraged
gradient; the descent objective and reported `energy` are `F_batch`.

Qualification is constrained numerical stationarity: outward gradients are
allowed at a box boundary and prediction errors need not vanish.
`prediction_residual` reports the largest absolute prediction error. The
objective is nonconvex, so different starts can reach different stationary
points; no uniqueness or global minimum is claimed, and neither number measures
task accuracy.

### Step rule

- The first trial uses `step` (default `1.0`). After an accepted move, let
  `s` be the change in eligible coordinates and `y` the change in their exact
  gradient. The next
  trial uses the scalar secant estimate `dot(s,s) / dot(s,y)` (the first
  Barzilai and Borwein step, [IMA J. Numer. Anal. 8, 1988](https://doi.org/10.1093/imanum/8.1.141))
  when curvature and result are positive and finite. Unchanged coordinates
  contribute nothing.
- The estimate is capped at `ldexp(step, min(backtracks - 1, 1023))`; unsafe
  arithmetic falls back to `step`. Adaptation resets on every solve.
- For a batch, state contributions to the secant numerator are `s_x²/B` and
  parameter contributions `s_theta²`; the denominator and the slope use the
  mean-objective gradient, so row repair does not slow down as `B` grows.
- A trial projects onto the boxes, must move with a finite negative slope, and
  is accepted when `E_new <= E_old + 1e-4 * dot(gradient_old, displacement)`.
  A failed trial halves the step, up to `backtracks` (default `32`) trials
  per sweep.
- A trial is also accepted when its full projected stationarity meets
  `tolerance` and `abs(E_new - E_old) <= 8 * ulp(E_old)`, so a stationary
  proposal can finish when rounded energy reads a few ulps higher.
- `budget` (default `settle_budget`, `2048`) bounds accepted sweeps. A
  stationary initial state qualifies with budget zero; a nonstationary one
  refuses. Budget exhaustion returns `reason="budget"`; failure to find an
  acceptable trial returns `reason="line_search"`. Final stationarity is
  recomputed on the complete graph after any proposal; a call with no proposal
  uses its own initial evaluation, never a previous call's result.

Invalid arguments raise `ValueError`. Numerical overflow may refuse a proposal
or raise `ValueError`; neither outcome commits anything. Outputs of a refused
solve are diagnostics only.

### Query derivatives and cache

Query solves skip derivatives of frozen parameters, so an overflow confined to
an unused parameter derivative cannot refuse a valid query. State derivatives
include every returning constraint through live state and error readback.
Learning computes and checks all parameter derivatives. Finite energy and
finite computed derivatives are always required.

In a single-row query, predictions whose incoming contacts are all fixed sensor
inputs (including bias-only predictions) are constant and are cached across
repair proposals; their errors, consumers and returning derivatives are
recomputed on each proposal. The cache is built only when the starting state
needs repair, is private to the solve, is never retained or serialized, and
is not used by the initial and final evaluations. Learning and batch solves
do not use it; single-row reference refinement after a tensor proposal does.

## Optional tensor execution

`device="python"` (default) is the standard-library float64 reference solver.
`"cpu"`, `"mps"` and `"cuda:N"` compute the same energy and analytic derivatives
with PyTorch tensors; `"cuda"` means `"cuda:0"`. MPS uses float32; CPU and CUDA
default to float64 and accept float32. PyTorch supplies arithmetic only (no
autograd optimizer). Tensor code parallelizes within residual-dependency
levels and vectorizes batch rows; there is no distributed multi-GPU solve.
Device arithmetic proposes and the reference engine admits:

- A stationary start or a zero budget skips the device and runs only the
  reference check. Values not representable in the device dtype raise
  `ValueError`.
- The device stops at `max(tolerance, 64 * dtype_epsilon)`. A float32 device
  phase is capped at `max(1, budget // 2)` accepted sweeps; float64 may use the
  full budget.
- The device candidate is rechecked in float64 against the original sensor
  values, exact output and intervention clamps, the original pre-experience
  anchors and the original frozen query parameters, with the requested
  `tolerance`.
- If the candidate is unqualified, the reference engine refines it with the
  remaining sweep budget. If its float64 energy exceeds the start by more than
  8 ulps, repair restarts from the original start with the remaining budget.
- `energy_history` mixes device and reference steps, so it proves no float64
  decrease per device step. `execution` reports device, precision, tensor and
  reference sweeps and any restart.

PyTorch and the device are checked on first solve: missing PyTorch raises
`ImportError`, an unavailable device raises `ValueError`, and no other device is
substituted. Device choice and cost are in [execution and cost](ACCELERATION.md).

## Continuation and admission

| Operation | Live state | Retained parameters | Event record |
| --- | --- | --- | --- |
| `settle` (with or without clamps) and `predict` | Unchanged | Unchanged | Unchanged |
| Qualified `step`, with or without clamps | Commit solved state | Unchanged | Unchanged |
| Qualified `observe` | Commit solved state | Commit solved relations | Advance once |
| Qualified `observe_batch` | Unchanged | Commit shared relations | Advance once for the batch |
| Refused operation | Unchanged | Unchanged | Unchanged |

- `settle` is pure, including with `targets` or `interventions`. `predict`
  raises `SettlementError` on refusal.
- `step` runs the same frozen-parameter solve and retains the complete activity
  only when every free state qualifies. Clamps apply to that call only: later
  calls start from the retained values and qualify under their own clamps. It
  admits no teaching evidence and consumes no event ID. An all-clamped state
  qualifies without repair because nothing is eligible, and a goal-clamped
  output constrains the solve without forecasting the outcome.
- `observe` needs at least one output target. Targets stay fixed through the
  joint state/parameter repair, and nothing commits unless the whole proposal
  qualifies.
- `observe_batch` validates a finite nonempty sequence of pairs before solving.
  It commits shared parameters and one event; row states are returned and not
  retained or checkpointed, because choosing one row as the live state would
  be arbitrary. A one-row batch solves the same parameters as `observe` and
  keeps live state.

### Event identity

- Event IDs are nonnegative and strictly increasing. An omitted ID is
  allocated on acceptance; applications that need retry safety keep their own.
- Retrying the latest admitted ID with identical sensor samples, clamps and
  source returns `duplicate=True` without solving or committing. A changed
  payload under that ID, or any older ID, is rejected. Only the latest event is
  remembered.
- A batch's identity includes its ordered input/clamp rows: reordering a batch
  under the same ID conflicts. `observe` and `observe_batch` share one event
  sequence.
- `source="witness"` labels actual observations and `source="estimate"`
  derived teaching targets. The label is caller-declared and unauthenticated,
  bound into event identity, shared by all rows of a batch, and changes
  neither the objective nor qualification.

## Reinforcement target and credit horizon

`Reinforcement` stores actual `(context, action, reward, next_context)` records
and fits discrete-action Q estimates through ordinary batch repair, with no
second optimizer and no convergence guarantee. Write `gamma` for `discount`
(default `0.95`), `v` for `value_scale` (default `0.9`) and `r` for
`reward_scale` (default `1.0`). The one-step teaching target is

```text
target = (1-gamma) * v * reward/r
         + gamma * clip(max_a Q(next_context, a), -v, v)
```

- Terminal transitions omit the second term. At `gamma = 0` every target is
  the scaled immediate reward and no next-value query runs, whatever
  `credit_horizon` is.
- Rewards must lie in `[-r, r]`, `gamma` in `[0, 1)`, and `v` strictly below
  both `state_bound` and `1 / (1 + state_prior)`, the attainable output bound.
  Candidate and next-action values require qualified queries; an unqualified
  bootstrap query refuses the replay.
- Replay draws `min(records, batch_size)` records uniformly without
  replacement from a bounded FIFO store and always includes the latest. All
  targets of one replay use the same pre-update brain, and one
  `observe_batch(source="estimate")` fits them. A refused fit keeps parameters
  and the replay RNG state; stored records stay for a later `replay()`.

With `credit_horizon > 1` (default `1`) a target follows up to that many
adjacent records. It crosses to the next record only when the observed next
context equals that record's context, both belong to the same episode, and the
next recorded action is greedy under the pre-update values. Each traversed
reward is scaled by `(1-gamma)*v/r` and discounted along the recorded path. A
terminal tail bootstraps with zero; a cut at a reset, context gap, off-policy
action, missing future record or the horizon bootstraps with the clipped next
value. A time limit is not terminal unless the caller says so. Starting from a
bootstrap in `[-v, v]`, each reverse step `(1-gamma)*v*reward/r + gamma*tail`
is a convex combination, so every target lies in `[-v, v]`.

Feedback names an issued decision and the action actually executed; credit
goes to the executed action. Validation precedes any mutation, and the record
and outcome receipt commit before replay starts, so a refused or failing fit
cannot lose the outcome. An identical retry of the latest outcome changes
nothing, `reset` marks an episode boundary, and these receipts are separate
from the brain's admission events. Retry, snapshot and result rules are in the
[reference](REFERENCE.md#reinforcement-discrete-reward-driven-choices).

`History`, `LearningProgress`, `LiveController` and `slew` sit outside the
energy and add no patch rule; their use is in [live operation](LIVE.md).

## Checkpoint contract

- A snapshot (schema `population-brain/1`) contains the complete layout,
  configuration, arrays, admission cursor and a digest. The JSON must have
  exactly the expected fields, no duplicate keys, finite numbers and consistent
  bounds, counts and identity, and must fit in 32 MiB. Connections and
  dimensions are rebuilt and checked before a continuation is installed.
- Compatibility is bound by a layout/configuration fingerprint and SHA-256
  hashes of `brain.py`, `cortex.py`, `column.py`, `ports.py`, `_repair.py`,
  `_validation.py` and `_tensor.py`. Any difference is rejected, including
  across releases.
- `restore` installs a validated continuation for the same graph and
  configuration, with no overrides. `Brain.from_snapshot` builds a new brain;
  its optional `device` and `dtype` apply only after the snapshot passes full
  validation, keep arrays and admission identity, and update the execution
  configuration and fingerprint. A new device with no dtype takes that device's
  default ([device transfer](ACCELERATION.md#move-a-continuation-between-devices)).
- Hashes and digests check integrity; they do not authenticate a witness.
  Checkpoint files are bounded data and are never executed. Signatures are in
  [checkpoints](REFERENCE.md#checkpoints).

## What the tests cover

```sh
python -m pytest -q tests/equilibrium
```

| Area | Tests |
| --- | --- |
| Analytic derivatives against finite differences, including error readback and private recurrent state contacts | `test_drsn_math.py`, `test_query_gradients.py`, `test_tensor_math.py` |
| Layout rules, connectivity refusals, dense and sparse coverage, exact counts | `test_drsn.py`, `test_setup_errors.py`, `test_inspection_coverage.py`, `test_coupled_default.py`, `test_dense_defaults.py`, `test_ordinary_defaults.py` |
| Step rule, finishing allowance, query cache | `test_spectral_step.py`, `test_query_cache.py`, `test_query_cache_tensor.py` |
| Batch objective, per-row qualification, tensor batches | `test_batch.py`, `test_batch_math.py`, `test_batch_tensor.py` |
| `step` with clamps, learning from witnesses, unclamped recall | `test_step_clamps.py`, `test_learning_basics.py`, `test_bootstrap.py` |
| Event identity, source labels, atomic refusal, checkpoints | `test_drsn.py`, `test_event_serialization.py`, `test_estimates.py`, `test_packaging.py` |
| Device configuration, tensor proposals, device transfer | `test_execution_config.py`, `test_tensor_math.py` |
| Reinforcement target, credit horizon, feedback ownership | `test_reinforcement.py`, `test_reinforcement_audit.py`, `test_credit_horizon.py`, `test_execution_custody.py`, `test_temporal_credit.py` |
| Executable documentation | `test_documentation.py` |

Tests check subsequent unclamped predictions on new inputs, never clamped
outputs, which equal their targets by construction.
