# API reference

Exact signatures, defaults, result fields and refusals for every public name
of `cadence.experimental.equilibrium`: `Cortex`, `Brain`, `Input`,
`Population`, `Output`, `SettlementError`, `bootstrap`, `History`,
`LearningProgress`, `Reinforcement`, `LiveController`, `slew` and `__version__`.
Import them from the package; the module files are internal.

```python
from cadence.experimental.equilibrium import (
    Cortex, Brain, Input, Population, Output, SettlementError, bootstrap,
    History, LearningProgress, Reinforcement, LiveController, slew, __version__,
)

cortex = Cortex(seed=0)
x = cortex.input("x", shape=2)
hidden = cortex.column("hidden", patches=4, inputs=x)
motor = cortex.column("motor", patches=2, inputs=hidden)
cortex.output("y", shape=(), reads=motor)
brain = cortex.build()

first = brain.observe({"x": [1.0, 0.0]}, {"y": 0.5}, event_id=0)
retry = brain.observe({"x": [1.0, 0.0]}, {"y": 0.5}, event_id=0)
assert first["accepted"] and retry["duplicate"]
copy = Brain.from_snapshot(brain.snapshot())
assert copy.weights == brain.weights and copy.inspect()["last_event_id"] == 0
```

Start with the [quickstart](QUICKSTART.md) and the
[design guide](BRAIN_DESIGN.md); the [specification](SPECIFICATION.md) holds
the equations, and [scope and limits](index.md#scope-and-limits) separates
observer wiring from automatic System 2 behavior, which is not implemented.
`__version__` is the package release string.

## Cortex: declare a layout

```text
Cortex(
    *,
    seed=0,
    fan_in=None,
    initial_scale=0.3,
    settle_budget=2048,
    tolerance=1e-6,
    state_prior=0.01,
    parameter_prior=0.1,
    state_bound=1.0,
    parameter_bound=4.0,
    step=1.0,
    backtracks=32,
    max_patches=10000,
    max_connections=1000000,
    max_inputs=1000000,
    device="python",
    dtype=None,
)
```

All arguments are keyword-only. Integer parameters reject booleans; `fan_in`
also accepts `None` for full connectivity. Real parameters must be finite and
strictly positive. Defaults suit small numerical and learning examples.

| Parameter | Contract |
| --- | --- |
| `seed` | Nonnegative integer controlling sampled wiring and initial relation coefficients. Reproduction also needs the same layout, configuration and implementation. |
| `fan_in` | `None` (default) connects every coordinate of each declared source to every destination patch. A positive integer requests sparse sampling: a minimum count per destination, capped at source size and raised to cover the complete source across the population. Applies separately to each source and signal kind. |
| `initial_scale` | Positive initial weight scale, at most `parameter_bound`. Each weight is sampled uniformly in `[-initial_scale, initial_scale]` and divided by the square root of its target's incoming connection count. Biases and live state start at zero. |
| `settle_budget` | Nonnegative maximum accepted repair sweeps per solve. Zero qualifies a stationary starting state without repair. A sweep may evaluate several rejected proposals. |
| `tolerance` | Positive maximum projected-gradient residual for qualification. An absolute numerical threshold, unrelated to prediction accuracy. |
| `state_prior` | Positive coefficient of the quadratic activity penalty. It changes preferred states and returning influence in free layered queries as well as solver speed. Zero is unsupported. |
| `parameter_prior` | Positive coefficient anchoring weights and biases to their pre-call values during `observe` or `observe_batch`. The anchor is fixed for the whole solve; a batch applies it once to shared parameters. |
| `state_bound` | Positive absolute bound on processing-patch states and output/intervention clamps. Sensor values are not clipped to it. |
| `parameter_bound` | Positive absolute bound on weights and biases; repair projects eligible parameter coordinates into this box. |
| `step` | Positive initial and fallback projected-gradient step. Later trials estimate a scalar step from the previous accepted displacement and gradient change; unsafe estimates fall back to this value. |
| `backtracks` | Positive maximum line-search proposals per sweep. Rejection halves the proposed step. |
| `max_patches` | Positive cap on processing patches across all populations. Sensor samples and output aliases do not count. |
| `max_connections` | Positive cap on compiled directed input, state and error-readback connections. |
| `max_inputs` | Positive cap on scalar samples across all sensors. |
| `device` | `"python"` (default) uses the standard-library reference engine. `"cpu"`, `"mps"`, `"cuda"` and `"cuda:N"` select optional PyTorch tensor execution. `"cuda"` resolves to `"cuda:0"`; `N` is a nonnegative device index. Availability is checked on the first solve. |
| `dtype` | `None` resolves to `"float32"` for `"mps"`, otherwise `"float64"`. Explicit `"float32"` or `"float64"` selects tensor proposal precision. `"python"` requires `"float64"`; `"mps"` requires `"float32"`. Final qualification always uses Python float64 with the original data. |

`cortex.config` is a read-only mapping of resolved settings. The caps count
graph elements and do not bound RAM or runtime; larger inputs raise coverage
cost even with a small `fan_in`. Tensor execution needs
`pip install "cadence-net[accel]"` and imports PyTorch only on the first solve,
which raises `ImportError` if it is missing or `ValueError` for an unavailable
device; no other device is chosen silently. See
[execution and cost](ACCELERATION.md).

### Layout methods

| Method | Return and behavior |
| --- | --- |
| `input(name, *, shape)` | An `Input` boundary whose supplied values stay fixed for the complete solve. |
| `column(name=None, *, patches, inputs=())` | A `Population` of exactly `patches` processing patches, a positive integer. `inputs` accepts existing sensor or population handles. |
| `observer(name=None, *, patches, inputs=(), observes)` | Experimental readback with the same patch primitive and at least one observed population. It reads their live states and exactly recomputed prediction errors in every joint solve and may also take ordinary `inputs`. |
| `output(name, *, shape, reads, indices=None)` | An `Output` exposing selected coordinates of one population, with no separate output network. |
| `build()` | A `Brain` with resolved wiring. Requires at least one output and one connected system: every population reads another population's states or errors or is read by one, no group of populations settles apart from the rest, and all resolved patches form one connected component, including under sparse wiring. A single population, an unread sensors-only population or disconnected patch groups raise `ValueError` naming the defect. A successful build freezes the layout; further construction or a second build raises `ValueError`. |

Names are unique across node types, UTF-8 strings of 1 to 256 characters.
`None` generates a name from the node type and an unused positive integer.
References must be handles returned by this `Cortex`; a matching name or a
handle from another layout is refused.

`inputs` and `observes` accept one handle or an iterable of distinct, existing
handles. `inputs` reads sensor values or population states; `observes` takes
populations only and also reads their errors (a population declared in both
gets a single state connection). Empty-input columns are permitted. All eligible
states are solved jointly, observers included, in every solve.

`shape` is a positive integer or a tuple/list of at most eight positive
integers; `shape=()` is one scalar. Output size cannot exceed `reads.patches`.
`indices` selects distinct zero-based local coordinates of `reads`, in output
order, with length equal to output size; the default is `range(size)`.
Different outputs may alias the same patch, so two scalar outputs reading one
population both default to index zero. Use `indices=(0,)` and `indices=(1,)`
for independent controls, or one vector output with `shape=2`.

Normalization, features and motor interpretation belong to the application.

### Immutable layout handles

Obtain these from the builder; do not construct them directly.

| Class | Public fields and properties |
| --- | --- |
| `Input` | `name`, `shape`; `size` counts scalar samples. |
| `Population` | `name`, `patches`, `inputs`, `observes`; `role` is `"observer"` when `observes` is nonempty, otherwise `"processing"`. |
| `Output` | `name`, `shape`, `reads`, `indices`. |

## Brain: query, continue and learn

Construct with `Cortex.build()` or `Brain.from_snapshot`; the `Brain`
constructor is internal. Use one serial owner per brain: methods have no
thread synchronization, and admission and restore are all-or-nothing under
serial access.

| Method | Contract |
| --- | --- |
| `settle(inputs, *, targets=None, interventions=None, budget=None)` | Pure query returning a complete solve result. Optional output targets and population interventions are hypothetical clamps. Changes neither live state nor parameters. |
| `predict(inputs, *, budget=None)` | Pure query returning an output-name-to-flat-tuple mapping. Raises `SettlementError` if the solve does not qualify. |
| `step(inputs, *, targets=None, interventions=None, budget=None)` | Repair activity with parameters frozen, optionally under the same clamps as `settle`. Retains the proposed state only if qualified; parameters and event history are unchanged. Returns the full result plus `accepted`. |
| `observe(inputs, targets, *, event_id=None, budget=None, source="witness")` | Jointly repair state, weights and biases under at least one output target. A qualified solve atomically retains state, parameters and event ownership; a refusal retains nothing. |
| `observe_batch(examples, *, event_id=None, budget=None, source="witness")` | Jointly repair private row states and shared parameters under a batch of targets. A qualified solve atomically retains parameters and one event identity and keeps the pre-call live state. A refusal commits nothing. |
| `inspect()` | Layout description, graph counts, topology, observation roles and continuation metadata. |
| `snapshot()` | Complete continuation as a JSON string of at most 32 MiB of UTF-8 text; a larger one raises `ValueError`. |
| `Brain.from_snapshot(text, *, device=None, dtype=None)` | Class method that validates the complete snapshot before constructing a new brain. Optional execution overrides keep arrays and event identity and change configuration and fingerprint. See [checkpoints](#checkpoints). |
| `restore(text)` | Validate, then replace this brain's continuation. Layout and configuration fingerprint must match. Returns `None`. |

`budget` overrides `settle_budget` for one attempted solve and must be a
nonnegative integer; `None` uses the configured budget. An identical
latest-event retry performs no solve and ignores `budget`. Queries start from
the retained live state and do not reset it.

`SettlementError` is a `RuntimeError` raised by `predict` for a valid but
unqualified solve. Its message gives the refusal reason, stationarity,
tolerance and sweep count; use `settle` for full diagnostics. Invalid
arguments and nonrepresentable initial numeric quantities raise `ValueError`.

### Values, clamps and events

`inputs` maps every declared sensor exactly once, by name or by that brain's
`Input` handle. Values may be correctly nested lists/tuples or flat
lists/tuples with the declared element count. Objects with `tolist()` are
accepted without a NumPy dependency; an exposed shape must match the declared
shape or its flat vector shape. Scalar boundaries take scalar values. Samples
must be finite reals; booleans are rejected. Nothing is clipped or
preprocessed.

`targets` maps selected output names or `Output` handles to values of the
output shape. `interventions` maps selected population names or `Population`
handles to vectors of `population.patches` values. Clamps must lie inside
`state_bound`, and aliased output/intervention clamps must agree exactly on
shared patches or the call raises `ValueError`. Returned output values are
always flat tuples.

In `settle` and `step`, clamps condition the solve and never become learning
admissions. `step` retains the qualified state, including clamped coordinates;
the clamps apply to that call only. A goal-clamped output expresses an
intention; leave the output free when evaluating a forecast. Invalid arguments
or a refused solve preserve the entire continuation.

`observe` requires a nonempty target mapping and takes no interventions.
`source="witness"` labels observed targets and `source="estimate"` labels
derived teaching targets, including Q estimates; no other label is accepted.
Both use identical repair. The label is caller-declared provenance, appears
in every admission result and is part of the event identity; a batch uses one
label for all rows. Outputs returned by learning equal their clamps, so
measure learning with later unclamped predictions.

`event_id` is an ordered nonnegative integer within the runtime's JSON integer
limits; excessive digit counts raise `ValueError` before any solve. Omitting
it selects the latest admitted identifier plus one, starting at `0`. Gaps are
allowed. The latest accepted identifier may be retried with identical inputs,
clamps and source, which performs no solve and no second admission. Reusing it
with different content, or supplying an older identifier, raises `ValueError`.
Refused proposals consume no inferred identifier.

### Batch experience

`observe_batch` takes a nonempty finite sequence of `(inputs, targets)` pairs,
each following the `observe` rules; generators are refused. All rows are
validated before solving, and an invalid row changes nothing. Rows may clamp
different outputs or different values for the same output: they are separate
experiences.

A batch of size `B` has `B` private patch-state vectors, each starting from
the **same pre-call live state**, and shared weights and biases. Its objective
is the mean of the per-row residual-plus-state-prior energies plus one
parameter-prior penalty anchored to the pre-batch parameters. Rows interact
only through shared parameters.

Qualification tests each row's **unaveraged** state gradient and the shared
parameter gradient, so a larger batch does not loosen a row's tolerance. A
qualified batch commits shared parameters and one admission and leaves
`brain.state` unchanged, even for one row.

One ordered batch owns one `event_id` in the `observe` sequence, with a
digest over every row in order plus the source. An identical latest retry
through the same method performs no solve; any change under that ID conflicts.
Serial calls reanchor after each admission while a batch has one fixed
anchor, so the two generally yield different parameters: batch grouping is a
learning choice. Graph caps do not bound the extra row-state and temporary
storage a batch needs; choose a bounded batch size.

### Read-only properties

| Property | Value |
| --- | --- |
| `config` | Resolved configuration mapping. |
| `state` | Tuple of current patch values, concatenated in population declaration order. |
| `weights` | Tuple of coefficients in `graph.edges` order. |
| `biases` | Tuple of prediction offsets in patch order. |
| `graph` | Immutable topology with `n_inputs`, `n_patches`, `edges`, `incoming` and `residual_order`. |

An edge is `(kind, source, target)` with `kind` `"input"`, `"state"` or
`"residual"`. Input sources index the flattened sensor vector; other sources
and all targets index the combined patch vector. `incoming` holds edge-index
tuples per target and `residual_order` orders recomputation of derived errors.

### Solve results

`settle`, `step` and newly attempted `observe` calls return a dictionary:

| Key | Meaning |
| --- | --- |
| `outputs` | Names mapped to selected state tuples. Outputs of a refused solve are diagnostic proposals and must not drive actions. |
| `state`, `weights`, `biases` | Proposed final coordinate tuples. A pure query leaves the brain's properties unchanged. |
| `predictions` | Exactly recomputed `tanh` predictions, one per patch. |
| `errors` | Exact `state - prediction` values, one per patch. |
| `energy` | Final objective, including the parameter anchor penalty during learning. |
| `stationarity` | Maximum absolute projected-gradient component over eligible coordinates. |
| `prediction_residual` | Largest absolute prediction error; can be nonzero at qualified stationarity. |
| `qualified` | Whether a fresh final evaluation meets the stationarity tolerance. |
| `reason` | `"qualified"`, `"budget"` or `"line_search"`. |
| `sweeps` | Accepted repair sweeps. |
| `energy_history` | With the Python engine, initial energy followed by each accepted proposal's energy. Tensor execution records its approximate device trajectory followed by any reference refinement; that combined trace does not certify float64 monotonicity. |
| `work` | Algorithmic work counts, including attempted work on refused solves and both tensor and reference stages. |
| `execution` | Present for tensor solves: device, precision, tensor library version and work split; see below. |

`step` adds `accepted`, equal to `qualified`. A newly attempted `observe` adds
`accepted`, `duplicate=False`, `event_id` and `source`. An identical
latest-event retry returns only `accepted=False`, `qualified=True`,
`duplicate=True`, `event_id` and `source`; its qualification refers to the
prior admission.

A newly attempted `observe_batch` returns the same common diagnostics and
admission fields plus `batch_size`, with these per-row fields:

| Key | Batch meaning |
| --- | --- |
| `states` | Tuple of proposed patch-state tuples in example order. There is no `state` field, and these states are not committed. |
| `predictions`, `errors` | Tuples of per-row patch vectors in example order. |
| `outputs` | Tuple of output mappings in example order. Targets are clamped, so these are not scores. |
| `weights`, `biases` | One shared proposed vector of each kind. |
| `energy` | Mean row energy plus one shared parameter-anchor penalty. |
| `stationarity` | Maximum of all unaveraged row-state residuals and the shared parameter residual. |
| `prediction_residual` | Maximum absolute prediction error across every row. |
| `batch_size` | Number of examples in this atomic admission. |

An identical latest batch retry returns `accepted=False`, `qualified=True`,
`duplicate=True`, `event_id`, `source` and `batch_size`, without solve
diagnostics.

`work` contains `evaluations` (energy/gradient computations including the
final check, counted once when no proposal was attempted), `patch_visits` and
`edge_visits` (prediction, error and derivative traversals, including partial
work before a numeric failure), `proposals` (line-search attempts) and
`backtracks` (rejected proposals). Query-local reuse of sensory predictions
lowers forward edge visits. Measure elapsed time and memory separately.

Tensor `execution` contains `device`, `dtype`, `torch` (library version),
`tensor_sweeps`, `reference_sweeps`, `reference_evaluations` and
`reference_restart` (a float64 energy increase discarded the device candidate
and restarted from the original coordinates). Float32 device repair uses at
most `max(1, budget // 2)` accepted sweeps, leaving the rest for float64
reference refinement; `sweeps` counts both stages. Returned `energy`,
`predictions`, `errors`, `stationarity` and `qualified` always come from the
float64 reference check on the original data. The two-stage rule is in the
[specification](SPECIFICATION.md).

### Inspector result

`inspect()` returns layout `inputs`, `populations` and `outputs`, plus
`config`, `patches`, `input_samples`, `connections`, `edges`,
`observed_fields`, `sensor_coverage`, `output_connected_patches`,
`fingerprint`, `implementation`, `admissions` and `last_event_id`.

- Population records add `role` and global patch `indices`.
- `observed_fields` is `("state", "prediction_error")`.
- `sensor_coverage` counts unique input coordinates attached anywhere.
- Each output record adds `sensor_coverage_by_coordinate`, a flat tuple of
  distinct reachable sensor-coordinate counts in output order.
- `output_connected_patches` counts processing patches in any output's coupled
  component. State and residual connections join components in both
  directions; shared fixed inputs do not join independent patches.
- `implementation` maps each hashed source file to its SHA-256.
- `last_event_id` is `-1` before any admission.

Coverage is structural: zero weights, saturation or learned cancellation can
suppress a path. Editing the returned copies does not modify the brain.

## Numerical and learning contract

Each patch computes `p = tanh(b + sum(weight * signal))` and error `e = x - p`.
Queries minimize `sum(e²)/2 + state_prior * sum(x²)/2` over unclamped states;
learning adds `parameter_prior * sum((parameter - pre_experience_parameter)²)/2`
and makes weights and biases eligible. Repair is synchronized projected-gradient
descent with a secant step and Armijo backtracking, and a solve qualifies when
every eligible projected residual is within `tolerance`. Qualification means
bounded projected stationarity of a nonconvex energy: it does not certify zero
disagreement, a unique state or a global optimum. The equations, step rule,
batch objective and query cache are in the
[specification](SPECIFICATION.md#repair-and-qualification).

`observe` fits labeled targets. `Reinforcement` builds discrete-action Q
targets and bounded replay on that same primitive. Episodic retrieval,
protected consolidation, planning and structural growth are outside `observe`.

## bootstrap

```text
bootstrap(
    brain,
    examples,
    *,
    checks,
    max_error,
    epochs=20,
    seed=0,
    budget=None,
    batch_size=1,
)
```

Replays examples into an existing `Brain` until unclamped checks pass, and
returns a plain report. This is the **bootstrapping phase**; the **live
phase** continues with `observe` on the same brain. No solver mode changes
between them. Usage is in the [design guide](BRAIN_DESIGN.md).

| Argument | Contract |
| --- | --- |
| `brain` | Existing `Brain`. Earlier admitted experience is kept. |
| `examples` | Nonempty finite sequence of `(inputs, targets)` pairs in the `observe` format, each with at least one target. Rows may repeat; generators are refused. |
| `checks` | Nonempty sequence in the same format, used only for unclamped evaluation. They influence stopping, so reserve a separate final test. |
| `max_error` | Required finite nonnegative maximum absolute error in encoded output units, separate from solver `tolerance`. |
| `epochs` | Nonnegative maximum complete replay passes, default 20. Zero evaluates without learning. |
| `seed` | Nonnegative integer for a private RNG that shuffles example indices each epoch. Python's global random state is untouched. |
| `budget` | Nonnegative solve-budget override for every admission and check; `None` uses the brain's configured budget. |
| `batch_size` | Positive integer, default 1. One gives ordered `observe` admissions; larger values group shuffled examples into `observe_batch` calls, and the last group may be smaller. |

Before the first epoch and after each complete epoch, pure `settle` calls score
the examples (recall) and then the checks with **no targets clamped**. Only
targeted coordinates are compared, and agreeing output aliases count once per
pair. Every query must qualify and both maximum errors must meet `max_error`
to pass. A brain that passes at epoch zero returns without replay. A refusal
stops immediately and contributes no score.

All inputs are validated and copied before any solve; a malformed row fails
with its collection and index and changes nothing. Each presentation or
minibatch takes a fresh automatic event ID. Only each admission is atomic, so
a later refusal keeps earlier commits, and a new call starts a new shuffle
sequence without resuming an interrupted one. For external retry identities,
streaming data or custom metrics, call `observe`, `observe_batch` and `settle`
directly.

| Report field | Meaning |
| --- | --- |
| `options` | Requested `max_error`, epoch allowance, shuffle `seed` and `batch_size`, plus the resolved integer solve `budget`. |
| `passed` | All recall and check queries qualified and both errors met the limit, for the supplied cases only. |
| `reason` | `"passed"`, `"epochs"` (allowance exhausted) or `"refused"`. |
| `epochs` | Fully admitted replay passes; a partial pass is excluded. |
| `presentations`, `accepted` | Attempted and admitted example presentations in this call. |
| `updates` | Successful atomic admissions. A batch counts once here and by its row count in `accepted`; with `batch_size=1`, `updates == accepted`. |
| `examples`, `checks` | Counts of supplied rows. |
| `history` | Epoch-zero assessment and one after each complete epoch, each with `epoch`, `recall` and `checks`. |
| `work` | Summed solver work over admissions and checks, including refused solves. |
| `failure` | `None`, or `stage` (`"observe"`, `"observe_batch"`, `"recall"`, `"checks"`), original row `index`, solver `reason` and `stationarity`. For a refused batch, `index` is its first original index and `indices` lists all of them in batch order. |

Each assessment metric contains `evaluated`, `qualified` and `max_error`. A
refused query sets that collection's `max_error` to `None`; a collection not
reached after a refusal has metric `None`. The report is not stored in the
brain's snapshot. Exact replay needs the **starting** checkpoint, the original
examples and checks, the preprocessing and the report options.

## History: explicit temporal context

```text
History(size, *, steps=4)
```

`size` is the positive number of raw scalars per sample and `steps` the
positive window length. Encodings hold oldest-to-newest blocks of
`[sample values..., present]`, with `present=1` for a supplied sample and zero
for initial padding. The oldest sample is discarded when the window fills.
Missing values inside a sample need the caller's own encoding.

| Property or method | Contract |
| --- | --- |
| `input_size`, `steps` | Raw sample width and window length. |
| `size`, `shape` | Encoded width `steps * (input_size + 1)` and `(size,)`, for an input port. Encoded width is at most one million coordinates. |
| `push(values)` | Validate a finite flat sample, retain it and return the padded encoding as a tuple. An invalid sample leaves history unchanged. |
| `preview(values)` | Return the same encoding without retaining the sample. |
| `reset()` | Clear retained samples. |
| `snapshot()` | JSON (`kind` `"cadence.history"`, `version` 1) of dimensions and retained samples. |
| `History.from_snapshot(text)` | Validate exact schema, finite rows, dimensions and window bounds before restoring. Text limit: 32 MiB. |

Samples follow flat boundary validation, including objects with a valid
`tolist()` and shape; booleans and nonfinite values are refused. `History`
stores caller observations; it is neither neural nor episodic memory. Save its
snapshot alongside the brain's.

## LearningProgress: predictor-error reduction

```text
LearningProgress(*, rate=0.1, capacity=128)
```

`rate` is finite in `(0,1]` and integer `capacity` in `[1,4096]`; both are
read-only properties. `update(key, error)` takes a nonempty string key of at
most 256 UTF-8 bytes and a finite nonnegative error, keeps one moving mean per
key and evicts the least recently updated context when full. A new context
scores zero; later updates use

```text
new_mean = old_mean + rate * (error - old_mean)
progress = max(0, old_mean - new_mean) / max(1, old_mean)
```

The score lies in `[0,1]` and depends on error units. A constant stream scores
zero, as does an error at or above the current mean; random downward
fluctuations can score positively. Feed predictor errors with one consistent
definition. The helper does not update a brain. `snapshot()` (`kind`
`"cadence.learning-progress"`, `version` 1) and
`LearningProgress.from_snapshot(text)` keep means and eviction order; loading
validates exact fields, capacity, unique keys and finite errors within 32 MiB.
An invalid update changes nothing.

## Reinforcement: discrete reward-driven choices

```text
Reinforcement(
    brain,
    *,
    actions,
    action_input="action",
    value_output="value",
    discount=0.95,
    exploration=0.1,
    reward_scale=1.0,
    value_scale=0.9,
    capacity=1024,
    batch_size=16,
    credit_horizon=1,
    seed=0,
)
```

In the default action-conditioned form, `brain` needs an `action_input` sensor
with exactly `actions` coordinates and a `value_output` selecting one scalar
patch state. With `action_input=None`, `value_output` is a tuple/list of
exactly `actions` scalar output names on distinct patches; their values come
from one joint query and only the selected output is clamped when teaching.
Other inputs carry the situation, explicit history and drives. Values are
settled outputs with no separate policy head. `brain` and the read-only
`config` mapping are attributes; one serial owner controls both objects.
Usage and ownership rules are in the [live guide](LIVE.md).

| Argument | Contract |
| --- | --- |
| `actions` | Integer of at least two: the number of discrete choices. |
| `action_input`, `value_output` | Sensor name plus scalar output name for one-hot action-conditioned queries, or `None` plus a tuple/list of distinct scalar output names for one joint vector query. Lists become tuples in `config`. |
| `discount` | Finite discount in `[0,1)`. |
| `exploration` | Finite probability in `[0,1]` of a uniform action during exploratory selection. |
| `reward_scale` | Positive finite bound on reward magnitude. Larger rewards are refused, never clipped. |
| `value_scale` | Positive finite target scale, strictly below both `state_bound` and `1 / (1 + state_prior)`. |
| `capacity` | Positive bound on stored transitions; the oldest is discarded when full. |
| `batch_size` | Positive integer at most `capacity`. Replay samples up to this many records, always including the latest. |
| `credit_horizon` | Experimental bounded credit: positive integer at most `capacity`, the maximum consecutive recorded transitions per return. One gives the one-step target. |
| `seed` | Nonnegative integer for private exploration, tie-breaking and replay sampling. |

For reward `r` the target is a normalized discounted-return estimate:

```text
y = (1-discount) * value_scale * r/reward_scale
    + discount * clip(max_a Q(next_inputs, a), -value_scale, value_scale)
```

Terminal records omit the second term. With `discount=0` every target uses its
immediate reward only and replay makes no next-action queries, whatever the
horizon; nonterminal feedback records and validates the next observation all
the same. All targets use the pre-update parameters and are fitted with
`observe_batch(source="estimate")`, which preserves live activity. Replay
gives no guarantee of protected retention or convergence.

With `credit_horizon > 1`, a return follows the next recorded transition only
if its observation exactly matches the preceding next observation, it belongs
to the same episode and its executed action is greedy under the same frozen
pre-update values (exact ties count as greedy). At an off-policy action,
missing future, context gap or the horizon, it bootstraps from the preceding
next observation; at a terminal record the future value is zero. For a path of
length `n` the target is

```text
sum(k=0..n-1, discount**k * (1-discount) * value_scale * reward[k]/reward_scale)
    + discount**n * bootstrap
```

This is a finite greedy-cut return in the spirit of Watkins traces; see
[Munos et al. (2016)](https://arxiv.org/abs/1606.02647) for off-policy return
operators, whose tabular guarantees do not carry over to nonlinear repair.
`reset()` cuts temporal links without clearing replay.

| Method | Contract |
| --- | --- |
| `act(inputs, *, explore=True, budget=None)` | Supply every sensor except a declared action input. Query each action (conditioned mode) or all value outputs jointly (vector mode), require qualification, choose an epsilon-exploratory or maximum-value action with uniform tie-breaking, and retain qualified activity with `step`. A pending action needs feedback or `reset` before the next `act`. |
| `feedback(reward, next_inputs=None, *, decision_id, executed_action, terminal=False, learn=True, budget=None)` | Acknowledge an issued decision and record the action actually executed, which may differ from the proposal. Nonterminal outcomes need next inputs; terminal outcomes need `None`. A first valid acknowledgment consumes the pending decision even if fitting then refuses. An identical retry of the latest outcome neither records nor learns. `learn=False` records without fitting. Invalid or conflicting arguments change nothing. |
| `replay(*, budget=None)` | Attempt one sampled batch update from retained records, sampling uniformly without replacement besides the mandatory latest record. A next-action query or fit refusal commits no parameters; retry with this method, not by resubmitting feedback. |
| `reset()` | End the episode and credit segment and discard a pending action, without inventing a reward or clearing records, RNG, parameters or live activity. |
| `inspect()` | Copy of `config`, current `records`, cumulative `transitions`, successful `updates`, boolean `pending`, current `episode`, `issued_decisions`, `pending_decision_id`, `pending_action` and `last_feedback_id`. Absent identities are `None`. |
| `snapshot()` | Bounded JSON with schema `reinforcement/3`: brain, replay records with episode and decision identifiers, pending proposal, decision counter, latest outcome receipt, RNG and counters. |
| `Reinforcement.from_snapshot(text)` | Validate the complete continuation, including the reinforcement source hash and the brain's source identity. Text limit: 32 MiB. No device or dtype override. |

Successful `act` returns `accepted=True`, integer `action`, positive integer
`decision_id`, candidate `values` (scaled return estimates), boolean
`exploratory`, the selected `settlement` and `work`. A query or step refusal
returns `accepted=False`, `action=None`, `decision_id=None`, `values`, `work`
and reason `"query_refused"` or `"step_refused"`, leaving pending state and
RNG unchanged.
`explore=False` disables epsilon moves and breaks exact ties randomly. Every
successful decision consumes RNG.

A completed replay returns the batch result plus sampled `indices`, derived
`targets`, `credit_horizons`, `credit_stops` and cumulative `updates`; its
`work` sums every next-action query and the fit, and a refusal restores the
RNG. Stop
reasons are `terminal`, `zero_discount`, `horizon`, `pending_future`,
`discontinuity` and `off_policy`; `zero_discount` marks an immediate-reward
nonterminal target. Earlier refusals return `accepted=False` with
`"empty_replay"` or `"bootstrap_refused"`.

A first `feedback` acknowledgment adds `stored=True`, `duplicate=False`,
`decision_id` and cumulative `transitions`; with learning disabled it has
`accepted=False` and reason `"learning_disabled"`. A duplicate returns
`accepted=False`, `stored=False`, `duplicate=True`, `decision_id`,
`transitions`, `reason="duplicate_feedback"` and `work={}`, with no
settlement or `qualified` field. Each `budget` applies to each underlying
solve. An action-conditioned `act` costs `actions` queries plus one `step`;
vector mode costs one query plus one `step`.

Pass `act`'s `decision_id` and the action the body performed as
`executed_action` to `feedback`, or call `reset` for a discarded command.
Decision IDs increase only after accepted decisions and are never reused.
`decision_id` is a positive integer and `executed_action` an integer in
`[0, actions)`; booleans are refused. Only the
latest acknowledged outcome can be retried: identical action, reward, next
context and terminal status return `duplicate=True` and leave a newer pending
decision intact. Changed payloads and older, canceled or unissued IDs are
refused atomically. `learn` and `budget` are execution options, so changing
them on a duplicate never triggers learning.

The latest outcome receipt survives replay eviction and snapshots. Decision
identity is separate from the brain's event identity. `learn=False` records a
transition, which can evict the oldest record; snapshot a separate learner for
an evaluation that must not touch the live replay, RNG or activity.

## LiveController and actuator limits

```text
LiveController(callback, *, fallback, max_age=0.25, clock=time.monotonic)
slew(current, target, *, rate, dt)
```

`callback` runs as `callback(observation)` on one daemon worker, started by
the constructor, and returns a
mapping with exact boolean `qualified`. A qualified result needs a finite
nonempty `command` sequence of the same length as `fallback`; a refused result
may omit it. Extra fields, including `accepted`, are ignored, so translate a
policy's result explicitly. Exceptions and malformed results count as errors
and later work continues. The worker must own every brain, history or
reinforcement helper the callback uses.

`fallback` and `command` are finite nonempty Python sequences; arrays are
refused, so convert them with `tolist()`. `max_age` is positive finite
seconds. `clock` is a fast callable returning finite nondecreasing seconds.
Scheduling is best-effort: the GIL, locks, copying and validation take time,
and an active callback cannot be cancelled. Usage is in the
[live guide](LIVE.md).

| Method | Contract |
| --- | --- |
| `submit(observation)` | Copy finite JSON-like data and return a positive submission ID without waiting. At most one observation waits behind the active callback; a newer submission replaces it. Submission after closure raises `ValueError`. |
| `read()` | The command from the latest fresh qualified completion; otherwise `fallback` (before any completion, after refusal or error, after expiry or after closure). A fresh earlier command is returned while newer work waits. |
| `inspect()` | Copy of counters, worker and queue state, status, result age and latest timing and error diagnostics. |
| `close(*, wait=True, timeout=1.0)` | Stop submissions, drop pending work and let active work finish. With `wait`, block at most `timeout` finite nonnegative seconds (at most `threading.TIMEOUT_MAX`). Returns whether the worker exited. |

Age runs from submission, so queue and solve time count. Observations are
string-keyed dictionaries, lists, tuples, strings, booleans, finite numbers
and `None`; cycles, arrays and other types are refused. The replaceable
pending slot is for sensory refreshes. **Do not put reward or transition
records in it**; store those losslessly and deliver them in order to the
serial owner. Expiry or closing does not undo changes made by a running
callback, and a false `close` result means the callback owns its brain until
the worker exits. There is no runtime checkpoint.

Counters are `submitted`, `dropped`, `started`, `completed`, `qualified`,
`refused` and `errors`. Other fields are `closed`, `busy`, `pending`,
`worker_alive`, `status`, `last_result_id`, `result_age`,
`last_solve_seconds`, `last_latency_seconds` and `last_error`. Status is
`"waiting"`, `"qualified"`, `"refused"`, `"error"`, `"stale"` or `"closed"`.
Solve time includes result validation; latency adds queue time. Timing and
result identity are `None` until available. `dropped` counts replaced and
closed-out pending work.

`slew` returns a tuple moving each `current` coordinate toward `target` by at
most `rate * dt`. Vectors are finite, nonempty Python sequences of equal
length; `rate` is a nonnegative scalar or equal-length sequence and `dt` finite
nonnegative seconds.
Nonfinite products are refused. It limits actuator speed and chooses no
target.

## Checkpoints

A `Brain` snapshot (schema `population-brain/1`) contains the full
configuration and layout, graph fingerprint, live state, weights, biases,
admission count and latest event identity and digest. It binds the SHA-256 of
`brain.py`, `cortex.py`, `column.py`, `ports.py`, `_repair.py`,
`_validation.py` and `_tensor.py`. Loading refuses different hashes even when
version labels match, so a comment-only edit to one of these files breaks
loading. Keep the exact source with saved models and never rewrite stored
hashes.

Application continuation also needs the observation encoding and normalizer,
action decoder, external history, per-call budget overrides, action timing and
environment identity. The snapshot saves the configured `settle_budget` and
none of these. See
[whole-life saves](LIVE.md#save-a-whole-life-and-run-the-small-gates).

Loading checks structure, configuration, deterministic topology, array
lengths, finite values, bounds and event-ownership consistency, and builds a
complete proposal before replacing anything. A loaded brain has new handles;
address it by names. `restore` keeps the receiving brain's handles and
requires the complete configuration, including device and dtype, to match.

`Brain.from_snapshot(text, device=..., dtype=...)` validates the original
snapshot first, so an override cannot bypass an invalid fingerprint, source
mismatch or malformed array. With both overrides omitted, saved settings are
kept. `device` alone selects that device's default precision; `dtype` alone
keeps the saved device. Arrays, bounds, topology and event ownership are kept
exactly; the configuration fingerprint changes. Float32 does not round stored
arrays, and hardware availability is checked on the first solve.

JSON loading executes no code. A valid checkpoint does not authenticate its
maker, prove its witness history or certify current stationarity: check
provenance separately and solve before acting. The 32 MiB text limit does not
bound peak decoding or construction memory.
