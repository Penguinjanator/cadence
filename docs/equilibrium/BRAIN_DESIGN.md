# Design and teach a population brain

This page continues the [quickstart](QUICKSTART.md): what to give the brain,
how to wire it, how to teach a skill and keep it, and what to measure. Signatures
are in the [API reference](REFERENCE.md), equations in the
[specification](SPECIFICATION.md). The examples share one namespace.

## Define the body's information and outcomes

Write down what the brain observes, what each output means, what the body
executes and which later observation establishes success. More patches cannot
reconstruct information that never reaches them: one camera frame omits velocity
and the last turn. One body adapter supplies every named sensor and receives
every named output; the application defines objective, reward units and
actuator behaviour.

Scale inputs and targets well inside the `tanh` range (with default bounds,
values near `-0.6` and `0.6` work). Fit preprocessing on teaching data only and
save it with the brain. Split recordings by episode before cutting windows, and
keep development and assessment sets apart.

The example body on this page answers `answer = 0.6 * signal`. Teaching,
development and assessment values are distinct:

```python
from cadence.experimental.equilibrium import Cortex, bootstrap

def rows(values):
    return [({"signal": [x]}, {"answer": [0.6 * x]}) for x in values]

examples = rows((-0.8, -0.4, 0.4, 0.8))   # admitted as teaching
checks = rows((-0.6, 0.6))                # development: decide when to stop
assessment = (-0.7, -0.2, 0.2, 0.7)       # final, no targets
```

## Wire a connected brain

Every patch predicts `p = tanh(bias + weighted incoming signals)` and carries
the error `state - p`. `column(..., inputs=...)` reads sensors or the live states
of earlier populations. An output exposes selected patch states; there is no
separate output network. `build()` requires at least two populations joined into
one connected graph and refuses a population or group that would settle alone.

```python
layout = Cortex(seed=2)
signal = layout.input("signal", shape=1)
features = layout.column("features", patches=4, inputs=signal)
response = layout.column("response", patches=1, inputs=features)
layout.output("answer", shape=1, reads=response)
brain = layout.build()

description = brain.inspect()
assert description["output_connected_patches"] == 5
assert description["outputs"][0]["sensor_coverage_by_coordinate"] == (1,)
```

| Choice | Construction | Meaning |
| --- | --- | --- |
| Width | `column(patches=256, inputs=...)` | 256 processing states and their incoming relations |
| Intermediate population | A column reading another column | A learned composition; downstream relations reshape upstream states through the joint energy |
| Sensory branches | Columns reading different sensors, read by a later column | Separate representations settling in the same solve |
| Direct paths | `inputs=(eyes, perception)` | A later population reads sensors as well as earlier states |
| Error readback (experimental) | `observer(observes=...)` | States and exact current errors of the observed populations |

Patches within one population do not read each other. Selecting one patch as an
output gives it no access to its neighbours, so widening a population while
reading only its first state adds no hidden representation: connect a second
population to create one. For several scalar outputs from one population, pass
distinct `indices`; each defaults to patch zero. Start with defaults and the
smallest brain that builds; for a small nonlinear relation, 4 to 16 connected
representation patches feeding one output patch per output coordinate is a
starting search range.

Sensory branches are the starting layout for a multimodal task:

```python
cortex = Cortex(seed=7)
eyes = cortex.input("eyes", shape=(4, 4))
ears = cortex.input("ears", shape=(4,))
vision = cortex.column("vision", patches=8, inputs=eyes)
hearing = cortex.column("hearing", patches=8, inputs=ears)
fusion = cortex.column("fusion", patches=8, inputs=(vision, hearing))
cortex.output("move", shape=2, reads=fusion)
senses = cortex.build()
assert senses.inspect()["patches"] == 24
assert senses.inspect()["sensor_coverage"] == 20
```

`inspect()` also reports `input_samples` and `connections`. Its counts are
structural paths that saturation or learned cancellation can suppress; perturb
inputs and measure free outputs before relying on one.

## Experimental recursive observer settlement

`observer(..., observes=...)` reads the observed populations' states and their
exact current prediction errors. An observer can observe another observer and can
also take ordinary `inputs`. Every relation joins the same energy: higher
populations do not wait for lower ones and then issue a separate correction.
Ordinary state contacts return influence as well; readback adds a current error
channel, recomputed as states change, and holds no record of a past forecast.

```python
recursive = Cortex(seed=7)
signal = recursive.input("signal", shape=1)
representation = recursive.column("representation", patches=4, inputs=signal)
integration = recursive.observer("integration", patches=2, observes=representation)
reflection = recursive.observer("reflection", patches=1, observes=integration)
recursive.output("answer", shape=1, reads=reflection)
recursive_brain = recursive.build()

report = bootstrap(recursive_brain, examples, checks=checks,
                   epochs=20, max_error=0.1, batch_size=4, seed=2)
assert report["passed"], report
```

An action need not come from the highest observer. In a mixed layout ordinary
columns produce the answer and `observer("monitor", patches=2,
observes=(integration, response))` reads their states and errors; its relations
return influence through the same settlement, with no second body interface.

Whether readback helps a task is open. Reading a clamped motor's error adds a
parameter-learning path into the motor's own relations, which changes what is
learned and assigns no reward credit. Compare each observer layout with an
equally informed state-coupled control at matched parameters and total work;
equal patch counts need not mean equal parameters, connections or work. With
direct sensor paths, a policy can use the bypass while its observers stay
numerically active: perturb sensory inputs with history held fixed, and treat
a detached lesion as a sensitivity measurement, distinct from a trained
control. Zero error leaves an observer's influence in place. The runnable
example is `layout_learning.py --layout recursive`.

## Connectivity and cost

| Setting | Default | Guidance |
| --- | --- | --- |
| `fan_in` | `None` (every source coordinate) | A positive integer requests sparse wiring. It is capped at the source width and raised until the destination population covers every source coordinate. |
| `max_connections` | `1000000` | Bounds compiled directed connections; `max_patches` (`10000`) and `max_inputs` (`1000000`) bound patches and sensor samples. The builder refuses and never drops inputs. |

Dense input wiring costs input width times population width; dense population
wiring costs the product of both widths. An observer adds error contacts to its
state contacts; reading the same population through `inputs` and `observes`
shares the state contact. A camera shows the scale:

```python
camera = Cortex(seed=1)
frame = camera.input("camera", shape=(240, 320, 3))
retina = camera.column("retina", patches=256, inputs=frame)
camera.output("move", shape=2, reads=camera.column("motor", patches=2, inputs=retina))
try:
    camera.build()  # 256 * 230,400 = 58,982,400 connections
except ValueError as error:
    assert "Connection budget exceeded" in str(error)

small = Cortex(seed=1, fan_in=8)
frame = small.input("camera", shape=(24, 32, 3))
retina = small.column("retina", patches=64, inputs=frame)
small.output("move", shape=1, reads=small.column("motor", patches=1, inputs=retina))
sparse = small.build().inspect()
assert sparse["sensor_coverage"] == 2304 and sparse["connections"] == 2368
```

Here `fan_in=8` is raised to 36 so that 64 patches cover all 2,304 samples.
Sparse wiring that splits the compiled patches into separate components is
refused (shared sensors do not couple patches); increase `fan_in` or add a
population reading across the components. `settle_budget` (default `2048`)
counts accepted repair sweeps; evaluations, rejected proposals and final
qualification add work. Devices and measured costs are in
[execution and cost](ACCELERATION.md).

## Make temporal information explicit

When the latest sample is ambiguous, give the brain a bounded `History` window
([usage](LIVE.md#supply-recent-context-explicitly)). It is explicit input memory
whose size counts in any comparison; choose its length from the task's causal
horizon and sampling rate. `step` retains activity between calls, which alone is
no evidence of temporal memory: the builder reads only previously declared
sources, so it has no recurrent state cycles, and `observe_batch` rows gain no
order from adjacency. To claim learned memory, remove the extra history in a
declared control, freeze learning during recall and test writes, retention,
replacement and reset.

## Teach a skill

Each teaching row has a meaning, and the operation must match it:

| Available evidence | Operation | What it can acquire |
| --- | --- | --- |
| Measured sensor or body consequence | `observe` / `observe_batch` with the default `source="witness"` | Prediction of that consequence |
| An actual demonstration action | The same witness operations | Imitation of that action from the supplied context |
| A transformed target, prediction or simulator preference | The same operations with `source="estimate"` | The declared estimated relation |
| Reward after an executed discrete action | `Reinforcement` act, feedback, replay | Estimated action values; see [live use](LIVE.md#preserve-execution-and-feedback-ownership) |

A reward is no motor target. Teaching the brain its own executed commands trains
imitation, even when the outcome was poor. Source labels bind retry identity;
they do not authenticate evidence. Missing outcomes are not zero rewards.
Sampling shapes what is learned: a predictor taught mostly on selected events
cannot recover how often they occur across all opportunities, so keep factual
forecasts representative. Which operation retains activity, parameters and
events is tabled in the [quickstart](QUICKSTART.md#3-save-it-and-use-it-live).

`bootstrap(brain, examples, *, checks, max_error, epochs=20, seed=0,
budget=None, batch_size=1)` replays examples until every free query qualifies
and recall of `examples` and the development `checks` both meet `max_error`,
in output units:

```python
report = bootstrap(brain, examples, checks=checks, max_error=0.1,
                   epochs=20, batch_size=4, seed=2)
assert report["passed"], report
assert report["accepted"] == 4 * report["updates"]

for x in assessment:
    query = brain.settle({"signal": [x]})  # no answer clamp
    assert query["qualified"]
    assert abs(query["outputs"]["answer"][0] - 0.6 * x) < 0.1
```

It copies the dataset before changing the brain, checks readiness before
teaching and after each epoch, and sets `report["reason"]` to `"passed"`,
`"epochs"` or `"refused"`; admissions before a refusal stay committed.
`epochs=0` only checks readiness, `seed` fixes the per-epoch shuffle and
`budget` overrides `settle_budget` for each solve. `history` holds per-epoch
errors and `failure` the refusal. Solver `tolerance` (default `1e-6`) bounds
stationarity: it certifies no accuracy, and loosening it adds no skill.

| Counter | Meaning |
| --- | --- |
| `presentations` | Attempted example rows, including a refused group |
| `accepted` | Example rows in admitted calls |
| `updates` | Accepted atomic learning calls |
| `work` | Summed numerical work, including readiness queries |

The default `batch_size=1` uses ordered `observe` calls, which retain activity
and parameters. A larger size uses `observe_batch`: rows have private activities
and share one parameter anchor, live activity is unchanged and a refused batch
retains nothing. Batch size changes the learning trajectory, so four admitted
rows can be one update and equal presentations need not mean equal work. For
external experience that may be retried, pass increasing `event_id`s:

```python
live_state = brain.state
update = brain.observe_batch(examples, event_id=100)
assert update["accepted"] and brain.state == live_state

retry = brain.observe_batch(examples, event_id=100)  # identical latest event
assert retry["duplicate"] and not retry["accepted"]
```

An identical retry of the latest event teaches nothing twice; older or changed
identities raise `ValueError`. An omitted ID is allocated only on acceptance and
identifies no external retry. See [batch experience](REFERENCE.md#batch-experience).
For large recordings, feed bounded row lists to `observe_batch`: bootstrap's
`batch_size` limits each joint solve, while the whole dataset is copied up front.

| Setting | Default | Effect and when to change it |
| --- | --- | --- |
| `initial_scale` | `0.3` | Initial weights are uniform in `±initial_scale / sqrt(incoming contacts)`; it must not exceed `parameter_bound` (`4.0`). More scale gives nonlinear curvature and makes settlement harder. |
| `state_prior` | `0.01` | Activity penalty. It changes preferred answers as well as speed; a free unread output settles at `tanh(drive) / (1 + state_prior)`. |
| `parameter_prior` | `0.1` | Anchors each admission to the parameters at its start. Larger values resist change and need more exposure; this is no long-term protection. |
| `state_bound` | `1.0` | Box for states and clamps. Raising it permits larger clamps without expanding `tanh`: a clamped target of `2` can qualify and be unreachable free. |

Compare a small declared set of settings on development data and freeze the
choice before assessment. For classes, expose one score patch per class and
decode with argmax over legal actions; scores are not probabilities.
Rebalancing rare actions is a curriculum choice; check rare and ordinary
decisions afterwards.

## Continue learning without losing the earlier skill

An accepted update can improve its clamped objective while worsening free
answers or an earlier skill. Interleave earlier examples with new experience and
recheck both skills without targets; neither `parameter_prior` nor replay
guarantees retention. A candidate restored from the actor's snapshot can be
assessed before it replaces the actor, which is kept for rollback.

Then run the deployed output and decoder with the teacher disconnected.
Agreement on demonstration states hides failures on the states the learned
policy reaches. Label demonstrations collected there as a supervised curriculum
and compare with an equal-exposure replay control. For control tasks, measure
routine behaviour, a real disturbance, recovery and retention after further
learning; [live use](LIVE.md) covers the body loop.

## Keep numerical error, forecast surprise and task value separate

| Measurement | What it tells you | What it cannot establish |
| --- | --- | --- |
| `stationarity` and `qualified` | Every eligible coordinate meets the repair tolerance under this call's clamps | Correct predictions, good actions or a unique answer |
| `prediction_residual` and patch `errors` | Current disagreement between state and local prediction | Whether an earlier forecast was contradicted |
| Forecast surprise | Difference between a stored issued forecast and the later observation | Whether that outcome is good or bad |
| Reward or goal deficit | Value or shortfall in declared task units | Credit for each population |

Store each forecast with its context, action and model identity before the
outcome arrives; recomputing it after learning erases the error you wanted.
A patch's local prediction and its settled state can differ under priors and
coupling, so declare which of the two your application issues as the forecast.
`LearningProgress` summarizes changes in supplied predictor errors and detects
no surprise by itself. A predictable failure carries zero surprise and needs
correction all the same. Clamped targets change the settling problem, so
evaluate with the answer free.

## Measure speed and retained correction

Each call solves synchronously. Every participating population stays in the
energy and every eligible coordinate in final qualification, so naming a branch
`fast` schedules nothing and an observer adds latency to every solve. Parallelize
independent lives; serialize calls to one brain. Query-cache rules are in the
[specification](SPECIFICATION.md#repair-and-qualification).

Count queries, candidate actions, teaching, replay, checks and refusals, summing
primitive `work` counters or a helper's aggregate, never both. Report task
quality, qualification rate, latency and wall time with `evaluations`,
`edge_visits`, `patch_visits`, `proposals`, `backtracks` and `sweeps`; a sweep
budget is no time deadline. For a correction test, establish competent routine
behaviour, disturb the actual body, measure recovery, then freeze learning and
test retained quality and later work. Include a routine-plus-factual-fit control
and count failed correction attempts.

## Diagnose before adding compute

| Observation | Check next |
| --- | --- |
| A solve refuses | `reason`, `stationarity`, `work`, bounds and budget headroom, from the unchanged continuation; report any tolerance change |
| Solves qualify but teaching examples are not recalled freely | Input sufficiency, output connectivity, target units, useful capacity, anchoring and exposure |
| Recall passes but new cases fail | Split independence, coverage, decoder and the states the learned controller reaches |
| A temporal task fails | Whether the supplied observation or history distinguishes the answers, before blaming depth |
| A new skill displaces an old one | Replay coverage, update frequency, parameter anchoring and explicit retention checks |
| Depth or observers add work without benefit | Keep the simpler qualified model; compare against an equally informed ordinary control at total task cost |

More exposure, depth or hardware cannot repair absent information or a
disconnected representation.
