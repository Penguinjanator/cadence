# Numerical and learning contracts

Each model family in Cadence solves its own equations, stops by its own rule and changes
its parameters under its own contract. This page sets them side by side. Read the row for
the family you use before reading a result off it, and before changing a solver or a
learning rule.

## Choose the computation

| API | Equations | Stopping rule | Update contract |
| --- | --- | --- | --- |
| `NeuralGraph`, `Learner` | Rate neurons exchange activity over a directed graph ([concepts](concepts.md#settling-and-equilibrium)). | `equilibrate` checks the full fixed-point equations; `settle` alone may run a fixed budget or stop on activity movement. | Free/nudged local contrast, finite phases by default; optional reward traces. The gradient reading needs symmetric effective weights, a smooth stable branch and a small nudge ([certificate](certificate.md#equilibrium-propagation-scope)). |
| `Brain` (composed) | One recurrent graph, with System 2 observers in the same equations and held trace, record and recall input. | `act`, `predict`, `accuracy` and `wait` need the full potential/adaptation residual below `learning.tolerance` within `learning.free_steps`; exhaustion refuses the answer. | The learner's contrast and the actor-critic's reward eligibility (always finite phases, `reward.eligibility_steps`), plus explicit trace and synaptic-memory writes. |
| `PatchNet` | The reciprocal graph core with persistent free activity and explicit evidence ports. | Free and nudged phases (two when `centered`, the default) must pass the equation residual. | Commits only when every phase qualifies; no replay-loss check ([PatchNet](patchnet.md#observed-ports-and-local-detuning)). |
| `TemporalPatchNet` | A causal free path and jointly repaired teaching paths over a finite window. | Whole-path residual and branch checks. | Centered contrast of parameter derivatives. The two teaching paths must sit centered on the free path (`symmetry_tolerance=0.15`), else beta halves up to `max_halvings=8` times and the step is refused; `backtrack=True` also requires a loss decrease on the observation ([temporal](temporal.md#checking-that-the-contrast-is-centered)). |
| `RecordPatchNet` | A gated causal context scan with local record reads. | The causal path solves its declared free equations. | `observe` uses an adjoint backward scan; record writes use a local delta rule. `detune` solves the two quadratic teaching phases as a check and changes nothing ([record patch](record-patch.md#detuning-as-the-acceptance-check)). |
| `RecordPatchStack`, `JointRecordPatches` | Two contexts in depth, or several patches joined by ports. | The stack runs its two causal scans; joined patches run a fixed number of Jacobi `rounds` (default 1) per moment, and the result need not be an equilibrium. | An adjoint through both scans, or through the moments and rounds (across ports with `cross_adjoint=True`, the default); records stay outside the gradient ([joined patches](record-patch.md#several-patches-joined-by-ports)). |
| `BeliefPatch` | A transition followed by a fixed number of damped nonlinear repair iterations per observation. | The iteration count. `residual` is the last damped move. | An adjoint through the iterations and the window; a step on a target is taken only after a replay of the chunk lowers its loss by the Armijo margin (`backtrack=False`, or an `output_gradient` in place of a target, takes the plain step) ([belief](belief.md#an-admitted-step)). |
| `Steered`, `Life` | A cortex, a steering patch and an optional governor executed in sequence; readbacks are taken as given, and no joint observer-observed fixed point is solved. | Each participant's own rule. | The participants' updates under one joint admission with replay; `Life` undoes a learn window whose loss did not fall ([steering](steering.md#the-switch-and-the-accounting)). |
| Population solver (`cadence.experimental.equilibrium`) | Populations of processing patches under one stated nonlinear energy. | Projected-gradient steps with sufficient-decrease backtracking; qualifies when the full projected stationarity residual is at most `tolerance` on every eligible coordinate. | `observe` solves states and learned relations together and commits only a qualified proposal ([specification](equilibrium/SPECIFICATION.md#repair-and-qualification)). |

The experimental [partitioned temporal model](partitioned.md) adds fixed masks to
`TemporalPatchNet` and keeps its contract; `TemporalMemory.observe` refuses it.

### Component limits

| Component | Implemented operation | Limit |
| --- | --- | --- |
| `Brain` (composed) | Qualified settles for every answer, local contrast and reward learning, working trace, synaptic and opt-in episodic and recall memory | A qualified answer covers the present settle. Memory enters as held input and is written after the settle, outside the equations; behaviour across ticks needs [persistence tests](evaluation.md#test-coherent-behavior-over-time). |
| `RecordPatchNet` | Gated causal context, residual records, adjoint learning, private reads and fixed-dream sleep | The context does not read record outputs; no action-port planner and no learned memory relevance |
| `RecordPatchNet.detune` | Centered contrast for the linear-readout slow patch | Check both solves; categorical ports are refused, and records stay outside the energy |
| `RecordPatchStack` | Two contexts in depth with an adjoint through both | No joint `detune` solver, and no guarantee for a jointly free nonlinear energy |
| `JointRecordPatches` | Patches joined by ports for a fixed number of rounds | Finite rounds carry no all-patch convergence theorem |
| `TemporalPatchNet` | Learned temporal relationships, private imagination and bounded continuous-port planning | Model accuracy and executed outcomes need separate validation |
| `BeliefPatch` | Finite repair with an admitted step | A finite repair budget is no convergence certificate |

A private `imagine` call is a component operation. The application defines what its inputs
and outputs mean and which future variables it generates or supplies; a learned simulator
needs its own validation, and a value evaluator inside an exact rules-based search is a
value evaluator.

To check a fixed-point claim, evaluate the declared equations at the returned state: finite
values and the largest relevant defect, including slow states when the model has them. A
small integration step can make activity barely move while the equation defect stays large.
A small residual shows the state satisfies the equations within tolerance; uniqueness,
stability and useful behaviour need their own tests.

Numerical qualification, parameter commitment and measured learning improvement are three
separate facts. `PatchNet.observe` qualifies its phases before it updates and does not
replay the update. APIs with replay admission accept only steps that lower the replayed loss
on the declared observation or window. Neither establishes retention on earlier tasks or
generalization; [evaluation](evaluation.md) covers those measurements. Eligibility, reward
and source identity stay attached to the action that was actually executed.

## Brain and Learner

- A composed `Brain` answers only from a qualified settle (1024 sweeps, residual `3e-3`, with
  a half-step fallback, or `damping` halvings under `qualified=True`); a refused `act` issues no action and changes no state ([qualification and refusal](continuous.md#qualification-and-refusal)).
- `LearnerConfig` teaches with finite phases by default. `qualified=True` is the opt-in that
  gates each phase on the full equations, with at most `damping=3` dt halvings by default. Give it a
  `nudged_steps` budget comparable to `free_steps` ([qualified teaching budgets](learning.md#qualified-teaching-budgets)).
- RMS-normalized rates are covered in [rates under normalization](learning.md#rates-under-normalization).

## State and production integration

Keep live state, learned parameters, record tables, optimizer and step-size state, random
state and accounting separate, and save every one needed for an exact continuation,
including the boundary of a replay window. A custom callable policy or weighing needs its
own reproducible configuration; an array snapshot cannot serialize application code.

Calls validate shapes and finite values before mutating anything. An invalid call or a
failed transaction installs no parameter and no record. Counters of attempted work may
advance on a refused update; report them as attempted work, never as accepted learning. An
`imagine` call does not teach or alter live activity, though it can count its work.

`Connectome` arrays and its population mapping are read-only. `NeuralGraph.efficacy`,
`bias`, `log_gain` and the effective `weights` return read-only views. Replace parameters
in one of two ways:

- assign a complete array to `efficacy`, `bias` or `log_gain`; the setter validates it and
  refreshes the running backend;
- call `graph.with_parameters(efficacy=..., bias=..., log_gain=...)`, which returns a new
  graph on the same connectome; an omitted argument keeps the current array, and a new
  `log_gain` rebuilds every derived weight.

Do not mutate parameter or connectome arrays in place: derived transport caches assume
their values stay fixed. For topology changes, build a new `Connectome`; for population
changes, call `connectome.with_populations(name=indices)`, which returns a new connectome
with the named populations merged in. After a device update, always read `learner.brain`:
an older `NeuralGraph` object can share the updated device kernel while holding stale host
copies. Other model families change parameters through their own `set_parameters` or update
interface. Pin the library revision and backend for reproducible results and run a parity
check when changing device or precision ([backends](backends.md), [API](api.md),
[receipts](receipts.md)).

## Recursive observation

An observer belongs to the same equilibrium as the system it observes only when its state
and feedback enter the same coupled equations and stopping check; the
[recursive-settlement example](recursive-settlement.md) builds one with `PatchNet`.

## Cost counters

The `BeliefPatch` and `Steered` counters (`cost`, `macs_per_moment`, `readback_macs`) are
dense forward-work estimates. They include moments replayed in an admission, and `Steered`
adds the readbacks and declared `Rule` work it executes; a direct `BeliefPatch.readback()`
call counts nothing. They exclude adjoints, record writes, nonlinearities, sorting,
optimizer and admission bookkeeping, application callbacks and memory traffic. `imagine` is
counted with the same moment model although it performs fewer operations. A zero gain does
not make the dense port kernel sparse. `Life` reports governor settling steps as moment
equivalents (steps times governor synapses over the brain's `macs_per_moment()`), a declared
cost model. Use full profiling for a training-efficiency claim; [evaluation](evaluation.md)
lists what a cost comparison counts.
