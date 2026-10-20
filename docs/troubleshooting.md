# Troubleshooting

Short answers to the problems builders meet first, each linking the page that owns the topic.
[Common missteps](evaluation.md#common-missteps) lists the ways a good-looking number can be wrong.

## Install and run

**`pip install cadence-net==0.84.3`, then `import cadence`.** Python 3.11 or newer and NumPy are
the only requirements.

**Do I need a GPU?** No; everything runs on NumPy float64. `[fast]` adds Numba and SciPy,
`[accel]` torch and `[apple]` MLX for the settling brain. `PopulationPatch` is the record patch's
torch twin for one-moment paths ([population](population.md)), `TorchBelief` the belief patch's
([backends](backends.md)). The temporal patch has no GPU path.

**Which interface do I want?** `Brain.compose(..., arousal=True)` and `brain.live(...)` for one
continuing stream; `step` for explicit teaching, batches or learning from every outcome. Other
contracts are in the [specialist guides](quickstart.md#specialist-guides).

## Shapes and encodings

**`ValueError` on shapes.** A settling brain takes a drive `(batch, n)`, one column per neuron,
and `Learner.step` integer labels `(batch,)`. Temporal, record and belief patches take paths
`(batch, time, ports)` and targets `(batch, time, outputs)`; one moment is `(batch, 1, ports)`
([shapes](orientation.md#shapes)).

**Categorical outputs.** A record patch takes `groups=(k,)` and one-hot targets (a softmax per
group). A settling brain's `step` takes the index of the wanted output neuron. A temporal patch's
outputs are continuous levels ([build from your data](build.md)).

**An absent observation reads as zero drive.** Add a presence port when an observed zero must
differ from no observation ([temporal](temporal.md#context-private-continuation-and-readback)).

## The settling brain

**Accuracy stays at chance.** Read `free_budget_exhausted` in the lesson report first
([finite free phases](learning.md#finite-free-phases-that-run-out-of-budget)).
`learner.calibrate(training_drive)` picks the sampled gain whose mean top output per row is
nearest 0.5 ([calibration](learning.md#calibrating-the-operating-point)). Keep output projections
reciprocal (`Projection(..., reciprocal=True)`, the default) so a nudge reaches the hidden
neurons; select `eta`, `eta_bias` and `beta` on development data ([knobs](learning.md#7-every-knob)).

**The phases do not converge.** Raise `free_steps` and `nudged_steps` or lower the gain. A small
movement per step can hide a large equation error under saturation: `brain.residual(drive, state)`
checks the equations at a state, `brain.equilibrate` continues to a tolerance within a budget
([concepts](concepts.md#settling-and-equilibrium)), and `cd.certificate(brain)` bounds uniqueness
and distance when its hypotheses hold ([certificate](certificate.md)).

**Qualified teaching refuses every lesson.** A `LearningPhaseError` in phase `nudged` or
`opposite` naming a budget mismatch means the nudged budget is below the free one; a composed
brain teaches with 12 nudged against 1024 free steps. Set `learning_qualified=True,
learning_nudged_steps=1024` through `retune` ([budgets](learning.md#qualified-teaching-budgets)).

**Two learners on one brain.** Build the second with `Learner.sharing(first, outputs, ...)` so
each update reaches the other, and give them disjoint plasticity masks
([two learners](learning.md#two-learners-on-one-graph), [heads](cortex.md#which-synapses-a-head-owns)).

## Memory

**A memory I switched on changes nothing.** Erase it in one of two loaded copies of the saved
brain and compare answers ([check](memory.md#check-that-a-memory-does-the-work)).

**`predict` ignores the cue shown a moment ago.** `predict`, `accuracy` and `fit` read only the
learned weights; test a memory through `act`, `step` or `live`
([memory](memory.md#memory-in-the-composed-brain)).

**Which memory carries this task?** See [which memory for which
task](memory.md#which-memory-for-which-task). No memory holds a state the observation never shows.

**What survives `reset()`?** Records, the recall address map and learned weights. The working
trace, efference copy, heard history, episodic context and recall slots are cleared
(`keep_memory=True` keeps the slots). An adult brain starts calm with no income or error
references, so at `need=0` only a surprise at its own forecasts rouses it. Where its world may
stop paying, restore its arousal state, give it a positive `need`, or keep it living without the
reset ([reset and save](continuous.md#reset-and-save)).

**Stored keys do not match a brain built with another `code`.** Keys are valid only under the
code they were stored with; `retune` refuses `code` ([the observation code](memory.md#the-observation-code)).

## The temporal patch

**`observe` returns `updated=False`.** Read `result.reason`. `contrast_asymmetric`: lower `beta`,
or shorten the path or hidden width. `no_decreasing_parameter_step`: with `backtrack=True`, no
rate lowered the loss in replay ([checking a learning step](temporal.md#checking-a-learning-step)).

**The plan reports `step_cap`.** The search stopped at its cap short of convergence; execute the
first action, measure and replan. `method="bfgs"` suits a badly conditioned action path
([planning](planning.md)).

**Zero free energy, wrong predictions.** The causal recurrence has zero defect by construction;
surprise is the observation minus the prediction ([temporal](temporal.md)).

**Learning is slow.** One detuned chain costs about `N * T * H^3`; keep hidden width and path
length modest ([cost and memory](temporal.md#cost-and-memory), [evaluation](evaluation.md)).

## The record patch

**The slow weights learn nothing, the records recall everything.** The loss is a mean over time
and outputs, so the useful `rate` grows with their product; try `rate=8` with `backtrack=True`.
A conjunction of this input and the last takes far longer, or a second patch
([two patches in depth](record-patch.md#two-patches-in-depth)).

**The records recall nothing.** Give every reading unit unit variance and count the cells in use
([diagnosing a record store](record-patch.md#diagnosing-a-record-store)). At `record_rate` near
one a cell keeps only its last writers; set `record_averaging=True` to retain a corpus
([records](records.md#records)).

**Sleep changed nothing.** Sleep restructures the store and corrects nothing; dreams from cues the
store never met teach its guesses as facts. Dream the cues of the day and score the slow weights
alone before and after ([two phases](record-patch.md#acquisition-in-two-phases-records-by-day-weights-by-night)).

**Recall by position fails.** Records are keyed by content and context; give position its own
ports ([recall is by content](record-patch.md#recall-is-by-content-not-by-position)).

**`detune` refuses categorical ports.** Quadratic detuning covers the linear readout only;
`observe` learns both ([detuning](record-patch.md#detuning-as-the-acceptance-check)).

**Which checkpoint format?** Format 2, or format 3 when the patch has `groups`,
`record_writes="batch"`, `record_width` or a structured `port`. `load` reads both.

## The belief patch

**Imagination decays while the one-step read improves.** Add the imagination loss, which
penalises drift from random starting moments ([the imagination loss](belief.md#training-the-transition-the-imagination-loss)).

**A wide picture port needs a huge rate.** Use a rate in the thousands, or the torch twin with
an ordinary optimizer ([belief](belief.md#learning)).

## Evaluation

**Which number do I report?** Held-out data split before tuning, slow weights and records
separately, a control (majority outcome, persistence, a linear predictor), and log loss with
calibration beside accuracy ([evaluation](evaluation.md)).

**How do I make a result reproducible?** Pin the release, fix every seed and bind numbers to
sources with a [receipt](receipts.md), as the `verify.py` scripts in [benchmarks](../benchmarks/) do.

## Learning from reward

**Before the first lesson.** Run `preflight(brain, outputs, plastic, drives)`; it names a remedy
for each finding ([check before the first lesson](reward.md#check-before-the-first-lesson)).

**The curve is flat, `report["saturation"]` near 1, `report["trace"]` near 0.** The readout
latched ([the traps](reward.md#traps-with-their-measurements)).

**The value runs away under `dopamine_center`.** Leave `critic_signal="auto"` (the default),
which gives the critic the raw error when the dopamine is centred.

**A decision costs ten times a settle on torch.** Clipping efficacies with `with_parameters`
after every decision re-uploads every weight; use `LearnerConfig(scale_cap=...)` instead.

**The policy always picks the same action.** Check that every action has an outcome and that
the outcomes are symmetric before touching the rule.

**The greedy choice ignores the observation.** Read `report["capped"]` (the share of outcomes
clipped at `dopamine_cap`) and `report["saturation"]`, compare `predict` with greedy `act`
([replaying a life](reward.md#replaying-a-life-through-step)), and tune through the
[rate ownership table](brain.md#defaults-and-expert-overrides).

**Every stimulus learns the same lesson.** The state code is shared. Score a `specific` row
between two stimuli where the plastic synapses leave, and select the igniting population's gain
by protocol ([connectomes](connectomes.md)).

**A readout never moves under large dopamine.** Its cells saturate; `calibrate_bias` centres
them, and `naive_efficacy` removes a naive preference.

**The plastic set is a few hundred synapses.** `seam_report` shows a seam thinned by the
synapse floor; rebuild the fixture keeping that seam.

## Connectomes

**A predicate over a large population fails.** `active` asks the mean over every member to reach
0.5; use `sparse` for a subset and `lateralized` for a left-right pair ([protocols](protocols.md)).

**A gate passes on the whole brain, fails on the sub-net.** Recruit more hops or lower the
synapse floor, and keep the closure receipt ([connectomes](connectomes.md)).

**The brain ignites or is silent at every gain.** Use a gain per cell class, selected by
protocol ([gain selection](protocols.md#gain-selection-under-a-sparsity-cap)).

## Where to ask

Open an [issue](https://github.com/muellerberndt/cadence/issues) with the smallest script that
shows the behaviour, the library version and the platform. [Contributing](../CONTRIBUTING.md)
says how to run the tests.
