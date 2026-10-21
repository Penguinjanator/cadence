# Troubleshooting

Short answers to the problems builders meet first, each linking the page that owns the topic.
[Common missteps](evaluation.md#common-missteps) lists the ways a good-looking number can be wrong.

## Install and run

**`pip install cadence-net==0.84.3`, then `import cadence`.** Python 3.11 or newer and NumPy are
the only requirements.

**Do I need a GPU?** No; everything runs on NumPy float64. `[fast]` adds Numba and SciPy,
`[accel]` torch and `[apple]` MLX for the settling brain ([backends](backends.md)).

**Which interface do I want?** `Brain.compose(..., arousal=True)` and `brain.live(...)` for one
continuing stream; `step` for explicit teaching, batches or learning from every outcome. Other
contracts are in the [specialist guides](quickstart.md#specialist-guides).

## Shapes and encodings

**`ValueError` on shapes.** A settling brain takes a drive `(batch, n)`, one column per neuron,
and `Learner.step` integer labels `(batch,)`; a composed `Brain` takes observations
`(batch, inputs)` ([shapes](orientation.md#shapes)).

**Categorical outputs.** A settling brain's `step` takes the index of the wanted output
neuron ([build from your data](build.md)).

**An absent observation reads as zero drive.** Add a presence input when an observed zero must
differ from no observation.

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
(`reset(keep=("slots",))` keeps the slots). An adult brain starts calm with no income or error
references and no reward scale, so at `need=0` only a surprise at its own forecasts rouses it
and a positive `need_scale` wants nothing until outcomes enter the new scale;
`reset(keep=("arousal",))` keeps its arousal state as it is. Where its world may stop paying,
reset it keeping the arousal, give it a positive `need`, or keep it living without the reset
([reset and save](continuous.md#reset-and-save)).

**Stored keys do not match a brain built with another `code`.** Keys are valid only under the
code they were stored with; `retune` refuses `code` ([the observation code](memory.md#the-observation-code)).

## Evaluation

**Which number do I report?** Held-out data split before tuning, a control (majority outcome,
persistence, a linear predictor), and log loss with calibration beside accuracy
([evaluation](evaluation.md)).

**How do I make a result reproducible?** Pin the release, fix every seed and bind numbers to
sources with a [receipt](receipts.md), as the `verify.py` scripts in [benchmarks](../benchmarks/) do.

## Learning from reward

**Before the first lesson.** Run `preflight(brain, outputs, plastic, drives)`; it names a remedy
for each finding ([check before the first lesson](reward.md#check-before-the-first-lesson)).

**The curve is flat, `report["saturation"]` near 1, `report["trace"]` near 0.** The readout
latched ([the traps](reward.md#traps-with-their-measurements)).

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
0.5; use `sparse` for a subset and `lateralized` for a left-right pair ([protocols](../benchmarks/instruments/README.md)).

**A gate passes on the whole brain, fails on the sub-net.** Recruit more hops or lower the
synapse floor, and keep the closure receipt ([connectomes](connectomes.md)).

**The brain ignites or is silent at every gain.** Use a gain per cell class, selected by
protocol ([gain selection](../benchmarks/instruments/README.md#gain-selection-under-a-sparsity-cap)).

## Where to ask

Open an [issue](https://github.com/muellerberndt/cadence/issues) with the smallest script that
shows the behaviour, the library version and the platform. [Contributing](../CONTRIBUTING.md)
says how to run the tests.
