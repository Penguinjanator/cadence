# Changelog

## Unreleased

- `ArousalConfig` gains `need`, the reward per moment the body requires. The share of
  the need that the recent reward leaves unmet is a want of its own, measured against
  the need rather than against the spread of outcomes: a reward that comes once in `L`
  moments has a mean of `1 / L` and a spread near `1 / sqrt(L)`, so a brain that loses
  it falls short of its long-run reward by only `1 / sqrt(L)` spreads and, with the
  founders' threshold, is not roused, while it is short of its whole need. A need never
  habituates and a life that never paid wants from its first moment. The founder is
  zero, which leaves the law and every released result unchanged; the gene is in
  `ArousalConfig.space()` with zero inside it.
- Add the key-door nursery (`benchmarks/keydoor`), the delayed key-door reward chamber of
  [issue 111](https://github.com/muellerberndt/cadence/issues/111), roadmap row 07: one
  continuing life walks a 14-cell corridor (floor, chest, lamp, 2, 5 or 10 levers varying by
  one, door) once per trip, +1 at the door with the key, a cost of 0.25 for touching anything
  without one, the key in the chest for 500 trips and then in the lamp, one trip in twenty
  cut before the door with `done` clear. The `live` arm at a declared operating point is
  measured against the always-learning loop, zero eligibility, yoked rewards, frozen, blind
  (no pouch sense), a tabular Q(lambda) with the same information and uniform random, with
  behaviour and base-policy probabilities per cell and pouch state, greedy probes on saved
  copies, the complete work ledger and source-bound receipts. The development seeds
  established three findings recorded in its README: a want measured in reward spreads is
  diluted by a sparse reward (the `need` gene above), punished exploration drives the two
  motor units to the same saturated answer for every cell (the latch of `docs/reward.md`,
  now in a continuing life), and the composed critic, whose step is divided by its trace
  energy, stays flat over a 14-cell trip so that only the actor's own eligibility carries
  the door's credit; the chamber's point raises the critic's rate to 5.0 with the
  eligibility decay and discount at 0.95, the composed values as controls. The frozen
  protocol gates delays 2 and 5 and reports delay 10. Two freezes were confirmed on fresh
  seeds; at the second, of the 20 gated lives 20 acquired rule A, 19 re-adapted after the
  key moved (fed 1.00 at the end of rule B at delay 5), 19 were frugal and 17 calm against
  the 18 the gate requires, so the gates did not pass; the first freeze's gates missed on
  frugality and calm, and its yoked control crashed on trips cut to one cell. Learning at
  every moment re-adapts at 0.97 and 0.68 while never resting; zero eligibility, yoked
  rewards and the composed critic fall short; the blind arm, without the pouch sense, is
  fed at 1.00 with no wrong interaction at both gated delays, so the working trace carries
  the key. At delay 10 half the lives acquire and 8 of 10 re-adapt; the tabular learner
  acquires it in every life. A scarcity variant (the door paying one time in two) acquires
  nothing: the need sits at the expected income and the creature never leaves arousal.

## 0.75.0 — 2026-10-06

- Add `Brain.live` and arousal for one continuing stream
  ([issue 88](https://github.com/muellerberndt/cadence/issues/88),
  [issue 122](https://github.com/muellerberndt/cadence/issues/122)). A brain
  constructed with `arousal=ArousalConfig()` answers a routine moment with the
  greedy choice of one qualified settle and changes no parameter, record or
  optimizer state, while the eligibility of earlier sampled actions fades by one
  step (`ActorCritic.fade`). An outcome that contradicts the forecast made before it
  (surprise) or a reward below what the stream usually pays (want) rouses it; an
  aroused brain samples at a temperature its want raises, keeps eligibility,
  learns from every outcome and writes memory, and the outcome that woke it is
  written to its memory. `Arousal` and `ArousalConfig` are exported; every
  constant of the law is a gene with hand-set founders and a declared space for
  `genes`. `Brain.act`, `ActorCritic.act` and `ActorCritic.probabilities` take
  an optional sampling `temperature`; `Brain.last_arousal` reports each moment;
  a brain with arousal saves it under the format name `cadence-generic/3`.
  `step`, the composed defaults, the settling and learning equations and the
  checkpoints of brains without arousal are unchanged.
- `ArousalConfig` gains `value_surprise` and `record_surprise`: a brain with an
  associative memory also forecasts the outcome of its chosen action from the record
  it holds, and the error of that record is a second surprise channel with its own
  usual size (`Arousal.usual_record`, `Brain.last_arousal["record_error"]`). The
  founders weigh it at zero, so the law is unchanged by default: on the nursery's
  development seeds the record channel woke the brain sooner and left more lives
  searching too briefly. Checkpoints of brains with arousal carry the record forecast
  of the awaited action.
- Add the odour nursery (`benchmarks/reversal`): one continuing life through
  acquisition, reversal and return with an unrelated stable skill, at pre-switch
  exposures from 100 to 10,000 trials, on a frozen protocol with ten fresh
  confirmation seeds, against always-learning, released-default, memory-only,
  graph-only, frozen, replay, reset, tabular and uniform-random arms, with
  source-bound receipts and a verifier. Its gates passed on fresh seeds at each of its
  three freezes; at the third, of the 40 gated lives 39 reversed, 39 returned and 40
  kept the stable pair, with median reversal lags of 15 to 30 trials, and one life
  never searched for the moved reward. It reproduces the historical finding that the
  always-learning brain stops sampling the choice that must change, counts the
  witnessed approaches until the greedy choice turns (one), and measures the
  routine share and the settling work of each mode. The associative memory
  carries the adaptation, the graph's reward learning alone does not acquire the
  task, a routine moment pays one full settle, and arousal on the released
  composition stays at chance: the operating point of one continuing stream it
  declares (working trace amplitude 0.3, consolidation 0.25, a tenth of the
  composed actor rate) is a development setting of that chamber; no default
  changes.
- Odour nursery, third freeze (`benchmarks/reversal`), for the readings
  [issue 88](https://github.com/muellerberndt/cadence/issues/88) still required:
  the probability of approaching each odour under the behaviour that acted and under
  the base policy, and the probability of each executed action, read from the living
  brain at every trial; a `replay` control that presents the brain's own witnessed
  records of the first rule to its memory again, one per trial, in place of the
  earlier counterfactual payoff; and the complete work of a life, with probes,
  checkpoint files, memory reads and writes, presentations, brains built, the sweeps
  of a refused attempt and the wall time of a moment per mode. Receipts of the earlier
  freezes verify by their own kind. The confirmation ran once on fresh seeds.
- Harden `live` continuation and its audit instruments: reject contradictory
  pending-action checkpoints and incomplete arousal genes; preserve routine
  feedback on a refused arousal update and identify already accepted sampled
  feedback. Keep arousal statistics atomic on numerical overflow, and support
  tiny positive averaging rates and sampling temperatures. Validate nursery
  receipt plans, sources and readings, and count discarded reset brains and
  frozen actions. Historical receipts retain their original sources and work
  limitations; the old-rule control is not equal-work witnessed replay. Issues
  88 and 122 retain their remaining acceptance requirements.
- Add the bounded steady-rhythm chamber (`benchmarks/rhythm`) for
  [issue 116](https://github.com/muellerberndt/cadence/issues/116): one
  continuing `Brain.compose` life taught to alternate two actions under constant
  drive, with frozen inputs, a frozen protocol, a declared physical event
  cadence, checkpoint-forked erased/shuffled/reset/static controls, a matched
  flip-flop control, a uniform-random baseline, pause/distractor disturbances,
  paced runs under solver-budget and host-load variation, and checkpoint
  continuation checks between actions and during a pause. Results are measured
  limits of the current System 1 on five fresh seeds; no core source, default,
  mechanism or gene changes.

## 0.74.0 — 2026-10-04

- `ActorCriticConfig.eta_bias` left unset derives `eta / 10` at construction, the
  rule of issue 126 applied to the actor ([issue 143](https://github.com/muellerberndt/cadence/issues/143)).
  An explicit value is used as given; construction warns when the bias rate exceeds a
  positive `eta`. The bare default (`eta=0.5`) resolves to the former 0.05, and the
  composed brain keeps its measured `eta_bias=0.05` at `eta=1.0`, so composed and
  default brains are unchanged; an actor with a lowered `eta` and no explicit bias
  rate learns with a bias step a tenth of its synapse step instead of 25 times it.
- Grouped motor slots on the composed brain ([issue 142](https://github.com/muellerberndt/cadence/issues/142)):
  `Brain.compose`, `Brain.build`, `Brain.genome` and `Brain(...)` take `slots`, a
  count of equal groups or one size per group covering the actions. Each slot is one
  softmax that settles with the others; `act` and `step` return one index per slot;
  lateral inhibition stays within a slot and the unset `lateral` follows the largest
  slot; episodic memory writes the chosen neuron of every slot; `Brain.load` rebuilds
  the actor on the saved grouping. One slot is the unchanged default.

- Report `capped`, the share of observed rows whose dopamine exceeded `dopamine_cap`
  before the clip, in every `learn` report and through `Brain.last_learning`
  ([issue 139](https://github.com/muellerberndt/cadence/issues/139)). Add the
  night-replay chamber (`benchmarks/replay/`): a day of decisions through `step`,
  a night in which a saved copy re-experiences that day, an equal-experience awake
  control, and frozen policy readings as the adoption gate. Document that a replay
  through `step` is more experience at the same rates, not consolidation, with
  the measured causes of a policy that ignores its observation (sign-only
  dopamine at the composed actor rate on one stream, the default working trace)
  and a setting under which the night helped as much as fresh experience, the
  equal-experience comparison of
  [issue 112](https://github.com/muellerberndt/cadence/issues/112). Defaults,
  equations and saved-state semantics are unchanged.

- Accumulate CUDA block transport directly into its destination to avoid a
  temporary product and a separate addition kernel per block. Add actual-device
  System 1 equation, gradient, refusal, memory and continuation checks, plus a
  source-bound CPU/CUDA runtime and memory comparison for issue 98.
  Preserve POSIX source keys in credit diagnostics and LF bytes in the frozen
  phrase fixture so the existing provenance checks also pass on Windows.

## 0.73.1 — 2026-10-04

- Distinguish signed activity below rest from silence (issue 106). Preserve the
  historical strict activity-fraction expected failure and add functional
  response, qualified acquisition/retention and full saved next-update guards
  for the existing zero and optional 0.5 processing-bias settings. Core equations,
  defaults and saved-state semantics are unchanged.

- Record the measured decline of a normalized composed default
  ([issue 131](https://github.com/muellerberndt/cadence/issues/131)): at the
  proposed `eta=0.003, normalize=0.99, momentum=0.9` for both composed
  learners, supervised acquisition contracts pass but the reward stream fails
  its re-adaptation contract (0.486 against 0.9 after a contingency change).
  Composed defaults remain unnormalized; normalized rates stay per-application
  settings behind the 0.72.1 construction warning. Documentation only; no
  default or equation changes.

- Harden the acquisition and retention instruments: admit source and input
  identities before execution, distinguish executed outcomes from accepted
  feedback, count memory writes only after commit, and charge final receipt
  writes against declared time and storage limits. Preserve refused work and
  complete case censuses, with independent verification and adversarial guards.
- Make retention preparation work in a standalone checkout or source
  distribution. Formal-source snapshots are explicitly requested inputs;
  omitting them does not claim a formal verification result.
- Add actual sampled-action association, partial-cue continuation and saved
  feedback guards, and bounded finite-horizon input/trace checks. Broaden the
  default test inventory to include the acquisition instrument guards.
- Add a portable, read-only acquisition/retention results demo and document the
  research wrap-up. The 360/360 partial-cue and 24/24 order-sensitive results are
  bounded development screens from pinned 0.73.0 workers; fresh retention
  confirmation, integrated replay and native transfer remain open in issues
  [85](https://github.com/muellerberndt/cadence/issues/85) and
  [110](https://github.com/muellerberndt/cadence/issues/110). Failed controls and
  source/custody limits remain visible.
- Require proposals to use local agreement repair into the same global
  equilibrium, test the simplest existing System 1 first, and demonstrate
  benefit while preserving acquired capabilities. Animal and human brains
  remain the reference, including their finite capacity and possible rigidity.
  Update contributor/agent instructions, review template and current guides.

The numerical runtime, public defaults, learning and memory equations, and
checkpoint contracts match 0.73.0. Experimental centering and a normalized
composed default are not promoted by this release.

## 0.73.0 — 2026-10-04

- Resolve the unset motor `lateral` of `Brain.compose`, `build` and `genome` by
  readout width: -0.5 up to 8 actions, 0.0 above
  ([issue 124](https://github.com/muellerberndt/cadence/issues/124)). Measured on
  composed brains, -0.5 settles a small action menu in the same few dozen sweeps
  as 0.0, while from 12 actions the undamped free solve stops settling and damped
  answers take about nine times the sweeps. An explicit `lateral` is used as
  given, and small-menu brains are unchanged.
- Calibrate to the competitive operating point by default
  ([issue 125](https://github.com/muellerberndt/cadence/issues/125)):
  `Learner.calibrate` left without `level` now places the mean top output per
  row and slot near 0.5 instead of the whole readout's mean, which on a 36-way
  readout had selected saturating gains (12, then 128). An explicit `level`
  keeps the mean target; reports carry `target`, `mean_output` and `top_output`
  per candidate. Single-output calibration selects as before.
- Derive an unset `LearnerConfig.eta_bias` as `eta / 10` at construction
  ([issue 126](https://github.com/muellerberndt/cadence/issues/126)); the
  standalone default stays 0.02 and the composed default becomes 0.05. An
  explicit value is kept, a resolved value rides through `dataclasses.replace`
  unless re-derived with `eta_bias=None`, and a bias rate above a positive
  synapse rate warns, since the bias step then dominates.
- Count and warn when a finite teaching free phase uses its entire `free_steps`
  budget under a movement tolerance
  ([issue 127](https://github.com/muellerberndt/cadence/issues/127)): the lesson
  was learned from a state that may not have settled. `Learner.step` reports
  `free_budget_exhausted` (`demonstration_free_budget_exhausted` through
  `Brain.step`); phases with `tolerance=None` remain declared fixed-length and
  silent. The finite update law is unchanged.
- Warn at `LearnerConfig` construction when `qualified=True` and `nudged_steps`
  is below `free_steps`, including through `dataclasses.replace` on a composed
  configuration: qualified phases are settle budgets, and the composed finite
  teaching default of 12 nudged sweeps refuses every realistic lesson
  ([issue 123](https://github.com/muellerberndt/cadence/issues/123)). A refused
  nudged or opposite phase under such a configuration names the budget mismatch
  in its `LearningPhaseError` message and new `hint` attribute. Document the
  budget semantics in the learning, composition and troubleshooting guides.
  Finite teaching, the composed defaults, the reward-eligibility contract and
  refusal transactions are unchanged; deliberately small qualified budgets
  remain allowed.

## 0.72.1 — 2026-10-03

- Warn at learner or actor-critic config construction when `normalize > 0` and
  either `eta` or `eta_bias` exceeds `0.05`; include the actor's independent bias
  rate and point the warning to the constructor caller. The diagnostic excludes
  the critic's separate rate and does not change optimizer equations or defaults.
- Document the RMS update, floor and momentum effects, independent bias rates
  and the limits of the reported Atari, Transcribe and Patch World pilots from
  [issue 131](https://github.com/muellerberndt/cadence/issues/131). The warning
  threshold and suggested development sweeps are not stability guarantees.

## 0.72.0 — 2026-10-03

- Add optional `resting_bias` to `Brain` and `Brain.compose`, with finite scalar
  validation, protected sensory/visual, working-memory and motor boundaries,
  and saved initialization metadata separate from learned biases. The default
  remains zero; responsiveness is not an acquisition or retention guarantee.
- Add the vanished-cue recall chamber (`benchmarks/recall/vanished_cue.py`, issue 84): a
  continuing brain's free recall of a cue across blank or distracting delays, against
  erased and shuffled trace controls and a separately trained history comparator.
  Freeze paired episodes, fork complete probe checkpoints, charge attempted
  teaching and action solves, and retain refusal and source records.
- Add the `lateral0-local-rms` and `lateral0-resting` candidate genes to the
  acquisition microscope, with explicit effective settings and unchanged
  first-refusal stopping rules.
- Correct acquisition evidence descriptions for effective rates, nudged budgets
  and accepted versus attempted updates. Document supervised-only continuing
  interaction separately from zero-reward transitions. Broader acquisition and
  recall acceptance remains open; no memory or learning default is changed.

- Expose immutable `Brain.last_settlement` diagnostics for successful and refused
  action/prediction solves: per-row residuals, qualification, sweeps, checks and
  damping. Keep diagnostic state outside checkpoints and preserve action and
  feedback transactions. The report explicitly excludes learning and memory work.
- Add a guided documentation entry and link previously orphaned guides. Clarify
  reciprocal composition, the equilibrium world-model hypothesis, separate
  transition predictors, concrete memory/readback mechanisms, configuration
  defaults and the scope of centered dopamine and `Life`.
- Give contributors and agents a task-to-guide map, continuing-brain construction
  recipe and explicit sleep/dream preservation guidance. Include `AGENTS.md` in
  source distributions and validate its documentation links.
- Extend the continuing example with witnessed corrective teaching, repair and
  continued use, reporting actual outcomes alongside free-answer and other
  settling work. Preserve saved pending-feedback continuation.

- Center introductory and contributor guidance on one continuing equilibrium
  brain across bootstrap, use, witnessed disruption and local correction. Add
  an executable world-model guide with explicit current integration boundaries;
  keep independent learning controls and distinct model families labeled.

## 0.71.1 — 2026-10-03

- Reject nonfinite or inconsistent states during finite bias calibration, even
  without a report. Check the final candidate before returning biases; an invalid
  solve raises `RuntimeError` without using its output to advance the search or
  changing the source graph. Finite searches still permit unsettled states.

## 0.71.0 — 2026-10-03

- Add opt-in qualified graph learning through `LearnerConfig.qualified`. The
  free, positive and required negative teaching phases must satisfy the original
  full equations before one local update. `LearningPhaseError` retains phase
  states and attempted work; refusal preserves parameters and optimizer history.
- Extend `NeuralGraph.equilibrate` with bounded numerical step halvings. Repeated
  complete-state checkpoints can move an unqualified attempt to a smaller step
  sooner, while every accepted state still meets the original residual and all
  attempts share the declared sweep budget.
- Report all teaching phases, original residuals, residual transports,
  stagnation comparisons and attempted/accepted row presentations. Preserve
  accepted and refused demonstration costs in `Brain.last_learning`.
- Make gain calibration honor qualified learning and reject invalid states.
  Its default grid spans the current gain's representable powers-of-two multiples
  from 1/256 to 256, trying the current gain first. Explicit grids retain their
  supplied order. `Learner.last_calibration` records every attempted candidate;
  an all-refused search preserves the graph and optimizer.
- Add opt-in qualification and reports to `calibrate_bias`, checking every
  midpoint and the final candidate before returning biases. Report observed means,
  target gaps and all solve work. Finite calibration remains available; a
  qualified operating point does not guarantee a requested target or acquisition.
- Expose motor competition through `Brain.compose(lateral=...)`, retaining the
  default per-pair weight of -0.5. Zero removes those lateral connections while
  preserving reciprocal processing/motor feedback.
- Separate reward eligibility duration with `ActorCriticConfig.eligibility_steps`.
  The Brain default stays at 12 finite nudged steps independently of supervised
  teaching budgets; standalone `None` retains learner-budget inheritance.
- Restore associative memories, separator state and terminal working traces when
  reward bootstrap qualification fails, preserving the actual outcome for retry.
  Accepted real feedback remains learned if a subsequent lesson or action refuses.
- Score `Brain.fit` epochs through qualified, memory-free public predictions.
  A refused score preserves lessons already accepted in that epoch.
- Add a runnable `Brain.compose` example and NumPy-only CI coverage for actual
  feedback, demonstrations, free recall and saved pending-feedback continuation.
- Add repository acquisition and retention instruments with source-frozen
  protocols, independent residual/contrast checks, charged rehearsal, preserved
  refusal/case censuses and saved continuation. Separate graph acquisition,
  working traces and consolidated associative storage.
- Clarify conditional gradient assumptions, independent learning-rate
  hyperparameters, standalone versus composed defaults, calibration limits and
  device execution. The adaptive `ActorCritic` optimizer uses host arrays;
  supported blocked PyTorch `Learner` updates remain on the device. Document
  the independent `eta_bias=0.02` and `temperature=0.2` defaults accurately.
- Add numerical and behavioral regressions for phase refusal, bounded damping,
  calibration admission, motor wiring, memory rollback and saved continuation
  across supported backends. Gradient checks retain their symmetry, smooth-branch,
  nudge-limit and loss-scaling hypotheses.
- Distinguish current application demos from archived research examples. Restore
  the recorded 0.61.0 release and 0.62.0 development history without inventing
  releases for unpublished version numbers.
- Update consumer guidance for explicit qualified protocols, calibration and
  refused-lesson retry. Correct Atari settings and pooled processing-time labels;
  retain historical finite-probe measurements and their source identity.

- Document focused local contract checks, separate foundation/population suites
  and a NumPy-only environment for shorter iteration. Keep full CI coverage,
  fixtures and assertions; optional backend skips remain explicit.

Finite supervised teaching remains the default. System 1 memory, plasticity,
private imagination and action remain available; optional System 2 continues
to join the same neural graph. Numerical qualification alone does not establish
general acquisition, lifelong retention or an efficiency advantage.

## 0.70.0 — 2026-10-02

- Make System 1 the default continuing brain, with working trace, fast and
  persistent associative memory, plasticity, action and private imagination.
- Provide `Brain.compose` for reciprocal base modules and optional
  System 2 observer regions within the same neural graph.
- Add `Brain.imagine` for private responses to supplied hypothetical
  observations. Learned environmental consequences and action planning use
  the temporal-model API.
- Qualify actions and independent predictions against the full state equations.
  Refusal preserves action state and pending feedback; real outcomes learned
  before a subsequent refusal remain learned.
- Use bounded numerical damping when a free solve needs it, then check the
  original model's residual. The total budget and finite teaching rule stay fixed.
- Provide event records and consolidation, learned temporal paths, continuous
  action planning and finite response protection through advanced APIs.
- Keep exact state-and-error feedback available in the advanced population solver.
- Simplify guides around current usage. This is experimental software; backward
  compatibility is not a design requirement. NumPy is required, with optional
  acceleration backends.

## 0.62.0 development revision — 2026-10-02

This entry records the development sources at
[`1f9daac`](https://github.com/muellerberndt/cadence/commit/1f9daac).
The recovered work became release 0.70.0; 0.62.0 was not published as a
GitHub release or on PyPI. Names below describe that development revision.

- Restore the capable pre-reset foundation from 930ee807: continuing
  `GenericBrain` interaction, `Trace`/`Afterglow`, consolidating
  `SynapticMemory`, record patches and sleep, temporal learning, private
  imagination, action planning and response protection. Preserve subsequent
  numerical, continuation and recursive-wiring hardening.
- Add `GenericBrain.compose` as a direct modular entry with working trace and
  consolidating memory, optional reciprocal observer regions, and the existing
  continuing interaction interface. Add private `GenericBrain.imagine` over
  supplied hypothetical observations; environment prediction remains the
  separate learned temporal-model contract.
- Qualify `GenericBrain.act`, `predict` and `accuracy` against the full state
  equations. Exhausted action repair preserves live state and pending feedback;
  consumed real outcomes stay learned if a following action refuses. Keep finite
  eligibility/training phases distinct from this free-answer qualification.
- Keep cortical observation optional. The foundation can already be deep and
  modular; observer feedback extends the shared graph rather than replacing
  working memory and learning with a narrower model.
- Preserve the newer state-and-error solver under
  `cadence.experimental.equilibrium`, with its own guides, examples and tests.
  Its sparse patch-connectivity checks and same-call stationary-evaluation
  optimization remain available there, without changing the restored APIs.
- Rewrite the entry guides around the biological-brain objective, working
  mechanisms and actual application source identities. The default package
  requires NumPy. Keep current GPL-3.0 licensing and historical attribution.
- Preserve original Amen, Connect Four and Atari checkpoints and browser
  engines. Library recovery, checkpoint parity and native application behavior
  require separate verification; no old receipt is silently promoted.
- Recover the capable foundation before releasing the narrower candidate as the
  default. Its separate numerical, CI and package evidence stays source-bound;
  the restored package requires its own verification.

## 0.61.0 — 2026-10-02

- Restore the principle as an enforced default: patches repair local
  disagreement to reach a coherent brain state, and further repair is driven by
  that state's mismatch with reality. `Cortex.build()` now refuses a layout in
  which a population settles with no other population, and a layout in which a
  group of populations settles apart from the rest. Every population must read
  another population's states or errors, or be read by one, and those reads
  must join all populations into one connected system; an unread sensors-only
  population or a disconnected group raises `ValueError` naming it. The
  smallest brain is two populations.
- Remove the input-only "flat" layout from the README, quickstart, layout and
  design guides, agent guides, examples and test fixtures. The layout example
  defaults to a two-population brain (`small`), with `deep` and the explicit
  `recursive` experiment. The query-cost and temporal-credit examples no longer
  build an input-only arm; their recorded receipts stay as recorded.
- Lead the README and the contributor guide with the main hypothesis and the
  simplicity premise, and contrast settlement with feed-forward backpropagation.
- The repair law, energy, qualification tolerance and admission contract are
  unchanged. `cortex.py` changed, so snapshots bind to this release's sources;
  snapshots saved by `0.60.0` load only in `0.60.0`.
- Error-reading observers remain experimental; this release still claims no
  automatic System 2, retained useful recursive correction or reproduced
  musical quality.

Earlier entries remain in the [source changelog before the guide simplification](https://github.com/muellerberndt/cadence/blob/1f9daac/CHANGELOG.md).
Version numbers 0.63.0–0.69.0 were not published; they are not missing release entries.
