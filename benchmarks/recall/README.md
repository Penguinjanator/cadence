# Vanished-cue recall chamber

This bounded instrument asks whether a continuing `Brain.compose` can answer from
an earlier cue when all current probe inputs are identical. It tests one declared
neural-graph/trace recipe; it does not establish general recall, choose defaults,
or complete [the recall contract](https://github.com/muellerberndt/cadence/issues/84).
Read the [world-model guide](../../docs/world-model.md) and
[numerical contracts](../../docs/contracts.md) before interpreting results.

## What runs

One brain carries its activity and working trace through cue, delay and probe
observations, then straight into the next episode. Each group of rows receives
every cue once, with independently permuted cue assignments per episode and
identical delay/distractor inputs within the group. Row identity and distractor
identity therefore do not supply the answer. The event-time unit is one accepted
observation; these runs do not model variable physical time.

Only the current probe has a teacher. The chamber calls
`Learner.step(brain.stimulus(probe), cue)` with the live trace, followed by a free
`act(..., greedy=True)`. Earlier observations use greedy acts only. No reward is
invented, no actor eligibility is created, and no action awaits reward feedback.
A qualified refused lesson is charged and skipped; the following free action has
its own qualification. A refused action stops that model's continuing life.
There is no automatic reset, repeated reward or silent cold restart.

The selected recipe uses `modules=(32,)`, `lateral=0`, no episodic memory, the
specified working-trace amplitude/decay, and the learner configuration in
`make_brain`. It is **not the complete compose-default configuration**. Qualified
teaching uses 128 nudged sweeps per phase; `--finite` selects the separate finite
12-sweep teaching contract. Labels at the current probe supply no credit through
past observations. Evaluation has no teachers or parameter updates.

## Controls and continuation

Before any model runs, the instrument writes the exact training/test observations,
labels and shuffle permutations. Those arrays are reused across decays and both
models. Test streams use a separate random generator; running or reordering a
control cannot change later inputs.

At each evaluation probe, the continuing vanished-cue brain saves its **complete**
state. Four private branches load that same checkpoint:

- **intact:** reads its retained context normally;
- **erased:** zeros the working trace/last activity and marks rows cold before
  the probe, retaining the neural warm state;
- **shuffled:** transplants all three trace arrays from a row with another cue;
- **reset:** clears both transient neural state and trace, retaining learned
  parameters.

The intact original life alone continues. A separate restored branch must give
an identical answer and byte-equal checkpoint arrays after the probe, including
optimizer and random state. This checks the saved state immediately before the
probe; it is not a general mid-cue/replacement continuation guarantee.

The **appended-history comparator** is a second, independently trained brain
with the same initial parameters, episodes and lesson budget. During both its
training and evaluation, an external history buffer supplies the original cue
in extra probe coordinates. It is a useful acquisition control only to the
extent its own measured free recall succeeds. It is **not a mathematical upper
bound**, and the extra input is never given to the vanished-cue brain. Appending
untrained coordinates to the vanished-cue model would not be a competent control.

## Run and verify

Install the intended checkout first (`python -m pip install -e .` from the repo).
Use an unused output directory; existing attempts are never overwritten. This
small command checks the protocol, not a useful retention horizon:

```sh
python benchmarks/recall/vanished_cue.py --out /tmp/recall-smoke --cues 2 --streams 4 --modules 4 --lessons 2 --test-episodes 2 --seeds 0 --decays 0.8 --delays 1
python benchmarks/recall/vanished_cue.py --verify /tmp/recall-smoke
```

`protocol.json` and compressed input arrays precede all training. Initial,
trained, pre-probe and continued checkpoints retain model custody.
`rows.jsonl` preserves completed conditions if a run is interrupted. A completed
`summary.json` is a `cadence.Receipt`; its `body` holds raw trial labels/answers,
scores, all teacher/free-solve reports, configuration and artifact hashes. Its
source manifest binds copied benchmark and imported library Python sources.
The verifier checks the receipt digest, source bytes and retained artifact hashes;
this is custody verification, not an independent reproduction of learned behavior.
A partial attempt has no completed summary and makes no recall claim.

Work includes accepted and refused teacher phases (free, positive and negative
nudges), every free answer, control fork and continuation check. Sweeps,
row-sweeps, residual checks and trace-update counts are separate. Per-call wall
time excludes checkpoint I/O; run wall time includes it. These measurements are
not physical energy or a cross-machine efficiency comparison. Scores report
attempted, refused and unrun rows; refusals count as wrong among attempted probes.
Stopping before a probe does not fabricate a scored answer.

## What the existing exploration establishes

The [earlier issue comment](https://github.com/muellerberndt/cadence/issues/84#issuecomment-5967884580)
reports an exploratory appended-cue acquisition difference under selected trace
amplitudes, decays and reset interventions. Its numerical table lacks a frozen
reproduction driver and source-bound checkpoint receipt here. Old recall
summaries also exist, but lack source binding and used an earlier control and
accounting protocol. Neither provides a general finding that the default trace
prevents learning; they motivate a controlled comparison.

This corrected instrument has not established a new recall horizon. Freeze
behavioral acceptance before a campaign and retain its failures. Replacement,
partial/noisy cues, finite-capacity competition, irregular event timing,
pre-replacement continuation and an independently checked retention/separation
bound remain outside this initial chamber. The original source-specific records
remain historical evidence; rerunning a corrected protocol produces a new result.

## Finite continuing input fixtures

[The reviewed finite-horizon protocol](FINITE_HORIZON_PROTOCOL.md) is a paused
scientific design. Its portable [pure input fixtures](finite_horizon_inputs.py)
cover delay, distractors, replacement, partial/noisy cues, finite load, ordered
histories and irregular event timing. Paired histories have opposite expected
responses with identical current queries; the explicit history control reads
only witnessed payloads. A separate literal recurrence checks trace updates,
including refusal of skipped or extra events.

These fixtures and their deterministic tests construct no brain and perform no
learning or solve. They provide protocol and preservation checks, not a measured
recall horizon. A finite-horizon worker/launcher is not shipped, and #84 remains
open. The existing `vanished_cue.py` instrument above retains its separate scope.

## The finite continuing recall chamber, 2026-10-08

[`finite_horizon.py`](finite_horizon.py) is the worker for the
[reviewed finite-horizon protocol](FINITE_HORIZON_PROTOCOL.md), with its
[frozen inputs](finite_horizon_inputs.py) and [`protocol-finite.json`](protocol-finite.json)
declaring every setting, the founders, the caps and the gates. One continuing
`Brain.compose` life per arm and founder: `vanished`, the declared recipe (working
trace decay 0.8, amplitude 1, learned at the QUERY lessons only, never a history
coordinate); `default`, the same brain with the composed working-trace defaults
(amplitude 3, decay 0.2), the simpler setting kept as the control of that gene;
`history`, the external-history comparator with byte-equal initial arrays, the same
lessons and the actual observed payloads of the last four WRITE events appended at
QUERY; `random`, the frozen uniform-random actions. 192 training episodes, then 24
evaluation episodes of each of the 13 conditions, every QUERY forked into intact,
erased trace, shuffled trace (transplanted from the paired row with the opposite
value) and full reset; the first episode of every condition saved after the first
cue, before a replacement and before QUERY with a clone resuming each seam; private
imagination and a refused act at an impossible tolerance leaving the pre-query
checkpoint unchanged; clean-2 replayed under two timestamp schedules; the trace
audited at every act against its literal recurrence; a 900-second wall and a 160 MiB
output cap per founder. The horizon is the largest contiguous passed prefix over 0,
1 and 2 intervening events under the prewritten gates; closure needs horizon 1 and
the nuisance, replacement, order, continuation, purity and resource gates.

```sh
python benchmarks/recall/finite_horizon.py --out /tmp/recall-finite
python benchmarks/recall/finite_horizon.py --verify /tmp/recall-finite
python -m pytest -q benchmarks/recall/test_finite_horizon.py
```

### recall/1 on fresh founders 301, 302 and 303: closure failed

Receipt `results/finite-1-2026-10-08.json.gz`, verified: every founder complete within
its caps (135 to 185 seconds), no refused act, two refused lessons in one history
arm, the trace audit at 1.1e-16 on 1,750 audited acts per brain arm, every seam,
imagination, refusal and timing check equal. Intact accuracy over planned rows,
with the paired-both-correct share, the erased fork, the shuffled fork against the
transplanted value, the default-trace control, the history control and the frozen
uniform-random policy:

| founder | condition | intact | paired | erased | shuffled→transplanted | default | history | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 301 | clean-0 | 0.52 | 0.04 | 0.50 | 0.52 | 0.57 | 1.00 | 0.46 |
| 301 | clean-1 | **1.00** | 1.00 | 0.50 | 1.00 | 0.51 | 1.00 | 0.43 |
| 301 | clean-2 | **1.00** | 1.00 | 0.50 | 1.00 | 0.52 | 1.00 | 0.55 |
| 301 | distractor-1 / -2 | 0.50 / 0.50 | 0.00 | 0.50 | 0.50 | 0.56 / 0.50 | 1.00 | 0.51 / 0.47 |
| 301 | noise-1 | **1.00** | 1.00 | 0.50 | 1.00 | 0.47 | 1.00 | 0.47 |
| 301 | replacement-1 | 0.87 | 0.74 | 0.50 | 0.87 | 0.54 | 1.00 | 0.52 |
| 301 | partial-1 / order-latest-1 | 0.66 / 0.50 | 0.31 / 0.00 | 0.50 | 0.66 / 0.50 | 0.50 / 0.53 | 0.86 / 1.00 | 0.48 / 0.44 |
| 302 | every condition | 0.50 | 0.00 | 0.50 | 0.50 | 0.43 to 0.55 | 1.00 | 0.43 to 0.56 |
| 303 | clean-0 | **0.99** | 0.99 | 0.50 | 0.99 | 0.50 | 1.00 | 0.43 |
| 303 | clean-1 | 0.85 | 0.71 | 0.50 | 0.85 | 0.50 | 1.00 | 0.51 |
| 303 | clean-2 | 0.62 | 0.24 | 0.50 | 0.62 | 0.50 | 1.00 | 0.54 |
| 303 | replacement-1 | 0.92 | 0.83 | 0.50 | 0.92 | 0.50 | 1.00 | 0.49 |
| 303 | distractor-1 / noise-1 / partial-1 / order-latest-1 | 0.69 / 0.66 / 0.50 / 0.50 | 0.38 / 0.31 / 0 / 0 | 0.50 | 0.69 / 0.66 / 0.50 / 0.50 | 0.50 | 1.00 / 1.00 / 0.96 / 1.00 | 0.48 to 0.56 |

Horizons: 301 none (clean-0 fails while delays 1 and 2 pass, so the contiguous-prefix
rule credits nothing), 302 none, 303 zero. Nuisance gates fail on every founder.
Capacity 2/4 and delays 4/8 are at 0.46 to 0.75 and never pass. Closure fails as
declared and #84 stays open.

What the receipt does establish. Where the declared recipe recalls, it recalls through
the trace exactly as the protocol demanded: founder 301 answers delays 1 and 2 and
the noisy cue at 1.00 while its erased and reset forks sit at 0.50, and every shuffled
fork answers the transplanted history's value at the same accuracy and the original's
at its complement, so the transplanted trace alone determines the answer. The external
history control reads 1.00 on nearly every condition, so the lessons teach the mapping
when the cue is present. The composed default trace (amplitude 3, decay 0.2) recalls
nothing on any founder or condition, the finding of the 2026-10-03 exploration now on
a frozen protocol with source-bound receipts. The declared recipe is founder-bound:
one founder learns nothing at all with the same lessons that teach its history twin,
one recalls at delays 1 and 2 but not at 0, one at 0 and partly at 1. The distractor
and order conditions fail everywhere. The next freeze selects the trace amplitude and
the lesson budget on development founders and runs new fresh founders; the current
`--repeats` override marks such development runs as not frozen.

### Development founders 0 and 1: the amplitude, the decay and the lesson budget

Receipts `results/development-finite-*-2026-10-08.json.gz`, every one verified, 32 training
repeats (384 lessons) and 24 evaluation repeats unless marked. Horizon per founder, and the
intact accuracy of the declared recipe on the clean delays, the one-event distractor, the
replacement and the latest-of-two order condition (founder 0 / founder 1):

| amplitude | decay | repeats | horizon | clean-0 | clean-1 | clean-2 | distractor-1 | replacement-1 | order-latest-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.5 | 32 | none / none | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| 0.1 | 0.8 | 32 | 0 / none | 1.00 / 0.50 | 0.76 / 0.50 | 0.74 / 0.50 | 1.00 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| 0.3 | 0.5 | 32 | 0 / none | 1.00 / 0.50 | 0.50 / 0.77 | 0.50 / 0.62 | 0.93 / 0.71 | 0.50 / 0.53 | 0.50 / 0.50 |
| **0.3** | **0.8** | **32** | **1 / 0** | 1.00 / 0.99 | 0.98 / 0.80 | 0.94 / 0.88 | 0.92 / 0.94 | 0.50 / 0.50 | 0.50 / 0.50 |
| 0.3 | 0.8 | 64 | none / none | 0.58 / 0.50 | 0.76 / 0.50 | 0.80 / 0.50 | 0.50 / 0.50 | 0.72 / 0.50 | 0.70 / 0.50 |
| 0.3 | 0.9 | 32 | none / none | 0.52 / 0.50 | 0.67 / 0.50 | 0.55 / 0.50 | 0.70 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| 0.5 | 0.8 | 32 | none / none | 0.50 / 0.81 | 0.50 / 0.90 | 0.50 / 1.00 | 0.50 / 0.79 | 0.50 / 0.50 | 0.50 / 0.50 |
| 1.0 (recall/1) | 0.8 | 32 | none / none | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| 3.0 (composed default) | 0.8 | 32 | none / none | 0.50 / 0.49 | 0.50 / 0.51 | 0.50 / 0.49 | 0.50 / 0.50 | 0.50 / 0.66 | 0.50 / 0.52 |
| 0.3 | 0.8 | 32, surprise rule | none / 2 | 0.77 / 0.99 | 0.76 / 1.00 | 0.61 / 1.00 | 0.74 / 1.00 | 0.50 / 0.50 | 0.50 / 0.50 |
| 1.0 | 0.8 | 32, surprise rule | none / none | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| 0.3 | 0.8 | 64, surprise rule | 0 / none | 1.00 / 0.50 | 0.79 / 0.50 | 0.79 / 0.50 | 0.92 / 0.50 | 0.97 / 0.50 | 0.67 / 0.50 |

Amplitude is the knob: at 1.0 and 3.0 both founders recall nothing at any delay, at 0.3 with
decay 0.8 one founder has horizon 1 and the other horizon 0 (clean-1 at 0.80 with a paired
share of 0.59). Doubling the lesson budget to 64 repeats takes the recall away again on
founder 0 (clean-0 from 1.00 to 0.58, while clean-8 rises to 0.94) and founder 1 learns
nothing: more lessons on a mapping already learned move it. Replacement and the order of two
cues are at chance at every point but one; the external history control reads 1.00 on them
throughout. The selected point, amplitude 0.3, decay 0.8, 32 repeats, was declared in
[`protocol-finite-2.json`](protocol-finite-2.json) with recall/1's gates, caps, conditions
and controls unchanged before founders 304 to 306 ran.

### recall/2 on fresh founders 304, 305 and 306: closure failed

Protocol SHA-256 `a91ddb6755ed476b1408ecc1c07c362fb9b8d8969ac5df3d665bad4d885237bb`; receipt
`results/finite-2-2026-10-08.json.gz`, verified: every founder complete within its caps
(185 seconds for the three), no refused act or lesson, the trace audit at 1.1e-16 on 1,750
audited acts per brain arm, every seam, imagination, refusal and timing check equal.

| founder | condition | intact | paired | erased | shuffled→transplanted | default | history | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 304 | every condition | 0.50 | 0.00 | 0.50 | 0.50 | 0.50 | 0.78 to 1.00 | 0.45 to 0.57 |
| 305 | every condition | 0.36 to 0.50 | 0.00 | 0.50 | 0.36 to 0.50 | 0.50 | 0.50 to 1.00 | 0.44 to 0.55 |
| 306 | clean-0 / clean-1 / clean-2 | **1.00 / 1.00 / 1.00** | 1.00 | 0.50 | 1.00 | 0.50 | 1.00 | 0.48 to 0.53 |
| 306 | distractor-1 / distractor-2 | 0.72 / 0.50 | 0.45 / 0.00 | 0.50 | 0.72 / 0.50 | 0.50 | 1.00 | 0.45 / 0.54 |
| 306 | replacement-1 / order-latest-1 | 0.79 / 0.56 | 0.58 / 0.11 | 0.50 | 0.79 / 0.56 | 0.50 | 1.00 | 0.43 / 0.52 |
| 306 | partial-1 / noise-1 / clean-4 / clean-8 | 0.50 / 0.65 / 0.72 / 0.50 | 0.00 / 0.29 / 0.45 / 0.00 | 0.50 | same as intact | 0.50 | 1.00 | 0.46 to 0.55 |

Horizons: 304 none, 305 none, 306 zero (delays 1 and 2 clean at 1.00 through the trace, with
the erased and reset forks at 0.50, but the one-event distractor at 0.72 fails the prefix).
Closure fails as declared and #84 stays open. The selected recipe is founder-bound exactly as
recall/1's was: two fresh founders learn nothing from the same 384 lessons that teach their
history twins to 1.00, one recalls a vanished cue across two fillers and loses it to a
distractor. The composed default trace again recalls nothing on any founder.

### recall/3 on fresh founders 307, 308 and 309 under the surprise rule: closure failed

The loop chamber of issue #140 (`benchmarks/rhythm/loop_rhythm.py`) found that a lesson on
every row makes a learned pattern come and go from pass to pass, and the 64-repeat row above
is the same finding here. The third freeze therefore declares the routine rule of
`Brain.live` as the worker's `surprise` teaching rule: at every QUERY the free greedy act
first, then one lesson on the rows it answered wrong, on the drive that act read, as the
steady-rhythm chamber's `mismatch` arm teaches; a right answer teaches nothing. On the
development founders the rule gave horizons 2 and none against 1 and 0 under the every rule
at the same trace and budget, nothing at amplitude 1.0 under either rule, and the recall lost
again at 64 repeats under either rule. [`protocol-finite-3.json`](protocol-finite-3.json),
SHA-256 `957a66fbf60f077c1ffa40d4bbc91bed05e9ff270a9a8a16d108df6061e172a1`, keeps recall/2's
trace, budget, gates, caps, conditions and controls; receipt
`results/finite-3-2026-10-08.json.gz`, verified: every founder complete within its caps (186
seconds for the three), no refused act or lesson, the trace audit at 2.8e-17 on 1,750 audited
acts per brain arm, every seam, imagination, refusal and timing check equal. The rule gave 167
to 184 lessons on 557 to 722 rows per founder, against 384 lessons on 3,072 rows under the
every rule; the history twins needed 29 to 35.

| founder | condition | intact | paired | erased | shuffled→transplanted | default | history | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 307 | clean-0 / clean-1 / clean-2 / clean-4 | 0.50 / 0.50 / 0.50 / 0.56 | 0.00 to 0.12 | 0.50 | same as intact | 0.44 to 0.51 | 1.00 | 0.50 to 0.55 |
| 307 | clean-8 / capacity-2 / capacity-4 | 0.93 / 0.74 / 0.62 | 0.85 / 0.49 / 0.25 | 0.50 | same as intact | 0.46 to 0.52 | 1.00 | 0.52 to 0.56 |
| 307 | every other condition | 0.50 to 0.52 | 0.00 to 0.03 | 0.50 | same as intact | 0.45 to 0.52 | 1.00 | 0.41 to 0.55 |
| 308 | clean-0 / clean-1 / clean-2 / clean-4 / clean-8 | 0.50 / 0.52 / 0.66 / 0.85 / 0.62 | 0.00 / 0.03 / 0.31 / 0.70 / 0.25 | 0.50 | same as intact | 0.50 | 1.00 | 0.47 to 0.56 |
| 308 | partial-1 / noise-1 | 0.59 / 0.68 | 0.19 / 0.35 | 0.50 | same as intact | 0.50 | 1.00 | 0.46 / 0.54 |
| 308 | every other condition | 0.50 to 0.62 | 0.00 to 0.24 | 0.50 | same as intact | 0.50 | 1.00 | 0.47 to 0.57 |
| 309 | clean-0 / clean-1 | **0.99 / 0.89** | 0.99 / 0.78 | 0.50 | 0.99 / 0.89 | 0.50 / 0.49 | 1.00 | 0.49 / 0.47 |
| 309 | clean-2 / clean-4 / clean-8 | 0.86 / 0.85 / 0.50 | 0.73 / 0.71 / 0.00 | 0.50 | same as intact | 0.59 / 0.67 / 0.49 | 1.00 | 0.45 to 0.52 |
| 309 | distractor-1 / distractor-2 / noise-1 / partial-1 | **0.99** / 0.84 / **1.00** / 0.83 | 0.98 / 0.68 / 1.00 / 0.73 | 0.50 | same as intact | 0.46 to 0.54 | 1.00 | 0.44 to 0.55 |
| 309 | replacement-1 / order-latest-1 | 0.50 / 0.79 | 0.00 / 0.58 | 0.50 | same as intact | 0.57 / 0.53 | 1.00 | 0.53 / 0.46 |

Horizons: 307 none (its only passing condition is the eight-event delay, with no prefix
under it), 308 none, 309 one (delay 2 reads 0.86 with a paired share of 0.73, under the
gate's 0.75). Closure fails as declared and #84 stays open. Where founder 309 recalls, it
recalls through the trace: its erased and reset forks sit at 0.50 and every shuffled fork
answers the transplanted history's value; the default trace recalls nothing on any founder.

What the three freezes establish for roadmap row 02. On the founders that learn it, the
working trace at decay 0.8 and amplitude 0.3 is a recall of one vanished cue across one or
two fillers, through the trace and nothing else, with the paired share dropping below the
gate at delay 2 and the eight-event delay out of reach; the composed amplitude of 3 recalls
nothing at any delay on any founder. Whether a founder learns it at all is a property of the
founder, under either teaching rule: 3 of 6 fresh founders (303, 306, 309) and 2 of 2
development founders across the two rules. The replacement of a cue by a later one and the
latest of two cues are at chance on every founder but one, under every setting tried, while
the history control reads them at 1.00: a decaying superposition of settled states does not
let the readout prefer the later of two cues 0.8 apart in weight, and the lessons those rows
keep producing are the ones that move a mapping already learned (the 64-repeat rows). A
closure of #84 needs either an acquisition that does not depend on the founder or a memory
that writes the later cue over the earlier one; this instrument can now tell which.
