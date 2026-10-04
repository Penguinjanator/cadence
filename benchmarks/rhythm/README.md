# Steady-rhythm chamber

This bounded instrument asks whether one continuing `Brain.compose` life can alternate
two learned actions, A, B, A, B, while every observation it receives is identical.
It is the continuing-action chamber of
[issue 116](https://github.com/muellerberndt/cadence/issues/116), roadmap row 06 of
[issue 109](https://github.com/muellerberndt/cadence/issues/109). It measures one
declared System 1 recipe and its compose-default control against lesioned forks of
the same checkpoint, a matched flip-flop control and a uniform-random baseline. It
establishes no default, no mechanism and no general timing capability. Read
[the protocol](RHYTHM_PROTOCOL.md), the [world-model guide](../../docs/world-model.md)
and [numerical contracts](../../docs/contracts.md) before interpreting results.

## What runs

One brain with four continuing rows is taught over 24 bouts. A bout opens with one
cue event (two rows cued A, two cued B, by a frozen permutation) and continues with
eleven identical drive events. The label at the cue event is the cued action; at
every later event it is the opposite of the row's own last executed action. Lessons
go through `Learner.step(brain.stimulus(x), labels)` with the live trace in the
stimulus, followed by a free greedy `act`. The `every` arm teaches before each act;
the `mismatch` arm acts first and teaches only the rows that were wrong. The working
trace is the only history the brain carries; `Learner.step` leaves it untouched and
only the free act advances it.

After teaching, a final bout runs its cue and four lead events, then saves the probe
checkpoint between two actions. The live life and eight forks of that checkpoint run
the same 64 drive events: intact, restored, erased trace, shuffled trace (transplanted
from a row with the other cue), reset, static (trace erased before every act),
static_cold (`reset` before every act), the flip-flop control and frozen random
actions. Disturbance forks run 8 drive events, then a pause of 1, 2 or 4 zero-drive
events or one distractor event, then 32 drive events. Paced forks run 32 events at a
declared cadence of one event per 100 ms, at 50 ms and 200 ms, with an extra event
in one slot, with a skipped slot, and at 100 ms beside one spinning process per
logical CPU; unpaced forks vary the free-step budget and the residual tolerance. Every
variant's per-event actions are compared with an unpaced reference from the same
checkpoint.

One event is one accepted observation and its one free act. The clock supplies due
times to the driver and nothing to the brain. Sweeps, residual checks, refusals and
checkpoint IO are work. A refused act is a missed action that leaves state and trace
unchanged while the clock consumes the event.

## Run and verify

```sh
python benchmarks/rhythm/steady_rhythm.py --out /tmp/rhythm-run
python benchmarks/rhythm/steady_rhythm.py --verify /tmp/rhythm-run
python benchmarks/rhythm/report.py /tmp/rhythm-run/summary.json
```

The default run executes the frozen protocol on the five confirmation seeds, both
recipes and both arms (20 founders) in about 7 minutes on one laptop CPU, most of it
paced sleep. Any override (`--bouts`, `--window`, `--decay`, ...) marks the receipt
`frozen_protocol: false`. `--no-cadence` skips the paced and repair-speed forks. The
output directory holds the protocol copy, frozen inputs, every checkpoint,
`reports.jsonl` with every settlement and teaching report, `runs.jsonl` and the
`summary.json` receipt. `results/` keeps the receipts quoted below; checkpoints and
per-call reports stay out of the repository. `develop_recipe.py` reproduces the
development grid on the development seeds.

## Results on the five fresh seeds, 2026-10-04

Receipt: `results/confirmation-2026-10-04.json.gz` (a `cadence.Receipt`; its body is
read by `report.py`). Protocol SHA-256
`424a1604308725b3fa2e7708a947c6362150d5318d44f7f0cba39edef1e3a81f`, frozen. Every
planned founder completed; none was capped, refused an act or refused a lesson. The
tables below are printed by `report.py` from the stored per-event actions.

### Window controls, means over founders (alternation / agreement / refusals)

| Branch | compose_default/every | compose_default/mismatch | selected/every | selected/mismatch |
| --- | --- | --- | --- | --- |
| intact | 0.37 / 0.50 / 0 | 0.36 / 0.61 / 0 | 0.44 / 0.60 / 0 | 0.44 / 0.49 / 0 |
| restored | 0.37 / 0.50 / 0 | 0.36 / 0.61 / 0 | 0.44 / 0.60 / 0 | 0.44 / 0.49 / 0 |
| erased | 0.35 / 0.53 / 0 | 0.37 / 0.46 / 0 | 0.42 / 0.50 / 0 | 0.40 / 0.51 / 0 |
| shuffled | 0.37 / 0.50 / 0 | 0.36 / 0.41 / 0 | 0.44 / 0.51 / 0 | 0.44 / 0.50 / 0 |
| reset | 0.35 / 0.53 / 0 | 0.37 / 0.46 / 0 | 0.42 / 0.50 / 0 | 0.40 / 0.51 / 0 |
| static | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 |
| static_cold | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 | 0.00 / 0.50 / 0 |
| flipflop | 1.00 / 1.00 / 0 | 1.00 / 1.00 / 0 | 1.00 / 1.00 / 0 | 1.00 / 1.00 / 0 |
| random | 0.50 / 0.50 / 0 | 0.50 / 0.51 / 0 | 0.50 / 0.48 / 0 | 0.50 / 0.50 / 0 |
| shuffled_donor | 0.37 / 0.50 / 0 | 0.36 / 0.61 / 0 | 0.44 / 0.60 / 0 | 0.44 / 0.49 / 0 |

### Founders

| Recipe | Arm | Seed | intact alternation | intact agreement | blocks of 8 | erased | reset | static | flipflop | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compose_default | every | 101 | 0.67 | 0.51 | 0.54 0.71 0.68 0.75 0.64 0.64 0.71 0.71 | 0.62 | 0.62 | 0.00 | 1.00 | 0.54 |
| compose_default | every | 102 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 |
| compose_default | every | 103 | 0.50 | 0.50 | 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 | 0.51 | 0.51 | 0.00 | 1.00 | 0.45 |
| compose_default | every | 104 | 0.67 | 0.49 | 0.68 0.68 0.64 0.68 0.68 0.64 0.68 0.68 | 0.60 | 0.60 | 0.00 | 1.00 | 0.48 |
| compose_default | every | 105 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.54 |
| compose_default | mismatch | 101 | 0.13 | 0.57 | 0.36 0.07 0.11 0.11 0.11 0.11 0.07 0.11 | 0.44 | 0.44 | 0.00 | 1.00 | 0.54 |
| compose_default | mismatch | 102 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 |
| compose_default | mismatch | 103 | 1.00 | 1.00 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 | 0.84 | 0.84 | 0.00 | 1.00 | 0.45 |
| compose_default | mismatch | 104 | 0.67 | 0.49 | 0.71 0.64 0.71 0.68 0.64 0.71 0.68 0.64 | 0.56 | 0.56 | 0.00 | 1.00 | 0.48 |
| compose_default | mismatch | 105 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.54 |
| selected | every | 101 | 0.54 | 0.37 | 0.57 0.61 0.57 0.57 0.64 0.57 0.61 0.57 | 0.56 | 0.56 | 0.00 | 1.00 | 0.54 |
| selected | every | 102 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 |
| selected | every | 103 | 1.00 | 1.00 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 | 0.95 | 0.95 | 0.00 | 1.00 | 0.45 |
| selected | every | 104 | 0.66 | 0.65 | 0.64 0.64 0.71 0.64 0.64 0.71 0.64 0.64 | 0.60 | 0.60 | 0.00 | 1.00 | 0.48 |
| selected | every | 105 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.54 |
| selected | mismatch | 101 | 0.84 | 0.49 | 0.86 0.86 0.86 0.86 0.86 0.86 0.86 0.86 | 0.79 | 0.79 | 0.00 | 1.00 | 0.54 |
| selected | mismatch | 102 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 |
| selected | mismatch | 103 | 0.73 | 0.49 | 0.68 0.79 0.68 0.75 0.75 0.79 0.68 0.71 | 0.63 | 0.63 | 0.00 | 1.00 | 0.45 |
| selected | mismatch | 104 | 0.65 | 0.49 | 0.46 0.71 0.64 0.71 0.75 0.61 0.68 0.75 | 0.56 | 0.56 | 0.00 | 1.00 | 0.48 |
| selected | mismatch | 105 | 0.00 | 0.50 | 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.54 |

### Disturbances, means over founders

| Disturbance | Model | pre alternation | post alternation | agreement hold | agreement continue | recovered rows / rows |
| --- | --- | --- | --- | --- | --- | --- |
| distractor | compose_default/every brain | 0.34 | 0.37 | 0.47 | 0.57 | 0/20 |
| distractor | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| distractor | random | 0.44 | 0.52 | 0.50 | 0.49 | 0/20 |
| pause1 | compose_default/every brain | 0.34 | 0.36 | 0.47 | 0.57 | 0/20 |
| pause1 | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| pause1 | random | 0.49 | 0.48 | 0.51 | 0.52 | 0/20 |
| pause2 | compose_default/every brain | 0.34 | 0.36 | 0.53 | 0.53 | 0/20 |
| pause2 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause2 | random | 0.55 | 0.52 | 0.49 | 0.49 | 0/20 |
| pause4 | compose_default/every brain | 0.34 | 0.35 | 0.52 | 0.51 | 0/20 |
| pause4 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause4 | random | 0.50 | 0.52 | 0.44 | 0.49 | 0/20 |
| distractor | compose_default/mismatch brain | 0.41 | 0.43 | 0.37 | 0.62 | 4/20 |
| distractor | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| distractor | random | 0.44 | 0.52 | 0.50 | 0.49 | 0/20 |
| pause1 | compose_default/mismatch brain | 0.41 | 0.42 | 0.37 | 0.61 | 3/20 |
| pause1 | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| pause1 | random | 0.49 | 0.48 | 0.51 | 0.52 | 0/20 |
| pause2 | compose_default/mismatch brain | 0.41 | 0.36 | 0.59 | 0.61 | 0/20 |
| pause2 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause2 | random | 0.55 | 0.52 | 0.49 | 0.49 | 0/20 |
| pause4 | compose_default/mismatch brain | 0.41 | 0.36 | 0.60 | 0.60 | 4/20 |
| pause4 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause4 | random | 0.50 | 0.52 | 0.44 | 0.49 | 0/20 |
| distractor | selected/every brain | 0.44 | 0.44 | 0.49 | 0.65 | 4/20 |
| distractor | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| distractor | random | 0.44 | 0.52 | 0.50 | 0.49 | 0/20 |
| pause1 | selected/every brain | 0.44 | 0.45 | 0.49 | 0.62 | 4/20 |
| pause1 | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| pause1 | random | 0.49 | 0.48 | 0.51 | 0.52 | 0/20 |
| pause2 | selected/every brain | 0.44 | 0.44 | 0.51 | 0.69 | 4/20 |
| pause2 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause2 | random | 0.55 | 0.52 | 0.49 | 0.49 | 0/20 |
| pause4 | selected/every brain | 0.44 | 0.45 | 0.52 | 0.57 | 4/20 |
| pause4 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause4 | random | 0.50 | 0.52 | 0.44 | 0.49 | 0/20 |
| distractor | selected/mismatch brain | 0.40 | 0.43 | 0.50 | 0.50 | 0/20 |
| distractor | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| distractor | random | 0.44 | 0.52 | 0.50 | 0.49 | 0/20 |
| pause1 | selected/mismatch brain | 0.40 | 0.43 | 0.50 | 0.50 | 0/20 |
| pause1 | flipflop | 1.00 | 1.00 | 0.00 | 1.00 | 20/20 |
| pause1 | random | 0.49 | 0.48 | 0.51 | 0.52 | 0/20 |
| pause2 | selected/mismatch brain | 0.40 | 0.44 | 0.49 | 0.50 | 0/20 |
| pause2 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause2 | random | 0.55 | 0.52 | 0.49 | 0.49 | 0/20 |
| pause4 | selected/mismatch brain | 0.40 | 0.44 | 0.50 | 0.48 | 0/20 |
| pause4 | flipflop | 1.00 | 1.00 | 1.00 | 1.00 | 20/20 |
| pause4 | random | 0.50 | 0.52 | 0.44 | 0.49 | 0/20 |

### Cadence, repair speed and host load

| Variant | founders identical to reference | refusals | sweeps per event | solve ms mean / max | lateness ms max | missed deadlines | slot agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| budget_16 | 20/20 | 0 | 5.5 | unpaced | unpaced | unpaced | unpaced |
| budget_256 | 20/20 | 0 | 20.4 | unpaced | unpaced | unpaced | unpaced |
| budget_64 | 20/20 | 0 | 20.4 | unpaced | unpaced | unpaced | unpaced |
| paced_extra | 20/20 | 0 | 20.4 | 1.07 / 2.26 | 10.91 | 0 | 0.47 |
| paced_regular | 20/20 | 0 | 20.4 | 0.89 / 12.31 | 10.05 | 0 | 0.55 |
| paced_regular200 | 20/20 | 0 | 20.4 | 1.07 / 2.72 | 89.97 | 0 | 0.55 |
| paced_regular50 | 20/20 | 0 | 20.4 | 0.98 / 1.98 | 10.09 | 0 | 0.55 |
| paced_regular_load | 20/20 | 0 | 20.4 | 0.87 / 140.57 | 47.51 | 1 | 0.55 |
| paced_skipped | 20/20 | 0 | 20.5 | 1.09 / 2.44 | 10.11 | 0 | 0.48 |
| reference | 20/20 | 0 | 20.4 | unpaced | unpaced | unpaced | unpaced |
| tolerance_0.0001 | 20/20 | 0 | 22.4 | unpaced | unpaced | unpaced | unpaced |
| tolerance_0.01 | 20/20 | 0 | 19.8 | unpaced | unpaced | unpaced | unpaced |

### Custody and work

- shuffled rows reproducing the donor row's intact sequence: 80/80; erased equals reset in 20/20 founders; window cue followed in 41/80 rows
- founders planned 20, completed 20, capped False, run seconds 425
- probe continuation equal (actions and saved arrays): 20/20
- mid-pause continuation equal: 20/20
- protocol sha256 424a1604308725b3fa2e7708a947c6362150d5318d44f7f0cba39edef1e3a81f, frozen True
- compose_default/every: lessons 1440, refused 0, free budget exhausted 0, last-bout agreement with the label 0.38, flip-flop lessons 5760
- compose_default/mismatch: lessons 1205, refused 0, free budget exhausted 0, last-bout agreement with the label 0.43, flip-flop lessons 130
- selected/every: lessons 1440, refused 0, free budget exhausted 0, last-bout agreement with the label 0.45, flip-flop lessons 5760
- selected/mismatch: lessons 1246, refused 0, free budget exhausted 0, last-bout agreement with the label 0.46, flip-flop lessons 130

## Reading the controls

`static_cold` is the instrument's zero: a brain reset before every act under an
identical drive gives the same action at every event. `static` adds the warm neural
state and shows whether that state alone carries phase; it scores zero as well, so
the trace is necessary for any alternation. `erased` and `reset` keep the learned
parameters and lose the phase once; their action sequences are identical in every
founder, so the warm neural state contributes nothing once the trace is gone, and
the alternation that resumes from them is re-established by the parameters with the
pre-probe phase lost (agreement near 0.5). Every `shuffled` row reproduces the donor
row's intact sequence event for event: the transplanted trace carries the donor's
phase and determines the whole trajectory. `flipflop` is what an explicit recurrence
over the previous output does with the same lessons: alternation 1.00 in every
founder, under every disturbance. `random` has an expected alternation rate of 0.5.

Transient cue-following and continuing alternation are separated by the blocks of 8
events in the 64-event window. The founders that alternate hold their rate through
the eighth block; the founders that hold one action do so from the first block. The
teaching-phase agreement with the label (0.38 to 0.46 in the last bout) is reported
separately from the free window.

## Limits

The measured System 1 holds one action in two of five founders at both recipes and
both arms, alternates with period 2 in one founder (seed 103) under three of the four
recipe/arm combinations, and alternates in 54 to 84 percent of event pairs in the
others. The mean across founders, with every founder in the denominator, is 0.44 for
the selected recipe and 0.37 for the compose default, both below the uniform-random
baseline of 0.50 and far below the flip-flop control. Disturbances change little
because the rhythm that exists is fragile before the disturbance; the flip-flop
recovers in 20 of 20 rows after every disturbance, the brain in 0 to 4. Four
rows share one set of parameters; a founder with another seed is another brain. The
paced measurements are from one laptop and describe its sleep granularity and solve
time, not an embedded clock. Identical per-event actions across cadences, budgets,
tolerances and host load follow from a deterministic CPU solve with no hidden thread;
they establish event-time semantics, not robustness to a solver that fails to
qualify, and no budget down to 16 sweeps produced a refusal on this 68-neuron graph.
Issue 93 consumes a passed continuing-action seam; this chamber reports a failed
one for the current System 1 and the instrument to measure a repair.
