# Key-door nursery

This bounded instrument asks whether one continuing `Brain.compose` life can credit an
action whose worth arrives several irrelevant choices later, keep exploring at a cost
when its food stops, and move its search when the source of the food moves. It is the
delayed key-door reward nursery of
[issue 111](https://github.com/muellerberndt/cadence/issues/111), roadmap row 07 of
[issue 109](https://github.com/muellerberndt/cadence/issues/109), built on the routine
and repair loop of [issue 122](https://github.com/muellerberndt/cadence/issues/122) and
modelled on the [odour nursery](../reversal/README.md) of row 05. It measures one declared
System 1 operating point with arousal against the same brain without it, the brain without
eligibility, with yoked rewards, frozen, without its pouch sense, a tabular learner with the
same information and uniform-random actions. It establishes no default. Read
[routine and repair](../../docs/continuous.md#routine-and-repair-live), the
[world-model guide](../../docs/world-model.md) and
[numerical contracts](../../docs/contracts.md) before interpreting results.

## What runs

A creature walks a corridor once per trip: empty floor, a chest, a lamp, `D` levers and a
door, met in that order, 14 cells in all, so that every trip takes the same number of
moments whatever the delay. At every cell it passes or interacts. Under rule A the chest
holds the key; under rule B the lamp does. Interacting at the door with the key in the
pouch pays +1 and ends the trip; interacting with the chest, the lamp or a lever when it
holds no key costs 0.25; the floor has nothing to interact with; taking the key pays
nothing by itself. Its worth arrives `D + 1` cells later, after the levers, whose number
varies by one from trip to trip, so event time is irregular and credit cannot follow the
last action blindly. The creature sees the kind of the cell it faces and, through the
pouch sense, whether it holds the key. One trip in twenty is cut short before the door at
a random cell, the key lost with it; such a trip ends with `done` clear, so the forecast
carries over into the next trip (a truncated bootstrap), while the door's end is terminal.
The outcome of a trip's last cell is delivered with the first observation of the next
trip, as `step` and `live` define it. A life is one stream without resets: rule A for 500
trips, then rule B for 500. Nothing announces the change. The delays are 2, 5 and 10.

The need of this body is 0.03 reward per moment, a gene of `ArousalConfig` added for this
chamber: at full feeding the income is 1/14 per moment, so a fed creature's need is met and
a starving one wants its whole need. Its selection and the reason for its law are recorded
[below](#how-the-operating-point-and-the-founders-were-selected).

Every arm of a seed lives the same corridor sequence:

| Arm | What it is |
| --- | --- |
| `live` | `Brain.compose(6, 2, modules=(32,))` at the protocol's operating point, with `ArousalConfig()` at its founders and the chamber's need, through `Brain.live` |
| `step` | the same brain and operating point without arousal, through `step`: it samples and learns at every moment (the simpler control) |
| `lambda-zero` | the `live` brain with the eligibility decay `lam` at zero: credit reaches only the last action |
| `yoked` | the `live` brain whose door outcome is banked and paid at a random cell of the next trip: the same rewards, credited to nothing it did for them |
| `frozen` | the `live` brain after rule A, answering greedily and receiving no outcome |
| `blind` | the `live` brain without the pouch sense: it cannot tell whether it holds the key |
| `tabular` | epsilon-greedy Q(lambda) over (cell, pouch), the conventional online learner with the same information; its settings selected on the development seeds |
| `random` | uniform random actions |

The operating point of one continuing stream is the odour nursery's (working-trace
amplitude 0.3, consolidation 0.25, actor rate 0.1 with a bias rate of 0.01) with the
critic's rate at 5.0 in place of the composed 0.3, the eligibility decay at 0.95 in place
of 0.8 and the discount at 0.95 in place of 0.9; the composed values are the controls, and
the selection on the development seeds is recorded
[below](#how-the-operating-point-and-the-founders-were-selected).

Readings per rule, over the trips that reached the door: the share of trips that ended with
food, the share in which the key was taken, the wrong interactions per trip and the share
of door visits with the key at which the creature interacted, each over the last 50 trips;
the lag, the first trip from which 90% of the next 20 trips are fed; the number of trips
cut short; the greedy choice and the policy's probability of interacting, per cell and
pouch state, of a saved and reloaded copy every 25 trips; the probability of interacting
under the behaviour that acted and under the base policy, per cell and pouch state and
over the visits, read from the living brain; the share of aroused moments over the rule
and over its second half; and the work of the life: moments and settling sweeps per mode,
learning sweeps, probes, checkpoints, memory reads and writes, brains, refused sweeps and
the latency of a moment in each mode. The greedy probes read a blank-trace copy and are
reported separately from the executed behaviour.

## Gates, fixed before the confirmation run

Over the confirmation lives of the `live` arm at delays 2 and 5, at least 90% end each
rule with 90% of their last 50 trips fed, waste at most 1.5 wrong interactions per trip in
the last 50 trips of each rule, and spend no more than 35% of the second half of each rule
aroused. Delay 10 is reported without a gate. [protocol.json](protocol.json) holds the
gates, the seeds and every setting. It was committed and pushed before its confirmation
seeds were run. It is the chamber's second freeze; [the first](#the-first-freeze-2026-10-06)
gated frugality at 0.5 wrong interactions per trip and calm at 20%, and is recorded below
with its result. The frugality bound of 1.5 is below the uniform-random policy's wrong
interactions at delay 2 (1.5 per trip) and a third of the lazy interact-everywhere
policy's at delay 2 (4 to 5); it admits one retained habit per trip, which the first
freeze showed to be a phenomenon of its own, reported per life below rather than gated. The
calm bound of 35% follows one trip in twenty being cut: each cut is a missed meal and a
contradicted forecast, and a bout of arousal follows it by the law.

## Run and verify

```sh
python benchmarks/keydoor/key_door.py --seeds confirmation --out /tmp/keydoor.json.gz
python benchmarks/keydoor/key_door.py --verify /tmp/keydoor.json.gz --current
python benchmarks/keydoor/key_door.py --report /tmp/keydoor.json.gz
python -m pytest -q benchmarks/keydoor
```

The first command runs the frozen protocol: eight arms, three delays and ten confirmation
seeds, 240 lives. `--arms`, `--seeds` and `--delays` select a part. `--genes`, `--point`,
`--cost`, `--food` and `--episodes` override the arousal genes, the operating point and the
world, and mark the receipt `frozen_protocol: false`; a `null` in `--point` leaves that
setting at its released default, and `--point '{"learner": {...}}'` sets the settling
learner. The receipt is a `cadence.Receipt` bound to the chamber's source and every module
of the library. `--verify` checks its canonical form and digest, source manifest integrity,
one row for every planned life in order, the ranges of the recorded shares, the visits
against the moments lived, the trips that reached the door and the cut trips against the
trips planned, the gates recomputed from the rows, the embedded protocol text against its
hash and any claim of frozen settings. `--current` also requires the source manifest and
the protocol hash of the files present. These checks validate the recorded summaries; they
cannot reconstruct unrecorded executed actions or independently prove that a run occurred.
`--report` prints the tables. `results/` keeps the receipts quoted here. The guards run
short lives of the `live` arm and its controls, the world's accounting through what the
learner is handed (the door terminal, a cut trip not), the checkpoint continuation inside
the delay with the preceding outcome pending, the independence of a life from its probes,
a refused answer charged and recorded, the gate arithmetic and the receipt's custody.

## Results on ten fresh seeds, 2026-10-06

Receipt: `results/confirmation-2026-10-06.json.gz`. Protocol SHA-256
`8cc9ed188c91a7f7098a978b2e5f963f43b8d5e46797d4be5144a66ee9b8124d`, frozen at commit
`b57f9fd`; seeds 800 to 809; cadence 0.75.0 with the `need` gene of this branch, NumPy 2.5.3,
Python 3.13.0, macOS arm64. All 240 lives completed and none refused an answer. **Three of
the four gates passed and the fourth did not, so the gates did not pass.** Of the 20 gated
lives of the `live` arm, 20 acquired rule A (fed 1.00 at delay 2 and 0.99 at delay 5 at its
end, median lags 9 and 66 trips), 19 re-adapted after the key moved (fed 0.97 and 1.00 at the
end of rule B, median lags 27 and 38 trips, the door opened with the key on 99% of the visits),
19 were frugal (0.31 and 0.13 wrong interactions per trip at the end of rule A, 0.45 and 0.24
at the end of rule B) and 17 were calm against the 18 required. The misses: one life at delay
2 kept touching the empty chest and the lamp while holding the key after the move, ended rule
B fed on 0.70 of its trips with 2.1 wrong interactions per trip and spent 69% of the late
moments aroused; one life at delay 2 spent 54% of the late half of rule A aroused while fed on
every trip without a wrong interaction; one life at delay 5 acquired at trip 316 and
re-adapted at trip 373 and was still searching in the second half of each rule (32% and 46%).
At delay 10, ungated, 5 of 10 lives acquired rule A and 8 of 10 ended rule B fed (0.89 at its
end, median lag 172 trips; the tabular learner 0.88).

The controls on the same corridors: `step`, learning at every moment, ended rule B fed at 0.97
and 0.68 while aroused at every moment and wasting 1.8 and 0.7 interactions per trip;
`lambda-zero` at 0.68 and 0.33, the door's credit reaching only the last action;
`yoked`, paid the same rewards at a random cell of the next trip, at 0.28 and 0.07, near
random; `frozen` at 0.00 (no adaptation without outcomes); `blind`, without the pouch sense,
at 1.00 and 1.00 with no wrong interaction at either delay, so the working trace carries the
key through the levers and the pouch sense adds nothing at these delays; `tabular` at 0.90
and 0.92, bounded by its epsilon; `random` at 0.22. The `live` arm's routine moment took
0.71 to 0.79 ms (median) against 2.0 to 2.2 ms aroused, at 25 to 31 settling sweeps per
routine moment and 22 to 24 per aroused moment: routine is cheaper in time (no learning
phases), not in sweeps. It spent 13% and 21% of its moments aroused over the whole life and
5% and 4% over the second half of rule A, wrote 1,870 and 2,943 records to memory and spent
12,800 sweeps per life on the probes of saved copies.

### Declared variants on the same seeds

`results/variant-scarcity-2026-10-06.json.gz` (`--food 0.5`, the door paying one time in two
with the key): no life acquired rule A or rule B at either delay (fed 0.25 and 0.22 at the end
of rule A, 0.35 and 0.27 at the end of rule B), 83% to 88% of the moments aroused. The
expected income at full feeding, 0.036 per moment, sits at the need of 0.03, and the forecast
of the door is contradicted on every other trip, so the creature never leaves arousal and the
flood of sampled wrong interactions holds it in the latch recorded below.
`results/variant-composed-critic-2026-10-06.json.gz` (critic rate 0.3, eligibility decay 0.8,
discount 0.9, all else the frozen point): 13 of 20 lives acquired rule A and 11 re-adapted
(at delay 5, 4 and 3 of 10), against 20 and 19 for the frozen point.

### The tables

Every number below is read from the receipt by `--report`.

### Rule A, chest holds the key: trips fed in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 1.00 (1.00) | 0.99 (0.92) | 0.69 (0.12) |
| `step` | 0.94 (0.86) | 0.90 (0.78) | 0.57 (0.00) |
| `lambda-zero` | 0.82 (0.04) | 0.51 (0.00) | 0.12 (0.00) |
| `yoked` | 0.35 (0.14) | 0.14 (0.02) | 0.15 (0.02) |
| `frozen` | 1.00 (1.00) | 0.99 (0.92) | 0.69 (0.12) |
| `blind` | 1.00 (1.00) | 1.00 (1.00) | 0.57 (0.04) |
| `tabular` | 0.68 (0.00) | 0.89 (0.84) | 0.89 (0.84) |
| `random` | 0.25 (0.16) | 0.26 (0.16) | 0.23 (0.14) |

### Rule B, lamp holds the key: trips fed in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.97 (0.70) | 1.00 (0.98) | 0.89 (0.02) |
| `step` | 0.97 (0.90) | 0.68 (0.02) | 0.43 (0.00) |
| `lambda-zero` | 0.68 (0.02) | 0.33 (0.00) | 0.13 (0.00) |
| `yoked` | 0.28 (0.06) | 0.07 (0.02) | 0.13 (0.00) |
| `frozen` | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| `blind` | 1.00 (1.00) | 1.00 (1.00) | 0.70 (0.02) |
| `tabular` | 0.90 (0.84) | 0.92 (0.82) | 0.88 (0.78) |
| `random` | 0.22 (0.10) | 0.22 (0.18) | 0.28 (0.12) |

### Rule A: key taken in the last 50 trips, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 1.00 (1.00) | 1.00 (0.98) | 0.76 (0.34) |
| `step` | 0.95 (0.90) | 0.92 (0.80) | 0.59 (0.04) |
| `lambda-zero` | 0.85 (0.22) | 0.59 (0.10) | 0.24 (0.06) |
| `yoked` | 0.52 (0.28) | 0.38 (0.18) | 0.35 (0.18) |
| `frozen` | 1.00 (1.00) | 1.00 (0.98) | 0.76 (0.34) |
| `blind` | 1.00 (1.00) | 1.00 (1.00) | 0.63 (0.22) |
| `tabular` | 0.75 (0.04) | 0.95 (0.90) | 0.93 (0.90) |
| `random` | 0.49 (0.38) | 0.50 (0.32) | 0.50 (0.44) |

### Rule B: key taken in the last 50 trips, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.98 (0.80) | 1.00 (1.00) | 0.90 (0.14) |
| `step` | 0.98 (0.94) | 0.69 (0.04) | 0.44 (0.00) |
| `lambda-zero` | 0.72 (0.14) | 0.41 (0.12) | 0.25 (0.08) |
| `yoked` | 0.49 (0.18) | 0.23 (0.14) | 0.28 (0.12) |
| `frozen` | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |
| `blind` | 1.00 (1.00) | 1.00 (1.00) | 0.75 (0.10) |
| `tabular` | 0.94 (0.86) | 0.97 (0.94) | 0.94 (0.88) |
| `random` | 0.49 (0.42) | 0.48 (0.36) | 0.49 (0.38) |

### Rule A: wrong interactions per trip in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.31 (0.00) | 0.13 (0.00) | 1.53 (0.00) |
| `step` | 0.82 (0.02) | 0.55 (0.06) | 0.31 (0.12) |
| `lambda-zero` | 0.53 (0.00) | 0.62 (0.00) | 1.39 (0.00) |
| `yoked` | 0.78 (0.02) | 1.53 (0.84) | 2.96 (1.84) |
| `frozen` | 0.31 (0.00) | 0.13 (0.00) | 1.53 (0.00) |
| `blind` | 0.00 (0.00) | 0.00 (0.00) | 1.62 (0.00) |
| `tabular` | 0.15 (0.04) | 0.49 (0.28) | 1.62 (1.10) |
| `random` | 1.45 (1.20) | 2.98 (2.68) | 5.47 (5.14) |

### Rule B: wrong interactions per trip in the last 50, mean (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.45 (0.00) | 0.24 (0.00) | 0.43 (0.00) |
| `step` | 1.78 (0.92) | 0.67 (0.10) | 0.50 (0.06) |
| `lambda-zero` | 0.39 (0.00) | 0.70 (0.00) | 1.37 (0.00) |
| `yoked` | 0.88 (0.00) | 1.09 (0.68) | 2.26 (1.46) |
| `frozen` | 1.00 (1.00) | 1.00 (1.00) | 0.77 (0.00) |
| `blind` | 0.00 (0.00) | 0.00 (0.00) | 0.87 (0.00) |
| `tabular` | 0.15 (0.06) | 0.53 (0.34) | 1.66 (1.06) |
| `random` | 1.49 (1.30) | 2.91 (2.36) | 5.36 (4.74) |

### Rule A: first 20-trip window 90% fed, median trip (lives / lives)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 9 (10/10) | 65 (10/10) | 303 (8/10) |
| `step` | 122 (10/10) | 188 (10/10) | 250 (3/10) |
| `lambda-zero` | 14 (8/10) | 6 (6/10) | 9 (2/10) |
| `yoked` | 49 (4/10) | 132 (2/10) | 50 (2/10) |
| `frozen` | 9 (10/10) | 65 (10/10) | 303 (8/10) |
| `blind` | 14 (10/10) | 113 (10/10) | 56 (6/10) |
| `tabular` | 20 (8/10) | 17 (10/10) | 33 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) |

### Rule B: first 20-trip window 90% fed, median trip (lives / lives)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 26 (10/10) | 38 (10/10) | 172 (9/10) |
| `step` | 23 (10/10) | 122 (8/10) | 140 (6/10) |
| `lambda-zero` | 29 (7/10) | 15 (3/10) | 116 (1/10) |
| `yoked` | 27 (2/10) | none (0/10) | 464 (1/10) |
| `frozen` | none (0/10) | none (0/10) | none (0/10) |
| `blind` | 13 (10/10) | 32 (10/10) | 311 (6/10) |
| `tabular` | 29 (10/10) | 9 (10/10) | 32 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) |

### Rule A behaviour: probability of taking at the chest without the key, median (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.965 (0.659) | 0.863 (0.558) | 0.555 (0.407) |
| `step` | 0.844 (0.666) | 0.793 (0.592) | 0.464 (0.132) |
| `lambda-zero` | 0.972 (0.241) | 0.728 (0.186) | 0.214 (0.172) |
| `yoked` | 0.486 (0.408) | 0.454 (0.338) | 0.428 (0.276) |
| `frozen` | 0.965 (0.659) | 0.863 (0.558) | 0.555 (0.407) |
| `blind` | 0.971 (0.544) | 0.849 (0.518) | 0.478 (0.227) |
| `tabular` | 0.939 (0.056) | 0.938 (0.691) | 0.889 (0.291) |
| `random` | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) |

### Rule B behaviour: probability of taking at the lamp without the key, median (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.956 (0.747) | 0.920 (0.477) | 0.726 (0.217) |
| `step` | 0.936 (0.731) | 0.598 (0.028) | 0.267 (0.015) |
| `lambda-zero` | 0.899 (0.154) | 0.176 (0.149) | 0.144 (0.141) |
| `yoked` | 0.424 (0.349) | 0.224 (0.152) | 0.280 (0.155) |
| `frozen` | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| `blind` | 0.978 (0.829) | 0.950 (0.689) | 0.466 (0.156) |
| `tabular` | 0.902 (0.757) | 0.927 (0.870) | 0.911 (0.857) |
| `random` | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) |

### Rule B behaviour: probability of interacting at a lever with the key, median (minimum)

| Arm | D=2 | D=5 | D=10 |
| --- | --- | --- | --- |
| `live` | 0.045 (0.006) | 0.024 (0.006) | 0.063 (0.012) |
| `step` | 0.315 (0.035) | 0.041 (0.020) | 0.019 (0.011) |
| `lambda-zero` | 0.135 (0.007) | 0.147 (0.005) | 0.141 (0.034) |
| `yoked` | 0.365 (0.022) | 0.226 (0.149) | 0.267 (0.148) |
| `frozen` | none | none | none |
| `blind` | 0.007 (0.002) | 0.011 (0.001) | 0.151 (0.006) |
| `tabular` | 0.051 (0.050) | 0.092 (0.080) | 0.148 (0.137) |
| `random` | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) |

### The live arm: arousal and work, medians over lives

| Delay | aroused, whole life | aroused, second half of rule A | sweeps per routine moment | per aroused moment | learning sweeps | probe sweeps | memory reads | memory writes | routine ms (median, p90) | aroused ms (median, p90) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.131 | 0.053 | 25.3 | 23.9 | 19287 | 12800 | 15449 | 1870 | 0.71, 2.42 | 2.01, 8.40 |
| 5 | 0.210 | 0.042 | 31.0 | 22.0 | 33100 | 12800 | 16556 | 2943 | 0.79, 2.43 | 2.16, 10.27 |
| 10 | 0.461 | 0.494 | 26.4 | 23.1 | 67488 | 12816 | 20110 | 6444 | 0.68, 2.03 | 1.94, 6.00 |

### Gates

```json
{
 "10": {
  "acquired": 0.5,
  "adapted": 0.8,
  "calm": 0.4,
  "crashed": 0,
  "frugal": 0.5,
  "lives": 10
 },
 "2": {
  "acquired": 1.0,
  "adapted": 0.9,
  "calm": 0.8,
  "crashed": 0,
  "frugal": 0.9,
  "lives": 10
 },
 "5": {
  "acquired": 1.0,
  "adapted": 1.0,
  "calm": 0.9,
  "crashed": 0,
  "frugal": 1.0,
  "lives": 10
 },
 "passed": false,
 "pooled": {
  "acquired": 1.0,
  "adapted": 0.95,
  "calm": 0.85,
  "crashed": 0,
  "frugal": 0.95,
  "lives": 20
 }
}
```

## The first freeze, 2026-10-06

Receipts: `results/freeze1-confirmation-2026-10-06.json.gz` and the two variants beside it.
Protocol SHA-256 `2f214aaebff44070e1c0428fec3159005cef7f2ec13cb949dddf5b15b7e170c0`, frozen at
commit `35fcb14`; seeds 700 to 709; the same world, operating point and need as the second
freeze, with the gates at 0.5 wrong interactions per trip and 20% late arousal. **Its gates
did not pass.** Of the 20 gated lives of the `live` arm, 18 ended rule A fed and all 20
ended rule B fed (fed 1.00 at both delays, median lags of 28 and 34 trips after the key
moved, with the key taken on every trip); 11 were frugal and 14 calm by the first gates.
The misses were of three kinds: seven lives kept one extra interaction per trip while fed,
at the empty chest under rule B or at the lamp while already holding the key under rule A
(wrong 1.0 to 1.2 with fed 1.00; one life 2.6); six spent 22% to 36% of a late half aroused
while fed and frugal, in bouts that follow the cut trips; and one life at delay 5 never
acquired rule A (fed 0.24, aroused 87% of the late half) and then re-adapted to 1.00 under
rule B. At delay 10, 4 of 10 acquired and 2 of 10 re-adapted. The controls: `step` ended
rule B fed at 0.98 and 0.77 while aroused at every moment and wasting 0.8 to 1.5
interactions per trip; `lambda-zero` 0.51 and 0.30; `frozen` 0.00 (no adaptation without
outcomes); `blind`, without the pouch sense, 0.98 and 0.91 (the working trace carries the
key); `tabular` 0.90 and 0.92, and 0.91 at delay 10 where the brain reached 0.48;
`random` 0.25. The `yoked` arm crashed in 28 of 30 lives on a trip cut to one cell, where
its random payment cell had no range (`integers(0)`): a fault of that control alone,
repaired in the second freeze. The scarcity variant (`--food 0.5`, the door paying one time
in two) acquired nothing at either delay (fed 0.14 to 0.36, aroused 88% to 92%): with the
expected income at the need, the forecast of the door is contradicted on every other trip
and the creature never leaves arousal. The composed-critic variant (critic rate 0.3,
eligibility decay 0.8, discount 0.9, all else the frozen point) acquired 11 of 20 and
re-adapted 10 of 20, against 18 and 20 for the frozen point. The gates of the second
freeze were revised on this record as stated above; the point and the world were not.

## How the operating point and the founders were selected

Development used seeds 0 to 3 at every delay; the confirmation seeds were first run on the
frozen protocol. The odour nursery's operating point (working-trace amplitude 0.3,
consolidation 0.25, actor rate 0.1 with a bias rate of 0.01) was the starting point, and
the chamber was built up from a short corridor to the one above.

- **A need, and its unit.** With the arousal founders of 0.75.0 a creature whose food
  stops is roused only by the shortfall of its recent reward below its long-run reward,
  and a newborn that has never been paid is content with nothing: at delay 5 no
  development life acquired the task. A need, the reward per moment the body requires,
  rouses both. Measured in the law's unit, the spread of the outcomes the brain has
  learned from, the need did not work on this corridor: a reward of +1 once in 14 moments
  has a mean of 1/14 and a spread near 1/sqrt(14), so a starving creature's want stayed
  near 0.2 whatever the need, at the edge of the threshold, and its exploration heat
  barely rose; on the 24 development lives of the first grid 1 of 24 re-adapted after the
  key moved. The need's want is therefore the share of the need the recent reward leaves
  unmet, 1 for a creature that is never paid and 0 for one whose income covers it. A need
  of zero removes it and leaves the law of 0.75.0 unchanged. With the need at 0.03 and an
  eligibility decay of 0.9, every development life at delay 2 acquired the task and
  re-adapted after the key moved; delay 5 did not.
- **Corridor length.** In the first corridor the trip grew with the delay (`D + 3`
  cells), so the achievable income per moment fell with the delay and no single need
  suited every rung. Every trip now has 14 cells, with floor cells before the chest making
  up the difference, so the income at full feeding is the same at every delay.
- **The cost of a wrong interaction.** At a cost of 0.1 the creatures that acquired the
  task interacted at every cell, levers included, and still fed: the frugal policy pays
  little more than the lazy one at delay 2. At 0.25 the lazy policy loses, and the
  creatures that learn the task stop touching the levers. The cost of 0.25 stays.
- **The latch under punished exploration.** At delay 5 the development lives that failed
  did so in one way, read from the living brain every 100 trips: within 200 to 300 trips
  the two motor units settled to the same activation for every cell and pouch state (a
  cosine of 1.000 between all ten situations), the pass unit near 0.95, the interact unit
  below rest, and the policy answered pass with a base probability of 0.005 everywhere.
  A unit at its rail has no slope left for the nudge, so no reward could move it again.
  The synapses stayed near initialization (mean efficacy 1.38 at birth, 1.41 after 300
  trips) and the motor bias reached 0.21, so the efficacy cap of 8 could not bite. The
  calm policy of a fed creature sits on motor activations near rest, with margins of a
  hundredth, so when its food stops and its arousal samples at three times the
  temperature, the behaviour is close to uniform, the levers punish about half the cells
  of every trip, and the flood of negative dopamine drives the pass unit to its rail.
  Dopamine centring made every nonzero outcome a unit signal and hastened the collapse;
  a sharper learner temperature of 0.1 or 0.05 deepened it; a heat of 0.5, an actor rate
  of 0.03, a trace amplitude of 0, a bias rate of 0, 64 modules and an efficacy cap of 2
  changed nothing. The lives that acquired rule A at delay 5 under these settings never
  re-adapted after the key moved, for the same reason.
- **The critic carries delayed credit.** The brains that acquired rule A at delay 5 had a
  flat critic: a value of 0.1 to 0.2 at every cell, 0.2 at the door with the key where +1
  follows with certainty, and the same value with and without the key. The composed
  critic learns at a rate of 0.3 divided by one plus the energy of its eligibility trace,
  which over a 14-cell trip is about 27, on a code whose activations average 0.03; it
  learns its bias and little else. Taking the key then produces no value jump, and the
  only credit for the chest is the actor's own eligibility, (gamma * lam)^k with k the
  cells to the door: 0.43 at delay 2 and 0.28 at delay 5, which is why delay 2 was solved
  and delay 5 was not. Raising the critic's rate makes the key's value appear (about
  +0.4 for holding it at the end of rule A) and the lives at delay 5 acquire and
  re-adapt; the rate is selected below with the composed 0.3 as the control. An
  unnormalized critic at 0.05 did the same for one seed and was slower for the other.
- **The other rail.** At a critic rate of 10 the values run high (above 1 at the door
  with the key, 0.4 on the empty floor) and after the key moves the development lives
  drift to the opposite latch: by trip 250 of rule B they interact at every cell with a
  behaviour probability of 0.85, five wrong interactions per trip, fed on two trips in
  three with a net income below zero, so the need is never met, the want stays at 1 and
  the creature never returns to routine; 800 trips did not end it. A rate of 30 reached
  that state under rule A. The rate is therefore chosen between the flat critic and the
  running one, on the readings of all three gates rather than on the fed share alone.
- **The point.** On seeds 0 to 3 at the three delays, critic rates of 3, 5 and 10 with an
  eligibility decay of 0.9 all acquired rule A at delays 2 and 5 in every life; after the
  key moved at delay 5, 3, 2 and 4 of the 4 lives re-adapted, the last with one wrong
  interaction per trip and a third of the late moments aroused. Rate 5 with the eligibility
  decay at 0.95 and the discount at 0.95, which carry the door's credit further back
  ((gamma * lam)^6 of 0.52 against 0.28), re-adapted 3 of 4 with 0.3 wrong interactions
  per trip and 12% of the late moments aroused. On seeds 0 to 7 at delays 2 and 5, with
  receipts, that point passed every gate reading in 6 of 8 lives at each delay (fed at
  the end of rule A 15 of 16, of rule B 15 of 16, frugal 13 of 16, calm 13 of 16); rate 5
  at the composed decay and discount passed 5 of 8 and 4 of 8, rate 3 there 3 of 8 and
  2 of 8. The misses of the frozen point: two lives at
  delay 2 kept interacting at the lamp while holding the key, a habit from acquisition
  that costs a quarter per trip and that a fed and calm creature does not unlearn; one
  life at delay 5 latched after the key moved; one life at delay 5 spent 21% of the late
  moments of rule A aroused. The frozen point is rate 5, decay 0.95, discount 0.95, and
  the gates stay as first declared. Delay 10 is left ungated: 0 to 2 of 4 development
  lives acquired it under any point tried, while the tabular learner acquires it.
- **The tabular learner.** Alpha in 0.1, 0.2 and 0.5, epsilon in 0.05, 0.1 and 0.2 and
  lambda in 0.8 and 0.9 were run on the development seeds at every delay, with gamma 0.9.
  Epsilon 0.05 left lives that never found the pair of interactions and epsilon 0.2 wasted
  one to three interactions per trip; at epsilon 0.1 every life ended both rules fed at
  0.74 or more, and alpha 0.5 with lambda 0.8 had the highest shares (0.89 to 0.93 under
  rule A, 0.89 to 0.92 under rule B, at the three delays) with lags of 7 to 95 trips. Its
  epsilon bounds it: one random pass at the chest or the door in twenty costs the trip.
  It solves delay 10 within the 500 trips; the brain's gates stop at delay 5.
