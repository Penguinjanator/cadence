# Odour nursery

This bounded instrument asks whether one continuing `Brain.compose` life can acquire a
rule, adapt when the rule turns over unannounced, return when it turns back, and keep an
unrelated skill throughout, after short and after long experience of the first rule. It
is the reversal chamber of
[issue 88](https://github.com/muellerberndt/cadence/issues/88), roadmap row 05 of
[issue 109](https://github.com/muellerberndt/cadence/issues/109), and the first
behavioural evidence for the routine-and-repair loop of
[issue 122](https://github.com/muellerberndt/cadence/issues/122). It measures one declared
System 1 operating point with arousal against the same brain without it, the released
defaults, two ablations of the brain, frozen, replay and reset controls, a tabular
learner with the same information and uniform-random actions. It establishes no default.
Read [routine and repair](../../docs/continuous.md#routine-and-repair-live), the
[world-model guide](../../docs/world-model.md) and
[numerical contracts](../../docs/contracts.md) before interpreting results.

## What runs

An animal meets one of four odours per trial, drawn at random, and avoids (0) or
approaches (1). Approaching the sugar odour pays +1, approaching another odour costs 1,
avoiding pays nothing. Odours 0 and 1 are the reversal pair: the sugar sits at odour 0
under rule A and at odour 1 under rule B. Odours 2 and 3 are the stable pair, the
unrelated skill: odour 2 is always sugar and odour 3 never. A life is one stream without
resets: rule A for the exposure (100, 300, 1,000, 3,000 or 10,000 trials), rule B for 600
trials and rule A again for 600. Nothing announces a change.

The world and its payoff are those of the reports behind the issue
([50](https://github.com/muellerberndt/cadence/issues/50),
[69](https://github.com/muellerberndt/cadence/issues/69)): a two-odour T-maze as a
bandit. Those reports ran cadence-net 0.20 and 0.42 learners whose interfaces no longer
exist, with three seeds and a lag read on greedy choices held for 20 trials. The nursery
keeps their world, adds the stable pair, runs the current `Brain.compose`, reads the lag
on executed actions and the greedy choices on saved copies, and ends rule B after 600
trials, so a slower reversal is reported as none.

Every arm of a seed lives the same odour sequence:

| Arm | What it is |
| --- | --- |
| `live` | `Brain.compose(4, 2, modules=(32,))` at the protocol's operating point, with `ArousalConfig()` at its founders, through `Brain.live` |
| `step` | the same brain and operating point without arousal, through `step`: it samples and learns at every moment (the simpler control) |
| `defaults` | `Brain.compose(4, 2, modules=(32,))` at the released defaults, through `step` |
| `memory-only` | the `live` brain with its actor's rates at zero: associative memory and critic learn, the graph's policy does not |
| `graph-only` | the `live` brain without its associative memory: the graph's reward learning alone |
| `frozen` | the `live` brain after rule A, answering greedily and receiving no outcome |
| `replay` | the `live` brain re-living rule A after the change: it is paid what its action earned under rule A, the same work on old evidence |
| `reset` | a newborn `live` brain at every rule change |
| `tabular` | epsilon-greedy tabular Q-learning with the same odour, action and reward; alpha 1.0 and epsilon 0.1, selected on the development seeds |
| `random` | uniform random actions |

The operating point of one continuing stream is working-trace amplitude 0.3,
consolidation 0.25 and an actor rate of 0.1 with a bias rate of 0.01; the composed
defaults are 3.0, 0.05, 1.0 and 0.05. Its selection on the development seeds is recorded
[below](#how-the-operating-point-and-the-founders-were-selected).

Readings per rule: the share of optimal executed actions in the last 100 trials; the lag,
the first trial from which the next 40 executed actions are at least 90% optimal; the
greedy choice and the policy's approach probability per odour of a saved and reloaded
copy every 25 trials; how often each odour was met and approached; the first executed
approach at the newly rewarded odour and the number of approaches executed there until
the greedy choice turned, read on a copy after each of the first 20 of them, which
separate too few contradicting witnesses from a failure to revise after them; whether the
stable pair stayed right at the probes; the share of aroused moments; and the free-solve
sweeps per moment in each mode with the eligibility and feedback sweeps of aroused
moments. The copies are read and the living brain is not: a life gives the same executed
actions with and without its probes. A refused answer raises and the life is recorded as
crashed.

## Gates, fixed before the confirmation run

Over the confirmation lives of the `live` arm at exposures of 300 and more, at least 90%
end each rule with 90% of their last 100 executed actions optimal, keep the stable pair
right on 95% of their probes, and spend no more than 20% of the second half of each rule
aroused; at each of those exposures the median reversal lag is at most 150 trials, a life
that never reverses counting as beyond it. Exposure 100 ends inside the youth and is
reported without a gate. [protocol.json](protocol.json) holds the gates, the seeds and
every setting. It was committed and pushed before its confirmation seeds were run. It is
the chamber's second freeze; [the first](#the-first-freeze) is recorded below.

## Run and verify

```sh
python benchmarks/reversal/odour_nursery.py --seeds confirmation --out /tmp/nursery.json.gz
python benchmarks/reversal/odour_nursery.py --verify /tmp/nursery.json.gz --current
python benchmarks/reversal/odour_nursery.py --report /tmp/nursery.json.gz
python -m pytest -q benchmarks/reversal
```

The first command runs the frozen protocol: ten arms, five exposures and ten
confirmation seeds, 500 lives, in about six minutes on nine laptop cores. `--arms`,
`--seeds` and `--exposures` select a part. `--genes`, `--point`, `--reliability`,
`--payoff` and `--jitter` override the arousal genes, the operating point and the world,
and mark the receipt `frozen_protocol: false`; a `null` in `--point` leaves that setting
at its released default. The receipt is a `cadence.Receipt` bound to the chamber's source
and every module of the library. `--verify` checks its canonical form and digest, one row
for every planned life and the gates recomputed from the rows; `--current` also requires
the source manifest and the protocol hash of the files present. `--report` prints the
tables below. `results/` keeps the receipts quoted here. The guards run short lives of
the `live` arm and its controls, the checkpoint continuation during a reversal, the
independence of a life from its probes, the gate arithmetic and the receipt's custody.

## Results on ten fresh seeds, 2026-10-05

Receipt: `results/confirmation-2026-10-05.json.gz`. Protocol SHA-256
`0a5f2b4e6c2e2767aebc57588caf3e338a0348f5ec14ab099f637413f09652c4`, frozen; seeds 300 to
309; cadence 0.74.0 source of this change, NumPy 2.5.3, Python 3.13.0, macOS arm64. All
500 lives completed and none refused an answer. **The gates passed.** Of the 40 gated
lives of the `live` arm, 40 acquired rule A, 39 reversed, 38 returned, 39 kept the stable
pair and 40 returned to routine; the median reversal lags were 22.5, 29, 20.5 and 22
trials at exposures of 300, 1,000, 3,000 and 10,000. Three lives missed a reading: one at
exposure 1,000 never approached the moved sugar and ended both later rules avoiding the
reversal pair; one at exposure 1,000 reversed and returned and ended the last rule at
0.87; one at exposure 10,000 ended every rule at 1.00 and had the stable pair right at
79% of the probes of a slow return.

### Rule A: optimal share of the last 100 actions, mean (minimum)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 0.75 (0.67) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `step` | 0.75 (0.67) | 0.89 (0.81) | 0.96 (0.77) | 0.98 (0.96) | 0.99 (0.97) |
| `defaults` | 0.49 (0.42) | 0.50 (0.43) | 0.50 (0.44) | 0.48 (0.42) | 0.50 (0.41) |
| `memory-only` | 0.73 (0.65) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `graph-only` | 0.47 (0.39) | 0.64 (0.45) | 0.64 (0.45) | 0.73 (0.46) | 0.74 (0.44) |
| `frozen` | 0.75 (0.67) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `replay` | 0.75 (0.67) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `reset` | 0.75 (0.67) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `tabular` | 0.69 (0.52) | 0.90 (0.71) | 0.95 (0.92) | 0.95 (0.90) | 0.94 (0.89) |
| `random` | 0.48 (0.39) | 0.53 (0.48) | 0.50 (0.42) | 0.51 (0.41) | 0.47 (0.38) |

### Rule B: optimal share of the last 100 actions, mean (minimum)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 1.00 (1.00) | 1.00 (1.00) | 0.97 (0.73) | 1.00 (1.00) | 1.00 (1.00) |
| `step` | 0.94 (0.71) | 0.94 (0.76) | 0.91 (0.75) | 0.84 (0.67) | 0.86 (0.43) |
| `defaults` | 0.47 (0.39) | 0.49 (0.41) | 0.50 (0.42) | 0.52 (0.45) | 0.49 (0.43) |
| `memory-only` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.98 (0.77) |
| `graph-only` | 0.50 (0.16) | 0.55 (0.40) | 0.58 (0.34) | 0.64 (0.46) | 0.65 (0.47) |
| `frozen` | 0.49 (0.43) | 0.49 (0.45) | 0.51 (0.43) | 0.52 (0.45) | 0.52 (0.42) |
| `replay` | 0.49 (0.43) | 0.49 (0.45) | 0.51 (0.43) | 0.52 (0.45) | 0.52 (0.42) |
| `reset` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `tabular` | 0.96 (0.91) | 0.94 (0.87) | 0.95 (0.93) | 0.95 (0.89) | 0.96 (0.91) |
| `random` | 0.52 (0.43) | 0.49 (0.42) | 0.48 (0.40) | 0.51 (0.38) | 0.46 (0.41) |

### Rule A again: optimal share of the last 100 actions, mean (minimum)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 0.98 (0.77) | 1.00 (1.00) | 0.97 (0.82) | 1.00 (1.00) | 1.00 (1.00) |
| `step` | 0.90 (0.62) | 0.93 (0.66) | 0.97 (0.89) | 0.93 (0.60) | 0.89 (0.57) |
| `defaults` | 0.48 (0.39) | 0.49 (0.41) | 0.51 (0.45) | 0.49 (0.43) | 0.51 (0.45) |
| `memory-only` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.96 (0.64) | 0.97 (0.74) |
| `graph-only` | 0.64 (0.39) | 0.56 (0.41) | 0.56 (0.44) | 0.63 (0.42) | 0.64 (0.45) |
| `frozen` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `replay` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `reset` | 1.00 (1.00) | 1.00 (0.98) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `tabular` | 0.96 (0.93) | 0.96 (0.95) | 0.96 (0.92) | 0.95 (0.93) | 0.95 (0.92) |
| `random` | 0.50 (0.40) | 0.48 (0.40) | 0.51 (0.41) | 0.47 (0.41) | 0.51 (0.44) |

### Reversal lag in trials, median (lives that reversed / lives)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 33 (10/10) | 22 (10/10) | 29 (10/10) | 20 (10/10) | 22 (10/10) |
| `step` | 58 (10/10) | 52 (10/10) | 155 (9/10) | 218 (7/10) | 139 (7/10) |
| `defaults` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `memory-only` | 31 (10/10) | 20 (10/10) | 19 (10/10) | 20 (10/10) | 23 (9/10) |
| `graph-only` | none (0/10) | 254 (1/10) | 415 (3/10) | 152 (2/10) | 355 (3/10) |
| `frozen` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `replay` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `reset` | 74 (10/10) | 77 (10/10) | 65 (10/10) | 53 (10/10) | 70 (10/10) |
| `tabular` | 60 (10/10) | 26 (10/10) | 46 (10/10) | 30 (10/10) | 68 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |

### Return lag in trials, median (lives that returned / lives)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 23 (9/10) | 17 (10/10) | 24 (9/10) | 19 (10/10) | 29 (10/10) |
| `step` | 158 (9/10) | 97 (10/10) | 118 (10/10) | 185 (9/10) | 148 (8/10) |
| `defaults` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `memory-only` | 22 (10/10) | 24 (10/10) | 24 (10/10) | 15 (10/10) | 20 (10/10) |
| `graph-only` | 333 (3/10) | 254 (2/10) | 305 (4/10) | 510 (3/10) | 91 (3/10) |
| `frozen` | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) |
| `replay` | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) |
| `reset` | 76 (10/10) | 77 (10/10) | 72 (10/10) | 66 (10/10) | 60 (10/10) |
| `tabular` | 71 (10/10) | 53 (10/10) | 89 (10/10) | 50 (10/10) | 76 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |

### Greedy choices after the reversal: first of two consecutive probes with every odour right, median trial (lives / lives)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 25 (10/10) | 37 (10/10) | 25 (9/10) | 25 (10/10) | 25 (10/10) |
| `step` | 25 (10/10) | 50 (10/10) | 187 (8/10) | 150 (4/10) | 237 (6/10) |
| `defaults` | none (0/10) | 525 (1/10) | 125 (1/10) | none (0/10) | none (0/10) |
| `memory-only` | 50 (10/10) | 25 (10/10) | 25 (10/10) | 25 (10/10) | 25 (9/10) |
| `graph-only` | none (0/10) | none (0/10) | 400 (1/10) | 175 (1/10) | 350 (1/10) |
| `frozen` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `replay` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `reset` | 25 (10/10) | 25 (10/10) | 25 (10/10) | 25 (10/10) | 25 (10/10) |
| `tabular` | 112 (10/10) | 50 (10/10) | 87 (10/10) | 50 (10/10) | 100 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |

### First executed approach at the new sugar odour after the reversal, median trial (lives / lives)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 23 (10/10) | 18 (10/10) | 20 (9/10) | 24 (10/10) | 20 (10/10) |
| `step` | 8 (10/10) | 37 (10/10) | 85 (9/10) | 185 (6/10) | 57 (7/10) |
| `defaults` | 11 (9/10) | 2 (7/10) | 10 (7/10) | 185 (8/10) | 2 (8/10) |
| `memory-only` | 28 (10/10) | 17 (10/10) | 18 (10/10) | 21 (10/10) | 21 (9/10) |
| `graph-only` | 21 (5/10) | 9 (6/10) | 6 (6/10) | 23 (6/10) | 17 (6/10) |
| `frozen` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `replay` | 69 (1/10) | 409 (1/10) | none (0/10) | 286 (1/10) | none (0/10) |
| `reset` | 2 (10/10) | 9 (10/10) | 7 (10/10) | 7 (10/10) | 3 (10/10) |
| `tabular` | 70 (10/10) | 38 (10/10) | 65 (10/10) | 38 (10/10) | 81 (10/10) |
| `random` | 6 (10/10) | 3 (10/10) | 1 (10/10) | 12 (10/10) | 3 (10/10) |

### Policy's approach probability at the new sugar odour when the rule turns, median (minimum)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 0.268 (0.174) | 0.251 (0.131) | 0.266 (0.167) | 0.238 (0.118) | 0.215 (0.078) |
| `step` | 0.270 (0.174) | 0.103 (0.043) | 0.023 (0.014) | 0.009 (0.008) | 0.007 (0.006) |
| `defaults` | 0.012 (0.005) | 0.008 (0.004) | 0.162 (0.004) | 0.005 (0.004) | 0.501 (0.004) |
| `memory-only` | 0.383 (0.326) | 0.378 (0.322) | 0.384 (0.328) | 0.381 (0.337) | 0.373 (0.327) |
| `graph-only` | 0.402 (0.241) | 0.376 (0.245) | 0.385 (0.260) | 0.328 (0.241) | 0.298 (0.177) |
| `frozen` | 0.268 (0.174) | 0.251 (0.131) | 0.266 (0.167) | 0.238 (0.118) | 0.215 (0.078) |
| `replay` | 0.268 (0.174) | 0.251 (0.131) | 0.266 (0.167) | 0.238 (0.118) | 0.215 (0.078) |
| `reset` | 0.505 (0.394) | 0.511 (0.393) | 0.509 (0.383) | 0.516 (0.383) | 0.510 (0.356) |
| `tabular` | 0.050 (0.050) | 0.050 (0.050) | 0.050 (0.050) | 0.050 (0.050) | 0.050 (0.050) |
| `random` | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) | 0.500 (0.500) |

### Approaches executed there until the greedy choice turned, median (lives whose choice turned / lives)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 1 (10/10) | 1 (10/10) | 1 (9/10) | 1 (10/10) | 1 (10/10) |
| `step` | 1 (10/10) | 1 (10/10) | 1 (9/10) | 1 (5/10) | 1 (7/10) |
| `defaults` | 0 (6/10) | 0 (4/10) | 1 (5/10) | 0 (4/10) | 0 (6/10) |
| `memory-only` | 1 (10/10) | 1 (10/10) | 1 (10/10) | 1 (10/10) | 1 (9/10) |
| `graph-only` | 6 (4/10) | 3 (5/10) | 7 (6/10) | 3 (6/10) | 4 (5/10) |
| `frozen` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `replay` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `reset` | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) |
| `tabular` | 1 (10/10) | 1 (10/10) | 1 (10/10) | 1 (10/10) | 1 (10/10) |
| `random` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |

### Lives whose greedy choice there never turned: approaches executed there under rule B, median (lives / lives)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | none (0/10) | none (0/10) | 0 (1/10) | none (0/10) | none (0/10) |
| `step` | none (0/10) | none (0/10) | 0 (1/10) | 0 (5/10) | 0 (3/10) |
| `defaults` | 1 (4/10) | 0 (6/10) | 0 (5/10) | 1 (6/10) | 0 (4/10) |
| `memory-only` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | 0 (1/10) |
| `graph-only` | 0 (6/10) | 0 (5/10) | 0 (4/10) | 0 (4/10) | 0 (5/10) |
| `frozen` | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) |
| `replay` | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) | 0 (10/10) |
| `reset` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `tabular` | none (0/10) | none (0/10) | none (0/10) | none (0/10) | none (0/10) |
| `random` | 70 (10/10) | 76 (10/10) | 73 (10/10) | 73 (10/10) | 70 (10/10) |

### Stable pair right at the probes, mean (minimum)

| Arm | 100 | 300 | 1,000 | 3,000 | 10,000 |
| --- | --- | --- | --- | --- | --- |
| `live` | 1.00 (0.92) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.99 (0.79) |
| `step` | 0.97 (0.71) | 1.00 (0.88) | 0.98 (0.67) | 0.96 (0.04) | 0.93 (0.00) |
| `defaults` | 0.00 (0.00) | 0.04 (0.00) | 0.05 (0.00) | 0.02 (0.00) | 0.01 (0.00) |
| `memory-only` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `graph-only` | 0.26 (0.00) | 0.21 (0.00) | 0.30 (0.00) | 0.36 (0.00) | 0.45 (0.00) |
| `frozen` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `replay` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| `reset` | 0.97 (0.96) | 0.98 (0.96) | 0.98 (0.96) | 0.98 (0.96) | 0.98 (0.96) |
| `tabular` | 0.97 (0.62) | 0.94 (0.00) | 0.99 (0.75) | 1.00 (0.92) | 1.00 (0.98) |
| `random` | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) |

### The live arm: witnesses, arousal and work, medians over lives

| Exposure | first approach at the new sugar odour | from it to the turned greedy choice | aroused, whole life | aroused, second half of rule A | sweeps per routine moment | sweeps per aroused moment | learning sweeps per aroused moment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 23 (10/10) | 0 (10/10) | 0.111 | 1.000 | 32.0 | 22.2 | 11.7 |
| 300 | 18 (10/10) | 0 (10/10) | 0.097 | 0.000 | 32.0 | 22.9 | 11.5 |
| 1,000 | 20 (9/10) | 0 (9/10) | 0.069 | 0.000 | 32.0 | 22.1 | 11.6 |
| 3,000 | 24 (10/10) | 0 (10/10) | 0.035 | 0.000 | 32.0 | 22.1 | 11.5 |
| 10,000 | 20 (10/10) | 0 (10/10) | 0.016 | 0.001 | 32.0 | 21.4 | 11.7 |

### Gates

```json
{
 "100": {
  "acquired": 0.0,
  "calm": 0.0,
  "crashed": 0,
  "lives": 10,
  "returned": 0.9,
  "reversed": 1.0,
  "stable_kept": 0.0
 },
 "1000": {
  "acquired": 1.0,
  "calm": 1.0,
  "crashed": 0,
  "lives": 10,
  "returned": 0.8,
  "reversed": 0.9,
  "stable_kept": 1.0
 },
 "10000": {
  "acquired": 1.0,
  "calm": 1.0,
  "crashed": 0,
  "lives": 10,
  "returned": 1.0,
  "reversed": 1.0,
  "stable_kept": 0.9
 },
 "300": {
  "acquired": 1.0,
  "calm": 1.0,
  "crashed": 0,
  "lives": 10,
  "returned": 1.0,
  "reversed": 1.0,
  "stable_kept": 1.0
 },
 "3000": {
  "acquired": 1.0,
  "calm": 1.0,
  "crashed": 0,
  "lives": 10,
  "returned": 1.0,
  "reversed": 1.0,
  "stable_kept": 1.0
 },
 "passed": true,
 "pooled": {
  "acquired": 1.0,
  "calm": 1.0,
  "crashed": 0,
  "lives": 40,
  "returned": 0.95,
  "reversed": 0.975,
  "stable_kept": 0.975
 },
 "reversal_lag": {
  "1000": 29.0,
  "10000": 22.0,
  "300": 22.5,
  "3000": 20.5
 }
}
```

### Reading the tables

**The historical failure and its cause.** The `step` arm reproduces what the reports
found. The longer it lives rule A, the lower its policy's probability of approaching the
odour that will carry the sugar when the rule turns (0.27, 0.10, 0.023, 0.009 and 0.007
across the exposures), the later its first approach there, and the fewer of its lives
end rule B with 90% of their last 100 actions optimal (9, 9, 7, 4 and 6 of 10). Of the
nine `step` lives whose greedy choice never turned, eight had not approached the new
sugar odour once and one had approached it once. In the 41 lives whose choice turned, one
approach sufficed in all but two, which needed two and four. The failure is too few
contradicting witnesses; at this operating point a witnessed outcome is taken.

A lag is the first window of 40 executed actions with 36 optimal. A brain that avoids
both odours of the reversal pair is right in three trials of four and can meet such a
window by chance, so for arms that end rule B below 0.9 the lag tables count more lives
than reversed. The last-100 shares and the greedy tables are the stricter readings.

**What arousal changes.** The `live` arm's approach probability at that odour stays
between 0.21 and 0.27 at every exposure, its first approach comes after about 20 trials
whatever the exposure, and the outcome of that first approach turns its greedy choice.
In 49 of its 50 lives the choice turned after exactly one approach; the other life never
approached. Its median reversal lag is 20 to 29 trials at exposures of 300 and more, and
in 47 of 50 lives its greedy choices are right for every odour at the probe of trial 25
or 50. At exposures of 300 and more it lives routine for 90% to 98% of its moments, and
the second half of rule A is aroused in 0.1% of moments or fewer.

**What carries the adaptation.** `memory-only` matches `live`: with the actor's rates at
zero the associative memory and the critic carry acquisition, reversal and return.
`graph-only` ends the rules between 0.47 and 0.74 and needs three to seven witnesses
where its choice turns at all: the graph's own reward learning does not acquire this task
in one stream at these budgets. `defaults` stays at chance.

**No adaptation without new evidence.** `frozen` and `replay` end rule B at chance for
the reversal pair (0.49 to 0.52 overall) and are right again at once when rule A returns.
`replay` does the work of a living brain on the old outcomes and changes nothing.

**Against starting over.** A newborn brain at every change (`reset`) reverses in 53 to 77
trials, inside its sampling youth, and has the stable pair right at 96% to 98% of the
probes, because it learns that pair again each time. `live` reverses sooner and keeps the
pair.

**Against the table.** At exposures of 300 and more the tabular learner ends the rules at
0.90 to 0.96, the price of exploring a tenth of its actions for life, and reverses with a
median lag of 26 to 68 trials, the wait for a random approach at the new sugar odour. It
needs no brain for four odours and it is the matched-information reference. `live` ends
the rules at 1.00 in the lives that pass and reverses in 20 to 29 trials: its exploration
is raised when reward is missing and absent otherwise.

**Work.** A routine moment settles once, 32 sweeps, and an aroused moment takes about 22
sweeps for its answer after about 11.5 for its feedback and eligibility. Routine saves
the eligibility phases, the parameter updates and the memory writes. It saves no settling
work: every moment pays one full qualified settle. Sweeps count numerical work; they are
neither wall time nor energy.

## Declared variants on the confirmation seeds

These runs change one thing each, at exposures of 300, 3,000 and 10,000, and are marked
`frozen_protocol: false`. They were declared before they were run and carry no gate; the
frozen gates are quoted where they help. Cells are the mean (minimum) share of optimal
actions in the last 100 trials of each rule, the median reversal lag with the lives that
have one, the stable pair at the probes, and the median share of aroused moments.
`--report` prints each receipt's full tables. At these three exposures 29 of the 30
confirmation lives of the `live` arm pass every reading of the gates.

### The world

A fifth of the outcomes withheld (`variant-unreliable-2026-10-05.json.gz`, `variant-unreliable-tabular-2026-10-05.json.gz`):

| Arm | Exposure | Rule A | Rule B | Rule A again | reversal lag | stable pair |
| --- | --- | --- | --- | --- | --- | --- |
| `live` | 300 | 1.00 (0.97) | 1.00 (1.00) | 1.00 (1.00) | 27 (10/10) | 0.98 (0.62) |
| `live` | 3,000 | 0.99 (0.92) | 0.96 (0.74) | 1.00 (0.97) | 41 (10/10) | 0.97 (0.50) |
| `live` | 10,000 | 1.00 (1.00) | 1.00 (0.99) | 1.00 (1.00) | 39 (10/10) | 0.99 (0.75) |
| `step` | 300 | 0.81 (0.74) | 0.90 (0.69) | 0.90 (0.79) | 68 (10/10) | 0.95 (0.75) |
| `step` | 3,000 | 0.92 (0.45) | 0.74 (0.47) | 0.92 (0.47) | 227 (5/10) | 0.80 (0.00) |
| `step` | 10,000 | 0.94 (0.57) | 0.65 (0.43) | 0.81 (0.43) | 175 (4/10) | 0.76 (0.00) |
| `tabular`, selected for this world | 300 | 0.90 (0.71) | 0.94 (0.87) | 0.96 (0.95) | 26 (10/10) | 0.93 (0.00) |
| `tabular`, selected for this world | 3,000 | 0.95 (0.90) | 0.95 (0.89) | 0.95 (0.93) | 30 (10/10) | 0.99 (0.92) |
| `tabular`, selected for this world | 10,000 | 0.94 (0.89) | 0.96 (0.91) | 0.95 (0.92) | 74 (10/10) | 1.00 (0.98) |
| `tabular`, protocol settings | 300 | 0.54 (0.48) | 0.56 (0.40) | 0.54 (0.45) | 353 (3/10) | 0.14 (0.00) |
| `tabular`, protocol settings | 3,000 | 0.56 (0.42) | 0.58 (0.47) | 0.58 (0.50) | 238 (1/10) | 0.16 (0.00) |
| `tabular`, protocol settings | 10,000 | 0.56 (0.48) | 0.58 (0.45) | 0.52 (0.42) | 185 (4/10) | 0.18 (0.08) |

Cost payoff with reward noise of 0.03 (`variant-cost-2026-10-05.json.gz`, `variant-cost-tabular-2026-10-05.json.gz`):

| Arm | Exposure | Rule A | Rule B | Rule A again | reversal lag | stable pair |
| --- | --- | --- | --- | --- | --- | --- |
| `live` | 300 | 1.00 (1.00) | 1.00 (0.99) | 0.99 (0.96) | 26 (10/10) | 1.00 (0.96) |
| `live` | 3,000 | 1.00 (1.00) | 1.00 (0.99) | 0.99 (0.93) | 34 (10/10) | 1.00 (1.00) |
| `live` | 10,000 | 1.00 (1.00) | 0.99 (0.91) | 0.98 (0.79) | 34 (10/10) | 0.98 (0.62) |
| `step` | 300 | 0.83 (0.74) | 0.92 (0.69) | 0.97 (0.79) | 55 (10/10) | 0.99 (0.71) |
| `step` | 3,000 | 0.99 (0.97) | 0.98 (0.94) | 1.00 (0.97) | 22 (10/10) | 1.00 (0.95) |
| `step` | 10,000 | 0.99 (0.97) | 0.90 (0.72) | 0.97 (0.85) | 32 (10/10) | 0.97 (0.78) |
| `tabular`, selected for this world | 300 | 0.97 (0.95) | 0.97 (0.94) | 0.96 (0.95) | 0 (10/10) | 1.00 (1.00) |
| `tabular`, selected for this world | 3,000 | 0.97 (0.93) | 0.98 (0.95) | 0.98 (0.96) | 1 (10/10) | 1.00 (1.00) |
| `tabular`, selected for this world | 10,000 | 0.98 (0.95) | 0.98 (0.95) | 0.97 (0.93) | 2 (10/10) | 1.00 (1.00) |
| `tabular`, protocol settings | 300 | 0.95 (0.92) | 0.94 (0.87) | 0.96 (0.95) | 1 (10/10) | 1.00 (1.00) |
| `tabular`, protocol settings | 3,000 | 0.95 (0.90) | 0.95 (0.89) | 0.95 (0.93) | 2 (10/10) | 1.00 (1.00) |
| `tabular`, protocol settings | 10,000 | 0.94 (0.89) | 0.96 (0.91) | 0.95 (0.92) | 11 (10/10) | 1.00 (1.00) |

With a fifth of the outcomes withheld, 28 of 30 `live` lives reverse and all return, and
24 keep the stable pair: a withheld reward at the stable sugar odour takes half of its
record, and the frozen gate on the stable pair is missed. The always-learning arm loses
the reversal with exposure (8, 4 and 1 of 10 lives end rule B at 0.9) and the stable pair
with it. The table is run twice: with the settings the protocol selected in the reliable
world it takes each withheld outcome at face value and stays near chance, and with the
settings the same rule selects in this world (alpha 0.5, epsilon 0.1) it ends the rules
at 0.90 to 0.96.

With the payoff as a cost, every wrong action is punished and carries its own evidence.
29 of 30 `live` lives pass every reading, with a median reversal lag of 26 to 34 trials.
The always-learning arm reverses about as fast here (22 to 55 trials) and ends rule B at
0.9 in 24 of 30 lives. The table with the settings selected for this world (alpha 1.0,
epsilon 0.05) reverses at once, a median lag of 0 to 2 trials: in this world it is the
fastest learner, and `live` ends the rules highest (0.98 to 1.00).

The two runs of the table with other settings used a copy of the protocol with those two
values; their receipts carry it and verify without `--current`.

### The arousal law

| Arousal law | Exposure | Rule A | Rule B | Rule A again | reversal lag | stable pair | aroused |
| --- | --- | --- | --- | --- | --- | --- | --- |
| founders | 300 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 22 (10/10) | 1.00 (1.00) | 0.097 |
| founders | 3,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 20 (10/10) | 1.00 (1.00) | 0.035 |
| founders | 10,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 22 (10/10) | 0.99 (0.79) | 0.016 |
| no want (`fast` at `slow`) | 300 | 1.00 (1.00) | 0.49 (0.45) | 1.00 (1.00) | none (0/10) | 1.00 (1.00) | 0.067 |
| no want (`fast` at `slow`) | 3,000 | 1.00 (1.00) | 0.52 (0.45) | 1.00 (1.00) | none (0/10) | 1.00 (1.00) | 0.024 |
| no want (`fast` at `slow`) | 10,000 | 1.00 (1.00) | 0.52 (0.42) | 1.00 (1.00) | none (0/10) | 1.00 (1.00) | 0.009 |
| no surprise (`tolerance` 20, `floor` 1) | 300 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 25 (10/10) | 1.00 (1.00) | 0.105 |
| no surprise (`tolerance` 20, `floor` 1) | 3,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 23 (10/10) | 1.00 (1.00) | 0.039 |
| no surprise (`tolerance` 20, `floor` 1) | 10,000 | 1.00 (1.00) | 0.99 (0.94) | 1.00 (1.00) | 23 (10/10) | 1.00 (1.00) | 0.016 |
| no heat (`heat` 0) | 300 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 22 (10/10) | 1.00 (1.00) | 0.100 |
| no heat (`heat` 0) | 3,000 | 1.00 (1.00) | 0.98 (0.80) | 0.98 (0.76) | 22 (10/10) | 1.00 (1.00) | 0.035 |
| no heat (`heat` 0) | 10,000 | 1.00 (1.00) | 0.95 (0.77) | 0.95 (0.74) | 21 (9/10) | 1.00 (0.96) | 0.018 |

Without the want no life reverses: the brain stays in routine through rule B, as the
frozen brain does. Without surprise all 30 lives pass every reading, so in this chamber
surprise adds nothing that can be measured. Without the heat 26 of 30 lives pass, against
29 of 30 with the founders; the four misses are at exposures of 3,000 and 10,000.

### The operating point and the size of the brain

| Operating point | Exposure | Rule A | Rule B | Rule A again | reversal lag | stable pair | aroused |
| --- | --- | --- | --- | --- | --- | --- | --- |
| declared | 300 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 22 (10/10) | 1.00 (1.00) | 0.097 |
| declared | 3,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 20 (10/10) | 1.00 (1.00) | 0.035 |
| declared | 10,000 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 22 (10/10) | 0.99 (0.79) | 0.016 |
| trace amplitude 3.0 | 300 | 0.74 (0.46) | 0.80 (0.52) | 0.81 (0.41) | 123 (6/10) | 0.59 (0.00) | 0.112 |
| trace amplitude 3.0 | 3,000 | 0.87 (0.49) | 0.82 (0.47) | 0.81 (0.48) | 80 (7/10) | 0.61 (0.00) | 0.054 |
| trace amplitude 3.0 | 10,000 | 0.84 (0.44) | 0.78 (0.43) | 0.67 (0.44) | 38 (6/10) | 0.56 (0.00) | 0.031 |
| consolidation 0.05 | 300 | 1.00 (1.00) | 0.97 (0.71) | 0.97 (0.69) | 27 (9/10) | 1.00 (1.00) | 0.099 |
| consolidation 0.05 | 3,000 | 1.00 (1.00) | 0.99 (0.93) | 0.97 (0.73) | 29 (10/10) | 1.00 (0.92) | 0.036 |
| consolidation 0.05 | 10,000 | 1.00 (0.99) | 0.96 (0.75) | 0.96 (0.77) | 53 (10/10) | 1.00 (0.92) | 0.017 |
| actor rates 1.0 and 0.05 | 300 | 0.78 (0.48) | 0.71 (0.40) | 0.72 (0.41) | 25 (5/10) | 0.47 (0.00) | 0.135 |
| actor rates 1.0 and 0.05 | 3,000 | 0.71 (0.43) | 0.57 (0.40) | 0.60 (0.42) | 87 (2/10) | 0.28 (0.00) | 0.064 |
| actor rates 1.0 and 0.05 | 10,000 | 0.67 (0.46) | 0.62 (0.49) | 0.60 (0.45) | 17 (2/10) | 0.28 (0.00) | 0.018 |
| all three released | 300 | 0.49 (0.41) | 0.49 (0.41) | 0.48 (0.41) | none (0/10) | 0.00 (0.00) | 0.067 |
| all three released | 3,000 | 0.57 (0.43) | 0.60 (0.46) | 0.52 (0.42) | 268 (2/10) | 0.12 (0.00) | 0.024 |
| all three released | 10,000 | 0.54 (0.41) | 0.55 (0.43) | 0.50 (0.44) | 80 (1/10) | 0.08 (0.00) | 0.009 |
| one module of 64 | 300 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 23 (10/10) | 1.00 (1.00) | 0.098 |
| one module of 64 | 3,000 | 1.00 (1.00) | 1.00 (1.00) | 0.96 (0.64) | 20 (10/10) | 1.00 (1.00) | 0.036 |
| one module of 64 | 10,000 | 1.00 (1.00) | 0.98 (0.77) | 0.95 (0.74) | 26 (9/10) | 1.00 (1.00) | 0.016 |
| two modules of 32 | 300 | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 23 (10/10) | 1.00 (1.00) | 0.100 |
| two modules of 32 | 3,000 | 1.00 (1.00) | 1.00 (0.98) | 1.00 (1.00) | 21 (10/10) | 1.00 (0.96) | 0.035 |
| two modules of 32 | 10,000 | 1.00 (1.00) | 0.97 (0.77) | 0.97 (0.74) | 19 (9/10) | 1.00 (1.00) | 0.014 |

The working trace and the actor rate are required: with either at its released value
most lives fail, and with all three released the life stays at chance, arousal included.
The released consolidation works less reliably (25 of 30 lives pass every reading). The
result does not depend on the 32-neuron module: one module of 64 neurons passes every
reading in 27 of 30 lives and two modules of 32 in 29 of 30.

## The first freeze

The chamber was first frozen with confirmation seeds 100 to 109 (protocol SHA-256
`381f23ca76287e0b83eeea99d0d5f5a3ae1adba2fce2be2eed0c814706ab1f8f`) and run once on
2026-10-05. Every gate passed: all 40 gated lives of the `live` arm acquired, reversed,
returned, kept the stable pair and returned to routine, with median reversal lags of
22.5, 31.5, 25.5 and 24 trials. Its receipt is
`results/first-freeze-confirmation-2026-10-05.json.gz`; it records the protocol hash and
the library version and predates the source manifest.

A review of the code after that run found that the actor's eligibility traces did not
fade while a brain lived in routine, so the first outcome learned after waking credited
actions sampled before the calm. The library was corrected (`ActorCritic.fade`), the
confirmation seeds of the first freeze were declared spent, and the protocol was frozen
again with fresh seeds, the same world, operating point, arousal founders and gates. The
second freeze also selects the tabular learner's two settings on the development seeds
(the first used alpha 0.2 and epsilon 0.1 unselected), gives a reset arm's newborn brain
a seed that meets no other generator of its life, adds the coverage and witness readings,
and binds its receipts to the source. The results above are the second freeze's.

## How the operating point and the founders were selected

Development used seeds 0 to 23, and seeds 400 to 447 at the longest exposure; the
confirmation seeds were first run on the frozen protocol.

- **The released defaults lock one stream.** With the composed actor rate of 1.0, selected
  on batches of streams, the policy of one stream saturates on one action for every odour
  within about a hundred trials. A tenth of the rate leaves the choice to the evidence.
- **The default working trace outweighs the present input of a continuing life.** With
  amplitude 3.0 into the scale-12 prefrontal projection, 1% of the variance of the
  association state follows the present odour and 1% the previous one; the state follows
  its own history. At amplitude 0.3, 76% follows the present odour and 16% the previous
  one. Amplitude 0 and 0.3 gave the same nursery results.
- **Lasting memory took too little of a witnessed outcome.** At the default consolidation
  of 0.05 a unit outcome moves the lasting record by a tenth while the transient copy
  fades by a tenth with every other record. After 10,000 trials of rule A a life sampled
  the new sugar odour, was paid, and returned to avoiding it before the record had turned:
  a failure to revise after sufficient witnesses. At 0.25 the lasting record takes half of
  a unit outcome. Consolidation 0.5 and 1.0 gave the same results in the reliable world;
  with a fifth of the outcomes withheld, 0.5 lost the stable pair on single withheld
  rewards and 0.25 kept it in all but one development life.
- **Whose outcomes enter the mood.** Letting every outcome enter it made the cost of
  exploring look like a shortfall and kept the brain awake: with the payoff as a cost
  (nothing for the right action, -1 for the wrong one, `--payoff cost`) the median
  reversal lag was 73 to 89 trials and 10% to 16% of the moments after a change were
  aroused. With only the outcomes of the brain's own greedy choices it was 17 to 25
  trials and 3% to 4%. In the sugar payoff the two rules did not differ.
- **The unit of reward.** A running RMS reward shrinks through a long calm at little
  reward and then makes small fluctuations look large. The unit is the spread of the
  outcomes the brain has learned from, which routine outcomes leave alone; the law is
  then unchanged by the scale and the zero of reward.
- **Arousal founders.** Removing the want left lives stuck after the reversal. With the
  hand-set heat of 1, 5 of 84 development lives at exposures of 300 and more ended a rule
  below the gate, having stopped searching before the moved sugar was found; with heat 2,
  and separately with a long-run rate of 0.002, none of 96 did before the eligibility
  correction. Heat 2 was kept as the founder: it spent the smaller share of moments
  aroused. Heat 3, a threshold of 0.3 and a faster recent rate were no better.
- **After the eligibility correction** the founders were left as they were. On the
  development seeds 94 of 96 lives at exposures of 300 and more then passed every reading,
  both misses at exposure 10,000. On the 48 diagnostic seeds at exposure 10,000, 45 lives
  ended every rule at 0.9 with the correction and 46 without it: the correction did not
  change the rate of misses.
- **The lag gate.** The gate on the reversal lag was first a ratio between the longest and
  the shortest exposure. A ratio of two medians of about 20 trials moves with single
  lives, and before the first confirmation it was replaced by the bound of 150 trials at
  each exposure.
- **The tabular learner.** Alpha in 0.1, 0.2, 0.5 and 1.0 and epsilon in 0.02, 0.05, 0.1,
  0.15, 0.2 and 0.3 were tried on the development seeds at exposures of 300 and more. The
  largest share of lives ending every rule at or above 0.9 was 94%, at epsilon 0.1 with
  alpha 0.5 or 1.0; alpha 1.0 had the lower median reversal lag, 38 trials against 52.

## What this does and does not establish

The `live` arm answers the two questions of issue 88 for this chamber. The choice that
must change is sampled again because a lasting shortfall of reward rouses the brain and
widens its sampling, and one witnessed outcome turns the choice because the lasting
record takes half of it. The always-learning arm shows the historical failure on the same
sequences, and its cause.

The repair is incomplete. Three of the 40 confirmation lives missed a reading, two at
exposure 1,000 and one at 10,000; on the development and diagnostic seeds 5 of 72 lives
at exposure 10,000 and none of 72 at exposures of 300 to 3,000 did. The founders' margin
is thin: in the confirmation life that never reversed, the arousal stayed under its
threshold for about 60 trials while the old sugar odour punished nine approaches; the
outcome that then woke the brain was recorded, the brain came to avoid that odour, the
shortfall that remained was within the threshold, and it never searched for the moved
sugar. The want
fades as the brain grows used to the poorer life, and a brain that avoids both odours of
the reversal pair is paid what it forecasts and stays calm. The dependence on exposure
that the issue describes is reduced and is still present.

The associative memory carries the adaptation. The odours are one-hot and the memory is a
direct record from sensory keys to action values, so this chamber does not test the
reciprocal graph's learned relations, generalization to unseen inputs, delayed outcomes
or short-term recall; those remain with issues
[110](https://github.com/muellerberndt/cadence/issues/110),
[111](https://github.com/muellerberndt/cadence/issues/111),
[84](https://github.com/muellerberndt/cadence/issues/84) and
[121](https://github.com/muellerberndt/cadence/issues/121). The declared operating point
is required: with arousal on the released composition the life stays at chance.

In this chamber the want does the rousing. The forecast is the critic's value of the
situation, and the next odour is random, so the usual error of a correct forecast is about
half a reward; a contradicted approach exceeds twice that by little and adds little
surprise. With the want removed no life reverses, and with surprise removed every life
passes. A forecast specific to the chosen action is not used.

The arousal law responds to change. In a world that does not change, noise in the reward
rate still rouses the founder genes for a small share of moments (the table reports it),
and a bout of needless exploration can lower the last 100 trials of a rule below the
gate. A brain whose life has always paid poorly, and whose youth has ended, is not roused.
The arousal level is a scalar computed from the brain's own temporal-difference error and
reward, as its dopamine is; it is a hand-set law whose constants are genes, and a
settling arousal patch inside the graph is not attempted here.

Routine is not cheaper in settling work. Reuse of a settled state across moments and
repair localized to declared dependencies, the remaining parts of issue 122, are not
attempted.

Sweeps count numerical work. They are not wall time or energy.
