# Sampled-income repair in saved robot lives

Development comparison for [#158](https://github.com/muellerberndt/cadence/issues/158),
2026-10-08. This checks the income repair on continuing acquired robot brains;
it does not establish improved fighting skill. The related exploration and greedy
policy limitations in #159/#160 remain tracked by #111.

The arena is [cadence-robot-arena](https://github.com/muellerberndt/cadence-robot-arena)
at `14eb800543923c9e1f64f28319490ea4435decb6`. Each run loads the same six evolved
checkpoints, preserving their weights, memory, trace, arousal state and pending
outcome. The declared `need=0` control isolates relative-income want; neither
arousal nor the brain is reset, and heat/temperature remain the saved genes.
The original positive-need condition can legitimately remain aroused when its
actual income does not meet its need.

Each arm runs three 600-moment royales (seeds 77, 78, 79; closing zone 500),
with no saving and no replay. Each run starts from the original acquired
checkpoints, rather than claiming a sequence of three continued fights.
The third arm uses uniform-random commands with the same six bodies. Cadence
control is `2566ccf2c23b76505a82b941111535ffd50f8ae4`; candidate is `b041dc7`,
with the same runtime change as `9c2ac1f`. The JSON retains source identities,
the arousal module hash, every checkpoint hash, per-fighter readings and work.
All checkpoint hashes are unchanged after each arm.

| Measured quantity | Released-source control | Income repair | Uniform random |
| --- | ---: | ---: | ---: |
| Total lived moments | 9,322 | 9,966 | 10,800 |
| Aroused share, weighted by lived moments | 93.4% | 66.2% | n/a |
| Moving closer to nearest rival, weighted share | 38.2% | 27.7% | 25.0% |
| Damage dealt, all fighters | 1,147.8 | 1,111.6 | 0.0 |
| Learning sweeps | 89,337 | 69,532 | 0 |
| Refusals | 0 | 0 | 0 |

Weighted shares use the arena's rounded per-fighter readings and are approximate.
Changed survival means denominators differ. Seeds 77 and 79 produce identical
behavioral readings in both brain arms; all three random runs also agree. These
are repeated development runs, not independent fresh-founder confirmation.

The repair reduces time spent aroused without a reset, but closing behavior
deteriorates. There is no demonstrated improvement in fighting tactics. Returning
these existing brains to greedy operation exposes their already documented weak
greedy policies; that is consistent with #160, but this comparison does not prove
the cause. No learning rule, temperature, heat, sensory gain or acceptance gate
is tuned to improve these results.

The [law regression](../../tests/test_arousal_income.py) isolates the actual bug:
after a reward drop, 1,500 non-greedy sampled outcomes leave the old income
reference frozen and the final 300 outcomes all aroused. The candidate follows
the new income and is calm on all final 300 outcomes, with own-error calibration
unchanged. [Paired nursery preservation](../reversal/README.md#income-tracking-repair--2026-10-08)
separately retains the original capability gates and records increased reversal
lags and learning cost.

## Reproduce

The harness is retained exactly as run. Copy `compare_arena.py` to
`<arena>/runs/cadence-issues-158-160-20261008/compare_arena.py`; it resolves that
arena checkout from its location. From the arena checkout, use its environment
and select the exact Cadence checkout with `PYTHONPATH`:

```sh
PYTHONPATH=/path/to/cadence/src:/path/to/cadence-robot-arena \
  .venv/bin/python runs/cadence-issues-158-160-20261008/compare_arena.py \
  --need-zero --output runs/issue158-comparison.json
```

Repeat with the control and candidate source checkouts; add `--policy random`
for the uniform-random arm. Output is local only. No league checkpoint is written.
The supplied files in `results/` retain both unfavorable and favorable readings.
