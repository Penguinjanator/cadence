# T1: the memory addressed the wrong moment

The installed POPGym environment scores `hand[-k]` **before** dealing the next
card. The current observation is `hand[-1]`, so the scored observation is
`k - 1` moments old. The original T1 ring read the observation `k` moments old.
Its chance result therefore cannot establish that a correctly addressed memory
fails to learn from reward.

An independent history oracle confirms the contract on 20 episodes per level:

| level | configured k | old lag: mean return | corrected lag | corrected return |
| --- | ---: | ---: | ---: | ---: |
| Easy | 4 | -0.5021 | 3 | 1.0000 |
| Medium | 32 | -0.5236 | 31 | 1.0000 |
| Hard | 64 | -0.5000 | 63 | 1.0000 |

The test uses the environment's actual rewards, independently of the patch
network. Its source file is hashed in both receipts. A future environment
change that alters this convention must change the adapter, not silently reuse
the delay.

## A bounded learning control

Three networks receive the same current observation, initialized weights,
allocated owner ranges, and reward learning rule. All have 599 parameters.
Their fast seam records are local state: an observed suit is written under a
clock address, read through a memory port, and the settled response is repaired
by the free/nudged contrast. The receipt records the readback and the reward.

Only the read condition differs: a period-three ring, the original period-four
ring, or a period-three ring whose memory output is masked. Four clock owners
are allocated in every arm; the corrected ring leaves the fourth unused.
The memory write and clock are designed. The readout policy learns from reward;
no correct-action labels enter its nudge.

The reward pays for the current answer, so this diagnosis uses `gamma = lam = 0`.
It does not test delayed reward credit. Other settings are recorded in the
receipt, including actor rate 0.02, bias rate 0.002, and critic rate 0.1.

After the five-episode pilot left all policies near chance, the fixed extended
schedule ran seeds 0, 1, 2 for 200 vector episodes of 32 environments, followed
by 100 fresh evaluation episodes per arm and seed. Every scheduled outcome is
retained in `receipt_200.json`; the earlier pilot is `receipt_pilot.json`.
The extended schedule used 326,400 environment steps per arm and seed and
119 seconds of total measured training/evaluation time on this machine.

| arm | mean evaluation return | standard deviation across seeds | memory matches scored target |
| --- | ---: | ---: | ---: |
| corrected delay | 0.08389 | 0.07730 | 1.0000 |
| original delay | -0.50042 | 0.00177 | 0.2317 |
| masked memory | -0.50000 | 0.00297 | 0.2491 |

The corrected policy improves, but remains far from the oracle's return of one.
Its returns by seed are 0.19167, 0.01417, and 0.04583. This supports an
observation-alignment correction and partial reward learning at the declared
budget. It demonstrates neither learned memory addresses, long-horizon credit,
nor superiority to a transformer. The original PPO window already contained
the correct observation, so the old patch/PPO comparison also had an
information-access mismatch.

## Reproduce

From the workspace root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  cadence-paper/.venv_suites/bin/python cadence-paper/experiments/t1_alignment/run.py \
  --episodes 200 --output cadence-paper/experiments/t1_alignment/receipt_200.json
cadence-paper/.venv_suites/bin/python cadence-paper/experiments/t1_alignment/run.py \
  --verify cadence-paper/experiments/t1_alignment/receipt_200.json
cadence-paper/.venv_suites/bin/python -m unittest discover \
  -s cadence-paper/experiments/t1_alignment -p 'test_*.py' -v
```

Both receipts were produced and verified against clean cadence revision
`af62201ee428022c3a31c3498ac75e474a313fad`. They hash every core Python module,
the experiment, the T1 adapter, and the installed environment implementation.
Later core edits can invalidate current-source comparison; the original
revision remains the reproduction target for these measurements.

The three regression tests check the environment oracle, actual fast-seam
readback, and episode lengths 51/103/155. The main T1 adapter now uses the
corrected delay, records actual episode lengths, accepts the current actor
configuration, and writes new `receipt_<level>_aligned.json` files by default.
The original Easy/Medium receipts remain immutable historical results; this
diagnosis does not reclassify their old implementation as successful.

## Verification with the refactored core

`receipt_refactor.json` reruns the exact 200-episode, three-seed schedule with
the refactored library. Every training-curve value and held-out result matches
`receipt_200.json` exactly; only source provenance and timing change. Mean
returns remain 0.08389 (corrected), -0.50042 (original), and -0.50000 (masked).
The nine runs take 115.93 seconds of measured training/evaluation time.

The schedule contains 2,937,600 training transitions, 45,900 learned-policy
evaluation transitions, and 12,360 transitions over 120 oracle episodes
(60 per lag condition), for 2,995,860 transitions in total. All 60 corrected
oracle episodes return one. The three regression tests pass against this core.

At that snapshot the receipt verified its canonical form, embedded digest, all 19
declared source files, and arithmetic. Subsequent S2 optimizations changed `learning.py`
and `settle.py`; current-source verification now intentionally rejects this historical
receipt. Its original library bytes are preserved under `../s2_efficiency/baseline/`.
Its source-manifest SHA-256 is
`5925eb89f68565bbca7c4bae781eb158be0e81507bb7b6210317ad3e3a6ae572`;
the receipt SHA-256 is
`f8100fa5868b0edd8c3a74e491d6f270d6ba1369cc8e5a45c1efd0474e17b6eb`.
The earlier two receipts retain their original source bindings.

The command below checks against the currently installed library; source differences
must be resolved by replaying the original snapshot or producing a new run, not by
re-signing the old metrics.

```sh
cadence-paper/.venv_suites/bin/python cadence-paper/experiments/t1_alignment/run.py \
  --verify cadence-paper/experiments/t1_alignment/receipt_refactor.json
```
