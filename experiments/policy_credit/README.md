# Delayed action credit

This small experiment tests whether reward can teach an action taken before a
delay. Only the first action affects the reward; subsequent observations are
blank and subsequent sampled actions are irrelevant. No teacher, replay,
memory address or reward-to-label conversion is provided.

Every arm receives 200 episodes in batches of 32, with five seeds (0–4), at
episode lengths 3 and 9. A 4-input, 8-hidden, 2-output Cadence circuit settles
all its moving neurons together. Its actor holds 72 parameters including the
unused critic. The conventional control is a 10-parameter linear softmax on
the same observed channels, with an exact policy-score eligibility trace.
Both trace rules decay by 0.95 per transition. The critic is disabled to
isolate delayed eligibility; this is not a test of a learned value model.

The actor correction makes its nudge differentiate the categorical policy,
even when the learner's imitation loss is quadratic. The legacy control uses
that quadratic imitation nudge for reward, reproducing the previous behavior.
All three Cadence arms use the same initial weights, rates and phase budgets.

Mean exact stochastic accuracy across five seeds:

| Rule | Length 3 | Length 9 |
|---|---:|---:|
| Corrected actor, eligibility trace | 0.8792 | 0.8047 |
| Corrected actor, no trace | 0.4736 | 0.4759 |
| Previous quadratic reward nudge | 0.5013 | 0.5001 |
| Conventional softmax score trace | 0.9834 | 0.9778 |

The corrected actor fails one seed at each delay. Mean greedy accuracy is
0.90; the conventional control reaches 1.00. The correction makes a substantial
difference here, but establishes neither general temporal credit nor an
advantage over conventional reinforcement learning. The parameters/rates of
the tabular control differ from Cadence; this comparison is a functional
control, not a matched-network gradient comparison. Numerical derivative tests
in `tests/test_policy_credit.py` separately check the local score identity.

Run from the repository root:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/policy_credit/run.py --output /tmp/cadence-credit-new/receipt.json
PYTHONPATH=src python experiments/policy_credit/verify.py
```

The committed receipt includes every scheduled seed, training curve, both
context probabilities and transition count. Its source archive binds the
complete executable core and runner used. CPU timings include library dispatch
and may share the host with other audits; they are not used for a speed claim.
No joules were measured. The failed seed-91 launch check (20 episodes) stopped
at a keyword mismatch in the legacy adapter before its completion; it is not
part of these five-seed results.
