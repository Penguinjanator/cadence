# Direct sensorimotor projections under delayed reward

This fixed wiring comparison tests a bounded circuit with local neuron state, sensory
input ports, two action outputs, eligibility readback and reward-modulated synaptic
updates. It is a capacity/wiring experiment, not evidence that a learning rule is better
than conventional policy gradients.

The schedule is frozen in `run.py`: lengths 3, 9 and 33; seeds 0–4; 200 episodes in
batches of 32; six arms and 90 completed rows. Only the first action affects reward;
intermediate observations are blank and intermediate sampled actions are irrelevant.
The exact integrated `policy_credit/run.py` training function supplies all interactions.

The three compared brains use the original 4-input, 8-hidden, 2-output topology, the
same topology with eight random `layered(skip=True)` input-to-output projections, and
the same skip topology with those eight efficacies initialized to zero. Existing
effective weights must match exactly in both skip arms; the zero arm must preserve
initial context probabilities exactly. Both skip variants add eight trainable parameters.
All learning rates, traces, phase budgets, temperatures, reward, and data/actor seed
rules remain identical. The original no-trace, previous quadratic-nudge and conventional
softmax-score controls are rerun at every length. The original 40-row reference receipt
and its exact source archive are also retained within this experiment's source bundle.

No thresholds, rates, seeds, durations or topologies are changed after results. Report
both exact stochastic accuracy and the number of seeds correctly classifying both
fresh contexts. No efficiency, energy, naturalistic control or general temporal-credit
claim follows from this toy protocol.

Run from this repository, with the integrated corrected actor in a separate checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python benchmarks/sensorimotor_skip/run.py --integration ../cadence-drawbacks-audit \
  --output benchmarks/sensorimotor_skip/results/receipt.json
PYTHONPATH=src python benchmarks/sensorimotor_skip/verify.py \
  benchmarks/sensorimotor_skip/results/receipt.json
```

The runner recursively freezes the complete integrated core, the original protocol,
itself, and reference evidence in `sources.zip` before training. A subprocess imports
and executes only that extracted snapshot. A private globals dictionary binds the
topology factory to the unchanged training bytecode; it never changes the imported
protocol module. `--archive <sources.zip> --output <fresh-directory>/receipt.json`
replays the fixed snapshot without the integration checkout. Existing outputs are never
overwritten. All training curves, probability readings, budgets, initial-state checks,
reference comparisons and source digests remain in the final receipt.

The recorded run froze integration commit `c292bdbbd5cb8e59735fa4721f49fe145ad54b42`.
All 40 original reference rows reproduce exactly. The 90-row receipt and both embedded
source bundles pass digest and arithmetic verification. Mean stochastic accuracy, with
the number of seeds correctly choosing both fresh contexts in parentheses:

| Arm | Length 3 | Length 9 | Length 33 |
| --- | ---: | ---: | ---: |
| Original corrected trace, 72 parameters | .879238 (4/5) | .804658 (4/5) | .715188 (2/5) |
| Random skip, 80 parameters | .981200 (5/5) | .945258 (5/5) | .820010 (5/5) |
| Zero-initialized skip, 80 parameters | .982820 (5/5) | .976571 (5/5) | .859394 (5/5) |
| Conventional score trace, 10 parameters | .983418 (5/5) | .977799 (5/5) | .904008 (5/5) |
| Corrected actor without trace | .473605 (0/5) | .475858 (0/5) | .473736 (0/5) |
| Previous quadratic reward nudge | .501251 (0/5) | .500102 (0/5) | .499749 (0/5) |

For this simple cue-conditioned control task, the zero-initialized sensory-to-action
projection is a useful optional brain pattern: it preserves the starting predictions
and gives sensory evidence a direct trainable path to the action outputs. This result
does not distinguish that path from the effect of the eight extra parameters. It also
does not solve the longest-delay stochastic gap: zero skip remains below the conventional
control there, and some individual seeds remain weak. Random skip makes a few previously
strong seeds worse even while correcting their greedy decisions. The complete outcomes
remain in the receipt; no follow-up tuning was performed.
