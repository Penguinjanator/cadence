# Witnessed neutral variation

The original minimal two-head run learns all eight answer pairs, but changing
the irrelevant neutral record changes answers. The direct single-head assay
also learns all 16 answers while depending on irrelevant context. Both failed
results remain prerequisites and retain their original criteria.

This next component run changes one training-data factor: the actual neutral
observation alternates between amplitudes 1 and .5 on successive rounds.
Every arm and founder sees the same amplitude schedule. Accepted observations
are witnessed normally; no training memory is edited to supply an answer.
The candidate, initialization, learning law, final-label teacher, numerical
qualification and case schedule remain unchanged.

Evaluation uses the original eight cases and all 13 original causal panels.
It retains the original donor, erasure, nuisance, distinct-head and retention
criteria. A positive result would show robustness within the amplitude family
seen during training. It would not establish general invariance, transfer,
learned meaning or a parallel-head advantage.

The runner requires fresh exposed and supplied-read capacity controls before
both learned H2 founders. Old capacity cannot admit changed training. The
source inventory includes both failed reader results and their complete cost
history. The finite stage budgets remain 600 seconds for capacity and 1800
seconds for H2, with actual work and complete parameter/optimizer snapshots.
No world comparison or cloud job opens automatically.

After source freeze:

```bash
python benchmarks/attention/neutral_variation_campaign.py --runtime . \
  --stage capacity --out /tmp/neutral-variation-capacity.json \
  --parameters /tmp/neutral-variation-capacity-parameters --commit <frozenSHA>
python benchmarks/attention/neutral_variation_campaign.py --runtime . \
  --verify /tmp/neutral-variation-capacity.json
```

Only a positive new capacity receipt admits H2:

```bash
python benchmarks/attention/neutral_variation_campaign.py --runtime . \
  --capacity /tmp/neutral-variation-capacity.json --stage h2 \
  --out /tmp/neutral-variation-h2.json \
  --parameters /tmp/neutral-variation-h2-parameters --commit <frozenSHA>
python benchmarks/attention/neutral_variation_campaign.py --runtime . \
  --capacity /tmp/neutral-variation-capacity.json \
  --verify /tmp/neutral-variation-h2.json
```

These are raw witnessed-memory component controls. Compressed learned-code
memory and acquired continuing behavior still require their own preservation
tests before an integrated attention comparison.
