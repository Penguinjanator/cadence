# Prospective robot training confirmation

This fixes a fresh cohort before running it, on the repaired source `750c8ef`.
The complete [protocol](protocol.json) names every body, founder, training world,
evaluation world, horizon and practical acceptance gate. It retains the existing
arena founder recipe without tuning. The previous mixed [development results](../arena_policy/README.md)
remain evidence: some early learned robots stood still, and the combined repair
underperformed the preceding source in the retained long-run and acquired-life
comparisons.

Nine continuing lives cover wheeled spike (Tumbler), four-legged hammer (Mantis),
and six-legged spinner (Hexapod) bodies. Each trains for exactly 80,000 nursery
moments. No checkpoint selection or interim behavioral probes are allowed.
Private frozen copies then act for 2,000 moments in each of two fresh worlds.
The controls are same-founder frozen newborns and same-body uniform random.
The target ablation zeros only the four target bearing/proximity channels;
other senses remain. All nine attempts count, including refusals and incomplete
attempts. The retained receipts are not replaced by retries.

For each life, average each metric across the two evaluation worlds. Its progress
gain is trained progress minus the larger of newborn and random progress; damage
gain is defined the same way. A positive gain must exceed `1e-9` in the metric's
units. At least eight of nine lives must have positive progress gain and at least
seven must have positive damage gain. A damage success also requires at least
30 hit points dealt in total across both evaluations (half a dummy's health),
so an incidental tiny collision does not pass. Each body must have at least
two of its three lives succeed at progress and at least two succeed at damage.

The sensory fraction is the sum over nine lives of intact minus ablated progress,
divided by the sum of progress gains. The denominator must exceed `1e-9`, or the
gate fails. The fraction must reach 0.5. No near-zero denominator is replaced or
clipped. Zero refusals, the complete cohort, source custody and exact saved
continuation are additional gates. These are fixed practical demo requirements,
not a statistical confidence bound on arbitrary future robots or combat wins.

The nursery's final real reward remains owed to its last executed action. The
receipt stores it, the next observation, every robot and arena state, both world
random generators, the dummy relocation counter, and the wrapper state alongside
the brain checkpoint. There is no synthetic reward, terminal flag or extra
unexecuted action. Before evaluation, a separate 32-moment fork compares the
unbroken brain with its loaded checkpoint in identically restored worlds. The
first call delivers the actual owed reward, once. Actions, rewards, complete
physics snapshots and resulting brain checkpoint arrays must match exactly.
These fork actions and all diagnostic work are charged separately; the training
checkpoint used for evaluations is unchanged.

Numerical work records unrounded answer/forecast sweeps and learning sweeps for
every training, continuation and frozen call. An observational wrapper counts
the returned sweeps from each actual `NeuralGraph.settle_batch` call, including
failed qualification and automatic retries, without changing its arithmetic.
Refusal attempts fail even when the arena wrapper's automatic retry succeeds;
the receipt retains completed partial totals and the refused call's work.
The eight-situation final probe
reports its own sweeps and elapsed time; there are no uncounted interim probes.
Wall-clock sections include policy calls and physics, and the outer receipt times
the full attempt. Sweeps are numerical work, not an energy or memory-I/O ledger.
Policy-dependent dummy destruction can change relocation times despite shared
initial world seeds. Frozen probes use private copies, receive no feedback, and
do not change the trained checkpoint.

Run each case in a fresh process, with at most three workers and one BLAS thread:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /path/to/arena/.venv/bin/python \
  benchmarks/arena_confirmation/confirm.py \
  --cadence /path/to/fixed-cadence --arena /path/to/cadence-robot-arena \
  --output /path/to/ignored-attempts --case tumbler-101
```

`--smoke` runs a separate declared non-cohort founder for a short lifecycle check;
its result cannot enter the confirmation census. The producer refuses to overwrite
an existing case directory and binds the exact source files, protocol, producer,
command, and checkpoint artifacts by SHA-256. [sources.json](sources.json) admits
only the fixed 55 Cadence and 15 arena Python source files before any training.
Large checkpoints and world
snapshots stay outside the repository.

Run `summarize.py /path/to/ignored-attempts` to check the complete census and all
gates. It checks required artifacts, their hashes, actual continuation records
and checkpoint arrays, source admission, horizons and the original metric math.
Missing and invalid cases fail. `--smoke --refusal-smoke` restricts only a separate
smoke brain's settling budget to cause a real qualification refusal and exercise
the failure receipt; it cannot enter the confirmation cohort.

The retained [preflight receipts](preflight/manifest.json) cover native-nursery
parity and exact 32-step continuation for all three bodies on non-cohort founders.
Every successful call's counted sweeps equal its answer plus learning sweeps.
A real one-step qualification refusal retains one attempted sweep and its partial
ledger, and a different runtime is rejected before training. The gate collator
rejects missing cohort cases and exits unsuccessfully when its gate is false.
