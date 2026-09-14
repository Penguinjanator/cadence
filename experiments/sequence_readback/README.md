# Causal sequence readback benchmark

The [sequence guide](../../docs/sequence.md) documents the reusable library API.

This package evaluates next-character prediction on small excerpts from three
disjoint public-domain books. It compares an embedded window patch, the existing
stateful patch with a raw hidden Echo, and the same stateful patch with a bounded,
centered hidden trace. Window and recurrent MLPs use the same input window,
embedding width, hidden width, stream boundaries, token exposure and epoch
budget. Their recurrent state is detached at each observation; no learner gets
credit through earlier observations. The bigram is an additional conventional
reference fitted to the full training excerpt, including boundary transitions.

Every patch also receives a causal sequence-cache evaluation using its frozen
free hidden features. The raw arm uses the cosine read from the earlier
`cadence-author` sequence notebook. The stale-coordinate ablation stores
normalized deviations while its running mean changes. The corrected cache
stores raw features and puts every stored key and query in the same current
coordinate system. The ablation uses the new cache's mean initialization; it is
not an exact reimplementation of the original author trainer. All three cache
arms use probability interpolation, so none is a full reproduction of the old
experiment's injection of positive current into the output neurons.

An independent scalar-stream cosine implementation receives exactly the same
keys, values, centering, capacity and temperature. It checks equality to the
library read at every validation and test step. It is a conventional control
with the same mathematical structure; its equality precludes an advantage
claim against matched attention.

The slow weights are frozen at evaluation. Each cache predicts before it sees
the actual next character, then stores that already observed association. The
first 16 characters of each evaluation stream form its unscored warm-up. The
cache is reset between training, validation and test. Neither book identity,
position label, target-derived key, nor a dictionary lookup enters the neural
features. The mean is learned from feature observations; the encoder is learned
by the patch's ordinary current-step local contrast. This does not establish
learned semantic addressing or language understanding.

## Reproduction

Install the package with its `dev` dependencies, including PyTorch, then run from
the repository root:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python experiments/sequence_readback/run.py --output /tmp/cadence-sequence-reproduction
PYTHONPATH=src python experiments/sequence_readback/verify.py \
  experiments/sequence_readback/runs/main/receipt.json
PYTHONPATH=src python experiments/sequence_readback/archive.py \
  experiments/sequence_readback/runs/main/receipt.json --verify
```

Use a fresh output directory for a reproduction. The current runner refuses to
overwrite a nonempty directory, preserving existing receipts and source copies.
The committed text files are the experiment inputs. `data/manifest.json` records
the Gutenberg IDs, original file hashes, normalization source hash and excerpt
hashes. `prepare.py` optionally regenerates those files from the sibling
`cadence-author/author/data/` cache. It requires that source checkout and is not
needed to run the experiment. The pilot used a disjoint earlier test excerpt;
the five-seed confirmation uses `test_confirmation.txt`.

The default confirmation schedule is fixed: seeds 0–4, 60,000 training
characters, 10,000 validation and test characters, four epochs, 64 training
streams, 16 evaluation streams, window 4, embedding 8, hidden 64. There is no
early stopping of the schedule. Best epoch, output temperature, cache
temperature and probability-mixture weight are selected on validation only.
All evaluated validation configurations are retained, including zero mixture.
The original seed-91 development pilot and the initial smoke check are also
retained. They used the previous runner revision and are historical diagnostics,
not confirmation receipts.

`runs/main/receipt.json` binds the experiment inputs, critical numerical sources,
configuration, every seed and every architecture to their hashes. Those critical source
bytes are copied under `runs/main/source/` before training. The full recursive
package, including `circuits/`, is supplied from immutable base `9f859bf` without
replacing any frozen file. `archive_manifest.json` binds that complete source
archive to the final receipt. New runs copy the full recursive package from the
start. The archived package can be imported by setting
`PYTHONPATH=experiments/sequence_readback/runs/main/source/src`. `seedN.npz` retains every test token's loss and correct
or incorrect decision for every neural/cache comparison. `verify.py` recomputes
all test aggregates and checks completeness, validation-only selection, artifact
hashes and independent cosine equality. It does not rerun training or claim
that a stored loss alone proves model correctness.

The implementation has no energy-in-joules measurement. Timings include each
training arm's validation passes and are ordinary CPU wall times. The benchmark
is a small controlled text experiment, not a replacement for the larger R31/R32
language evaluations.
