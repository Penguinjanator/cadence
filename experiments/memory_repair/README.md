# Content and continual-memory experiments

These scripts exercise the public `ContentMemory`, `ReservoirReplay`, and existing
`Learner`/`FastSynapses` APIs. They produce canonical receipts with code hashes, declared
comparison grids, seeds, every scheduled outcome, and recomputable metrics. They implement
bounded observer-like patches with input ports, local records, readback, and corrective
feedback; they do not establish a general solution to lifelong learning or learned
address-policy discovery.

`content.py` compares noisy content retrieval with delta fast synapses, separated delta
synapses, FIFO nearest exemplars, and fixed prototypes. There are no supplied record
addresses: the observed feature vector is the cue. Identical-cue aliases and overflow
are mandatory controls. `content_spec.json` is written before the comparison; the receipt
retains the entire 5-seed, 4-condition, 5-arm grid. The extra sparse expansion is counted
as fixed and mutable arrays rather than free capacity.

`continual.py` uses MNIST's original 60,000/10,000 train/test separation, average-pools
2x2 pixels to 196 inputs, and trains a 48-neuron hidden layer with one ten-class head.
It draws 2,000 training examples once per task and tests on 256 held-out examples per
task. Split mode presents pairs 0/1 through 8/9. Permuted mode presents ten pixel
permutations. Task identity is never passed to the model; there is no boundary action
or test-time head restriction. All arms see the identical training and test examples
for a given seed. Reading all test tasks after each stage measures retention only;
those measurements do not select hyperparameters or further updates.

There are six arms: Cadence and a same-shape ReLU MLP with Adam, each with no replay,
current-only repetition, or reservoir replay. One update follows each 32-example
current minibatch. Replay adds up to 32 previous examples from a 256-example reservoir;
the repetition control adds exactly the same number of current examples. Both therefore
have equal training-row and update budgets. No-replay uses fewer rows. Adam and Cadence
have different computation per update; wall time and Cadence settling steps are retained.
Model parameter counts and replay array bytes are reported separately. The replay store
contains labels: they are explicit information unavailable to the no-replay arms.

The MNIST file is downloaded from the TensorFlow Keras dataset mirror, kept under ignored
`data/`, and its bytes are SHA256-pinned in each continual receipt. It is not added to Git.

```sh
mkdir -p experiments/memory_repair/data
curl -L --fail https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz \
  -o experiments/memory_repair/data/mnist.npz
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHONPATH=src python experiments/memory_repair/content.py
PYTHONPATH=src python experiments/memory_repair/continual.py --mode split
PYTHONPATH=src python experiments/memory_repair/continual.py --mode permuted
PYTHONPATH=src python experiments/memory_repair/content.py --verify experiments/memory_repair/content.json
PYTHONPATH=src python experiments/memory_repair/continual.py --verify experiments/memory_repair/split.json
PYTHONPATH=src python experiments/memory_repair/continual.py --verify experiments/memory_repair/permuted.json
```

The seed-99 split pilot uses 1,000 examples per task and is retained separately. It was
used for execution checks, with no changes to the specified model hyperparameters.
The main seeds are 101 through 105. This is a smaller protocol than the historical
full-image, thirty-task permuted-MNIST experiment and must not replace its negative
result. The previous no-replay split-MNIST and long-horizon game drift likewise remain
separate evidence.

`frozen_sources.zip` contains the exact experiment scripts and complete Cadence Python
package used for these comparisons; `frozen_sources.json` binds the archive and every
member. `freeze.py` creates this archive once and refuses to overwrite it. After later
library changes, verify the historical receipts against these archived executable bytes:

```sh
PYTHONPATH=src python experiments/memory_repair/verify.py
```

To rerun that version, unpack the archive into a new directory, set `PYTHONPATH` to its
`src/`, and run its experiment script with `--data` pointing to the downloaded MNIST file.
Use the receipt's Python/NumPy/Torch versions when comparing exact numeric results. The
ordinary `--verify` commands above intentionally verify against the current working
tree and may report a source mismatch after integration; archived-source verification
preserves the original evidence rather than rebinding it to changed code.
