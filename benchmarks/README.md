# Reproducing runtime and fixed-input checks

These scripts compare the numerical equations of a bounded brain: local neuron state,
input/output populations, readback through the equation residual, local synaptic updates,
and receipts binding the observations to their implementation.

From the repository root with the Torch extra installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=src python benchmarks/runtime.py --device cpu --out benchmarks/receipts/runtime_cpu.json
PYTHONPATH=src python benchmarks/runtime.py --device mps --out benchmarks/receipts/runtime_mps.json
PYTHONPATH=src python benchmarks/ep_inputs.py --out benchmarks/receipts/ep_inputs.json
```

Run each timed experiment by itself. Both runtime arms use Torch on the same device and
precision; the control loads the original residual, activation and block/contrast functions
from the immutable Git revision named by the script. A frozen source bundle includes the
original baseline file and can run outside the Git clone. Setup and result inspection are outside the timer; GPU synchronization is inside
the timing boundary. Warmup samples are retained separately, every declared seed is kept,
and each receipt reports numerical deviations and host-reference residuals. A speed ratio
below one is a slower current arm and remains in the receipt. The update workload uses a
fixed number of centered updates with identical synthetic batches; it is not a learning
quality benchmark. Its last free state predates its last parameter update, so the separately
reported residual under the final parameters is a diagnostic, not a convergence assertion.

The EP script compares ordinary cross-entropy finite differences with contrasts converted
by the contact/gain factor divided by temperature. Autonomous source parameters are held
fixed, while their outgoing projection weights are differentiated. Every phase is checked
to the same equation tolerance. The script includes a control with asymmetric effective
weights between free neurons.

Both scripts pass their stored raw measurements to an arithmetic verifier when writing a
receipt. To recheck a stored receipt without repeating the timings:

```python
from pathlib import Path
from cadence import Receipt
from benchmarks.runtime import verify

path = Path("benchmarks/receipts/runtime_cpu.json")
receipt = Receipt.read(path)
sources = [(entry["path"], Path(entry["path"])) for entry in receipt.source["files"]]
print(Receipt.verify(path, sources=sources, check=verify))
```

Use `benchmarks.ep_inputs.verify` for the EP receipt. Each receipt binds a content-addressed
archive under `benchmarks/frozen/` containing the whole Cadence package, experiment scripts,
and original baseline. Later core edits do not invalidate the archived experiment. Extract
the archive into a new directory and run the same commands there to replay with those exact
sources; dependencies and device still need to match the recorded environment. Archive
names change when source bytes change, so an existing experiment bundle is never overwritten.
Timings are specific to the recorded hardware/software. No joules, CUDA performance,
comparison against a conventional learner, or historical trained-checkpoint certificate
is inferred from these receipts.

A separate [isolated CUDA benchmark](device_energy/README.md) measures the same
frozen runtime pair with long timed intervals and cumulative GPU energy counters.
It reports raw device energy and an adjacent-idle subtraction separately, including
the cold-residual regression under that subtraction. Its repetition boundary differs
from the single-unit workstation timings above.
