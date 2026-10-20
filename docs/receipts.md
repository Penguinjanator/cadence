# Receipts

A receipt stores outcomes, an embedded digest and a manifest of selected source files.
The caller decides which outcomes to record and what the verifier checks.

## Write and verify a small receipt

Save this as `receipt_demo.py` and run `python receipt_demo.py`. It binds a small
arithmetic result to its own script and to the installed receipt implementation:

```python
from pathlib import Path
import cadence as cd

sources = [
    ("receipt_demo.py", Path(__file__)),
    ("cadence/receipts.py", Path(cd.__file__).with_name("receipts.py")),
]
values = [1, 2, 3]
receipt = cd.Receipt.build(
    "sum-demo/v1", {"values": values, "total": sum(values)}, sources=sources,
)
path = receipt.write(Path("sum_receipt.json"))

def check(body):
    if body["values"] != [1, 2, 3] or body["total"] != sum(body["values"]):
        return "stored sum differs from the declared inputs"
    return None

ok, message = cd.Receipt.verify(path, sources=sources, check=check)
assert ok, message
assert cd.Receipt.read(path).digest == receipt.digest
print(message)
```

Expected output: `canonical form, digest, sources, arithmetic agree`. For an experiment,
bind every consumed source and dataset, and supply checks for its schedule, outcomes
and metrics.

## Shape

```json
{
  "kind": "experiment/v1",
  "body": { "...": "recorded readings, rows, gain tables, custody" },
  "source": { "files": [{"path": "experiment.py", "sha256": "..."}], "manifest_sha256": "..." },
  "digest": "sha256 of the canonical JSON of kind, body and source"
}
```

`Receipt.build(kind, body, sources=())` hashes each `(relative, path)` pair in `sources`
(an empty tuple gives an empty manifest) and the canonical JSON of the payload
(`cd.canonical_json`: sorted keys, no whitespace). Mapping keys become strings, tuples become
lists, NumPy arrays and scalars and `Path` values become plain JSON, and NaN or infinity
raises `ValueError`. `write(path)` creates parent directories, writes the newline-terminated
canonical form and returns the path; `Receipt.read(path)` loads a file without checking it;
`to_dict()` returns the four fields. `cadence.receipts.canonical_sha256(value)` and
`cadence.receipts.source_manifest(pairs)` (not exported at top level) build the digest and the
`source` block.

## Verification

`Receipt.verify(path, *, sources=None, check=None)` returns `(ok, message)` and stops at the
first disagreement: (1) the file is byte for byte the canonical form of its content;
(2) the embedded digest matches; (3) when `sources` is supplied, the complete source
manifest matches; (4) when `check` is supplied, `check(body)` returns no problem text (for a
protocol receipt: every pass flag follows from the stored readings and every tally from the
rows). The message lists what was checked, or names the disagreement or the check's text.

## What belongs in the body

The connectome summary and digest (`Connectome.summary()`, `Connectome.digest()`) with the
custody block for measured data; the graph description (`NeuralGraph.to_dict()`: connectome
summary, neuron model, backend, transport, layout); the gain selection table with every gain
tried and its admissibility; the protocol as data, with the reference for every row; the
score with readings and reference readings on every row, and the control's score; the
backend's conformance report (`cd.conformance(graph, stimulus)` on a `NeuralGraph`); and a
boundary block saying what the experiment supplies rather than derives and what it does not
claim.

## The cost and the orienting fields

Every arm of a comparison carries the same accounting: `macs_per_moment`
(`BeliefPatch.macs_per_moment()` or `Steered.macs_per_moment()`), `moments_per_decision`
and the `cost` counters (`moments`, `macs`, `replays`) over the life. `Life.compute()`
returns these with the governor's steps added as moments and the decisions by mode.

A demo that reads a gain's response to events carries `cd.orienting` per kind (count,
capture of the first and last `bins` events, curve, latency shares, mean return) and the
`cd.dishabituation` pair, so that two demos' captures mean the same thing. An unobserved
return is `null`, with `return_censored=true` after a positive capture, and is not
counted as recovery. Partial final curve bins stay in the curve.

MAC counts are dense forward-work estimates that exclude adjoints, writes, nonlinearities,
callbacks and memory traffic; state these exclusions and any governor step-to-moment
conversion. An efficiency comparison also reports elapsed time, peak memory, learning and
search work and every failed attempt. Test targets never calibrate the governor or choose a
baseline; record the disjoint calibration and validation split and its seeds.

## Editing invalidates, on purpose

A source check fails once any file in the manifest changes. Record source hashes beside the
library version, because one version string can cover several source revisions. A receipt
is evidence for its own source snapshot: keep its bytes and verify against that snapshot
instead of re-signing an old result against later code. A run records data generation,
seeds, every scheduled condition, predictions and targets, and its verifier rejects missing
outcomes instead of averaging whichever rows remain.
