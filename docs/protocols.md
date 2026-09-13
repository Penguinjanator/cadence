# Protocols: held-out tests with preconditions

A protocol is a declared list of what a settled net will be asked, and what it may be
shown before it is asked. It can score a supplied graph model, and the same
discipline (a validation split, a test read once, a control) is what the learning
examples follow in their scripts.

## Stimuli, rows, training facts

For a complete runnable circuit and protocol, see
[`examples/ring_protocol.py`](../examples/ring_protocol.py). This fragment assumes
an engine whose wiring has `sensors` and `motors` sets:

```python
import cadence as cd
from cadence.protocol import Levels

protocol = cd.Protocol(
    stimuli={"rest": (), "touch": ("sensors",)},              # name -> sets clamped at full amplitude
    training=[("touch", "motors", "active")],                  # what the model may be shown
    rows=[
        cd.Row("R1", "rest", "motors", "inactive", "no input, no output"),
        cd.Row("R2", "touch", "motors", "reduced", "cutting the sensors", ablate=("sensors",)),
    ],
    levels=Levels(active=0.5, inactive=0.2, margin=0.15),
    steps=60,
)
report = protocol.score(engine)       # {"rows": [...], "passed": k, "total": m, ...}
```

A `Row` names a stimulus, a readout set, a predicate, a citation-style reference, and
optionally sets to ablate (owners zeroed for that row) and a second readout to compare
against (`relative_to`). For `exceeds` and `lateralized`, a matching readout
name takes priority over a stimulus name; other predicates prefer a matching
stimulus. An unknown nonempty name raises `KeyError`. `tier` groups results.

## Predicates and their preconditions

| predicate | passes when | precondition |
|---|---|---|
| `active` | mean(readout) ≥ `active` | |
| `inactive` | mean(readout) ≤ `inactive` | |
| `reduced` | mean(readout) ≤ mean(reference) − `margin` | the reference (intact) net was active |
| `retained` | mean(readout) ≥ `active` | the reference was active |
| `released` | mean(readout) ≥ mean(reference) + `margin` | the reference was inactive |
| `exceeds` | mean(readout) ≥ `active` and ≥ mean(other) + `margin` | |
| `lateralized` | absolute mean difference ≥ `margin` | at least one mean ≥ `inactive` |
| `sparse` | `sparse_min` ≤ fraction active ≤ `sparse_max` | |
| `densified` | fraction(readout) ≥ fraction(reference) + `densify_margin` | the reference was sparse |

A precondition that fails makes the row fail. That is what stops a dead net from passing
`reduced` or a saturated one from passing `released`. `evaluate_predicate(predicate,
value, reference, levels)` is the pure function, and a receipt's verifier reruns it on the
stored readings.

## The shuffled control

`cd.shuffled(wiring, seed)` permutes the postsynaptic endpoints of every overlap and keeps
everything else: every count, every sign, every owner's out-degree, every named set, and
no autapses. Score the protocol on it with the same rule and gain. A difference
is evidence that the particular connections matter under the declared model and
protocol. Use several shuffle seeds; one shuffled failure does not isolate a
biological explanation or every possible graph confound.

## Gain selection under a sparsity cap

```python
gain, table = cd.select_gain(lambda g: cd.Settlement(w, rule.replace(gain=g)), protocol,
                             grid=(0.01, 0.02, 0.03, 0.05), sparsity_cap=0.05)
```

Every gain on the grid is tried. A gain is admissible when no more than the
cap's fraction of owners is active under any training stimulus. Selection
maximizes training facts passed among admissible gains, breaking ties by smallest
gain. It can return a gain that fails some or all training facts: inspect the
table's `facts_passed` as well as `admissible`.

An empty grid or no admissible candidate raises `ValueError`. Declare a wider
grid or a different cap before evaluation, rather than silently using a rejected
candidate. Pass `sparsity_cap=None` for toy nets deliberately allowed to light
entirely. Include the complete selection table in the receipt.

## For learned nets

The learning examples do not use `Protocol`; their held-out facts are a test set. The
same three habits carry over: selection on a validation split of the training data only,
the test set read once after selection, and a control that should fail (a net trained on
shuffled labels scoring at chance).
