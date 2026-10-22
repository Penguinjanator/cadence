# Coupled memory relevance: admission before integration

This is a **component experiment**, on branch `coupled-memory-attention` from
v0.85.1. It does not change `Brain.compose` or the existing held-query reader.
One query selects both an event and its witnessed successor; these are not two
attention heads. Multiple heads remain a later, matched-cost comparison.

The demonstrated limitation of held-query retrieval is that the current
interpretation cannot revise its selection during a settle. The earlier
competitive reader described in [the binding instrument](../binding/index_read.py)
had a different failure: content strengthened its own selection, sometimes
completing a nearly random stored moment. Reinstalling it would not finish the
relevance mechanism.

The [protocol](protocol.json) fixes two candidate equations, two development
founders, a local budget and unused confirmation founders before behavioral
results. It keeps the original named-fact instrument and its acquisition floor
unchanged. If supplying both facts still fails to teach the query relation,
behavioral attention comparisons stop: that failure has not been localized to
memory. Numerical checks alone cannot admit a library option.

## Equations and boundaries

With witnessed keys `K`, contents `V` and a zero-content null slot, let
`p(q) = softmax(q K / T - recency * age, null / T)` and `R(q) = sum p_i V_i`.
Event and successor coordinates are concatenated in `V`; the selector is shared.
The candidate is an agreement energy on the live read activation `r`:

```
E_memory(q, r) = alpha / 2 * |r - R(q)|²
drive_read = alpha * (R(q) - r)
drive_query = alpha * J_R(q).T * (r - R(q))
```

The implementation evaluates the actual softmax Jacobian, including the mean
subtraction and null probability. A transpose of the normalized recall is not
that derivative. Reciprocal read/association and query/association projections
let the present interpretation and memory selection influence each other.
Query projections start at zero. The existing graph's local free/nudged contrast
teaches them; there is no separate address lesson or backpropagated answer head.
Derivative checks use finite differences as diagnostics only.

The second, simpler control is `E = -alpha q M r`, where
`M = K.T V / number_of_held_slots`. It supplies paired linear drives and avoids
claiming that a linear association is a normalized content selector. The old
competitive energy remains a negative control. A test retains its unsupported
positive and negative completions from tiny different starts, even with a null
slot. Agreement removes those completions in that example; that is not a proof
against all spurious equilibria.

The agreement candidate also has explicit negative checks: without query
evidence, biased stored contents can still drive a nonzero read, and duplicating
records changes their aggregate probability against the null. Neither a null
slot nor restoring terms establish warranted recall or learned abstention.

Observation codes, the working trace, witnessed keys, values, successors, ages,
slot masks and parameters are held across each free/positive/negative phase.
Query, read, association and motor activations are live. `p(q)` is an algebraic
relation recomputed in **every sweep and residual check**. No unresolved query
or read is frozen to publish an answer. A slot is written only after an actual
accepted observation, never from a teacher label or an imagined event.

The prototype uses the CPU float64 reference path and refuses damping rather
than allowing a fallback to drop the memory equations. It has no supported
reward, private imagination or pending-checkpoint continuation contract yet.
These lifecycle gates, backend conformance, partial-row teaching, causal
relevance, learned semantic codes and retention must pass before integration.
Raw coordinates in the admission chamber are explicitly a control. Neither its
finite acquisition nor a small joint residual establishes learned meaning,
two-link binding, sentence generation or coherent behavior over time.

## Run

From the repository root:

```sh
PYTHONPATH=src python -m pytest benchmarks/attention -q
PYTHONPATH=src python benchmarks/attention/admission.py --out /tmp/attention-admission.json
PYTHONPATH=src python benchmarks/attention/verify.py benchmarks/attention/results/admission-256.json
PYTHONPATH=src python benchmarks/attention/verify.py benchmarks/attention/results/admission-1024.json
```

The admission receipt records source hashes, the declared recipe, resolved
settings, training completion, decisions, refusals and solver work. Its wall
budget is checked between bounded trials; bounded evaluation follows. Host
seconds and solver work are not physical energy. A budget-truncated run is a
failure to establish acquisition, not evidence that the learner cannot acquire
the task with more work. Confirmation seeds stay unused on admission failure.

## 22 October 2026: admission remains open

Both recipes used the same normalized teaching, development seeds and task;
only the free/nudged sweep caps rose from 256 to 1,024 in the declared
[numerical retry](protocol-longer.json). Failed or unrun answers remain in the
denominator. These are final acquired readings, not best checkpoints selected
along the training path.

| Sweep cap per phase | Founder | Completed rounds / 256 | Final overall accuracy | Unequal-fact accuracy | Teaching refusals | Training action refusals |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 0 | 256 | 87.5% | 75% | 204 | 0 |
| 256 | 1 | 60 | 100% | 100% | 0 | 1 |
| 1,024 | 0 | 66 | 37.5% | 25% | 0 | 1 |
| 1,024 | 1 | 71 | 50% | 50% | 0 | 1 |

In the longer-budget final evaluations, half the queries refused for each
founder. A 100% answer reading after an incomplete training run does not pass
the declared completion gate. The two admissions took 15.14 and 5.76 host
seconds; their [receipts](results/) retain decisions, completion and all recorded
teaching/action solver work. No behavioral candidate round or confirmation seed
was used. Historical source bytes changed during CLI and guard work are retained
by content hash in `results/sources/`; the verifier checks those exact bytes,
receipt digests and recomputed scores/gates. It does not claim independent
behavioral reproduction.

The [diagnostic replay](diagnose.py) repeats the same failed 1,024-sweep recipe
and tries the existing three-halving damping fallback at each refused query.
Its [readback](results/damping-readback.json) retains the extra training and solve
work separately. Both queries still fail the original residual tolerance. This
does not establish that all solvers or teaching recipes fail; it establishes
that enabling that existing fallback at these boundaries does not close this
admission. The local Jacobian is measured at an **unqualified attempted state**,
not at an equilibrium, and is not a stability theorem.

The next behavioral delivery is an admitted acquisition/content-use chamber
under a separately declared recipe, preserving these failures and the 0.95
floor. Then compare held LMS and `toward` before the two bounded coupled
candidates, keeping information and teacher budget matched. The tiny equation
and local-credit probe is complete; learned causal relevance and lifecycle
integration are pending. This experiment must not be merged as completed
attention or used to close a relevance capability issue.
