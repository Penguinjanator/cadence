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
memory. The original admissions below failed; the separately declared
[teaching-policy repair](repair-protocol.json) now passes on both development
and all twenty fresh confirmation founders. Numerical checks and that acquisition
control alone cannot admit a library attention option.

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

## Earlier 22 October 2026 admissions: retained failures

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

These failures establish no retrieval deficit. The separately declared repair
below addresses exposed-fact acquisition, preserving these results and the 0.95
floor. Supplied-content use, held LMS versus `toward`, causal learned relevance
and lifecycle integration remain separate gates. This experiment must not be
merged as completed attention or used to close a relevance capability issue.

## Acquisition repair: the existing wrong-answer teaching control

The failed admission omitted an existing successful control in
[`acquisition_controls.py`](../relevance/acquisition_controls.py): teach only
incorrect answers from the **complete pre-query stimulus**. The new
[protocol](repair-protocol.json) fixes the task, two development founders,
twenty unused confirmation founders, a 600-host-second total budget and the
original 0.95 acquisition floor before running. It keeps the same 32-unit
reciprocal System 1, lateral 0, normalized teaching, `dt=0.1` and qualified
1,024-sweep free/nudged phases. Both matched arms now act before teaching;
one teaches every row, the other only wrong rows. The old admission's different
teach-before-act schedule remains explicit in its retained receipts.

Capture the complete drive before the query act advances the working trace,
then restrict that saved drive to the rows actually answered incorrectly.
Rebuilding it from a smaller observation batch would lose the original context;
rebuilding after acting would teach from the query-advanced trace. Teachers
still supply only final answers, with no slot index or relevance mask. An
all-correct round performs no lesson. Every phase of each accepted lesson
must qualify under the existing full-equation rule.

| Policy | Development founder | Completed rounds / 512 | Final overall accuracy | Lessons / taught rows | Training refusals |
| --- | --- | --- | --- | --- | --- |
| Every row | 0 | 67 | 37.5% | 67 / 1,072 | 1 query |
| Every row | 1 | 72 | 50% | 72 / 1,152 | 1 query |
| Wrong rows | 0 | 512 | 100% | 36 / 157 | 0 |
| Wrong rows | 1 | 512 | 100% | 31 / 103 | 0 |

Both wrong-row founders also reach 100% on unequal facts and each query. The
first development run was retained, then repeated after tightening only the
verifier. The replay reproduces **every complete per-run record**, including
parameter/checkpoint digests, decisions and work counts. The first passing
recipe was frozen before opening confirmation; the historical finite-teaching
fallback with lateral -0.5 was never run or restored as a default.

**Confirmation: all 20 fresh founders, seeds 100–119, pass.** Each completes
512 rounds and reaches 100% overall, unequal-fact and both-query accuracy,
with zero teaching or action refusals. Final acquired readings are used,
without selecting the best checkpoint along training. Initial mean accuracy
is 51.25%, final accuracy 100%, and the separate uniform-random control averages
50%. Confirmation uses 564 lessons on 2,579 taught rows. The original
development, verifier-only replay and confirmation take 29.47, 21.76 and
180.33 host seconds respectively, all charged within the declared 600-second
budget. Each evaluation forks saved state and checks that the acquired brain
is unchanged. Checkpoint contents are hashed in the receipts; binary checkpoints
remain local. This is not an efficiency comparison: the failed every-row arms
stop early and therefore do less work.

This repairs the acquisition **instrument's teaching recipe**, not a runtime
gradient or retrieval defect. Continuing to teach an already correct decision
can still move its parameters; wrong-row teaching removes that pressure. The
matched comparison measures this policy's effect here; it does not identify
the precise dynamical cause of the every-row refusals or propose a new default.

All sixteen fact/order/query cases are taught, using raw input coordinates as
an explicit control. Fresh founders do not establish unseen-binding or semantic
generalization. Trace-erased queries also qualify, but their mean accuracy is
77.5%; the intact acquisition result uses the continuing trial's prefix context.
This is not yet a test that the cortex can use content delivered through the
recall ports. The coupled reader still needs that content-use check, correct
partial-row memory snapshots, learned causal relevance, abstention, retention
and lifecycle integration. No behavioral coupled candidate was run. Seeds
100–119 are now opened for acquisition and must not be described as untouched
confirmation seeds for a later relevance candidate.

Run or verify the repair from the repository root:

```sh
PYTHONPATH=src python benchmarks/attention/repair_admission.py --stage development \
  --checkpoints /tmp/attention-dev-checkpoints --out /tmp/attention-development.json
PYTHONPATH=src python benchmarks/attention/repair_admission.py --stage confirmation \
  --development /tmp/attention-development.json --checkpoints /tmp/attention-confirm-checkpoints \
  --out /tmp/attention-confirmation.json
PYTHONPATH=src python benchmarks/attention/verify.py benchmarks/attention/results/repair-development.json
PYTHONPATH=src python benchmarks/attention/verify.py benchmarks/attention/results/repair-confirmation.json
```

The verifier binds the frozen protocol, complete ordered founder/arm sets,
matched development settings and schedules, actual answered cases, and original
checkpoint preservation. Optimistic flags cannot hide refused or unrun queries.
The earlier development source is retained by content hash alongside its first
receipt; a stricter verifier does not silently rewrite historical evidence.
