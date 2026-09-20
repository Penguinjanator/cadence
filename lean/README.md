# Cadence proof library

This package keeps the relevant mathematical sources with the implementation.
Cadence's observer-like patches hold local state, publish values at ports,
read neighbouring values and repair discrepancies. Witnessed target nudges
can change persistent connections. The proofs below describe particular
equations and limits of this structure; they do not prove general intelligence
or turn a converged prediction into a true observation.

The default import is `CadenceProofs`. `CadenceRecords` is a separate import
for the optional older linear-record components. `PatchNet` does not use that
record store. Original theorem namespaces are retained so existing formal
references remain unambiguous.

## Reproduce the checks

The package pins Lean **4.29.1** and Mathlib revision
`5e932f97dd25535344f80f9dd8da3aab83df0fe6`. The committed
`lake-manifest.json` locks Mathlib and its transitive dependencies. From this
directory, with the pinned dependency cache already provisioned:

```sh
lake build CadenceProofs
lake build CadenceRecords       # optional record algebra
python3 -m unittest test_check.py
python3 check.py
```

`check.py` validates source hashes and dependency revisions, builds both
imports, and asks Lean to print the axioms of **every inventoried theorem**.
It rejects admissions, project-defined axioms, missing declarations and
unexpected axioms. The permitted foundational axioms are `propext`,
`Classical.choice` and `Quot.sound`; `sorryAx` is rejected. Successful checks
write the compact `verification.json` receipt. This is a check of the stated
mathematics, not a formal refinement proof for Python or numerical hardware.

Dependency caches and compiled artifacts belong under ignored `.lake/`.
They are not vendored or committed. An existing matching cache can be reused
without copying it:

```sh
mkdir -p .lake
ln -s /absolute/path/to/existing/pinned/.lake/packages .lake/packages
python3 check.py
```

On a fresh build machine with Lean's `elan` toolchain manager installed, first
run `lake exe cache get`. Lake provisions the revisions already recorded in
the committed manifest, and Mathlib retrieves the corresponding compiled
cache. Then run the checks above. Do not run `lake update` during reproduction:
that is an explicit dependency-update operation. This initial provisioning
requires several gigabytes; the local build reused the existing workspace
cache and downloaded no new one. `check.py` itself refuses missing or
mismatched dependencies rather than fetching them.

## What is connected to the implementation

| Proof module | Mathematical result | Code and limits |
| --- | --- | --- |
| `CadenceFlagship/Activation.lean` | Rectified sigmoid values and Lipschitz bounds. | [neuron.py](../src/cadence/neuron.py) implements the same real-valued formula numerically. A kink, rounding and backend differences remain outside these real-arithmetic proofs. |
| `CadenceFlagship/Settling.lean` | Fixed-drive contraction, uniqueness and error/lesion bounds under `0 < dt <= 1` and `L * row_mass < 1`. | [brain.py](../src/cadence/brain.py), [certificate.py](../src/cadence/certificate.py). Adaptation is absent from this map. Arbitrary trained weights, negative teaching phases and temporal anchors are not automatically certified by the unmodified row-mass bound. A small measured residual alone proves neither uniqueness nor local stability. |
| `Cadence/Budget.lean` | A geometric contraction certificate reaches a tolerance at an explicit logarithmic step threshold; a scalar contraction realizes the bound exactly. | Explains why a spent iteration cap alone cannot certify convergence in [brain.py](../src/cadence/brain.py). The theorem bounds distance to a fixed point, whereas the runtime measures an equation residual. Its supplied contraction premise is not automatically established for anchored, nudged or learned networks. An upper bound above tolerance does not imply that the actual error exceeds tolerance. |
| `CadenceFlagship/Credit.lean` | Endpoint contrast is local, symmetric and zero when both endpoints agree. | [learning.py](../src/cadence/learning.py). These algebraic identities do **not** prove the equilibrium-propagation gradient theorem. Centered contrast uses the positive and negative endpoints and denominator `2 * beta`; one-sided contrast uses `beta`. Momentum and decay can move parameters even when raw contrast is zero. |
| `Cadence/UpdateRule.lean` | Bias-corrected moving-average identities and conditional normalized-step bounds; zero-leak rectifier properties. | [learning.py](../src/cadence/learning.py) and [plasticity.py](../src/cadence/plasticity.py). The tight step bound assumes equal averaging rates; it does not cover arbitrary unequal-rate optimizers. Tying, clipping, finite precision and successful task learning require separate checks. |
| `CadenceFlagship/Wiring.lean` | Conditional contraction bounds survive specified weakening, masking or bounded added matrix terms. | Connectome/certificate mathematics in a fixed ambient dimension. It does not establish autonomous graph growth, learned useful specialization, or preservation under unrestricted learning. |

The Python interface and invariants for `PatchNet` are tested in
[test_patch.py](../tests/test_patch.py) and
[test_temporal_patch.py](../tests/test_temporal_patch.py). Exact checkpoint
continuation, source-ID bookkeeping, isolated imagination and Python residual
checks currently have executable tests, not end-to-end Lean correctness proofs.

## Abstract memory obligations

These modules state conditions a useful memory system would have to satisfy.
Their premises must be established before applying a conclusion to a network.

| Proof module | Result and interpretation |
| --- | --- |
| `CadenceMission/MemoryBoundary.lean` | An adequate representation must distinguish histories with different admitted future answers. Exact finite capacity, decoder/update compatibility, and causal-topology counterexamples. A DAG or maximal event IDs alone do not retain missing payloads. No learned address or semantic relevance theorem is supplied. |
| `CadenceMission/NoiseBoundary.lean` | Repair can preserve whatever record it is given, including a wrong one. Identical available evidence forces a noise-versus-change tradeoff. The record-preservation premise is explicit; it is not a discovered property of `PatchNet`. |
| `CadenceMission/BasinRetention.lean` | A supplied basin survives accepted moves under a positive allowed-exit margin and a sufficiently small energy change relative to one fixed reference. The actual network has no automatically certified exit margin, accepted-descent trace or unrestricted-learning guarantee. |
| `CadenceMission/SpecializationBoundary.lean` | Symmetric deterministic dynamics cannot break an invariant state under invariant inputs; useful complementary roles may still optimize a symmetric task. Minimizing disagreement plus wiring cost alone can select disconnection. These are constraints, not evidence of learned roles. |
| `CadenceMission/TemporalOverlap.lean` | For the scalar energy `a*z^2/2 + k*(z-c)^2/2`, `a>0`, `k>=0`, the exact next value is `k/(a+k)*c`. Its iterates and history separation decay geometrically to zero. This explains why finite temporal coupling need not create permanent memory in a convex example. It is not a theorem about every nonlinear, driven, learning `PatchNet`. |

The opt-in temporal boundary in [patch.py](../src/cadence/patch.py) uses the
preceding **free** activity, held fixed across the next free and nudged phases.
Fresh/reset streams use zero activity. A boundary value is not an external
teaching target. The scalar theorem does not certify its discrete solver or
provide credit assignment through prior observations.

## Temporal paths and compressed boundaries

`CadenceMission/CausalTemporalUniqueness.lean` proves that a finite causal
path with a fixed initial boundary and deterministic transition maps has
exactly one defect-free realization: its forward recurrence. This applies
to the target-free construction in [temporal.py](../src/cadence/temporal.py).
It does not assert uniqueness of a nonlinear detuned stationary point or of
an arbitrary cyclic network. Its four theorems need no axioms.

`CadenceMission/QuadraticBoundary.lean` proves six scalar identities and bounds
for eliminating an old state from a quadratic temporal energy. For positive
prior precision and nonnegative transition precision, the reduced boundary
energy is attained and bounds the full energy from below. Information and
precision preserve later minimization; a constant is needed for absolute
energy. All model coefficients are fixed, and later objectives may read only
the retained boundary. The result does not establish nonlinear compression,
learned importance, or exact reinterpretation after changing the model.

These sources are copied byte-for-byte from the reviewed research modules.
They are included in the default import and the complete axiom audit.

## Optional linear-record algebra

`CadenceRecords` imports `CadenceFlagship/Memory.lean` and `Sparse.lean`.
They prove ideal linear read/write identities, exact unit-key replacement,
orthogonal/disjoint noninterference and finite-dimensional capacity bounds.
The addresses, normalization and update law are supplied assumptions. The
runtime's learning rates, decay, consolidation and finite precision are not
silently included. These results can inform the older
[memory.py](../src/cadence/memory.py) and
[stream.py](../src/cadence/stream.py) components; they do not establish
automatic noninterference or describe a hidden store inside `PatchNet`.

## Source management

`sources.json` records the original workspace source and SHA256 for each
curated module, its current local SHA256, and its scope. Builds work without
those sibling projects. The imported mathematical statements are preserved;
documentation clarifications and compatibility repairs are recorded in the
manifest. To change a
proof, review its hypotheses and code relationship, update the local source
hash and provenance note, and rerun `check.py`. Do not silently refresh copies
from a moving sibling checkout.

The package intentionally excludes unrelated physics proofs, prototype
quadratic/star experiments, historical asymptotic cost claims, rollback and
aversion mechanisms, Bayesian filter candidates, and binary structural-weight
wells. None is required to
state the implemented common core's current guarantees.

### Conditional protected-path retention

`CadenceMission/ProtectedTemporalPath.lean` proves three conditional statements:
if a new transition agrees with the old transition along one protected finite
path, that path remains defect-free and equals the new forward recurrence.
Agreement of the readback maps on that path preserves its published values.
The proof reuses causal-path uniqueness and requires no axioms. It assumes
exact map agreement; it does not verify floating-point SVD/projection, select
importance, protect neighboring cues or establish unlimited plastic capacity.
[TemporalMemory](../docs/temporal-memory.md) exposes the corresponding explicit
response constraints and their finite numerical scope.
