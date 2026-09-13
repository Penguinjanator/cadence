# A small core

Cadence's OPH-inspired unit is an observer-like patch: bounded state, input/output
ports, a readback, retained records, and a feedback move. A net composes those patches
through declared seams. Simplicity means a few explicit state transitions whose costs
and consequences can be checked; it does not require every transition to do the same job.

## Three kinds of state

| state | lifetime | update |
|---|---|---|
| activation and potential | a settlement or a continuing interaction | repair from neighbouring emissions and external drive |
| traces and fast records | an episode or an explicitly retained stream | decay, outer product, or residual correction |
| learned seam weights | across many observations | local free/nudged contrast, optionally weighted by reward credit |

A unique attracting equilibrium under a fixed drive erases its initial condition.
That can be useful for inference, but history must then enter through explicit records
or a changed drive. Multiple attractors and capped dynamics can retain history too;
they need their own tests. Calling all three state lifetimes 'one equilibrium' hides
this distinction and makes temporal bugs difficult to see.

## Keep the contracts small

- `Settlement` repairs owners and offers a separate residual diagnostic. It does not
  claim every capped run has converged.
- `Learner` coordinates free and nudged phases and their endpoint contrast. Its existing
  optimizer and scale conventions remain compatible.
- `Trace` keeps decaying activity. `Echo` and `Afterglow` provide familiar defaults.
- `FastSeams` keeps pairwise records. Direct `observe` and `recall` avoid building fake
  full-network states for a simple write or read. Delta mode writes prediction error;
  the original additive and one-hot replacement modes remain available.
- `ActorCritic` combines a contrast trace, value readout, and reward prediction error.
  Domain-specific rewards, curricula, notebooks, and exploration policies belong in
  their applications until controls justify a reusable primitive.

Key normalization and output softmax are declared group operations; the seam contrast
is endpoint-local. The reference ledger certifies the tested settlement arithmetic,
not every auxiliary computation. Public evidence bundles record both the mechanism
and its boundary.

## What would justify more machinery

A new mechanism needs a failure it addresses and an ablation on the same inputs,
budget and seed schedule. Count state, all updates, and inference cost. Check that a
simple direct solver does not already provide the same behaviour. Keep contradictory
results: residual memory can improve revision yet damage correlated older records.

Nature motivates local state, feedback and reuse. It does not prove this implementation
is optimal. The engineering question is which work each local correction removes and
which capability requires interaction between patches. [Memory](memory.md) and
[learning](learning.md) state the implemented answers.

API removals and compatibility decisions are recorded in the
[changelog](../CHANGELOG.md).
