# Concepts

Cadence's observer-like patch is a bounded piece of software state with ports,
readback, records, and a feedback move. This is a design for computation; it does
not ascribe experience to the software or establish a biological brain model.

## Owners and seams

An *owner* holds a potential `v` and publishes an activation `s`. A directed
*overlap* carries one owner's published value to another. A *seam* may tie an
overlap and its reverse to one learned weight. `Wiring` declares these connections
and named sets of owners, such as `input` and `output`.

`Settlement` holds the wiring, rule, and parameter arrays. On each step it
collects incoming messages, then each owner updates its own state:

```text
inbox[i] = sum(weights[e] * s[pre[e]] for edges ending at i)
v[i]    += dt * (inbox[i] + drive[i] + bias[i] - strength * a[i] - v[i])
s[i]     = activation(v[i])
```

The adaptation term is absent unless enabled. The effective edge weight is
`gain * count[e] * edge_scale[e] * exp(log_gain[pre[e]])`.
`edge_scale` starts from the wiring's signed values and can be learned.
Transport follows the declared edges; the owner update reads its own state and
inbox. Softmax and key normalization additionally read their declared groups.

The external *drive* is added on every step. Although the API calls it a
*clamp*, it does not forcibly hold an owner's potential at that value when the
owner also receives feedback or bias.

## Settlement and equilibrium

Settlement means running the update, from rest or a supplied state. A returned
`SettledState` can be a transient, a fixed point, or part of an oscillation.
`steps` limits work and `tolerance` stops on small activation movement.
`Settlement.residual` separately measures the remaining fixed-point equation
error. Saturation can produce small movement with a large residual.

An equilibrium need not be unique or stable. Carrying a state between inputs can
save work or select a different attractor. Test both cold and warm starts, and
reset state at independent episode boundaries. A unique attracting equilibrium
under a fixed drive erases its initial condition; keeping history then requires
an explicit record, a trace, or a drive that carries history.

## Activation and adaptation

`GradedRule` rebases a sigmoid so that zero potential emits zero. With zero
drive, zero bias, and zero adaptation, the all-zero state is an exact fixed
point. `learning_rule()` uses a gentler slope and a small negative leak to give
the learner a responsive starting point. Neither setting guarantees convergence.

`Adaptation` adds one variable per owner that follows its activation and
subtracts from its drive. In suitable mutually inhibitory circuits this can
produce an oscillation. The [half-center example](../examples/half_center.py)
demonstrates one such circuit. Leave adaptation off when beginning with
equilibrium learning; its gradient interpretation needs additional assumptions.

## Three state lifetimes

| State | What changes it | How to manage it |
|---|---|---|
| Potential, activation, adaptation | Settlement steps | Pass `state=` to continue; omit it to start from rest |
| A trace or fast-memory matrix | Explicit activity or observation updates | Reset at episode boundaries and preserve batch row identities |
| Learned weights and biases | `Learner.step` or `Learner.update` | Save with `Learner.save`; evaluate with `predict` or `free` |

A [trace](api.md#streams-cadencestream) retains fading activity.
[Fast memory](memory.md) retains associations between supplied keys and values.
[Learning](learning.md) changes a reusable response through free/nudged endpoint
contrasts. The centered learner uses three phases: free, positive nudge, and
negative nudge. Under its equilibrium assumptions, the small-nudge contrast
corresponds to a loss gradient with the stated parameter scaling.

## Evidence and controls

A [protocol](protocols.md) declares stimuli, readouts, interventions, and predicates.
A shuffled wiring tests whether a response depends on the particular connections
under the same rule. It is one control, not proof of a biological mechanism.
High gains can saturate an excitatory circuit, so protocols can limit the active
fraction during gain selection.

`conformance` compares a trajectory with the owner-by-owner reference.
It checks the tested transport and update. A [receipt](receipts.md) binds stored
results to sources when those files are included and checked; its caller supplies
the arithmetic verifier. Neither check establishes benchmark fairness.

The [small-core rationale](condense.md) explains why inference, memory, and
learning remain separate operations. Simplicity is a design constraint, not a
proof that one architecture is optimal.
