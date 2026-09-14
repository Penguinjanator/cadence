# Learning: the free/nudged rule

Start with the runnable [two-label example](quickstart.md#learn-a-response).
This page explains supervised free/nudged learning without adaptation: the owner
equations, a numerical update, the gradient assumptions, and configuration choices.
Full comparisons live in [cadence-examples](https://github.com/muellerberndt/cadence-examples).

## 1. What is being learned

A settlement has three parameter arrays, all on the `Settlement`:

- `edge_scale`, one signed number per overlap. The effective drive of overlap `e` per unit
  of presynaptic activation is `gain · count[e] · edge_scale[e] · exp(log_gain[pre[e]])`;
  in a learned net `gain`, `count`, and `log_gain` are 1, 1, and 0, so the drive is just
  `edge_scale[e]`. Call it `W[e]`.
- `bias`, one number per owner.
- `log_gain`, one per owner, which the learner leaves alone.

Learning moves `edge_scale` and `bias`. It never touches the wiring: who talks to whom is
fixed, and only how strongly changes.

## 2. What a settlement computes

Each owner `i` holds a potential `v[i]` and publishes an activation `s[i]`. One step of the
rule, for every owner at once:

    inbox[i] = sum over overlaps e with post[e] = i of  W[e] · s[pre[e]]
    total[i] = inbox[i] + clamp[i] + bias[i]            (+ nudge[i] in the nudged phase)
    v[i]    <- v[i] + dt · (total[i] - v[i])
    s[i]     = act(v[i])

`act` is the rule's activation. `learning_rule()` sets slope 1, threshold 0, and leak 0.1:

    act(v) = tanh(v / 2)          for v >= 0
    act(v) = 0.1 · tanh(v / 2)    for v <  0

so rest (`v = 0`) publishes exactly 0, positive drive saturates toward 1 around `v ≈ 4`,
and negative drive publishes a small negative number instead of a hard zero. A
settlement runs this step from rest (or from a given state) until no owner's activation
moved more than `tolerance` in a step, or until the step cap. That state is what a
readout sees.

An input is a *clamp*: a fixed drive on the input owners, one number per owner
(a pixel's level, a board cell's occupancy). In a `layered` wiring, input owners
have no inbound overlaps. With zero input bias, at equilibrium each sits at
`v = clamp` and publishes `act(clamp)`. Other wirings can feed back into inputs.

## 3. The two phases and the update

For a batch of inputs with targets:

**Free phase.** Settle under the input clamp alone. Call the rest state `s⁰`. The output
owners' activations are the net's answer; no target has entered.

**Nudged phases.** From `s⁰`, settle again with an extra drive on the output owners only:

    nudge[i] = beta · weight · (target[i] - p[i])      for output owners i
    p        = softmax(s[outputs] / T)

`p` is a softmax over the output owners, so pushing the target's owner up pushes the
others down by their share (a cross-entropy nudge). `weight` is 1 for a label; for a
reward it is the advantage of the action (see [reward](reward.md)). With `centered=True`
(the default) two such settlements run from the same `s⁰`: one with `+beta`, rest state
`s⁺`, and one with `-beta`, rest state `s⁻`. The nudge travels back over the feedback
overlaps, so hidden owners rest at slightly different activations in the two phases.

**Update.** Every overlap reads its own two endpoints in the two phases, and every owner
reads itself:

    contrast[e] = mean over the batch of ( s⁺[pre[e]] · s⁺[post[e]] - s⁻[pre[e]] · s⁻[post[e]] ) / (2 beta)
    owner[i]    = mean over the batch of ( s⁺[i] - s⁻[i] ) / (2 beta)

    edge_scale[e] += eta   · contrast[e]
    bias[i]       += eta_b · owner[i]

Two more details, both in `Learner.update`: an overlap and its reverse (`i → j` and
`j → i`) form one *seam* and share one scale, so their two contrasts are averaged and both
move by the same amount; and a scale's magnitude is clipped to eight.
With `centered=False` the second nudged phase is skipped and `s⁻` is replaced by `s⁰` with
`beta` in place of `2 beta`.

The implementation retains the free and nudged endpoint states and optimizer history;
it does not retain a backward computation graph. Softmax reads its declared output
group, and weight tying aggregates the members of a declared tie group. These are
patch-level operations in addition to the endpoint-local contrast.

## 4. A worked example, with the numbers

Six owners: inputs 0 and 1, hidden 2 and 3, outputs 4 and 5. `cd.layered(2, 2, 2,
density=1.0, seed=3)` gives dense forward overlaps, feedback overlaps tied to them, and
two lateral overlaps between the outputs starting at 0. The rule is `learning_rule(dt=1.0)`;
the learner uses `eta=1.0`, `beta=0.1`, `T=0.2`, tolerance `1e-9` for tightly settled numerical phases.
The input is `x = (1.0, 0.2)` with label 1.

Initial seams (the reverse of each hidden↔output overlap shares its scale):

    0→2 +0.115   1→2 −0.587   0→3 +0.530   1→3 −0.196
    2→4 −0.479   3→4 +0.527   2→5 +0.633   3→5 +0.719     (feedback 4→2, 4→3, 5→2, 5→3 equal)
    4→5  0.000   5→4  0.000

Clamp: owner 0 gets 1.0, owner 1 gets 0.2, everyone else 0.

Free phase, 27 steps to rest:

    v = [1.000, 0.200, 0.011, 0.282, 0.071, 0.104]
    s = [0.462, 0.100, 0.005, 0.140, 0.036, 0.052]

Output owners publish 0.036 and 0.052; `softmax(s/T)` is (0.480, 0.520); the answer is
class 1, which happens to be right, but barely. Target: owner 5 at 1, owner 4 at 0.

Nudge drive at the first `+beta` step: `0.1 · ((0, 1) − (0.480, 0.520)) = (−0.048, +0.048)`
on owners 4 and 5. Nudged rest states:

    +beta (26 steps): s = [0.462, 0.100,  0.019, 0.143, 0.012, 0.078]
    −beta (21 steps): s = [0.462, 0.100, −0.001, 0.136, 0.064, 0.021]

Owner 5 went up and owner 4 went down under `+beta`, the reverse under `−beta`; the
hidden owners moved too, through the feedback seams, which is the only way the target
reaches them. Contrasts, `(s⁺s⁺ − s⁻s⁻) / 0.2`:

    0→2 +0.047   1→2 +0.010   0→3 +0.016   1→3 +0.003
    2→4 +0.002   3→4 −0.035   2→5 +0.008   3→5 +0.042
    4→5 −0.002   5→4 −0.002

Read the two largest. Seam 3↔5: hidden owner 3 was active (0.14) in both phases and output
5 rose under the nudge, so their product rose and the seam strengthens by +0.042. Seam
3↔4: the same hidden owner and output 4, which fell, so the seam weakens by −0.035. The
input seams into owner 3 also strengthen (0→3 by +0.016) because owner 3 itself ended a
little higher under `+beta` than `−beta`, which is the credit that flowed back. With
`eta = 1` those contrasts are the scale changes. Biases move by `eta_b · owner`:
owner 5 +0.0057, owner 4 −0.0052. After this single update a fresh free settlement gives
outputs (0.028, 0.067) instead of (0.036, 0.052): the right answer, with more room.

From a checkout, rerun the calculation with `python examples/worked_update.py`.

## 5. Why the contrast is a gradient

Let `F` be the recurrent owners and hold the input emissions fixed. With no adaptation,
symmetric **effective** recurrent weights, and a monotone differentiable activation,
the continuous-time dynamics descend

    E(v_F) = Σ_{i∈F} ∫₀^{v[i]} u · act'(u) du
             − ½ Σ_{i,j∈F} W[i→j] s[i] s[j] − Σ_{i∈F} d[i] s[i]
    d[i] = clamp[i] + bias[i] + Σ_{k input} W[k→i] s[k]

The one-way input projections belong in `d`, not in the symmetric recurrent sum.
A finite Euler step need not lower this energy. A stationary point need not be a
minimum or unique. Tying `edge_scale` alone does not make effective weights symmetric
if opposite contacts or presynaptic gains differ.

On a smooth stable equilibrium branch, converged free/nudged phases obey the
[equilibrium-propagation identity](https://arxiv.org/abs/1602.05179): the limiting
endpoint contrast is minus the loss gradient with respect to a physical seam weight.
The symmetric pair is one physical seam, so its two identical directed contrasts
must not be summed twice. For a quadratic nudge the loss is half squared error.
Cadence's cross-entropy drive omits `1/T`, so its loss is **T times cross-entropy**.
The [centered estimator](https://arxiv.org/abs/2006.03824) cancels the leading nudge
bias on a sufficiently smooth branch; a finite nudge crossing a kink or a different
attractor need not have that accuracy.

`Learner.contrast` returns a statistic, and `Learner.update` keeps the library's
existing step convention. When `c[e] = gain * count[e] * exp(log_gain[pre[e]])`
is not one, conversion to minus the gradient with respect to a tied `edge_scale`
requires multiplying by `c[e]` (assuming the reciprocal factors agree). For ordinary
cross-entropy also divide by `T`. Bias gradients need the same loss scaling but no
contact factor. Sharing a parameter across several distinct seams gives a sum of
contributions; the implementation uses their mean as a step-size convention.
`tests/test_equilibrium.py` checks the scale conversion against finite differences.

A small activation change is not a fixed-point certificate: saturation or tiny `dt`
can make it small while the potential is far from rest. Check the owner equations:

```python
free = learner.free(drive)
remaining = learner.engine.residual(drive, free)  # one diagnostic value per batch row
```

For a nudged phase, pass its actual `nudge` too, and inspect before changing the
engine's parameters. The diagnostic uses one transport evaluation and does not settle
again. `learning_rule` provides responsive defaults, but cannot guarantee convergence,
uniqueness, or accurate credit for every wiring. Leakage reduces dead regions; it
does not remove saturation or make the piecewise activation globally smooth.

## Feedback and the reach of a nudge

An endpoint contrast is zero if neither endpoint changes under the nudge.
Hidden owners therefore need a directed feedback path from nudged outputs.
`layered` supplies tied feedback between hidden and output owners. A general
directed graph may provide some such paths and omit others; asymmetry also breaks
the energy-gradient argument above. Feedback reachability alone does not ensure
useful credit if activations saturate or phases fail to converge.

For measured wiring, distinguish supplied topology from an assumed learning
mechanism. Adding reverse edges changes the model. Test that modelling choice
with controls rather than treating it as a biological consequence.

## 6. Using it

The [quickstart](quickstart.md#learn-a-response) creates and trains a learner from
scratch. For a dataset, construct `(batch, wiring.n)` drives and put features in
the input columns. `step` takes integer class indices within the output group;
for `slots`, use one index per row and slot. `accuracy` is the fraction of correct
choices over all rows and slots. Use [explicit target patterns](tasks.md#pattern-targets)
for regression or reconstruction.

`LearnerConfig` is frozen. To change the learning rate between epochs:

```python
from dataclasses import replace

# Continue with the learner from the quickstart.
learner.config = replace(learner.config, eta=0.5)
```

`learner.engine` is a plain `Settlement` at every moment: settle it, run `conformance` on
it, export `engine.dense()` for a page, put `to_dict()` in a receipt.

## 7. Every knob

| knob | where | what it does | where to start |
|---|---|---|---|
| `beta` | `LearnerConfig` | nudge strength; smaller is closer to the gradient, larger a stronger signal | 0.1 |
| `free_steps`, `nudged_steps` | `LearnerConfig` | the most steps a free and a nudged settlement may take before the contrast is read | 100, 50 |
| `tolerance` | `LearnerConfig` | a settlement stops once no owner moves more than this (None: the step cap alone) | 1e-4 |
| `eta` | `LearnerConfig` | seam step; the contrast is divided by `2 beta` | default 0.2; the two-label quickstart uses 2.0; tune on validation |
| `eta_bias` | `LearnerConfig` | bias step | `eta / 100` |
| `temperature` | `LearnerConfig` | softmax temperature of the cross-entropy nudge; also the policy temperature when sampling actions | 0.1 (labels), 0.2 (actions) |
| `centered` | `LearnerConfig` | contrast `+beta` against `−beta` (two nudged phases) rather than against the free state | `True` |
| `nudge` | `LearnerConfig` | `"cross_entropy"` or `"quadratic"` (`beta · (target − s)`) | cross-entropy for classes |
| `momentum` | `LearnerConfig` | each seam steps on a bias-corrected running average of its contrast | 0 (off); tune with the learning rate |
| `decay` | `LearnerConfig` | every update shrinks each trainable seam and bias by this fraction | 0 by default; decay also forgets useful weights |
| `normalize`, `normalize_floor` | `LearnerConfig` | divide each seam's step by its bias-corrected running RMS plus a floor | 0 (off); tune on validation; combining momentum and RMS gives an Adam-style update |
| `symmetric` | `Learner` | tie an overlap and its reverse into one seam | `True` |
| `trainable_overlaps` | `Learner` | bool per overlap; freeze the rest | all |
| `trainable_owners` | `Learner` | bool per owner bias; freeze the rest | all |
| `leak`, `slope`, `dt` | `learning_rule` | sub-rest response, activation slope, step of the owner update | 0.1, 1.0, 0.5 to 1.0 |
| `density`, `feedback`, `lateral`, `init`, `skip` | `layered` | input→hidden density, feedback scale, output↔output scale, initial magnitude, direct input→output seams | 1.0, 1.0, 0, 1.0, `False` |

`Learner.parameters()` counts trainable seam scales (shared pairs/groups count
once) plus unfrozen owner biases. A seam whose reverse is frozen still contributes
one trainable scale. Report optimizer arrays and episodic memory separately when comparing
storage; parameter count alone does not measure execution cost.

## 8. Warm starts, costs, and what to expect

A free phase may start from an earlier state (`learner.free(drive, warm=state)` or
`learner.step(..., warm=state)`). Within the same attracting basin this can save repairs;
with multiple attractors it can change the answer. A capped warm phase deliberately
retains transients and is not necessarily an equilibrium. Reset state at independent
episode boundaries and compare warm and cold inference on changing inputs.

Each centered update runs three phases. Measure all phase steps and wall time; fewer
epochs do not by themselves mean greater sample efficiency or lower compute. If the
task is simply storing and revising observations, [fast memory](memory.md) supplies a
single read and residual write without running three settlements for every record.
