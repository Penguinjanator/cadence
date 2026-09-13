# Learning from reward

Cadence composes settlement contrasts, eligibility traces and a reward prediction
error without a backward computation graph. Its observer-like patches retain bounded
local state, read through declared ports, and use feedback to repair their response.
This page describes those components and the public evidence for reward-weighted learning.

## Three factors: the trace, the critic, the dopamine

`ActorCritic` wraps a `Learner`. Acting is a free settlement; the action is a draw from the
softmax over the output owners (or, with `Bins`, one softmax draw per dimension of a
population code). Then two nudged settlements toward and away from the
action taken give every seam its per-row contrast, an estimate of that action's score
under the [equilibrium assumptions](learning.md). Each seam keeps an eligibility trace
of those contrasts:

```
e[e]  <- gamma * lam * e[e] + contrast[e]
delta =  r + gamma * V(s') - V(s)
scale[e] += eta * delta * e[e]
```

`V` is a linear reading of the owners named as the critic (the hidden set, usually), trained
by its own trace and the same `delta`. The dopamine signal is clipped (`dopamine_cap`), can
be centred and scaled by its own running statistics (`dopamine_center`), and the critic's step
is divided by its trace's energy (`critic_normalize`). A time limit is not a terminal
state: pass `bootstrap=` the value of
the last observation for a truncated row.

The adaptive local step uses `momentum` and `normalize` to keep a running mean and RMS
of each seam's own steps, with corrections for its short history. Each seam reads its
own optimizer state; this does not guarantee stable learning for every task or setting.

```python
ac = cd.ActorCritic(learner, critic=wiring.sets["hidden"],
                    config=cd.ActorCriticConfig(gamma=0.99, lam=0.95, eta=0.001, eta_bias=0.0001,
                                         eta_critic=0.5, normalize=0.999, momentum=0.9))
action = ac.act(drive)                       # settle, draw, keep eligibility
obs, reward, done = env.step(action)
ac.learn(reward, done, next_drive)           # settle the next state, dopamine, every seam moves
```

## Public reward evidence

The [Pong example](https://github.com/muellerberndt/cadence-examples/tree/main/04_pong)
compares a reward-weighted free/nudged learner with backprop REINFORCE using the same
interaction budget, reward and evaluation protocol. Its historical receipt reports
about 88% of balls returned by the patch net and 93% by REINFORCE. The policies generate
their own rollouts, so this comparison does not isolate one cause of their difference.
It tests that example's learner, rather than the complete `ActorCritic` trace/critic
composition described above. The tutorial identifies the preserved source version
and explains how to verify or rerun it.

Pong adds the undiscounted reduction in paddle-to-ball distance as a shaping reward.
With the script's discount `gamma=0.5`, this is not guaranteed to preserve the best
policy for the original return/miss reward. Discounted potential shaping would require
`gamma * Phi(next) - Phi(now)`, with suitable terminal handling. Both policies receive
the same implemented reward; final ball-return rate is evaluated separately from
shaped training return.

## Deployment

Choose the settling budget using changing observations and the deployed policy's
evaluation metric. Compare cold and warm starts, reset independent episodes, and
measure all decision work. A small step count or a cached response to one unchanged
input does not establish reliable control or lower inference cost.
