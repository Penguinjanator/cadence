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
starts with teacher imitation, then practices with advantage-weighted local
nudges, teacher corrections, and rehearsal. Its policy sees one current pixel
frame plus an input `Afterglow`. The shipped run retained the imitation
checkpoint because practice did not improve validation win rate.

The reward-only backprop REINFORCE control receives no teacher demonstrations
and uses different observations and reward/discount settings. These results do
not isolate the learning rule or establish an efficiency advantage. They also
do not test the complete `ActorCritic` trace/critic composition described above.
The example's tutorial and receipts distinguish the training stages and controls.

Pong uses a tracking-distance shaping term alongside game outcomes. Treat shaped
training return as a separate metric from points won, lost, or drawn. An
undiscounted potential difference does not generally preserve the original
objective under discounting; that guarantee requires the discounted form
`gamma * Phi(next) - Phi(now)` with appropriate terminal handling.

The [biology-to-Cadence map](biology.md#memory-and-learning) distinguishes
eligibility, prediction error, and memory, and explains the scope of the dopamine
analogy.

## Deployment

Choose the settling budget using changing observations and the deployed policy's
evaluation metric. Compare cold and warm starts, reset independent episodes, and
measure all decision work. A small step count or a cached response to one unchanged
input does not establish reliable control or lower inference cost.
