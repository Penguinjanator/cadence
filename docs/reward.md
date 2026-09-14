# Learning from reward

Cadence combines free/nudged contrasts, eligibility traces and a reward prediction
error without a backward computation graph. Each synapse keeps its own eligibility
trace, and one broadcast dopamine signal gates every trace into a weight change.
This page describes those components and the public evidence for reward-weighted learning.

## Eligibility traces and dopamine

`ActorCritic` wraps a `Learner`. Acting runs the free phase; the action is a draw from the
softmax over the output neurons (or, with `Bins`, one softmax draw per dimension of a
population code). Two nudged phases, toward and away from the
action taken, give every synapse its per-row contrast, an estimate of that action's score
under the [equilibrium assumptions](learning.md). Each synapse keeps an eligibility trace
of those contrasts:

```
elig[e]      <- gamma * lam * elig[e] + contrast[e]
delta         = r + gamma * V(s') - V(s)
efficacy[e] += eta * delta * elig[e]
```

This is a three-factor rule. Presynaptic and postsynaptic activity enter through the
contrast and accumulate in the synapse's eligibility trace; the prediction error `delta`,
the dopamine signal, is the same for every synapse and sets the sign and size of the
weight change. `V` is a linear readout of the neurons named as the critic (usually the
hidden population), trained by its own trace and the same `delta`. The dopamine signal is
clipped (`dopamine_cap`), can be centred and scaled by its own running statistics
(`dopamine_center`), and the critic's step is divided by its trace's energy
(`critic_normalize`). A time limit is not a terminal state: pass `bootstrap=` the value of
the last observation for a truncated row.

The adaptive local step uses `momentum` and `normalize` to keep a running mean and RMS
of each synapse's own steps, with corrections for its short history. Each synapse reads its
own optimizer state; this does not guarantee stable learning for every task or setting.

Continue with the learner and drive from the [quickstart](quickstart.md#learn-a-response):

```python
ac = cd.ActorCritic(learner, critic=connectome.populations["hidden"],
                    config=cd.ActorCriticConfig(gamma=0.99, lam=0.95, eta=0.001, eta_bias=0.0001,
                                                eta_critic=0.5, normalize=0.999, momentum=0.9))
action = ac.act(drive)                       # settle, draw, keep eligibility
reward, done, next_drive = np.ones(len(drive)), np.zeros(len(drive), bool), drive  # from your environment
ac.learn(reward, done, next_drive)           # settle the next state, dopamine, every synapse moves
```

## Decision and reset contract

Drives are nonempty finite `(batch, neurons)` arrays. Rewards and optional bootstrap
values have shape `(batch,)`; `done` is boolean. Validation precedes trace and parameter
updates. Each batch row keeps its own potential, adaptation, eligibility and reward
centering. Ended rows reset before the next free phase; their neighbours get the usual
single phase. The next free phase is computed before the parameter update and reused
for the next action if its drive matches. This is a finite-step approximation; compare
it with freshly settled deployment behavior when tuning budgets.

`act(..., greedy=True)` is an evaluation read: it clears eligibility and cannot be
followed by `learn`. Calling another `act` replaces the pending action. Finish the
reward transition before changing learner parameters elsewhere, or call `reset`.
`reset` clears stream state and centering, but preserves the learned critic, actor and
optimizer history. `GenericBrain.save/load` includes this full state; `Learner.save`
does not save a separately constructed actor-critic.

## Reward evidence

The archived [Pong experiment](https://github.com/muellerberndt/cadence-examples/tree/7302f2af3dc0638bbafd1da1446ed96ffabaa9dd/04_pong)
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

The [function map](biology.md#learning) places eligibility, prediction error
and valence among the other mechanisms.

## Deployment

Choose the settling budget using changing observations and the deployed policy's
evaluation metric. Compare cold and warm starts, reset independent episodes, and
measure all decision work. A small step count or a cached response to one unchanged
input does not establish reliable control or lower inference cost.
