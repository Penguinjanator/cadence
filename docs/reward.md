# Learning from reward

A body learns from what happens to it, not from a label. Cadence composes settlement contrasts, eligibility traces, and a reward prediction error
without a backward computation graph. This page says what each one
is, what it was measured to do, and what it was measured not to do.

## Three factors: the trace, the critic, the dopamine

`ActorCritic` wraps a `Learner`. Acting is a free settlement; the action is a draw from the
softmax over the output owners (or, with `Bins`, one softmax draw per dimension of a
population code). Then two nudged settlements toward and away from the
action taken give every seam its per-row contrast, the score of that action. Each seam keeps
an eligibility trace of those contrasts:

```
e[e]  <- gamma * lam * e[e] + contrast[e]
delta =  r + gamma * V(s') - V(s)
scale[e] += eta * delta * e[e]
```

`V` is a linear reading of the owners named as the critic (the hidden set, usually), trained
by its own trace and the same `delta`. The dopamine signal is clipped (`dopamine_cap`), can
be centred and scaled by its own running statistics (`dopamine_center`), and the critic's step
is divided by its trace's energy (`critic_normalize`), which is what stopped the value from
running away on Hopper. A time limit is not a terminal state: pass `bootstrap=` the value of
the last observation for a truncated row.

The adaptive local step is Adam written per seam: `momentum` and `normalize` keep a running
mean and RMS of each seam's own steps, bias-corrected. It matters. On cart-pole it took every
seed to 500 with the threshold at 20k to 60k steps; the uncorrected version explodes on its
first steps and the policy collapses.

```python
ac = cd.ActorCritic(learner, critic=wiring.sets["hidden"],
                    config=cd.ActorCriticConfig(gamma=0.99, lam=0.95, eta=0.001, eta_bias=0.0001,
                                         eta_critic=0.5, normalize=0.999, momentum=0.9))
action = ac.act(drive)                       # settle, draw, keep eligibility
obs, reward, done = env.step(action)
ac.learn(reward, done, next_drive)           # settle the next state, dopamine, every seam moves
```

## What the gates measured

Everything below is from `cadence-paper/experiments/gate_*`, five seeds unless noted, against
PPO with an MLP policy and PPO with a transformer policy trained in the same script.

| gate | patch net | PPO MLP | PPO transformer |
|---|---|---|---|
| cart-pole, final length | 500 | 497 | 500 |
| cart-pole, steps to 475 | 40k to 60k | 20k to 100k | 82k to 162k |
| cart-pole, parameters | 1,716 | 9,155 | 17,443 |
| cart-pole, multiply-adds per decision | 10.35k in the stored comparison receipt | 8.9k | 67.8k |
| cart-pole, wall-clock per run | 28 s | 14 s | 37 s |
| Pong, balls returned | 0.73 (three factors) | 0.93 (REINFORCE, rung 04) | |
| Hopper-v4 at 300k steps | 177 at best, then fades | about 1,000 | |

This cart-pole comparison favours the patch net on parameter count and its median
learning threshold. Its recorded compute per decision exceeds the MLP's, and its
wall-clock is higher. The two-warm-step deployment probe is a separate configuration. Pong and Hopper do not reach the bar with either rule; the numbers
are in the receipts and the paper's limitations, and the next candidates are a fused
learning loop and a critic with its own hidden owners.

## Deployment

One historical cart-pole deployment probe retained length 500 at two warm steps and
fell to 115 at one. This is not a general step budget for trained policies. Deploy with `engine.settle_batch(drive, steps=2, state=previous)`.
