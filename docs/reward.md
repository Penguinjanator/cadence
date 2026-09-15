# Learning from reward

Cadence combines free/nudged contrasts, eligibility traces and a reward prediction
error without a backward computation graph. Each synapse keeps its own eligibility
trace, and one broadcast dopamine signal gates every trace into a weight change.
This page describes those components and the tests that check them.

For application code, [`GenericBrain.step`](continuous.md) receives this moment's
observation and the preceding action's reward, updates plasticity, and returns the next
action. There is no separate training mode. The phase-level API below exposes the same
mechanism for custom architectures and measurement.

## Eligibility traces and dopamine

`ActorCritic` wraps a `Learner`. Acting runs the free phase; the action is a draw from the
softmax over each categorical motor slot (or, with `Bins`, one softmax draw per
continuous dimension). Unequal categorical slots use the existing `Learner(slots=[...])`
layout; every slot participates in the same settlement. Two nudged phases, toward and away from the
action taken, give every synapse its per-row contrast, an estimate of that action's score
under the [equilibrium assumptions](learning.md). The actor always uses a softmax
nudge, even when the learner uses quadratic imitation. For a fixed temperature `T`,
the contrast estimates `T * grad(log policy)`. Its common scale is absorbed in the
actor learning rate; several slots estimate the sum of their log probabilities. Each synapse keeps an eligibility trace
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
hidden population). The actor signal is clipped (`dopamine_cap`) and can be centred
and scaled by its own running statistics (`dopamine_center`). The critic has two
explicit choices:

- `critic_signal="modulated"` preserves the original rule: the critic shares the
  actor signal. This can stabilize an update, but clipping or centring means it need
  not predict mean discounted return in reward units.
- `critic_signal="td"` uses the raw prediction error for the critic, independently
  of actor modulation. Use this when a calibrated return prediction is required.
  It can require retuning the critic rate and reward scale. The default is
  `"modulated"`.

The critic's step can be divided by its trace's energy (`critic_normalize`). Reports
include the absolute raw `td_error`, the absolute modulated `delta`, and signed
`dopamine`. A time limit is not a terminal state: pass `bootstrap=` the value of
the last observation for a truncated row.

The adaptive local step uses `momentum` and `normalize` to keep a running mean and RMS
of each synapse's own steps, with corrections for its short history. Each synapse reads its
own optimizer state; this does not guarantee stable learning for every task or setting.

For custom wiring, build an actor on a declared output population:

```python
import numpy as np
import cadence as cd

connectome = cd.layered(2, 8, 3, density=1.0, seed=0)
learner = cd.Learner(
    cd.Brain(connectome, cd.learning_neuron_model()), connectome.populations["output"]
)
drive = np.zeros((1, connectome.n))
drive[:, list(connectome.populations["input"])] = [[0.5, 0.8]]
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
single phase. The next free phase estimates the bootstrap value before the parameter
update. The next action settles again from that warm state under the updated parameters.
A cached free phase is reused only when both the drive and the brain parameters are
unchanged. Budget for both free phases and the action’s two nudged phases.

`act(..., greedy=True)` is an evaluation read: it clears eligibility and cannot be
followed by `learn`. Calling another `act` replaces the pending action. Finish the
reward transition before changing learner parameters elsewhere, or call `reset`.
`reset` clears stream state and centering, but preserves the learned critic, actor and
optimizer history. `GenericBrain.save/load` includes this full state; `Learner.save`
does not save a separately constructed actor-critic.

## Reward evidence

Measure actual task outcomes separately from shaped reward. An
undiscounted potential difference does not generally preserve the original
objective under discounting; that guarantee requires the discounted form
`gamma * Phi(next) - Phi(now)` with appropriate terminal handling.

The [experience guide](experience.md) connects eligibility, prediction error
and valence to the rest of the learning life.

## Budget the ongoing loop

Choose the settling budget using changing observations and the task's outcome
metric. Compare cold and warm starts, reset independent episodes, and
measure all decision work. A small step count or a cached response to one unchanged
input does not establish reliable control or lower decision cost.

## What delayed credit requires

A reward that follows the reading it belongs to is a record: written in one exposure at
the fast rate and read back at that reading ([records](memory.md#records)). The
eligibility trace is for the policy where the reward arrives after other decisions.

An eligibility trace remembers which action could have caused a later reward; it
does not remember an arbitrary observation or learn a memory address. At a delay of
`d` transitions its contribution is multiplied by `(gamma * lam) ** d`. A trace
that fades too soon cannot assign useful credit, while a long trace includes more
unrelated actions. State representation, exploration and the critic matter too.

The [delayed-credit test](../tests/test_child.py) pays a choice three moments later,
after blank moments with presses of their own; the learned greedy choice reaches a hit
rate of at least 0.8 with the trace (`lam=0.9`) and at most 0.65 without it (`lam=0`). The
[policy credit tests](../tests/test_policy_credit.py) check that the eligibility
differentiates the sampled policy. Passing these small tests does not establish
performance on any larger task.

For a real task, record raw held-out return, forgetting and every interaction used
by rehearsal or planning. Choose the eligibility horizon from actual action-to-reward
delays, and test imitation, reward practice and rehearsal separately. A biological
analogy supplies a hypothesis; the behavioral test determines whether it works.
