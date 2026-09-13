# Learning to play a game

Two ways a patch net learns to act, both with the same free/nudged rule as
classification. The full, runnable versions with receipts and playable pages are
[04 Pong](https://github.com/muellerberndt/cadence-examples/tree/main/04_pong) in
cadence-examples and [03 Connect Four](https://github.com/muellerberndt/cadence-examples/tree/v0.5.0/03_connect_four)
at its tag v0.5.0; this page is the mechanism.

## Acting is a settlement

A game state becomes a clamp on the input owners: a board as one owner per cell per
side (1 where a disc sits), a screen as one owner per pixel (its brightness). The net
settles under that clamp; the output owners are the actions, and their rest activations
are the net's preferences. To act, take the most active legal output (a game), or draw
from `softmax(s_out / T)` (a policy that explores). There is no search inside the net
and no memory beyond what the clamp carries: two consecutive frames if the game needs
velocity, as Pong does.

## Way one: imitate a teacher (Connect Four)

If something can tell you the right move for a position, learning to play is
classification, and everything in [learning](learning.md) applies unchanged:

1. **Positions.** Play games with cheap players that make some random moves, so the
   positions are varied and plausible. Deduplicate.
2. **Labels.** For each non-terminal position, the move a deeper search prefers for the
   side to move. The teacher's depth is the ceiling of what the net can learn.
3. **Encoding from the mover's side.** One plane of the mover's discs and one of the
   opponent's, so one net plays both colours. Mirror positions left-right for free
   augmentation; mirror the labels with them.
4. **Learn.** Free phase under the position, nudged phases toward the teacher's column,
   local update. Select the hidden size on a validation split of the positions.
5. **Play.** Settle, mask illegal columns, take the most active output. Measure strength
   by matches against fixed opponents, alternating who starts, from random openings (a
   deterministic net against a deterministic search would otherwise replay one game).

What to expect: agreement with the teacher is the training signal, and play strength is
what it buys. A one-hidden-layer net that agrees with a depth-4 search on about half the
positions beats a random mover almost always and loses to a two-ply search almost always,
because it misses forced blocks often enough for an opponent that never does. An MLP of
the same size trained by Adam on the same positions lands in the same place. Imitation of
a shallow search is exactly that strong, whichever rule learns it.

## Way two: learn from reward (Pong)

When nothing says what the right action is, only how things went, the rule still applies
with one change: **the target of the nudge is the action the net took, and the strength
of the nudge is that action's advantage.**

```python
# one iteration
frames, actions, returns = rollout(policy, envs, steps)         # act by sampling softmax(s_out / T)
advantages = (returns - returns.mean()) / returns.std()
learner.step(drive(frames), actions, weight=advantages)         # free, +beta·A, -beta·A, update
```

`Nudge.weight` scales the nudge row by row. For a transition whose action paid
(`A > 0`) the output owner of that action is pulled up; for one that cost (`A < 0`) it is
pushed down; the limiting update is proportional to `A · d log π(a|s) / dW` under the
equilibrium conditions in [learning](learning.md). Finite nudges and incomplete
settlement introduce bias; policy temperature sets the proportionality. The goal, a scalar reward, enters through
the nudge alone.

### Setting up the reward so credit lands on the right step

The paddle's job in Pong is to be where the ball will be. The plain reward, `+1` for a
return and `−1` for a miss, arrives only at the end of a rally, many steps after the
moves that mattered, and a policy-gradient learner with that reward and a modest budget
learns something crude: "ball high, go up; ball low, go down", in absolute rows, ignoring
where its own paddle is. It returns some balls and looks bad.

The public Pong example adds a dense tracking reward proportional to the reduction
in paddle-to-ball distance. It changes the learning objective: with discount `gamma=0.5`,
the un-discounted difference `distance_before - distance_after` does not have the
policy-invariance guarantee of discounted potential shaping. That guarantee would
require `gamma * Phi(next) - Phi(now)` with suitable terminal handling. Both compared
policies receive the same implemented reward; evaluate the final ball-return rate
separately from shaped training return.

### Two frames

A single frame does not say which way the ball is moving, so a paddle that reads one
frame cannot anticipate a diagonal ball and both learners return fewer than 40% of them.
Clamp the previous frame alongside the current one and the direction is visible. The
page does the same: it keeps one earlier frame, nothing more.

### Evaluating a policy

Greedy play (most active output) on fresh seeds until a fixed number of points have
ended; report balls returned over balls faced, and returns per point. Cap rally length,
or two competent paddles can keep a horizontal ball going forever. Put a same-sized
network trained by backprop REINFORCE with Adam through the same interaction budget, reward, and
evaluation, and report both, with wall-clock.

## What the reward rungs found

The public Pong run returns about 88% of balls against 93% for backprop REINFORCE
with Adam; imitation of a tracker reaches about 96%. Each policy generates its own
rollouts under the same interaction budget and evaluation conditions. The adaptive
local step uses a running mean and RMS with short-history corrections; the current
`LearnerConfig.momentum` and `normalize` implement those corrections too.

These results do not identify one universal cause of the remaining gap. Reward timing,
exploration, credit assignment, finite-nudge bias, convergence, and representation all
need separate controls. Check the environment's observation/action/reward order before
interpreting a failed memory or delayed-reward experiment.

## Putting a trained net in a page

Every page settles the net in JavaScript, owner by owner, with the same rule; see
[pages](pages.md).
