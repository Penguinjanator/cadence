# Quickstart: the existing GenericBrain composition

This guide retains the `GenericBrain` interface for existing applications.
Start with [TemporalPatchNet](temporal.md) and
[learn, act and observe](interaction.md) for the current temporal core.

`GenericBrain` supports **ongoing experience**: the same composed brain
observes, acts, remembers and learns throughout its life. This interface has
no training/inference switch or separate model to deploy after learning.

Install from the [README](../README.md#current-library). The complete walkthrough below
needs only NumPy and Cadence.

## One mode, one loop

Each `GenericBrain.step` receives the current observation and the outcome of its
**previous action**, incorporates that experience, and chooses the next action.
The first call only chooses an action, because no previous outcome exists.

Here a body moves left, stays still or moves right along a line. Its observation
contains its position and a goal. Reward is the actual reduction in distance
after the body moves. Halfway through, the goal changes; the same brain continues.

```python
import numpy as np
import cadence as cd

brain = cd.GenericBrain.build(2, 3, hidden=16, working_memory=True, seed=0)
moves = np.array([-0.1, 0.0, 0.1])
position, goal = 0.5, 0.8
action = brain.step([[position, goal]])

for moment in range(64):
    previous_distance = abs(goal - position)
    position = float(np.clip(position + moves[int(action[0])], 0.0, 1.0))
    reward = previous_distance - abs(goal - position)  # actual consequence

    if moment == 31:
        goal = 0.2
    action = brain.step([[position, goal]], reward=[reward])

assert brain.basal_ganglia.updates == 64
print("Observed and learned from 64 action outcomes.")
```

The loop makes 64 real transitions and leaves one next action ready to execute.
It demonstrates the interface; learning a reliable controller requires its own
curriculum, measurements and controls. The body, goal and reward rule are supplied.

The brain retains its policy, critic, associative reward memory and working trace.
It never needs to enter a different mode to put a learned response to use.
`GenericBrain` is a starting composition. A [records cortex](memory.md#records) learns the
consequences of actions and their reward from the same stream, and
[compose a brain](brain.md) writes one experience step with records beside a policy.
Learned language needs additional wiring; see [experience-based architectures](experience.md).

## Keep feedback attached to the right action

| Event | What to do |
|---|---|
| First observation | Call `brain.step(observation)`. |
| Action has executed | Pass its reward with the resulting observation to the next `step`. |
| Real transition with no reward event | Pass zero, or omit `reward`; learning and eligibility still advance. |
| Outcome has not arrived | Wait for it. Another `step` would consume the pending action as a zero-reward transition. |
| Episode ended | Pass `done=[True]` with the next episode's reset observation and the previous action's reward. |
| Current demonstration | Pass `teacher=[action_index]` for the current observation, alongside any previous-action feedback. |

Observations have shape `(batch, inputs)`: `[[position, goal]]` is one stream.
Actions, rewards and `done` have one entry per row. Keep each row attached to the
same stream. For truncation and custom scheduling, see [continuous interaction](continuous.md).

## Continue the same life after saving

```python
path = brain.save("living_brain.npz")
brain = cd.GenericBrain.load(path)
assert brain.basal_ganglia.updates == 64
```

This preserves learned parameters, memories, random state and the pending action's
eligibility. Save the environment and issued action separately. After restoring,
execute that action or receive its actual outcome, then continue the same loop.
`reset()` clears current activity and pending credit; it is not a mode switch.

## One mode, distinct operations

Settling changes neural activity. Real observations, demonstrations and rewards
supply the signals that change memories and synapses. The local learning rule
holds weights fixed during its free and nudged phases, then applies an update.
Those are numerical operations inside the ongoing loop.

Calling a raw `Brain.settle` does not automatically teach its weights. Imagined
outcomes are predictions; keep them out of the records of observed evidence. Custom
cortexes use the same [local learning primitives](learning.md) and own their feedback
timing; [write a cortex](cortex.md) describes their ports and heads.

For a frozen measurement, use a separate saved copy. Independent-sample helpers
such as `fit` and `predict` do not toggle the live brain's operating mode.
Check residuals before claiming equilibrium, and observed task outcomes before
claiming a capability. See [concepts and limits](concepts.md).
