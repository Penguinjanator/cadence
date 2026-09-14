# Learning to play a game

Build enough capacity, imitate useful behavior, practice, and revisit mistakes.
Cadence supplies settlement, local learning, memory, and reward primitives; the
application supplies observations, actions, a teacher, and an environment. The
[biology-to-Cadence map](biology.md) explains these functional correspondences.

This guide describes historical game experiments. Their training, receipts and
playable pages are retained at a fixed source revision in
[03 Connect Four](https://github.com/muellerberndt/cadence-examples/tree/7302f2af3dc0638bbafd1da1446ed96ffabaa9dd/03_connect_four)
and [04 Pong](https://github.com/muellerberndt/cadence-examples/tree/7302f2af3dc0638bbafd1da1446ed96ffabaa9dd/04_pong).
The current [public showcase](https://github.com/muellerberndt/cadence-examples)
focuses on embodiment and continual memory; the game launchers below apply only
to that historical checkout.

## Observe, remember, act

A game observation becomes a drive on input owners. Settlement produces output
activations; mask illegal actions, then take the most active output or sample
from a softmax to explore. Inputs and memory should match the information needed
for the decision:

- **Connect Four:** two 42-cell planes encode the current player's and opponent's
  discs. The board is fully visible. The browser can additionally compare futures
  through supplied game-rule search, with learned scores breaking ties.
- **Pong:** one current 12 × 16 pixel frame plus `Afterglow` of input activity,
  with 192 trace owners, decay 0.5, and `focus=0`. The learned policy uses current
  pixels and their fading history to choose among three paddle actions.

Pong does not receive a second frame as an observation. Advance its trace once
per real decision and reset at point boundaries. Replaying a training sample
uses its recorded causal trace; shuffled samples must not advance live memory.
Warm recurrent state and an explicit trace are different records; see
[concepts](concepts.md) and [memory lifetimes](biology.md#which-memory-survives-what).

## Imitate, experiment, revisit

1. **Choose the layout.** Define observation and action ports, necessary memory,
   and sufficient hidden capacity. Select sizes using validation data.
2. **Imitate.** Collect varied demonstrations and use `Learner.step` to teach
   actions. Keep equivalent positions and their augmentations in the same split.
3. **Practice.** Play games and learn from outcomes or teacher corrections on
   encountered states. Stochastic exploration helps expose different situations.
4. **Revisit weaknesses.** Add relevant demonstrations and rehearse earlier
   examples. Keep held-out evaluation states out of corrective teaching.
5. **Continue learning.** Retain episodes and checkpoints, test candidate updates,
   and evaluate playing strength independently of teacher agreement.

Connect Four imitates a depth-4 teacher, then practices with corrective teaching
and rehearsal. Pong imitates a teacher that uses simulator velocity; the policy
must infer motion from pixels and its trace. Pong practice combines
advantage-weighted nudges, teacher corrections, and rehearsal. Its shipped run
retained the imitation checkpoint because practice did not improve validation
win rate. More experience supplies learning opportunities; it does not guarantee
that every update improves play.

## Learning from an outcome

An action's advantage can weight its local nudge:

```python
learner.step(recorded_drives, actions, weight=advantages)
```

Positive advantage favors the action; negative advantage discourages it. These
are recorded observations and causal memory clamps from the rollout. Construct
advantages from measured outcomes and the chosen baseline; do not substitute
imagined rewards for measured experience without identifying that model-based
training explicitly. [Learning](learning.md) gives the equilibrium assumptions
and finite-nudge limits. [Reward](reward.md) describes `ActorCritic`, an additional
composition with a critic and eligibility traces for delayed feedback.

Reward shaping changes what is optimized. Report actual wins, losses, and draws
as well as shaped training return. For Pong, a paddle can return balls repeatedly
without winning a point, so cap rallies and count those draws explicitly.

## Comparing futures

Planning is a useful architecture built around the policy. Connect Four's browser
uses four-ply search through supplied rules and a supplied threat heuristic;
learned scores break ties. Evaluate the raw policy, search alone, and their
combination with the same search budget. Search contributes most of the measured
playing-strength gain in the historical example.

The separate [deliberation pattern](deliberation.md) demonstrates independent
imagined branches, a learned terminal evaluator, and learning from real outcomes.
Neither example learns a transition model. Include the cost and information
provided by that model when comparing architectures.

## Evaluate and deploy

Use fresh seeds and fixed opponents, report wins/losses/draws, and distinguish
validation used to select weights from the final test. Compare opponents of
several strengths. A high win rate against a scripted opponent does not establish
a human win rate.

For learning-rule comparisons, match observations, demonstrations, reward,
training budget, and action-selection work. The historical Pong run's reward-only MLP
control has different observations and reward/discount settings and receives no
teacher demonstrations. It is an additional control, not a matched efficiency
comparison. Connect Four's MLP controls do not receive the patch net's extra
practice corrections.

The static pages settle exported weights in JavaScript. Run `python serve.py pong
--learn` or `python serve.py connect-four --learn` from that historical checkout to save
completed games and train locally. Retention checks on rehearsal examples limit
some regressions but do not guarantee a monotonic win rate. See
[Training a player](https://github.com/muellerberndt/cadence-examples/blob/7302f2af3dc0638bbafd1da1446ed96ffabaa9dd/TRAINING.md)
for persistence and [browser pages](pages.md) for deployment.
