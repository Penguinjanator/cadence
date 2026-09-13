# Condensing cadence: the plan for 0.8

Bernhard, 2026-09-14: everything condensed into very, very simple and efficient basic
elements; nothing optional in the sense of add-on modules; minimize, optimize, document;
then a new set of examples. This is the plan, written before the work so the work can be
checked against it. The receipts it rests on are in `docs/child.md` and in the NES player's
experiments (`cadence-gamer/experiments`).

## Five elements and one step

| element | what it is | today |
|---|---|---|
| **owners and seams** | the state and the world model: a wiring with named ranges | `Wiring`, `Region`/`Projection`/`Constitution`/`grow`/`mutate`/`evolve`, `layered`/`embedded`/`stateful` |
| **settlement** | repair to rest from a clamp, from rest or from the previous state, capped or run out; perception when run out, memory when capped | `Settlement`, `GradedRule`, `Adaptation`, `Nudge`, `SettledState` |
| **contrast** | free against nudged at every seam: what a nudge would change; the local signal of every kind of learning | inside `Learner.update` and `ActorCritic.learn` |
| **trace** | a quantity decaying across moments; over the contrast it is the eligibility, over a range's activation weighted by change it is the afterimage, over the reward it is the usual level | `Echo`, `Afterglow`, `FastSeams`, the eligibility arrays of `ActorCritic`, `delta_mean`/`delta_var` |
| **valence** | the reward less its expectation, quiet within the usual, in proportion, capped; the expectation learned by a critic of the same kind as everything else | `ActorCriticConfig.dopamine_*`, `center_*`, `ValueNet`, the critic inside `ActorCritic` |

and one step: seams move by trace times valence (imitation is the same step with the nudge
toward the target and the valence one).

In 0.8 these are five classes and a function: `Wiring`, `Settlement`, `contrast`, `Trace`,
`Valence`, and `Learner.step`; an `Agent` composes them for a stream of moments (act,
learn) and is what the player and the games use. The critic is an `Agent` part, not an
option: without it the child's rule on 1942 fell (589 to 280 points a game) where with it
the same rule rose (605 to 1,482).

## What each public name becomes

From an inventory of every Python file in the four repos that use the core
(cadence-examples, cadence-gamer, cadence-paper, cadence-author; 2026-09-14):

| today | used by | 0.8 |
|---|---|---|
| `Wiring`, `layered`, `Constitution`, `Region`, `Projection`, `grow`, `evolve` | examples, gamer, paper, author | kept, under `cadence.wiring` |
| `embedded`, `stateful` | author, paper | kept as builders in `cadence.wiring` |
| `mutate` | none | kept inside `evolve` only |
| `Settlement`, `GradedRule`, `learning_rule`, `Adaptation` | all | kept; the rule is the settlement's, `learning_rule` stays the default maker |
| `Nudge`, `SettledState` | internal | internal |
| `Learner`, `LearnerConfig` (60 fields) | all | `Learner` with a config of: rate and bias rate, beta, temperature, tolerance, step caps, and one adaptive option (the bias-corrected momentum and RMS that Pong needed); `consolidate`/`restore` go (E4: no gain) |
| `ActorCritic`, `ActorCriticConfig` (65 fields) | gamer, paper | `Agent` = settlement + eligibility `Trace` + `Valence` + critic `Learner` + an action code; a config of: trace decay (one number in place of gamma and lam), rate, the valence's level rate, floor and cap, the critic's rate |
| `Bins` | gamer, paper | kept as the action code (a softmax draw per dimension) |
| `Population`, `DiscreteCode`, `actor_critic_wiring`, `Seams` | paper only, or none | dropped |
| `Rehearsal`, `RehearsalConfig`, `ValueNet`, `ValueConfig` | gamer, paper | dropped (E4: the clipped rehearsal fails; the critic is a `Learner`) |
| `DreamActorCritic`, `DreamConfig`, `SleepConfig` (`dream.py`, 529 lines) | paper only | dropped from the core; the paper pins its own version |
| `Echo`, `Afterglow` | gamer, paper, author | one `Trace(decay, focus, source, target)`: focus 0 is the Echo |
| `FastSeams` | paper, author | `Trace` over pairs (`pairs=True`), the same class |
| `dopamine_center`, `center_per_stream`, `center_scale`, `dopamine_floor`, `dopamine_cap` | gamer | `Valence(level, floor, cap)`, per stream always, in the reward's own units always (the scaled centre reads a surprise of any size as root two; the units form is the one that rose on 1942); the Mario self-play re-run on it is the gate before the scaled form is removed |
| `Receipt`, `Source`, `fetch`, `manifest`, `canonical_json` | all | kept under `cadence.receipts` |
| `canonical_sha256`, `source_manifest` | none | dropped from the top level |
| `conformance` | examples, author | kept (the owner-by-owner check of every page) |
| `Protocol`, `Row`, `Ledger`, `select_gain`, `shuffled`, `evaluate_predicate`, `settle_owner_by_owner` | paper (57 files), the core's own examples | kept under `cadence.protocol`, out of the top-level story |
| `save`, `load` | examples, paper | kept: `Learner.save`/`load`, `Agent.save`/`load` |
| `available_backends` | author | kept |

The top level goes from fifty-three names to about fifteen.

## Optimizations, each with its number

- The afterimage on the device. Today it is host-side NumPy over the batch times the retina
  (256 by 7,056 a moment); the afterimage brains trained at half the speed of the plain
  one (7,000 against 13,400 updates in the same time on an A10G).
- The trace and the step on the device for the imitation `Learner` as they are for the
  agent; one settlement per moment shared by act and learn (already); warm starts across
  moments in the trainer's stream mode (E1 measured the step cap costs nothing in gradient
  direction; a warm start halves the steps).
- The fused contrast kernel on the device path.
- The measure for all of it: the player's update time at 512 owners on the library, and
  the examples' wall-clock, before and after.

## Documentation

`docs/index.md` restructured around the five elements, one page each with its receipt
(the child tests, the ladder); `api.md` regenerated from the fifteen names; `CHANGELOG`
0.8 lists every dropped name and its replacement.

## Gates

1. The core suite, `tests/test_child.py` included, green on 0.8.
2. The four example rungs re-run on 0.8 with their receipts verifying or improving; then
   the new set of examples.
3. The player: the one-frame afterimage brain's 0.579 bits at 512 owners reproduced by
   the 0.8 `Learner`; the 1942 curves and the sixteen-game self-play reproduced by the
   0.8 `Agent`.

## Order

1. 0.7.1 first: release what exists (the afterimage, the salience, the valence options)
   so the public library and the public examples agree (PyPI is at 0.4.1 against a repo
   at 0.7.0).
2. 0.8 on a branch: the five elements and their tests; the examples ported and re-run; the
   player ported and measured; the docs; the release; then the new examples.
