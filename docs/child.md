# The child's brain out of simpler components

A child learning games keeps a permanent world model ("these buttons move the character",
"evade or kill the enemies"), sees one frame at a time, keeps each moment reverberating for
seconds before it vanishes, and changes the world model only when a moment carried a
positive or negative signal, in proportion to it. This note says which component of the
core carries each of those, what the small experiments in `tests/test_child.py` measured on
2026-09-14, and what the receipts from the NES player say about the parts that still fall
short.

## The four capabilities and their components

| the child | the component | the experiment | the number |
|---|---|---|---|
| a permanent world model | the seams: `Settlement` weights moved by `Learner` on the free/nudged contrast, or by `ActorCritic` on the trace times the dopamine | every rung of the examples ladder; the NES player's imitation | — |
| one frame, and a moment that reverberates for seconds then vanishes | an owned state clamped beside the settlement: `Echo` (the trace of the hidden equilibria), or `Afterglow` (the trace weighted by what changed since the last moment, of the hidden owners or of the input owners themselves, an afterimage) | predict the symbol seen one moment ago from one moment's input while twelve owners of background are always on | echo 0.25 (chance), afterglow of the interpretation 0.30, afterimage of the picture unfocused 0.58, focused **1.00** |
| the moment before still in the settlement itself | a warm start with a capped repair: `settle_batch(drive, state=previous, steps=k)` | the state after three steps of repair from the previous equilibrium, against a cold settlement | distance from the moment before 0.99 (partial) against 1.28 (cold and full): a capped repair carries the past, a full one forgets it |
| a press whose consequence comes later | the eligibility trace of `ActorCritic` (`lam`, `gamma`) | a two-context bandit paid three moments after the press, with two moments of other presses between | hit rate **1.00** with the trace at 0.9, 0.47 without |
| plasticity only for a positive or negative signal, in proportion | the dopamine centred on its running level per stream, in the reward's own units, nothing within the usual, capped (`dopamine_center`, `center_per_stream`, `center_scale=False`, `dopamine_floor`, `dopamine_cap`), no critic | after thirty moments of a reward of one: the same reward again, a missing reward, a reward ten larger | seam change **0.000**, 0.082, 0.279 |

What the experiments settle: the memory of the moment before that works is not a trace of
the interpretation but a fading picture of what changed on the retina, with the change
weighting doing most of the work (0.58 to 1.00); the trace does credit a press three
moments on; the dopamine can be made quiet for the usual and proportional for the rest with
the components in the core, no critic needed. What they leave open is the coupling: today
the afterimage is a clamp beside the settlement and the eligibility trace a separate array,
where in the child what is still ringing is what gets written when the signal comes.

## What the NES player says the parts are still missing

The receipts are in `cadence-gamer/experiments/e6_transfer`.

- With the dopamine centred and scaled to unit size on every decision (the recipe that raised
  Mario from 626 to 970 pixels with the recordings as an anchor), a brain dropped on 1942, a
  game it never watched, plays worse than itself not learning at every rate tried, and the
  anchor halves its press rate on a game played by holding fire. The scaled centre reads a
  surprise of any size as root two (the surprise inflates its own scale), so a plane and a
  level cleared moved the seams alike.
- With a dopamine floor the loss is delayed by a quarter of the run and not stopped; with
  the dopamine in the reward's own units and a cap of ten, the run starts at the untrained
  level, falls and half recovers. The child's rule proper (no critic, quiet for the usual,
  proportional otherwise, nothing replayed) is the receipt in flight.
- On sixteen games at once, one dopamine level shared by every stream lets the richly paid
  games push Mario's presses down until it stops running; with a level per stream the press
  rate holds and the brain gains where it had little (the never-seen Mario 8-1 from 336 to
  595 pixels, Ghosts 'n Goblins from 533 to 1,700 points) and loses where it had most
  (Mario 1-1 from 735 to 548).

## The order of the next work

1. The afterimage in the player: one frame on the retina and a fading, change-weighted
   picture of it beside, against the frame-and-previous-frame retina, on the fifteen-game
   library (in flight on a four-GPU instance: `gamer/run_memory.sh`).
2. The child's rule as the self-play recipe once its 1942 curve is in.
3. The coupling: the eligibility trace weighted by the afterimage, so that what is still
   ringing is what the dopamine writes through; a small experiment first, a rare cue that
   predicts a later reward against a static background.
4. The settlement as the memory in play: no reset between moments, a capped repair from the
   previous equilibrium; the trainer's stream mode needs the warm start threaded through.
