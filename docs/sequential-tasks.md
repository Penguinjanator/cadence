# Sequential tasks and prerequisite chains

A continuing brain that learns a single step well, food in one move, often fails a chain of
steps: gather wood, gather stone, make fire, warm up. The usual sign is a policy that settles
on the one action that pays most often. The measurements below come from the
[prerequisite-chain nursery](../benchmarks/chain/README.md), the
[key-door nursery](../benchmarks/keydoor/README.md) and the
[competing-skill ring](../benchmarks/competing/README.md).

## What breaks a chain

**Hidden state.** If the observation does not show what the creature holds, the right action
depends on history and the chain is a short-term memory task. In the chain nursery with the
stage hidden, no learning arm beats the memoryless ceiling (62 completions per thousand
moments, uniform over the chain's actions): stages with identical observations cannot be told
apart, and the policy settles on one action or keeps sampling. The brain carries history in
its working trace ([memory](memory.md#working-trace)); give the last module enough width
when history has to carry the state.

**Late pay.** Credit for an action across intervening choices travels through the actor's
eligibility and the critic. With pay only at the completion of a four-step observed chain,
the greedy policy completes at 42 per thousand moments, equal to uniform random, while the
sampled policy reaches 77 and the greedy answer is right at 0.67 of the stages: the early
steps are learned and the whole chain is not. The key-door nursery acquires at delays of two
and five choices and fails at ten; at five, only 4 of 10 lives acquire with the composed
critic rate, against 10 of 10 with a raised one (item 2 below).

**Readout saturation.** Near saturation a finite update produces little contrast between
actions, and exploration cannot escape the latch; the key-door chamber records it under
punished exploration. Read the motor activations and the margin between the chosen and the
next action before changing a rate ([reward traps](reward.md#traps-with-their-measurements)).

**Loud carried context.** In a chain the next action depends on what the brain sees now. A
carried interpretation fed back too strongly makes it depend on the moment before. The
founder `working_memory_scale` of 1.2 lets the present decide at birth, and learning
strengthens the carry where the past predicts the outcome ([memory](memory.md#working-trace)).

**Continued learning.** A policy acquired at the nursery's actor step can erode under its own
continued learning, most clearly when the later pay is signed and noisy
([#169](https://github.com/muellerberndt/cadence/issues/169)). In the
[competing ring](../benchmarks/competing/README.md), under signed noisy pay the loss reproduces
at every actor step, because the noisy pay overwrites the associative store's one-shot record
of the chosen action's reward (`memory_rate` 1.0; an averaged record at 0.2 halves the loss).
More drilling past acquisition moves the policy instead of consolidating it.

## The pattern the measurements support

1. **Observe the chain's state and shape the pay.** Expose inventory as senses (has wood,
   has stone, has fire) and keep a hidden-inventory condition as a separate memory test. Pay
   executed actions through `Brain.live`; label teaching of a sequence is a separate contract
   ([#121](https://github.com/muellerberndt/cadence/issues/121)). With the state observed and
   pay at every correct step, the chamber's brain completes 204 per thousand moments against a
   tabular learner's 250 and answers 0.92 of the stages right. Backward lessons (episodes that
   start at a random stage with the prerequisites granted) reach 33: the granted completions
   meet the need, the brain calms and learns less. Disclose supplied shaping and verify the
   full chain from stage zero under end-only pay.
2. **Raise the critic rate when credit spans about five choices.** The key-door nursery runs
   `actor_eta_critic=5.0`; at 10 the values run high and the creature latches the opposite
   way. Select the rate on the task with the composed 0.3 as control.
3. **Average the associative record where noisy pay follows acquisition.** In the competing
   ring, actor steps of 0.003 and 0.001 retain no more than the nursery's 0.03 and do not
   prevent the loss under signed noisy pay; a brain composed with an averaged associative
   record (`memory_rate=0.2`) acquires as before and loses half as much
   ([memory](memory.md#associative-store-the-hippocampus)).
4. **Choose `live` or `step` after the caregiver leaves.** `live` learns only while aroused,
   by a surprising outcome or by pay below what life usually brings; `step` learns from every
   outcome ([routine and repair](continuous.md#routine-and-repair-live)). Which path erodes a taught
   chain depends on the founder. Where most surprises are punishments, as in the melee,
   `live` alone does not hold the skill; lower `memory_rate` instead. Teaching by demonstration
   and correction goes through the episodic memory ([episodic](episodic.md)).
5. **Read the greedy policy stage by stage.** Whole-chain success hides a stalled stage.
   Read the greedy answer, the right action's probability and its margin at every stage, and
   the sampled policy's completion rate beside the greedy one: a frozen greedy copy can
   answer one action in every situation while the sampled probabilities move with the
   situation.
6. **Declare the probe schedule and freeze the budget.** Longer training can degrade an
   acquired skill. Probe saved copies at a declared schedule (the chain nursery probes every
   2,000 of 20,000 moments), select the training budget in development, freeze it before
   testing fresh founders and disclose any checkpoint selection.

## Settings involved

The composed founders ([brain](brain.md#defaults-and-expert-overrides)) are `actor_gamma`
0.9, `actor_lam` 0.8, `actor_eta` 1.0 and `actor_eta_critic` 0.3; their meaning is in
[reward](reward.md#eligibility-traces-and-dopamine). The instruments run their own operating
points. These are chamber settings, not library defaults:

| Chamber | Operating point | Controls |
| --- | --- | --- |
| Prerequisite chain | modules (48, 24), `actor_eta` 0.03 (bias 0.003), `actor_lam` 0.6, `actor_gamma` 0.95, `actor_eta_critic` 5, temperature 0.3 | a tenth of the step; uniform random, memoryless ceiling, frozen newborn, tabular Q-learning |
| Key-door | `actor_eta_critic` 5.0, `actor_lam` 0.95, `actor_gamma` 0.95, `actor_eta` 0.1, selected on development seeds | the composed 0.3, 0.8 and 0.9 |
| Competing ring | the chain founder above; arms retune `actor_eta` to 0.003 or 0.001 at the ring's door, or compose with `memory_rate=0.2` | the nursery's 0.03 and `memory_rate` 1.0 |

Set them through `Brain.compose(..., actor_eta=0.03, ...)` or retune a running life
([retune](brain.md#retune-the-same-life)).

## Work in progress

- A measured memory horizon for a hidden chain state: [#84](https://github.com/muellerberndt/cadence/issues/84).
- How carried context persists and is revised across a task: [#189](https://github.com/muellerberndt/cadence/issues/189).
- A mechanism that keeps an acquired skill at the nursery's own step: [#169](https://github.com/muellerberndt/cadence/issues/169).
- Delayed credit beyond five choices: [#111](https://github.com/muellerberndt/cadence/issues/111).
- Conventional competitors (online Elman and GRU actor-critics) on the chain: [#184](https://github.com/muellerberndt/cadence/issues/184).
