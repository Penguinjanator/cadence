# Cadence documentation

Cadence builds bounded observer-like software patches with local state, ports,
readback, records, and feedback. The central loop is: read the current state, measure
a discrepancy, correct it through declared seams, and keep the result for the next
moment. Public examples supply executable comparisons and evidence receipts.

| start here | purpose |
|---|---|
| [concepts](concepts.md) | owners, seams, state, feedback, and the limits of the analogy |
| [memory](memory.md) | direct key/value ports and one-line residual writes |
| [learning](learning.md) | free/nudged settlement, exact assumptions, and numerical checks |
| [quickstart](quickstart.md) | wiring, a held-out protocol, a control, and a receipt |
| [API](api.md) | current public names and module helpers |

## Compose only what the task needs

| operation | implementation | boundary |
|---|---|---|
| repair interacting state | `Settlement` over a `Wiring` | a capped run need not reach equilibrium |
| learn slow responses | `Learner` free/nudged endpoint contrast | conditional equilibrium gradient, finite-step bias |
| retain recent activity | `Trace`; `Echo` and `Afterglow` are defaults | fading history, not a permanent record |
| store and revise observations | `FastSeams` | fixed capacity, supplied keys, interference |
| assign reward credit | `ActorCritic` trace and critic; `Valence` | a temporal-credit estimator, not a solved general RL system |

The [small-core rationale](condense.md) explains why these operations remain distinct.
The [small component experiments](child.md) test what composing them adds.

## Build and measure

[Task recipes](tasks.md) · [Games](games.md) · [Reward](reward.md) ·
[Embodiment](embodied.md) · [Browser pages](pages.md) · [Backends and timing](backends.md) ·
[Protocols](protocols.md) · [Receipts](receipts.md) · [Comparisons](differences.md)

Runnable examples live in [cadence-examples](https://github.com/muellerberndt/cadence-examples):
digits, associative recall, Connect Four, Pong, changing memory, and circuit interventions.
The intervention example compares a supplied local model with a trained MLP, fixed-depth
graph unrolling and an independent solver. It checks new wiring and ablations without
retraining. The memory example
includes a transformer that learns its training-length task and exact lookup controls;
its length-extrapolation gains and correlated-key failures are both retained. Historical
examples keep their original source-bound receipts, which are distinct from new results.
