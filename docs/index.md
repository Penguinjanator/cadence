# Cadence documentation

Cadence builds bounded observer-like software patches with local state, ports,
readback, records, and feedback. The core loop reads a state, measures a
discrepancy, corrects it through declared seams, and keeps what the task needs.

## Start here

1. [Install Cadence](../README.md#install), then run the short memory example.
2. [Quickstart](quickstart.md): settle a three-owner circuit, remove a relay,
   check the residual, fit two labels, and save a checkpoint.
3. [Concepts](concepts.md): learn what an owner, seam, clamp, and state mean.
4. [Task recipes](tasks.md): choose inference, memory, learning, or reward.

## Choose a mechanism

| Guide | What it covers |
|---|---|
| [Memory](memory.md) | Direct key/value ports, residual writes, interference, and resets |
| [Learning](learning.md) | Free/nudged phases, gradient assumptions, and numerical checks |
| [Reward](reward.md) | Eligibility traces, a critic, and reward prediction error |
| [Small core](condense.md) | State lifetimes and why the operations stay separate |
| [Small component experiments](child.md) | What a few tested compositions add |

## Build and measure

[Games](games.md) · [Embodiment](embodied.md) · [Browser pages](pages.md) ·
[Backends and timing](backends.md) · [Protocols](protocols.md) ·
[Receipts](receipts.md) · [Comparisons](differences.md) · [API reference](api.md)

The [public examples](https://github.com/muellerberndt/cadence-examples) cover digits,
associative recall, Connect Four, Pong, changing memory, and circuit interventions.
Their tutorials state the data, controls, selection budgets, and source version
for each receipt. Historical receipts describe their original source snapshots;
rerunning an example produces a separate result.
