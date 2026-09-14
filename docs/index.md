# Cadence documentation

Cadence builds patch nets: owners with local state exchange messages over declared
seams and repair their state until the net settles. Records, traces and local
learning keep what a task needs.

## Start here

1. [Install Cadence](../README.md#install) and run the short memory example.
2. [Quickstart](quickstart.md): settle a circuit, cut a relay, check the residual,
   fit two labels and save a checkpoint.
3. [Concepts](concepts.md): owners, seams, settlement and the three state lifetimes.
4. [Patterns](patterns.md): assemble regions, bodies, records, futures and monitors
   into a brain.
5. [Function map](biology.md): from a nervous-system function to its wiring and pattern.

## Mechanisms

| Guide | What it covers |
|---|---|
| [Memory](memory.md) | Key/value records, residual writes, interference and resets |
| [Learning](learning.md) | Free and nudged phases, gradient conditions, every knob |
| [Reward](reward.md) | Eligibility traces, a critic and the prediction error |
| [Task recipes](tasks.md) | Input encoding, pattern targets, streams, several learners |

## Build and measure

[Browser pages](pages.md) · [Backends and timing](backends.md) ·
[Protocols](protocols.md) · [Receipts](receipts.md) · [API reference](api.md)

## Examples

Six [interactive websites](https://floatingpragma.io/cadence-examples/) apply these
patterns: an eye and drawing arm, a teachable mouse, a C. elegans habitat, a
fly-inspired forager, changing memory and Connect Four. Their code, evidence and
local launcher are in [cadence-examples](https://github.com/muellerberndt/cadence-examples).
