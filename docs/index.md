# Cadence documentation

Cadence builds brains out of neurons that hold local state, exchange activity over
declared synapses and relax toward their synaptic input until the brain settles.
Records, traces and local learning keep what a task needs.

## Start here

1. [Install Cadence](../README.md#install) and run the small learning example.
2. [Quickstart](quickstart.md): settle a circuit, cut a relay, check the residual,
   fit two labels and save a checkpoint.
3. [Concepts](concepts.md): neurons, synapses, settling and the three state lifetimes.
4. [Design a brain](design.md): choose capacity and ports, couple regions, imitate,
   practice, correct and retain. [Patterns](patterns.md) supplies the component recipes.
5. [Function map](biology.md): from a nervous-system function to its connectome and pattern.
6. [A generic brain](patterns.md#a-generic-brain): a ready brain of standard regions that
   learns labels, pictures and rewards.

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
