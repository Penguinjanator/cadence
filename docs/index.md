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
7. [One ongoing brain](continuous.md): continuous interaction, repetition, salience and
   persistent synaptic consolidation.

## Mechanisms

| Guide | What it covers |
|---|---|
| [Memory](memory.md), [consolidation](continuous.md) | Synaptic records, transient/persistent weights, salience, interference, pattern separation and resets |
| [Content prototypes](content_memory.md) | Content-selected associations, novelty, stable coordinates and capacity limits |
| [Rehearsal](replay.md) | Bounded past observations, explicit labels and extra training work |
| [Sequence readback](sequence.md) | A causal content cache and bounded fading traces |
| [Certificate](certificate.md) | When settling is a contraction: row mass, slope bound, the error bounds and the warm-start budget |
| [Learning](learning.md) | Free and nudged phases, gradient conditions, every knob |
| [Reward](reward.md) | Eligibility traces, a critic and the prediction error |
| [Task recipes](tasks.md) | Input encoding, pattern targets, streams, several learners |

## Build and measure

[Measured comparisons](comparisons.md) · [Browser pages](pages.md) · [Backends and timing](backends.md) ·
[Protocols](protocols.md) · [Receipts](receipts.md) · [API reference](api.md)

## Examples

Five [official examples](https://floatingpragma.io/cadence-examples/) apply these
patterns: an eye and drawing arm, a C. elegans habitat, a composer, a fly-inspired
forager and Connect Four. Their code, evidence, local launcher and a catalogue of
reusable cortices are in [cadence-examples](https://github.com/muellerberndt/cadence-examples).
