# Patch nets and networks trained by backprop

Both learn weighted responses from data. Cadence offers a different implementation of
inference and credit, with explicit retained state. That difference is useful only when
it fits the task and its cost is measured.

| property | ordinary feed-forward model | Cadence settlement |
|---|---|---|
| inference | evaluate layers | repeatedly repair state over declared seams |
| state between inputs | supplied by a cache or separate memory when needed | retained explicitly in a state, trace, or fast-memory patch |
| training credit | reverse-mode differentiation | free/nudged endpoint contrasts |
| differentiation storage | activations or recomputation checkpoints, plus optimizer history | phase endpoints and optimizer history; traces when used |
| exact gradient conditions | differentiable executed computation | stable smooth equilibrium branch, symmetric effective recurrent weights, converged phases, vanishing nudge |
| work | forward/backward operations | all free and nudged repair steps plus the update |
| memory capacity | model and context dependent | model and fast-store dimensions dependent |

A transformer can also use recurrent state, external memory, local updates, or a
retrieval module. An explicit store beating one trained transformer on a structured
memory task does not show that these resources are exclusive to patch nets.

## What to measure

Separate inference, learning, and record maintenance. On an addressed lookup, one read
already computes the answer; repeated settlement can add cost without accuracy. On
interacting constraints, feedback may change the answer, and the cost includes every
repair. A warm start may help on slowly changing inputs, but comparing one unchanged
cached input with another model's fresh decision does not measure a control workload.

Compare validation-selected models on the same held-out examples and include a simple
algorithmic solver when the task has one. Count mutable records, training examples,
architecture search, parameters, wall time, and all retained failures. A receipt hash
verifies custody; it cannot repair mismatched budgets or an unlearned baseline.

## Public comparisons

The supervised public examples reach broadly similar accuracy to their MLP baselines
at greater wall-clock cost; widths and selection budgets are not always matched.
Pong's historical reward run returns about 88% of balls against 93% for REINFORCE with
Adam using the same interaction budget, reward, and evaluation protocol.

The [changing-memory example](https://github.com/muellerberndt/cadence-examples/tree/main/05_memory)
tests a more specific benefit: residual writes revise bounded records without
retraining a feature extractor. Exact lookup controls also solve it. The transformer
learns the training-length task, and correlated keys expose interference in Cadence's
store. These are task-specific results, not a general efficiency theorem.
