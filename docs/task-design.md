# From a task to patch ports

Cadence aims to reuse local overlap repair and equilibrium detuning across
tasks. Scaling a network is useful only when its observations, actions and
learning signal preserve the distinctions the task requires. The current
library does not establish that arbitrary tasks become solvable by increasing
width, data or compute.

Design a [continuing life](world-model.md): bootstrap, ordinary use, witnessed
disruption and local correction in the same acquired brain. Keep isolated
classification and calibration as mechanism controls. Measure retained and new
behavior after repair, alongside stable-use work and repair work; do not replace
the brain or bypass its memory between observations to stand in for this test.

Every issue and plan declares whether it tests a bounded component, integrated
continuing behavior, or infrastructure. State the applicable equilibrium
boundaries, simpler control and preservation gates. Infrastructure can explain
applicability without requiring an unrelated whole-brain demonstration. Use the
[existing capability owners and canonical principle](../CONTRIBUTING.md#preserve-the-capable-foundation).

## Start at the boundary

A patch has bounded local state, declared ports, readback and a repair rule.
Give each external port a meaning, unit, availability condition and source.
Distinguish an absent observation from an observed zero. Distinguish a goal,
an executed action, a measured consequence and a private prediction even when
their arrays have the same shape. The application supplies these semantics;
array dimensions alone cannot enforce them.

First supply correct action records directly to the environment or instrument.
Check what actually happens. Then compare the learner's proposed actions using
the same boundary. This separates a learning failure from an interface that
cannot realize the requested behavior.

For a sampler, an event can expose sample identity, onset time, gain and
duration. Those are playable controls. Spectral brightness and onset density
are observations of sound; mapping them directly to pitch or repeated drum
hits is an additional instrument design. It is not an acquired inverse model.
Known action demonstrations are also not a transcription of those actions
from a musical mix. That inference requires its own evidence.

The same distinction applies to robots, games and interactive software: a
forecast of an observation is not automatically an executable action.

## Check which differences survive encoding

Use two cases whose distinction matters. If the encoder maps them to the same
ports, a deterministic learner cannot recover that distinction from those
ports alone. More context or a learned prior can support a guess, but cannot
make the missing observation unique.

For example, consider one event in each sixteenth-note interval. Times
`j + 0.25` are equally spaced. Times `j + 0.25` for even `j` and `j + 0.75`
for odd `j` alternate between long and short gaps. Both have the same event
count, occupied intervals and unsigned distance to the nearest grid point.
An unsigned timing-spread summary cannot distinguish them. This is a
counterexample for that timing summary, not a claim that all audio features
of the two rendered waveforms are identical.

Use explicit signed event offsets when the task requires them. Quantized
patterns can still have musical groove; preserving microtiming alone does
not establish rhythmic quality, correct instrument choices or arrangement.

## Match the update to the behavior

`Brain.lesson` teaches the current observation with the declared motor labels; reward
feedback instead belongs to the preceding executed action. Slot sizes, event boundaries,
units and the information available before each action must match the task. Use the
[teaching guide](teach.md) and [reward guide](reward.md) to select the actual update.

Check directly cued acquisition before interpreting a failure as missing context. If a
relation is individually learnable but competing experience changes it, measure the
surviving memory, read, free answer and update direction separately. A local numerical
residual and a task prediction error have different meanings. Preserve the frozen task,
capacity, exposure and original gates while diagnosing a failure.

## Test coherent behavior over time

Sequence competence requires coherent, continuing learned context and temporal
organization to govern behavior. Sequential output and sensory feedback are
allowed. Working traces, associative memory and efference are legitimate; the
question is what behavior their acquired context supports. Next-event accuracy
and self-feeding alone are bounded component results, not evidence of phrase or
whole-sequence competence.

For an integrated sequence claim, declare the relevant context and time span,
then compare the same acquired brain with retained context, absent context and
perturbed context under matched exposure. Where relevant, test recovery after a
disturbance and counterfactual contexts requiring different continuations. Keep
the simpler control, earlier skills, pending continuation, refusals and work in
the acceptance gates. These comparisons must test the claimed behavior, rather
than only its aggregate next-event score.

Separate equation residual from musical surprise, structural coherence and task
success. State which coordinates are live in the solve and which are held:
default `Brain.act` qualifies a present state with held trace and memory input,
not a joint future trajectory. This does not require a new phrase module or
whole-song-at-once generation. Check existing state, information and experience
first; any added mechanism must use local repair in the same coupled global
equilibrium. Adapters must not conceal answers, phase or arrangement. Supplied
context or clocks belong in the declared information budget and appropriate
controls, not in a claim that the brain learned that organization itself.

## Test learned memory relevance

The criterion is that the same acquired brain uses the same memories differently
when context changes, and experience improves that selection. Keep three
questions separate: does information survive (retention), does the current query
select what matters (read), and does an outcome update the appropriate earlier
actions (credit)? A longer trace or a larger store alone does not establish
learned selection. This capability is not yet demonstrated.

First require acquisition on a directly cued control with the same answer
relations. If that fails, diagnose encoding and learning before attributing the
failure to context or proposing another memory mechanism. Start with existing
System 1 state and memory. Then use a bounded component assay:

1. Acquire several facts in one brain. Save it and branch copies with identical
   memories and parameters; change only the query or legitimate task context.
   Require the answer to change with which retained fact is relevant. The
   adapter supplies observations and queries, never the relevant record index.
2. Perturb relevant and irrelevant memories separately under matched exposure.
   Relevant changes should affect the answer; irrelevant changes should not.
   Compare before and after learning the query relationships to establish that
   experience improved selection, rather than a supplied key deciding it.
3. Test fresh bindings, presentation orders and held-out role combinations so
   a fixed cue-to-answer table or recency rule cannot pass. Keep the learned
   query semantics available; disclose which distinctions training supplies.
4. Add competing memories and switch the question in the same acquired life.
   Measure earlier skills, revision of obsolete facts, recovery and saved
   continuation. Use uniform-random, frozen-copy and simple learners with the
   same information, including a control for the strongest task shortcut.

Declare live and held coordinates. Learned selection must participate through
local repair in the same coupled equilibrium; a separate answer vote after
settlement is not evidence of this property. Qualification checks the equations,
while the comparisons above check behavior. Count storage and all acquisition,
queries, memory reads/writes, settling, replay, interventions and refused work.
Freeze the task, capacity, work limits and gates before testing fresh founders.
Preserve existing capability gates and failed attempts. A passed component assay
would support the next sequence test; it would not establish sentence formation
or authorize a default change. These tests do not prescribe a new module.

The initial [named-fact selection instrument](../benchmarks/relevance/README.md)
implements the acquisition diagnostic on the existing graph: balanced facts,
late queries from identical saved state, trace erasure and frozen/random
controls. It covers finite acquisition with reset trials; the generalization,
interference and continuing-life gates above remain separate work.

## Close the action and consequence loop

For an acting system, test this full sequence:

1. Observe the current situation and retain the relevant context.
2. Form a private action proposal under a sustained intention.
3. Predict the proposal's consequences using relationships learned from
   actual action/consequence records.
4. Execute the selected action and measure the actual consequence.
5. Repair the discrepancy and test earlier skills after the update.

`Brain.imagine` evaluates supplied observation sequences with a private trace and read-only
durable memory. It does not predict the body's next observation or learn an environmental
transition model. Supply a tested transition model as an explicit control when studying
planning, and account for its extra information and work. Replay a selected action through
the actual body before treating the desired consequence as acquired competence.

## Test coordination with the same patch rule

A possible shared workspace is a bounded patch that reads proposals and
mismatches from other patches and feeds back a common intention or constraint.
Specialized patches can still use the same overlap and detuning operations.
Recursively exposing summaries of this process is a research hypothesis about
coordination, with no claim about consciousness.

Before treating such a workspace as necessary, compare the same task with
and without useful summary feedback. Count its state, ports and work. Include
a matched recurrent model, disconnected feedback and shuffled feedback.
Require an improvement in a named behavior, such as maintaining a motif while
changing accompaniment, or revising a strategy under new constraints. Activity
in the new patch alone does not establish useful coordination.
For specialization, selectively perturb a region and measure both its distinct
contribution and the continued coordination of the whole system.

Optional observing regions return signals inside the same neural graph, while private
imagination keeps its branch separate. Neither establishes an automatically specialized
workspace. The [anatomy guide](anatomy.md#write-a-cortex) describes the supported regions;
the [contracts](contracts.md) delimit their numerical guarantees.

## Scale after the joint test

A useful small test checks acquisition, retention after competing experience,
private prediction, successful action and the task's actual quality together.
Use independent tasks or withheld cases for generalization. Larger runs should
report quality against data, state, time and compute, with simple controls.
Formal consistency and numerical convergence are necessary checks on the
implementation; neither substitutes for that behavioral evidence.

Use [evaluation](evaluate.md), [the continuing brain](start.md#one-continuing-equilibrium-brain)
and [the numerical contracts](contracts.md) when turning these criteria into an application.
Musical examples illustrate the criteria; they do not define a separate learning architecture.
