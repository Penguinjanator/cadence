# Concepts

## Owners, overlaps, settlement

A patch net is a stateful system whose input/output behaviour depends on its dynamics
and retained records. It contains bounded observer-like patches with local state, ports,
readback and feedback; owners exchange messages over declared overlaps. In settlement:
every owner repairs its own patch from its own state, its inbox, its clamp, and its bias.
Iterating that repair from rest is *settlement*, and the state the net rests in is what a
readout sees.

The graded rule is

    v <- v + dt * ( -v + inbox + clamp + bias - strength * a )
    inbox = sum over inbound overlaps of  gain * count * sign * exp(log_gain[pre]) * s[pre]
    s(v)  = rectified sigmoid, exactly zero at rest

The drive is absolute per contact: a hub integrates every contact it receives, and the
one global parameter is the drive of one contact. That follows the leaky integrate-and-fire
convention used for the fly brain, and it is why gains are small numbers.

## Why rest emits nothing

A plain sigmoid emits a few percent at rest. Multiplied by thousands of contacts on a hub
that leak ignites the net with no input at all. The activation is therefore re-based so
that an owner at exactly rest publishes exactly zero, in the engine's own precision. Rest
is then a fixed point of the whole net, and "nothing in, nothing out" is a testable fact.

## Why adaptation

Some graded wirings approach a fixed point under a constant clamp. Others can oscillate
or have several attractors; a constant input alone does not guarantee convergence. Adaptation adds one slow variable per owner that follows its own
activation and subtracts from its own drive. With mutual inhibition, which every real
wiring has in abundance, that is the half-center oscillator, and the net can carry rhythm.
It is still owner-local: an owner reads only its own adaptation. It is off by default, and
a lane that turns it on says how it chose the two numbers.

## Why sparsity gates the gain

Raise the gain enough and any wiring runs away: a large fraction of owners saturate and
every readout lights. In that regime a protocol passes for reasons that have nothing to do
with the wiring, and a shuffled control passes just as well. So a gain is admissible only
while the net stays sparse under the training stimuli. The cap is declared, the table of
gains tried is recorded, and the same rule is applied to the control.

## Why learning is two settlements

A patch net learns without a backward pass. It settles free, with only its input clamped;
then it settles again from that state with its output owners nudged toward the target; then
every overlap moves its own scale on the difference between what its two endpoints did in
the two phases, and every owner moves its bias on its own difference. For symmetric effective recurrent weights and converged phases on a stable smooth
equilibrium branch, the limiting small-nudge contrast gives the corresponding
loss gradient, with the parameter and temperature scaling stated in [learning](learning.md). The goal enters through one door,
the nudge, and the free phase, which is what a readout sees, never meets it. See
[learning](learning.md).

## Why float64

Owners sit on knife edges. In the fly brain, one motor neuron under one taste settled to
1.0 in one run and to 0.01 in another with the same wiring and clamp, because float32
summation order differed. Receipts are made on the float64 CPU backend for that reason.
Accelerated backends are for looking, and they come with a conformance number.

## Why a shuffled control

A protocol scored on a wiring alone measures the protocol as much as the wiring. The
control keeps everything about the wiring that is not the wiring: every count, every sign,
every owner's out-degree, every named set; only who talks to whom is destroyed. What the
wiring passes and the control fails is what the wiring predicted.

## Why receipts

Numbers in a notebook rot. A receipt is canonical JSON with its own digest, the digests of
the code and data that produced it, and enough stored readings that every pass flag can be
recomputed by the verifier. Editing any source file the receipt names invalidates it, on
purpose: a result belongs to the code that made it.

## What Cadence does not claim

Settling a measured wiring and passing held-out facts is evidence that the wiring carries
those facts under this rule. It is not a claim about biology beyond the scored predicates,
and nothing in the library ascribes experience to anything.

## Fast records and simple parts

A trace keeps recent activity; an associative matrix keeps key/value records; slow seam
weights learn a reusable response. These have different update contracts even though
they share the pattern of state, readback, discrepancy, and local correction.
`FastSeams(rule="delta")` makes this explicit: read the value already predicted, then
write only the error. [Memory](memory.md) derives the interference and capacity limits.

Simplicity is a design constraint, not a proof of optimality. Keep a mechanism when a
controlled experiment shows what it adds, and measure the work it saves. Current
Cadence experiments do not establish that nature implements this particular software
rule or that patch nets dominate every feed-forward or transformer architecture.
