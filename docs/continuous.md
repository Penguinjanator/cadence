# One ongoing brain

Use `GenericBrain.step` for an interacting agent. Each moment brings an observation,
feedback from the preceding action, and optionally a demonstration. The same recurrent
brain responds and changes its synapses throughout its life; there is no `train()` or
`eval()` switch. No replay buffer or separate training network is required.
Neurons hold bounded local state, exchange signals through declared synapses and read
back their current activity; records and feedback make the system self-reading.

```python
import numpy as np
import cadence as cd

brain = cd.GenericBrain.build(4, 4, hidden=32, episodic=True, seed=0)
observation = np.eye(4)[[0]]
action = brain.step(observation)
# The environment now executes that action and reveals the next moment.
reward = (action == 0).astype(float)
action = brain.step(np.eye(4)[[1]], reward=reward, done=np.array([True]))
# A teacher can supply the correct action for the current observation.
action = brain.step(np.eye(4)[[2]], teacher=np.array([2]))
# Saving includes the current action's eligibility, even before its reward arrives.
path = brain.save("living_brain.npz")
resumed = cd.GenericBrain.load(path)
resumed.step(np.eye(4)[[3]], reward=np.array([1.0]))
```

Reward and `done` concern the **previous action**; a teacher labels the **current
observation**. Each has one entry per batch row. Omitted reward means no reward event,
numerically zero; this is appropriate for an environment that supplies reward only
when events occur. The critic and eligibility still advance. It is not a way to skip
unknown transitions in offline data. For an ended row, supply its next episode's reset
observation; truncation can supply the previous episode's final value with `bootstrap`.
The first call has no previous action to reward. Keep stream identities until `reset()`.

## Fast activity, slow plasticity

One ongoing system still has distinct physical quantities and numerical timescales.
Neuron potentials change during settling. Synaptic weights change as observations,
demonstrations or reward supply a learning signal. Cadence alternates these updates:
weights are held fixed during each free/nudged phase, then local contrasts update them.
That separation preserves the numerical rule and its conditional gradient interpretation.
It does not claim that a biological brain runs these exact phases or that perception
and plasticity are identical processes.

An input is not automatically a correct action label. Repeating the brain's own guesses
as targets would strengthen mistakes. `step` uses actual rewards for the actor/critic
and actual teacher labels for imitation. The associative memory records only the chosen
action's observed reward; it never treats unobserved actions' predictions as evidence.
Lower-level `act`/`learn` and `Learner` remain available for experiments and custom wiring.
Use a separate instance for frozen `predict` or greedy `act` measurements.

## Repetition and salience become lasting synaptic changes

`GenericBrain.build(..., episodic=True)` adds `SynapticMemory`. It has a persistent
matrix `C` shared across streams and a transient residual `F` per stream. Each entry is
a synapse from a declared key neuron to a declared value neuron. The effective weight
is `C + F`; there is no list of remembered examples. For one unit key `k` and an actually
observed value `v`, its update is:

```text
F *= decay
alpha = min(1, consolidation * (1 + salience))
C += alpha * outer(k, v - k @ C)
F += rate * outer(k, v - k @ (C + F))
```

Defaults are `decay=0.9`, `consolidation=0.05`, `rate=1`. Repetition updates the slow
matrix even when the immediate fast response is already correct. Salience accelerates
that slow change. In the generic reward loop it defaults to `abs(reward)`; explicit
`salience=` is a nonnegative vector. This is a supplied importance signal, not a measured
neurotransmitter or a detector of subjective meaning. Positive and aversive outcomes can
both be salient, while the signed observed value determines what is remembered.

For a batch, slow updates average over writing rows using the same pre-update matrix.
Fast corrections remain per stream. `value_mask` limits learning to observed output
components. Keys are normalized; orthogonal keys preserve each other, correlated keys
can interfere. Storage capacity stays fixed, and another observed value can revise a
consolidated association. The mechanism does not guarantee recall of every experience.

```python
memory = cd.SynapticMemory(np.arange(3), np.arange(3, 5))
cue, outcome = np.array([[1., 0., 0.]]), np.array([[1., 0.]])
for _ in range(40):
    memory.observe(cue, outcome)
memory.reset(1)  # remove the transient residual; retain persistent synapses
assert memory.recall(cue)[0, 0] > 0.85
memory.observe(cue, np.array([[0., 1.]]), salience=np.array([19.]))
memory.reset(1)
assert np.allclose(memory.recall(cue), [[0., 1.]])
```

The controlled retention test leaves **0.05** of a unit target after one ordinary
exposure, **0.8715** after 40 repetitions, and **1.0** after one exposure with salience
19, after clearing all transient residuals. These are model responses, not human
retention rates. [Executable tests](../tests/test_continuous.py) also cover distraction,
correction, unseen value components, ongoing reward learning and checkpoint recovery.

## What changes in the wiring

Plasticity changes connection strengths, including previously zero weights. The declared
key/value contacts and neuron counts stay fixed. Adding/pruning anatomical connections
is a separate structural mechanism, and is not necessary to make a lasting change in
these synapses. This implementation does not model dendritic growth, protein synthesis,
biological synaptic tags or automatic transfer into the association cortex's weights.
The policy's own plastic weights continue learning from contrasts and eligibility.

Biological experiments motivate separating transient changes from their persistence:
repeated stimulation can establish lasting synaptic potentiation, and dopamine can
modulate spine plasticity in a restricted time window. The equations above are an
engineering model, not a reconstruction of those cellular processes.
[Frey and Morris (1997)](https://pubmed.ncbi.nlm.nih.gov/9020359/),
[Yagishita et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/25258080/).

## Reset, save and cost

`brain.reset()` clears the current neural/eligibility state while retaining its memories.
`brain.hippocampus.reset(batch)` clears transient residuals and keeps consolidated
synapses. `brain.hippocampus.clear()` explicitly erases both. Changing memory batch size
also clears transient residuals, while the shared consolidated matrix survives.
Unlike independent `FastSynapses` streams, these streams share long-term knowledge.

`GenericBrain.save/load` saves both memory timescales, policy, critic, optimizers,
working activity, random generators, and pending-action states. Resume the same row
identities and supply the pending action's actual outcome once. The environment/body
must be saved separately. Old generic checkpoints retain their original fast-memory rule.
`parameters()` includes the consolidated matrix; per-stream residuals and eligibility
are additional storage. Persistent memory costs `key_width × value_width` numbers,
plus the same amount per stream for effective fast weights. Reads do not consolidate
or decay memory; only a new observation advances its update clock.
