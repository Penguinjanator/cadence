# Reading a sequence through local state

`cadence.sequence` supplies two small readback patterns. Both maintain local
state that a caller reads through an explicit input port. Their updates depend
on observations available at that moment; neither differentiates through time.
The resulting state, diagnostics, and experiment receipts make their behavior
inspectable. These are observer-like software patches with local records and
feedback, not pretrained language understanding.

`BoundedTrace(width, decay=0.5, radius=1.0, center=True)` keeps a leaky trace of a
feature vector. `read()` optionally removes its common scalar component and
caps its L2 norm at `radius`. This prevents a wide, dense positive trace from
silently overwhelming a sparse input. It never amplifies a weak trace, changes
its width, or changes the network's trainable parameter count. Scaling a useful
signal can also impair prediction: the appropriate radius and centering choice
must be validated for the task. Use `center=False` when a common component is
meaningful. The recurrence itself provides no temporal credit assignment.

`SequenceCache(features, values, capacity=128, temperature=0.1,
center_rate=0.02)` stores a finite ring of feature/value associations for each
stream. Call `reset(batch)` explicitly before a new set of streams. `read(cue)`
returns `SequenceRead(value, entropy, maximum_weight, entries)` without changing
state. `observe(features, values)` writes an association and updates its running
feature mean. Both stored raw features and the current cue are normalized in
the same coordinate system when queried. `center_rate=0` selects ordinary
uncentered cosine attention. Scaled centering and unit norms keep opposite
large finite cues representable; softmax subtracts the maximum cosine before
dividing by temperature, so read probabilities remain finite even at a very
small positive temperature.

```python
from cadence.sequence import SequenceCache

cache = SequenceCache(features=64, values=75)
cache.reset(batch=16)

# At each time, features come from the current free equilibrium.
read = cache.read(features)
# Score the next-token prediction before the next token is revealed.
# Only after the environment reveals it:
cache.observe(features, observed_next_token_one_hot)
```

When values are normalized next-token distributions, the read is another
normalized distribution whenever at least one record exists. An empty cache
returns zero. Replace that zero with the base prediction before probability
interpolation. Select interpolation weights and temperatures on validation
sequences, and keep them fixed during held-out evaluation. Adding a positive
cache vector directly to output currents is a different model; it is not
probability interpolation.

This cache learns its mean and records observed associations. It does not learn
the feature encoder, discover a semantic address, or outperform conventional
attention given the same keys. The causal benchmark in
`experiments/sequence_readback/` includes that equality control and conventional
window/recurrent MLPs. It tests next-character prediction on disjoint novels,
with every result and a per-token verification artifact retained.
