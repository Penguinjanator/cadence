# Learning content prototypes

`ContentMemory` is a bounded observer-like software patch: its cue and observed-value
ports feed local state, its readback predicts from a record, and feedback corrects the
selected association. It receives features of an observation, rather than a slot number,
class ID, or task ID as its address. It is a record store with a learned address; the
[records cortex](memory.md#records) uses a fixed one.

```python
import numpy as np
from cadence.content_memory import ContentMemory

memory = ContentMemory(inputs=3, outputs=2, capacity=32, match=0.75)
memory.observe(np.eye(3)[:2], np.eye(2))
prediction = memory.recall(np.array([[0.99, 0.1, 0.0]]))
assert np.array_equal(prediction, [[1., 0.]])
```

Competition compares unit cue vectors with stored unit prototypes. If the best cosine
is at least `match`, that prototype wins; otherwise an observation recruits an unused
slot, or replaces the least recently written one when the store is full. A matched
prototype moves toward the observed cue by `key_rate`; its value moves toward the actual
observed value by `value_rate`. New associations are stored in one observation. Setting
`key_rate=0` gives the fixed-prototype ablation. Values never select the slot.

Call `recall(cue)` before revealing the observed value to `observe(cue, value)`. Reads
below the match threshold and zero cues return zero; `select(cue)` returns the selected
slot (`-1` for abstention) and its cosine score. Reads leave all memory unchanged. Array
rows are observations of one shared store; use separate instances for isolated streams.
`clear()` erases all records. The caller owns checkpointing this object separately from
a `Learner`.

This is competitive prototype learning and content retrieval in a supplied feature
space. It does not learn which feature dimensions matter, train an encoder, infer a
latent task, or learn when an observation should be trusted. It cannot distinguish two
meanings with identical cues, and eviction limits retention when the content exceeds
capacity. Read cost is proportional to capacity times feature width. The `mutable_bytes`
field counts NumPy array payloads, excluding Python object/counter overhead.

Keep the input coordinate system stable. This implementation deliberately has no
running centering transform. A moving mean or changing encoder can invalidate stored
prototypes even when the memory itself is functioning correctly. A frozen transform
must be fitted from training inputs and applied identically at writes and reads.

The [content memory tests](../tests/test_content_memory.py) cover selection by noisy
content, learned prototypes, revision and capacity, aliasing and invalid input.

The mechanism is compatible with the broad complementary-learning-systems motivation
for rapid records plus slower learning; its cosine search and allocation threshold are
engineering choices. See the original
[McClelland, McNaughton and O'Reilly paper](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf).
