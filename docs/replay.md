# Explicit experience rehearsal

In a learner whose every parameter takes part in every prediction, each update moves
every other prediction, and on a correlated stream later updates overwrite what earlier
ones taught. `ReservoirReplay` adds a finite store of actual feature/label observations to
the loop and gives the existing `Learner.step` examples from earlier experience, so a
single head can receive positive targets for classes absent from the current minibatch.
The mechanism needs labels and additional training rows; an anchor on old weights alone
does not supply them. It is the control a dense baseline receives; a
[records cortex](memory.md#records) over a sparse code learns from one stream without it.

```python
import numpy as np
from cadence.replay import ReservoirReplay

replay = ReservoirReplay(capacity=256, inputs=196, seed=0)

# In the ongoing loop, after obtaining a labeled minibatch x, y:
# old_x, old_y = replay.sample(len(x))            # only previously seen observations
# train_x = np.concatenate([x, old_x])
# train_y = np.concatenate([y, old_y])
# learner.step(make_drive(train_x), train_y)      # unchanged local learning rule
# replay.observe(x, y)                           # real observations only
```

The store uses uniform reservoir admission across all observed examples and uniform
sampling without replacement among retained records. It receives no task identifier,
class-balancing oracle, future label, or test data. Admission has a separate random
generator from sampling, so sampling more often does not change which examples survive.
Stored and returned arrays are independent copies. Labels are scalar nonnegative integer
class indices; this is not a trajectory or return estimator.

Storage is fixed at `capacity * (inputs + 1) * 8` bytes for the feature and label arrays
(float64/int64), plus Python objects, counters, and the two random-generator states.
`to_dict()` exposes the array payload as `mutable_bytes`, the number of observations
offered, and the number of sampled training examples. The caller owns isolation,
checkpointing, and the point at which a label becomes available.

Replaying old distributions can slow learning of a changed distribution. A reservoir
also loses rare events and eventually evicts examples. Replay does not solve conflicting
tasks with identical observations, does not guarantee lifetime retention, and does not
restore a guarantee for a learner explicitly restricted to no replay.

The motivation is reactivation of experience, as observed by
[Wilson and McNaughton](https://pubmed.ncbi.nlm.nih.gov/8036517/) and developed in
[complementary learning systems](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf).
Uniform reservoir sampling is a conventional engineering policy. The
[replay tests](../tests/test_replay.py) check that the store retains prior observations
only, that admission is independent of sampling, and that an invalid batch admits nothing.
