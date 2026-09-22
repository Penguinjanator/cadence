# The belief patch

`BeliefPatch` keeps a belief that a learned transition carries forward under the action
that was executed, lets the evidence of the moment repair that belief through a few
iterations of one nonlinear map, reads the record store inside those iterations, and
answers through a linear readout the store patches. It is the smallest patch in the
library that can imagine on its own: the record patch's context is computed from the input
of the moment, so without an input it has no next state; the belief patch's transition
needs only the belief and a declared action.

```text
p[t]    = g * z[t-1] + (1 - g) * tanh(T [z[t-1]; a[t-1]] + t_b),   g = sigmoid(G [z; a] + g_b)
e[t]    = tanh(port(o[t]) + e_b)
z(0)    = p[t]
m(k)    = read(code([e[t], z(k)]))
z(k+1)  = (1 - alpha) z(k) + alpha tanh(F [z(k); e[t]; p[t]; m(k); 1] + f_b)      k < K
y[t]    = C z(K) + c + decode(m(K))
```

The three clocks stay apart: the environment's moments `t`, the repair iterations `k`
within a moment with the evidence held fixed, and learning between chunks. An imagined
step is the transition alone, with the store read at the expectation and nothing observed.

## What each part is for

- **The transition** answers what follows under an action. Scene and action meet inside
  the repair map, so the effect of an action can depend on what is where; the guide's test
  is the mixed difference of the belief over an observation and an action, which is zero
  for a port whose blocks never meet and nonzero here.
- **The repair** assimilates evidence: the map reads the belief, the encoded observation,
  the expectation and the store's read together, and moves the belief by a damped step.
  `BeliefPath.residual` is the size of the last step; a familiar moment ends near zero.
- **The store** holds the residual of the slow readout at the code of the final reading,
  coded to `record_width` signs, written once per observed moment. Its read enters the
  repair and patches the readout. No gradient reaches the store; that is the record
  patch's rule and it stays.
- **Imagination** (`imagine(actions)`) leaves parameters, records, state and counters
  unchanged and consumes no observation. A continuation conditioned on recorded future
  inputs is a different measurement; this method cannot make one.

```python
import numpy as np
from cadence import BeliefPatch, DenseBlock, MapBlock, StructuredPort

port = StructuredPort(3 * 24 * 24 + 52, [MapBlock(0, 3, 24, 24, 8, 5, 2), DenseBlock(1728, 52, 64)], broadcast=(1728, 52))
brain = BeliefPatch(port, actions=9, belief=256, outputs=1728 + 5, iterations=2, cells=4096, active=32)
brain.reset()
seen = brain.assimilate(observations, actions)          # (N, T, inputs), (N, T, actions): the belief follows the evidence
ahead = brain.imagine(candidate_actions)                # (N, H, actions): private, from the live belief
learned = brain.observe(observations, actions, targets, rate=1.0)   # one backward scan through every iteration; the store takes the residuals
```

## Learning

`observe` computes the adjoint of the precision-weighted half mean squared error through
every repair iteration and the transition over the chunk, treating each store read as
given, and moves the parameters by `rate` times that gradient. The adjoint matches finite
differences to 1e-5 on every parameter group (`tests/test_belief.py`). The loss is a mean
over time and outputs, so the useful rate scales with their product; a wide picture port
needs a rate in the thousands, or the torch backend with an ordinary optimiser.

`belief_torch.TorchBelief` is the same slow half as a torch module with autograd, for
batches of streams on a GPU; `export()` and `load()` move the parameters in the library's
packing, so a patch trained there continues in NumPy with its records and its custody.

## What it does not do

It has no certificate: the repair map is nonlinear and its contraction is measured, not
proved. It does not learn its keys; the store's code is the fixed sparse code of
`Records`. It does not partition the store by action or provenance; those are the
application's declarations. Categorical outputs and the night (`dream`, `sleep`) are the
record patch's; the belief patch's readout is linear.
