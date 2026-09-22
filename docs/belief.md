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
packing, so a patch trained there continues in NumPy with its records.

### Training the transition: the imagination loss

A belief patch trained on the one-step read alone learns to lean on the next frame. The
repair takes what it needs from the evidence, the transition carries less each epoch, and
the imagination decays while the one-step read improves. On Seaquest from pixels the
one-step explained variance of the frame-to-frame change rose from 0.06 to 0.35 over
twenty epochs while the open-loop imagination four decisions ahead fell from 0.03 above
persistence to 0.12 below it. This is teacher forcing, and the unified world-model report's
rule names the repair: a rollout must predict every input it consumes.

The imagination loss trains that. From random moments `t0` of each chunk, imagine `H`
steps under the recorded actions from the belief at `t0 - 1` with no observation, and
penalise the drift of the imagined output from the truth along with the one-step loss:

```text
L = L_one_step + w * mean over starts of  1/2 * mean( W * [cumsum_k(y_hat[t0+k] - y[t0+k]) ; fields]^2 )
```

where the cumulative sum runs over the retina outputs (the imagined retina against the
true one), the fields are compared step by step, and `W` is the same output weighting as
the one-step loss. The gradient flows through the imagined transitions into `T`, `G`, `C`
and back through the repair into the belief the rollout started from, so the belief is
trained to carry what the transition needs. Two starts per chunk, `H = 8` and `w = 0.3`
were enough on Atari: on Pong after twenty epochs the imagination with no observation
explained 0.42, 0.55, 0.62 and 0.61 of the changed cells at one, two, four and eight
decisions and beat persistence on 99 percent of the moments; the one-step read explained
0.76 against 0.70 for a convolutional GRU of the same size, and the imagined danger told
which of the six plans loses the point (area under the curve 0.92 against 0.47 shuffled)
when the emulator was restored to the same moment to try every plan. Not every run takes:
a second seed learned a one-step read of 0.70 and no imagination at all, the transition
having settled on persistence while the repair did the work, so the receipt of every run
carries the open-loop curve and the imagined term's weight is the first knob to raise. The
recipe lives in the application (`cadence-atari/tools/rung1.py`), fifteen lines around
the ordinary loss; the library supplies the imagination with a `state` and no observation.

## What it does not do

It has no certificate: the repair map is nonlinear and its contraction is measured, not
proved. It does not learn its keys; the store's code is the fixed sparse code of
`Records`. It does not partition the store by action or provenance; those are the
application's declarations. Categorical outputs and the night (`dream`, `sleep`) are the
record patch's; the belief patch's readout is linear.
