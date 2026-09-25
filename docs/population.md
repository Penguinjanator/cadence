# Many brains at once: populations on a device

One class runs a whole population of record patches on a graphics processor (or any torch
device): `PopulationPatch`. Every brain is an instance with its own parameters; every
world a brain is in is a stream with its own record store; every settle of the population
is one batched product.

```python
import torch
from cadence import RecordPatchNet
from cadence.population import PopulationPatch

net = RecordPatchNet(20, 16, 3, groups=(3,), cells=512, active=8, slowest=2.0)
pop = PopulationPatch.from_patch(net, instances=8, streams=32)   # 8 brains, each in 32 worlds
x = torch.rand(8, 32, 20, device=pop.dev)                          # one reading per brain and world
out = pop.imagine(x)                                               # out["output"]: (8, 32, 3)
target = torch.nn.functional.one_hot(torch.randint(0, 3, (8, 32)), 3).to(pop.dev, out["output"].dtype)
pop.observe(x, target, rate=0.3)                                   # every brain its own step, every world its own write
pop.inherit(torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]), sigma=0.1)      # selection: copies of the parents, mutated
```

The device is chosen for you (cuda, then mps, then cpu). `from_patch` starts every brain as
a copy of a NumPy patch, which is how a schooled patch becomes a population; the plain
constructor takes the same arguments as `RecordPatchNet` plus `instances` and `streams`.

## What batches, and what does not

Brains, worlds and (through several `PopulationPatch` objects) the patches of a brain all
batch, because nothing crosses them: each brain's slow step is the adjoint of its own
one-moment loss, each world's records take their own residual, and no gradient reaches a
store or another patch. The moments of one world do not batch; a temporal patch settles
them in order, as any recurrent system does. This class is the twin of a one-moment path
from rest, the reading a decision or a school batch gives a patch.

## Parity and measurement

The tests hold the twin to the NumPy patch on the cpu in float64: the prediction, the slow
step against the NumPy adjoint (linear and categorical) and a write with its read, all to
1e-9. On an Apple M-series GPU, brains of three patches at 32 channels and 256 cells playing
rock, paper, scissors ran 2,048 games at 7,400 game-rounds per second and 32,768 games at
27,000, against about 40 per core on the NumPy path (`cadence-games/receipts/throughput.json`).
The tables cost `instances * streams * cells * outputs` values; keep stores small and brains
many.
