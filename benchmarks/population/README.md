# Population throughput

`throughput.py` measures the population kernel, `cadence.population.PopulationPatch`, against
the NumPy patch on one processor core, and writes `receipt.json` beside itself.

The work is one moment per brain and stream: an imagined reading, a one-moment path from
rest, and an observation with its write, on a patch of 32 inputs, 32 channels, 8 outputs and
a store of 256 cells (8 active), in a fixed random linear world per stream. The population
runs instances x streams in lockstep on the torch device (cuda, then mps, then cpu); the
reference runs the same patch, one brain in one stream with its own store, on one core with
the BLAS thread count at one. A second reference row runs 64 streams per call through one
shared store, which is a different world and is reported only for scale. The receipt also
holds the parity of the two paths on the cpu in float64: the reading and the slow step after
one observed moment.

```bash
python benchmarks/population/throughput.py              # the first available device
python benchmarks/population/throughput.py --device cpu --steps 20
```

The receipt of 2026-09-25 on an Apple M4 laptop (torch 2.14, the graphics processor):

| instances x streams | moments per step | moments per second |
| --- | --- | --- |
| 8 x 64 | 512 | 127,840 |
| 32 x 64 | 2,048 | 368,839 |
| 64 x 128 | 8,192 | 498,436 |
| 64 x 256 | 16,384 | 423,054 |

The reference, one brain in one stream on one core, ran 1,276 moments per second at the
same work (18,783 with 64 streams through one shared store); the best population row is
390 times the reference. The parity of the reading is 2e-16 and of the slow step 3e-17.
The dip at 16,384 streams is the store tables (139 MB) on that device. Wall-clock numbers
depend on the machine; the receipt records the load average at the end of the run.
