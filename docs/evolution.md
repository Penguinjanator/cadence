# Evolve a brain

Evolution changes a genome between lives. `mutate` draws an offspring, and `evolve` keeps
the genomes whose lives score best under a fitness the application supplies. A `Genome`
decides which regions a brain has, how large they are and how they project; any other
genome (a dict of thresholds, a port topology, a patch's sizes) works the same way with its
own mutation. Learning inside a life is separate and covered in [learning](learning.md).

## Any genome

`cadence.genes(space, *, rate=1.0)` returns a mutation for dict genomes over a declared space:

| Kind | Declaration | Mutation |
|---|---|---|
| `log` | `("log", step, low, high)` | multiply by `exp(normal(0, step))`, clip; for scales and thresholds |
| `linear` | `("linear", step, low, high)` | add `normal(0, step)`, clip |
| `int` | `("int", low, high)` | move by one, up or down, within the bounds |
| `choice` | `("choice", option, ...)` | redraw among the options |

Each named gene mutates independently with probability `rate`, which must lie in `(0, 1]`.
Keys that `space` does not name are copied unchanged; a named gene the genome lacks is
skipped. A malformed declaration (unknown kind, wrong length, `step <= 0`, `low > high`, no
options) raises `ValueError`. Any hand-set constant becomes a gene this way: the hand-set
value starts the lineage and stays as the control.

```python
import numpy as np
from cadence import evolve, genes

space = {"threshold": ("log", 0.3, 0.5, 50.0), "budget": ("int", 1, 20), "law": ("choice", "a", "b")}
start = {"threshold": 4.0, "budget": 3, "law": "a"}

def fitness(genome, seed):  # the caller's: a threshold near 8, a budget near 12, law b
    return -np.log(genome["threshold"] / 8.0) ** 2 - abs(genome["budget"] - 12) / 10 + (genome["law"] == "b")

lineage = evolve(fitness, start, mutate=genes(space), generations=12, population=10, keep=3, seed=5)
assert lineage.best["law"] == "b" and lineage.best["budget"] > 3
assert lineage.best_fitness > fitness(start, 0)  # the hand-set start is the control
```

## Mutation

`cadence.genome.mutate(genome, rng, *, size_step=0.25, fixed=(), tied=())` returns one
`Genome` offspring with the same regions and projections; nothing is added or removed.
A negative or non-finite `size_step` raises `ValueError`, and a name in `fixed` or `tied`
that is not a region raises `KeyError`.

- Each blank region outside `fixed` takes the size `round(size * exp(normal(0, size_step)))`,
  at least one. Designed regions and regions in `fixed` keep their size.
- Each `(leader, follower)` pair in `tied` gives the follower the leader's size, for a
  `prefrontal_cortex` with one neuron per neuron of its region.
- Each projection's `density` is multiplied by `exp(normal(0, 0.2))` within `[0.01, 1]`,
  its `sign` moves by `normal(0, 0.2)` within `[-1, 1]`, and its `scale` is multiplied by
  `exp(normal(0, 0.2))` within `[0.05, 20]`. `count` and `reciprocal` stay as declared.

Fix the regions whose width the environment sets. In `Genome.composed`, the layout
`Brain.compose` develops, the motor cortex is designed and keeps its size; the `sensory`
region is blank and belongs in `fixed`, and the `prefrontal` region of the working trace
keeps the size of the `association` region it holds through `tied`.

```python
import cadence as cd
from cadence.genome import mutate

genome = cd.Genome.composed(2, 2, modules=(12,))
held = (("association", "prefrontal"),)
child = mutate(genome, np.random.default_rng(3), fixed=("sensory",), tied=held)
for parent, offspring in zip(genome.regions, child.regions):
    print(parent.name, parent.size, "->", offspring.size)
```

## Selection

```text
evolve(fitness, genome, *, generations=10, population=8, keep=2, seed=0,
       mapper=map, report=None, mutate=None, grow=None, **mutation)
```

Generation 0 scores the starting genome and `population - 1` offspring of it. Each later
generation scores `population` offspring, each of a parent drawn at random from the `keep`
best of the generation before. Life `index` of generation `g` gets the seed
`seed + 1000 * g + index`; its genome is grown at that seed and scored with
`fitness(grown, seed)`. Scores must be finite, and higher is better.

| Genome | Grown by | Mutated by |
|---|---|---|
| `Genome` | `develop(genome, seed=seed)`, or `grow` | `cadence.genome.mutate`; the `**mutation` keywords go to it |
| anything else | passed as it is, or `grow(genome, seed)` | the `mutate(genome, rng)` you pass (required) |

`evolve` raises `ValueError` when `generations`, `population` or `keep` is not a positive
integer, when `keep` exceeds `population`, when a non-`Genome` genome comes without
`mutate`, and when `**mutation` keywords come with a custom `mutate`.

This fitness lives forty moments, learning which of two actions each cue pays for, and
scores the reward of the last twenty moments less a small cost per synapse. Each grown
anatomy is wrapped with the composed founders.

```python
def life(connectome, seed):
    brain = cd.Brain.compose(2, 2, connectome=connectome, seed=seed)
    rng = np.random.default_rng(seed)
    cue = int(rng.integers(2))
    action = brain.step([np.eye(2)[cue]])
    earned = []
    for moment in range(40):
        reward = float(action[0] == cue)
        earned.append(reward)
        cue = int(rng.integers(2))
        action = brain.step([np.eye(2)[cue]], reward=[reward])
    return float(np.mean(earned[-20:])) - 1e-4 * connectome.synapses

lineage = cd.evolve(life, genome, generations=3, population=4, keep=2, seed=0,
                    fixed=("sensory",), tied=held)
for record in lineage.generations:
    print(record["generation"], round(record["best_fitness"], 3), round(record["mean_fitness"], 3))
```

`Lineage` (`cadence.genome.Lineage`) records what selection did. `generations` holds one
record per generation with `generation`, `best_fitness`, `mean_fitness` and `best`, that
generation's best genome (as its `to_dict()` when it has one, so a `Genome` is recorded
as a dict). `best` is the best genome
over all generations, the earliest on a tie, and `best_fitness` its score.
`report=callback` receives the lineage after every generation, so a long run writes its
record as it goes. `Genome.from_dict(record, designed=...)` reads a record back; each
designed region is supplied by name and must match the circuit digest in the record
([genome records](cortex.md#genome-records)).

```python
import json
from pathlib import Path

log = Path("lineage.json")
cd.evolve(life, genome, generations=2, population=4, keep=2, seed=0, fixed=("sensory",),
          tied=held, report=lambda so_far: log.write_text(json.dumps(so_far.generations)))
best = cd.Genome.from_dict(json.loads(log.read_text())[-1]["best"], designed={"motor": genome.region("motor")})
connectome = cd.develop(best, seed=0)
```

### The selection protocol

Select every genome the same way. The hand-set genome is the lineage's first member, and
the selected genome must beat it on held-out worlds, never on the worlds it was selected
on. A random search over the same space at the same number of evaluations is the second
control. On a small, smooth space like the one below, random search matches or beats
selection; report both and claim only what the lineage gains over each. The fitness reads
only what the genome cannot reweight, and prices both compute and surprise: a price on
compute alone favours genomes that switch learning off. Keep every outcome, including
genomes that prune what the designer put in.

```python
space = {"span": ("log", 0.3, 0.5, 8.0), "retain": ("linear", 0.1, 0.0, 1.0), "reads_probe": ("choice", 0, 1)}
hand_set = {"span": 2.0, "retain": 1.0, "reads_probe": 1}

def world(genome, seed):  # a noisy world whose best span is 4 and best retain 0.3
    noise = np.random.default_rng(seed).normal(0.0, 0.05)
    return -np.log(genome["span"] / 4.0) ** 2 - (genome["retain"] - 0.3) ** 2 - 0.1 * genome["reads_probe"] + noise

def draw(rng):  # one uniform draw from the same space
    return {"span": float(np.exp(rng.uniform(np.log(0.5), np.log(8.0)))),
            "retain": float(rng.uniform(0.0, 1.0)), "reads_probe": int(rng.integers(2))}

generations, population = 12, 4
selected = evolve(world, hand_set, mutate=genes(space), generations=generations,
                  population=population, keep=2, seed=0).best
rng = np.random.default_rng(1)
drawn = [draw(rng) for _ in range(generations * population)]   # equal evaluations
lucky = max(enumerate(drawn), key=lambda pair: world(pair[1], pair[0]))[1]
held_out = range(10_000, 10_020)                               # worlds selection never saw
score = {name: np.mean([world(g, s) for s in held_out])
         for name, g in (("hand_set", hand_set), ("selected", selected), ("random", lucky))}
print({name: round(float(value), 3) for name, value in score.items()})
assert score["selected"] > score["hand_set"]
```

## Parallel lives

`mapper` runs the lives of one generation. It receives a callable and an iterable of
`(genome, seed)` jobs, as `map` does, and returns their scores in order. A pool's `map`
gives the same lineage whenever the fitness depends only on its arguments. A thread
pool runs the lives in one process. A process pool (`multiprocessing.Pool.map`,
`concurrent.futures.ProcessPoolExecutor.map`) runs them on separate cores; its workers
import the fitness (and `grow`), so each is a module-level function of a file and the script
creates the pool under `if __name__ == "__main__":`.

```python
from multiprocessing.pool import ThreadPool

with ThreadPool(4) as pool:
    parallel = cd.evolve(life, genome, generations=3, population=4, keep=2, seed=0,
                         fixed=("sensory",), tied=held, mapper=pool.map)
assert parallel.best == lineage.best and parallel.best_fitness == lineage.best_fitness
```

## What selection, learning and candidate sampling change

| Mechanism | Changes | Leaves as it is |
|---|---|---|
| Selection (`evolve`) | The genome of the next generation: region sizes and projection densities, signs and scales, and so the connectome a life starts from | Every synapse and record inside a life |
| Learning inside a life | Synapses (`Learner`, `ActorCritic`) and memory writes (`SynapticMemory.observe`) | The genome |
| [Candidate sampling](concepts.md#detuning-inside-a-life) | The drive of one decision, and so the equilibrium the brain reads | Parameters, records and genome |

The mechanisms combine inside the fitness. A life that samples candidate actions scores the
genome together with that sampling, so give every genome the same perturbation bound,
candidate count and settling budget, and count every candidate's settling steps in the cost
of the life.
