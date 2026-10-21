# Cadence documentation

Cadence builds an animal-like brain: one continuing equilibrium that settles each observation
into an interpretation, acts on it and repairs its local relationships from the outcome.
System 1 is the default brain, System 2 observers are optional regions in the same settlement,
and an MLP or classifier setup is a control to compare against. The
[manifesto](../MANIFESTO.md) states the aims and [AGENTS.md](../AGENTS.md) the construction rules.

## Install

```sh
python -m pip install cadence-net==0.84.3
```

Cadence needs Python 3.11 or later and NumPy. From a checkout, run `python -m pip install -e .`
at the repository root. The extras `[fast]` (Numba and SciPy), `[accel]` (PyTorch) and
`[apple]` (MLX) add backends; [backends](backends.md) says which to choose. The
[v0.84.3 documentation](https://github.com/muellerberndt/cadence/blob/v0.84.3/docs/README.md)
is pinned to the release tag.

## Start here

1. [Quickstart](quickstart.md): compose a brain, run it live, imagine, save and resume; then
   run [examples/continuing_brain.py](../examples/continuing_brain.py).
2. [World model](world-model.md): the lifecycle of one continuing brain and the order of one moment.
3. [Continuous interaction](continuous.md): connect a body through `live` and `step`.
4. [Memory](memory.md): what the brain keeps between moments and which memory to switch on.
5. [Compose a brain](brain.md): regions, observers and the founder values of every setting.

## Choose the interface for the task

| Goal | Start with | Read next |
| --- | --- | --- |
| One continuing stream of observations and actions | `Brain.compose(..., arousal=True)`, `brain.live(...)` | [Continuous interaction](continuous.md) |
| Explicit teaching, batched streams, learning from every outcome | `brain.step(...)` | [Continuous interaction](continuous.md), [learning](learning.md) |
| Memory across moments | `Brain.compose` memory settings | [Memory](memory.md) |
| Change settings of a living brain | `brain.retune(...)`, `brain.describe()` | [Retune the same life](brain.md#retune-the-same-life) |
| Learned consequences and planning | `TemporalPatchNet`, `TemporalPatchNet.plan` | [Temporal](temporal.md), [planning](planning.md) |
| Sequence records | `RecordPatchNet` | [Record patch](record-patch.md) |
| Belief under action | `BeliefPatch` | [Belief](belief.md) |
| Self-readback gains and a governed life | `Steered`, `Life` | [Steering](steering.md) |
| Custom anatomy | `Genome`, `develop`, `NeuralGraph` | [Write a cortex](cortex.md) |
| Connectome import | `NeuralGraph` from a measured wiring diagram | [Connectomes](connectomes.md) |
| Selection across lives | `genes`, `evolve`, `cadence.genome.mutate` | [Evolution](evolution.md) |
| Many brains on a GPU | `cadence.population.PopulationPatch` (torch) | [Population](population.md) |
| The population solver | `cadence.experimental.equilibrium` | [Population solver](equilibrium/index.md) |

Each model family carries its own guarantee, and one family's guarantee does not certify a
composition of families ([contracts](contracts.md)).

## All guides

| Group | Page | Covers |
| --- | --- | --- |
| Start | [Quickstart](quickstart.md) | first continuing brain, imagination, save and resume |
| | [World model](world-model.md) | lifecycle and causal order of one moment |
| | [Orientation](orientation.md) | Cadence for machine-learning readers |
| | [Concepts](concepts.md) | neuron equations, settling, state lifetimes |
| Run the brain | [Compose a brain](brain.md) | composition settings and founder values |
| | [Continuous interaction](continuous.md) | `live`, `step`, arousal, refusal, saving |
| | [Sequential tasks](sequential-tasks.md) | multi-step prerequisite chains |
| Memory | [Memory](memory.md) | every memory type and when to use it |
| | [Episodic memory](episodic.md) | `memory_choice=True` records that vote in the choice |
| | [Recall memory](recall.md) | `recall=True` slots and the learned address |
| | [The learned code](code.md) | `learned_code=units`: a code region settled with the brain, the memories keyed on it |
| | [Standalone stores](records.md) | `Records`, `FastSynapses` and other stores used directly |
| Learning and reward | [Learning](learning.md) | the free/nudged rule and teaching budgets |
| | [Reward](reward.md) | three-factor rule, critic, dopamine, `ActorCritic` |
| Build and wire | [Build your own brain](build.md) | tutorials on your own data |
| | [Write a cortex](cortex.md) | regions, projections, ports |
| | [Connectomes](connectomes.md) | a measured wiring diagram as one `NeuralGraph` |
| | [Evolution](evolution.md) | genomes, mutation and selection |
| Other model families | [PatchNet](patchnet.md) | reciprocal patches on continuous observations |
| | [Recursive settlement](recursive-settlement.md) | observer populations in one graph |
| | [Temporal](temporal.md) | `TemporalPatchNet` transitions |
| | [Planning](planning.md) | private repair of input ports toward a goal |
| | [Protect temporal responses](temporal-memory.md) | `TemporalPatchNet` tool that protects chosen responses; not a brain memory |
| | [Partitioned](partitioned.md) | fixed connectivity masks, experimental |
| | [Record patch](record-patch.md) | `RecordPatchNet` and its compositions |
| | [Population](population.md) | `PopulationPatch` on torch |
| | [Belief](belief.md) | `BeliefPatch` assimilation and imagination |
| | [Steering](steering.md) | `Steered` and `Life` |
| Evaluate | [Evaluate a brain](evaluation.md) | task design, baselines, controls |
| | [Protocols](protocols.md) | held-out predicate tests |
| | [Receipts](receipts.md) | outcomes bound to source hashes |
| | [Troubleshooting](troubleshooting.md) | symptom to remedy |
| Reference | [Contracts](contracts.md) | numerical and learning guarantee per family |
| | [Certificate](certificate.md) | the settling contraction certificate |
| | [Backends](backends.md) | devices, precision, extras |
| | [API reference](api.md) | every public name and signature |
| Population solver | [Population solver](equilibrium/index.md) | `cadence.experimental.equilibrium` |
| | [Quickstart](equilibrium/QUICKSTART.md) | build, teach, test, save and resume |
| | [Brain design](equilibrium/BRAIN_DESIGN.md) | wiring, teaching rows, capacity, diagnosis |
| | [Live operation](equilibrium/LIVE.md) | bodies, reward learning, retention |
| | [Execution and cost](equilibrium/ACCELERATION.md) | devices, batching, parallel lives |
| | [Specification](equilibrium/SPECIFICATION.md) | equations, qualification, continuation |
| | [Solver API reference](equilibrium/REFERENCE.md) | classes, parameters, diagnostics |
| | [Agent guide](equilibrium/AGENTS.md) | application workflow for coding agents |

## Evaluate and contribute

Design a claim with [evaluation](evaluation.md), test it with [protocols](protocols.md) and
report it with [receipts](receipts.md). [CONTRIBUTING.md](../CONTRIBUTING.md) lists the checks,
[CHANGELOG.md](../CHANGELOG.md) the releases, the [paper](https://philpapers.org/rec/MUECAP-2)
the hypothesis and [cadence-demos](https://github.com/muellerberndt/cadence-demos) the
application demos.
