<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: a mesh of stateful neural patches, feedback loops and synaptic signals" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/) · [Cadence page](https://floatingpragma.io/cadence/) · [PyPI](https://pypi.org/project/cadence-net/) · [Documentation](https://github.com/muellerberndt/cadence/blob/main/docs/index.md)

**An experimental neural library for learning through local overlap repair and equilibrium detuning.**

Cadence explores an animal-inspired hypothesis: the same network that interprets
an observation should carry context, change its learned relationships through
experience, and explore possible continuations. It does not claim to reproduce
an animal or human brain.

> **Under active development.** Pin a release or exact commit for reproducible work.
> Version 0.10.0 includes temporal learning, explicit response protection and a
> small equilibrium actor. Earlier applications use other library compositions;
> their results do not validate the revised architecture automatically.

[Architecture and integration](https://github.com/muellerberndt/cadence/blob/main/docs/architecture.md) maps context, learning,
explicit memory protection, self-readback, imagination and action planning to
the implemented base-library APIs.

[Design a task](https://github.com/muellerberndt/cadence/blob/main/docs/task-design.md) explains how observations, action ports,
teaching and readback connect. [Common missteps](https://github.com/muellerberndt/cadence/blob/main/docs/missteps.md) covers
information loss, misleading proxy scores, memory interference and premature
claims about coordination or scaling.

## Temporal paths and persistent context

Version 0.10.0 adds experimental [`TemporalPatchNet`](https://github.com/muellerberndt/cadence/blob/main/docs/temporal.md). Each
moment is an observer-like patch with input/output ports, bounded activity and
a residual against its preceding state. During learning, adjacent patches repair
a complete observed path; centered equilibrium detuning changes their shared
relationships. Free inference carries hidden context, and private continuation
uses the same learned relationships without changing live state. Targets never
become the live hidden state.

The NumPy implementation exposes measured residuals, curvature checks, phase
work and complete checkpoints. It extends the verified scalar-cue solver to
time-varying observations. Sequence acquisition and long-term retention still
require behavioral tests: a path can satisfy every model equation and predict
poorly. This interface does not add a replay store or autonomous planning policy.
The existing 0.9.0 graph API remains compatible.

[`TemporalMemory`](https://github.com/muellerberndt/cadence/blob/main/docs/temporal-memory.md) adds explicit response protection to
temporal learning: caller-selected activity directions constrain later EP
updates without replaying raw examples. Protected-path retention and remaining
plasticity must be tested together; the available subspace is finite.

The base package also exposes [`EquilibriumActor`](https://github.com/muellerberndt/cadence/blob/main/docs/actor.md): a minimal
fixed-model observer plus joint future-state/action repair. It admits actual
readings, retains a supplied goal, proposes an action privately and replans after
real readback. Its linear Gaussian body assumptions and model-bound compressed
history are explicit; it is not yet a general nonlinear composer.

## Start with PatchNet

`PatchNet` is the existing general graph interface. It uses the existing
nonlinear neural dynamics and local free/nudged learning rule, with a fully
reciprocal graph by default. Its observer-like patches have bounded activity,
declared ports, local readback and feedback/repair; experiments expose their
observations, residuals and learned changes through reproducible evidence.

- **Current context lives in neural activity.** Activity continues between
  observations. Optional temporal overlap holds each solve against the previous
  free activity. Whether a particular graph retains a cue through a delay must
  be measured; a converged network can also forget its previous input.
- **Acquired relationships live in continuous synapses and biases.** Resetting
  activity leaves learned parameters intact. No external fact store or replay
  buffer is required by this interface. Interference during further learning
  remains a separate test.
- **Real observations detune the network.** Continuous targets nudge only
  declared observed ports. Each synapse changes from its endpoints' free/nudged
  activity contrast. The implementation does not construct a backward graph.
- **Convergence is checked.** Every required phase must satisfy the neural
  equations within the declared residual tolerance before learning commits.
  A capped solve is reported as unfinished. A small residual does not establish
  a unique, stable or correct answer.
- **Imagined continuations are isolated.** Branches use the same learned net
  without changing live activity, parameters or evidence bookkeeping. Branch
  isolation is implemented; useful planning and musical improvisation need
  empirical validation.
- **A complete checkpoint resumes the learner.** It includes parameters,
  optimizer state, current activity and the optional bounded source-ID window.
  Repeated IDs are suppressed within that window; IDs do not prove that two
  environmental reports are independent.

Follow the [PatchNet guide](https://github.com/muellerberndt/cadence/blob/main/docs/patchnet.md) for the running example, memory
semantics, continuous targets and rehearsal. The earlier `Brain`, `Learner`,
`GenericBrain`, `Records` and circuit APIs remain available for existing
applications. Their separate associative memories are optional compositions,
not required components of `PatchNet`.

The research target is fewer local mechanisms supporting acquisition, selective
forgetting, retention and correction together. Learned importance, robust
lifelong memory, autonomous specialization and animal-level capability remain
open. In particular, the revised core does not freeze each weight into a binary
state: that candidate prevented compatible learning through shared connections.

## Install

Python 3.11+, with NumPy as the only required dependency:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install cadence-net==0.10.0
```

For development from this checkout, use `python -m pip install -e .`.
On Windows activate with `.venv\Scripts\Activate.ps1`.
[Optional backends](https://github.com/muellerberndt/cadence/blob/main/docs/backends.md) support Numba, PyTorch and MLX.

## Reference

[Experience](https://github.com/muellerberndt/cadence/blob/main/docs/experience.md) · [Quickstart](https://github.com/muellerberndt/cadence/blob/main/docs/quickstart.md) ·
[Continuous interaction](https://github.com/muellerberndt/cadence/blob/main/docs/continuous.md) · [Records](https://github.com/muellerberndt/cadence/blob/main/docs/memory.md#records) ·
[Write a cortex](https://github.com/muellerberndt/cadence/blob/main/docs/cortex.md) · [Compose a brain](https://github.com/muellerberndt/cadence/blob/main/docs/brain.md) ·
[Evolve a brain](https://github.com/muellerberndt/cadence/blob/main/docs/evolution.md) · [Local learning](https://github.com/muellerberndt/cadence/blob/main/docs/learning.md) ·
[Reward](https://github.com/muellerberndt/cadence/blob/main/docs/reward.md) · [API](https://github.com/muellerberndt/cadence/blob/main/docs/api.md) · [All docs](https://github.com/muellerberndt/cadence/blob/main/docs/index.md) ·
[Lean proofs](https://github.com/muellerberndt/cadence/blob/main/lean/README.md)

Check equation residuals before claiming equilibrium. Measure task quality and
learning cost; local updates alone guarantee neither capability nor speed.
[Concepts and limits](https://github.com/muellerberndt/cadence/blob/main/docs/concepts.md). MIT licensed.
