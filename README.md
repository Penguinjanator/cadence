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
> The revised `PatchNet` interface is included in version 0.9.0; earlier applications
> use other library compositions and their results do not validate it automatically.

## Start with PatchNet

`PatchNet` is the common starting point for new experiments. It uses the existing
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
python -m pip install cadence-net==0.9.0
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
