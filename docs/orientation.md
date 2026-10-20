# Cadence for machine-learning people

This page maps Cadence onto models you know: which brain corresponds to which
familiar model, where each one's learning signal comes from, what the arrays look
like and what the words mean. The application interface is
`Brain.compose(..., arousal=True)`, then `brain.live(observation, reward=...)` for
each real outcome ([composition guide](brain.md), [quickstart](quickstart.md)); the
families below are the numerical mechanisms underneath and beside it.

## The picture

A Cadence brain is a set of patches joined by ports. A patch holds a bounded local
state (the potentials of a group of neurons, or one context vector), reads its
inputs through ports and updates by its declared equations. Graph and temporal
models qualify a settled state and learn from detuned phases. Record models run a
causal scan and belief models run finite repair iterations; both train their slow
weights by an adjoint backward scan through the observed window. The
[contract guide](contracts.md) separates these cases.

![Idealized equilibrium contrast under a symmetric energy model](assets/learning-cycle.svg)

## The four brains against models you know

| Cadence | Nearest familiar model | What is different |
| --- | --- | --- |
| Settling brain (`Genome`, `develop`, `NeuralGraph`, `Learner`) | A continuous Hopfield or energy-based recurrent network trained by [equilibrium propagation](https://arxiv.org/abs/1602.05179): rate neurons relax to a fixed point, and the gradient is the contrast of a free and a nudged fixed point. | Any wiring, including a measured connectome. Reciprocal learning ties paired weights; an arbitrary directed import need not satisfy the gradient contract. A [certificate](certificate.md) says when settling is a contraction and how far the answer is from the equilibrium. Reward learns through eligibility traces and one broadcast prediction error. |
| Temporal patch (`TemporalPatchNet`) | A tanh RNN, `h[t] = A tanh(h[t-1]) + B u[t]`, `y[t] = C tanh(h[t])`. | Trained by two detuned solves of the whole path's energy and their contrast, with the solves checked for symmetry, in place of backpropagation through time. The same contrast on the input ports with weights frozen plans through the learned model. Chosen responses can be protected by projecting later updates ([orthogonal weight modification](https://www.nature.com/articles/s42256-019-0080-x)). |
| Record patch (`RecordPatchNet`) | A gated linear RNN with per-channel retention (a minimal GRU or a linear recurrent unit) plus a [delta-rule](https://arxiv.org/abs/2406.06484) table read through a sparse code, the cerebellar memory of Marr and Albus. | The store sits inside the patch and takes an outcome in one write with no gradient; readings generalise by the overlap of their codes. `sleep` teaches the slow weights the store's own completions with the data gone and rewrites the store at dawn. Categorical ports end in a softmax per group. |
| Belief patch (`BeliefPatch`) | A recurrent state-space world model: a transition under the executed action, evidence assimilated into the belief, imagination as an open-loop rollout ([Ha and Schmidhuber](https://arxiv.org/abs/1803.10122), [Dreamer](https://arxiv.org/abs/1912.01603)). | Evidence repairs the belief by a few iterations of one nonlinear map that also reads the record store. Imagination is the transition alone and consumes no observation. The [imagination loss](belief.md#training-the-transition-the-imagination-loss) makes the transition carry the belief instead of leaning on the next frame. |

The record store's key is a fixed random expansion, described under *Code* in the glossary.
The composed brain has no `sleep` cycle; its memories key on the observation or on the fixed
sparse code `code=units` ([the observation code](memory.md#the-observation-code)).

- `RecordPatchStack` puts two record patches in depth; `JointRecordPatches` couples
  several through finite repair rounds on declared ports. `StructuredPort` reads a grid
  through a tied local kernel, a convolution at the port. `evolve` mutates and selects
  genomes across lives.
- `Steered` ([a brain that reads itself](steering.md)) weighs a belief cortex's
  senses by a second belief patch that reads its readback. The nearest familiar model
  is an attention gate; here the gain acts inside the repair and is learned within the
  life from the cortex's own loss.
- `Life` lets a governor decide from the brain's own surprise when to imagine and
  when to learn. `PatchGovernor` is a graph of fourteen rate neurons by default (seven
  readback, `cortex=4` cortex, three motor) whose synapses are genes;
  `ThresholdGovernor` is the hand-set rule and its control.

## Where the learning signal comes from

- **The settling brain and the temporal patch use no backward pass through time.**
  The free settle is the answer. A second settle with the outputs nudged by `beta`
  toward the target (and, centered, a third nudged away) gives every synapse the
  product of its two neurons in each phase; the difference over `2 beta` is the
  update. On a smooth stable equilibrium with symmetric effective weights the contrast
  tends to the loss gradient as `beta` tends to zero
  ([learning](learning.md#5-why-the-contrast-is-a-gradient)). At a finite `beta` it is
  an estimate with a `beta^2` error term and a phase-solve term that grows as
  `1/beta`; the default activation's kink at zero lies outside the smooth-branch
  argument. [benchmarks/local_equilibrium](../benchmarks/local_equilibrium/README.md)
  compares the contrast learner with an implicit-gradient learner on the same network.
  The temporal patch does the same on a whole path: it solves the path's energy plus
  and minus `beta` times the loss by damped Newton steps and contrasts the parameter
  derivatives ([temporal learning](temporal.md#energy-and-update)).
- **The record patch and the belief patch train their slow weights by an adjoint
  scan** over the observed chunk: one forward scan, one backward scan, the ordinary
  gradient of the slow readout's loss. The record store is a port value added to the
  readout and receives no gradient; after the scan the delta rule writes whatever the
  slow readout got wrong. The record patch's energy is quadratic, and `detune` shows
  that the contrast of its two detuned equilibria equals the adjoint
  ([record patch](record-patch.md#detuning-as-the-acceptance-check)).
- **Records learn by one write** of the error into the active cells' rows, so two
  readings share records exactly as far as their codes overlap.
- **Reward** enters the settling brain as a broadcast prediction error gating every
  synapse's eligibility trace, a three-factor rule ([reward](reward.md)).
- **Backtracking** (`observe(..., backtrack=True)`) accepts a parameter step only
  after a target-free replay from the original boundary lowers the loss, halving the
  rate up to sixteen times. It replaces a learning-rate schedule. The temporal and
  record patches take it as an opt-in flag; `BeliefPatch.observe` admits this way by
  default whenever it has a target.

## Compared with a backprop network

| | A network trained by backprop | A Cadence brain |
| --- | --- | --- |
| Reading an answer | Evaluate the layers | Settle to a fixed point, or run the causal scan, with a residual check |
| State between inputs | A cache or a separate memory | A settled state, a carried context, a trace or a record store, each explicit |
| Credit | Reverse-mode differentiation | Free and nudged contrasts for the settling brain and the temporal patch; an adjoint scan of the slow half for the record and belief patches; delta-rule writes for records |
| Exact-gradient conditions | A differentiable computation | For the contrast: a stable smooth equilibrium, symmetric effective weights, converged phases and a vanishing nudge; the adjoint scan is the exact gradient of the slow readout's loss |
| Work | A forward and a backward pass | Every step of the free and nudged phases plus the update; one forward and one backward scan for the adjoint |
| Memory of one event | A replay buffer or fine-tuning | One write into the store, read back at the same reading |
| Planning | A separate policy or a search | The same contrast on the action ports with the weights frozen; imagination under declared actions |
| Hardware | GPU tensors | NumPy float64 on the CPU; optional torch, MLX or Numba for the settling brain's graph transport; torch for the belief patch's slow half ([backends](backends.md)) |

The table compares update mechanisms. Networks trained by backprop can also be
recurrent, online, model-based and memory-using, so a fair comparison measures
acquired behaviour and total work on the same task and information.

## Shapes

| Model | Inputs | Targets | State between calls | Checkpoint |
| --- | --- | --- | --- | --- |
| Composed `Brain` | `(batch, inputs)` observations; one row for `live` | `step(teacher=...)` labels the current observation; reward labels the preceding executed action | Neural activity, working trace, optional efference copy, heard history, recall slots and their address map, the associative and episodic stores with the episodic temporal context, pending outcomes | `Brain.save` / `Brain.load` keep the complete continuation |
| Neural graph / `Learner` | a drive `(batch, n)`, one column per neuron; features go on the input neurons | integer labels `(batch,)` or `(batch, slots)` for `Learner.step`, or a target pattern `(batch, n)` for the phase calls | `BrainState` per row; pass `warm=` to continue | `Learner.save` / `Learner.load`, one `.npz` |
| Temporal patch | `(batch, time, inputs)` | `(batch, time, outputs)` | the hidden boundary `(batch, hidden)`; `reset()` clears it | `save` / `load`, `snapshot` / `restore` |
| Record patch | `(batch, time, inputs)` | `(batch, time, outputs)`, one-hot per group for categorical ports | the context `(batch, hidden)` plus the record tables | `save` / `load`, `snapshot` / `restore` |
| Belief patch | observations `(batch, time, inputs)` and actions `(batch, time, actions)` | `(batch, time, outputs)` | the belief `(batch, belief)` plus the record tables | `save` / `load`, `snapshot` / `restore` |
| Records cortex (`Records`) | a reading `(inputs,)` or `(batch, inputs)` | a dict of field targets | the tables, the running mean and the counters | `Records(**r.to_dict())` rebuilds the fixed cells; `state()` / `load_state()` carry the learned arrays |

Batch rows are independent streams sharing parameters. `imagine` is private: it
changes no learned parameter, record or live activity, though cost counters may count
its work. `advance` carries context without learning. `observe` learns and, for a
record patch, writes.

## Glossary

- **Patch.** A bounded unit with local state, ports, readback and a repair rule: a
  group of neurons with their synapses, one temporal or record patch, or a belief.
- **Port.** A declared band of a patch's state that another patch, or the world,
  reads or drives: a population of a connectome, the input and output ports of a
  path, a `Port` between record patches.
- **Settle, settling.** Repeating local updates under fixed evidence. A settle
  qualifies when the component's declared equation or stationarity check passes; small
  movement or an exhausted iteration budget is insufficient.
- **Equilibrium.** A state satisfying the component's settlement contract under fixed
  boundary conditions. For the neural graph these include drive, parameters and held
  memory readback; the population engine checks constrained stationarity. For a free
  path, the causal recurrence satisfies its temporal equations with zero defect.
- **Residual.** A model-specific number: neural equation error, population
  stationarity error or temporal seam defect. `BeliefPatch.residual` reports the last
  damped move and is no independent fixed-point certificate
  ([contracts](contracts.md)). A residual says nothing about task success.
- **Certificate.** The contraction bound that says settling has a unique equilibrium
  and how far the last step's movement leaves the state from it.
- **Free phase, nudged phase, beta, centered.** The settle under the inputs alone; the
  settle with the outputs pulled toward (or away from) the target with strength `beta`
  (default 0.1), where smaller is closer to the gradient and weaker; `centered` (default
  on) contrasts the plus and minus `beta` phases over `2 beta`.
- **Detuning, contrast.** Perturbing an equilibrium and reading the difference between
  the perturbed and the free state; the learning signal of the settling brain and the
  temporal patch, and the planning signal on input ports.
- **Adjoint scan.** The backward pass of the record and belief patches over one chunk.
- **Reading.** The vector a record store codes: for `Records` whatever the application
  assembles; for the record patch the input block and the context block.
- **Code, cells, active.** The fixed random expansion of a reading onto `cells` units,
  of which the `active` strongest keep their rectified drive, scaled to unit length; the
  sparse key of the store.
- **Record, table, write.** One row per cell in one table per predicted field; a write
  is the delta-rule step into the active cells' rows at the write rate (`record_rate` on
  a patch, `rate` on `Records`; one takes the outcome whole).
- **Slow weights, slow readout.** The patch's parameters and the readout `C h + c`
  without the record read.
- **Dream, sleep, dawn.** The store's completion of a cue from rest; teaching the slow
  weights those fixed completions with the data gone; rewriting the store afterwards so
  it holds only what the weights did not take.
- **Context.** The carried state of a record or temporal patch between moments.
- **Readback.** The detached state, residuals, counters and revision of a patch, as
  observations an application or another patch's port can read.
- **Genome, develop, region, cortex, projection.** The declaration of a settling
  brain's regions and projections; its deterministic layout into one connectome; a
  named population; a builder of a standard region; a declared bundle of synapses.
- **Connectome.** The synapse list of a settling brain (`pre`, `post`, `count`, `sign`)
  and its named populations.
- **Efficacy, gain, log_gain, bias.** The learned signed strength of a synapse; the
  neuron model's global gain (`NeuronModel.gain`); the per-neuron scale of a neuron's
  outgoing synapses, applied as `exp(log_gain[pre])` and fixed by the learner; the
  learned per-neuron offset.
- **Eligibility trace, dopamine, valence.** Per-synapse memory of recent contrasts; the
  broadcast prediction error that gates them; the object that centers and caps it.
- **Steering patch.** A patch of the same kind whose evidence is the readback of the
  rest of the brain and whose settled state sets their gains, gaze, goals or budgets.
- **Receipt.** A JSON file binding a result to the sources that produced it, with a
  digest and a verifier ([receipts](receipts.md)).

## Reading the numbers

A score with records is a different number from the score of the slow weights, so report
both. Measure settling, learning and record maintenance
separately on the same held-out data, and compare against the simple control the task
admits: the majority outcome, persistence, a linear predictor, the search alone or the
algorithmic solver. [Evaluate a brain](evaluation.md) lists the checks and
[common missteps](evaluation.md#common-missteps).

## Further reading

- Learning rule: [equilibrium propagation](https://arxiv.org/abs/1602.05179), its
  [centered estimator](https://arxiv.org/abs/2006.03824) and
  [orthogonal weight modification](https://www.nature.com/articles/s42256-019-0080-x).
- Records: the cerebellar sparse expansion with a linear readout of
  [Marr (1969)](https://doi.org/10.1113/jphysiol.1969.sp008820) and
  [Albus (1971)](https://doi.org/10.1016/0025-5564%2871%2990051-4); the
  [delta-rule fast-weight memory](https://arxiv.org/abs/2406.06484); fast and slow
  [complementary learning systems](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf)
  (McClelland et al., 1995) with their retention and interference tradeoffs.
- Reward: [prediction errors](https://www.gatsby.ucl.ac.uk/~dayan/papers/sdm97.html)
  (Schultz et al., 1997); eligibility-based online recurrent learning
  ([Bellec et al., 2020](https://www.nature.com/articles/s41467-020-17236-y)), from
  whose e-prop Cadence's rule differs.
- Acquisition and language: word/referent learning across ambiguous situations
  ([Smith and Yu, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2271000/)); live
  interaction in infant phonetic learning ([Kuhl et al., 2003](https://doi.org/10.1073/pnas.1532872100));
  communication and thought as distinct ([Fedorenko et al., 2024](https://www.nature.com/articles/s41586-024-07522-w));
  [speech-planning experiments](https://pubmed.ncbi.nlm.nih.gov/26858678/) on
  incremental formulation.

The [paper](https://philpapers.org/rec/MUECAP-2) states the hypotheses, the theorems
and the evidence.
