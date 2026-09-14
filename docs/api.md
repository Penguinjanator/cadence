# API reference

Top-level exports and module-qualified helpers are listed below. Begin with the
[quickstart](quickstart.md) for complete runnable examples; the source docstrings
give additional details. Prefer keyword arguments for optional configuration.

## Connectome (`cadence.connectome`)

- `Connectome.from_synapses(n, *, pre, post, count=None, sign=None, populations=None, label="connectome", min_count=0.0)`:
  build from synapse lists. Parallel synapses merge (counts add), autapses drop, synapses
  below `min_count` drop, and the arrays are sorted by `(post, pre)`. `count` (synaptic
  contacts per synapse) defaults to 1 and `sign` to +1.
- Fields: `n`, `pre`, `post`, `count`, `sign` (arrays), `populations` (name → tuple of
  neurons), `label`. Property `synapses` (the number of synapses).
- `members(*names)`, `with_populations(**populations)`, `in_degree()`, `out_degree()`,
  `summary()`, `digest()` (SHA-256 of the sorted arrays).

## Neuron model (`cadence.neuron`)

- `NeuronModel(dt=0.2, slope=4.0, threshold=1.5, gain=0.02, stimulus_amplitude=3.0, adaptation=None, leak=0.0)`:
  the graded (rate) neuron. `activation(v)` is the rectified, re-based sigmoid with optional leak;
  `slope_at(v)` its derivative; `rest_emission` the raw sigmoid's value at rest, which is
  subtracted so rest publishes zero. `replace(**changes)`, `to_dict()`.
- `Adaptation(tau_steps=50.0, strength=1.0)`: a slow per-neuron variable that can
  produce rhythm in suitable circuits.
- `learning_neuron_model(gain=1.0, *, slope=1.0, leak=0.1, dt=0.5, stimulus_amplitude=1.0)` (in
  `cadence.learning`): responsive starting settings for local learning, to validate for your task.

## Brain (`cadence.brain`)

- `Brain(connectome, neuron_model, *, backend="cpu" | "torch" | "mlx", efficacy=None, log_gain=None, bias=None, device=None, dense_limit=2048, layout=None, precision=None)`:
  the runnable brain. `efficacy` (the learned synaptic efficacy) defaults to the connectome's
  signs; `log_gain` and `bias` to zero. Attributes include `connectome`, `neuron_model`,
  `efficacy`, `log_gain` and `bias`. When the connectome's dense blocks fit in `dense_limit`
  squared entries the transport is the block transport (see `cadence.blocks`); `layout`
  passes a precomputed cut, as `with_parameters` does. `brain.layout` is the cut in use.
  `precision` (torch only) is `"float32"` or `"float64"`; the default is float64 except on
  MPS. Float32 is the speed of a consumer GPU; compare it with float64 and record the
  measured precision error.
- `settle(stimulus=None, *, steps=60, state=None, mask=None, trajectory=False, nudge=None, tolerance=None) -> BrainState`:
  one stimulus; `stimulus` is a list of neurons at full amplitude, a `{neuron: level}` map, or a
  dense vector. Map values are levels multiplied by `stimulus_amplitude`; dense vectors
  are drives directly. `settle_batch(drive, ...)` takes `(batch, n)` drives.
  `mask` accepts `(n,)`, `(1, n)`, or `(batch, n)`, with finite values in `[0, 1]`; zero suppresses a neuron.
  With a positive `tolerance`, both methods stop once every activation moves less than
  that amount in a step. `None` or zero uses the full step cap. Nonfinite drives and warm
  potentials/adaptation are rejected. Use floating arrays for dense drives and maps for
  selected indices: the legacy integer vector of length `n` containing only 0/1 is a drive.
- `equilibrate(drive, *, budget=512, chunk=32, tolerance=1e-5, state=None, mask=None, nudge=None) -> Equilibrium`:
  seek a joint state whose equation residual is below tolerance, checking after each chunk.
  `budget` caps additional settling steps exactly, including a short final chunk;
  zero checks the starting state. Each check uses one transport; unread float64 Torch
  states stay on their device and return one scalar per row. The returned
  `Equilibrium` has `state`, per-row `residual`, total `steps`, `tolerance`, and a boolean
  per-row `converged` property. `state.steps` is the last chunk's count. Convergence here
  does not prove stability, uniqueness or task quality.
- `residual(drive, state, *, nudge=None, mask=None, on_device=True)`: per-row maximum remaining
  fixed-point equation discrepancy, including adaptation when enabled. One transport
  evaluation, no state change. Unread float64 Torch states use the resident kernel;
  float32 states and other backends use the float64 CPU reference. `on_device=False`
  selects that reference explicitly. It is a diagnostic and does not prove stability or uniqueness.
- `ep_structure(brain, *, fixed_inputs=(), tolerance=1e-12) -> EPStructure`: reports the
  maximum asymmetry of effective free/free weights, incoming mass on excluded source
  neurons, and adaptation. Its `compatible` flag covers structure only: inspect phase
  residuals, unchanged unnudged inputs, smoothness, stability, finite-beta bias, and
  parameter/loss units separately. See [the certificate guide](certificate.md#equilibrium-propagation-scope).
- `stimulus_vector(stimulus)`, `stimulus_levels(levels)` (finite levels times the stimulus
  amplitude, with signed values allowed), `readings(state, names, i=0)`, `with_parameters(*, efficacy=None, log_gain=None, bias=None)`,
  `weights` (effective drive per synapse), `dense()` (the `W[pre, post]` matrix), `to_dict()`.
- `Brain.contrast_on_device(plus, minus)`: the learning rule's per-synapse and per-neuron
  contrast computed on the device when both states carry its handle; `None` otherwise.
- `BrainState`: `v`, `activation`, `adaptation`, `steps`, `trajectory`, `activity_change`
  (the total movement of the activations while settling, per row), `device` (the same
  state on the accelerator that produced it, or `None`). State arrays have shape `(n,)`
  from `settle` or `(batch, n)` from `settle_batch`; trajectories add a leading step axis.
  `row(i)`, `mean(members, i)`, `fraction_active(members, level, i)`, `active(level, i)`,
  `batched`.
- `Nudge(target, mask, beta, softmax_temperature=None, weight=None, groups=None)`: extra drive
  `beta · (target − s)` on the masked neurons, or `beta · (target − softmax(s/T))` over the
  masked group with a temperature; `weight` scales rows. `groups` assigns a separate
  softmax group per neuron (`-1` excludes a neuron). `drive(s)`.
- `available_backends()`: `{"cpu": "numpy float64", "torch": "mps float32" | "cuda float64" | "cpu float64", "mlx": "gpu float32"}`, for what is installed.

## Blocks (`cadence.blocks`)

- `layout(connectome, *, max_pairs=256) -> Layout`: cut the neurons into contiguous ranges at
  the boundaries of the connectome's contiguous populations and pair the ranges that carry
  synapses. A connectome with no contiguous populations, or one that fragments into more
  than `max_pairs` blocks, gets one block, the full matrix.
- `Layout`: `starts`, `pair_pre`, `pair_post`, `offset`, `edge_index`; `ranges`, `pairs`,
  `size`, `bounds(k)`, `sources()` (ranges that receive no synapses), `flat(weights)`,
  `blocks(flat)`, `to_dict()`.
- `BlockTransport(layout, flat)`: `synaptic_input(s)` returns the synaptic input for the
  activations `s`, reusing the product of every source range that did not move since the
  previous call.
- `block_contrast(layout, s_plus, s_minus)`: the learning rule's per-synapse contrast as one
  Gram product per block.

## Streams (`cadence.stream`)

- `stateful(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0, context_init=1.0) -> (Connectome, tie_groups)`:
  `embedded` plus a `context` range of `hidden` neurons that receive no synapses and reach
  every hidden neuron. Populations `input`, `context`, `embedding`, `hidden`, `output`.
- `Trace(connectome, decay=0.5, amplitude=1.0, focus=0.0, source="hidden", target="context")`:
  the memory of the moment before, per stream. It keeps a trace of the `source` range's
  activation and adds it to the next settling as a stimulus on the paired `target` range
  (one neuron per source neuron). `focus` above zero weights each neuron by its movement
  since the last moment (its share of the row's mean movement, to that power), so the trace
  is brightest where the moment changed. `reset(batch, rows=None)`, `stimulate(drive)`,
  `update(state)`, `keep(rows)`, `ringing(floor=0.1)` (each source neuron's share of what is
  still ringing, one elsewhere; a salience for `ActorCritic.salience`), `to_dict()`.
- `Echo(connectome, decay=0.5, amplitude=1.0, ...)`: a `Trace` with the same defaults, used at
  focus 0 into the `context` range (the carried state of the earlier moments).
- `Afterglow(connectome, decay=0.5, amplitude=1.0, focus=1.0, source="hidden", target="afterglow")`:
  the focused `Trace`. With `source="input"` it is an afterimage of the picture itself, the
  memory that reads a cue against a static background (`tests/test_child.py`: 1.00 where the
  Echo reads chance).
- `FastSynapses(pre, post, decay=1.0, rate=1.0, amplitude=1.0, normalize=False, replace=False, rule="hebb", writes=0)`:
  one mutable `(pre, post)` matrix per stream; `writes` counts the rows written. `rule="delta"`
  uses unit keys and writes `rate * outer(key, value - prediction)`; it rejects
  `normalize`/`replace` and rates outside `[0, 1]`. The default `rule="hebb"` is additive
  Hebbian memory, with optional count-averaged reads (`normalize`) and one-hot row
  replacement (`replace`).
  `observe(key, value, write=None)` uses `(batch, width)` ports and writes all rows unless
  given a boolean mask; `recall(key)` reads without decay. `update(state, write=None, post=None)`
  reads named neuron activations instead and defaults to decay-only; `read(drive)` reads
  key neurons' drive columns. `stimulate(drive, inplace=False)` adds the read into post columns.
  `reset(batch, rows=None)`, `keep(rows)`, `to_dict()`. See [memory](memory.md) for stream
  identity, representation alignment, key interference, and checkpoint boundaries.
- `cadence.stream.columns(index)`: a slice when the neurons are one contiguous range,
  else the index array. Slices can avoid the copies required by advanced indexing.
- `SynapticMemory(pre, post, decay=0.9, rate=1.0, consolidation=0.05, ...)`:
  normalized delta synapses with a shared persistent `consolidated` matrix and per-stream
  effective `strength` matrices. `observe(key, value, write=None, *, salience=None,
  value_mask=None)` consolidates only observed values. Salience is a finite nonnegative
  `(batch,)` vector; the observed-value mask is boolean with the values' shape.
  `reset(batch, rows=None)` clears transient residuals and retains persistent synapses,
  including across batch changes. `clear()` erases both. Inherits `recall`, `read`,
  `stimulate`, `update` and `keep`; its rule is always delta. Reads never learn.
  See [the equations and lifecycle](continuous.md#repetition-and-salience-become-lasting-synaptic-changes).

## Regions (`cadence.regions`)

- `Region(name, size=0, circuit=None, inputs=None, outputs=None)`: a named group of neurons.
  A blank region has a `size` and no synapses of its own. A designed region has a `circuit`
  (a `Connectome`) and takes its size from it; `inputs` and `outputs` name the circuit
  populations that receive and send projections, and default to the whole region.
  `designed`, `neurons(population=None)` (region-local indices), `to_dict()`.
- `visual_cortex(height, width, *, channels=1, features=8, field=3, stride=1, init=1.0, seed=0, name="visual")`:
  population `input` with one neuron per pixel and channel, in the order of an image array
  `(height, width, channels)` flattened row by row, and population `output` with `features`
  feature maps. Each feature neuron receives synapses from one `field` by `field` window of
  the input; windows step by `stride`. Efficacies start random and fan-scaled.
- `cortex(size, *, lateral=0.0, name="association")`: a blank region, or with a negative
  `lateral` a designed one whose neurons inhibit each other pairwise.
- `motor_cortex(actions, *, lateral=0.0, name="motor")`: population `actions` with one neuron
  per action and optional pairwise lateral inhibition.
- `prefrontal_cortex(holds, *, name="prefrontal")`: a blank region with one neuron per neuron
  of `holds` (a `Region` or a size), for a `Trace` from the held region.

## Generic brain (`cadence.generic`)

- `GenericBrain.build(inputs, actions, *, hidden=64, density=1.0, lateral=-0.5, working_memory=False, memory_scale=12.0, episodic=True, features=8, field=3, seed=0, **options)`:
  develops `GenericBrain.genome(...)` and wraps it. `inputs` is a vector length, or an image
  shape `(height, width)` or `(height, width, channels)` for a `visual_cortex`. `options` go
  to the constructor.
- `GenericBrain.genome(inputs, actions, ...) -> Genome`: regions `sensory` (or `visual`),
  `association` (`hidden` neurons), `motor` (`motor_cortex(actions, lateral=lateral)`) and,
  with `working_memory`, `prefrontal`; projections sensory to association (reciprocal for a
  visual cortex), association to motor (reciprocal), and prefrontal to association at
  `memory_scale`.
- `GenericBrain(connectome, *, episodic=True, consolidation=0.05, working_memory_decay=0.2, working_memory_amplitude=3.0, learning=None, reward=None, seed=0, backend="cpu", device=None)`:
  needs populations `sensory` or `visual/input`, `association` and `motor`, and uses
  `prefrontal` for a working memory when present. `learning` defaults to
  `LearnerConfig(beta=0.1, eta=0.5, temperature=0.2, tolerance=3e-3, nudged_steps=12, momentum=0.9)`,
  `reward` to `ActorCriticConfig(gamma=0.9, lam=0.8, eta=1.0, eta_critic=0.3)`.
  Attributes `connectome`, `brain`, `learner`, `basal_ganglia` (`ActorCritic` reading the
  association cortex), `working_memory` (`Trace` or `None`), `hippocampus` (`SynapticMemory`
  from sensory to motor neurons, or `None`; old checkpoints retain `FastSynapses`), `sensory_index`, `association_index`,
  `motor_index`.
  - `stimulus(observations, *, memory=True)`: the drive of a batch; with `memory`, the
    working memory and the hippocampal recall are added.
  - `step(observations, *, reward=None, done=None, teacher=None, salience=None, bootstrap=None)`:
    the ongoing interaction API, returning the next sampled actions. Reward/done concern
    the preceding action; teacher labels concern the current observation. Omitted reward
    means zero (no reward event). The first call cannot receive past-action feedback.
    `last_learning` exposes the previous transition's report and demonstration count.
    Supplied salience controls memory consolidation; by default it is absolute reward.
  - `fit(observations, labels, *, epochs=30, batch=32) -> list[float]` (training accuracy per
    epoch), `predict(observations)`, `accuracy(observations, labels)`: independent samples,
    without memory.
  - `act(observations, *, greedy=False) -> actions`: one row per stream; updates the working
    memory. `learn(reward, done, next_observations, *, bootstrap=None, salience=None) -> report`: the hippocampus records the
    reward of the chosen action for its situation, `done` rows reset their working memory,
    and the basal ganglia learn from dopamine. For truncated episodes `bootstrap` supplies
    the value of the old episode's final observation; `next_observations` holds the reset
    observation for ended rows. Reward and bootstrap are finite batch vectors; done is boolean.
  - `reset()` clears working state, action cache, eligibility and reward centering; hippocampal
    records and slow parameters are kept. `parameters()` counts actor/critic parameters
    and the shared consolidated memory matrix; per-stream state is additional storage.
  - `save(path) -> Path`, `GenericBrain.load(path, *, backend="cpu", device=None, precision=None)`:
    complete composition checkpoints, including both optimizers, critic, random state,
    stream traces, prepared state, both memory timescales and any action awaiting feedback.
    Format 2 also retains pending nudged states; format 1 still loads.
    The archive is replaced atomically. A learner-only checkpoint
    is rejected by `GenericBrain.load`; `Learner.load` can extract a learner from either.
  Observations must be a nonempty finite batch, with image dimensions flattened per row.
  `fit` rejects noninteger labels and mismatched batches before updating. It resets current
  action/working state but keeps episodic records. `brain` always returns the current
  `learner.brain`, including after training.

## Genome (`cadence.genome`)

- `Projection(pre, post, density=1.0, sign=0.0, scale=1.0, count=1.0, reciprocal=True)`:
  synapses from `pre` to a fraction `density` of `post`. An end is a region name, meaning
  the region's `outputs` (for `pre`) or `inputs` (for `post`), or `region/population`.
  `sign` is the mean sign (−1 all inhibitory, +1 all excitatory, 0 mixed); `scale`
  multiplies the fan-scaled magnitudes; `reciprocal` adds the reverse synapses with the
  same weights.
  `Genome(regions, projections, label="genome")`: a connectome before development;
  `region(name)`, `to_dict()`, `Genome.from_dict(d, designed=None)` (a record keeps a
  designed region's circuit label and digest, and `designed` supplies the region by name).
- `develop(genome, seed=0) -> Connectome`: development, deterministic in the seed. Regions
  are laid out in order as contiguous populations named after them; a designed region adds
  its circuit's synapses and its populations as `region/population`; each projection is
  drawn between its ends.
- `cadence.genome.mutate(genome, rng, *, size_step=0.25, fixed=(), tied=())`: one offspring
  (module level; `evolve` uses it). Designed regions and regions in `fixed` keep their size;
  each `(leader, follower)` pair in `tied` keeps the follower the size of the leader.
- `evolve(fitness, genome, *, generations=10, population=8, keep=2, seed=0, mapper=map, report=None, **mutation) -> Lineage`:
  selection under `fitness(connectome, seed) -> float`. `mapper` runs a generation's
  developed connectomes (a process pool's `map` needs a picklable `fitness`); `report` is
  called with the lineage after every generation. `Lineage.best`, `.best_fitness`,
  `.generations`.

## Timing (`cadence.timing`)

- `latency(decide, *, repeats=1000, warmup=20)`: time `decide()` `repeats` times; the median,
  90th and 99th percentiles and maximum in microseconds, the mean, `jitter` (p99 over p50
  minus one), and the voluntary and involuntary context switches during the measurement.
- `environment()`: machine, cores, Python, thread limits, pinned cores (Linux), load average,
  library versions.

## Neuron-by-neuron reference (`cadence.reference`)

- `cadence.reference.settle_neuron_by_neuron(connectome, neuron_model, stimulus, *, steps, log_gain=None, bias=None, efficacy=None) -> (trajectory, Ledger)` (module level; `conformance` uses it):
  one neuron at a time, each reading only its own state and its synaptic input, counting one
  transmission per declared synapse per step.
- `conformance(brain, stimulus, *, steps=60) -> dict`: the brain against the reference on
  the same stimulus; `max_abs_deviation`, the ledger, the backend.
- `cadence.reference.Ledger`: `declared_synapses`, `steps`, `transmissions`, `undeclared`;
  property `clean`, also reported by `to_dict()`.

## Protocols (`cadence.protocol`)

- `Row(id, stimulus, readout, predicate, reference="", ablate=(), relative_to="", tier="experiment")`.
- `Protocol(stimuli, rows, training=(), levels=Levels(), steps=60)`: `score(brain)`,
  `neurons_for(connectome, stimulus)`, `to_dict()`. `stimuli` maps each stimulus name to the
  populations it drives at full amplitude.
- `cadence.protocol.Levels(active=0.5, inactive=0.2, margin=0.15, sparse_min=0.005, sparse_max=0.2, densify_margin=0.05)` (module level).
- `evaluate_predicate(predicate, value, reference, levels=None) -> bool`; `PREDICATES`
  maps each name to its definition.
- `shuffled(connectome, seed, *, keep=None) -> Connectome`: the control.
- `select_gain(make_brain, protocol, grid, *, sparsity_cap=0.05) -> (gain, table)`:
  among admissible gains, maximize training facts passed and break ties by smallest
  gain. `sparsity_cap=None` admits every gain. An empty grid or no admissible gain raises
  `ValueError`.

## Checkpoints (`cadence.checkpoint`)

- `save(learner, path) -> Path` and `load(path, *, backend=None, device=None, config=None, precision=None) -> Learner`,
  also as `Learner.save(path)` and `Learner.load(path, ...)`: one `.npz` file holding the
  connectome, every synapse's efficacy, every neuron's gain and bias, the neuron model, the
  configuration, the outputs and slots, the plasticity masks, tie groups, momentum and
  normalisation state, and the update count. `backend` and `device` may differ from the
  saved ones; `precision` overrides saved precision and `config` replaces the saved
  configuration. Inference through `predict` or `free` does not update parameters.
  Separate `FastSynapses`, `Trace` and `ActorCritic` objects are not saved by this API.
  Use `GenericBrain.save/load` for the full standard composition. Archive replacement is
  atomic, so a failed write leaves the previous checkpoint intact.

## Learning (`cadence.learning`)

- `LearnerConfig(beta=0.1, eta=0.2, eta_bias=0.02, centered=True, free_steps=100, nudged_steps=50, tolerance=1e-4, nudge="cross_entropy", temperature=0.2, normalize=0.0, normalize_floor=1e-3, momentum=0.0, decay=0.0)`:
  `momentum` steps each synapse on a running average of its own contrast; `decay` shrinks every
  plastic synapse's efficacy and every plastic neuron's bias by that fraction on each update
  (a leak on the synapses, for streams).
- `Learner(brain, outputs, config=LearnerConfig(), plastic_synapses=None, plastic_neurons=None, reciprocal=True, tie_groups=None, slots=1, updates=0, contrast_updates=0)`:
  `plastic_synapses` and `plastic_neurons` are bool masks over synapses and neurons; only those
  move and decay, so two learners can share one brain without one's decay eroding the other's
  synapses. With `reciprocal`, each reciprocal synapse pair shares one efficacy.
  `tie_groups` is an int per synapse (−1 for none); synapses in a group share one efficacy and
  move by the mean of their contrasts, which is how an embedding is shared across positions.
  Overlapping reciprocal/explicit ties form one group; arbitrary group IDs are compacted.
  Ties constrain increments, so initialize tied values equally to keep them equal. Frozen
  members keep their values, including under decay and clipping, and contribute zero to
  the group's mean increment. Reciprocal learning rejects ambiguous parallel pairs;
  merge those with `from_synapses`, or use `reciprocal=False`. Masks remain mutable;
  changing the original boolean array affects later updates. Rebuild the learner to
  change tie topology.
  `slots` splits the outputs into softmax groups (a count of equal groups, or one size per
  group); `updates` counts all applied updates, while `contrast_updates` counts only
  this learner's own optimizer history, excluding external reward/direct updates.
  - `free(drive, warm=None)`, `nudged(drive, free, target, sign=1.0, weight=None)`,
    `targets(labels)`, `nudge_for(target, beta, weight=None)`;
  - `contrast(free, nudged, opposite=None) -> (per_synapse, per_neuron)`,
    `update(free, nudged, opposite=None) -> {"scale_step", "bias_step"}`,
    `step(drive, labels, warm=None, weight=None) -> (LearnedState, report)`;
    labels are integer indices within each output group: `(batch,)` for one group,
    `(batch, slots)` for several; `accuracy` averages all row/slot choices;
  - `calibrate(drive, *, level=0.5, grid=None)`, `predict(drive)`, `accuracy(drive, labels, batch=256)`,
    `parameters()`, `to_dict()`; attributes `brain`, `reverse` (index of each synapse's
    reverse, or −1), `second_moment` (when normalising).
- `layered(inputs, hidden, outputs, *, density=0.3, feedback=1.0, lateral=0.0, seed=0, count=1.0, init=1.0, skip=False, skip_init=None, excitatory_forward=False) -> Connectome`
  with populations `input`, `hidden`, `output`. `skip_init=None` uses `init` for
  direct input-to-output projections. A finite nonnegative value overrides that
  scale; `skip=True, skip_init=0.0` adds trainable zero-efficacy projections while
  preserving the other effective parameters and initial predictions.
- `embedded(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0) -> (Connectome, tie_groups)`:
  a window of one-hot tokens through one embedding table shared across positions, then a
  dense hidden layer and the outputs, feedback synapses tied in pairs; populations `input`,
  `embedding`, `hidden`, `output`.
- `LearnedState(free, nudged, opposite=None)`.

## The agent and the valence (`cadence.plasticity`)

- `ActorCritic(learner, critic, config=None, seed=0, population=None)`: the agent of a stream
  of moments, the composition of the elements (`reward.md`). `learner` is a `Learner` whose
  outputs are the action neurons; `critic` the neurons whose settled activation, read through a
  learned linear readout, is the expectation of the reward to come; `population` a `Bins` for
  a continuous action.
  - `act(drive, greedy=False) -> action`: one free phase, then a draw from the softmax
    over the output neurons (with `Bins`, one draw per dimension), or the most probable.
    Cache reuse requires the same drive; caller buffers are copied. A greedy action clears
    pending eligibility and cannot be followed by `learn`. Repeated `act` replaces the
    pending decision. `Bins` requires at least two levels per dimension. Without
    `Bins`, `Learner(slots=[2, 3])` returns two categorical action indices per row;
    padding is never sampled. Actor nudges differentiate the softmax policy,
    independently of the learner's imitation loss;
  - `learn(reward, done, next_drive, bootstrap=None, *, observed=None) -> report`: the prediction error
    `reward + gamma * V(next) - V(now)` made into the dopamine by the valence and written
    through every synapse's eligibility, the trace of the last act's contrast decaying by
    `gamma * lam` a moment. The critic uses its own trace and `critic_signal`: raw
    prediction error (`"td"`) or modulated error (`"modulated"`, the default).
    Reports include absolute raw `td_error`, absolute modulated `delta`, and signed
    `dopamine`.
    `done` rows start their next life from rest; a truncated row passes `value_of` its last
    observation as `bootstrap`;
    `observed` is a boolean batch vector for real transitions. Padding rows do not
    teach the actor or critic or enter reward statistics; their eligibility resets.
    Updates average over observed rows. At least one row must be observed.
  - `reset()` (cached input/state, eligibility, salience and centering cleared; learned
    parameters and optimizer history retained), `probabilities(state)` (shape `(batch, actions)`
    or `(batch, slots, max_size)` for categorical slots, with exact zero padding;
    `(batch, dims, size)` with `Bins`), `settle(drive)`,
    `value(state)`, `value_of(drive)`, `parameters()`, `to_dict()`; the attributes `valence`,
    `salience`, `delta_mean`, `delta_var`.
- `ActorCriticConfig(gamma=0.99, lam=0.9, eta=0.5, eta_bias=0.05, eta_critic=0.05, normalize=0.0, momentum=0.0, dopamine_cap=1.0, dopamine_center=0.0, dopamine_floor=0.0, center_scale=True, critic_normalize=True, critic_signal="modulated")`:
  `gamma` the discount and `lam` the trace's decay; `eta` and `eta_bias` the actor's rates,
  `eta_critic` the critic's; `normalize` and `momentum` the adaptive local step, as the
  learner's; `dopamine_center` the rate at which the reward's running level and scale follow
  it (0 for no centring), `dopamine_floor` the band around the level, in scales, within
  which the dopamine is zero, `dopamine_cap` its cap, `center_scale` whether the surprise is
  measured in scales of the usual (`True`) or in the reward's own units; `critic_normalize`
  divides the critic's step by its trace's energy. `critic_signal="td"` keeps the
  critic target in reward units; `"modulated"` may change its fixed point through
  clipping or centring. See [the choice and its measured tradeoff](reward.md).
- `Bins(dims, size=9)`: the population code for `dims` continuous dimensions, each a softmax
  over `size` bins (`centres`, `groups`, `read`, `size`).
- `Valence(level=0.0, floor=0.0, cap=1.0, units=True, per_stream=True, mean=0.0, var=1.0)`: the
  reward less its expectation, made into the dopamine. Called on `delta` (a prediction error,
  or the reward alone), it subtracts the running level (`level` is the forgetting factor; 0 for
  none), kept per stream with `per_stream` or shared across streams otherwise; `mean` and
  `var` hold that running level and variance. The result is in the reward's own units or over
  its running scale (`units`), zero within `floor` scales of the level (quiet while the reward
  is what it usually is), and capped at `cap` (0 for no cap). `reset()`. `ActorCritic.valence`
  is the agent's, built per stream from `dopamine_center`, `dopamine_floor` and `center_scale`
  of its config; the cap is `dopamine_cap`, applied by `learn`.
  Calling `valence(delta, observed=mask)` excludes unobserved rows from the
  running statistics and returns zero for them. An all-false mask leaves it unchanged.
- `ActorCritic.state`: the free phase of the latest moment, the state `act` read or `learn`
  settled; `None` after `reset`.
- `ActorCritic.salience`: `(batch, neurons)`, set before `learn`; each synapse's eligibility is
  weighted by its pre neuron's entry (a `Trace.ringing`), so that what is still ringing is
  what a signal writes through. None by default.

## Receipts and custody (`cadence.receipts`, `cadence.custody`)

- `Receipt.build(kind, body, sources=()) -> Receipt`; `write(path)`; `Receipt.read(path)`;
  `Receipt.verify(path, *, sources=None, check=None) -> (ok, message)`; `to_dict()`.
- `canonical_json(value)`; `cadence.receipts.canonical_sha256(value)` and `cadence.receipts.source_manifest(files)` at module level.
- `Source(key, file, url, sha256, citation="")`, `fetch(sources, root, *, allow_download=False)`,
  `manifest(sources, extra=None)`, `sha256_of(path)`, `CustodyError`.

## Optional task compositions

`from cadence.circuits import assemble, reflex_arc, imagine, Deliberator, ActivityMonitor` imports small
sensorimotor, counterfactual-search and self-reading compositions. See [patterns](patterns.md)
for ports, budgets, supplied-model boundaries and examples. These optional architectural
helpers compose ordinary neuron dynamics with explicit host-side orchestration.

`assemble(regions, synapses=()) -> Connectome` merges an insertion-ordered mapping of
region names to connectomes into one connectome. Each entry of `synapses` is
`(source_region, local_neuron, target_region, local_neuron, weight)`, a directed synapse
with count 1 and sign `weight`. Each region remains addressable as a population, and each
of its populations as `region/population`. Use the result with one `Brain` and concatenate
drives in region insertion order. Unknown regions, neurons outside their region, autapses
and nonfinite weights are rejected. The helper adds topology only; convergence depends on
the combined system.


- `Deliberator(actions, transition, evaluate, terminal, *, depth=6, max_nodes=10000, adversarial=False, prune=False, clone=deepcopy)`:
  resumable iterative deepening over isolated futures. `start(live)` snapshots new input
  and replaces old work; `tick(nodes=128)` visits at most that many new positions and
  returns an isolated copy of the last completed `Deliberation`, or `None`.
  `pending` reports unfinished work; `pause()`/`resume()` preserve it; `cancel()` drops
  the continuation and obsolete result. `nodes` counts total work for this observation,
  `budget_exhausted` reports the hard limit, and `result` is the last completed depth.
  No work starts until the caller supplies input and ticks. Callbacks obey `imagine`'s
  isolation contract and remain fixed for one search; node limits do not preempt callbacks.
  This object stores no learned weights and writes no real-action feedback.
  See [defaults and scheduling](continuous.md#defaults-and-the-thinking-clock).
- `imagine(live, actions, transition, evaluate, terminal, *, depth=2, max_nodes=10000, adversarial=False, prune=False, clone=deepcopy) -> Deliberation`:
  compare copied futures. Scores use the root actor's perspective; adversarial layers
  alternate min/max. Pruning uses alpha-beta bounds with fresh bounds per root action.
  `clone` must isolate mutable branch state; other callbacks must not mutate external
  objects. Invalid budgets/nonfinite scores raise. Exhaustion raises before the next
  transition and returns no partial ranking. `Deliberation.futures` are sorted by score;
  each `Future` holds `action`, `score`, `sequence`, `state`. `nodes` counts visited
  successor states, and `depth` records the requested horizon.
- `ActivityMonitor().read(activity, scores, *, pressure=0) -> Readback`: finite 1D vectors
  and pressure in `[0, 1]`. Returns `activity_change`, `ambiguity`, `pressure`, `uncertainty`,
  `request_more`, `state`. `reset()` clears its previous reading and state. This heuristic
  does not learn by itself and does not establish consciousness.
- `reflex_arc(axes=2)`: sensory error ports and opposing motor pairs; the application
  supplies body dynamics and interprets the motor readout.

## Record every settling step

`record_settlements(callback, label="")` captures calls made inside its context,
including calls inside `Learner`, `ActorCritic` and supplied imagination routines.
The callback receives a `SettlementRecord` after each call. It contains the
connectome, neuron model, effective weights, bias, drive, mask, nudge and full
potential, activation and adaptation histories. Histories have shape
`(steps + 1, batch, neurons)`: the first row is the initial state, then one row
for every actual iteration. Differences of successive potentials are the signed
local repairs. Even a zero-step call has its initial row.

```python
records = []
with cd.record_settlements(records.append, label="observe and act"):
    result = brain.settle(stimulus={0: 1.0}, steps=32)
repairs = np.diff(records[0].potential, axis=0)
```

Recording is opt-in. It copies every neuron's state after every step and can be
expensive for large brains. Write bounded chunks from the callback rather than
keeping an entire long task in RAM. Callback diagnostics do not recursively
record themselves; nested recording contexts restore the previous callback on
exit. Callback failures propagate. State and parameter arrays own their storage;
subsequent learning cannot change those saved arrays. Treat the shared connectome
as read-only.

CPU recording uses the inspectable NumPy kernel rather than the fused path, so
round-off and wall time may differ. These are measurements of the recorded run,
not reconstructions of a previous unrecorded run. Iteration traces are simulated
neural activity, not measured biological EEG or neurotransmitter concentrations.

`ActorCritic.learn` reports signed mean `dopamine` alongside the existing mean
absolute `delta`. For a one-stream agent it is that transition's signed,
centered and capped learning signal. It is a global modulation signal; spatial
neurotransmitter diffusion is not part of this model.


## Deprecated names (`cadence.legacy`)

Every name of cadence 0.8 resolves for one release to its current counterpart with a
`DeprecationWarning` naming the replacement: top-level names such as `Wiring`,
`Settlement`, `GradedRule`, `FastSeams`, `Constitution`, `grow` and `learning_rule`; the
module paths `cadence.wiring`, `cadence.rules`, `cadence.settle`, `cadence.constitution` and
`cadence.brains`; keyword arguments such as `clamp=`, `sets=`, `edge_scale=`, `engine=`,
`trainable_overlaps=` and `symmetric=`; and attributes and methods such as `.sets`,
`.edges`, `.wiring`, `.rule`, `.repair`, `clamp_levels()` and `Trace.clamp()`. The module
docstring lists the full map. `cadence.legacy.OLD_NAMES` maps old names to
`(module, name)`.

## Bounded memory and rehearsal

- `ContentMemory(inputs, outputs, capacity, match=0.75, key_rate=0.1, value_rate=1.0)`:
  `select(cue)` returns slots/scores; `recall(cue)` reads without mutation;
  `observe(cue, value, write=None)` learns from observed values; `clear()` erases
  the shared store. Novel cues allocate or evict a slot. See [content memory](content_memory.md).
- `ReservoirReplay(capacity, inputs, seed=0)`: `sample(count)` returns owned prior
  feature/label rows; `observe(features, labels)` admits actual observations into
  a uniform bounded reservoir. Learning and checkpointing are caller-owned.
  See [rehearsal](replay.md) for information and storage costs.
- `cadence.sequence.SequenceCache(features, values, capacity=128, temperature=0.1,
  center_rate=0.02)`: per-stream content readback; call `reset(batch)`, then
  `read(features)` before `observe(features, observed_values)`.
  `SequenceRead` exposes value, entropy, maximum weight and record count.
- `cadence.sequence.BoundedTrace(width, decay=0.5, radius=1.0, center=True)`:
  `reset(batch)`, `observe(value)` and non-mutating `read()`. Readback has at most
  the declared L2 radius; this is no guarantee of better sequence prediction.
  See [sequence readback](sequence.md).
