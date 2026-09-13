# API reference

Top-level exports and module-qualified helpers are listed below. Begin with the
[quickstart](quickstart.md) for complete runnable examples; the source docstrings
give additional details. Prefer keyword arguments for optional configuration.

## Wiring (`cadence.wiring`)

- `Wiring.from_edges(n, *, pre, post, count=None, sign=None, sets=None, label="wiring", min_count=0.0)`:
  build from edge lists. Parallel overlaps merge (counts add), autapses drop, overlaps
  below `min_count` drop, and the arrays are sorted by `(post, pre)`. `count` defaults to
  1 and `sign` to +1.
- Fields: `n`, `pre`, `post`, `count`, `sign` (arrays), `sets` (name → tuple of owners),
  `label`. Property `edges`.
- `members(*names)`, `with_sets(**sets)`, `in_degree()`, `out_degree()`, `summary()`,
  `digest()` (SHA-256 of the sorted arrays).

## Rules (`cadence.rules`)

- `GradedRule(dt=0.2, slope=4.0, threshold=1.5, gain=0.02, clamp_amplitude=3.0, adaptation=None, leak=0.0)`:
  the owner rule. `activation(v)` is the rectified, re-based sigmoid with optional leak;
  `slope_at(v)` its derivative; `rest_emission` the raw sigmoid's value at rest, which is
  subtracted so rest publishes zero. `replace(**changes)`, `to_dict()`.
- `Adaptation(tau_steps=50.0, strength=1.0)`: a slow per-owner variable that can
  produce rhythm in suitable circuits.
- `learning_rule(gain=1.0, *, slope=1.0, leak=0.1, dt=0.5, clamp_amplitude=1.0)` (in
  `cadence.learning`): the rule settings a learnable net needs.

## Settlement (`cadence.settle`)

- `Settlement(wiring, rule, *, backend="cpu" | "torch" | "mlx", edge_scale=None, log_gain=None, bias=None, device=None, dense_limit=2048, layout=None, precision=None)`:
  the engine. `edge_scale` defaults to the wiring's signs; `log_gain` and `bias` to zero.
  When the wiring's dense blocks fit in `dense_limit` squared entries the transport is the
  block transport (see `cadence.blocks`); `layout` passes a precomputed cut, as
  `with_parameters` does. `engine.layout` is the cut in use. `precision` (torch only) is
  `"float32"` or `"float64"`; the default is float64 except on MPS. Float32 is the speed of a
  consumer GPU; compare it with float64 and record the measured precision error.
- `settle(clamp=None, *, steps=60, state=None, mask=None, trajectory=False, nudge=None, tolerance=None) -> SettledState`:
  one clamp; `clamp` is a list of owners at full amplitude, a `{owner: level}` map, or a
  dense vector. Map values are levels multiplied by `clamp_amplitude`; dense vectors
  are drives directly. `settle_batch(drive, ...)` takes `(batch, n)` drives.
  `mask` accepts `(n,)`, `(1, n)`, or `(batch, n)`; zero suppresses an owner.
  Both methods stop at the activation-movement `tolerance` and report steps taken.
- `residual(drive, state, *, nudge=None, mask=None)`: per-row maximum remaining
  fixed-point equation discrepancy, including adaptation when enabled. One CPU transport
  evaluation, no state change; a diagnostic rather than a stability/uniqueness proof.
- `clamp_vector(clamp)`, `clamp_levels(levels)` (levels in [0, 1] times the clamp amplitude),
  `readings(state, names)`, `with_parameters(*, edge_scale, log_gain, bias)`, `weights`
  (effective drive per overlap), `dense()` (the `W[pre, post]` matrix), `to_dict()`.
- `Settlement.contrast_on_device(plus, minus)`: the learning rule's per-overlap and per-owner
  contrast computed on the device when both states carry its handle; `None` otherwise.
- `SettledState`: `v`, `activation`, `adaptation`, `steps`, `trajectory`, `device` (the
  same state on the accelerator that produced it, or `None`), `repair` (the total
  movement of the activations, per row). State arrays have shape `(n,)` from `settle`
  or `(batch, n)` from `settle_batch`; trajectories add a leading step axis. `row(i)`,
  `mean(members, i)`, `fraction_active(members, level, i)`, `active(level, i)`, `batched`.
- `Nudge(target, mask, beta, softmax_temperature=None, weight=None, groups=None)`: extra drive
  `beta · (target − s)` on the masked owners, or `beta · (target − softmax(s/T))` over the
  masked group with a temperature; `weight` scales rows. `groups` assigns a separate
  softmax group per owner (`-1` excludes an owner). `drive(s)`.
- `available_backends()`: `{"cpu": "numpy float64", "torch": "mps float32" | "cuda float64" | "cpu float64", "mlx": "gpu float32"}`, for what is installed.

## Blocks (`cadence.blocks`)

- `layout(wiring, *, max_pairs=256) -> Layout`: cut the owners into contiguous ranges at the
  boundaries of the wiring's contiguous named sets and pair the ranges that carry overlaps.
  A wiring with no contiguous sets, or one that fragments into more than `max_pairs` blocks,
  gets one block: the full matrix.
- `Layout`: `starts`, `pair_pre`, `pair_post`, `offset`, `edge_index`; `ranges`, `pairs`,
  `size`, `bounds(k)`, `sources()` (ranges that hear nothing), `flat(weights)`,
  `blocks(flat)`, `to_dict()`.
- `BlockTransport(layout, flat)`: `inbox(s)` for one settlement, reusing the product of every
  source range that did not move since the previous call.
- `block_contrast(layout, s_plus, s_minus)`: the learning rule's per-overlap contrast as one
  Gram product per block.

## Streams (`cadence.stream`)

- `stateful(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0, context_init=1.0) -> (Wiring, tie_groups)`:
  `embedded` plus a `context` range of `hidden` owners that hear nothing and reach every
  hidden owner. Sets `input`, `context`, `embedding`, `hidden`, `output`.
- `Trace(wiring, decay=0.5, amplitude=1.0, focus=0.0, source="hidden", target="context")`:
  the memory of the moment before, per stream: the trace of the `source` range's activation,
  entering the next settlement as a clamp on the paired `target` range (one owner per source
  owner). `focus` above zero weights each owner by its movement since the last moment (its
  share of the row's mean movement, to that power): brightest where the moment changed.
  `reset(batch, rows=None)`, `clamp(drive)`, `update(state)`, `keep(rows)`, `ringing(floor=0.1)`
  (each source owner's share of what is still ringing, one elsewhere: a salience for
  `ActorCritic.salience`), `to_dict()`.
- `Echo(wiring, decay, amplitude)`: the `Trace` at focus 0 into the `context` range (the
  carried state of the earlier releases, unchanged).
- `Afterglow(wiring, decay, amplitude, focus=1.0, source="hidden", target="afterglow")`: the
  focused `Trace`; with `source="input"` an afterimage of the picture itself, the memory that
  reads a cue against a static background (`tests/test_child.py`: 1.00 where the Echo reads
  chance).
- `FastSeams(pre, post, decay=1.0, rate=1.0, amplitude=1.0, normalize=False, replace=False, rule="hebb")`:
  one mutable `(pre, post)` matrix per stream. `rule="delta"` uses unit keys and writes
  `rate * outer(key, value - prediction)`; it rejects `normalize`/`replace` and rates
  outside `[0, 1]`. The default preserves additive Hebbian memory, including optional
  count-averaged reads (`normalize`) and one-hot row replacement (`replace`).
  `observe(key, value, write=None)` uses `(batch, width)` ports and writes all rows unless
  given a boolean mask; `recall(key)` reads without decay. `update(state, write=None, post=None)`
  instead reads named owner activations and defaults to decay-only; `read(drive)` reads
  key owners' drive columns. `clamp(drive, inplace=False)` adds the read into post columns.
  `reset(batch, rows=None)`, `keep(rows)`, `to_dict()`. See [memory](memory.md) for stream
  identity, representation alignment, key interference, and checkpoint boundaries.
- `cadence.stream.columns(index)`: a slice when the owners are one contiguous range,
  else the index array. Slices can avoid the copies required by advanced indexing.

## Constitution (`cadence.constitution`)

- `Region(name, size)`, `Projection(pre, post, density=1.0, sign=0.0, scale=1.0, count=1.0, symmetric=True)`,
  `Constitution(regions, projections, label)`: a wiring before it is grown.
- `grow(constitution, seed=0) -> Wiring`: development, deterministic in the seed; sets named
  after the regions, contiguous.
- `cadence.constitution.mutate(constitution, rng, *, size_step=0.25, fixed=())`: one offspring (module level; `evolve` uses it).
- `evolve(fitness, constitution, *, generations=10, population=8, keep=2, seed=0, **mutation) -> Lineage`:
  selection under `fitness(wiring, seed) -> float`; `Lineage.best`, `.best_fitness`, `.generations`.

## Timing (`cadence.timing`)

- `latency(decide, *, repeats=1000, warmup=20)`: time `decide()` `repeats` times; the median,
  90th and 99th percentiles and maximum in microseconds, the mean, `jitter` (p99 over p50
  minus one), and the voluntary and involuntary context switches during the measurement.
- `environment()`: machine, cores, Python, thread limits, pinned cores (Linux), load average,
  library versions.

## Reference engine (`cadence.reference`)

- `cadence.reference.settle_owner_by_owner(wiring, rule, clamp, *, steps, log_gain=None, bias=None, edge_scale=None) -> (trajectory, Ledger)` (module level; `conformance` uses it):
  one owner at a time, reading only its own row and its inbox slice, counting one delivery
  per declared overlap per step.
- `conformance(engine, clamp, *, steps=60) -> dict`: the engine against the reference on
  the same clamp; `max_abs_deviation`, the ledger, the backend.
- `cadence.reference.Ledger`: `declared_overlaps`, `steps`, `deliveries`, `undeclared`; `clean` in `to_dict()`.

## Protocols (`cadence.protocol`)

- `Row(id, stimulus, readout, predicate, reference="", ablate=(), relative_to="", tier="experiment")`.
- `Protocol(stimuli, rows, training=(), levels=Levels(), steps=60)`: `score(engine)`,
  `clamp_for(wiring, stimulus)`, `to_dict()`.
- `cadence.protocol.Levels(active=0.5, inactive=0.2, margin=0.15, sparse_min=0.005, sparse_max=0.2, densify_margin=0.05)` (module level).
- `evaluate_predicate(predicate, value, reference, levels=None) -> bool`; `PREDICATES`
  maps each name to its definition.
- `shuffled(wiring, seed, *, keep=None) -> Wiring`: the control.
- `select_gain(make_engine, protocol, grid, *, sparsity_cap=0.05) -> (gain, table)`:
  among admissible gains, maximize training facts passed and break ties by smallest
  gain. An empty grid or no admissible gain raises `ValueError`.

## Checkpoints (`cadence.checkpoint`)

- `save(learner, path) -> Path` and `load(path, *, backend=None, device=None, config=None, precision=None) -> Learner`,
  also as `Learner.save(path)` and `Learner.load(path, ...)`: one `.npz` file holding the wiring,
  every seam's scale, every owner's gain and bias, the rule, the configuration, the masks, tie
  groups, momentum and normalisation state, and the update count. `backend` and `device` may
  differ from the saved ones; `precision` overrides saved precision and `config`
  replaces the saved configuration. Inference through `predict` or `free` does
  not update parameters. Separate `FastSeams` and `Trace` objects are not saved.

## Learning (`cadence.learning`)

- `LearnerConfig(beta=0.1, eta=0.2, eta_bias=0.02, centered=True, free_steps=100, nudged_steps=50, tolerance=1e-4, nudge="cross_entropy", temperature=0.2, normalize=0.0, normalize_floor=1e-3, momentum=0.0, decay=0.0)`:
  `momentum` steps each seam on a running average of its own contrast; `decay` shrinks every
  trainable seam and bias by that fraction on each update (a leak on the seams, for streams).
- `Learner(engine, outputs, config=LearnerConfig(), trainable_overlaps=None, trainable_owners=None, symmetric=True, tie_groups=None, slots=1)`:
  `trainable_overlaps` and `trainable_owners` are bool masks over overlaps and owners; only those
  move and decay, so two learners can share one net without one's decay eroding the other's seams;
  `tie_groups` is an int per overlap (−1 for none); overlaps in a group share one scale and
  move by the mean of their contrasts, which is how an embedding is shared across positions;
  - `free(drive, warm=None)`, `nudged(drive, free, target, sign=1.0, weight=None)`,
    `targets(labels)`, `nudge_for(target, beta, weight=None)`;
  - `contrast(free, nudged, opposite=None) -> (per_overlap, per_owner)`,
    `update(free, nudged, opposite=None) -> {"scale_step", "bias_step"}`,
    `step(drive, labels, warm=None, weight=None) -> (LearnedState, report)`;
    labels are integer indices within each output group: `(batch,)` for one group,
    `(batch, slots)` for several; `accuracy` averages all row/slot choices;
  - `calibrate(drive, *, level=0.5, grid=None)`, `predict(drive)`, `accuracy(drive, labels, batch=256)`,
    `parameters()`, `to_dict()`; attributes `engine`, `reverse` (index of each overlap's
    reverse, or −1), `second_moment` (when normalising).
- `layered(inputs, hidden, outputs, *, density=0.3, feedback=1.0, lateral=0.0, seed=0, count=1.0, init=1.0, skip=False, excitatory_forward=False) -> Wiring`
  with sets `input`, `hidden`, `output`.
- `embedded(vocabulary, positions, dim, hidden, outputs, *, seed=0, init=1.0) -> (Wiring, tie_groups)`:
  a window of one-hot tokens through one embedding table shared across positions, then a
  dense hidden layer and the outputs, feedback seams tied in pairs; sets `input`,
  `embedding`, `hidden`, `output`.
- `LearnedState(free, nudged, opposite)`.

## The agent and the valence (`cadence.plasticity`)

- `ActorCritic(learner, critic, config=None, seed=0, population=None)`: the agent of a stream
  of moments, the composition of the elements (`reward.md`). `learner` is a `Learner` whose
  outputs are the action owners; `critic` the owners whose settled activation, read through a
  learned linear readout, is the expectation of the reward to come; `population` a `Bins` for
  a continuous action.
  - `act(drive, greedy=False) -> action`: one free settlement, then a draw from the softmax
    over the output owners (with `Bins`, one draw per dimension), or the most probable;
  - `learn(reward, done, next_drive, bootstrap=None) -> report`: the prediction error
    `reward + gamma * V(next) - V(now)` made into the dopamine by the valence and written
    through every seam's eligibility, the trace of the last act's contrast decaying by
    `gamma * lam` a moment; the critic's readout moves by its own trace and the same error.
    `done` rows start their next life from rest; a truncated row passes `value_of` its last
    observation as `bootstrap`;
  - `reset()` (the traces cleared, at a life's end), `probabilities(state)`, `settle(drive)`,
    `value(state)`, `value_of(drive)`, `parameters()`, `to_dict()`; the attributes `valence`,
    `salience`, `delta_mean`, `delta_var`.
- `ActorCriticConfig(gamma=0.99, lam=0.9, eta=0.5, eta_bias=0.05, eta_critic=0.05, normalize=0.0, momentum=0.0, dopamine_cap=1.0, dopamine_center=0.0, dopamine_floor=0.0, center_scale=True, critic_normalize=True)`:
  `gamma` the discount and `lam` the trace's decay; `eta` and `eta_bias` the actor's rates,
  `eta_critic` the critic's; `normalize` and `momentum` the adaptive local step, as the
  learner's; `dopamine_center` the rate at which the reward's running level and scale follow
  it (0 for no centring), `dopamine_floor` the band around the level, in scales, within
  which the dopamine is zero, `dopamine_cap` its cap, `center_scale` whether the surprise is
  measured in scales of the usual (`True`) or in the reward's own units; `critic_normalize`
  divides the critic's step by its trace's energy.
- `Bins(dims, size=9)`: the population code for `dims` continuous dimensions, each a softmax
  over `size` bins (`centres`, `groups`, `read`, `size`).
- `Valence(level=0.0, floor=0.0, cap=1.0, units=True)`: the reward less its
  expectation, made into the dopamine. Called on `delta` (a prediction error, or the reward
  alone): less its running level per stream (`level` is the forgetting factor; 0 for none),
  in the reward's own units or over its running scale (`units`), nothing within `floor`
  scales of the level (quiet while the reward is what it usually is), capped at `cap`.
  `reset()`. `ActorCritic.valence` is the agent's, built from `dopamine_center`,
  `dopamine_floor` and `center_scale` of its config, per stream always; the cap is
  `dopamine_cap`, applied by `learn`.
- `ActorCritic.salience`: `(batch, owners)`, set before `learn`; each seam's eligibility is
  weighted by its pre owner's entry (a `Trace.ringing`), so that what is still ringing is
  what a signal writes through. None by default.

## Receipts and custody (`cadence.receipts`, `cadence.custody`)

- `Receipt.build(kind, body, sources=()) -> Receipt`; `write(path)`;
  `Receipt.verify(path, *, sources=None, check=None) -> (ok, message)`; `to_dict()`.
- `canonical_json(value)`; `cadence.receipts.canonical_sha256(value)` and `cadence.receipts.source_manifest(files)` at module level.
- `Source(key, file, url, sha256, citation="")`, `fetch(sources, root, *, allow_download=False)`,
  `manifest(sources, extra=None)`, `sha256_of(path)`, `CustodyError`.
