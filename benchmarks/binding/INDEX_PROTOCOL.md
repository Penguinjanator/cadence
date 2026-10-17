# The index memory: learned addresses over the brain's own moments

Declared 16 October 2026, before the first declared run, for issues
[#199](https://github.com/muellerberndt/cadence/issues/199) and
[#201](https://github.com/muellerberndt/cadence/issues/201). Instrument: `index_read.py`
(the memory), `index_runs.py` (the lives); contracts: `test_index_read.py`. Companion to
[token binding](BINDING_PROTOCOL.md), [succession](SEQUENCE_PROTOCOL.md),
[chained recall](CHAIN_PROTOCOL.md) and [fresh bindings](../relevance/BINDINGS_PROTOCOL.md).

## Question

A store read by the present event generalizes to bindings never trained when the question
shares an event with the stored moment (token binding, succession), and fails where it shares
nothing (fresh bindings, finite recall). Learned reads tried so far either saturate (the
bilinear read) or complete on a stored moment nearly at random (a read competing inside the
settle), and lose part of what the fixed read has. Does a memory that addresses each moment before its
settle, by the event heard plus a learned map of the event and the cortex of the moment
before, trained by the address lesson, keep the stored-event read's binding and succession,
learn the addresses the present does not share, and hold in a continuing life and at camera
width?

## Mechanism

| Part | Definition |
| --- | --- |
| slot | one per accepted observation and stream; with `capacity`, the oldest dropped first |
| key `K_t` | the fixed part of the moment's address: `heard * x_t` (`x_t` the observation, or its sparse code at camera width) |
| address `q_t` | `K_t + x_t W_x + c_{t-1} W_c + b`, computed before the moment's settle; `c_{t-1}` the settled association activity of the moment before |
| read | `p = softmax_i(q_t . K_i / T - recency * age_i)`; recall drive `alpha sum_i p_i V_i`, next drive `alpha sum_i p_i N_i` on two source regions with one-way synapses into the association region (initial scale 1.2) |
| value, next | the event as witnessed; the next slot's value, zero until it comes |
| settle | the library's own; the recall and next synapses learn by its contrast like the sensory ones |
| address lesson | after a lesson, `d = W_{source->association} (c_+ - c_-)` per source; slot scores `(d_r . V_i + d_n . N_i) / |d|`; `p* = softmax(score / 0.1 + log p)`; the map takes `0.01 * (K* - q_t)` on `[x_t, c_{t-1}, 1]` at the taught rows, `K* = sum_i p*_i K_i` |
| sparse code (camera width) | 1024 code neurons, each reading 32 of the inputs with weights of variance 1/32; 24 winners kept; unit length |

## Arms

| Arm | Genes |
| --- | --- |
| `index` | alpha 3, T 0.25, heard 1, recency 0.25, next on, value the event, address 0.01, address temperature 0.1, address prior 1, context 0; capacity 0 (12 in the continuing task); the sparse code in the wide task only |
| `heard` | the same with address 0: the address stays the event heard |

## Tasks, readings and gates

Each life is read at every budget on forks; the reading asserts nothing changes the life.
Development founders 0 to 5; lessons judged by the act (`learning_surprise`).

| Task | Instrument | Budgets | Gate per founder at the last budget | Development passes at |
| --- | --- | --- | --- | --- |
| `token` | `token_binding`'s trials, panels and conditions | 256, 1024 | held-out at least .75; slots erased at most 1/8 + .15; a donor's slots followed on at least .75 | 5 of 6 |
| `fresh` | `relevance/bindings` (3 bits, m64, the trace arm) through this instrument's runner | 512, 2048 | the runner's gate: held-out exact at least .75 with both carriers erased at chance | 4 of 6 |
| `succession` | `sequence_binding`'s trials and panels | 256, 1024 | held-out at least .75; slots erased within .15 of 1/8 | 5 of 6 |
| `continuing` | token binding without a reset between trials; the panel read after two earlier trials of the same life | 256, 1024 | held-out at least .75; slots erased at most 1/8 + .15 | 5 of 6 |
| `wide` | token binding with each token a fixed patch of 400 of 12,824 inputs, noise pixels at .01 each frame | 256, 1024 | held-out at least .75; slots erased at most 1/8 + .15 | 5 of 6 |
| `chain` | `chain_binding`'s trials and panels | 1024, 4096 | two-held at least .60 and one-held at least .75 | 5 of 6 |

The `heard` arm runs on `token`, `fresh` and `succession` as the address lesson's control. The
comparisons (the fixed stores, the composed graph, the supervised GRU) are the retained rows of
those instruments' own readings, not rerun.

## Confirmation

Founders 400 to 411 stay unopened until the development readings are in and this protocol's
arms are frozen. A task whose development passes is confirmed on them with the same arm, budgets
and gate; it passes at 10 of 12 founders (`fresh`: 8 of 12).

## Sizing before this note (founders outside the declared set; not declared)

Sizing on founders outside the declared set fixed the arm's genes and three parts of the
mechanism, each from a failure of the form before it: the address is computed before the
settle (a read competing inside the settle from rest completed on a stored moment nearly at
random); the answer's address is named among the moments the address attended to (without that
prior the lesson named the moment whose own value is the answer); the keys hold the fixed part
of the address (keys written with the learned map made the lesson chase its own output in a
continuing life). The readings are retained outside this repository.

## Declared expectations

- `token`, `succession`: at the gate from 256 rounds on most founders; the slots erased at
  chance. The working trace carries some facts on its own, near the edge of the erased
  criterion.
- `fresh`: the `index` arm at the gate by 2048 rounds; the `heard` arm reads the mixture of both
  facts and stays below it on some founders.
- `continuing`, `wide`: at the gate if the address and the code carry the binding; a failure
  says which (the readings' erased conditions).
- `chain`: below the gate. The second link's address depends on the question marker and the
  recalled token together, a selection the linear address map makes only through conjunctions
  in the cortex of the moment before.

## What this does not claim

A bounded component on supplied event streams with lessons at the answer only. A pass
establishes that a memory of the brain's own moments, addressed before each settle by the event
heard plus a learned map trained by the address lesson, answers bindings and successions never
trained, learns an address the question does not share, holds in a continuing life of three
trials and at camera width, on the founders named. It establishes no chained inference, no
sentence understanding, no runtime default and nothing about C64 transcription or spoken words,
which wait on their own instruments.

## Commands

From the repository root, with Cadence on the Python path, one BLAS thread and at most two
workers; `DIR` lies outside the repository.

```sh
for task in token fresh succession continuing wide chain; do
  PYTHONPATH=src python benchmarks/binding/index_runs.py --task $task --out DIR/$task \
    --arms index --seeds 0 1 2 3 4 5 --workers 2
done
for task in token fresh succession; do
  PYTHONPATH=src python benchmarks/binding/index_runs.py --task $task --out DIR/$task-heard \
    --arms heard --seeds 0 1 2 3 4 5 --workers 2
done
python -m pytest -q benchmarks/binding
```

## Amendment 1, after the development readings (16 October 2026)

The development readings are retained as measured, outside this repository. The development
selected `token`, `succession`, `continuing` and `fresh`; their confirmation on founders 400
to 411 is declared, with the arm, budgets and gate above. `chain` and `wide` are kept as
declared. The camera task, declared before its first run:

| Item | `camera` |
| --- | --- |
| world | 12,824 inputs: a 128 x 100 image plus 24 body inputs; each token a contiguous 20 x 20 block on its own cell of the image; an event lights its tokens' blocks; .01 of the inputs lit at random each frame |
| retina | a fixed first stage between the pixels and the brain: 744 local pooling units (8 x 8 windows at stride 4, the drive the window's mean), the units at or above .5 kept (at most 128), the code scaled to unit length, the 24 body inputs passed through: 768 units; the cortex's observation and the memory's key are this code |
| brain | the composition of the token task on 768 inputs: 49,152 sensory synapses |
| genes | the `index` arm's with `heard` 1.5 (unit-length codes: a query shares most of its code with its fact's and little with the other fact's); `code` 0 |
| arms | `index` and `heard` |
| budgets, panels, gate | the `wide` task's: 256 and 1024 rounds; train, held-out, three facts, new name; held-out at least .75 with the slots erased at most 1/8 + .15; development 5 of 6, confirmation on founders 400 to 411 at 10 of 12 |

The retina is a construction of the task, as a seeing brain's first visual stage is of its
anatomy: the claim is that a memory keyed on a compressed code of the scene, read by a cortex
on the same code, binds what the scene shows; nothing here learns the code.

```sh
PYTHONPATH=src python benchmarks/binding/index_runs.py --task camera --out DIR/camera \
  --arms index heard --seeds 0 1 2 3 4 5 --workers 2
```

## Amendment 2, the generic code in place of the retina (16 October 2026)

The camera task's retina is a stage of the task's world. The library carries a generic code
only: a fixed sparse code of the whole observation with no spatial assumption (1,024 code
neurons, each reading 32 random inputs with fixed weights, the largest positive sums kept,
unit length). Declared before its first run:

| Item | `camera` with the generic code |
| --- | --- |
| observation | the generic code of the frame (1,024 units, fan-in 32, 96 winners) plus the 24 body inputs: 1,048 units, the cortex's observation and the memory's key alike (`--generic 96`) |
| everything else | amendment 1's: the `index` arm, budgets, panels, gate; development 5 of 6 on founders 0 to 5 |

The sizing selected 96 winners; the development on founders 0 to 5 selected the arm, and its
confirmation on founders 400 to 411 is declared with the same gate. The readings are retained
outside the repository.

```sh
PYTHONPATH=src python benchmarks/binding/index_runs.py --task camera --out DIR/camera-generic \
  --arms index --seeds 0 1 2 3 4 5 --generic 96 --workers 2
```

## Amendment 3, the input schedule, the address step and the attention record (17 October 2026)

Three changes to the instrument; no gene, panel, budget or gate changes.

| Item | Change |
| --- | --- |
| input schedule (`wide`, `camera`) | the training frames take their noise from the training stream's generator alone; each reading panel takes noise of its own, drawn afresh from the founder and the panel at every reading, so every reading and every condition (intact, slots erased, trace erased) sees the same frames, and a reading changes no later training frame (`test_index_runs.py`) |
| address step | each taught row's rate is `min(0.01, 1 / (z . z))` on `z = [x_t, c_{t-1}, 1]`, the rows' steps averaged as before; a step whose map or address is not finite is refused and counted (`address_refused`). The declared tasks' inputs (at most 27 values, or a unit-length code with at most 24 body inputs) and an association region of 64 neurons whose activations lie below one keep `0.01 * (z . z)` below 0.92, so no row of them reaches the bound and the step is the earlier one bit for bit |
| attention record | the memory keeps the attention each read used (`attended`), recency included, recorded before the moment's write; `fresh_competition` reports it in place of a weight recomputed without the recency charge. The scores and the gates do not read it |

The readings taken before this amendment are kept as measured. In them the reading panels of
`wide` and `camera` drew their noise from the training stream's generator, so their conditions
saw different noise and the training after a reading followed that reading's draws; and the
fresh-bindings `p_asked` and `decided` omit the recency charge. `wide` and `camera` are read
again under this schedule before their gates are cited.
