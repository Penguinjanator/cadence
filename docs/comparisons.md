# Measured tradeoffs

Cadence combines local neuron state, synaptic transport, readback and plasticity.
The useful question is which task benefits from that combination at a stated
quality, memory and computation budget. These controlled comparisons retain
every scheduled seed, conventional controls and frozen executable sources.

| Question | Measured result | What the comparison establishes |
|---|---|---|
| Can runtime overhead be reduced? | Median old/new time ratios for 60 fixed steps are 1.13 on CPU and 2.91 on Apple MPS; five centered updates give 1.10 and 2.18. | Same Cadence workload and numerical budgets, five seeds. These are implementation speedups, not comparisons against an MLP. [Protocol and receipts](../benchmarks/README.md). |
| Does the improvement save GPU energy? | On one isolated A10G, five centered updates use about 16% fewer raw device joules. | GPU energy only. Idle-corrected cold-residual energy increases in all five seeds. Host/system energy and matched task quality are outside this operation benchmark. [Measurements](../benchmarks/device_energy/README.md). |
| Can one head retain earlier classes? | Final split-MNIST accuracy rises from 0.1922 to 0.7792 with a 256-row reservoir. | Earlier real targets matter. Rehearsal adds storage and training examples; repeating current examples does not produce this gain. The fixed-rate Adam control is untuned. [Memory comparisons](../experiments/memory_repair/README.md). |
| Does rehearsal prevent plasticity drift? | It improves ten-task retention, while current-task accuracy declines. Thirty-task reset controls identify no reliable simple reset remedy. | Retaining old skills and continuing to learn new ones are separate measurements. [Drift diagnostic](../experiments/memory_repair/drift.json). |
| Can memory select by content? | Learned prototypes score 1.000 on noisy cues; fixed prototypes score 0.9981. Identical observable cues with conflicting outcomes yield 0.5. | Selection works within declared feature coordinates. It does not learn a general encoder or recover information absent from the cue. [Content memory](content_memory.md). |
| Do local reward updates handle delay? | A small delayed-choice circuit with a direct trainable route reaches greedy accuracy 1.0 across five seeds at episode lengths 3, 9 and 33. | The conventional score-rule control also succeeds and has better long-delay stochastic accuracy. This is a bounded task, not a solution to general temporal credit. [Wiring comparison](../benchmarks/sensorimotor_skip/README.md). |
| Can a policy use earlier observations? | On the actual four-suit repeat-card task, a three-stage sensory register lets both patch policies and a smaller linear policy reach return 1.0 in all five seeds. Their memory-masked controls remain near chance. | The register is an explicit controller component outside the neural solve. This isolates a useful history representation, with no learned address or general delayed-reward claim. [Task and controls](../experiments/temporal_address/README.md). |
| Does recurrence solve language modelling? | On the primary text excerpt, the recurrent patch scores 3.7443 bits/character and its content cache 3.7130; the windowed MLP scores 3.3228. Lower is better. | The language gap persists. A separate excerpt supports probability readback over current injection and uniform recent-token counts; its scores cannot be compared directly to the primary MLP scores. [Text experiments](../experiments/sequence_readback/README.md). |

These experiments use five seeds each, but small fixtures and one dataset or
hardware instance do not establish broad generalization. Runtime ratios are
medians of paired ratios; language deviations in the experiment reports are
sample standard deviations across initializations, not uncertainty across books.

Two distinctions matter when designing the next experiment:

- **Convergence and task accuracy:** a small equation residual means the declared
  dynamics are self-consistent. It does not mean the answer is correct or that
  semantic surprise determines the work. A valid contraction certificate adds
  a distance bound; an uncertified residual does not establish uniqueness.
- **Structure and learning rule:** a supplied address, controller history, lookup
  store, simulator or direct motor route contributes information or computation.
  Give a conventional control the same structure and count all mutable memory,
  settling phases, validation work and additional training rows.

See [brain design](design.md), [reward learning](reward.md) and
[convergence conditions](certificate.md) for the implementation contracts.
