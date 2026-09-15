# Learning through experience

Design the brain and its learning life together. What can it observe and change?
What persists, what counts as evidence, and what makes a decision useful?

**Observe → remember → predict → act or communicate → learn from the outcome.**

There is one ongoing application loop, with no separate training/inference mode.
Thinking reads existing state; real observations, demonstrations and rewards
supply learning signals. A predicted event cannot become a witnessed fact.
Cadence implements observer-like software patches: bounded local state, declared
ports, readback, records and feedback/repair, with checkable evidence.

## Connect functions through actual ports

These six functions are a candidate architecture, not an anatomical parts list.

| Function | Build with | What must be learned or supplied explicitly |
|---|---|---|
| Perception and grounding | Input regions, associations, local updates | Features, entities and word/referent bindings |
| World model | Belief/action ports and predictive outputs | Action-conditioned consequences and uncertainty |
| Episodes and consolidation | `FastSynapses`, `SynapticMemory`, bounded event records | Addressing, relevant writes, retention and correction |
| Workspace and goals | Recurrent state, `Trace`, goal/readback ports | Persistent context, useful goals and subgoals |
| Action selection and value | `ActorCritic`, eligibility, `Valence`, optional `Deliberator` | Exploration, delayed credit and useful planning |
| Language and readback | Intention, language input/output and feedback | Expressing an intention, understanding replies and asking questions |

`assemble` or `Genome`/`develop` connect regions by declared synapses. Use one
`Brain` when they should settle jointly: independent solves merged in a picture
are not coupled dynamics. Auxiliary memories have separate state and clocks;
wire recall into declared drive ports and include those stores in cost accounting.
A named region has no function until its connections and experience make it useful.

`GenericBrain` supplies a recurrent sensory/association/motor policy, a critic,
optional working trace and associative reward memory. Its `hippocampus` remembers
chosen-action rewards for sensory cues. World models, learned hierarchical goals
and language require additional compositions and evidence.

## Keep the causal order

1. Receive an observation and feedback for the **preceding action**.
2. Score its saved prediction before revealing the outcome to the learner.
   Update from observed components; apply reward to the saved action eligibility.
3. Retain relevant actual events with their time, context and provenance.
4. Combine current evidence, remembered events and a goal. If imagining, use
   isolated branch state and read-only model parameters.
5. Save the next prediction, execute an action or utterance, and attend to the result.

A missing feature is not an observed zero. A UI tick is not another action outcome.
Replay may teach from retained real observations, but needs its own scheduling and
must not advance live episode state. Replaying obsolete evidence can undo a correction.
See [continuous interaction](continuous.md) for exact feedback and reset semantics.

Words can be learned in ambiguous situations and connected to remembered events.
Language is an input/output channel for the persistent system. Planning and speaking
can interleave; predicting observations or words can provide useful supporting
signals. A complete sentence need not be prepared before its first word.

## Two runnable examples

```bash
python examples/generic_brain.py
python examples/experience.py
```

**Ongoing reward:** one `GenericBrain.step` loop learns rewarded choices, encounters
a changed reward rule and adapts. No phase switch or teacher action is needed.

**Experience and planning:** three rooms, two actions and three synaptic stores.
The learner explores and predicts destinations before seeing them; hears one word
in two ambiguous scenes; sees a relevant object once; then acts on a request using
its acquired associations. Action outcomes keep teaching through the same operation.

The example removes event memory or word grounding and corrupts the learned model.
The corresponding route disappears or changes. These are interventions on copies
of the same acquired experience; they do not teach the live agent. The
[tests](../tests/test_experience.py) check the causal dependencies and isolation.

Learned quantities are transition values, word/object associations and a location.
Categorical perception, conjunction keys, request interpretation and search are
supplied. This small reference demonstrates integration, not learned perception,
recurrent representations, general reasoning, fluent language or human sample efficiency.

## Grow the experience with the brain

Start with consequences, occlusion and delayed goals. Add situated words,
demonstrations and corrections; then withheld combinations of objects, attributes
and relations. Add responsive partners, useful questions, descriptions and reading.
Measure transfer from those experiences into decisions and conversation.

The training unit is a trajectory: lifetime, episode, time, observation and masks,
heard utterance, action, pre-action prediction, outcome, reward and provenance.
A text corpus can extend experience; it does not supply goal formation or temporal
credit by itself. Distinguish directly observed events from a speaker's reports.

Withhold worlds, vocabulary mappings, compositions and complete lifetimes.
Measure learning over exposures. For adaptation measurements, report the allowed
feedback and update budget. For frozen measurements, use a separate snapshot.
Ablate memory, grounding, learned dynamics and active information gathering;
compare replay at equal compute. Supplied search and perception count in comparisons.

Learned latent states also need stable target representations while encoders change.
Local equilibrium repair alone does not solve arbitrary temporal credit assignment.
Test those mechanisms before increasing the number of regions, neurons or GPUs.

## State and evidence

Preserve private episode records, random state and pending actions per stream.
`SynapticMemory` explicitly shares slow weights across streams; a measurement
snapshot must not modify the live learner's shared matrix. Save the environment,
partner, curriculum and replay state alongside the brain when resuming a whole life.
`GenericBrain.save` covers its standard composition; custom stores have their own owner.

Check equation residuals for equilibrium claims and task outcomes for capability
claims. A settled system can be wrong. A unique attracting equilibrium erases its
initial condition; history then needs a trace, record or context in the drive.
Memory, reward and language functions do not follow from anatomical names.
Cadence does not model spikes, neurotransmitter chemistry or consciousness.

## Research basis

These sources motivate experiments, not a complete biological learning algorithm:

- [Smith & Yu (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2271000/): word/referent
  acquisition across ambiguous situations.
- [McClelland et al. (1995)](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf):
  complementary fast and slow learning systems, with retention/interference tradeoffs.
- [Schultz et al. (1997)](https://www.gatsby.ucl.ac.uk/~dayan/papers/sdm97.html):
  reward prediction errors; [Bellec et al. (2020)](https://www.nature.com/articles/s41467-020-17236-y):
  eligibility-based online recurrent learning. Cadence's rule is not e-prop.
- [Kuhl et al. (2003)](https://doi.org/10.1073/pnas.1532872100): live interaction
  supported learning in the studied infant phonetic-learning conditions.
- [Fedorenko et al. (2024)](https://www.nature.com/articles/s41586-024-07522-w):
  distinctions between communication and thought; [speech-planning experiments](https://pubmed.ncbi.nlm.nih.gov/26858678/)
  support incremental formulation.
