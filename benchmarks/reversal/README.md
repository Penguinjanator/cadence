# Odour nursery

This bounded instrument asks whether one continuing `Brain.compose` life can acquire a
rule, adapt when the rule turns over unannounced, return when it turns back, and keep an
unrelated skill throughout, after short and after long experience of the first rule. It
is the reversal chamber of
[issue 88](https://github.com/muellerberndt/cadence/issues/88), roadmap row 05 of
[issue 109](https://github.com/muellerberndt/cadence/issues/109), and the first
behavioural evidence for the routine-and-repair loop of
[issue 122](https://github.com/muellerberndt/cadence/issues/122). It measures one declared
System 1 operating point with arousal against the same brain without it, the released
defaults, two ablations of the brain, frozen, replay and reset controls, a tabular
learner with the same information and uniform-random actions. It establishes no default. Read
[routine and repair](../../docs/continuous.md#routine-and-repair-live), the
[world-model guide](../../docs/world-model.md) and
[numerical contracts](../../docs/contracts.md) before interpreting results.

## What runs

An animal meets one of four odours per trial, drawn at random, and avoids (0) or
approaches (1). Approaching the sugar odour pays +1, approaching another odour costs 1,
avoiding pays nothing. Odours 0 and 1 are the reversal pair: the sugar sits at odour 0
under rule A and at odour 1 under rule B. Odours 2 and 3 are the stable pair, the
unrelated skill: odour 2 is always sugar and odour 3 never. A life is one stream without
resets: rule A for the exposure (100, 300, 1,000, 3,000 or 10,000 trials), rule B for 600
trials and rule A again for 600. Nothing announces a change. This is the two-odour T-maze
of the reports behind the issue ([50](https://github.com/muellerberndt/cadence/issues/50),
[69](https://github.com/muellerberndt/cadence/issues/69)), with the stable pair added.

Every arm of a seed lives the same odour sequence:

| Arm | What it is |
| --- | --- |
| `live` | `Brain.compose(4, 2, modules=(32,))` at the protocol's operating point, with `ArousalConfig()` at its founders, through `Brain.live` |
| `step` | the same brain and operating point without arousal, through `step`: it samples and learns at every moment (the simpler control) |
| `defaults` | `Brain.compose(4, 2, modules=(32,))` at the released defaults, through `step` |
| `memory-only` | the `live` brain with its actor's rates at zero: associative memory and critic learn, the graph's policy does not |
| `graph-only` | the `live` brain without its associative memory: the graph's reward learning alone |
| `frozen` | the `live` brain after rule A, answering greedily and receiving no outcome |
| `replay` | the `live` brain re-living rule A after the change: it is paid what its action earned under rule A, the same work on old evidence |
| `reset` | a newborn `live` brain at every rule change |
| `tabular` | epsilon-greedy tabular Q-learning (alpha 0.2, epsilon 0.1): the conventional online learner with the same odour, action and reward |
| `random` | uniform random actions |

The operating point of one continuing stream is working-trace amplitude 0.3,
consolidation 0.25 and actor rates 0.1 and 0.01, a tenth of the composed values. Its
selection on the development seeds is recorded
[below](#how-the-operating-point-and-the-founders-were-selected).

Readings per rule: the share of optimal executed actions in the last 100 trials; the lag,
the first trial from which the next 40 executed actions are at least 90% optimal; the
greedy choice per odour of a saved and reloaded copy every 25 trials; the first executed
approach at the newly rewarded odour and the trials from it to the greedy flip there,
which separate too few contradicting witnesses from a failure to revise after them;
whether the stable pair stayed right at the probes; the share of aroused moments; and the
free-solve sweeps per moment in each mode with the eligibility and feedback sweeps of
aroused moments. A refused answer raises and the life is recorded as crashed.

## Gates, fixed before the confirmation run

Over the confirmation lives of the `live` arm at exposures of 300 and more, at least 90%
end each rule with 90% of their last 100 executed actions optimal, keep the stable pair
right on 95% of their probes, and spend no more than 20% of the second half of each rule
aroused; at each of those exposures the median reversal lag is at most 150 trials, a life
that never reverses counting as beyond it. Exposure 100 ends inside the youth and is
reported without a gate. [protocol.json](protocol.json) holds the gates, the seeds and
every setting; it was committed before the confirmation seeds were run.

## Run and verify

```sh
python benchmarks/reversal/odour_nursery.py --seeds confirmation --out /tmp/nursery.json
python benchmarks/reversal/odour_nursery.py --report /tmp/nursery.json
python -m pytest -q benchmarks/reversal
```

The first command runs the frozen protocol: ten arms, five exposures and ten
confirmation seeds, 500 lives. `--arms`, `--seeds` and `--exposures` select a part.
`--genes`, `--point`, `--reliability`, `--payoff` and `--jitter` override the arousal
genes, the operating point and the world, and mark the receipt `frozen_protocol: false`.
`--report` prints the tables below from a receipt. The guards run short lives of the
`live` arm and its controls, the checkpoint continuation during a reversal and the gate
arithmetic. `evolve_genes.py` runs selection over the arousal genes with the founders as
the control and scores the winner on seeds that selection never saw.

<!-- RESULTS -->

## How the operating point and the founders were selected

Development used seeds 0 to 23 only; the confirmation seeds were first run on the frozen
protocol.

- **The released defaults lock one stream.** With the composed actor rate of 1.0, selected
  on batches of streams, the policy of one stream saturates on one action for every odour
  within about a hundred trials. A tenth of the rate leaves the choice to the evidence.
- **The default working trace outweighs the present input of a continuing life.** With
  amplitude 3.0 into the scale-12 prefrontal projection, 1% of the variance of the
  association state follows the present odour and 1% the previous one; the state follows
  its own history. At amplitude 0.3, 76% follows the present odour and 16% the previous
  one. Amplitude 0 and 0.3 gave the same nursery results.
- **Lasting memory took too little of a witnessed outcome.** At the default consolidation
  of 0.05 a unit outcome moves the lasting record by a tenth while the transient copy
  fades by a tenth with every other record. After 10,000 trials of rule A a life sampled
  the new sugar odour, was paid, and returned to avoiding it before the record had turned:
  a failure to revise after sufficient witnesses. At 0.25 the lasting record takes half of
  a unit outcome, and the lives of the development seeds reversed at every exposure.
  Consolidation 0.5 and 1.0 gave the same results in the reliable world; with a fifth of
  the outcomes withheld, 0.5 lost the stable pair on single withheld rewards and 0.25
  kept it in all but one life.
- **Whose outcomes enter the mood.** Letting every outcome enter it made the cost of
  exploring look like a shortfall and kept the brain awake: with the payoff as a cost
  (nothing for the right action, -1 for the wrong one, `--payoff cost`) the median
  reversal lag was 73 to 89 trials and 10% to 16% of the moments after a change were
  aroused. With only the outcomes of the brain's own greedy choices it was 17 to 25
  trials and 3% to 4%. In the sugar payoff the two rules did not differ.
- **The unit of reward.** A running RMS reward shrinks through a long calm at little
  reward and then makes small fluctuations look large. The unit is the spread of the
  outcomes the brain has learned from, which routine outcomes leave alone; the law is
  then unchanged by the scale and the zero of reward.
- **Arousal founders.** Removing the want left lives stuck after the reversal. With the
  hand-set heat of 1, 5 of 84 development lives at exposures of 300 and more ended a rule
  below the gate, having stopped searching before the moved sugar was found; with heat 2,
  and separately with a long-run rate of 0.002, none of 96 did. Heat 2 was kept as the
  founder: it spent the smaller share of moments aroused. Heat 3, a threshold of 0.3 and a
  faster recent rate were no better.

## What this does and does not establish

The `live` arm answers the two questions of issue 88 for this chamber. The choice that
must change is sampled again because a lasting shortfall of reward rouses the brain and
widens its sampling, and the witnessed outcome turns the lasting record because the
record takes half of it. The always-learning arm shows the historical failure on the same
sequences.

The associative memory carries the adaptation: `memory-only` matches `live`, and
`graph-only` does not acquire the task in one stream at these budgets. The odours are
one-hot and the memory is a direct record from sensory keys to action values, so this
chamber does not test the reciprocal graph's learned relations, generalization to unseen
inputs, delayed outcomes or short-term recall; those remain with issues
[110](https://github.com/muellerberndt/cadence/issues/110),
[111](https://github.com/muellerberndt/cadence/issues/111),
[84](https://github.com/muellerberndt/cadence/issues/84) and
[121](https://github.com/muellerberndt/cadence/issues/121).

The arousal law responds to change. In a world that does not change, noise in the reward
rate still rouses the founder genes for a small share of moments (the table reports it),
and a bout of needless exploration can lower the last 100 trials of a rule below the
gate. A brain whose life has always paid poorly, and whose youth has ended, is not roused.
The tabular learner needs no brain for a table of four odours; it is the matched-information
baseline, and its exploration is a fixed share of its actions for life.

Sweeps count numerical work. They are not wall time or energy.
