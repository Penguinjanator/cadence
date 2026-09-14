# Comparing possible futures

Deliberation is an architectural pattern built from Cadence's existing mechanisms.
It needs no new settlement law: bounded software patches hold local state, read
candidate observations at their ports, record predicted consequences, and use
feedback from observed outcomes to repair their predictions.

A useful composition has three parts:

- A **transition model** predicts what an action changes. It may be a supplied
  simulator, as in Connect Four, or a separately learned model.
- An **evaluator** reads a predicted state and estimates its benefit or cost. It
  can be a patch net trained with the ordinary free/nudged local update.
- A **controller** compares candidate sequences under a bounded search budget and
  executes only the first action of the preferred sequence.

[`examples/deliberation.py`](../examples/deliberation.py) provides an executable
composition. It copies the live state into independent branches, simulates a small
set of action sequences, and asks a learned evaluator about their terminal
observations. The evaluator has two output owners for positive and negative
outcomes. Thinking reads those owners; it does not train on invented observations.

```bash
python examples/deliberation.py
```

The supplied toy world offers an immediate temptation followed by a bad outcome,
or a delayed good outcome. Over 100 seeded trials, two-step deliberation selects
the good route in every trial, while a one-step view selects the temptation.
An exact evaluator also succeeds. Deliberation using an incorrect transition
model selects the wrong route in every trial. These controls locate the result:
the transition model and access to the delayed consequence matter; this is not
evidence that a patch net learns world dynamics or outperforms other planners.

## Prediction error after acting

After the real outcome arrives, compare it with the predicted return:

```text
prediction error = observed return - predicted return
```

A positive error means the outcome was better than expected; a negative error
means it was worse. This is the computational role of a dopamine-like teaching
signal, not a claim to reproduce biological dopamine. The example trains its
terminal evaluator from an observed outcome. `ActorCritic` additionally supports
reward-modulated action learning with a critic and eligibility traces. A predicted
reward inside a branch is a prediction, not new evidence about the real world.

## State and evaluation boundaries

Each branch must own its temporary state, including any trace arrays. A transition
must not mutate an external live object. The example's evaluator is a read-only
settlement; a stateful evaluator must keep its trace inside the branch too. Discard
unchosen branches. Carry the selected real experience forward, and save learned
parameters separately from transient traces.

`tests/test_deliberation.py` checks independent branch memory, unchanged live
state, no learner updates during imagination, the delayed-consequence result,
and failure with a wrong world model. The simple enumerator is deliberately
bounded to 4,096 branches; larger tasks need pruning or a different search method.

The historical Connect Four experiment uses this broad pattern with a supplied game simulator and
four-ply search. Its evaluator is a supplied threat heuristic with the
learned policy breaking ties. Its reported win rate therefore includes planning;
raw-policy and search-only controls are reported separately. The core example
above isolates the complementary case of a learned terminal evaluator.
