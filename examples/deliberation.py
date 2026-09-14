"""Compare futures with a learned evaluator; learn only from observed outcomes.

An architectural pattern, not an additional Cadence learning rule. The transition
model here is supplied. Each imagined branch gets its own copied state.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from itertools import product
from typing import Any

import numpy as np

import cadence as cd


def compare_futures(
    live: dict[str, Any],
    actions: tuple[int, ...],
    transition: Callable[[dict[str, Any], int], tuple[dict[str, Any], float, bool]],
    value: Callable[[dict[str, Any]], float],
    *,
    horizon: int = 2,
    discount: float = 0.9,
) -> list[dict[str, Any]]:
    """Enumerate a small action tree. Value reads a candidate without training on it.

    ``live`` may include a Trace's arrays or other temporary state. The transition
    must mutate only the branch state it receives, not external shared objects.
    A transition reward is an immediate cost/benefit; value estimates the remaining
    consequence, including the final outcome if it has not entered reward yet.
    """
    if not actions or horizon < 1 or not 0 <= discount <= 1:
        raise ValueError("nonempty actions, positive horizon and discount in [0, 1] required")
    if len(actions) ** horizon > 4096:
        raise ValueError("this small explicit planner is limited to 4096 branches")
    futures = []
    for sequence in product(actions, repeat=horizon):
        state = deepcopy(live)
        score = 0.0
        elapsed = 0
        for action in sequence:
            state, reward, done = transition(state, action)
            score += discount**elapsed * reward
            elapsed += 1
            if done:
                break
        score += discount**elapsed * value(state)
        futures.append({"actions": sequence[:elapsed], "score": float(score), "state": state})
    return sorted(futures, key=lambda row: -row["score"])


class Evaluator:
    """A tiny patch net learns whether a predicted terminal observation is good or bad."""

    def __init__(self, seed: int = 0) -> None:
        wiring = cd.layered(2, 8, 2, density=1, seed=seed)
        config = cd.LearnerConfig(eta=3, eta_bias=0.03, temperature=0.1, tolerance=1e-4)
        self.learner = cd.Learner(
            cd.Settlement(wiring, cd.learning_rule(dt=1)), wiring.sets["output"], config
        )
        self.n = wiring.n

    def drive(self, features: np.ndarray) -> np.ndarray:
        return np.pad(features, ((0, 0), (0, self.n - 2)))

    def teach(self, features: np.ndarray, outcomes: np.ndarray) -> None:
        self.learner.step(self.drive(features), (outcomes < 0).astype(int))

    def value(self, state: dict[str, Any]) -> float:
        free = self.learner.free(self.drive(state["features"][None]))
        logits = free.activation[0, self.learner.output_index] / 0.1
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return float(p[0] - p[1])

    def observe_outcome(self, state: dict[str, Any], actual: float) -> float:
        """Report a signed prediction error and teach the observed terminal outcome."""
        error = actual - self.value(state)
        self.teach(state["features"][None], np.array([actual]))
        return error


def transition(state: dict[str, Any], action: int) -> tuple[dict[str, Any], float, bool]:
    """A supplied two-step world: the tempting route hides a bad terminal outcome."""
    state["trail"].append(action)
    if state["phase"] == 0:
        state["chosen"] = action
        state["phase"] = 1
        return state, float(state["temptation"] if action != state["safe"] else 0), False
    good = state["chosen"] == state["safe"]
    state["features"] = np.array(
        [state["intensity"] if good else 0, 0 if good else state["intensity"]]
    )
    state["phase"] = 2
    return state, 0.0, True  # terminal utility is evaluated separately


def run(seed: int = 0, trials: int = 100) -> dict[str, Any]:
    brain = Evaluator(seed)
    rng = np.random.default_rng(seed)
    features = np.eye(2)[np.arange(64) % 2] * rng.uniform(0.5, 1, (64, 1))
    outcomes = np.where(np.arange(64) % 2 == 0, 1, -1)
    for _ in range(100):
        brain.teach(features, outcomes)
    wins = {"one_step": 0, "two_steps": 0, "exact_value_control": 0, "wrong_model": 0}
    errors = []
    for _ in range(trials):
        live = {
            "phase": 0,
            "safe": int(rng.integers(2)),
            "features": np.zeros(2),
            "intensity": float(rng.uniform(0.6, 1)),
            "temptation": float(rng.uniform(0.05, 0.3)),
            "trail": [],
        }
        wrong = {**live, "safe": 1 - live["safe"]}

        def exact(state: dict[str, Any]) -> float:
            return float(np.sign(state["features"][0] - state["features"][1]))

        for arm, initial, horizon, evaluator in (
            ("one_step", live, 1, brain.value),
            ("two_steps", live, 2, brain.value),
            ("exact_value_control", live, 2, exact),
            ("wrong_model", wrong, 2, brain.value),
        ):
            candidate = compare_futures(initial, (0, 1), transition, evaluator, horizon=horizon)[0]
            action = candidate["actions"][0]
            wins[arm] += action == live["safe"]
            if arm == "two_steps":
                actual = 1.0 if action == live["safe"] else -1.0
                predicted = brain.value(candidate["state"])
                errors.append(actual - predicted)  # signed outcome prediction error
        assert live["phase"] == 0 and live["trail"] == []
    # Only an observed terminal outcome supplies a teaching update, never a hypothetical one.
    updates_before = brain.learner.updates
    observed = {"features": np.array([0.8, 0.0])}
    positive_error = brain.observe_outcome(observed, 1.0)
    negative_error = brain.observe_outcome(observed, -1.0)
    return {
        "trials": trials,
        "correct": wins,
        "accuracy": {k: v / trials for k, v in wins.items()},
        "observed_feedback": {
            "better_than_expected": positive_error,
            "worse_than_expected": negative_error,
        },
        "mean_outcome_prediction_error": float(np.mean(errors)),
        "real_outcome_updates": brain.learner.updates - updates_before,
        "boundary": (
            "Supplied transition model; learned terminal evaluator; "
            "exact-value and wrong-model controls. "
            "Not a learned world model or a general planning advantage."
        ),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
