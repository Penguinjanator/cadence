"""Counterfactual reasoning is composed from isolated state and a learned readout."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "deliberation", Path(__file__).parents[1] / "examples/deliberation.py"
)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_delayed_consequence_and_wrong_world_model():
    result = m.run(trials=20)
    assert result["accuracy"]["two_steps"] == 1
    assert result["accuracy"]["one_step"] == 0
    assert result["accuracy"]["exact_value_control"] == 1
    assert result["accuracy"]["wrong_model"] == 0
    assert result["real_outcome_updates"] == 2
    assert result["observed_feedback"]["better_than_expected"] > 0
    assert result["observed_feedback"]["worse_than_expected"] < 0


def test_branches_do_not_share_live_or_other_branch_memory():
    live = {"memory": np.zeros(2), "trail": []}

    def step(state, action):
        state["memory"][action] += 1
        state["trail"].append(action)
        return state, 0.0, False

    futures = m.compare_futures(live, (0, 1), step, lambda s: float(s["memory"][1]), horizon=2)
    np.testing.assert_array_equal(live["memory"], [0, 0])
    assert live["trail"] == []
    assert len(futures) == 2
    assert {tuple(f["state"]["trail"]) for f in futures} == {(0, 1), (1, 1)}
    assert futures[0]["actions"] == (1, 1)


def test_imagination_reads_without_changing_learner():
    brain = m.Evaluator()
    before = brain.learner.brain.dense().copy()
    state = {
        "phase": 0,
        "safe": 1,
        "features": np.zeros(2),
        "intensity": 0.8,
        "temptation": 0.2,
        "trail": [],
    }
    m.compare_futures(state, (0, 1), m.transition, brain.value)
    np.testing.assert_array_equal(brain.learner.brain.dense(), before)
    assert brain.learner.updates == 0
    with pytest.raises(ValueError):
        m.compare_futures(state, (0, 1), m.transition, brain.value, horizon=20)


def test_unexpected_bad_outcome_reduces_its_predicted_value():
    brain = m.Evaluator()
    state = {"features": np.array([0.8, 0.0])}
    for _ in range(20):
        brain.observe_outcome(state, 1.0)
    before = brain.value(state)
    error = brain.observe_outcome(state, -1.0)
    assert error < 0
    assert brain.value(state) < before
