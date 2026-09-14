from __future__ import annotations

import copy
import importlib
from pathlib import Path

import numpy as np


def test_diagnostic_resets_partition_head_and_representation(monkeypatch) -> None:
    folder = Path(__file__).resolve().parents[1] / "experiments/memory_repair"
    monkeypatch.syspath_prepend(str(folder))
    drift = importlib.import_module("drift")
    initial = drift.continual.Patch(17)
    model = copy.deepcopy(initial)
    brain = model.learner.brain
    model.learner.brain = brain.with_parameters(
        efficacy=brain.efficacy + 0.5, bias=brain.bias + 0.1
    )
    edge_mask, neuron_mask = drift.masks(model)
    for arm, edge, neuron in [
        ("reset_head", edge_mask, neuron_mask),
        ("reset_representation", ~edge_mask, ~neuron_mask),
    ]:
        reset = drift.reset(model, initial, arm).learner.brain
        assert np.array_equal(reset.efficacy[edge], initial.learner.brain.efficacy[edge])
        assert np.array_equal(reset.efficacy[~edge], model.learner.brain.efficacy[~edge])
        assert np.array_equal(reset.bias[neuron], initial.learner.brain.bias[neuron])
        assert np.array_equal(reset.bias[~neuron], model.learner.brain.bias[~neuron])
    reset = drift.reset(model, initial, "reset_all")
    assert np.array_equal(reset.learner.brain.efficacy, initial.learner.brain.efficacy)
    assert np.array_equal(reset.learner.brain.bias, initial.learner.brain.bias)
    # The untouched branch and parent must retain all trained parameters.
    carried = drift.reset(model, initial, "carried")
    assert np.array_equal(carried.learner.brain.efficacy, model.learner.brain.efficacy)
    carried.learner.brain.efficacy.fill(0)
    assert np.any(model.learner.brain.efficacy != 0)
