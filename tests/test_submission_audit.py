"""Adversarial regressions for the submission's numerical and store boundaries."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd
from cadence.temporal import contrast_asymmetry


def test_parameter_replacement_cannot_overflow_effective_weights():
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1], count=[1e100])
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model())
    before = brain.weights.copy()
    with pytest.raises(ValueError, match="effective synaptic weights"):
        brain.with_parameters(efficacy=np.array([1e300]))
    with pytest.raises(ValueError, match="effective synaptic weights"):
        brain.efficacy = np.array([1e300])
    np.testing.assert_array_equal(brain.weights, before)


def test_large_finite_protected_input_does_not_overflow_rank_threshold():
    net = cd.TemporalPatchNet(1, 1, 1)
    net.set_parameters({"A": np.zeros((1, 1)), "B": np.ones((1, 1)), "C": np.ones((1, 1))})
    inputs = np.array([[[1e200]]])
    original = net.imagine(inputs).output
    memory = cd.TemporalMemory()
    report = memory.protect(net, inputs)
    assert report.ranks["B"] == 1 and report.maximum_residual == 0
    before = net.parameters()
    proposed = {key: value.copy() for key, value in before.items()}
    proposed["B"] *= -1
    net.set_parameters(memory.project(before, proposed))
    np.testing.assert_array_equal(net.imagine(inputs).output, original)


def test_coincident_detuned_paths_cannot_hide_a_displaced_midpoint():
    net = cd.TemporalPatchNet(1, 1, 1)
    free = net.imagine(np.zeros((1, 1, 1)))
    displaced = replace(free, hidden=free.hidden + 1)
    assert contrast_asymmetry(free, free, free) == 0
    assert np.isinf(contrast_asymmetry(free, displaced, displaced))
