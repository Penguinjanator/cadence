"""Causal checks on the R32-style decoder follow-up."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest


def test_injected_memory_never_reads_current_or_future_target(monkeypatch) -> None:
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1] / "experiments/sequence_readback"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("sequence_decoder", root / "decoder.py")
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    cfg = Namespace(window=2, dim=2, hidden=4, radius=1.0, eta=1.5)
    net = decoder.Patch(3, cfg, seed=0, mode="echo")
    rng = np.random.default_rng(0)
    windows = rng.integers(0, 3, size=(8, 2, 2))
    targets = rng.integers(0, 3, size=(8, 2))
    changed = targets.copy()
    changed[3:] = (changed[3:] + 1) % 3
    efficacy = net.learner.brain.efficacy.copy()
    bias = net.learner.brain.bias.copy()
    old_logits, old_memory = decoder.outputs(net, windows, targets, 1.0, second_phase=True)
    new_logits, new_memory = decoder.outputs(net, windows, changed, 1.0, second_phase=True)
    # The target at t=3 is observed after, not before, prediction t=3.
    np.testing.assert_array_equal(old_logits[:4], new_logits[:4])
    np.testing.assert_array_equal(old_memory[:4], new_memory[:4])
    assert np.max(np.abs(old_logits[4:] - new_logits[4:])) > 1e-6
    np.testing.assert_array_equal(net.learner.brain.efficacy, efficacy)
    np.testing.assert_array_equal(net.learner.brain.bias, bias)


def test_uniform_control_reads_only_prior_tokens_and_evicts_oldest(monkeypatch) -> None:
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1] / "experiments/sequence_readback"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("uniform_control", root / "uniform_control.py")
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    targets = np.array([[0], [1], [2], [2], [0]])
    read = control.uniform(targets, vocabulary=3, capacity=2)
    np.testing.assert_array_equal(read[0], [[0, 0, 0]])
    np.testing.assert_array_equal(read[1], [[1, 0, 0]])
    np.testing.assert_array_equal(read[2], [[0.5, 0.5, 0]])
    np.testing.assert_array_equal(read[3], [[0, 0.5, 0.5]])
    np.testing.assert_array_equal(read[4], [[0, 0, 1]])
