"""Actual immutable raw11 banks; no brain construction or neural phases."""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    runtime = os.environ.get("CADENCE_ATTENTION_RUNTIME")
    root = Path(runtime) if runtime else Path(__file__).resolve().parents[2]
    if not (root / "benchmarks/attention/heads.py").is_file():
        pytest.fail("set CADENCE_ATTENTION_RUNTIME to the frozen Cadence source")
    heads = load("single_bank_actual_heads", root / "benchmarks/attention/heads.py")
    campaign = load("single_bank_campaign",
                    Path(__file__).with_name("single_relevance_campaign.py"))
    facts = np.array(list(itertools.product(range(2), repeat=3)), dtype=np.int64)
    campaign._CONTEXT = SimpleNamespace(sr=SimpleNamespace(FACTS=facts))
    return heads, campaign, facts


@pytest.fixture(autouse=True)
def forbid_brains_and_solves(monkeypatch, modules):
    heads, _, _ = modules

    def forbidden(*args, **kwargs):
        raise AssertionError("bank interventions must not construct a brain or settle")

    monkeypatch.setattr(heads.MultiHeadBrain, "compose", forbidden)
    monkeypatch.setattr(heads.coupled.Brain, "compose", forbidden)
    monkeypatch.setattr(heads.Graph, "equilibrate", forbidden)


def witnessed(modules):
    heads, _, facts = modules
    heard = 0.37
    values = np.zeros((8, 6, 11))
    for stream, (a, b, order) in enumerate(facts):
        ordered = (a, 2 + b) if order == 0 else (2 + b, a)
        values[stream, 1] = np.eye(11)[ordered[0]]
        values[stream, 4] = np.eye(11)[ordered[1]]
    values[:, 2, 4] = 1  # actual neutral witness
    values[:, 0, 7] = 0.6  # a separate nonfact record must stay untouched
    values[:, 3, 0] = 1  # unheld A-like and B-like data are not donor targets
    values[:, 5, 2] = 1
    held = np.tile([True, True, True, False, True, False], (8, 1))
    memory = heads.Memory(np.arange(11), np.arange(11, 22), heard * values, values,
                          held, np.array([6., 5., 3., 2., 1., 0.]),
                          amplitude=3., temperature=0.25, mode="agreement", empty_anchor=True)
    return heads.MultiMemory((memory,)), SimpleNamespace(recall=SimpleNamespace(heard=heard))


def arrays(memory):
    return {name: np.array(getattr(memory, name), copy=True)
            for name in ("keys", "values", "held", "ages")}


def named(memory, name):
    return memory.held & (memory.values[:, :, 2 * name:2 * name + 2].sum(axis=2) > 0)


@pytest.mark.parametrize("query", [0, 1])
@pytest.mark.parametrize("condition", ["flip_relevant", "flip_irrelevant", "donor_both"])
def test_donors_change_joint_actual_K_V_only_for_requested_names(modules, query, condition):
    _, campaign, facts = modules
    memory, brain = witnessed(modules)
    before = arrays(memory)
    host_query = np.eye(11)[np.full(8, 5 + query)]
    complete_drive = np.random.default_rng(2).normal(size=(8, 37))
    query_before, drive_before = host_query.copy(), complete_drive.copy()
    changed = campaign.bank_change(memory, condition, query, brain)
    names = (0, 1) if condition == "donor_both" else (
        query if condition == "flip_relevant" else 1 - query,
    )
    all_selected = np.zeros_like(memory.held)
    for name in names:
        selected = named(memory, name)
        all_selected |= selected
        for stream in range(8):
            slot = int(np.flatnonzero(selected[stream])[0])
            event = np.eye(11)[2 * name + 1 - facts[stream, name]]
            np.testing.assert_array_equal(changed.values[stream, slot], event)
            np.testing.assert_array_equal(changed.keys[stream, slot], brain.recall.heard * event)
    np.testing.assert_array_equal(changed.values[~all_selected], memory.values[~all_selected])
    np.testing.assert_array_equal(changed.keys[~all_selected], memory.keys[~all_selected])
    np.testing.assert_array_equal(changed.held, memory.held)
    np.testing.assert_array_equal(changed.ages, memory.ages)
    for field, expected in before.items():
        np.testing.assert_array_equal(getattr(memory, field), expected)
    np.testing.assert_array_equal(host_query, query_before)
    np.testing.assert_array_equal(complete_drive, drive_before)
    # The returned bank owns immutable arrays and preserves the complete source descriptor.
    for field in before:
        assert not getattr(changed, field).flags.writeable
    for field in ("query", "read", "amplitude", "temperature", "recency", "null",
                  "mode", "empty_anchor"):
        np.testing.assert_array_equal(getattr(changed.memories[0], field),
                                      getattr(memory.memories[0], field))


@pytest.mark.parametrize("condition,name", [("erase_A", 0), ("erase_B", 1),
                                            ("erase_neutral", None)])
def test_erasure_changes_only_actual_selected_held_flags(modules, condition, name):
    _, campaign, _ = modules
    memory, brain = witnessed(modules)
    before = arrays(memory)
    selected = (memory.held & (memory.values[:, :, 4] > 0)
                if name is None else named(memory, name))
    changed = campaign.bank_change(memory, condition, 0, brain)
    expected = memory.held.copy()
    expected[selected] = False
    np.testing.assert_array_equal(changed.held, expected)
    for field in ("keys", "values", "ages"):
        np.testing.assert_array_equal(getattr(changed, field), before[field])
    np.testing.assert_array_equal(memory.held, before["held"])


def test_reorder_moves_K_V_held_and_ages_together_and_preserves_law(modules):
    _, campaign, _ = modules
    memory, brain = witnessed(modules)
    state = np.random.default_rng(19).normal(size=(8, 24))
    changed = campaign.bank_change(memory, "joint_K_V_held_age_reorder", 1, brain)
    for field in ("keys", "values", "held"):
        np.testing.assert_array_equal(getattr(changed, field), getattr(memory, field)[:, ::-1])
    np.testing.assert_array_equal(changed.ages, memory.ages[::-1])
    np.testing.assert_allclose(changed.weights(state), memory.weights(state)[:, :, ::-1],
                               atol=1e-15)
    np.testing.assert_allclose(changed.drive(state), memory.drive(state), atol=1e-14)
    np.testing.assert_allclose(changed.energy(state), memory.energy(state), atol=1e-14)


def test_all_erased_and_unchanged_conditions_keep_data_and_settings(modules):
    _, campaign, _ = modules
    memory, brain = witnessed(modules)
    for condition in ("intact", "frozen_interface", "shuffled_interface"):
        assert campaign.bank_change(memory, condition, 0, brain) is memory
    erased = campaign.bank_change(memory, "all_erased", 0, brain)
    assert not erased.held.any() and memory.held.any()
    for field in ("keys", "values", "ages"):
        np.testing.assert_array_equal(getattr(erased, field), getattr(memory, field))
    assert erased.empty_anchor


def test_missing_duplicate_named_or_neutral_witness_refuses(modules):
    _, campaign, _ = modules
    memory, brain = witnessed(modules)
    head = memory.memories[0]
    for condition, extra_slot in (("flip_relevant", 3), ("erase_neutral", 5)):
        held, values = memory.held.copy(), memory.values.copy()
        if condition == "erase_neutral":
            values[:, extra_slot] = np.eye(11)[4]
        held[:, extra_slot] = True
        malformed = replace(memory, memories=(replace(head, values=values, held=held),))
        with pytest.raises(ValueError, match="one actual"):
            campaign.bank_change(malformed, condition, 0, brain)
