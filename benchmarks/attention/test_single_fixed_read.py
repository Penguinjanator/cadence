"""Array-only oracle law, chronology and immutable phase-boundary checks."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "single_asked_read_tests", Path(__file__).with_name("single_fixed_read.py")
)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


def bank(values: np.ndarray, *, keys: np.ndarray | None = None,
         held: np.ndarray | None = None, ages: np.ndarray | None = None):
    head = SimpleNamespace(
        query=np.arange(11), read=np.arange(11, 22),
        keys=values.copy() if keys is None else keys,
        values=values, held=np.ones(values.shape[:2], bool) if held is None else held,
        ages=np.arange(values.shape[1], dtype=float) if ages is None else ages,
        amplitude=3.0, temperature=0.25, recency=0.0, null=0.0,
        mode="agreement", empty_anchor=True,
    )
    return SimpleNamespace(memories=(head,), mode="agreement", empty_anchor=True)


def cases():
    rows = np.array(list(itertools.product(range(2), repeat=4)), dtype=np.int64)
    a, b, order, query = rows.T
    values = np.zeros((16, 3, 11))
    a_fact, b_fact = np.eye(11)[a], np.eye(11)[2 + b]
    values[:, 0] = np.where(order[:, None] == 0, a_fact, b_fact)
    values[:, 1] = np.where(order[:, None] == 0, b_fact, a_fact)
    values[:, 2, 4] = 1
    observed = np.eye(11)[5 + query]
    return rows, bank(values, ages=np.array([2., 1., 0.])), observed


def test_all_sixteen_actual_histories_and_questions_choose_the_named_held_record():
    rows, source, observation = cases()
    memory = oracle.AskedRead(source, observation)
    state = np.random.default_rng(123).normal(size=(16, 24))
    p = memory.weights(state)
    assert p.shape == (1, 16, 3)
    assert memory.heads == 1 and memory.streams == 16
    assert memory.fixed_selector and memory.mode == "held" and memory.empty_anchor
    for stream, (a, b, order, query) in enumerate(rows):
        selected = query if order == 0 else 1 - query
        assert p[0, stream, selected] == 1 and p[0, stream].sum() == 1
        expected = np.eye(11)[a if query == 0 else 2 + b]
        np.testing.assert_array_equal(memory.targets[0, stream], expected)
    # The selector is frozen; live cortical/query/read states do not address the oracle.
    state[:, :22] *= -100
    assert memory.weights(state) is p


def test_prefix_and_neutral_events_select_null_with_full_empty_restoring_force():
    _, source, _ = cases()
    for event in range(5):
        memory = oracle.AskedRead(source, np.tile(np.eye(11)[event], (16, 1)))
        assert not memory.p_fixed.any() and not memory.targets.any()
        state = np.ones((16, 24))
        np.testing.assert_array_equal(memory.drive(state)[:, 11:22], -3 * np.ones((16, 11)))
        np.testing.assert_array_equal(memory.energy(state), 16.5 * np.ones(16))
    source.memories[0].held[:] = False
    empty = oracle.AskedRead(source, np.tile(np.eye(11)[5], (16, 1)))
    np.testing.assert_array_equal(empty.drive(state)[:, 11:22], -3 * np.ones((16, 11)))
    assert not empty.p_fixed.any()


def test_name_comes_from_actual_key_and_read_content_from_whole_actual_value():
    values = np.zeros((1, 4, 11))
    keys = np.eye(11)[[0, 2, 1, 4]][None]
    values[0, 0, 10] = 4
    values[0, 1, 7] = 8
    values[0, 2, 9] = -3
    source = bank(values, keys=keys, ages=np.array([4., 0., 1., 2.]))
    memory = oracle.AskedRead(source, np.eye(11)[[5]])
    np.testing.assert_array_equal(memory.p_fixed[0, 0], [0, 0, 1, 0])
    np.testing.assert_array_equal(memory.targets[0, 0], values[0, 2])
    # Unheld newer witnesses cannot replace a held older fact.
    source.memories[0].held[0, 2] = False
    older = oracle.AskedRead(source, np.eye(11)[[5]])
    assert older.p_fixed[0, 0, 0] == 1
    source.memories[0].held[0, 0] = False
    absent = oracle.AskedRead(source, np.eye(11)[[5]])
    assert not absent.p_fixed.any() and not absent.targets.any()


def test_equal_age_conflicts_refuse_and_duplicate_identical_facts_are_equivalent():
    values = np.eye(11)[[0, 1]][None]
    source = bank(values, ages=np.zeros(2))
    with pytest.raises(ValueError, match="conflicting"):
        oracle.AskedRead(source, np.eye(11)[[5]])
    source.memories[0].values[0, 1] = source.memories[0].values[0, 0]
    source.memories[0].keys[0, 1] = source.memories[0].keys[0, 0]
    memory = oracle.AskedRead(source, np.eye(11)[[5]])
    np.testing.assert_array_equal(memory.targets[0, 0], values[0, 0])


def test_energy_gradient_and_full_jacobian_are_the_declared_restoring_law():
    _, source, observed = cases()
    memory = oracle.AskedRead(source, observed).take([3, 8])
    state = np.random.default_rng(9).normal(size=(2, 25))
    eps = 1e-5
    for coordinate in range(state.shape[1]):
        delta = np.zeros_like(state)
        delta[:, coordinate] = eps
        numerical = (memory.energy(state + delta) - memory.energy(state - delta)) / (2 * eps)
        np.testing.assert_allclose(memory.drive(state)[:, coordinate], -numerical, atol=2e-9)
        jacobian_column = (memory.drive(state + delta) - memory.drive(state - delta)) / (2 * eps)
        expected = np.zeros_like(state)
        if 11 <= coordinate < 22:
            expected[:, coordinate] = -3
        np.testing.assert_allclose(jacobian_column, expected, atol=2e-9)


def test_take_carries_captured_selector_and_readonly_custody_for_masks_and_reorders(monkeypatch):
    _, source, observed = cases()
    memory = oracle.AskedRead(source, observed)
    before = memory.values.copy()
    source.memories[0].values[:] = 99
    source.memories[0].keys[:] = 99
    source.memories[0].query[:] = 99
    observed[:] = 99
    np.testing.assert_array_equal(memory.values, before)
    for array in (memory.values, memory.keys, memory.held, memory.ages,
                  memory.p_fixed, memory.targets, memory.observations,
                  memory.memories[0].query, memory.memories[0].read):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0
    with pytest.raises(FrozenInstanceError):
        memory.p_fixed = np.zeros((1, 16, 3))
    # take is a restriction of the captured boundary, not a call to the selector constructor.
    def forbidden(*args, **kwargs):
        raise AssertionError("selector must not be recomputed during row selection")
    monkeypatch.setattr(oracle.AskedRead, "__init__", forbidden)
    for selection, expected in (([7, 2, 11], [7, 2, 11]),
                                (np.arange(16) % 4 == 0, [0, 4, 8, 12]), ([], [])):
        part = memory.take(selection)
        assert part.streams == len(expected)
        np.testing.assert_array_equal(part.p_fixed, memory.p_fixed[:, expected])
        np.testing.assert_array_equal(part.targets, memory.targets[:, expected])
        np.testing.assert_array_equal(part.observations, memory.observations[expected])
        assert part.ages is memory.ages
        for setting in ("amplitude", "temperature", "recency", "null", "mode", "empty_anchor"):
            assert getattr(part.memories[0], setting) == getattr(memory.memories[0], setting)


def test_slot_reorder_carries_keys_values_ages_and_selector_together():
    _, source, observed = cases()
    memory = oracle.AskedRead(source, observed)
    order = np.array([2, 0, 1])
    head = memory.memories[0]
    reordered = replace(memory.bank, memories=(replace(
        head, keys=head.keys[:, order], values=head.values[:, order],
        held=head.held[:, order], ages=head.ages[order],
    ),))
    changed = oracle.AskedRead(reordered, observed)
    np.testing.assert_array_equal(changed.p_fixed, memory.p_fixed[:, :, order])
    np.testing.assert_array_equal(changed.targets, memory.targets)


@pytest.mark.parametrize("selection", [[0, 0], [-1], [16], [1.2], [[1]], True,
                                       np.array([2**64 - 1], dtype=np.uint64), [True]])
def test_invalid_row_ownership_refuses(selection):
    _, source, observed = cases()
    with pytest.raises(ValueError):
        oracle.AskedRead(source, observed).take(selection)


@pytest.mark.parametrize("name,value", [("amplitude", -1), ("amplitude", np.inf),
                                         ("amplitude", True), ("amplitude", "3"),
                                         ("temperature", 0), ("recency", -1),
                                         ("null", np.nan), ("empty_anchor", False),
                                         ("mode", "linear")])
def test_invalid_or_different_source_settings_refuse(name, value):
    _, source, observed = cases()
    setattr(source.memories[0], name, value)
    with pytest.raises(ValueError):
        oracle.AskedRead(source, observed)


def test_invalid_head_data_ports_queries_and_state_refuse():
    _, source, observed = cases()
    with pytest.raises(ValueError, match="exactly one"):
        oracle.AskedRead(SimpleNamespace(memories=source.memories * 2), observed)
    for field, value in (("query", np.arange(11, dtype=float)),
                         ("read", np.arange(11)), ("held", np.ones((16, 3))),
                         ("ages", np.array([-1., 1., 2.]))):
        _, malformed, _ = cases()
        setattr(malformed.memories[0], field, value)
        with pytest.raises(ValueError):
            oracle.AskedRead(malformed, observed)
    for observation in (observed[:-1], np.full((16, 11), np.nan),
                        np.tile(np.eye(11)[5] + np.eye(11)[6], (16, 1))):
        with pytest.raises(ValueError):
            oracle.AskedRead(source, observation)
    memory = oracle.AskedRead(source, observed)
    for state in (np.zeros((15, 23)), np.zeros((16, 21)), np.zeros(22),
                  np.full((16, 22), np.inf)):
        with pytest.raises(ValueError):
            memory.drive(state)


def test_zero_slot_bank_is_valid_null_and_has_full_anchor():
    memory = oracle.AskedRead(bank(np.zeros((2, 0, 11))), np.eye(11)[[5, 6]])
    assert memory.p_fixed.shape == (1, 2, 0)
    assert not memory.targets.any()
    np.testing.assert_array_equal(memory.drive(np.ones((2, 22)))[:, 11:], -3 * np.ones((2, 11)))
