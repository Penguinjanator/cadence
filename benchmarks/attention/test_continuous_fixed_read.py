"""Array-only raw9 oracle checks; no Cadence imports, constructors or solves."""

from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).parent
oracle_module = _load("continuous_asked_read_tests", HERE / "continuous_fixed_read.py")
task = _load("continuous_asked_read_task_tests", HERE / "continuous_relevance_task.py")


@pytest.fixture(autouse=True)
def forbid_model_imports(monkeypatch):
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name == "cadence" or name.startswith("cadence."):
            pytest.fail("oracle arrays must not import or construct a model")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)


@pytest.fixture
def oracle():
    runtime = os.environ.get("CADENCE_ATTENTION_RUNTIME")
    if runtime is None and (HERE / "single_fixed_read.py").is_file():
        runtime = HERE.parents[1]
    if not runtime:
        pytest.fail("set CADENCE_ATTENTION_RUNTIME to the exact frozen source")
    return SimpleNamespace(AskedRead=oracle_module.asked_read_class(runtime))


def bank(values, *, keys=None, held=None, ages=None):
    head = SimpleNamespace(
        query=np.arange(9),
        read=np.arange(9, 18),
        keys=values.copy() if keys is None else keys,
        values=values,
        held=np.ones(values.shape[:2], bool) if held is None else held,
        ages=np.arange(values.shape[1], dtype=float) if ages is None else ages,
        amplitude=3.0,
        temperature=0.25,
        recency=0.0,
        null=0.0,
        mode="agreement",
        empty_anchor=True,
    )
    return SimpleNamespace(memories=(head,), mode="agreement", empty_anchor=True)


def cases(query=0):
    histories = task.draw(0, 0)
    prefix, observed = task.events(histories, query)
    values = np.stack(prefix, axis=1)
    return histories, bank(values, ages=np.array([2.0, 1.0, 0.0])), observed


def test_source_bound_parent_law_is_reused_without_its_raw11_constructor(oracle, monkeypatch):
    parent = oracle.AskedRead.__mro__[1]
    assert oracle.AskedRead.energy is parent.energy
    assert oracle.AskedRead.drive is parent.drive
    assert oracle.AskedRead._state is parent._state

    def forbidden(*args, **kwargs):
        raise AssertionError("the raw11 constructor must not interpret raw9 keys")

    monkeypatch.setattr(parent, "__init__", forbidden)
    _, source, observed = cases()
    memory = oracle.AskedRead(source, observed)
    assert memory.targets.shape == (1, 8, 9)


@pytest.mark.parametrize("query", [0, 1])
def test_both_questions_select_actual_named_payload_across_orders_and_classes(oracle, query):
    histories, source, observed = cases(query)
    memory = oracle.AskedRead(source, observed)
    assert memory.heads == 1 and memory.streams == 8
    assert memory.fixed_selector and memory.mode == "held" and memory.empty_anchor
    for stream, (_, _, order) in enumerate(histories.rows):
        selected = query if order == 0 else 1 - query
        assert memory.p_fixed[0, stream, selected] == 1
        assert memory.p_fixed[0, stream].sum() == 1
        np.testing.assert_array_equal(
            memory.targets[0, stream], source.memories[0].values[stream, selected]
        )
        assert memory.targets[0, stream, 2] == histories.quantities[stream, query]
        assert memory.targets[0, stream, query] == 1
        assert memory.targets[0, stream, 3] == 1
    state = np.random.default_rng(1).normal(size=(8, 21))
    captured = memory.weights(state)
    state[:, :18] *= -100
    assert memory.weights(state) is captured


def test_prefixes_and_absent_names_keep_full_null_anchor(oracle):
    histories, source, observed = cases()
    prefix, _ = task.events(histories, 0)
    state = np.ones((8, 19))
    for observation in prefix:
        memory = oracle.AskedRead(source, observation)
        assert not memory.p_fixed.any() and not memory.targets.any()
        np.testing.assert_array_equal(memory.drive(state)[:, 9:18], -3 * np.ones((8, 9)))
        np.testing.assert_array_equal(memory.energy(state), np.full(8, 13.5))
    source.memories[0].held[:] = False
    memory = oracle.AskedRead(source, observed)
    assert not memory.p_fixed.any()
    np.testing.assert_array_equal(memory.drive(state)[:, 9:18], -3 * np.ones((8, 9)))


def test_actual_key_name_and_fact_marker_select_whole_value_not_an_answer_code(oracle):
    keys = np.zeros((1, 4, 9))
    keys[0, :, 0] = 1
    keys[0, :3, 3] = 1
    values = np.arange(36, dtype=float).reshape(1, 4, 9) - 10
    source = bank(values, keys=keys, ages=np.array([4.0, 2.0, 1.0, 0.0]))
    memory = oracle.AskedRead(source, np.eye(9)[[5]])
    np.testing.assert_array_equal(memory.p_fixed[0, 0], [0, 0, 1, 0])
    np.testing.assert_array_equal(memory.targets[0, 0], values[0, 2])
    source.memories[0].held[0, 2] = False
    memory = oracle.AskedRead(source, np.eye(9)[[5]])
    assert memory.p_fixed[0, 0, 1] == 1


def test_equal_age_conflicts_refuse_but_identical_duplicates_are_equivalent(oracle):
    keys = np.zeros((1, 2, 9))
    keys[0, :, [0, 3]] = 1
    values = keys.copy()
    values[0, :, 2] = [0.2, 0.8]
    source = bank(values, keys=keys, ages=np.zeros(2))
    with pytest.raises(ValueError, match="conflicting"):
        oracle.AskedRead(source, np.eye(9)[[5]])
    source.memories[0].values[0, 1] = source.memories[0].values[0, 0]
    memory = oracle.AskedRead(source, np.eye(9)[[5]])
    np.testing.assert_array_equal(memory.targets[0, 0], values[0, 0])


def test_energy_gradient_and_jacobian_match_full_restoring_force(oracle):
    _, source, observed = cases()
    memory = oracle.AskedRead(source, observed).take([2, 6])
    state = np.random.default_rng(3).normal(size=(2, 21))
    eps = 1e-5
    for coordinate in range(state.shape[1]):
        delta = np.zeros_like(state)
        delta[:, coordinate] = eps
        derivative = (memory.energy(state + delta) - memory.energy(state - delta)) / (2 * eps)
        np.testing.assert_allclose(memory.drive(state)[:, coordinate], -derivative, atol=3e-9)
        jacobian = (memory.drive(state + delta) - memory.drive(state - delta)) / (2 * eps)
        expected = np.zeros_like(state)
        if 9 <= coordinate < 18:
            expected[:, coordinate] = -3
        np.testing.assert_allclose(jacobian, expected, atol=3e-9)


def test_take_keeps_captured_selector_unique_rows_and_immutable_owned_data(oracle, monkeypatch):
    _, source, observed = cases()
    memory = oracle.AskedRead(source, observed)
    before = memory.values.copy()
    source.memories[0].values[:] = 99
    source.memories[0].keys[:] = 99
    source.memories[0].query[:] = 99
    observed[:] = 99
    np.testing.assert_array_equal(memory.values, before)
    for array in (
        memory.values,
        memory.keys,
        memory.held,
        memory.ages,
        memory.p_fixed,
        memory.targets,
        memory.observations,
        memory.memories[0].query,
        memory.memories[0].read,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0
    with pytest.raises(FrozenInstanceError):
        memory.targets = np.zeros((1, 8, 9))

    def forbidden(*args, **kwargs):
        raise AssertionError("take must not recompute the selector")

    monkeypatch.setattr(oracle.AskedRead, "__init__", forbidden)
    for selection, expected in (
        ([6, 1, 4], [6, 1, 4]),
        (np.arange(8) % 3 == 0, [0, 3, 6]),
        ([], []),
    ):
        part = memory.take(selection)
        assert type(part) is oracle.AskedRead
        assert part.streams == len(expected)
        np.testing.assert_array_equal(part.p_fixed, memory.p_fixed[:, expected])
        np.testing.assert_array_equal(part.targets, memory.targets[:, expected])
        np.testing.assert_array_equal(part.observations, memory.observations[expected])
        assert part.ages is memory.ages
        assert not np.shares_memory(part.values, memory.values)
        for name in ("amplitude", "temperature", "recency", "null", "mode", "empty_anchor"):
            assert getattr(part.memories[0], name) == getattr(memory.memories[0], name)


def test_slot_permutation_carries_names_values_masks_and_ages_together(oracle):
    _, source, observed = cases(1)
    memory = oracle.AskedRead(source, observed)
    order = np.array([2, 0, 1])
    head = memory.memories[0]
    changed = replace(
        memory.bank,
        memories=(
            replace(
                head,
                keys=head.keys[:, order],
                values=head.values[:, order],
                held=head.held[:, order],
                ages=head.ages[order],
            ),
        ),
    )
    permuted = oracle.AskedRead(changed, observed)
    np.testing.assert_array_equal(permuted.targets, memory.targets)
    np.testing.assert_array_equal(permuted.p_fixed, memory.p_fixed[:, :, order])


@pytest.mark.parametrize(
    "selection",
    [[0, 0], [-1], [8], [1.2], [[1]], True, np.array([2**64 - 1], dtype=np.uint64), [True]],
)
def test_invalid_row_ownership_refuses(oracle, selection):
    _, source, observed = cases()
    with pytest.raises(ValueError):
        oracle.AskedRead(source, observed).take(selection)


@pytest.mark.parametrize(
    "name,value",
    [
        ("amplitude", -1),
        ("amplitude", np.inf),
        ("amplitude", True),
        ("amplitude", "3"),
        ("temperature", 0),
        ("recency", -1),
        ("null", np.nan),
        ("empty_anchor", False),
        ("mode", "linear"),
    ],
)
def test_invalid_or_different_settings_refuse(oracle, name, value):
    _, source, observed = cases()
    setattr(source.memories[0], name, value)
    with pytest.raises(ValueError):
        oracle.AskedRead(source, observed)


def test_invalid_heads_data_ports_observations_and_state_refuse(oracle):
    _, source, observed = cases()
    with pytest.raises(ValueError, match="exactly one"):
        oracle.AskedRead(SimpleNamespace(memories=source.memories * 2), observed)
    for field, value in (
        ("query", np.arange(9, dtype=float)),
        ("read", np.arange(9)),
        ("query", np.arange(11)),
        ("held", np.ones((8, 3))),
        ("ages", np.array([-1.0, 1.0, 2.0])),
        ("keys", np.zeros((8, 3, 11))),
    ):
        _, malformed, _ = cases()
        setattr(malformed.memories[0], field, value)
        with pytest.raises(ValueError):
            oracle.AskedRead(malformed, observed)
    both_names = source.memories[0].keys.copy()
    both_names[0, 0, :2] = 1
    with pytest.raises(ValueError, match="both A and B"):
        oracle.AskedRead(bank(source.memories[0].values, keys=both_names), observed)
    for observation in (
        observed[:-1],
        np.full((8, 9), np.nan),
        np.tile(np.eye(9)[5] + np.eye(9)[6], (8, 1)),
    ):
        with pytest.raises(ValueError):
            oracle.AskedRead(source, observation)
    memory = oracle.AskedRead(source, observed)
    for state in (np.zeros((7, 18)), np.zeros((8, 17)), np.zeros(18), np.full((8, 18), np.inf)):
        with pytest.raises(ValueError):
            memory.drive(state)


def test_zero_slot_bank_selects_null_and_keeps_full_anchor(oracle):
    memory = oracle.AskedRead(bank(np.zeros((2, 0, 9))), np.eye(9)[[5, 6]])
    assert memory.p_fixed.shape == (1, 2, 0) and not memory.targets.any()
    np.testing.assert_array_equal(memory.drive(np.ones((2, 18)))[:, 9:], -3 * np.ones((2, 9)))
