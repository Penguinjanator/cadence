"""Independent held-read derivatives, witnessed chronology and immutable custody."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "attention_fixed_read_tests", Path(__file__).with_name("fixed_read.py")
)
assert SPEC is not None and SPEC.loader is not None
fixed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixed
SPEC.loader.exec_module(fixed)


def fact(name: int, value: int) -> np.ndarray:
    out = np.zeros(19)
    out[[0, 5 + name, 7 + value]] = 1
    return out


def bank():
    values = np.zeros((4, 5, 19))
    values[:, 0] = fact(0, 0)
    values[:, 1] = fact(1, 2)
    values[:, 2, 1] = 1  # neutral
    values[:, 3] = fact(0, 1)
    values[:, 4] = fact(1, 3)
    held = np.array([[True, True, True, True, True],
                     [False, True, False, False, False],
                     [False, False, False, False, False],
                     [False, False, True, False, False]])
    common = dict(keys=values * 0.5, values=values, held=held,
                  ages=np.array([4., 3., 2., 1., 0.]), amplitude=1.5)
    first = SimpleNamespace(query=np.arange(19), read=np.arange(19, 38), **common)
    second = SimpleNamespace(query=np.arange(38, 57), read=np.arange(57, 76), **common)
    return SimpleNamespace(memories=(first, second))


def test_oracle_chooses_latest_actual_held_named_fact_and_is_query_independent() -> None:
    memory = fixed.FrozenRead(bank())
    state = np.random.default_rng(31).normal(size=(4, 79))
    p = memory.weights(state)
    assert p.shape == (2, 4, 5)
    np.testing.assert_array_equal(p[0, 0], [0, 0, 0, 1, 0])
    np.testing.assert_array_equal(p[1, 0], [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(p[1, 1], [0, 1, 0, 0, 0])
    assert not p[0, 1:].any() and not p[:, 2:].any()
    np.testing.assert_array_equal(memory.targets[0, 0], fact(0, 1))
    np.testing.assert_array_equal(memory.targets[1, 0], fact(1, 3))
    state[:, :19] += 1000
    state[:, 38:57] -= 1000
    assert memory.weights(state) is p
    assert memory.mode == "held" and memory.fixed_selector and memory.streams == 4


def test_absent_name_restores_in_nonempty_bank_but_empty_stream_has_zero_force() -> None:
    memory = fixed.FrozenRead(bank())
    state = np.ones((4, 79))
    drive = memory.drive(state)
    np.testing.assert_array_equal(drive[1, memory.memories[0].read], -1.5 * np.ones(19))
    np.testing.assert_array_equal(drive[3, memory.memories[1].read], -1.5 * np.ones(19))
    np.testing.assert_array_equal(drive[2], np.zeros(79))
    assert memory.energy(state)[2] == 0
    for member in memory.memories:
        np.testing.assert_array_equal(drive[:, member.query], np.zeros((4, 19)))
    assert memory.energy(state)[1] > 0 and memory.energy(state)[3] > 0


def test_drive_matches_independent_energy_derivative_and_symmetric_jacobian() -> None:
    memory = fixed.FrozenRead(bank())
    state = np.random.default_rng(22).normal(size=(4, 79))
    epsilon = 1e-5
    jacobian = []
    for coordinate in range(state.shape[1]):
        delta = np.zeros_like(state)
        delta[:, coordinate] = epsilon
        numerical = (memory.energy(state + delta) - memory.energy(state - delta)) / (2 * epsilon)
        np.testing.assert_allclose(memory.drive(state)[:, coordinate], -numerical, atol=2e-9)
        jacobian.append((memory.drive(state + delta) - memory.drive(state - delta)) / (2 * epsilon))
    jacobian = np.stack(jacobian, axis=-1)
    np.testing.assert_allclose(jacobian, jacobian.swapaxes(1, 2), atol=1e-10)
    # Read curvature is restoring; all query and non-port derivatives vanish.
    expected = np.zeros((4, 79))
    for member in memory.memories:
        expected[:, member.read] = -member.amplitude * memory.held.any(axis=1)[:, None]
    np.testing.assert_allclose(np.diagonal(jacobian, axis1=1, axis2=2), expected, atol=1e-9)


def test_reordering_records_keeps_ages_selectors_targets_and_energy_together() -> None:
    memory = fixed.FrozenRead(bank())
    order = np.array([4, 2, 0, 3, 1])
    changed = replace(memory.bank, memories=tuple(replace(
        head, keys=head.keys[:, order], values=head.values[:, order],
        held=head.held[:, order], ages=head.ages[order],
    ) for head in memory.memories))
    reordered = fixed.FrozenRead(changed)
    state = np.random.default_rng(9).normal(size=(4, 79))
    np.testing.assert_array_equal(reordered.p_fixed, memory.p_fixed[:, :, order])
    np.testing.assert_array_equal(reordered.targets, memory.targets)
    np.testing.assert_array_equal(reordered.energy(state), memory.energy(state))
    np.testing.assert_array_equal(reordered.drive(state), memory.drive(state))


def test_wrapper_owns_one_readonly_bank_ports_and_fixed_targets() -> None:
    supplied = bank()
    memory = fixed.FrozenRead(supplied)
    before = memory.values.copy()
    supplied.memories[0].values[:] = 100
    supplied.memories[0].query[:] = 0
    np.testing.assert_array_equal(memory.values, before)
    np.testing.assert_array_equal(memory.memories[0].query, np.arange(19))
    for name in ("keys", "values", "held", "ages"):
        value = getattr(memory, name)
        assert all(getattr(head, name) is value for head in memory.memories)
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 0
    for value in (memory.p_fixed, memory.targets,
                  *(p for head in memory.memories for p in (head.query, head.read))):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 0
    with pytest.raises(FrozenInstanceError):
        memory.memories = ()
    assert not hasattr(fixed, "Graph") and not hasattr(fixed, "Brain")


@pytest.mark.parametrize("selection, expected", [
    ([3, 0, 1], [3, 0, 1]), (np.array([False, True, False, True]), [1, 3]),
    (np.array([], dtype=np.int64), []),
])
def test_take_preserves_fixed_selector_and_stream_identity(selection, expected) -> None:
    memory = fixed.FrozenRead(bank())
    taken = memory.take(selection)
    state = np.random.default_rng(17).normal(size=(4, 79))
    np.testing.assert_array_equal(taken.p_fixed, memory.p_fixed[:, expected])
    np.testing.assert_array_equal(taken.targets, memory.targets[:, expected])
    np.testing.assert_array_equal(taken.drive(state[expected]), memory.drive(state)[expected])
    np.testing.assert_array_equal(taken.energy(state[expected]), memory.energy(state)[expected])
    assert taken.streams == len(expected)
    assert taken.memories[0].keys is taken.memories[1].keys


@pytest.mark.parametrize("selection", [[True], [0, 0], [-1], [4], [0.5],
                                        np.array([2**64 - 1], dtype=np.uint64)])
def test_invalid_subsets_leave_the_owned_boundary_unchanged(selection) -> None:
    memory = fixed.FrozenRead(bank())
    before = memory.p_fixed.copy()
    with pytest.raises(ValueError):
        memory.take(selection)
    np.testing.assert_array_equal(memory.p_fixed, before)


def test_empty_slot_bank_has_zero_energy_and_drive() -> None:
    original = fixed.FrozenRead(bank())
    empty = fixed.FrozenRead(replace(original.bank, memories=tuple(replace(
        head, keys=head.keys[:, :0], values=head.values[:, :0],
        held=head.held[:, :0], ages=head.ages[:0],
    ) for head in original.memories)))
    state = np.ones((4, 79))
    assert empty.p_fixed.shape == (2, 4, 0)
    np.testing.assert_array_equal(empty.energy(state), np.zeros(4))
    np.testing.assert_array_equal(empty.drive(state), np.zeros((4, 79)))


@pytest.mark.parametrize("change", ["one_head", "wide_port", "overlap", "different_bank",
                                    "nonfinite", "negative_age", "negative_alpha", "tied_latest"])
def test_invalid_or_ambiguous_boundaries_are_rejected(change: str) -> None:
    supplied = bank()
    if change == "one_head":
        supplied.memories = supplied.memories[:1]
    elif change == "wide_port":
        supplied.memories[0].read = np.arange(38)
    elif change == "overlap":
        supplied.memories[1].read = supplied.memories[0].read.copy()
    elif change == "different_bank":
        supplied.memories[1].values = supplied.memories[1].values + 1
    elif change == "nonfinite":
        supplied.memories[0].keys[0, 0, 0] = np.nan
    elif change == "negative_age":
        supplied.memories[0].ages[0] = -1
    elif change == "negative_alpha":
        supplied.memories[0].amplitude = -1
    elif change == "tied_latest":
        supplied.memories[0].ages[0] = supplied.memories[0].ages[3]
    with pytest.raises(ValueError):
        fixed.FrozenRead(supplied)


@pytest.mark.parametrize("state", [np.zeros((3, 79)), np.zeros((4, 75)),
                                   np.full((4, 79), np.nan)])
def test_invalid_activity_is_rejected_by_all_read_operations(state) -> None:
    memory = fixed.FrozenRead(bank())
    for operation in (memory.energy, memory.drive, memory.weights):
        with pytest.raises(ValueError):
            operation(state)
