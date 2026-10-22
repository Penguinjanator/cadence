"""Inherited empty-read agreement, unchanged occupied rows and frozen custody."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "attention_fixed_empty_boundary_tests", Path(__file__).with_name("fixed_read.py")
)
assert SPEC is not None and SPEC.loader is not None
fixed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixed
SPEC.loader.exec_module(fixed)


def bank(anchor=None):
    values = np.zeros((3, 3, 19))
    values[:, 0, [0, 5, 7]] = 1
    values[:, 1, [0, 6, 8]] = 1
    values[:, 2, 1] = 1
    held = np.array([[True, True, True], [False, False, False], [False, False, True]])
    common = dict(keys=values.copy(), values=values, held=held,
                  ages=np.array([2., 1., 0.]), amplitude=1.5, mode="agreement")
    if anchor is not None:
        common["empty_anchor"] = anchor
    first = SimpleNamespace(query=np.arange(19), read=np.arange(19, 38), **common)
    second = SimpleNamespace(query=np.arange(38, 57), read=np.arange(57, 76), **common)
    out = SimpleNamespace(memories=(first, second), mode="agreement")
    if anchor is not None:
        out.empty_anchor = anchor
    return out


def test_historical_default_is_preserved_and_occupied_rows_are_bit_exact() -> None:
    historical = fixed.FrozenRead(bank())
    explicit = fixed.FrozenRead(bank(False))
    revised = fixed.FrozenRead(bank(True))
    state = np.random.default_rng(11).normal(size=(3, 79))
    assert historical.empty_anchor is explicit.empty_anchor is False
    np.testing.assert_array_equal(historical.energy(state), explicit.energy(state))
    np.testing.assert_array_equal(historical.drive(state), explicit.drive(state))
    occupied = historical.held.any(axis=1)
    np.testing.assert_array_equal(
        revised.energy(state)[occupied], historical.energy(state)[occupied])
    np.testing.assert_array_equal(revised.drive(state)[occupied], historical.drive(state)[occupied])
    np.testing.assert_array_equal(revised.p_fixed, historical.p_fixed)
    np.testing.assert_array_equal(revised.targets, historical.targets)
    assert historical.energy(state)[1] == 0 and not historical.drive(state)[1].any()
    assert revised.energy(state)[1] > 0


def test_allocated_empty_and_zero_slot_banks_both_restore_toward_zero() -> None:
    memory = fixed.FrozenRead(bank(True))
    erased = fixed.FrozenRead(replace(memory.bank, memories=tuple(replace(
        head, held=np.zeros_like(head.held),
    ) for head in memory.memories)))
    empty = fixed.FrozenRead(replace(memory.bank, memories=tuple(replace(
        head, keys=head.keys[:, :0], values=head.values[:, :0],
        held=head.held[:, :0], ages=head.ages[:0],
    ) for head in memory.memories)))
    state = np.random.default_rng(4).normal(size=(3, 79))
    expected, energy = np.zeros_like(state), np.zeros(3)
    for head in memory.memories:
        expected[:, head.read] = -head.amplitude * state[:, head.read]
        energy += 0.5 * head.amplitude * np.square(state[:, head.read]).sum(axis=1)
    for boundary in (erased, empty):
        assert boundary.empty_anchor and not boundary.targets.any() and not boundary.p_fixed.any()
        np.testing.assert_array_equal(boundary.drive(state), expected)
        np.testing.assert_allclose(boundary.energy(state), energy, rtol=0, atol=1e-13)
        for head in boundary.memories:
            assert not boundary.drive(state)[:, head.query].any()
    assert empty.p_fixed.shape == (2, 3, 0)
    np.testing.assert_array_equal(empty.energy(state), erased.energy(state))


def test_zero_content_and_absent_names_keep_restoring_law_without_fabricating_a_fact() -> None:
    memory = fixed.FrozenRead(bank(True))
    null = fixed.FrozenRead(replace(memory.bank, memories=tuple(replace(
        head, keys=np.zeros_like(head.keys), values=np.zeros_like(head.values),
        held=np.ones_like(head.held),
    ) for head in memory.memories)))
    state = np.ones((3, 79))
    assert not null.targets.any() and not null.weights(state).any()
    for head in null.memories:
        np.testing.assert_array_equal(null.drive(state)[:, head.read], -1.5 * np.ones((3, 19)))
    # A nonempty neutral-only row already used this zero-content law historically.
    historical = fixed.FrozenRead(bank(False))
    np.testing.assert_array_equal(memory.drive(state)[2], historical.drive(state)[2])
    zero = np.zeros_like(state)
    assert not null.energy(zero).any() and not null.drive(zero).any()
    powerless = fixed.FrozenRead(replace(null.bank, memories=tuple(
        replace(head, amplitude=0.0) for head in null.memories)))
    assert not powerless.energy(state).any() and not powerless.drive(state).any()


def test_revised_drive_is_negative_energy_derivative_with_symmetric_restoring_jacobian() -> None:
    memory = fixed.FrozenRead(bank(True))
    state = np.random.default_rng(91).normal(size=(3, 79))
    epsilon = 1e-5
    columns = []
    for coordinate in range(state.shape[1]):
        delta = np.zeros_like(state)
        delta[:, coordinate] = epsilon
        numerical = (memory.energy(state + delta) - memory.energy(state - delta)) / (2 * epsilon)
        np.testing.assert_allclose(memory.drive(state)[:, coordinate], -numerical, atol=3e-9)
        columns.append((memory.drive(state + delta) - memory.drive(state - delta)) / (2 * epsilon))
    jacobian = np.stack(columns, axis=-1)
    np.testing.assert_allclose(jacobian, jacobian.swapaxes(1, 2), atol=1e-10)
    diagonal = np.zeros_like(state)
    for head in memory.memories:
        diagonal[:, head.read] = -head.amplitude
    np.testing.assert_allclose(np.diagonal(jacobian, axis1=1, axis2=2), diagonal, atol=1e-9)


@pytest.mark.parametrize("selection, expected", [
    ([2, 1, 0], [2, 1, 0]), (np.array([False, True, True]), [1, 2]),
    (np.array([], dtype=np.int64), []),
])
def test_take_and_rebuilding_owned_bank_preserve_setting_and_stream_identity(selection, expected):
    memory = fixed.FrozenRead(bank(True))
    taken = memory.take(selection)
    rebuilt = fixed.FrozenRead(taken.bank)
    state = np.random.default_rng(10).normal(size=(3, 79))
    assert taken.empty_anchor is rebuilt.empty_anchor is True
    assert taken.bank.empty_anchor is True
    assert all(head.empty_anchor is True for head in taken.memories)
    np.testing.assert_array_equal(taken.targets, memory.targets[:, expected])
    np.testing.assert_array_equal(taken.drive(state[expected]), memory.drive(state)[expected])
    np.testing.assert_array_equal(rebuilt.drive(state[expected]), taken.drive(state[expected]))


def test_boundary_setting_is_owned_immutable_and_can_be_inherited_from_member_descriptors() -> None:
    source = bank(True)
    del source.empty_anchor
    memory = fixed.FrozenRead(source)
    assert memory.empty_anchor is True
    source.memories[0].empty_anchor = False
    assert memory.empty_anchor and memory.bank.empty_anchor
    with pytest.raises(FrozenInstanceError):
        memory.empty_anchor = False
    for head in memory.memories:
        with pytest.raises(FrozenInstanceError):
            head.empty_anchor = False


@pytest.mark.parametrize("setting", [0, 1, "yes", None, np.array([True])])
def test_nonboolean_boundary_setting_is_rejected(setting) -> None:
    source = bank()
    source.empty_anchor = setting
    with pytest.raises(ValueError, match="empty_anchor"):
        fixed.FrozenRead(source)


def test_inconsistent_head_and_bank_settings_or_nonagreement_source_are_rejected() -> None:
    for location in ("first", "second", "bank"):
        source = bank(True)
        if location == "bank":
            source.empty_anchor = False
        else:
            source.memories[0 if location == "first" else 1].empty_anchor = False
        with pytest.raises(ValueError, match="empty_anchor"):
            fixed.FrozenRead(source)
    source = bank(True)
    source.memories[1].empty_anchor = 1
    with pytest.raises(ValueError, match="empty_anchor"):
        fixed.FrozenRead(source)
    source = bank(True)
    source.mode = "linear"
    with pytest.raises(ValueError, match="agreement"):
        fixed.FrozenRead(source)
    source = bank(np.bool_(True))
    assert fixed.FrozenRead(source).empty_anchor is True
