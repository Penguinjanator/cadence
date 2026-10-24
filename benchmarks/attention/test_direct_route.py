"""Regression checks for the retained route-only attention candidate."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "attention_direct_route", HERE / "direct_route.py")
assert SPEC is not None and SPEC.loader is not None
route = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = route
SPEC.loader.exec_module(route)
DirectRoute = route.DirectRoute


def boundary(gate: float = 1.0, *, zero_weight: bool = False) -> DirectRoute:
    rng = np.random.default_rng(76421)
    keys = rng.normal(0.0, 0.2, (2, 4, 3))
    values = rng.uniform(-0.08, 0.5, (2, 4, 3))
    weight = np.zeros((3, 3)) if zero_weight else rng.normal(0.0, 0.08, (3, 3))
    return DirectRoute(keys, values,
        np.array([[True, True, False, True], [True, False, True, True]]),
        np.array([3.0, 2.0, 1.0, 0.0]),
        rng.normal(0.0, 0.2, (2, 3)), weight,
        np.full(2, gate, dtype=float), amplitude=2.1, temperature=0.63,
        recency=0.17, null=0.31, gamma=6.75,
        route_stiffness=36.0, route_damping=1.3)


def numerical_force(memory: DirectRoute, answer: np.ndarray, state: np.ndarray,
                    kind: str, index: tuple[int, ...], epsilon: float = 1e-6) -> float:
    def changed(sign: float) -> tuple[DirectRoute, np.ndarray, np.ndarray]:
        if kind == "answer":
            value = answer.copy()
            value[index] += sign * epsilon
            return memory, value, state
        if kind == "route":
            value = state.copy()
            value[index] += sign * epsilon
            return memory, answer, value
        if kind == "meaning":
            value = memory.meaning.copy()
            value[index] += sign * epsilon
            return replace(memory, meaning=value), answer, state
        if kind == "weight":
            value = memory.weight.copy()
            value[index] += sign * epsilon
            return replace(memory, weight=value), answer, state
        raise AssertionError(kind)

    plus, plus_answer, plus_route = changed(1.0)
    minus, minus_answer, minus_route = changed(-1.0)
    return -float((plus.energy(plus_answer, plus_route).sum()
                  - minus.energy(minus_answer, minus_route).sum()) / (2.0 * epsilon))


@pytest.mark.parametrize("gate", [0.0, 0.37, 1.0])
def test_all_route_forces_are_negative_derivatives_of_one_energy(gate: float) -> None:
    memory = boundary(gate)
    rng = np.random.default_rng(9)
    answer = rng.normal(0.1, 0.2, (2, 3))
    state = rng.normal(0.0, 0.2, (2, 3))
    forces = memory.forces(answer, state)
    for kind in ("answer", "route", "meaning"):
        for index in np.ndindex(forces[kind].shape):
            assert numerical_force(memory, answer, state, kind, index) \
                == pytest.approx(forces[kind][index], rel=4e-5, abs=4e-7)
    for index in np.ndindex(memory.weight.shape):
        expected = forces["weight"][:, index[0], index[1]].sum()
        assert numerical_force(memory, answer, state, "weight", index) \
            == pytest.approx(expected, rel=4e-5, abs=4e-7)


def test_gate_off_is_the_held_meaning_reader_and_has_no_route_or_W_force() -> None:
    memory = boundary(0.0, zero_weight=True)
    answer = np.random.default_rng(4).normal(size=(2, 3))
    state = np.zeros((2, 3))
    parts = memory.parts(answer, state)
    np.testing.assert_array_equal(parts["query"], memory.meaning)
    np.testing.assert_array_equal(memory.energy(answer, state),
                                  parts["answer"]["energy"])
    forces = memory.forces(answer, state)
    assert not forces["route"].any()
    assert not forces["weight"].any()


def test_born_zero_map_receives_a_nonzero_centered_local_contrast() -> None:
    memory = boundary(1.0, zero_weight=True)
    rng = np.random.default_rng(17)
    answer = rng.normal(0.1, 0.2, (2, 3))
    state = rng.normal(0.0, 0.2, (2, 3))
    gradient = route.centered_weight_gradient(
        memory, answer + 0.04, state, answer - 0.04, state, 0.1)
    assert gradient.shape == memory.weight.shape
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 1e-9


def test_retained_actual_cadence_acquisition_and_causal_gates() -> None:
    receipt = json.loads((HERE / "results/direct-route-development.json").read_text())
    assert receipt["schema"] == "cadence-attention-direct-route-development/1"
    assert receipt["scope"].startswith("single development founder")
    assert receipt["lessons"] == 64
    assert receipt["auxiliary_coordinates_per_stream"] == 64
    assert receipt["allowed_accuracy"] >= 0.95
    assert receipt["held_accuracy"] >= 0.95
    assert receipt["causal"]["learned"] == {
        "answer": 1.0, "top1": 1.0, "material": 0.984375}
    assert receipt["causal"]["W_zero"]["answer"] < 0.4
    assert receipt["causal"]["relation_erased"]["answer"] < 0.6
    assert receipt["causal"]["relevant_erased"]["answer"] == 0.0
    assert receipt["causal"]["unrelated_erased"]["answer"] == 1.0
    assert receipt["causal"]["relevant_content"]["answer"] == 1.0
    assert receipt["causal"]["unrelated_content"]["answer"] == 1.0
    assert receipt["maximum_relative_query_shift"]["unrelated_content"] <= 0.06
    assert all(receipt["gates"].values())
    assert receipt["passed"] is True
