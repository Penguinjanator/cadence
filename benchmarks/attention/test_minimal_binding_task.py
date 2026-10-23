"""Pure task contracts only: no Cadence import, constructor or neural phase."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "minimal_binding_task", HERE / "minimal_binding_task.py"
)
assert SPEC is not None and SPEC.loader is not None
task = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task)


def test_exact_full_factorial_has_eight_cases_one_query_and_no_held_split():
    expected = np.asarray([(*row, 0) for row in itertools.product(range(2), range(2), range(2))])
    np.testing.assert_array_equal(task.CASES, expected)
    assert task.CASES.shape == (8, 4) and not task.CASES.flags.writeable
    assert task.INPUTS == 19 and task.VALUES * task.SLOTS == 8
    assert task.MOTOR_CHOICES == 4 and task.ACTIVE_VALUES == (0, 1)
    assert not hasattr(task, "HELD") and not hasattr(task, "HELD_PAIRS")
    train, all_rows = task.panel("train"), task.panel("all")
    np.testing.assert_array_equal(train, all_rows)
    train[0, 0] = 1
    assert task.CASES[0, 0] == 0
    for name in ("held_out", "held", "test", "validation"):
        with pytest.raises(ValueError, match="no held-out"):
            task.panel(name)


def test_actual_fact_order_varies_while_candidate_query_and_targets_remain_fixed():
    rows = task.panel("all")
    prefix, query = task.events(rows)
    assert len(prefix) == 3 and query.shape == (8, 19)
    np.testing.assert_array_equal(task.targets(rows), rows[:, :2])
    expected_query = np.zeros_like(query)
    expected_query[:, (2, 3)] = 1
    np.testing.assert_array_equal(query, expected_query)
    for i, (a, b, order, _) in enumerate(rows):
        for fact, name, value in ((prefix[order][i], 0, a), (prefix[1 - order][i], 1, b)):
            expected = np.zeros(19)
            expected[[0, 5 + name, 7 + value]] = 1
            np.testing.assert_array_equal(fact, expected)
    expected_neutral = np.zeros((8, 19))
    expected_neutral[:, 1] = 1
    np.testing.assert_array_equal(prefix[2], expected_neutral)
    assert all(not array[:, (4, 9, 10)].any() for array in (*prefix, query))
    assert all(not array[:, 11:].any() for array in (*prefix, query))


def test_exposed_control_supplies_both_unselected_values_only_in_control_coordinates():
    rows = task.panel("all")
    original_prefix, query = task.events(rows)
    exposed_prefix, exposed = task.events(rows, exposed=True)
    for original, candidate in zip(original_prefix, exposed_prefix, strict=True):
        np.testing.assert_array_equal(original, candidate)
    np.testing.assert_array_equal(query[:, :11], exposed[:, :11])
    for i, (a, b, _, _) in enumerate(rows):
        assert np.flatnonzero(exposed[i, 11:]).tolist() == [int(a), int(4 + b)]
    assert not exposed[:, (13, 14, 17, 18)].any()


def test_either_fact_can_change_independently_without_changing_the_other_or_query():
    rows = task.panel("all")
    prefix, query = task.events(rows)
    for name in (0, 1):
        donor = task.flipped(rows, name)
        np.testing.assert_array_equal(donor[:, name], 1 - rows[:, name])
        np.testing.assert_array_equal(donor[:, 1 - name], rows[:, 1 - name])
        np.testing.assert_array_equal(donor[:, 2:], rows[:, 2:])
        np.testing.assert_array_equal(task.flipped(donor, name), rows)
        donor_prefix, donor_query = task.events(donor)
        np.testing.assert_array_equal(donor_query, query)
        np.testing.assert_array_equal(donor_prefix[2], prefix[2])
        for i, row in enumerate(rows):
            position = int(row[2]) if name == 0 else 1 - int(row[2])
            np.testing.assert_array_equal(donor_prefix[1 - position][i], prefix[1 - position][i])
            assert not np.array_equal(donor_prefix[position][i], prefix[position][i])
    swapped = task.swapped(rows)
    np.testing.assert_array_equal(swapped[:, :2], rows[:, 1::-1])
    np.testing.assert_array_equal(task.events(swapped)[1], query)
    assert (task.targets(swapped) != task.targets(rows)).any(axis=1).sum() == 4


def test_scores_count_all_cases_refusals_and_inactive_motor_errors():
    rows = task.panel("all")
    perfect = task.targets(rows)
    score = task.scores(perfect, rows)
    assert score["cases"] == score["correct_pairs"] == 8
    assert score["per_slot_correct"] == [8, 8]
    assert score["unequal_cases"] == score["unequal_correct_pairs"] == 4
    assert score["per_query_accuracy"] == [1.0] and score["per_prefix_accuracy"] == [1.0, 1.0]
    damaged = perfect.copy()
    damaged[0, 0], damaged[1, 1], damaged[2, 0] = -1, 2, 3
    score = task.scores(damaged, rows)
    assert score["correct_pairs"] == 5 and score["exact_pair_accuracy"] == 5 / 8
    assert score["per_slot_correct"] == [6, 7] and score["unanswered_slots"] == 1
    constant = task.scores(np.zeros((8, 2), dtype=np.int64), rows)
    assert constant["exact_pair_accuracy"] == 0.25
    assert constant["per_slot_accuracy"] == [0.5, 0.5]
    assert constant["unequal_pair_accuracy"] == 0


def test_schedule_is_reproducible_and_uses_every_pair_in_the_full_factorial():
    schedule = task.draw(0, 512)
    assert schedule.shape == (512, 16, 4)
    np.testing.assert_array_equal(schedule, task.draw(0, 512))
    assert not np.array_equal(schedule, task.draw(1, 512))
    assert set(map(tuple, schedule.reshape(-1, 4))) == set(map(tuple, task.CASES))
    expected_rng = np.random.default_rng(np.random.SeedSequence([622022, 0]))
    np.testing.assert_array_equal(schedule, task.CASES[expected_rng.integers(8, size=(512, 16))])
    assert task.draw(0, 0).shape == (0, 16, 4)


def test_actual_random_references_include_the_stronger_binary_aware_policy():
    before = task.draw(0, 2)
    references = task.random_references(0)
    for policy, count, namespace in (("active_binary", 2, 622024), ("full_motor", 4, 622023)):
        reference = references[policy]
        expected_rng = np.random.default_rng(np.random.SeedSequence([namespace, 0]))
        expected = expected_rng.integers(count, size=(8, 2))
        np.testing.assert_array_equal(reference["answers"], expected)
        assert reference["scores"] == task.scores(expected, task.CASES)
        assert reference["choices"] == list(range(count))
        assert reference["expected_exact_pair_accuracy"] == 1 / count**2
        assert reference["expected_per_slot_accuracy"] == 1 / count
    assert references["active_binary"]["expected_exact_pair_accuracy"] == 0.25
    assert references["full_motor"]["expected_exact_pair_accuracy"] == 0.0625
    assert task.random_references(0) == references
    np.testing.assert_array_equal(task.draw(0, 2), before)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [[0, 0, 0]],
        [[0, 0, 0, 1]],
        [[0, 0, 2, 0]],
        [[2, 0, 0, 0]],
        [[0, -1, 0, 0]],
        [[0.0, 0.0, 0.0, 0.0]],
        [[False, False, False, False]],
        [[0, 0, 0, 2**64 - 1]],
    ],
)
def test_invalid_rows_are_not_reinterpreted_or_assigned_a_second_query(rows):
    for function in (task.targets, task.events, task.scores, task.swapped):
        args = (np.zeros((1, 2), dtype=np.int64), rows) if function is task.scores else (rows,)
        with pytest.raises(ValueError):
            function(*args)


@pytest.mark.parametrize(
    "answer",
    [
        np.zeros((8, 1), dtype=np.int64),
        np.zeros((8, 2), dtype=float),
        np.zeros((8, 2), dtype=bool),
        np.full((8, 2), -2),
        np.full((8, 2), 4),
    ],
)
def test_invalid_answer_shapes_or_motor_choices_refuse(answer):
    with pytest.raises(ValueError):
        task.scores(answer, task.CASES)


@pytest.mark.parametrize("value", [True, -1, 0.5, np.nan, np.inf, "0", None])
def test_invalid_schedule_values_refuse(value):
    with pytest.raises(ValueError):
        task.draw(value, 2)
    with pytest.raises(ValueError):
        task.draw(0, value)
    with pytest.raises(ValueError):
        task.draw(0, 2, value)


def test_no_hidden_value_or_exposure_fallbacks():
    with pytest.raises(ValueError):
        task.draw(0, 2, 0)
    for name in (True, -1, 2, 0.5):
        with pytest.raises(ValueError):
            task.flipped(task.CASES, name)
    with pytest.raises(TypeError):
        task.events(task.CASES, exposed=1)
    with pytest.raises(ValueError):
        task.random_reference(0, policy="best_of_binary_and_full")
