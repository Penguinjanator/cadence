"""The adversarial task must require bindings, preserve holdout and count refusals."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("dual_binding_task", Path(__file__).with_name(
    "dual_binding.py"))
task = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = task
spec.loader.exec_module(task)


def test_swapping_bindings_keeps_query_and_token_multiset_but_changes_both_answers():
    rows = task.CASES[task.CASES[:, 0] != task.CASES[:, 1]]
    prefix, query = task.events(rows)
    swapped_prefix, swapped_query = task.events(task.swapped(rows))
    np.testing.assert_array_equal(query, swapped_query)
    np.testing.assert_array_equal(sum(prefix), sum(swapped_prefix))
    assert (task.targets(rows) != task.targets(task.swapped(rows))).all()
    # Each first/second fact shares value coordinates; a uniform memory mixture loses binding.
    assert not np.array_equal(prefix[0], swapped_prefix[0])


def test_holdout_contains_complete_unseen_pairs_with_balanced_value_marginals():
    train, held = task.panel("train"), task.panel("held_out")
    assert (len(train), len(held)) == (48, 16)
    assert set(map(tuple, train[:, :2])).isdisjoint(set(map(tuple, held[:, :2])))
    for slot in range(2):
        np.testing.assert_array_equal(np.bincount(train[:, slot]), np.full(4, 12))
        np.testing.assert_array_equal(np.bincount(held[:, slot]), np.full(4, 4))
    schedule = task.draw(0, 512)
    assert schedule.shape == (512, 16, 4)
    assert not any(tuple(row[:2]) in task.HELD_PAIRS for row in schedule.reshape(-1, 4))
    np.testing.assert_array_equal(task.draw(0, 512), schedule)


def test_query_contains_no_values_and_exposed_control_provides_both_unselected_facts():
    _, query = task.events(task.CASES)
    _, exposed = task.events(task.CASES, exposed=True)
    assert not query[:, 5:].any()
    assert (query[:, 2] == 1).all()
    assert (query[:, 3:5].sum(axis=1) == 1).all()
    assert (exposed[:, 11:15].sum(axis=1) == 1).all()
    assert (exposed[:, 15:19].sum(axis=1) == 1).all()
    np.testing.assert_array_equal(exposed[:, :11], query[:, :11])


def test_pair_score_does_not_hide_one_correct_slot_or_refused_cases():
    answers = task.targets(task.CASES)
    assert task.scores(answers, task.CASES)["exact_pair_accuracy"] == 1
    answers[:, 1] = -1
    score = task.scores(answers, task.CASES)
    assert score["exact_pair_accuracy"] == 0
    assert score["per_slot_accuracy"] == [1.0, 0.0]
    assert score["unanswered_slots"] == len(task.CASES)


@pytest.mark.parametrize("rows", [[], [[0, 1, 0]], [[4, 0, 0, 0]], [[0, 0, 2, 0]],
                                  [[0.0, 0.0, 0.0, 0.0]], [[-1, 0, 0, 0]]])
def test_invalid_cases_are_rejected(rows):
    with pytest.raises(ValueError):
        task.events(rows)
