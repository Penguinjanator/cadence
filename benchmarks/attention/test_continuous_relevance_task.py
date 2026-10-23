"""Pure NumPy task tests; importing any Cadence/model module is forbidden."""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "continuous_relevance_task_tests", Path(__file__).with_name("continuous_relevance_task.py")
)
task = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = task
SPEC.loader.exec_module(task)


@pytest.fixture(autouse=True)
def forbid_model_imports(monkeypatch):
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name == "cadence" or name.startswith("cadence."):
            pytest.fail("array fixtures must not import or construct a model")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)


def test_every_draw_has_all_sixteen_strata_and_pairs_the_same_actual_quantities():
    for index in range(12):
        batch = task.draw(0, index)
        record = task.record(batch)
        rows, values = np.array(record["rows"]), np.array(record["quantities"])
        np.testing.assert_array_equal(
            np.unique(rows, axis=0), np.array([(*r, q) for r in task.HISTORIES for q in (0, 1)])
        )
        np.testing.assert_array_equal(values[::2], values[1::2])
        np.testing.assert_array_equal(values[::2], batch.quantities)
        assert np.all((values >= 0.5).astype(int) == rows[:, :2])
        assert np.all(values >= np.where(rows[:, :2] == 0, 0.1, 0.6))
        assert np.all(values <= np.where(rows[:, :2] == 0, 0.4, 0.9))
    assert not np.array_equal(task.draw(0, 0).rows, task.draw(0, 1).rows)


def test_reproducible_independent_streams_keep_full_fresh_payload_schedules():
    before = task.schedule(3, 3, split="held")
    task.schedule(3, 11, split="train")
    after = task.schedule(3, 3, split="held")
    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left.rows, right.rows)
        np.testing.assert_array_equal(left.quantities, right.quantities)
    schedules = [task.schedule(3, 3, split=s) for s in task.SPLITS]
    assert len({json.dumps([task.record(b) for b in schedule]) for schedule in schedules}) == 3
    train = task.schedule(3, 3)
    assert len({float(x) for b in train for x in b.quantities.flat}) == 48
    assert task.schedule(3, 0) == ()


def test_only_witnessed_facts_have_a_shared_payload_question_has_no_answer_or_address():
    batch = task.draw(5, 0)
    prefixes = []
    for query in (0, 1):
        prefix, question = task.events(batch, query)
        prefixes.append(prefix)
        np.testing.assert_array_equal(question, np.tile(np.eye(9)[5 + query], (8, 1)))
        for i, row in enumerate(batch.rows):
            for tick, name in enumerate((int(row[2]), 1 - int(row[2]))):
                expected = np.zeros(9)
                expected[[name, task.IS]] = 1
                expected[task.PAYLOAD] = batch.quantities[i, name]
                np.testing.assert_array_equal(prefix[tick][i], expected)
        np.testing.assert_array_equal(prefix[2], np.tile(np.eye(9)[4], (8, 1)))
        np.testing.assert_array_equal(task.targets(batch, query), batch.rows[:, query])
    np.testing.assert_array_equal(prefixes[0], prefixes[1])


def test_exposed_control_receives_both_actual_unselected_scalars_for_each_question():
    batch = task.draw(8, 2)
    for query in (0, 1):
        plain, question = task.events(batch, query)
        exposed, given = task.events(batch, query, exposed=True)
        np.testing.assert_array_equal(plain, exposed)
        np.testing.assert_array_equal(given[:, :7], question[:, :7])
        np.testing.assert_array_equal(given[:, 7:], batch.quantities)


@pytest.mark.parametrize("query", [0, 1])
@pytest.mark.parametrize("name", [0, 1])
@pytest.mark.parametrize("opposite", [False, True])
def test_actual_donor_changes_one_continuous_quantity_and_preserves_host_question(
    query, name, opposite
):
    host = task.draw(6, 0)
    changed = task.donor(host, name, 6, 0, opposite_class=opposite)
    np.testing.assert_array_equal(changed.quantities[:, 1 - name], host.quantities[:, 1 - name])
    np.testing.assert_array_equal(changed.rows[:, 2], host.rows[:, 2])
    assert np.all(changed.quantities[:, name] != host.quantities[:, name])
    _, original_query = task.events(host, query)
    _, donor_query = task.events(changed, query)
    np.testing.assert_array_equal(original_query, donor_query)
    expected = task.targets(host, query) ^ int(opposite and name == query)
    np.testing.assert_array_equal(task.targets(changed, query), expected)
    # The alternate event changes only the chosen fact's shared payload coordinate.
    before, _ = task.events(host, query)
    after, _ = task.events(changed, query)
    for tick in (0, 1):
        difference = before[tick] != after[tick]
        assert not difference[:, np.arange(9) != task.PAYLOAD].any()
        np.testing.assert_array_equal(difference[:, task.PAYLOAD], before[tick][:, name] == 1)


def test_neutral_amplitude_and_prefix_order_controls_preserve_quantities_and_question():
    batch = task.draw(1, 7)
    original, question = task.events(batch, 1)
    for amplitude in (0.0, 0.37, 0.5, 1.0):
        changed, asked = task.events(batch, 1, neutral_amplitude=amplitude)
        np.testing.assert_array_equal(changed[:2], original[:2])
        np.testing.assert_array_equal(changed[2], amplitude * original[2])
        np.testing.assert_array_equal(asked, question)
    swapped, asked = task.events(batch, 1, reverse_order=True)
    np.testing.assert_array_equal(swapped[0], original[1])
    np.testing.assert_array_equal(swapped[1], original[0])
    np.testing.assert_array_equal(swapped[2], original[2])
    np.testing.assert_array_equal(asked, question)


def test_frozen_arrays_and_records_have_independent_custody():
    batch = task.draw(2, 2)
    with pytest.raises(ValueError):
        batch.quantities[0, 0] = 0
    with pytest.raises(FrozenInstanceError):
        batch.rows = task.HISTORIES
    original = task.record(batch)
    prefix, question = task.events(batch, 0)
    prefix[0][:] = 0
    question[:] = 99
    data = task.record(batch)
    data["quantities"][0][0] = 99
    assert task.record(batch) == original


def test_unanswered_and_wrong_queries_remain_errors_in_every_score_denominator():
    batch = task.draw(2, 1)
    answers = np.column_stack([task.targets(batch, q) for q in (0, 1)])
    assert task.scores(answers, batch)["accuracy"] == 1
    answers[0, 0] = -1
    score = task.scores(answers, batch)
    assert score["accuracy"] == 15 / 16 and score["unanswered"] == 1
    assert score["both_questions_correct"] == 7 / 8


def test_equal_fixed_mixture_can_hide_opposite_answers():
    # At equal fixed weights these two permitted histories give the SAME complete read.
    facts = np.zeros((2, 2, 9))
    facts[:, 0, [task.NAME_A, task.IS]] = 1
    facts[:, 1, [task.NAME_B, task.IS]] = 1
    facts[:, :, task.PAYLOAD] = [[0.2, 0.8], [0.8, 0.2]]
    np.testing.assert_array_equal(facts[0].mean(axis=0), facts[1].mean(axis=0))
    assert facts[0, 0, task.PAYLOAD] < 0.5 < facts[1, 0, task.PAYLOAD]


def test_soft_content_dependent_weights_can_still_encode_both_quantities():
    # Counterexample to a necessity claim: exposed name masses plus a scalar mixture
    # can be decoded without a majority fact choice.
    batch = task.draw(9, 3, split="held")
    values = batch.quantities
    coefficient = 0.2
    logits = np.column_stack((coefficient * values, np.full(8, 2.0), np.zeros(8)))
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    pa, pb = weights[:, 0], weights[:, 1]
    assert max(pa.max(), pb.max()) < 0.5
    mixture = pa * values[:, 0] + pb * values[:, 1]
    difference = np.log(pa / pb) / coefficient
    recovered_b = (mixture - pa * difference) / (pa + pb)
    recovered_a = recovered_b + difference
    np.testing.assert_allclose(np.column_stack((recovered_a, recovered_b)), values, atol=1e-14)


@pytest.mark.parametrize(
    "call",
    [
        lambda: task.draw(True, 0),
        lambda: task.draw(0, -1),
        lambda: task.draw(0, 0, split="unknown"),
        lambda: task.schedule(0, 0.5),
        lambda: task.events(task.draw(0, 0), 2),
        lambda: task.events(task.draw(0, 0), True),
        lambda: task.events(task.draw(0, 0), 0, neutral_amplitude=np.nan),
        lambda: task.Histories(task.HISTORIES, np.full((8, 2), 0.5)),
    ],
)
def test_invalid_or_gap_payload_inputs_refuse(call):
    with pytest.raises(ValueError):
        call()
