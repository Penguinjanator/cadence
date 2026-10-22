"""Fail-closed capacity admission, witnessed-drive custody and actual-work checks."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "attention_heads_admission_tests", Path(__file__).with_name("heads_admission.py")
)
assert SPEC is not None and SPEC.loader is not None
admission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = admission
SPEC.loader.exec_module(admission)
task = admission.task


def protocol() -> dict:
    data = json.loads(admission.PROTOCOL.read_text())
    data["brain"]["modules"] = [4]
    return data


def admitted_result() -> dict:
    rows = task.panel("train")
    answers = task.targets(rows)
    return {
        "training": {"completed": True, "rounds": 2, "wrong_counts": [3, 1],
                     "work": {"action_refusals": 0, "lesson_refusals": 0}},
        "parameters_preserved": True,
        "evaluation": {"train": {
            "answers": answers.tolist(), "scores": task.scores(answers, rows),
            "work": {"action_refusals": 0},
        }},
    }


def test_complete_perfect_capacity_panel_passes() -> None:
    assert admission.acquired(admitted_result(), 0.95)


@pytest.mark.parametrize("failure", [
    "partial_training", "changed_parameters", "training_action_refusal",
    "training_lesson_refusal", "evaluation_refusal", "missing_query",
    "empty_queries", "missing_slot", "nonfinite_query", "missing_answer",
    "unanswered", "wrong_scores", "incomplete_rounds",
])
def test_capacity_gate_rejects_partial_or_inconsistent_evidence(failure: str) -> None:
    result = admitted_result()
    training = result["training"]
    evaluation = result["evaluation"]["train"]
    if failure == "partial_training":
        training["completed"] = False
    elif failure == "changed_parameters":
        result["parameters_preserved"] = False
    elif failure.startswith("training_"):
        training["work"][failure.removeprefix("training_") + "s"] = 1
    elif failure == "evaluation_refusal":
        evaluation["work"]["action_refusals"] = 1
    elif failure in ("missing_query", "empty_queries"):
        evaluation["scores"]["per_query_accuracy"] = [1.0] if failure == "missing_query" else []
    elif failure == "missing_slot":
        evaluation["scores"]["per_slot_accuracy"] = [1.0]
    elif failure == "nonfinite_query":
        evaluation["scores"]["per_query_accuracy"][1] = float("nan")
    elif failure == "missing_answer":
        evaluation["answers"] = evaluation["answers"][:-1]
    elif failure == "unanswered":
        evaluation["answers"][0][1] = -1
        evaluation["scores"] = task.scores(evaluation["answers"], task.panel("train"))
    elif failure == "wrong_scores":
        evaluation["answers"][0][0] = (evaluation["answers"][0][0] + 1) % task.VALUES
    elif failure == "incomplete_rounds":
        training["rounds"] += 1
    assert not admission.acquired(result, 0.95)


@pytest.mark.parametrize("arm", ["exposed", "supplied_reads", "h1", "h2", "h1_wide_ports"])
def test_each_motor_slot_can_express_every_shared_value(arm: str) -> None:
    brain = admission.make_brain(arm, 0, protocol())
    np.testing.assert_array_equal(brain.learner.slot_sizes, [4, 4])
    assert len(brain.learner.output_index) == 8


def fake_brain(rows: np.ndarray):
    """A reusable stimulus buffer and genuine immutable two-head bank, without solving."""
    batch, width = len(rows), task.INPUTS
    first = admission.heads.Memory(
        np.arange(width), np.arange(width, 2 * width),
        np.zeros((batch, 2, width)), np.zeros((batch, 2, width)),
        np.ones((batch, 2), dtype=bool), np.arange(2.0), amplitude=1.5,
    )
    second = replace(first, query=np.arange(2 * width, 3 * width),
                     read=np.arange(3 * width, 4 * width))
    memory = admission.heads.MultiMemory((first, second))
    buffer = np.arange(batch * 4 * width, dtype=float).reshape(batch, 4 * width)
    brain = SimpleNamespace(
        brain=SimpleNamespace(memory=memory), recall=SimpleNamespace(amplitude=3.0),
        working_memory=SimpleNamespace(reset=lambda batch: None),
        basal_ganglia=SimpleNamespace(reset=lambda: None),
        stimulus=lambda query: buffer, reset=lambda: None,
    )
    return brain, buffer


def test_supplied_read_ports_contain_both_actual_facts_in_fixed_name_order() -> None:
    # Both prefix orders and both query orders; A/B values deliberately differ.
    rows = np.array([[0, 3, 0, 0], [0, 3, 1, 0], [2, 1, 0, 1], [2, 1, 1, 1]])
    brain, buffer = fake_brain(rows)
    old = brain.brain.memory
    query, drive, saved = admission.prepare(brain, rows, "supplied_reads")
    assert saved is brain.brain.memory
    assert all(member.amplitude == 0 for member in saved.memories)
    for h, member in enumerate(saved.memories):
        expected = np.zeros((len(rows), task.INPUTS))
        expected[:, 0] = 1  # IS
        expected[:, 5 + h] = 1  # fixed A/B identity, never ASK_AB/ASK_BA
        expected[np.arange(len(rows)), 7 + rows[:, h]] = 1
        np.testing.assert_array_equal(drive[:, member.read], 3 * expected)
    for name in ("keys", "values", "held", "ages"):
        np.testing.assert_array_equal(getattr(saved, name), getattr(old, name))
    assert not np.shares_memory(drive, buffer)
    assert not query[:, 5:].any()


def test_training_teaches_wrong_pairs_from_the_same_prequery_drive_and_bank(monkeypatch) -> None:
    rows = np.array([[0, 3, 0, 0], [2, 1, 1, 1]])
    brain, buffer = fake_brain(rows)
    saved_bank = brain.brain.memory
    saved_drive = buffer.copy()
    seen = []
    monkeypatch.setattr(admission.sr.Work, "act", lambda self, brain, observation: np.zeros((2, 2)))

    def act(brain, query, drive, work):
        # A query can replace the live boundary and overwrite a reusable buffer.
        brain.brain.memory = None
        buffer[:] = -999
        answer = task.targets(rows)
        answer[1, 0] = (answer[1, 0] + 1) % task.VALUES
        return answer

    def teach(brain, drive, target, selected, memory, work):
        seen.append(memory)
        np.testing.assert_array_equal(drive, saved_drive)
        np.testing.assert_array_equal(target, task.targets(rows))
        np.testing.assert_array_equal(selected, [False, True])

    monkeypatch.setattr(admission, "act_drive", act)
    monkeypatch.setattr(admission, "teach", teach)
    result = admission.train(brain, "h2", [rows], time.monotonic() + 5)
    assert result["completed"] and result["wrong_counts"] == [1]
    assert seen == [saved_bank]


def test_meter_charges_residual_checks_and_saved_teaching_and_restores_patch() -> None:
    brain = admission.make_brain("supplied_reads", 3, protocol())
    rows = np.array([[0, 3, 0, 0]])
    # These are numerical one-row phase checks, not a learning campaign.
    brain.act(task.events(rows)[0][0])
    query, drive, saved = admission.prepare(brain, rows, "supplied_reads")
    cost = {name: 0 for name in ("cortical_synapse_transports", "memory_scan_macs",
                                "softmax_elements", "memory_drive_evaluations")}
    original = admission.heads.Graph._synaptic_input
    with admission.metered(cost):
        work = admission.sr.Work()
        assert admission.act_drive(brain, query, drive, work) is not None
        before_residual = deepcopy(cost)
        brain.brain.residual(drive, brain.basal_ganglia.state)
        assert cost["cortical_synapse_transports"] > before_residual["cortical_synapse_transports"]
        assert cost["memory_scan_macs"] > before_residual["memory_scan_macs"]
        before_teacher = deepcopy(cost)
        admission.teach(brain, drive, task.targets(rows), np.array([True]), saved, work)
        assert cost["cortical_synapse_transports"] > before_teacher["cortical_synapse_transports"]
        assert cost["memory_drive_evaluations"] > before_teacher["memory_drive_evaluations"]
        assert work.lessons == 1 and work.teaching_residual_checks > 0
        assert brain.brain.memory is saved
    assert admission.heads.Graph._synaptic_input is original
