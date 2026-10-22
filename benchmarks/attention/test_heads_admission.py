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


def test_supplied_capacity_control_disables_memory_for_every_prefix_moment():
    protocol = json.loads(admission.PROTOCOL.read_text())
    brain = admission.make_brain("supplied_reads", 0, protocol)
    rows = admission.task.panel("train")[:2]
    prefix, _ = admission.task.events(rows)
    work = admission.sr.Work()
    for event in prefix:
        assert work.act(brain, event) is not None
        assert all(member.amplitude == 0.0 for member in brain.brain.memory.memories)
    assert brain.recall.held.any()  # The control still witnesses the actual facts.
    _, drive, memory = admission.prepare(brain, rows, "supplied_reads")
    assert all(member.amplitude == 0.0 for member in memory.memories)
    assert admission.protocol_amplitude(brain) == 3.0
    assert max(drive[:, memory.memories[0].read].ravel()) == 3.0


@pytest.mark.parametrize("selector", [
    "../heads-protocol.json", "other.json", "/heads-protocol.json",
])
def test_protocol_selector_rejects_any_unfrozen_file(selector):
    with pytest.raises(ValueError, match="declared"):
        admission.protocol_path(selector)


def test_variant_sources_bind_both_active_and_original_protocols(monkeypatch):
    original = admission.HERE / "heads-protocol.json"
    longer = admission.protocol_path("heads-longer-protocol.json")
    monkeypatch.setattr(admission, "PROTOCOL", longer)
    paths = dict(admission.sources())
    assert paths["benchmarks/attention/heads-longer-protocol.json"] == longer
    assert paths["benchmarks/attention/heads-protocol.json"] == original
    assert len(paths) == len(admission.sources())


def test_historical_check_uses_its_frozen_base_protocol_when_variant_is_active(monkeypatch):
    path = admission.HERE / "results" / "heads-capacity-corrected.json"
    body = json.loads(path.read_text())["body"]
    base = json.loads((admission.HERE / "heads-protocol.json").read_text())
    longer = admission.protocol_path("heads-longer-protocol.json")
    monkeypatch.setattr(admission, "PROTOCOL", longer)
    assert admission.check(body, protocol=base) is None
    assert admission.check(body) == "heads protocol selector differs"
    assert admission.verify_prior(path).body["protocol"] == base


def test_numeric_retry_changes_only_declared_solver_budgets_in_the_brain():
    base = json.loads((admission.HERE / "heads-protocol.json").read_text())
    longer = json.loads(admission.protocol_path("heads-longer-protocol.json").read_text())
    for data in (base, longer):
        data["brain"]["modules"] = [4]
    before = admission.make_brain("h1", 0, base)
    after = admission.make_brain("h1", 0, longer)
    work_control = admission.make_brain("h1_more_steps", 0, longer)
    assert after.learner.config.free_steps == after.learner.config.nudged_steps == 2048
    assert work_control.learner.config.free_steps == 4096
    assert work_control.learner.config.nudged_steps == 4096
    for key in before.learner.config.__slots__:
        if key not in ("free_steps", "nudged_steps"):
            assert getattr(before.learner.config, key) == getattr(after.learner.config, key)
    for key in ("efficacy", "bias", "log_gain"):
        np.testing.assert_array_equal(getattr(before.brain, key), getattr(after.brain, key))
    assert before.brain.neuron_model == after.brain.neuron_model


def failed_result(arm, seed, data):
    evaluation = {}
    for name in ("train", "held_out", "all"):
        rows = task.panel(name)
        answers = np.full((len(rows), 2), -1, dtype=np.int64)
        evaluation[name] = {"answers": answers.tolist(), "scores": task.scores(answers, rows),
                            "work": {"action_refusals": 0}}
    return {"arm": arm, "seed": seed, "acquired": False, "parameters_preserved": True,
            "training": {"completed": False, "rounds": 0, "wrong_counts": [],
                         "work": {"action_refusals": 0, "lesson_refusals": 0}},
            "initial": evaluation["all"], "evaluation": evaluation,
            "schedule_sha256": admission.canonical_sha256(task.draw(
                seed, data["task"]["rounds"], data["task"]["batch"]))}


def test_retry_charges_the_complete_prior_chain_and_declared_diagnostics(tmp_path, monkeypatch):
    data = json.loads(admission.protocol_path("heads-longer-protocol.json").read_text())
    data["task"]["rounds"] = 3
    frozen = tmp_path / "heads-longer-protocol.json"
    frozen.write_text(json.dumps(data))
    monkeypatch.setattr(admission, "PROTOCOL", frozen)
    monkeypatch.setattr(admission.time, "monotonic", lambda: 100.0)
    prior = SimpleNamespace(body={"elapsed_host_seconds": 10.0, "previous_host_seconds": 20.0},
                            digest="d" * 64)
    monkeypatch.setattr(admission, "verify_prior", lambda path: prior)
    deadlines = []

    def trial(arm, seed, protocol, deadline):
        deadlines.append(deadline)
        return failed_result(arm, seed, protocol)

    monkeypatch.setattr(admission, "trial", trial)
    receipt = admission.run(Path("retained-prior.json"))
    expected = 30.0 + data["diagnostic_charge_seconds"]
    assert receipt.body["previous_host_seconds"] == expected
    assert receipt.body["prior_reference"] == prior.digest
    assert receipt.body["protocol_file"] == "heads-longer-protocol.json"
    assert deadlines == [100.0 + 600.0 - expected] * 2
    assert admission.check(receipt.body) is None
    with pytest.raises(ValueError, match="requires"):
        admission.run()
    prior.body["protocol_file"] = "heads-longer-protocol.json"
    with pytest.raises(ValueError, match="one numerical retry"):
        admission.run(Path("retained-prior.json"))


def fixed_protocol():
    data = json.loads(admission.protocol_path("heads-fixed-read-protocol.json").read_text())
    data["brain"]["modules"] = [4]
    return data


def test_anchored_control_keeps_candidate_anatomy_and_read_force_without_external_answers():
    data = fixed_protocol()
    supplied = admission.make_brain("supplied_reads", 0, data)
    candidate = admission.make_brain("h2", 0, data)
    assert isinstance(supplied, admission.FixedReadBrain)
    assert not isinstance(candidate, admission.FixedReadBrain)
    assert not hasattr(supplied, "_supplied_content_amplitude")
    for key in ("pre", "post", "count", "sign"):
        np.testing.assert_array_equal(getattr(supplied.connectome, key),
                                      getattr(candidate.connectome, key))
    np.testing.assert_array_equal(supplied.brain.efficacy, candidate.brain.efficacy)
    rows = np.array([[0, 3, 0, 0], [2, 1, 1, 1]])
    work = admission.sr.Work()
    for event in admission.task.events(rows)[0]:
        assert work.act(supplied, event) is not None
        assert supplied.brain.memory.fixed_selector
        assert [member.amplitude for member in supplied.brain.memory.memories] == [1.5, 1.5]
    query, drive, snapshot = admission.prepare(supplied, rows, "supplied_reads")
    assert snapshot.fixed_selector and snapshot.mode == "held"
    assert not query[:, 5:].any()
    for name, member in enumerate(snapshot.memories):
        assert not drive[:, member.read].any()
        expected = np.zeros((len(rows), task.INPUTS))
        expected[:, 0] = expected[:, 5 + name] = 1.0
        expected[np.arange(len(rows)), 7 + rows[:, name]] = 1.0
        np.testing.assert_array_equal(snapshot.targets[name], expected)


def test_fixed_capacity_row_teaching_and_lesions_keep_the_held_boundary_and_cost_scope():
    brain = admission.make_brain("supplied_reads", 1, fixed_protocol())
    rows = np.array([[0, 3, 0, 0], [2, 1, 1, 1]])
    for event in admission.task.events(rows)[0]:
        brain.act(event)
    query, drive, snapshot = admission.prepare(brain, rows, "supplied_reads")
    cost = {name: 0 for name in ("cortical_synapse_transports", "memory_scan_macs",
                                "softmax_elements", "memory_drive_evaluations")}
    work = admission.sr.Work()
    with admission.metered(cost):
        assert admission.act_drive(brain, query, drive, work) is not None
        admission.teach(brain, drive, task.targets(rows), np.array([False, True]), snapshot, work)
    assert brain.brain.memory is snapshot and snapshot.streams == 2
    assert cost["fixed_read_coordinate_updates"] > 0
    assert cost["memory_scan_macs"] == cost["softmax_elements"] == 0
    assert cost["memory_drive_evaluations"] > 0
    assert work.accepted_presentations == 1 and work.lesson_refusals == 0
    _, _, erased = admission.prepare(brain, rows, "supplied_reads", "erased")
    assert erased.fixed_selector and not erased.held.any() and not erased.targets.any()
    _, _, lesioned = admission.prepare(brain, rows, "supplied_reads", "head_0")
    assert [member.amplitude for member in lesioned.memories] == [0.0, 1.5]


def test_fixed_variant_sources_and_prior_preserve_both_earlier_protocols(monkeypatch):
    fixed = admission.protocol_path("heads-fixed-read-protocol.json")
    monkeypatch.setattr(admission, "PROTOCOL", fixed)
    paths = dict(admission.sources())
    for name in ("heads-protocol.json", "heads-longer-protocol.json",
                 "heads-fixed-read-protocol.json", "fixed_read.py"):
        assert paths["benchmarks/attention/" + name] == admission.HERE / name
    prior = admission.verify_prior(admission.HERE / "results" / "heads-capacity-longer.json")
    assert prior.body["protocol_file"] == "heads-longer-protocol.json"
    assert not prior.body["capacity_admitted"]


def test_one_anchored_revision_requires_and_charges_the_failed_longer_run(tmp_path, monkeypatch):
    data = fixed_protocol()
    data["task"]["rounds"] = 3
    frozen = tmp_path / "heads-fixed-read-protocol.json"
    frozen.write_text(json.dumps(data))
    monkeypatch.setattr(admission, "PROTOCOL", frozen)
    prior = SimpleNamespace(body={"protocol_file": "heads-longer-protocol.json",
                                  "elapsed_host_seconds": 10.0, "previous_host_seconds": 20.0},
                            digest="d" * 64)
    monkeypatch.setattr(admission, "verify_prior", lambda path: prior)
    monkeypatch.setattr(admission, "trial", lambda arm, seed, protocol, deadline:
                        failed_result(arm, seed, protocol))
    result = admission.run(Path("failed-longer.json"))
    assert result.body["previous_host_seconds"] == 30.0
    assert result.body["protocol_file"] == "heads-fixed-read-protocol.json"
    with pytest.raises(ValueError, match="requires"):
        admission.run()
    prior.body["protocol_file"] = "heads-fixed-read-protocol.json"
    with pytest.raises(ValueError, match="one anchored control revision"):
        admission.run(Path("failed-fixed.json"))

