"""Admission arithmetic and phase custody, using synthetic records and no neural solves."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from cadence.receipts import Receipt, canonical_sha256

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "empty_boundary_subject", HERE / "empty_boundary_admission.py")
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def hashed(value):
    return canonical_sha256(value)


def action(rows=16, *, refused=False):
    residual = [0.004 if refused and row == 0 else 0.001 for row in range(rows)]
    flags = tuple(x <= 0.003 for x in residual)  # actual live Brain reports use tuples
    return {"steps": 32, "residual_checks": 2, "residual": tuple(residual),
            "max_residual": max(residual), "qualified": all(flags), "row_qualified": flags,
            "tolerance": 0.003, "budget": 2048}


def state(seed, update):
    return {"parameters": hashed(["parameters", seed, update]),
            "optimizer": hashed(["optimizer", seed, update]),
            "updates": update, "contrast_updates": update}


def evaluation(panel):
    rows = subject.task.panel(panel)
    answers = subject.task.targets(rows)
    work = subject.trace_work([{"prefix": [action(len(rows))] * 3,
                                "query": action(len(rows)), "answer": answers.tolist(),
                                "selected": [], "phases": []}], len(rows))
    return {"answers": answers.tolist(), "scores": subject.task.scores(answers, rows),
            "work": work, "association": None, "attention": None, "completed": True}


def round_record(seed, index, anchored):
    rows = subject.task.draw(seed, 2, 16)[index]
    before = state(seed, min(index, 1))
    failed = index == 1 and not anchored
    record = {"round": index + 1, "before": before, "after": before,
              "prefix": [action(refused=True)] if failed else [action()] * 3,
              "query": None if failed else action(), "query_drive": None,
              "bank": None, "held_any": None, "answer": None, "selected": [],
              "phases": [], "teacher_drive": None, "teacher_bank": None,
              "memory_empty_anchor": None, "phase_empty_anchor": [],
              "prefix_index": 0 if failed else None,
              "stop": "refused_prefix" if failed else None}
    if failed:
        refused = record["prefix"][0]
        record["failed_boundary"] = {
            "state": hashed(["failed", seed]), "steps": 32, "residual_checks": 2,
            "residual": list(refused["residual"]), "qualified": list(refused["row_qualified"]),
            "drive": hashed("failed_drive"), "bank": hashed("empty_bank"),
            "held_any": [False] * 16, "prefix_index": 0,
            "parameters": before["parameters"], "empty_anchor": False}
        return record
    answer = subject.task.targets(rows)
    record.update(query_drive=hashed(["drive", seed, index]), bank=hashed(["bank", seed, index]),
                  held_any=[True] * 16, memory_empty_anchor=anchored)
    if index == 0:
        answer[0, 0] = (answer[0, 0] + 1) % 4
        record.update(selected=[0], teacher_drive=hashed(["teacher_drive", seed]),
                      teacher_bank=hashed(["teacher_bank", seed]), after=state(seed, 1))
        record["phase_empty_anchor"] = [anchored] * 3
        record["phases"] = [{
            "state": hashed(["phase", seed, name]), "steps": 32, "residual_checks": 2,
            "residual": [0.001], "qualified": [True], "drive": record["teacher_drive"],
            "bank": record["teacher_bank"], "parameters": before["parameters"],
        } for name in ("free", "nudged", "opposite")]
    record["answer"] = answer.tolist()
    return record


@pytest.fixture
def specimen(monkeypatch, tmp_path):
    declared = json.loads(subject.PROTOCOL.read_text())
    declared["rounds"] = 2  # smaller synthetic ledger, not a behavioral recipe/run
    protocol_path = tmp_path / subject.PROTOCOL.name
    protocol_path.write_text(json.dumps(declared))
    monkeypatch.setattr(subject, "PROTOCOL", protocol_path)
    prior_real = Receipt.read(subject.REFERENCE)
    runs = []
    old_runs = []
    for variant in declared["variants"]:
        for seed in declared["development_seeds"]:
            old = next(r for r in prior_real.body["runs"]
                       if r["arm"] == "supplied_reads" and r["seed"] == seed)
            anchored = variant["empty_anchor"]
            trace = [round_record(seed, i, anchored) for i in range(2)]
            training = {"completed": anchored, "rounds": 2 if anchored else 1,
                        "wrong_counts": [1, 0] if anchored else [1],
                        "stop": None if anchored else "refused_prefix",
                        "work": subject.trace_work(trace, 16)}
            run = {"variant": variant["name"], "seed": seed, "empty_anchor": anchored,
                   "training": training, "trace": trace, "initial": evaluation("all"),
                   "evaluation": {panel: evaluation(panel) for panel in ("train", "held_out")},
                   "initial_parameters": hashed(["initial", seed]),
                   "final_parameters": hashed(["final", seed]),
                   "initial_state": state(seed, 0), "final_state": state(seed, 1),
                   "evaluation_parameters": hashed(["final", seed]),
                   "evaluation_state": state(seed, 1),
                   "parameters_preserved": True,
                   "schedule_sha256": canonical_sha256(subject.task.draw(seed, 2, 16)),
                   "neuron_model": deepcopy(old["neuron_model"]),
                   "resolved": deepcopy(old["resolved"]),
                   "learning_config": deepcopy(declared["learning_config"]),
                   "elapsed_host_seconds": 0.1}
            run["work"] = subject.transport_work(run)
            run["admitted"] = subject.admitted(run, declared)
            runs.append(run)
            if not anchored:
                old_runs.append({"arm": "supplied_reads", "seed": seed,
                                 "training": deepcopy(training),
                                 "final_parameters": run["final_parameters"],
                                 "neuron_model": deepcopy(old["neuron_model"]),
                                 "resolved": deepcopy(old["resolved"])})
    prior = SimpleNamespace(body={"runs": old_runs}, digest="f" * 64)
    monkeypatch.setattr(subject, "reference", lambda: prior)
    elapsed = 1.0
    body = {"protocol_file": protocol_path.name, "protocol": declared, "runs": runs,
            "reference_receipt_digest": prior.digest, "masked_historical_reproduced": True,
            "occupied_query_overlap_identical": True, "passed": True,
            "decision": "empty_boundary_acquisition_admitted; selection_pending",
            "candidate_rounds_used": 0, "confirmation_seeds_used": [],
            "elapsed_host_seconds": elapsed,
            "previous_capacity_host_seconds": declared["budget"]["old_capacity_phase_host_seconds"],
            "previous_diagnostic_host_seconds": declared["budget"][
                "preceding_separate_diagnostic_host_seconds"],
            "cumulative_host_seconds": declared["budget"]["old_capacity_phase_host_seconds"]
            + declared["budget"]["preceding_separate_diagnostic_host_seconds"] + elapsed}
    return body


def test_synthetic_complete_ledger_passes_with_actual_tuple_action_report_format(specimen):
    assert subject.check(specimen) is None


def test_overlap_requires_every_old_answer_and_the_last_refused_rounds_prior_state(specimen):
    masked, anchored = specimen["runs"][0], specimen["runs"][2]
    assert subject.matched(masked, anchored)
    anchored["trace"][0]["answer"] = None
    assert not subject.matched(masked, anchored)


def test_overlap_rejects_truncation_and_before_state_drift(specimen):
    masked, anchored = specimen["runs"][0], specimen["runs"][2]
    short = deepcopy(anchored)
    short["trace"] = short["trace"][:1]
    assert not subject.matched(masked, short)
    anchored["trace"][1]["before"]["parameters"] = hashed("drift")
    assert not subject.matched(masked, anchored)


def test_prefix_solver_work_may_differ_while_occupied_query_identity_is_preserved(specimen):
    masked, anchored = specimen["runs"][0], specimen["runs"][2]
    anchored["trace"][0]["prefix"][0]["steps"] += 32
    assert subject.matched(masked, anchored)


def test_gate_recomputes_raw_decisions_despite_optimistic_scores(specimen):
    result = specimen["runs"][2]
    assert subject.admitted(result, specimen["protocol"])
    result["evaluation"]["train"]["answers"] = [[0, 0]] * 48
    assert not subject.admitted(result, specimen["protocol"])


@pytest.mark.parametrize("change", [
    "missing_run", "founder", "flag", "protocol", "reference", "schedule", "model",
    "config", "genes", "layout", "answer", "selection", "phase_residual", "phase_qualified",
    "phase_boundary", "phase_parameters", "teacher_flag", "query_flag", "failed_state",
    "failed_residual", "failed_held", "training_work", "transport", "completed", "stop",
    "after_state", "score", "unanswered", "hidden_eval_refusal", "eval_learned", "admitted",
    "overlap", "historical", "confirmation", "candidate", "charge", "infinite_time",
    "extra_panel", "eval_state", "preservation_flag",
])
def test_strict_verifier_rejects_forged_admission_phase_custody_and_work(specimen, change):
    body = specimen
    anchored = body["runs"][2]
    row = anchored["trace"][0]
    failed = body["runs"][0]["trace"][-1]["failed_boundary"]
    reading = anchored["evaluation"]["train"]
    if change == "missing_run":
        body["runs"].pop()
    elif change == "founder":
        anchored["seed"] = 2
    elif change == "flag":
        anchored["empty_anchor"] = False
    elif change == "protocol":
        body["protocol"]["gate"]["trained_floor"] = 0.0
    elif change == "reference":
        body["reference_receipt_digest"] = "0" * 64
    elif change == "schedule":
        anchored["schedule_sha256"] = "0" * 64
    elif change == "model":
        anchored["neuron_model"]["dt"] = 0.2
    elif change == "config":
        anchored["learning_config"]["beta"] = 0.2
    elif change == "genes":
        anchored["resolved"]["genes"]["recall_amplitude"] = 6.0
    elif change == "layout":
        anchored["resolved"]["layout"]["modules"] = [64]
    elif change == "answer":
        row["answer"][0][0] = -1
    elif change == "selection":
        row["selected"] = []
    elif change == "phase_residual":
        row["phases"][0]["residual"] = [0.5]
    elif change == "phase_qualified":
        row["phases"][0]["qualified"] = [False]
    elif change == "phase_boundary":
        row["phases"][1]["bank"] = hashed("different bank")
    elif change == "phase_parameters":
        row["phases"][1]["parameters"] = hashed("different parameters")
    elif change == "teacher_flag":
        row["phase_empty_anchor"][1] = False
    elif change == "query_flag":
        row["memory_empty_anchor"] = False
    elif change == "failed_state":
        failed["state"] = "invalid"
    elif change == "failed_residual":
        failed["residual"][0] = 0.005
    elif change == "failed_held":
        failed["held_any"][0] = True
    elif change == "training_work":
        anchored["training"]["work"]["action_sweeps"] += 1
    elif change == "transport":
        anchored["work"]["fixed_read_coordinate_updates"] += 1
    elif change == "completed":
        anchored["training"]["completed"] = False
    elif change == "stop":
        anchored["training"]["stop"] = "wall_budget"
    elif change == "after_state":
        row["after"]["contrast_updates"] += 1
    elif change == "score":
        reading["scores"]["exact_pair_accuracy"] = 0.5
    elif change == "unanswered":
        reading["answers"][0] = [-1, -1]
        reading["scores"] = subject.task.scores(reading["answers"], subject.task.panel("train"))
    elif change == "hidden_eval_refusal":
        reading["work"]["action_refusals"] = 1
    elif change == "eval_learned":
        reading["work"]["lessons"] = 1
    elif change == "admitted":
        anchored["admitted"] = False
    elif change == "overlap":
        body["occupied_query_overlap_identical"] = False
    elif change == "historical":
        body["masked_historical_reproduced"] = False
    elif change == "confirmation":
        body["confirmation_seeds_used"] = [200]
    elif change == "candidate":
        body["candidate_rounds_used"] = 1
    elif change == "charge":
        body["cumulative_host_seconds"] -= 1.0
    elif change == "extra_panel":
        anchored["evaluation"]["undeclared"] = deepcopy(reading)
    elif change == "eval_state":
        anchored["evaluation_state"]["optimizer"] = hashed("changed evaluation optimizer")
    elif change == "preservation_flag":
        anchored["parameters_preserved"] = False
    else:
        body["elapsed_host_seconds"] = float("inf")
    assert subject.check(body) is not None


def test_atomic_partial_journal_preserves_unverified_body_before_final_checks(tmp_path):
    path = tmp_path / "development.json.partial.json"
    original = {"status": "unverified", "body": {"actual_answers": [[1, 2]], "passed": False}}
    subject.journal_write(path, original)
    assert json.loads(path.read_text()) == original
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_failed_final_check_preserves_all_actual_trial_records(specimen, monkeypatch, tmp_path):
    def recorded_trial(variant, seed, declared, recipe, deadline):
        return deepcopy(next(r for r in specimen["runs"]
                             if r["variant"] == variant["name"] and r["seed"] == seed))

    monkeypatch.setattr(subject, "trial", recorded_trial)
    monkeypatch.setattr(subject, "check", lambda body: "forced arithmetic-check failure")
    journal = tmp_path / "failed.json.partial.json"
    with pytest.raises(AssertionError, match="forced arithmetic"):
        subject.run(journal=journal)
    stored = json.loads(journal.read_text())
    assert stored["status"] == "check_failed" and len(stored["body"]["runs"]) == 4
    assert stored["body"]["runs"][0]["trace"][0]["answer"] == (
        specimen["runs"][0]["trace"][0]["answer"])


def test_tiny_actual_model_instrumentation_counts_actual_work_and_restores_wrappers():
    # One tiny software-contract trial, not the declared behavioral campaign.
    recipe = json.loads(subject.BASE.read_text())
    recipe["brain"].update(modules=[4], empty_anchor=True)
    brain = subject.admission.make_brain("supplied_reads", 0, recipe)
    graph_type = subject.admission.heads.Graph
    original_solve = graph_type.equilibrate
    counter = {name: 0 for name in ("cortical_synapse_transports", "memory_scan_macs",
                                   "softmax_elements", "memory_drive_evaluations",
                                   "fixed_read_coordinate_updates")}
    with subject.admission.metered(counter):
        training, trace = subject.train(brain, subject.task.draw(0, 1, 2), time.monotonic() + 10)
    assert training["completed"] and len(trace) == 1
    assert subject.trace_work(trace, 2) == training["work"]
    empty = {"work": asdict(subject.sr.Work())}
    result = {"training": training, "initial": empty, "evaluation": {},
              "resolved": {"layout": {"synapses": brain.connectome.synapses}}}
    assert subject.transport_work(result) == counter
    assert graph_type.equilibrate is original_solve
    assert "_qualified_phase" not in brain.learner.__dict__
    subject._phase(trace[0]["prefix"][0], 2, 0.003, 2048)
    assert all(flag is True for flag in trace[0]["phase_empty_anchor"])
