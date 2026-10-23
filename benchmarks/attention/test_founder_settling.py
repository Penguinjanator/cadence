"""Founder numerical-policy custody and separate, cumulative phase budgets."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from cadence import Brain
from cadence.receipts import Receipt, canonical_sha256

SPEC = importlib.util.spec_from_file_location(
    "heads_founder_settling_tests", Path(__file__).with_name("heads_relevance.py"))
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)
admission, task = subject.admission, subject.task


def protocol():
    return json.loads(admission.protocol_path(admission.FOUNDER_PROTOCOL).read_text())


def result(arm, seed, data):
    brain = admission.make_brain(arm, seed, data)
    evaluation = {}
    for panel in ("train", "held_out", "all"):
        rows = task.panel(panel)
        answers = task.targets(rows)
        evaluation[panel] = {"answers": answers.tolist(), "scores": task.scores(answers, rows),
                             "work": {"action_refusals": 0}}
    return {
        "arm": arm, "seed": seed, "mode": "agreement", "acquired": True,
        "training": {"completed": True, "rounds": data["task"]["rounds"],
                     "wrong_counts": [1] * data["task"]["rounds"],
                     "work": {"action_refusals": 0, "lesson_refusals": 0}},
        "initial": deepcopy(evaluation["all"]), "evaluation": evaluation,
        "initial_parameters": "a" * 64, "final_parameters": "b" * 64,
        "evaluation_parameters": "b" * 64, "parameters_preserved": True,
        "schedule_sha256": canonical_sha256(task.draw(
            seed, data["task"]["rounds"], data["task"]["batch"])),
        "resolved": brain.describe(), "neuron_model": brain.brain.neuron_model.to_dict(),
        "learning_config": brain.learner.config.to_dict(),
        "numerical_policy": admission.numerical_settings(brain),
        "neurons": brain.connectome.n, "synapses": brain.connectome.synapses,
    }


@pytest.fixture
def fixture(monkeypatch):
    data = protocol()
    monkeypatch.setattr(admission, "PROTOCOL", admission.protocol_path(admission.FOUNDER_PROTOCOL))
    prior = SimpleNamespace(digest=data["prior"]["receipt_digest"], body={
        "cumulative_host_seconds": data["prior"]["receipt_cumulative_host_seconds"]})
    monkeypatch.setattr(admission, "founder_prior", lambda path, protocol: prior)
    body = {
        "protocol": data, "protocol_file": admission.FOUNDER_PROTOCOL,
        "runs": [result(arm, seed, data) for arm in ("exposed", "supplied_reads")
                 for seed in data["development_seeds"]],
        "capacity_admitted": True,
        "decision": "capacity_admitted; selection_and_causal_gates_pending",
        "candidate_rounds_used": 0, "confirmation_seeds_used": [],
        "founder_defaults": admission.founder_defaults(),
        "budget_scope": "additional_founder_capacity_stage",
        "prior_reference": prior.digest,
        "previous_host_seconds": data["prior"]["previous_host_seconds"],
        "elapsed_host_seconds": 12.0,
        "cumulative_host_seconds": data["prior"]["previous_host_seconds"] + 12.0,
    }
    return SimpleNamespace(protocol=data, prior=prior, body=body)


def test_actual_composed_founder_defaults_are_step_one_and_existing_damping_three():
    brain = Brain.compose(task.INPUTS, 8, modules=(32,), episodic=False, seed=0)
    assert brain.brain.neuron_model.dt == 1.0
    assert brain.learner.config.damping == brain.describe()["genes"]["learning_damping"] == 3
    assert admission.founder_defaults() == {
        "neuron_model": brain.brain.neuron_model.to_dict(), "learning_damping": 3}


@pytest.mark.parametrize("arm", ["exposed", "supplied_reads", *subject.ARMS])
def test_every_constructor_uses_the_same_frozen_numerical_policy(arm):
    data = protocol()
    brain = admission.make_brain(arm, 0, data)
    assert brain.brain.neuron_model.dt == 1.0
    assert brain.learner.config.damping == 3
    assert bool(getattr(brain, "empty_anchor", False)) == (arm != "exposed")
    cap = 4096 if arm == "h1_more_steps" else 2048
    assert brain.learner.config.free_steps == brain.learner.config.nudged_steps == cap


def test_new_control_changes_only_declared_brain_genes_and_step():
    old = json.loads((admission.HERE / "heads-fixed-read-protocol.json").read_text())
    new = protocol()
    assert new["task"] == old["task"] and new["gates"] == old["gates"]
    assert new["brain"] == {**old["brain"], "learning_damping": 3, "empty_anchor": True}
    assert new["dt"] == 1.0 and new["candidate_order"] == ["agreement"]
    assert new["maximum_candidate_rounds"] == 1
    assert new["budget"]["development_total_host_seconds"] == 360
    assert new["relevance_stage"]["development_total_host_seconds"] == 600
    assert new["prior"]["previous_host_seconds"] == 501.3375700454424


def test_complete_current_source_capacity_can_be_admitted(fixture):
    assert admission.check(fixture.body) is None
    assert subject.capacity_check(fixture.body) is None


@pytest.mark.parametrize("change", ["dt", "damping", "gene", "anchor", "config",
                                    "anatomy", "mode", "evaluation_mutation"])
def test_forged_numerical_metadata_or_evaluation_preservation_fails(fixture, change):
    body = deepcopy(fixture.body)
    run = body["runs"][2]
    if change == "dt":
        run["neuron_model"]["dt"] = 0.5
    elif change == "damping":
        run["numerical_policy"]["damping"] = 0
    elif change == "gene":
        run["resolved"]["genes"]["learning_damping"] = 0
    elif change == "anchor":
        run["numerical_policy"]["empty_anchor"] = False
    elif change == "config":
        run["learning_config"]["beta"] = 0.2
    elif change == "anatomy":
        run["neurons"] += 1
    elif change == "mode":
        run["mode"] = "linear"
    else:
        run["evaluation_parameters"] = "c" * 64
    assert admission.check(body) is not None


@pytest.mark.parametrize("change", ["prior", "past", "total", "scope", "negative_time",
                                    "defaults", "refusal", "partial", "scores", "founder"])
def test_prior_cost_and_actual_answers_cannot_be_replaced_by_pass_flags(fixture, change):
    body = deepcopy(fixture.body)
    if change == "prior":
        body["prior_reference"] = "f" * 64
    elif change == "past":
        body["previous_host_seconds"] -= 1
    elif change == "total":
        body["cumulative_host_seconds"] -= 1
    elif change == "scope":
        body["budget_scope"] = "same_budget"
    elif change == "negative_time":
        body["elapsed_host_seconds"] = -1
    elif change == "defaults":
        body["founder_defaults"]["learning_damping"] = 0
    elif change == "refusal":
        body["runs"][2]["training"]["work"]["action_refusals"] = 1
    elif change == "partial":
        body["runs"][2]["training"]["rounds"] -= 1
    elif change == "scores":
        body["runs"][2]["evaluation"]["train"]["answers"][0][0] = -1
    else:
        body["runs"].pop()
    assert admission.check(body) is not None


def test_new_capacity_budget_does_not_subtract_past_work(fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "sources", lambda: [])
    monkeypatch.setattr(admission.time, "monotonic", lambda: 100.0)
    deadlines = []

    def trial(arm, seed, data, deadline):
        deadlines.append(deadline)
        return result(arm, seed, data)

    monkeypatch.setattr(admission, "trial", trial)
    journal = tmp_path / "capacity.partial.json"
    receipt = admission.run(Path("prior.json"), journal=journal)
    assert deadlines == [460.0] * 4
    assert receipt.body["previous_host_seconds"] == 501.3375700454424
    assert receipt.body["cumulative_host_seconds"] == 501.3375700454424
    assert json.loads(journal.read_text())["body"] == receipt.body
    with pytest.raises(ValueError, match="requires"):
        admission.run()


def test_expired_founder_cell_keeps_complete_unanswered_panels_without_solves(monkeypatch):
    monkeypatch.setattr(admission.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(admission, "evaluate", lambda *a, **kw: pytest.fail("expired neural panel"))
    measured = admission.trial("supplied_reads", 0, protocol(), 99.0)
    assert measured["training"]["rounds"] == 0
    assert measured["training"]["stop"] == "wall_budget"
    assert not measured["acquired"] and not any(measured["work"].values())
    for name in ("train", "held_out", "all", "erased"):
        panel = task.panel("all" if name == "erased" else name)
        reading = measured["evaluation"][name]
        assert reading["status"] == "unrun_budget"
        assert reading["scores"]["unanswered_slots"] == 2 * len(panel)
        assert reading["association"] is reading["attention"] is None
        assert reading["work"] == admission.asdict(admission.sr.Work())


def test_only_a_panel_already_started_may_cross_the_phase_deadline(monkeypatch):
    now, seen = [0.0], []
    monkeypatch.setattr(admission.time, "monotonic", lambda: now[0])

    def evaluate(brain, arm, rows, condition):
        seen.append(condition)
        now[0] = 101.0
        answers = task.targets(rows)
        return {"answers": answers.tolist(), "scores": task.scores(answers, rows)}

    monkeypatch.setattr(admission, "evaluate", evaluate)
    first = admission.budgeted_evaluation(None, "exposed", task.panel("train"), 100.0)
    second = admission.budgeted_evaluation(None, "exposed", task.panel("held_out"), 100.0)
    assert first["scores"]["exact_pair_accuracy"] == 1.0
    assert second["status"] == "unrun_budget"
    assert seen == ["intact"]


@pytest.mark.parametrize("panel", ["held_out", "all"])
def test_perfect_train_cannot_hide_unrun_or_refused_intact_panels(fixture, panel):
    body = deepcopy(fixture.body)
    run = body["runs"][2]
    reading = run["evaluation"][panel]
    reading["status"] = "unrun_budget"
    assert admission.acquired(run, 0.95)  # Historical train-only gate is unchanged.
    assert admission.check(body) == "capacity gate differs"
    reading.pop("status")
    reading["work"]["action_refusals"] = 1
    assert admission.check(body) == "capacity gate differs"
    reading["work"]["action_refusals"] = 0
    reading["answers"][0][0] = -1
    reading["scores"] = task.scores(reading["answers"], task.panel(panel))
    assert admission.check(body) == "capacity gate differs"


def test_complete_held_panel_accuracy_is_not_an_acquisition_floor(fixture):
    body = deepcopy(fixture.body)
    run = body["runs"][2]
    reading = run["evaluation"]["held_out"]
    reading["answers"] = [[(a + 1) % task.VALUES, (b + 1) % task.VALUES]
                          for a, b in reading["answers"]]
    reading["scores"] = task.scores(reading["answers"], task.panel("held_out"))
    assert reading["scores"]["exact_pair_accuracy"] == 0.0
    assert admission.check(body) is None


def test_prior_failed_empty_receipt_is_checked_with_its_archived_sources():
    data = protocol()
    path = admission.HERE / "results" / data["prior"]["receipt_file"]
    receipt = admission.founder_prior(path, data)
    assert not receipt.body["passed"]
    assert admission.founder_previous(data, receipt) == 501.3375700454424


def test_all_old_source_path_inventories_remain_identical_to_retained_receipts():
    for name in ("heads-capacity-corrected.json", "heads-capacity-longer.json",
                 "heads-capacity-fixed-read.json"):
        receipt = Receipt.read(admission.HERE / "results" / name)
        active = admission.protocol_path(receipt.body.get("protocol_file", "heads-protocol.json"))
        assert [entry["path"] for entry in receipt.source["files"]] == [
            name for name, _ in admission.sources(active)]
    founder = dict(admission.sources(admission.protocol_path(admission.FOUNDER_PROTOCOL)))
    assert "benchmarks/attention/results/empty-boundary-development.json" in founder
    assert "benchmarks/attention/results/founder-capacity-development.json" not in founder


def test_positive_capacity_requires_exact_current_shared_runtime(fixture, tmp_path, monkeypatch):
    source = tmp_path / "runtime.py"
    source.write_text("runtime = 1\n")
    monkeypatch.setattr(admission, "sources", lambda: [("runtime.py", source)])
    path = tmp_path / "capacity.json"
    Receipt.build("cadence-attention-heads-admission/1", fixture.body,
                  admission.sources()).write(path)
    assert subject.verify_capacity(path).body["capacity_admitted"]
    source.write_text("runtime = 2\n")
    with pytest.raises(ValueError, match="source"):
        subject.verify_capacity(path)


@pytest.mark.parametrize("change", ["refusal", "short_training", "false_admission", "missing_cell"])
def test_no_failed_or_partial_capacity_can_open_relevance(fixture, tmp_path, monkeypatch, change):
    monkeypatch.setattr(admission, "sources", lambda: [])
    body = deepcopy(fixture.body)
    if change == "refusal":
        body["runs"][2]["training"]["work"]["lesson_refusals"] = 1
    elif change == "short_training":
        body["runs"][2]["training"]["rounds"] -= 1
    elif change == "false_admission":
        body["capacity_admitted"] = False
    else:
        body["runs"].pop()
    path = tmp_path / "capacity.json"
    Receipt.build("cadence-attention-heads-admission/1", body).write(path)
    with pytest.raises(ValueError, match="prerequisite"):
        subject.verify_capacity(path)


def test_relevance_stage_keeps_all_ten_cells_and_all_cumulative_work(
    fixture, tmp_path, monkeypatch,
):
    capacity = Receipt.build("cadence-attention-heads-admission/1", fixture.body)
    root = tmp_path
    retained = root / "results" / fixture.protocol["relevance_stage"]["capacity_receipt_file"]
    capacity.write(retained)
    monkeypatch.setattr(subject, "HERE", root)
    monkeypatch.setattr(subject, "verify_capacity", lambda path: capacity)
    monkeypatch.setattr(subject, "sources", lambda: [])
    clock = iter([0.0])
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(clock, 601.0))
    monkeypatch.setattr(subject, "trial", lambda *a, **kw: pytest.fail("expired stage ran trial"))
    body = subject.run(retained).body
    assert len(body["runs"]) == 10 and all(r["status"] == "unrun_budget" for r in body["runs"])
    assert body["capacity_host_seconds"] == fixture.body["cumulative_host_seconds"]
    assert body["cumulative_host_seconds"] == fixture.body["cumulative_host_seconds"] + 601.0
    assert body["candidate_rounds_used"] == 1 and body["confirmation_seeds_used"] == []
    assert subject.check(body) is None
    changed = deepcopy(body)
    changed["previous_host_seconds"] = 0.0
    assert subject.check(changed) == "founder prerequisite or cumulative budget differs"


def test_relevance_sources_add_capacity_without_a_capacity_manifest_cycle(fixture, monkeypatch):
    monkeypatch.setattr(admission, "sources", lambda: [])
    paths = dict(subject.sources())
    name = fixture.protocol["relevance_stage"]["capacity_receipt_file"]
    assert "benchmarks/attention/results/" + name in paths
