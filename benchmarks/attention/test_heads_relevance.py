"""Independent prerequisites, witnessed interventions, gates and budget custody."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadence.receipts import Receipt, canonical_sha256

SPEC = importlib.util.spec_from_file_location(
    "heads_relevance_tests", Path(__file__).with_name("heads_relevance.py")
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)
task = subject.task


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    declared = json.loads(subject.admission.PROTOCOL.read_text())
    declared["task"]["rounds"] = 3
    declared["gates"]["retention_rounds"] = 2
    path = tmp_path / "heads-protocol.json"
    path.write_text(json.dumps(declared))
    monkeypatch.setattr(subject.admission, "PROTOCOL", path)
    original_selector = subject.admission.protocol_path

    def selected(name):
        original_selector(name)  # Keep the production whitelist in the fixture.
        return tmp_path / name

    for selector in subject.admission.PROTOCOL_NAMES[1:]:
        variant = json.loads(original_selector(selector).read_text())
        variant["task"]["rounds"], variant["gates"]["retention_rounds"] = 3, 2
        (tmp_path / selector).write_text(json.dumps(variant))
    monkeypatch.setattr(subject.admission, "protocol_path", selected)
    return declared


def training(rounds):
    return {"completed": True, "rounds": rounds, "wrong_counts": [0] * rounds,
            "work": {"action_refusals": 0, "lesson_refusals": 0}}


def perfect(rows):
    return subject.reading(task.targets(rows), rows)


def capacity_body(protocol):
    runs = []
    for arm in ("exposed", "supplied_reads"):
        for seed in protocol["development_seeds"]:
            runs.append({
                "arm": arm, "seed": seed, "training": training(protocol["task"]["rounds"]),
                "initial": perfect(task.panel("all")),
                "evaluation": {name: perfect(task.panel(name))
                               for name in ("train", "held_out", "all")},
                "acquired": True, "parameters_preserved": True,
                "schedule_sha256": canonical_sha256(task.draw(
                    seed, protocol["task"]["rounds"], protocol["task"]["batch"])),
            })
    return {"protocol": protocol, "runs": runs, "capacity_admitted": True,
            "candidate_rounds_used": 0, "confirmation_seeds_used": [],
            "decision": "capacity_admitted; selection_and_causal_gates_pending",
            "elapsed_host_seconds": 1.0}


def result(protocol, *, arm="h1", seed=0, mode="agreement"):
    rows = task.panel("all")
    conditions = {name: perfect(rows) for name in (
        "intact", "erase_0", "erase_1", "neutral_sham", "reorder",
        "frozen_interface", "shuffled_interface", "head_order", "head_0")}
    conditions["donor_swap"] = perfect(task.swapped(rows))
    if arm == "h2":
        conditions.update(head_1=perfect(rows), cloned_heads=perfect(rows))
    for name in range(2):
        answers = task.targets(rows)
        slot = rows[:, 3] ^ name
        chosen = np.arange(len(rows))
        answers[chosen, slot] = (answers[chosen, slot] + 1) % task.VALUES
        conditions[f"erase_{name}"] = subject.reading(answers, rows)
    measured = {
        "arm": arm, "seed": seed, "mode": mode,
        "training": training(protocol["task"]["rounds"]), "parameters_preserved": True,
        "evaluation": {name: perfect(task.panel(name)) for name in ("train", "held_out", "all")},
        "causal": {"conditions": conditions},
        "schedule_sha256": canonical_sha256(task.draw(
            seed, protocol["task"]["rounds"], protocol["task"]["batch"])),
        "retention": {"status": "not_run_budget", "passed": False}, "passed": False,
        "representation_probe": {"status": "pending_behavioral_gates", "passed": False},
    }
    measured["metrics"] = subject.metrics(measured, protocol)
    return measured


def body(protocol):
    return {"protocol": protocol, "runs": [
        result(protocol, arm=arm, seed=seed, mode=mode)
        for mode in protocol["candidate_order"] for arm in subject.ARMS
        for seed in protocol["development_seeds"]
    ], "selected": None, "passed": False, "candidate_rounds_used": 2,
        "confirmation_seeds_used": []}


def bank(rows, copies=1):
    prefix, _ = task.events(rows)
    values = np.stack(prefix, axis=1)
    values = np.tile(values, (1, 1, copies)) / np.sqrt(copies)
    width = values.shape[2]
    first = subject.heads.Memory(
        np.arange(width), np.arange(width, 2 * width), values, values,
        np.ones(values.shape[:2], dtype=bool), np.array([2.0, 1.0, 0.0]), amplitude=1.5,
    )
    from dataclasses import replace

    return subject.heads.MultiMemory((first, replace(
        first, query=np.arange(2 * width, 3 * width), read=np.arange(3 * width, 4 * width),
    )))


def test_capacity_requires_both_independently_passing_controls(protocol):
    measured = capacity_body(protocol)
    assert subject.capacity_check(measured) is None
    for change in ("missing", "duplicate", "optimism", "unrun", "rounds"):
        altered = deepcopy(measured)
        if change == "missing":
            altered["runs"].pop()
        elif change == "duplicate":
            altered["runs"][-1] = deepcopy(altered["runs"][-2])
        elif change == "optimism":
            altered["runs"][-1]["evaluation"]["train"]["scores"]["exact_pair_accuracy"] = 0.0
        elif change == "unrun":
            altered["runs"][-1]["evaluation"]["train"]["answers"][0] = [-1, -1]
        else:
            altered["runs"][-1]["training"]["rounds"] = 2
        assert subject.capacity_check(altered) is not None


def test_capacity_receipt_binds_sources_before_selection(protocol, tmp_path, monkeypatch):
    declared = subject.admission.PROTOCOL
    monkeypatch.setattr(subject.admission, "sources", lambda: [("heads-protocol.json", declared)])
    path = tmp_path / "capacity.json"
    Receipt.build("cadence-attention-heads-admission/1", capacity_body(protocol),
                  subject.admission.sources()).write(path)
    assert subject.verify_capacity(path).body["capacity_admitted"]
    declared.write_text(declared.read_text() + "\n")
    with pytest.raises(ValueError, match="source manifest"):
        subject.verify_capacity(path)


@pytest.mark.parametrize("selector", [
    "heads-longer-protocol.json", "heads-fixed-read-protocol.json",
])
def test_capacity_selects_its_whitelisted_variant_before_protocol_and_source_checks(
    protocol, tmp_path, monkeypatch, selector,
):
    base = subject.admission.PROTOCOL
    longer = subject.admission.protocol_path(selector)
    data = json.loads(longer.read_text())
    measured = capacity_body(data)
    measured["protocol_file"] = longer.name

    def selected_sources():
        active = subject.admission.PROTOCOL
        paths = [("benchmarks/attention/" + active.name, active),
                 ("benchmarks/attention/heads-protocol.json", base)]
        if active.name == "heads-fixed-read-protocol.json":
            paths += [("benchmarks/attention/heads-longer-protocol.json",
                       subject.admission.protocol_path("heads-longer-protocol.json")),
                      ("benchmarks/attention/fixed_read.py", subject.HERE / "fixed_read.py")]
        return paths

    monkeypatch.setattr(subject.admission, "sources", selected_sources)
    subject.admission.PROTOCOL = longer
    path = tmp_path / "capacity-variant.json"
    Receipt.build("cadence-attention-heads-admission/1", measured, selected_sources()).write(path)
    subject.admission.PROTOCOL = base
    assert subject.verify_capacity(path).body["protocol"] == data
    assert subject.admission.PROTOCOL == longer
    measured["protocol_file"] = "unfrozen.json"
    Receipt.build("cadence-attention-heads-admission/1", measured, selected_sources()).write(path)
    with pytest.raises(ValueError, match="declared"):
        subject.verify_capacity(path)


def test_archived_capacity_runner_can_only_differ_by_ast_identical_formatting(
    protocol, tmp_path, monkeypatch,
):
    current = tmp_path / "heads_admission.py"
    current.write_text("x=1\n")
    monkeypatch.setattr(subject.admission, "sources", lambda: [
        ("benchmarks/attention/heads_admission.py", current)])
    path = tmp_path / "capacity.json"
    Receipt.build("cadence-attention-heads-admission/1", capacity_body(protocol),
                  subject.admission.sources()).write(path)
    digest = hashlib.sha256(current.read_bytes()).hexdigest()
    snapshot = tmp_path / "results" / "sources" / (digest + ".txt")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(current.read_bytes())
    monkeypatch.setattr(subject, "HERE", tmp_path)
    current.write_text("x = 1\n")
    assert subject.verify_capacity(path).body["capacity_admitted"]
    current.write_text("x = 2\n")
    with pytest.raises(ValueError, match="source manifest"):
        subject.verify_capacity(path)


@pytest.mark.parametrize("copies", [1, 2])
def test_witnessed_bank_erasures_and_donor_keep_query_independent_information(copies):
    rows = task.CASES[[5, 18, 39, 52]]
    original = bank(rows, copies)
    before = original.values.copy()
    for name in range(2):
        erased = subject.bank_change(original, f"erase_{name}")
        named = original.values[:, :, 5 + name] > 0
        np.testing.assert_array_equal(erased.held, original.held & ~named)
        np.testing.assert_array_equal(erased.values, before)
        assert erased.memories[0].held is erased.memories[1].held
    brain = SimpleNamespace(port_copies=copies, recall=SimpleNamespace(heard=1.0))
    transplanted = subject.bank_change(original, "donor_swap", brain, rows)
    donor_prefix, host_query = task.events(task.swapped(rows))
    np.testing.assert_array_equal(host_query, task.events(rows)[1])
    expected = np.tile(np.stack(donor_prefix, axis=1), (1, 1, copies)) / np.sqrt(copies)
    np.testing.assert_array_equal(transplanted.values, expected)
    np.testing.assert_array_equal(transplanted.keys, expected)
    np.testing.assert_array_equal(original.values, before)


def test_reordering_keeps_ages_and_records_together_and_neutral_sham_touches_no_fact():
    original = bank(task.CASES[[5, 18]])
    reordered = subject.bank_change(original, "reorder")
    np.testing.assert_array_equal(reordered.values, original.values[:, ::-1])
    np.testing.assert_array_equal(reordered.keys, original.keys[:, ::-1])
    np.testing.assert_array_equal(reordered.ages, original.ages[::-1])
    sham = subject.bank_change(original, "neutral_sham")
    np.testing.assert_array_equal(sham.values[:, :2], original.values[:, :2])
    np.testing.assert_array_equal(sham.values[:, 2], original.values[:, 2] * 0.5)


def test_every_query_intervention_forks_the_same_drive_and_cold_cortex(monkeypatch):
    brain = subject.heads.MultiHeadBrain.compose(
        task.INPUTS, 8, slots=2, heads=2, modules=(4,), episodic=False,
        recall_next=False, learning_free_steps=512, learning_nudged_steps=512,
    )
    rows = task.CASES[[5, 18]]
    recorded = []

    def refuse(twin, query, drive, work):
        assert twin.basal_ganglia.state is None
        assert not twin.working_memory.trace.any()
        recorded.append((query.copy(), drive.copy()))
        work.action_refusals += 1
        return None

    monkeypatch.setattr(subject.admission, "act_drive", refuse)
    before = subject.sr.parameter_digest(brain)
    measured = subject.evaluate_conditions(brain, "h2", rows, brain.brain.efficacy.copy(),
                                           float("inf"))
    assert len(recorded) == len(measured["conditions"]) == 12
    for query, drive in recorded[1:]:
        np.testing.assert_array_equal(query, recorded[0][0])
        np.testing.assert_array_equal(drive, recorded[0][1])
    assert subject.sr.parameter_digest(brain) == before
    assert all(r["scores"]["unanswered_slots"] == 4 for r in measured["conditions"].values())


def test_primary_gate_is_recomputed_from_causal_slots_and_actual_donor_values(protocol):
    measured = result(protocol)
    computed = subject.metrics(measured, protocol)
    assert computed["primary_passed"] and computed["record_erasure_slot_loss"] == [1.0, 1.0]
    assert computed["record_erasure_other_slot_loss"] == [0.0, 0.0]
    assert computed["donor_pair_follow"] == 1.0 and computed["donor_unequal_cases"] == 48
    assert not measured["passed"]  # Retention remains a separate, unrun gate.
    measured["causal"]["conditions"]["donor_swap"] = perfect(task.panel("all"))
    assert not subject.metrics(measured, protocol)["primary_passed"]


def test_passing_aggregate_swaps_cannot_hide_failed_held_out_swaps(protocol):
    measured = result(protocol)
    donor = measured["causal"]["conditions"]["donor_swap"]
    answers = np.asarray(donor["answers"])
    failed = np.flatnonzero(task.HELD)[:4]
    answers[failed, 0] = (answers[failed, 0] + 1) % task.VALUES
    measured["causal"]["conditions"]["donor_swap"] = subject.reading(answers, donor["rows"])
    computed = subject.metrics(measured, protocol)
    assert computed["swapped_unequal_exact_pair_accuracy"] == pytest.approx(44 / 48)
    assert computed["swapped_held_out_exact_pair_accuracy"] == 0.75
    assert not computed["primary_passed"]


@pytest.mark.parametrize("panel, field", [
    ("train", "status"), ("train", "work"), ("erase_0", "status"), ("erase_0", "work"),
])
def test_refused_status_or_work_cannot_be_hidden_by_complete_looking_answers(
    protocol, panel, field,
):
    measured = result(protocol)
    reading = (measured["evaluation"][panel] if panel == "train"
               else measured["causal"]["conditions"][panel])
    if field == "status":
        reading["status"] = "refused"
    else:
        reading["work"]["action_refusals"] = 1
    assert not subject.metrics(measured, protocol)["primary_passed"]


@pytest.mark.parametrize("change", ["unrun", "no_erasure_effect", "sham", "training", "refusal"])
def test_optimistic_or_missing_causal_evidence_cannot_pass(protocol, change):
    measured = result(protocol)
    if change == "unrun":
        measured["causal"]["conditions"]["erase_0"]["answers"] = [[-1, -1]] * 64
    elif change == "no_erasure_effect":
        measured["causal"]["conditions"]["erase_0"] = perfect(task.panel("all"))
    elif change == "sham":
        measured["causal"]["conditions"]["neutral_sham"]["answers"] = [[0, 0]] * 64
    elif change == "training":
        measured["training"]["rounds"] = 2
    else:
        measured["training"]["work"]["lesson_refusals"] = 1
    assert not subject.metrics(measured, protocol)["primary_passed"]


def test_verifier_preserves_all_arms_and_rejects_optimism(protocol):
    measured = body(protocol)
    assert subject.check(measured) is None
    for change in ("missing", "duplicate", "score", "gate", "condition", "schedule"):
        altered = deepcopy(measured)
        if change == "missing":
            altered["runs"].pop()
        elif change == "duplicate":
            altered["runs"][1] = deepcopy(altered["runs"][0])
        elif change == "score":
            altered["runs"][0]["evaluation"]["held_out"]["scores"]["exact_pair_accuracy"] = 0.0
        elif change == "gate":
            altered["runs"][0]["passed"] = True
        elif change == "condition":
            del altered["runs"][0]["causal"]["conditions"]["erase_0"]
        else:
            altered["runs"][0]["schedule_sha256"] = "f" * 64
        assert subject.check(altered) is not None


def test_positive_selection_requires_retention_of_both_founders_and_stops_equation_search(protocol):
    measured = body(protocol)
    measured["runs"] = measured["runs"][:10]
    for run in measured["runs"][:2]:
        run["retention"] = {
            "status": "completed", "training": training(protocol["gates"]["retention_rounds"]),
            "evaluation": {name: perfect(task.panel(name)) for name in ("train", "held_out")},
            "loss": 0.0, "passed": True,
        }
        run["passed"] = True
    measured["selected"] = {"mode": "agreement", "arm": "h1"}
    measured["passed"], measured["candidate_rounds_used"] = True, 1
    assert subject.check(measured) is None
    measured["runs"][1]["retention"]["training"]["rounds"] = 1
    assert subject.check(measured) is not None


def probe_result(protocol):
    measured = result(protocol)
    measured["initial"] = perfect(task.panel("all"))
    measured["initial"]["association"] = np.zeros((64, 8)).tolist()
    for reading in measured["evaluation"].values():
        reading["association"] = np.eye(4)[task.targets(reading["rows"])].reshape(-1, 8).tolist()
    for name, reading in measured["causal"]["conditions"].items():
        reading["association"] = (
            np.zeros((64, 8)).tolist() if name.startswith("erase_")
            else np.eye(4)[task.targets(reading["rows"])].reshape(-1, 8).tolist()
        )
    measured["passed"] = True
    return measured


def test_representation_uses_only_recorded_arrays_after_behavior_and_retention(
    protocol, monkeypatch,
):
    measured = probe_result(protocol)
    monkeypatch.setattr(subject.admission, "make_brain", lambda *args: pytest.fail("neural work"))
    probe = subject.representation_probe(measured, float("inf"))
    assert probe["status"] == "completed" and probe["passed"]
    assert probe["result"]["learned"]["held_out"]["scores"]["exact_pair_accuracy"] == 1.0
    assert probe["result"]["controls"]["raw_query"]["held_out"]["scores"][
        "exact_pair_accuracy"] < 0.5
    measured["passed"] = False
    pending = subject.representation_probe(measured, float("inf"))
    assert pending["status"] == "pending_behavioral_gates"
    measured["passed"] = True
    assert subject.representation_probe(measured, -float("inf"))["status"] == "pending_budget"


def test_completed_representation_is_recomputed_from_stored_association_arrays(protocol):
    measured = body(protocol)
    measured["runs"] = measured["runs"][:10]
    for index in range(2):
        run = probe_result(protocol)
        run["seed"] = index
        run["schedule_sha256"] = canonical_sha256(task.draw(
            index, protocol["task"]["rounds"], protocol["task"]["batch"]))
        run["retention"] = {
            "status": "completed", "training": training(protocol["gates"]["retention_rounds"]),
            "evaluation": {name: perfect(task.panel(name)) for name in ("train", "held_out")},
            "loss": 0.0, "passed": True,
        }
        run["representation_probe"] = subject.representation_probe(run, float("inf"))
        measured["runs"][index] = run
    measured["selected"] = {"mode": "agreement", "arm": "h1"}
    measured["passed"], measured["candidate_rounds_used"] = True, 1
    assert subject.check(measured) is None
    probe = measured["runs"][0]["representation_probe"]["result"]["learned"]["probe"]
    probe["weights"][0][0][0] += 1
    assert subject.check(measured) == "representation diagnostic differs"


def test_parameter_snapshots_are_path_independent_nonoverwriting_research_arrays(tmp_path):
    graph = SimpleNamespace(efficacy=np.array([0.2, -0.3]), bias=np.array([0.1, 0.0]),
                            log_gain=np.zeros(2))
    learner = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {"beta": 0.1}),
                              velocity=np.array([0.01, 0.02]), updates=3, contrast_updates=3)
    brain = SimpleNamespace(brain=graph, learner=learner)
    assert subject.parameter_snapshot(brain, None, arm="h1", seed=0, mode="agreement") is None
    first = subject.parameter_snapshot(
        brain, tmp_path / "first", arm="h1", seed=0, mode="agreement")
    other = subject.parameter_snapshot(
        brain, tmp_path / "other", arm="h1", seed=0, mode="agreement")
    assert first["content_sha256"] == other["content_sha256"]
    assert str(tmp_path) not in json.dumps(first)
    with np.load(tmp_path / "first" / first["file"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["efficacy"], graph.efficacy)
        np.testing.assert_array_equal(archive["optimizer_velocity"], learner.velocity)
        metadata = json.loads(str(archive["metadata"]))
        assert metadata["updates"] == 3
        assert "not a supported continuing Brain checkpoint" in metadata["scope"]
        content = canonical_sha256({name: archive[name].tolist() for name in archive.files})
        assert content == first["content_sha256"]
    with pytest.raises(FileExistsError):
        subject.parameter_snapshot(brain, tmp_path / "first", arm="h1", seed=0, mode="agreement")


def test_retention_balances_existing_pairs_and_alternates_query_order(protocol):
    protocol["gates"]["retention_rounds"] = 6
    schedule = subject.retention_schedule(0, protocol)
    assert schedule.shape == (6, 16, 4)
    for tick, rows in enumerate(schedule):
        assert (rows[:, 3] == tick % 2).all()
        assert all(tuple(row[:2]) not in task.HELD_PAIRS for row in rows)
    _, counts = np.unique(schedule[:, :, :2].reshape(-1, 2), axis=0, return_counts=True)
    assert counts.tolist() == [8] * 12


def test_exhausted_shared_budget_retains_all_unrun_cells_without_starting_training(
    protocol, monkeypatch,
):
    capacity = SimpleNamespace(body={"elapsed_host_seconds": 500.0, "previous_host_seconds": 101.0},
                               digest="d" * 64)
    monkeypatch.setattr(subject, "verify_capacity", lambda path: capacity)
    monkeypatch.setattr(subject, "trial", lambda *args: pytest.fail("expired budget started trial"))
    receipt = subject.run(Path("unused"))
    assert not receipt.body["passed"] and receipt.body["selected"] is None
    assert len(receipt.body["runs"]) == 20
    assert all(run["status"] == "unrun_budget" for run in receipt.body["runs"])
    assert not receipt.body["confirmation_seeds_used"]
    assert subject.check(receipt.body) is None
