"""Pure stage/custody checks; no brain construction, settling or campaign."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadence.receipts import Receipt, canonical_sha256

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "query_campaign_tests", HERE / "query_founder_campaign.py")
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def training(rounds):
    ctx = subject.context()
    return {"rounds": rounds, "completed": True, "stop": None,
            "wrong_counts": [0] * rounds, "work": ctx.admission.asdict(ctx.sr.Work())}


def perfect(rows):
    ctx = subject.context()
    return ctx.relevance.reading(ctx.task.targets(rows), rows)


def measured(arm="h2", seed=0):
    ctx, profile = subject.context(), subject.point_protocol("h2")
    rows = ctx.task.panel("all")
    conditions = {name: perfect(rows) for name in (
        "intact", "erase_0", "erase_1", "neutral_sham", "reorder", "frozen_interface",
        "shuffled_interface", "head_order", "head_0")}
    conditions["donor_swap"] = perfect(ctx.task.swapped(rows))
    if arm == "h2":
        conditions.update(head_1=perfect(rows), cloned_heads=perfect(rows))
    for name in (0, 1):
        answer = ctx.task.targets(rows)
        slot = rows[:, 3] ^ name
        answer[np.arange(len(rows)), slot] = (answer[np.arange(len(rows)), slot] + 1) % 4
        conditions[f"erase_{name}"] = ctx.relevance.reading(answer, rows)
    result = {"arm": arm, "seed": seed, "mode": "agreement", "status": "measured",
              "training": training(512), "parameters_preserved": True,
              "evaluation": {name: perfect(ctx.task.panel(name))
                             for name in ("train", "held_out", "all")},
              "causal": {"conditions": conditions},
              "representation_probe": {"status": "pending_budget", "passed": False},
              "elapsed_host_seconds": 1.}
    result["metrics"] = ctx.relevance.metrics(result, profile)
    result["retention"] = {"status": "completed", "training": training(256),
        "evaluation": {name: perfect(ctx.task.panel(name)) for name in ("train", "held_out")},
        "loss": 0., "passed": True}
    result["passed"] = True
    assert subject.relevance_check(result, profile) is None
    return result


def stage_body(stage, runs):
    elapsed = sum(run.get("elapsed_host_seconds", 0.) for run in runs) + 1.
    body = {"protocol": subject.declared(), "stage": stage, "query_scale": 1, "runs": runs,
            "prerequisites": {"prior": "a" * 64}, "previous_host_seconds": 12.,
            "elapsed_host_seconds": elapsed, "cumulative_host_seconds": 12. + elapsed,
            "head_advantage_claimed": False, "confirmation_seeds_used": [],
            "candidate_points_used": 1, "stone_age_opened": False}
    body["source_revision"] = "1" * 40
    if stage == "h2":
        body.update(h2_admitted=True, resource_stage_permitted=True, comparators_opened=False)
    elif stage == "capacity":
        body["capacity_admitted"] = True
    else:
        body["comparison_measurements_complete"] = True
    return body


@pytest.fixture
def arithmetic(monkeypatch):
    # Isolate constructor/source identity checks; all behavior, work, causal,
    # retention, scope and order arithmetic below uses the real frozen helpers.
    monkeypatch.setattr(subject, "prerequisites", lambda stage: {
        "prior": SimpleNamespace(digest="a" * 64, body={"cumulative_host_seconds": 12.})})
    monkeypatch.setattr(subject, "common_check", lambda result, stage, profile, **kwargs: None)
    return monkeypatch


def test_protocol_is_one_initializer_point_with_the_declared_staged_limits():
    spec = subject.declared()
    assert spec["query_scale"] == 1 and spec["development_seeds"] == [0, 1]
    assert [(spec["stages"][name]["total_host_seconds"],
             spec["stages"][name]["training_seconds_per_call"]) for name in subject.STAGES] == [
                 (600, 240), (1800, 600), (3000, 600)]
    assert spec["stages"]["h2"]["order"] == [["h2", 0], ["h2", 1]]
    assert len(spec["stages"]["comparators"]["order"]) == 8
    base = json.loads((HERE / spec["base_protocol_file"]).read_text())
    for stage in subject.STAGES:
        point = subject.point_protocol(stage)
        point["budget"]["training_seconds_per_arm"] = base["budget"]["training_seconds_per_arm"]
        assert point == base


def test_h2_requires_both_founders_and_preserves_causal_retention_gates(arithmetic):
    runs = [measured(seed=seed) for seed in (0, 1)]
    assert subject.check(stage_body("h2", runs)) is None


@pytest.mark.parametrize("change", ["missing", "duplicate", "order", "held", "donor",
                                    "retention", "refusal", "unanswered", "gate",
                                    "advantage", "stone_age", "confirmation", "scope"])
def test_h2_cannot_open_comparators_from_incomplete_or_optimistic_readings(arithmetic, change):
    body = stage_body("h2", [measured(seed=seed) for seed in (0, 1)])
    run = body["runs"][1]
    if change == "missing":
        body["runs"].pop()
    elif change == "duplicate":
        body["runs"][1] = deepcopy(body["runs"][0])
    elif change == "order":
        body["runs"].reverse()
    elif change in ("held", "unanswered"):
        reading = run["evaluation"]["held_out"]
        reading["answers"][0] = [-1, -1] if change == "unanswered" else [3, 3]
        reading["scores"] = subject.context().task.scores(
            reading["answers"], subject.context().task.panel("held_out"))
    elif change == "donor":
        reading = run["causal"]["conditions"]["donor_swap"]
        reading["status"] = "unrun_budget"
    elif change == "retention":
        run["retention"]["training"]["completed"] = False
    elif change == "refusal":
        run["training"]["work"]["lesson_refusals"] = 1
    elif change == "gate":
        run["passed"] = False
    elif change == "advantage":
        body["head_advantage_claimed"] = True
    elif change == "stone_age":
        body["stone_age_opened"] = True
    elif change == "confirmation":
        body["confirmation_seeds_used"] = [200]
    else:
        body["comparators_opened"] = True
    assert subject.check(body) is not None


def test_valid_negative_h2_closes_the_resource_stage(arithmetic):
    body = stage_body("h2", [measured(seed=seed) for seed in (0, 1)])
    run = body["runs"][0]
    run["retention"] = {"status": "not_run_budget", "passed": False}
    run["passed"] = False
    body.update(h2_admitted=False, resource_stage_permitted=False)
    assert subject.check(body) is None


def test_missing_or_capped_comparators_never_count_as_complete(arithmetic):
    runs = [measured(arm, seed)
            for arm, seed in subject.declared()["stages"]["comparators"]["order"]]
    body = stage_body("comparators", runs)
    assert subject.check(body) is None
    runs[0]["training"].update(completed=False, rounds=511, wrong_counts=[0] * 511,
                               stop="wall_budget")
    runs[0]["metrics"] = subject.context().relevance.metrics(runs[0], subject.point_protocol("h2"))
    runs[0]["passed"] = False
    assert subject.check(body) == "comparator completeness differs"
    body["comparison_measurements_complete"] = False
    assert subject.check(body) is None


def test_old_capacity_kind_cannot_authorize_query_one(tmp_path):
    old = Receipt.build("cadence-attention-heads-admission/1", {"capacity_admitted": True})
    path = tmp_path / "old.json"
    old.write(path)
    assert subject.verify(path) == (False, "not a matched query-founder stage receipt")


def test_stage_receipt_rejects_changed_initializer_bytes(tmp_path, monkeypatch):
    initializer = tmp_path / "initializer.py"
    initializer.write_text("query_scale = 1\n")
    monkeypatch.setattr(subject, "sources", lambda stage: [("initializer.py", initializer)])
    monkeypatch.setattr(subject, "check", lambda body: None)
    receipt = Receipt.build(subject.declared()["stages"]["capacity"]["receipt_kind"],
                            {"stage": "capacity"}, subject.sources("capacity"))
    path = tmp_path / "capacity.json"
    receipt.write(path)
    assert subject.verify(path)[0]
    initializer.write_text("query_scale = 0\n")
    assert not subject.verify(path)[0]


def fake_context():
    calls = []

    def make(arm, seed, profile, mode="agreement"):
        calls.append(("base", arm, seed))
        return SimpleNamespace(arm=arm, seed=seed)

    def initialize(brain, *, seed, query_scale):
        calls.append(("initialize", brain.arm, seed, query_scale))
        brain.query_initialization = {"query_scale": query_scale, "seed": seed}

    old_snapshot = object()
    ctx = SimpleNamespace(admission=SimpleNamespace(make_brain=make,
                                                    metered=lambda work: nullcontext()),
        initializer=SimpleNamespace(initialize=initialize),
        relevance=SimpleNamespace(parameter_snapshot=old_snapshot),
        sr=SimpleNamespace(revision=lambda: "freeze"))
    return ctx, calls, make, old_snapshot


def test_factory_scope_has_explicit_namespace_and_restores_after_exception(monkeypatch):
    ctx, calls, original, old_snapshot = fake_context()
    monkeypatch.setattr(subject, "context", lambda: ctx)
    captures = {}
    with pytest.raises(RuntimeError, match="scoped failure"), subject.factory_scope("h2", captures):
        brain = ctx.admission.make_brain("h2", 0, {})
        assert captures[("h2", 0)] is brain
        assert calls == [("base", "h2", 0), ("initialize", "h2", 0, 1)]
        raise RuntimeError("scoped failure")
    assert ctx.admission.make_brain is original
    assert ctx.relevance.parameter_snapshot is old_snapshot


def test_both_h2_cells_are_attempted_after_baseline_custody_before_any_injection(
    tmp_path, monkeypatch,
):
    ctx, calls, original, old_snapshot = fake_context()

    def prior(stage):
        assert ctx.admission.make_brain is original
        calls.append(("baseline_verified", stage))
        return {"prior": SimpleNamespace(digest="a" * 64,
                                          body={"cumulative_host_seconds": 12.})}

    def trial(arm, seed, mode, profile, deadline, *, parameters):
        ctx.admission.make_brain(arm, seed, profile, mode)
        calls.append(("trial", seed))
        return {"arm": arm, "seed": seed, "mode": mode, "passed": False}

    ctx.relevance.trial = trial
    monkeypatch.setattr(subject, "context", lambda: ctx)
    monkeypatch.setattr(subject, "prerequisites", prior)
    monkeypatch.setattr(subject, "sources", lambda stage: [])
    monkeypatch.setattr(subject, "check", lambda body: None)
    receipt = subject.run("h2", tmp_path / "h2.json", tmp_path / "parameters", "freeze")
    assert calls[0] == ("baseline_verified", "h2")
    assert [call for call in calls if call[0] == "trial"] == [("trial", 0), ("trial", 1)]
    assert not receipt.body["h2_admitted"] and not receipt.body["resource_stage_permitted"]
    assert not receipt.body["comparators_opened"] and not receipt.body["head_advantage_claimed"]
    assert ctx.admission.make_brain is original and ctx.relevance.parameter_snapshot is old_snapshot


def test_research_snapshot_metadata_binds_the_new_full_source_closure(tmp_path, monkeypatch):
    paths = [("runner.py", HERE / "query_founder_campaign.py"),
             ("initializer.py", HERE / "query_founder.py"),
             ("protocol.json", subject.PROTOCOL)]
    ctx, _, _, _ = fake_context()
    ctx.sr.parameter_digest = lambda brain: "b" * 64
    monkeypatch.setattr(subject, "context", lambda: ctx)
    monkeypatch.setattr(subject, "sources", lambda stage: paths)
    learner = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {"beta": .1}),
                              updates=2, contrast_updates=2,
                              **{name: np.zeros(2) for name in subject.MOMENTS})
    brain = SimpleNamespace(brain=SimpleNamespace(**{name: np.zeros(2)
                                                    for name in subject.PARAMETERS}),
                            learner=learner, query_initialization={"query_scale": 1})
    record = subject.snapshot(brain, tmp_path, stage="h2", arm="h2", seed=0, mode="agreement")
    with np.load(tmp_path / record["file"], allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
    assert [row["path"] for row in metadata["source"]["files"]] == [name for name, _ in paths]
    assert metadata["query_founder"] == {"query_scale": 1}
    assert metadata["source"] == record["metadata"]["source"]
    assert canonical_sha256(metadata["source"]["files"]) == metadata["source"]["manifest_sha256"]


def test_capacity_controller_to_typed_receipt_uses_real_q1_constructors_without_solving(
    tmp_path, monkeypatch,
):
    """Synthetic trial answers isolate controller plumbing, not scientific acquisition."""
    ctx = subject.context()
    # Source archives need not contain Git. This synthetic instrument test
    # supplies its own revision while production execution keeps its guard.
    monkeypatch.setattr(ctx.sr, "revision", lambda: "a" * 40)
    original_factory = ctx.admission.make_brain
    original_snapshot = ctx.relevance.parameter_snapshot

    def no_phase(*args, **kwargs):
        pytest.fail("constructor-only controller smoke attempted a neural phase")

    monkeypatch.setattr(ctx.admission.heads.Graph, "equilibrate", no_phase)

    def synthetic_trial(arm, seed, profile, deadline, mode="agreement"):
        brain = ctx.admission.make_brain(arm, seed, profile, mode)
        assert brain.query_initialization["query_scale"] == 1
        if arm == "supplied_reads":
            assert brain.query_initialization["applied"]

        def panel(name):
            rows = ctx.task.panel(name)
            answer = ctx.task.targets(rows)
            # Capacity must report held-out failure without adding a hidden
            # held-out accuracy floor to the original trained-content gate.
            mask = np.array([tuple(row[:2]) in ctx.task.HELD_PAIRS for row in rows])
            answer[mask] = (answer[mask] + 1) % 4
            return ctx.relevance.reading(answer, rows)

        initial = ctx.sr.parameter_digest(brain)
        schedule = ctx.task.draw(seed, 512, 16)
        return {"arm": arm, "seed": seed, "mode": mode,
            "training": training(512), "initial": panel("all"),
            "evaluation": {name: panel(name) for name in ("train", "held_out", "all")},
            "initial_parameters": initial, "final_parameters": initial,
            "evaluation_parameters": initial, "parameters_preserved": True,
            "resolved": brain.describe(), "neuron_model": brain.brain.neuron_model.to_dict(),
            "numerical_policy": ctx.admission.numerical_settings(brain),
            "learning_config": brain.learner.config.to_dict(),
            "neurons": brain.connectome.n, "synapses": brain.connectome.synapses,
            "schedule_sha256": canonical_sha256(schedule),
            "work": {name: 0 for name in ("cortical_synapse_transports", "memory_scan_macs",
                "softmax_elements", "memory_drive_evaluations", "fixed_read_coordinate_updates")},
            "elapsed_host_seconds": 0., "acquired": True}

    monkeypatch.setattr(ctx.admission, "trial", synthetic_trial)
    path = tmp_path / "capacity.json"
    receipt = subject.run("capacity", path, tmp_path / "parameters", ctx.sr.revision())
    receipt.write(path)
    valid, message = subject.verify(path)
    assert valid, message
    assert receipt.body["capacity_admitted"]
    assert len(receipt.body["runs"]) == 4
    for result in receipt.body["runs"]:
        assert result["evaluation"]["held_out"]["scores"]["exact_pair_accuracy"] == 0.
        metadata = result["parameter_snapshot"]["metadata"]
        assert metadata["query_founder"] == result["query_initialization"]
        paths = {row["path"] for row in metadata["source"]["files"]}
        assert {"benchmarks/attention/query_founder_campaign.py",
                "benchmarks/attention/query_founder.py",
                "benchmarks/attention/query-founder-protocol.json"} <= paths
    assert ctx.admission.make_brain is original_factory
    assert ctx.relevance.parameter_snapshot is original_snapshot
    for damage in ("revision", "mode", "counters"):
        corrupt = deepcopy(receipt.body)
        metadata = corrupt["runs"][0]["parameter_snapshot"]["metadata"]
        if damage == "revision":
            metadata["source_revision"] = "f" * 40
        elif damage == "mode":
            metadata["mode"] = "linear"
        else:
            metadata["contrast_updates"] = metadata["updates"] + 1
        changed = tmp_path / f"corrupt-{damage}.json"
        Receipt.build(receipt.kind, corrupt, subject.sources("capacity")).write(changed)
        assert not subject.verify(changed)[0]
    # The actual legacy gate is evaluated after the injected namespace is gone.
    old = ctx.relevance.verify_capacity(HERE / "results/founder-capacity-extended.json")
    assert old.kind == "cadence-attention-heads-admission/1" and old.body["capacity_admitted"]


def test_unexpected_trial_failure_keeps_current_arrays_work_and_restores_hooks(
    tmp_path, monkeypatch,
):
    ctx, calls, original, old_snapshot = fake_context()
    original_meter = ctx.admission.metered

    def trial(arm, seed, profile, deadline, mode="agreement"):
        ctx.admission.make_brain(arm, seed, profile, mode)
        with ctx.admission.metered({"cortical_synapse_transports": 0}) as _:
            work = {"cortical_synapse_transports": 17}
            with ctx.admission.metered(work):
                raise RuntimeError("unexpected fixture failure")

    monkeypatch.setattr(subject, "context", lambda: ctx)
    monkeypatch.setattr(subject, "prerequisites", lambda stage: {
        "prior": SimpleNamespace(digest="a" * 64, body={"cumulative_host_seconds": 12.})})
    monkeypatch.setattr(subject, "sources", lambda stage: [])
    captured = []

    def snapshot(brain, directory, **kwargs):
        captured.append((brain.arm, brain.seed, Path(directory).name))
        return {"file": "fixture-interrupted.npz"}

    monkeypatch.setattr(subject, "snapshot", snapshot)
    ctx.admission.trial = trial
    path = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="fixture failure"):
        subject.run("capacity", path, tmp_path / "parameters", "freeze")
    journal = json.loads(path.with_suffix(".partial.json").read_text())
    assert journal["status"] == "cell_failed_unverified" and not path.exists()
    assert journal["actual_transport_work"] == {"cortical_synapse_transports": 17}
    assert journal["interrupted_parameters"]["file"] == "fixture-interrupted.npz"
    assert captured == [("exposed", 0, "interrupted")]
    assert ctx.admission.make_brain is original
    assert ctx.admission.metered is original_meter
    assert ctx.relevance.parameter_snapshot is old_snapshot


def test_execution_revision_change_retains_unverified_work_and_refuses_receipt(
    tmp_path, monkeypatch,
):
    ctx, _, original, old_snapshot = fake_context()
    revisions = iter(("freeze", "changed"))
    ctx.sr.revision = lambda: next(revisions)

    def trial(arm, seed, mode, profile, deadline, *, parameters):
        ctx.admission.make_brain(arm, seed, profile, mode)
        return {"arm": arm, "seed": seed, "mode": mode, "passed": False}

    ctx.relevance.trial = trial
    monkeypatch.setattr(subject, "context", lambda: ctx)
    monkeypatch.setattr(subject, "sources", lambda stage: [])
    monkeypatch.setattr(subject, "prerequisites", lambda stage: {
        "prior": SimpleNamespace(digest="a" * 64, body={"cumulative_host_seconds": 12.})})
    path = tmp_path / "changed.json"
    with pytest.raises(ValueError, match="execution commit changed"):
        subject.run("h2", path, tmp_path / "parameters", "freeze")
    journal = json.loads(path.with_suffix(".partial.json").read_text())
    assert journal["status"] == "revision_changed_unverified" and len(journal["runs"]) == 2
    assert ctx.admission.make_brain is original
    assert ctx.relevance.parameter_snapshot is old_snapshot
