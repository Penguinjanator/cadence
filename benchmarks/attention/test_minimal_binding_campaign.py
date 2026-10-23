"""Array and constructor/instrument checks; every neural solve is forbidden."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "minimal_campaign_tests", HERE / "minimal_binding_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)
RUNTIME = Path(os.getenv("CADENCE_ATTENTION_RUNTIME", str(HERE.parents[1])))


@pytest.fixture
def ctx(monkeypatch):
    result = subject.configure(RUNTIME)
    monkeypatch.setattr(result.sr, "revision", lambda: "1" * 40)  # Archive-safe fixture identity.

    def forbidden(*args, **kwargs):
        raise AssertionError("a neural phase was attempted by a pure instrument test")

    monkeypatch.setattr(result.admission.heads.Graph, "equilibrate", forbidden)
    return result


@pytest.fixture
def archived_prior(ctx, monkeypatch):
    """Isolate the retained prerequisite from this controller's current-source checks."""
    verifier = subject.load("minimal_tests_historical_verifier", HERE / "verify.py")
    expected = (ctx.runtime / "benchmarks/attention/results"
                / subject.declared()["prior"]["file"]).resolve()
    original = ctx.campaign.verify

    def verify(path):
        return verifier.verify(path) if Path(path).resolve() == expected else original(path)

    monkeypatch.setattr(ctx.campaign, "verify", verify)
    return ctx


def phase_work(ctx, actions=1, rows=8):
    work = ctx.sr.Work()
    work.actions, work.action_rows = actions, actions * rows
    work.action_sweeps, work.action_row_sweeps = actions, actions * rows
    work.action_residual_checks, work.action_row_residual_checks = actions, actions * rows
    work.max_action_residual = 0.001
    return work


def fake_evaluate(brain, arm, initial, deadline, *, causal=False):
    """Synthetic old-trial boundary only; no claimed neural reading."""
    del brain, arm, initial, deadline
    ctx = subject.context()
    rows = ctx.task.panel("all")
    result = {}
    for condition in subject.CONDITIONS if causal else ("intact",):
        expected = (
            ctx.task.flipped(rows, int(condition == "flip_B"))
            if condition.startswith("flip_")
            else rows
        )
        answer = ctx.task.targets(expected)
        slot = {
            "erase_0": 0,
            "erase_1": 1,
            "head_0": 0,
            "head_1": 1,
            "cloned_heads": 1,
            "frozen_interface": 0,
            "shuffled_interface": 1,
        }.get(condition)
        if slot is not None:
            answer[:, slot] = 1 - answer[:, slot]
        weights = np.zeros((2, 8, 3))
        weights[0, np.arange(8), rows[:, 2]] = 0.9
        weights[1, np.arange(8), 1 - rows[:, 2]] = 0.9
        result[condition] = subject.reading(
            answer, expected, phase_work(ctx), "completed", attention=weights.tolist()
        )
    return {"prefix_work": asdict(phase_work(ctx, 3)), "conditions": result}


def fake_train(brain, arm, schedule, deadline):
    del arm, deadline
    if brain is not None:
        brain.learner.updates += len(schedule)
        brain.learner.contrast_updates += len(schedule)
    work = phase_work(subject.context(), 4 * len(schedule), len(schedule[0]))
    work.lessons = len(schedule)
    work.presentations = work.accepted_presentations = len(schedule) * len(schedule[0])
    return {
        "completed": True,
        "rounds": len(schedule),
        "wrong_counts": [len(schedule[0])] * len(schedule),
        "stop": None,
        "work": asdict(work),
    }


@pytest.fixture
def synthetic(archived_prior, monkeypatch):
    ctx = archived_prior
    monkeypatch.setattr(subject, "evaluate", fake_evaluate)
    monkeypatch.setattr(ctx.admission, "train", fake_train)
    return ctx


def minimal_result(ctx):
    return {
        "arm": "h2",
        "training": fake_train(None, None, ctx.task.draw(0, 512), float("inf")),
        "initial": fake_evaluate(None, None, None, None),
        "evaluation": fake_evaluate(None, None, None, None),
        "parameters_preserved": True,
        "causal": fake_evaluate(None, None, None, None, causal=True),
    }


def test_protocol_has_no_held_or_comparator_stage_and_preserves_founder_point(ctx):
    spec = subject.declared()
    assert set(spec["stages"]) == {"capacity", "h2"}
    assert [
        (spec["stages"][k]["total_host_seconds"], spec["stages"][k]["training_seconds_per_call"])
        for k in subject.STAGES
    ] == [(600, 240), (1800, 600)]
    assert spec["task"]["held_out_cases"] == 0
    assert spec["prior"]["cumulative_host_seconds"] == 3518.474519670309
    for stage in subject.STAGES:
        point = subject.profile(stage)
        assert point["brain"] == spec["brain"]
        assert point["dt"] == 1 and point["brain"]["learning_damping"] == 3
    assert "at least .10" in spec["future_head_benefit_assessment"]["accuracy"]


def test_retention_is_balanced_all_eight_without_query_alternation(ctx):
    for seed in (0, 1):
        schedule = subject.retention_schedule(seed)
        assert schedule.shape == (256, 16, 4)
        for rows in schedule:
            unique, counts = np.unique(rows, axis=0, return_counts=True)
            np.testing.assert_array_equal(unique, ctx.task.CASES)
            np.testing.assert_array_equal(counts, np.full(8, 2))


def test_primary_and_distinct_usefulness_refuse_separately(ctx):
    r = minimal_result(ctx)
    assert subject.metrics(r)["primary_passed"]
    assert subject.metrics(r)["distinct_useful_heads"]
    lesion = r["causal"]["conditions"]["head_0"]
    lesion["status"] = "refused"
    lesion["work"]["action_refusals"] = 1
    m = subject.metrics(r)
    assert m["primary_passed"] and not m["distinct_useful_heads"]
    r["causal"]["conditions"]["flip_A"]["status"] = "refused"
    assert not subject.metrics(r)["primary_passed"]


def test_row_wise_roles_may_swap_and_slot_loss_uses_all_eight(ctx):
    r = minimal_result(ctx)
    rows = ctx.task.panel("all")
    weights = np.zeros((2, 8, 3))
    for h in range(2):
        preferred = h ^ rows[:, 2]
        slots = np.where(preferred == 0, rows[:, 2], 1 - rows[:, 2])
        weights[h, np.arange(8), slots] = 0.9
        answer = ctx.task.targets(rows)
        answer[np.arange(8), preferred] = 1 - answer[np.arange(8), preferred]
        r["causal"]["conditions"][f"head_{h}"]["answers"] = answer.tolist()
        r["causal"]["conditions"][f"head_{h}"]["scores"] = ctx.task.scores(answer, rows)
    r["causal"]["conditions"]["intact"]["attention"] = weights.tolist()
    m = subject.metrics(r)
    assert m["distinct_useful_heads"]
    assert m["specialization"]["slot_loss_denominator"] == 8
    assert m["specialization"]["predicted_slot_losses"] == [1.0, 1.0]
    answer = ctx.task.targets(rows)
    answer[2, 0] = 1 - answer[2, 0]
    r["causal"]["conditions"]["head_0"]["answers"] = answer.tolist()
    r["causal"]["conditions"]["head_0"]["scores"] = ctx.task.scores(answer, rows)
    assert not subject.metrics(r)["distinct_useful_heads"]


def test_real_controller_with_synthetic_boundaries_verifies_full_receipt_and_arrays(
    synthetic, tmp_path
):
    ctx = synthetic
    original_factory, original_task = ctx.admission.make_brain, ctx.admission.task
    out, parameters = tmp_path / "capacity.json", tmp_path / "arrays"
    receipt = subject.run("capacity", out, parameters, ctx.sr.revision())
    receipt.write(out)
    assert subject.verify(out)[0]
    assert receipt.body["capacity_admitted"]
    assert [r["arm"] for r in receipt.body["runs"]] == [
        "exposed",
        "exposed",
        "supplied_reads",
        "supplied_reads",
    ]
    assert ctx.admission.make_brain is original_factory and ctx.admission.task is original_task
    for r in receipt.body["runs"]:
        descriptor = r["parameter_snapshot"]
        with np.load(parameters / descriptor["file"], allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"]))
            assert metadata == descriptor["metadata"]
            assert metadata["source"] == subject.source_manifest(subject.sources("capacity"))
            assert metadata["query_founder"] == r["query_initialization"]
    for change in (
        "duplicate",
        "missing",
        "revision",
        "snapshot_source",
        "cost",
        "false_gate",
        "unrun",
        "wrong_score",
    ):
        body = deepcopy(receipt.body)
        r = body["runs"][0]
        if change == "duplicate":
            body["runs"][1] = deepcopy(r)
        elif change == "missing":
            body["runs"].pop()
        elif change == "revision":
            body["source_revision"] = "0" * 40
        elif change == "snapshot_source":
            r["parameter_snapshot"]["metadata"]["source"]["files"].pop()
        elif change == "cost":
            r["work"][subject.COST_KEYS[0]] = -1
        elif change == "false_gate":
            body["capacity_admitted"] = False
        elif change == "unrun":
            r["status"] = "unrun_budget"
        elif change == "wrong_score":
            r["evaluation"]["conditions"]["intact"]["answers"][0] = [3, 3]
        assert subject.check(body), change
    capped = deepcopy(receipt.body)
    capped["runs"] = [
        {"arm": arm, "seed": seed, "status": "unrun_budget", "passed": False}
        for arm, seed in subject.declared()["stages"]["capacity"]["order"]
    ]
    capped["capacity_admitted"] = False
    capped["elapsed_host_seconds"] = 1.0
    capped["cumulative_host_seconds"] = capped["previous_host_seconds"] + 1.0
    assert subject.check(capped) == "unrun cell before the declared stage deadline"
    capped["elapsed_host_seconds"] = 600.0
    capped["cumulative_host_seconds"] = capped["previous_host_seconds"] + 600.0
    assert subject.check(capped) is None


def test_h2_requires_own_positive_capacity_and_no_old_kind(synthetic, tmp_path, monkeypatch):
    ctx = synthetic
    old = ctx.runtime / "benchmarks/attention/results/query-founder-capacity.json"
    monkeypatch.setattr(subject, "_CAPACITY", old)
    with pytest.raises(ValueError, match="new matched minimal capacity"):
        subject.run("h2", tmp_path / "h2.json", tmp_path / "arrays", ctx.sr.revision())


def test_real_matched_capacity_to_both_h2_and_retention_with_no_neural_phases(
    synthetic, tmp_path, monkeypatch
):
    ctx = synthetic
    cap = tmp_path / "capacity.json"
    subject.run("capacity", cap, tmp_path / "capacity-arrays", ctx.sr.revision()).write(cap)
    monkeypatch.setattr(subject, "_CAPACITY", cap)
    out = tmp_path / "h2.json"
    receipt = subject.run("h2", out, tmp_path / "h2-arrays", ctx.sr.revision())
    receipt.write(out)
    assert subject.verify(out)[0]
    assert receipt.body["h2_admitted"] and receipt.body["distinct_useful_heads"]
    assert [r["seed"] for r in receipt.body["runs"]] == [0, 1]
    assert receipt.body["prerequisites"]["capacity"] == subject.Receipt.read(cap).digest
    for r in receipt.body["runs"]:
        assert r["retention"]["passed"] and r["retention"]["training"]["rounds"] == 256
        assert len(r["causal"]["conditions"]) == 13
        metadata = r["retention"]["parameter_snapshot"]["metadata"]
        assert metadata["phase"] == "retention" and metadata["source"] == subject.source_manifest(
            subject.sources("h2")
        )
    for change in ("retention", "donor_other_slot", "missing_panel", "distinct_flag"):
        body = deepcopy(receipt.body)
        r = body["runs"][1]
        if change == "retention":
            r["retention"]["training"]["work"]["lesson_refusals"] = 1
        elif change == "donor_other_slot":
            panel = r["causal"]["conditions"]["flip_A"]
            panel["answers"][0][1] = 1 - panel["answers"][0][1]
            panel["scores"] = ctx.task.scores(panel["answers"], panel["rows"])
        elif change == "missing_panel":
            r["causal"]["conditions"].pop("cloned_heads")
        else:
            body["distinct_useful_heads"] = False
        assert subject.check(body), change
    original_sources = subject.sources
    changed = tmp_path / "changed-task.py"
    changed.write_text(subject.TASK.read_text() + "\n# synthetic changed dependency\n")
    with monkeypatch.context() as scoped:
        scoped.setattr(
            subject,
            "sources",
            lambda stage: [
                (name, changed if name.endswith("minimal_binding_task.py") else path)
                for name, path in original_sources(stage)
            ],
        )
        assert not subject.verify(cap)[0]
        with pytest.raises(ValueError, match="new matched minimal capacity"):
            subject.run(
                "h2",
                tmp_path / "refused-changed.json",
                tmp_path / "refused-arrays",
                ctx.sr.revision(),
            )


def test_prior_verification_cache_invalidates_on_exact_dependency_change(
    ctx, monkeypatch, tmp_path
):
    dependency = tmp_path / "source.txt"
    dependency.write_text("original")
    calls = []
    monkeypatch.setattr(ctx.campaign, "sources", lambda stage: [("fixture/source", dependency)])
    old_path = ctx.runtime / "benchmarks/attention/results" / subject.declared()["prior"]["file"]
    original_bytes = old_path.read_bytes()
    copied = tmp_path / "runtime/benchmarks/attention/results" / old_path.name
    copied.parent.mkdir(parents=True)
    copied.write_bytes(original_bytes)
    monkeypatch.setattr(ctx, "runtime", tmp_path / "runtime")

    def verify(path):
        calls.append(path)
        return True, "synthetic custody boundary"

    monkeypatch.setattr(ctx.campaign, "verify", verify)
    first = subject.prior()
    first.body["h2_admitted"] = True
    assert subject.prior().body["h2_admitted"] is False
    assert len(calls) == 1
    dependency.write_text("changed")
    subject.prior()
    assert len(calls) == 2
    mutated = deepcopy(subject.declared())
    mutated["prior"]["digest"] = "f" * 64
    with monkeypatch.context() as scoped:
        scoped.setattr(subject, "declared", lambda: mutated)
        with pytest.raises(ValueError, match="negative prior differs"):
            subject.prior()
    assert len(calls) == 3
    copied.write_bytes(original_bytes + b"\n")
    with pytest.raises(ValueError, match="negative prior differs"):
        subject.prior()
    assert len(calls) == 4


def test_exception_keeps_actual_work_inherited_charge_and_full_source_snapshot(
    archived_prior, tmp_path, monkeypatch
):
    ctx = archived_prior
    monkeypatch.setattr(subject, "evaluate", fake_evaluate)

    def fail(brain, arm, schedule, deadline):
        del brain, arm, schedule, deadline
        raise RuntimeError("deliberate instrument fixture")

    monkeypatch.setattr(ctx.admission, "train", fail)
    actual_meter = ctx.admission.metered

    @contextmanager
    def metered(work):
        work["memory_drive_evaluations"] += 7  # Fake transport at the instrumentation boundary.
        with actual_meter(work):
            yield

    monkeypatch.setattr(ctx.admission, "metered", metered)
    original = ctx.admission.heads.Graph._synaptic_input
    path = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="deliberate instrument fixture"):
        subject.run("capacity", path, tmp_path / "arrays", ctx.sr.revision())
    journal = json.loads(path.with_suffix(".partial.json").read_text())
    assert journal["status"] == "cell_failed_unverified"
    assert set(journal["actual_transport_work"]) == set(subject.COST_KEYS)
    assert journal["actual_transport_work"]["memory_drive_evaluations"] == 7
    assert journal["previous_host_seconds"] == 3518.474519670309
    assert (
        journal["cumulative_host_seconds"]
        == journal["previous_host_seconds"] + journal["elapsed_host_seconds"]
    )
    assert journal["source_revision"] == ctx.sr.revision()
    assert journal["brain_snapshot"]["metadata"]["source"] == journal["source"]
    assert ctx.admission.heads.Graph._synaptic_input is original


def test_no_answer_or_lesson_refusal_can_qualify(ctx):
    r = minimal_result(ctx)
    for field in ("action_refusals", "lesson_refusals"):
        broken = deepcopy(r)
        broken["training"]["work"][field] = 1
        assert not subject.metrics(broken)["primary_passed"]
    panel = r["causal"]["conditions"]["flip_B"]
    panel["answers"][0] = [-1, -1]
    panel["scores"] = ctx.task.scores(panel["answers"], panel["rows"])
    assert not subject.metrics(r)["primary_passed"]
