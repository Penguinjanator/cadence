"""Pure arrays and constructor/instrument tests; neural phases are forbidden."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "single_campaign_tests", HERE / "single_relevance_campaign.py"
)
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)
RUNTIME = Path(os.environ.get("CADENCE_ATTENTION_RUNTIME", str(HERE.parents[1])))


@pytest.fixture
def ctx(monkeypatch):
    ctx = subject.configure(RUNTIME)
    monkeypatch.setattr(ctx.sr, "revision", lambda: "1" * 40)

    def forbidden(*args, **kwargs):
        raise AssertionError("no neural phases in the instrument tests")

    monkeypatch.setattr(ctx.admission.heads.Graph, "equilibrate", forbidden)
    return ctx


@pytest.fixture
def archived_prior(ctx, monkeypatch):
    """Verify only the declared historical boundary with the offline reader."""
    verifier = ctx.base.load("single_tests_historical_verifier", HERE / "verify.py")
    expected = (ctx.runtime / "benchmarks/attention/results"
                / subject.declared()["prior"]["file"]).resolve()
    original = ctx.base.verify

    def verify(path):
        return verifier.verify(path) if Path(path).resolve() == expected else original(path)

    monkeypatch.setattr(ctx.base, "verify", verify)
    return ctx


def work(actions=1, rows=8):
    w = subject.context().sr.Work()
    w.actions, w.action_rows = actions, actions * rows
    w.action_sweeps, w.action_row_sweeps = actions, actions * rows
    w.action_residual_checks, w.action_row_residual_checks = actions, actions * rows
    w.max_action_residual = 0.001
    return w


def fake_train(brain, arm, order, deadline):
    del arm, deadline
    if brain is not None:
        brain.learner.updates += len(order)
        brain.learner.contrast_updates += len(order)
        if brain.recall is not None:
            brain.recall.writes += 4 * len(order) * 16
    w = work(4 * len(order), 16)
    w.lessons = len(order)
    w.presentations = w.accepted_presentations = 16 * len(order)
    return {
        "completed": True,
        "rounds": len(order),
        "wrong_counts": [16] * len(order),
        "stop": None,
        "work": asdict(w),
    }


def fake_evaluate(brain, arm, initial, deadline, *, causal=False):
    del initial, deadline
    ctx = subject.context()
    if brain is None:
        arm, brain = "h1", subject.make_brain("h1", 0)
    conditions = {}
    facts = ctx.sr.FACTS
    for condition in subject.CONDITIONS if causal else ("intact",):
        answers = subject.expected(condition).copy()
        if condition == "erase_A":
            answers[:, 0] = 1 - answers[:, 0]
        elif condition == "erase_B":
            answers[:, 1] = 1 - answers[:, 1]
        elif condition in ("frozen_interface", "shuffled_interface"):
            answers[:] = 1 - answers
        queries = []
        for q in range(2):
            rows = np.column_stack((facts, np.full(8, q, dtype=np.int64)))
            _, drive, _, _ = subject.prepare(brain, rows, arm)
            boundary = None if arm == "exposed" else subject.expected_boundary(brain, q, condition)
            activity = {"association": np.zeros((8, 32)).tolist()}
            attention = None
            if boundary is not None:
                state = np.zeros((8, brain.connectome.n))
                query = np.eye(11)[2 * q + facts[:, q]]
                state[:, brain.head_queries[0]] = query
                activity.update(query=query.tolist(), read=np.zeros((8, 11)).tolist())
                attention = boundary.weights(state)[0].tolist()
            queries.append(
                {
                    "query": q,
                    "status": "completed",
                    "work": asdict(work()),
                    "bank": subject.bank_data(boundary),
                    "attention": attention,
                    "activation": activity,
                    "drive_sha256": subject.canonical_sha256(drive),
                }
            )
        conditions[condition] = {
            "answers": answers.tolist(),
            "scores": subject.scores(answers, condition),
            "queries": queries,
        }
    return {"rows": ctx.sr.CASES.tolist(), "prefix_work": asdict(work(3)), "conditions": conditions}


def result():
    return {
        "arm": "h1",
        "training": fake_train(None, None, subject.schedule(0, 512), None),
        "initial": fake_evaluate(None, None, None, None),
        "evaluation": fake_evaluate(None, None, None, None),
        "causal": fake_evaluate(None, None, None, None, causal=True),
        "parameters_preserved": True,
        "initial_parameters_preserved": True,
    }


def test_exact_factory_uses_one_native_output_and_matched_oracle(ctx):
    p = subject.declared()
    assert p["prior"]["cumulative_host_seconds"] == 4152.979600170228
    assert p["factory"]["inputs"] == 11 and p["factory"]["actions"] == 2
    brains = [subject.make_brain(arm, 0) for arm in ("exposed", "supplied_address", "h1")]
    assert [b.connectome.n for b in brains] == [77, 99, 99]
    for b in brains:
        c = b.learner.config
        assert c.beta == 0.1 and c.damping == 3 and c.eta == 0.003 and c.normalize == 0.99
        assert c.free_steps == c.nudged_steps == 2048 and b.brain.neuron_model.dt == 1
    np.testing.assert_array_equal(brains[1].brain.efficacy, brains[2].brain.efficacy)
    assert brains[1].query_initialization == brains[2].query_initialization
    assert brains[2].query_initialization["applied"]
    assert not brains[0].query_initialization["applicable"]


def test_schedules_are_balanced_same_point_and_independent(ctx):
    for seed in (0, 1):
        regular, retained = subject.schedule(seed, 512), subject.schedule(seed, 256, retention=True)
        assert regular.shape == (512, 16, 4) and retained.shape == (256, 16, 4)
        for rows in (regular[0], regular[-1], retained[0], retained[-1]):
            np.testing.assert_array_equal(np.unique(rows, axis=0), ctx.sr.CASES)
        np.testing.assert_array_equal(retained[0], retained[-1])
        assert not np.array_equal(regular[0], retained[0])


def test_metrics_require_relevant_erasure_each_name_and_each_query_donor(ctx):
    r = result()
    assert subject.metrics(r)["primary_passed"]
    r["causal"]["conditions"]["erase_B"]["answers"] = subject.expected("intact").tolist()
    assert not subject.metrics(r)["primary_passed"]
    r = result()
    donor = r["causal"]["conditions"]["donor_both"]
    a = np.asarray(donor["answers"])
    a[:2, 1] = 1 - a[:2, 1]  # Overall14/16=.875 would hide the B-query6/8 failure.
    donor["answers"], donor["scores"] = a.tolist(), subject.scores(a, "donor_both")
    assert not subject.metrics(r)["primary_passed"]


def test_selector_keeps_null_mass_and_demands_each_question(ctx):
    r = result()
    assert subject.metrics(r)["selection_passed"]
    weights = np.asarray(r["causal"]["conditions"]["intact"]["queries"][1]["attention"])
    # Half the mass is null: requested=.45 fails even though conditional-on-record=1.
    weights *= 0.5
    r["causal"]["conditions"]["intact"]["queries"][1]["attention"] = weights.tolist()
    assert not subject.metrics(r)["selection_passed"]
    r = result()
    unequal = np.flatnonzero(ctx.sr.FACTS[:, 0] != ctx.sr.FACTS[:, 1])
    weights = np.asarray(r["causal"]["conditions"]["intact"]["queries"][0]["attention"])
    weights[unequal[0]] = 0
    r["causal"]["conditions"]["intact"]["queries"][0]["attention"] = weights.tolist()
    assert not subject.metrics(r)["selection_passed"]  # Three/four < .8.


def test_erased_unavailable_is_diagnostic_but_interface_refusal_blocks(ctx):
    r = result()
    r["causal"]["conditions"]["all_erased"]["queries"][0]["status"] = "refused"
    assert subject.metrics(r)["primary_passed"]
    r["causal"]["conditions"]["frozen_interface"]["queries"][0]["status"] = "refused"
    assert not subject.metrics(r)["primary_passed"]


def test_unanswered_or_high_residual_cannot_be_relabelled_complete(ctx):
    panel = fake_evaluate(None, None, None, None)
    panel["conditions"]["intact"]["queries"][0]["work"]["max_action_residual"] = 0.1
    with pytest.raises(ValueError, match="qualify"):
        subject.panel_check(panel)
    panel = fake_evaluate(None, None, None, None)
    panel["conditions"]["intact"]["answers"][0][0] = -1
    panel["conditions"]["intact"]["scores"] = subject.scores(
        panel["conditions"]["intact"]["answers"]
    )
    with pytest.raises(ValueError, match="qualify"):
        subject.panel_check(panel)


def test_saved_complete_drive_teaches_only_postact_wrong_rows(ctx, monkeypatch):
    rows = ctx.sr.CASES.copy()
    captures = []
    shared_teach = ctx.admission.teach
    brain = subject.make_brain("h1", 0)
    monkeypatch.setattr(ctx.sr.Work, "act", lambda w, b, event: np.zeros(len(event), dtype=int))
    full_drive = np.arange(16 * brain.connectome.n).reshape(16, brain.connectome.n)
    memory = object()
    target = rows[np.arange(16), rows[:, 3]]
    monkeypatch.setattr(
        subject, "prepare", lambda b, r, a: (np.zeros((16, 11)), full_drive.copy(), memory, target)
    )
    answer = target.copy()
    answer[[1, 5, 9]] = 1 - answer[[1, 5, 9]]
    monkeypatch.setattr(ctx.admission, "act_drive", lambda *args: answer.copy())

    def teach(b, drive, t, selected, m, w):
        captures.append((drive.copy(), t.copy(), selected.copy(), m))

    monkeypatch.setattr(ctx.admission, "teach", teach)
    actual = subject.train(brain, "h1", rows[None], float("inf"))
    assert actual["wrong_counts"] == [3]
    np.testing.assert_array_equal(captures[0][0], full_drive)
    np.testing.assert_array_equal(np.flatnonzero(captures[0][2]), [1, 5, 9])
    assert captures[0][3] is memory
    answer[:] = target
    subject.train(brain, "h1", rows[None], float("inf"))
    assert len(captures) == 2 and not captures[1][2].any()
    # The existing shared teach helper skips an all-false mask, without a learner call.
    updates = brain.learner.updates
    shared_teach(brain, full_drive, target, np.zeros(16, dtype=bool), memory, ctx.sr.Work())
    assert brain.learner.updates == updates


def test_real_controller_synthetic_capacity_then_h1_and_retention(
    archived_prior, monkeypatch, tmp_path
):
    ctx = archived_prior
    monkeypatch.setattr(subject, "evaluate", fake_evaluate)
    monkeypatch.setattr(subject, "train", fake_train)
    out, parameters = tmp_path / "capacity.json", tmp_path / "parameters"
    receipt = subject.run("capacity", out, parameters, ctx.sr.revision())
    receipt.write(out)
    assert receipt.body["capacity_admitted"] and subject.verify(out)[0]
    assert len(receipt.body["runs"]) == 4
    subject._CAPACITY = out
    second = tmp_path / "h1.json"
    h1 = subject.run("learned_h1", second, parameters, ctx.sr.revision())
    h1.write(second)
    assert h1.body["h1_admitted"] and subject.verify(second)[0]
    assert [r["retention"]["status"] for r in h1.body["runs"]] == ["measured", "measured"]
    for r in h1.body["runs"]:
        descriptor = r["parameter_snapshot"]
        with np.load(parameters / descriptor["file"], allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            assert metadata == descriptor["metadata"]
            assert metadata["source"] == subject.source_manifest(subject.sources("learned_h1"))
            assert metadata["learning"]["beta"] == 0.1
    for change in ("revision", "duplicate", "omission", "optimism", "source"):
        body = deepcopy(h1.body)
        if change == "revision":
            body["runs"][0]["parameter_snapshot"]["metadata"]["source_revision"] = "2" * 40
        elif change == "duplicate":
            body["runs"][1] = deepcopy(body["runs"][0])
        elif change == "omission":
            body["runs"].pop()
        elif change == "optimism":
            body["runs"][0]["causal"]["conditions"]["flip_relevant"]["answers"][0][0] = -1
        else:
            source = body["runs"][0]["parameter_snapshot"]["metadata"]["source"]
            source.pop(next(iter(source)))
        assert subject.check(body) is not None


def test_stage_prerequisite_rejects_original_or_negative_capacity(
    archived_prior, monkeypatch, tmp_path
):
    out = tmp_path / "old-kind.json"
    subject.Receipt.build("cadence-attention-heads-admission/1", {}, []).write(out)
    subject._CAPACITY = out
    with pytest.raises(ValueError, match="new matched single capacity"):
        subject.prerequisites("learned_h1")


def test_interrupted_cell_preserves_charge_sources_and_full_arrays(
    archived_prior, monkeypatch, tmp_path
):
    ctx = archived_prior
    def fail(arm, seed, stage, deadline, directory, captures):
        captures["brain"] = subject.make_brain(arm, seed)
        captures["actual_transport_work"] = dict.fromkeys(ctx.base.COST_KEYS, 7)
        raise RuntimeError("synthetic interrupted cell")

    monkeypatch.setattr(subject, "trial", fail)
    out = tmp_path / "interrupted.json"
    with pytest.raises(RuntimeError, match="synthetic"):
        subject.run("capacity", out, tmp_path / "arrays", ctx.sr.revision())
    journal = json.loads(out.with_suffix(".partial.json").read_text())
    assert journal["status"] == "cell_failed_unverified" and not out.exists()
    assert (
        journal["cumulative_host_seconds"]
        == journal["previous_host_seconds"] + journal["elapsed_host_seconds"]
    )
    assert journal["source"] == subject.source_manifest(subject.sources("capacity"))
    assert journal["actual_transport_work"] == dict.fromkeys(ctx.base.COST_KEYS, 7)
    assert journal["brain_snapshot"]["metadata"]["source_revision"] == ctx.sr.revision()


@pytest.mark.parametrize("arm", ["h1", "supplied_address"])
def test_actual_evaluate_forks_same_eight_history_bank_for_both_questions(ctx, monkeypatch, arm):
    parent = subject.make_brain(arm, 0)
    before = subject.state_digest(parent)
    witnessed, queries = [], []

    def charge(destination):
        one = asdict(work())
        for key, value in one.items():
            if key == "max_action_residual":
                destination.max_action_residual = max(destination.max_action_residual, value)
            else:
                setattr(destination, key, getattr(destination, key) + value)

    def prefix_act(w, brain, observation):
        witnessed.append(observation.copy())
        brain.stimulus(observation)
        brain.recall.witness(
            observation, SimpleNamespace(activation=np.zeros((8, brain.connectome.n)))
        )
        charge(w)
        return np.zeros(8, dtype=int)

    def query_act(brain, query, drive, w):
        state = np.zeros((8, brain.connectome.n))
        state[:, brain.head_queries[0]] = query
        brain._test_state = SimpleNamespace(activation=state)
        q = int(query[0, 6] == 1)
        queries.append((q, query.copy(), drive.copy(), subject.bank_data(brain.brain.memory)))
        if arm == "supplied_address":
            expected = np.eye(11)[2 * q + ctx.sr.FACTS[:, q]]
            np.testing.assert_array_equal(brain.brain.memory.targets[0], expected)
        charge(w)
        return np.zeros(8, dtype=int)

    monkeypatch.setattr(ctx.sr.Work, "act", prefix_act)
    monkeypatch.setattr(ctx.admission, "act_drive", query_act)
    monkeypatch.setattr(ctx.admission.heads.MultiHeadBrain, "_activity", lambda b: b._test_state)
    panel = subject.evaluate(parent, arm, parent.brain.efficacy.copy(), float("inf"))
    assert len(witnessed) == 3 and len(queries) == 2
    assert [q[0] for q in queries] == [0, 1]
    assert queries[0][3] == queries[1][3]
    for q, query, drive, bank in queries:
        assert np.flatnonzero(query[0]).tolist() == [5 + q]
        assert query.shape == (8, 11) and drive.shape == (8, parent.connectome.n)
        assert bank["ages"] == [2.0, 1.0, 0.0]
    assert subject.state_digest(parent) == before and parent.recall.writes == 0
    subject.panel_check(panel, brain=subject.make_brain(arm, 0), arm=arm)


@pytest.mark.parametrize("change", ["prior_spec", "old_sources", "raw_prior"])
def test_historical_cache_never_hides_changed_declared_or_actual_dependency(
    ctx, monkeypatch, tmp_path, change
):
    calls = []
    monkeypatch.setattr(
        ctx.base, "verify", lambda path: (calls.append(str(path)) or True, "unit fixture")
    )
    p = deepcopy(subject.declared())
    prior = ctx.runtime / "benchmarks/attention/results" / p["prior"]["file"]
    copied = tmp_path / "benchmarks/attention/results" / prior.name
    copied.parent.mkdir(parents=True)
    copied.write_bytes(prior.read_bytes())
    ctx.runtime = tmp_path
    monkeypatch.setattr(subject, "declared", lambda: deepcopy(p))
    manifest = {"old_sources": {"sha256": "1" * 64}}
    monkeypatch.setattr(subject, "source_manifest", lambda paths: deepcopy(manifest))
    first = subject.prior()
    first.body["h2_admitted"] = True  # Returned data cannot mutate the cached verified body.
    assert subject.prior().body["h2_admitted"] is False and len(calls) == 1
    if change == "prior_spec":
        p["prior"]["digest"] = "0" * 64
    elif change == "old_sources":
        manifest["old_sources"]["sha256"] = "2" * 64
    else:
        copied.write_bytes(copied.read_bytes() + b"\n")
    if change == "old_sources":
        subject.prior()
    else:
        with pytest.raises(ValueError, match="prior differs"):
            subject.prior()
    assert len(calls) == 2
