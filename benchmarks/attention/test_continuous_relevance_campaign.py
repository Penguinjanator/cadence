"""Source-bound constructor/instrument tests; every neural phase is forbidden."""

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
    "continuous_campaign_tests", HERE / "continuous_relevance_campaign.py"
)
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)
RUNTIME = Path(os.environ.get("CADENCE_ATTENTION_RUNTIME", str(HERE.parents[1])))


@pytest.fixture
def ctx(monkeypatch):
    ctx = subject.configure(RUNTIME)
    monkeypatch.setattr(ctx.sr, "revision", lambda: "1" * 40)
    # This fixture tests this controller, not a recursive rerun of retained controllers.
    # Real prior bytes, typed digest, archive hash and source closure remain checked.
    monkeypatch.setattr(ctx.neutral, "verify", lambda path: (True, "synthetic boundary"))

    def forbidden(*args, **kwargs):
        raise AssertionError("no neural phases in the instrument tests")

    monkeypatch.setattr(ctx.heads.Graph, "equilibrate", forbidden)
    return ctx


def work(actions=1, rows=8):
    w = subject.context().sr.Work()
    w.actions, w.action_rows = actions, actions * rows
    w.action_sweeps, w.action_row_sweeps = actions, actions * rows
    w.action_residual_checks, w.action_row_residual_checks = actions, actions * rows
    w.max_action_residual = 0.001
    return w


def fake_train(brain, arm, records, deadline, progress=None):
    del arm, deadline
    if brain is not None:
        brain.learner.updates += len(records)
        brain.learner.contrast_updates += len(records)
    w = work(4 * len(records), 16)
    w.lessons = len(records)
    w.presentations = w.accepted_presentations = 16 * len(records)
    if progress is not None:
        progress(rounds=len(records), work=asdict(w), wrong_last=16)
    return {
        "completed": True,
        "rounds": len(records),
        "wrong_counts": [16] * len(records),
        "stop": None,
        "work": asdict(w),
        "schedule": records,
        "schedule_sha256": subject.canonical_sha256(records),
    }


def fake_evaluate(brain, arm, initial, seed, deadline, *, causal=False, start=0, progress=None):
    del initial, deadline
    ctx = subject.context()
    blocks = []
    for index in range(start, start + 32):
        batch = ctx.task.draw(seed, index, split="held")
        donors = [ctx.task.donor(batch, name, seed, index) for name in range(2)]
        block = {
            "index": index,
            "task": ctx.task.record(batch),
            "prefix": {
                "status": "completed",
                "work": asdict(work(3)),
                "bank": subject.bank_data(subject.expected_memory(brain, batch, 0, arm)),
            },
            "donors": [
                {
                    "name": name,
                    "task": ctx.task.record(donor),
                    "status": "completed",
                    "work": asdict(work(3)),
                    "bank": subject.bank_data(subject.expected_memory(brain, donor, 0, arm)),
                }
                for name, donor in enumerate(donors)
            ]
            if causal
            else [],
            "conditions": {},
        }
        for condition in subject.CONDITIONS if causal else ("intact",):
            answer = batch.rows[:, :2].copy()
            if condition == "flip_relevant":
                answer = 1 - answer
            elif condition == "erase_A":
                answer[:, 0] = 1 - answer[:, 0]
            elif condition == "erase_B":
                answer[:, 1] = 1 - answer[:, 1]
            elif condition in ("frozen_interface", "shuffled_interface"):
                answer = 1 - answer
            queries = []
            for q in range(2):
                donor = donors[q if condition == "flip_relevant" else 1 - q]
                memory = subject.expected_memory(
                    brain,
                    batch,
                    q,
                    arm,
                    condition,
                    donor_batch=donor if condition.startswith("flip_") else None,
                )
                _, drive, _ = subject.prepare(brain, batch, q, arm)
                activity = np.zeros((8, brain.connectome.n))
                if memory is not None:
                    query = np.zeros((8, 9))
                    query[:, q] = 10
                    query[:, 3] = 10
                    activity[:, brain.head_queries[0]] = query
                activity[:, brain.motor_index] = 0.2
                activity[np.arange(8), np.asarray(brain.motor_index)[answer[:, q]]] = 0.7
                queries.append(
                    {
                        "query": q,
                        "status": "completed",
                        "work": asdict(work()),
                        "bank": subject.bank_data(memory),
                        "drive": drive.tolist(),
                        "activation": activity.tolist(),
                        "potential": np.zeros_like(activity).tolist(),
                        "attention": None
                        if memory is None
                        else memory.weights(activity)[0].tolist(),
                    }
                )
            block["conditions"][condition] = {"answers": answer.tolist(), "queries": queries}
        blocks.append(block)
    panel = {"seed": seed, "start": start, "blocks": blocks}
    panel["scores"] = {
        c: subject.pooled(panel, c)["scores"]
        for c in (subject.CONDITIONS if causal else ("intact",))
    }
    if progress is not None:
        progress(blocks=32, planned=32)
    return panel


def result():
    brain = subject.make_brain("h1", 0)
    return {
        "arm": "h1",
        "training": fake_train(None, None, subject.schedule(0, 512), None),
        "initial": fake_evaluate(brain, "h1", None, 0, None),
        "evaluation": fake_evaluate(brain, "h1", None, 0, None),
        "causal": fake_evaluate(brain, "h1", None, 0, None, causal=True),
        "parameters_preserved": True,
        "initial_parameters_preserved": True,
    }


def test_factory_uses_raw9_one_native_output_same_parameters_and_q1(ctx):
    p = subject.declared()
    assert p["prior"]["cumulative_host_seconds"] == 5512.356352962387
    assert p["factory"]["inputs"] == 9 and p["factory"]["actions"] == 2
    brains = [subject.make_brain(a, 0) for a in ("exposed", "supplied_address", "h1")]
    assert [b.connectome.n for b in brains] == [75, 93, 93]
    for b in brains:
        c = b.learner.config
        assert c.beta == 0.1 and c.eta == 0.003 and c.damping == 3
        assert c.free_steps == c.nudged_steps == 2048 and c.normalize == 0.99
        assert b.brain.neuron_model.dt == 1 and b._episodic is None
    np.testing.assert_array_equal(brains[1].brain.efficacy, brains[2].brain.efficacy)
    assert brains[1].query_initialization == brains[2].query_initialization
    assert brains[2].query_initialization["applied"]
    assert not brains[0].query_initialization["applicable"]


def test_joint_schedule_preserves_exact_paired_fresh_payloads_and_neutral_offset(ctx):
    main, retention = subject.schedule(0, 512), subject.schedule(0, 256, offset=512)
    assert len(main) == 512 and len(retention) == 256
    assert [r["index"] for r in retention] == list(range(512, 768))
    assert [r["neutral_amplitude"] for r in main[:4]] == [1.0, 0.5, 1.0, 0.5]
    assert retention[0]["neutral_amplitude"] == 1
    for record in (main[0], main[-1], retention[0], retention[-1]):
        batch = subject.history(record["task"])
        prefix, question, target = subject.training_events(batch, "h1", record["neutral_amplitude"])
        np.testing.assert_array_equal(question[::2, 5], np.ones(8))
        np.testing.assert_array_equal(question[1::2, 6], np.ones(8))
        np.testing.assert_array_equal(target.reshape(8, 2), batch.rows[:, :2])
        for event in prefix:
            np.testing.assert_array_equal(event[::2], event[1::2])
    changed = deepcopy(main[0]["task"])
    changed["quantities"][1][0] += 0.01
    with pytest.raises(ValueError, match="paired"):
        subject.history(changed)


def test_actual_store_capture_owns_native_prefix_without_choosing_an_address(ctx):
    brain = subject.make_brain("h1", 0)
    batch = ctx.task.draw(0, 0)
    prefix, _ = ctx.task.events(batch, 0)
    brain.recall.reset(8)
    for event in prefix:
        brain.recall._observe(brain.recall.heard * event, event)
    memory = subject.witnessed_memory(brain)
    assert subject.bank_data(memory) == subject.bank_data(
        subject.expected_memory(brain, batch, 0, "h1")
    )
    before = memory.values.copy()
    brain.recall.values[:] = -99
    np.testing.assert_array_equal(memory.values, before)
    assert not memory.values.flags.writeable


def test_witness_uses_actual_stored_prefix_then_forks_same_host_for_both_questions(
    ctx, monkeypatch
):
    brain = subject.make_brain("h1", 0)
    batch, events = ctx.task.draw(0, 2), []

    def act(work, twin, event):
        twin.stimulus(event)
        twin.recall.witness(
            event, SimpleNamespace(activation=np.zeros((len(event), twin.connectome.n)))
        )
        events.append(event.copy())
        work.actions += 1
        work.action_rows += len(event)
        return np.zeros(len(event), int)

    monkeypatch.setattr(ctx.sr.Work, "act", act)
    host, memory, recorded = subject.witness(brain, "h1", batch, float("inf"))
    assert recorded["status"] == "completed" and len(events) == 3
    expected = subject.bank_data(subject.expected_memory(brain, batch, 0, "h1"))
    assert recorded["bank"] == subject.bank_data(memory) == expected
    assert brain.recall.writes == 0 and host.recall.writes == 24
    for query in (0, 1):
        twin = ctx.admission.binding.fork(host)
        question, _, bank = subject.prepare(twin, batch, query, "h1")
        assert subject.bank_data(bank) == expected
        assert np.all(question[:, 5 + query] == 1)
        assert host.recall.writes == 24


def test_actual_bank_interventions_preserve_joint_values_keys_and_unrelated_custody(ctx):
    brain = subject.make_brain("h1", 0)
    batch = ctx.task.draw(0, 0)
    bank = subject.expected_memory(brain, batch, 0, "h1")
    donor_batch = ctx.task.donor(batch, 0, 0, 0)
    donor = subject.expected_memory(brain, donor_batch, 0, "h1")
    changed = subject.bank_change(bank, "flip_relevant", 0, donor_bank=donor)
    selected = bank.held & (bank.keys[:, :, 0] > 0)
    np.testing.assert_array_equal(changed.keys[selected], donor.keys[selected])
    np.testing.assert_array_equal(changed.values[selected], donor.values[selected])
    np.testing.assert_array_equal(changed.values[~selected], bank.values[~selected])
    np.testing.assert_array_equal(changed.held, bank.held)
    neutral = subject.bank_change(bank, "neutral_half", 0)
    neutral_rows = bank.keys[:, :, 4] > 0
    np.testing.assert_array_equal(neutral.values[neutral_rows], bank.values[neutral_rows] / 2)
    np.testing.assert_array_equal(neutral.keys[neutral_rows], bank.keys[neutral_rows] / 2)
    erased = subject.bank_change(bank, "erase_neutral", 0)
    np.testing.assert_array_equal(erased.values, bank.values)
    assert not erased.held[neutral_rows].any()
    order = subject.bank_change(bank, "reorder", 0)
    for name in ("keys", "values", "held"):
        np.testing.assert_array_equal(getattr(order, name), getattr(bank, name)[:, ::-1])
    np.testing.assert_array_equal(order.ages, bank.ages[::-1])


def test_teacher_receives_captured_complete_drive_bank_and_postact_wrong_rows(ctx, monkeypatch):
    brain = subject.make_brain("h1", 0)
    records = subject.schedule(0, 1)
    target = np.asarray(records[0]["task"]["rows"])
    target = target[np.arange(16), target[:, 3]]
    answer = target.copy()
    answer[[1, 5, 9]] = 1 - answer[[1, 5, 9]]
    drive = np.arange(16 * brain.connectome.n).reshape(16, brain.connectome.n)
    marker, captures = object(), []
    monkeypatch.setattr(ctx.sr.Work, "act", lambda *args: np.zeros(16, int))

    def stimulus(question):
        brain.brain.memory = marker
        return drive.copy()

    monkeypatch.setattr(brain, "stimulus", stimulus)
    monkeypatch.setattr(ctx.admission, "act_drive", lambda *args: answer.copy())
    monkeypatch.setattr(
        ctx.admission,
        "teach",
        lambda b, d, t, selected, memory, w: captures.append(
            (d.copy(), t.copy(), selected, memory)
        ),
    )
    actual = subject.train(brain, "h1", records, float("inf"))
    assert actual["wrong_counts"] == [3]
    np.testing.assert_array_equal(captures[0][0], drive)
    np.testing.assert_array_equal(captures[0][1], target)
    np.testing.assert_array_equal(np.flatnonzero(captures[0][2]), [1, 5, 9])
    assert captures[0][3] is marker


def test_training_heartbeat_and_exact_census(ctx, monkeypatch):
    records = subject.schedule(0, 64)
    brain = subject.make_brain("exposed", 0)
    messages = []
    monkeypatch.setattr(ctx.sr.Work, "act", lambda *args: np.zeros(16, int))
    monkeypatch.setattr(ctx.admission, "act_drive", lambda *args: np.zeros(16, int))
    monkeypatch.setattr(ctx.admission, "teach", lambda *args: None)
    subject.train(brain, "exposed", records, float("inf"), progress=lambda **d: messages.append(d))
    assert [m["rounds"] for m in messages] == [32, 64]
    training = fake_train(None, None, subject.schedule(0, 512), None)
    subject.training_check(training, 0)
    for field in ("lessons", "presentations", "actions", "action_rows", "max_action_residual"):
        bad = deepcopy(training)
        bad["work"][field] += 1
        with pytest.raises(ValueError, match="census|work"):
            subject.training_check(bad, 0)


def test_metrics_require_each_query_relevance_and_preserve_null_competition(ctx):
    r = result()
    assert subject.metrics(r)["primary_passed"]
    for block in r["causal"]["blocks"]:
        block["conditions"]["erase_B"]["answers"] = block["conditions"]["intact"]["answers"]
    assert not subject.metrics(r)["primary_passed"]
    r = result()
    for block in r["causal"]["blocks"]:
        weights = np.array(block["conditions"]["intact"]["queries"][1]["attention"])
        block["conditions"]["intact"]["queries"][1]["attention"] = (weights * 0.45).tolist()
    metric = subject.metrics(r)
    assert not metric["selection_passed"] and metric["selective_unequal_shares"][1] == 0


def test_null_is_diagnostic_but_required_interface_refusal_blocks_admission(ctx):
    r = result()
    r["causal"]["blocks"][0]["conditions"]["all_erased"]["queries"][0]["status"] = "refused"
    assert subject.metrics(r)["primary_passed"]
    r["causal"]["blocks"][0]["conditions"]["frozen_interface"]["queries"][0]["status"] = "refused"
    assert not subject.metrics(r)["primary_passed"]


@pytest.mark.parametrize(
    "change", ["payload", "answer", "selector", "drive", "residual", "teaching"]
)
def test_panel_custody_rejects_changed_real_coordinates_or_qualified_state(ctx, change):
    brain = subject.make_brain("h1", 0)
    panel = fake_evaluate(brain, "h1", None, 0, None)
    row = panel["blocks"][0]["conditions"]["intact"]["queries"][0]
    if change == "payload":
        panel["blocks"][0]["task"]["quantities"][0][0] += 0.01
    elif change == "answer":
        panel["blocks"][0]["conditions"]["intact"]["answers"][0][0] ^= 1
    elif change == "selector":
        row["attention"][0][0] += 0.1
    elif change == "drive":
        row["drive"][0][0] += 1
    elif change == "residual":
        row["work"]["max_action_residual"] = 0.1
    else:
        row["work"]["lessons"] = 1
    with pytest.raises(ValueError):
        subject.panel_check(panel, brain, "h1", 0)


def test_prior_cache_rechecks_actual_source_and_finished_binding(ctx, monkeypatch):
    calls = []
    monkeypatch.setattr(ctx.neutral, "verify", lambda path: (calls.append(path) is None, "ok"))
    first = subject.prior()
    assert first.digest == subject.finished_prior()["digest"]
    assert subject.prior().digest == first.digest and len(calls) == 1
    original = ctx.neutral.sources
    monkeypatch.setattr(ctx.neutral, "sources", lambda stage: original(stage)[:-1])
    subject.prior()
    assert len(calls) == 2
    spec = deepcopy(subject.declared())
    spec["prior"]["status"] = "unfinished"
    monkeypatch.setattr(subject, "declared", lambda: spec)
    with pytest.raises(ValueError, match="unfinished"):
        subject.prior()


def test_controller_synthetic_capacity_h1_retention_snapshots_and_mutations(
    ctx, monkeypatch, tmp_path
):
    monkeypatch.setattr(subject, "evaluate", fake_evaluate)
    monkeypatch.setattr(subject, "train", fake_train)
    out, arrays = tmp_path / "capacity.json", tmp_path / "parameters"
    capacity = subject.run("capacity", out, arrays, ctx.sr.revision())
    capacity.write(out)
    assert capacity.body["capacity_admitted"] and subject.verify(out)[0]
    subject._CAPACITY = out
    learned = tmp_path / "h1.json"
    h1 = subject.run("learned_h1", learned, arrays, ctx.sr.revision())
    h1.write(learned)
    assert h1.body["h1_admitted"] and subject.verify(learned)[0]
    assert [r["retention"]["status"] for r in h1.body["runs"]] == ["measured", "measured"]
    assert (
        json.loads(learned.with_suffix(".progress.json").read_text())["phase"] == "cell_completed"
    )
    for r in h1.body["runs"]:
        descriptor = r["parameter_snapshot"]
        with np.load(arrays / descriptor["file"], allow_pickle=False) as archive:
            assert json.loads(str(archive["metadata"])) == descriptor["metadata"]
            assert set(archive.files) == {
                "metadata",
                *subject.PARAMETERS,
                *("optimizer_" + k for k in subject.MOMENTS),
            }
    for change in ("revision", "duplicate", "omission", "optimism", "source", "cost"):
        body = deepcopy(h1.body)
        if change == "revision":
            body["runs"][0]["parameter_snapshot"]["metadata"]["source_revision"] = "2" * 40
        elif change == "duplicate":
            body["runs"][1] = deepcopy(body["runs"][0])
        elif change == "omission":
            body["runs"].pop()
        elif change == "optimism":
            body["runs"][0]["causal"]["blocks"][0]["conditions"]["flip_relevant"]["answers"][0][
                0
            ] = -1
        elif change == "source":
            source = body["runs"][0]["parameter_snapshot"]["metadata"]["source"]
            source.pop(next(iter(source)))
        else:
            body["previous_host_seconds"] = 0
        assert subject.check(body) is not None


def test_earlier_capacity_or_revision_cannot_admit_fresh_candidate(ctx, tmp_path):
    old = tmp_path / "old-kind.json"
    subject.Receipt.build("cadence-attention-heads-admission/1", {}, []).write(old)
    subject._CAPACITY = old
    with pytest.raises(ValueError, match="new matched continuous capacity"):
        subject.prerequisites("learned_h1")
    with pytest.raises(ValueError, match="execution commit"):
        subject.run("capacity", tmp_path / "new.json", tmp_path / "arrays", "2" * 40)


def test_interruption_retains_elapsed_exact_schedule_actual_work_and_arrays(
    ctx, monkeypatch, tmp_path
):
    def fail(arm, seed, stage, deadline, directory, captures, progress):
        captures["brain"] = subject.make_brain(arm, seed)
        captures["actual_transport_work"] = dict.fromkeys(ctx.cost_keys, 7)
        captures["training_schedule"] = subject.schedule(seed, 512)
        progress(
            phase="synthetic_interruption", actual_transport_work=captures["actual_transport_work"]
        )
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
    assert journal["training_schedule"] == subject.schedule(0, 512)
    assert journal["actual_transport_work"] == dict.fromkeys(ctx.cost_keys, 7)
    assert journal["brain_snapshot"]["metadata"]["source_revision"] == ctx.sr.revision()
