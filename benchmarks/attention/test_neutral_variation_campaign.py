"""Data/custody checks only. Every actual neural phase is forbidden."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadence.receipts import canonical_sha256

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "neutral_variation_subject", HERE / "neutral_variation_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


@pytest.fixture
def ctx(monkeypatch):
    runtime = Path(os.environ.get("CADENCE_ATTENTION_RUNTIME", HERE.parents[1]))
    result = subject.configure(runtime)

    def forbidden(*args, **kwargs):
        raise AssertionError("a neural phase was attempted by a data/custody test")

    monkeypatch.setattr(result.admission.heads.Graph, "equilibrate", forbidden)
    return result


@pytest.fixture
def archived_priors(ctx, monkeypatch):
    """Read both retained prerequisites without changing live admission policy."""
    verifier = ctx.base.load("neutral_tests_historical_verifier", HERE / "verify.py")
    for owner, spec in ((ctx.base, subject.declared()["retained_negative"]),
                        (subject.h1_verifier(), subject.finished_prior_spec())):
        expected = (ctx.runtime / "benchmarks/attention/results" / spec["file"]).resolve()
        original = owner.verify

        def verify(path, *, expected=expected, original=original):
            return verifier.verify(path) if Path(path).resolve() == expected else original(path)

        monkeypatch.setattr(owner, "verify", verify)
    return ctx


def test_only_data_change_preserves_every_original_gate_order_and_budget(archived_priors):
    ctx = archived_priors
    p, old = subject.declared(), ctx.base.declared()
    for key in ("brain", "dt", "task", "gates", "retention", "causal_controls"):
        assert p[key] == old[key]
    assert [p["stages"][s]["order"] for s in subject.STAGES] == [
        old["stages"][s]["order"] for s in subject.STAGES
    ]
    assert p["retained_negative"]["digest"] == (
        "2f0ef07ba3d8dc312cd25b2d323389be63069e3630f1187564a5e17c759226c7"
    )
    assert p["immediate_prior"]["cumulative_host_seconds"] == 4428.81996425341
    assert subject.finished_prior_spec()["source_stage"] == "learned_h1"
    actual = subject.prior()
    assert actual.digest == p["immediate_prior"]["digest"]
    assert actual.body["cumulative_host_seconds"] == p["immediate_prior"]["cumulative_host_seconds"]
    paths = dict(subject.sources("capacity"))
    assert paths["benchmarks/attention/results/single-relevance-h1.json"].is_file()
    assert paths["benchmarks/attention/results/single-relevance-capacity.json"].is_file()
    assert p["stages"]["capacity"]["receipt_kind"] != old["stages"]["capacity"]["receipt_kind"]


def test_cannot_weaken_an_old_gate_through_new_protocol(ctx, monkeypatch, tmp_path):
    p = subject.declared()
    p["gates"]["sham_answer_change_maximum"] = 0.5
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(p))
    monkeypatch.setattr(subject, "PROTOCOL", changed)
    with pytest.raises(ValueError, match="point/gates differ"):
        subject.declared()


def test_unfinished_h1_prior_blocks_prior_sources_and_run_before_any_constructor(
    ctx, monkeypatch, tmp_path
):
    def forbidden(*args, **kwargs):
        pytest.fail("unfinished prerequisite reached a fresh constructor")

    unfinished = subject.declared()
    unfinished["immediate_prior"]["status"] = (
        "unfinished_h1_stages_required_before_freeze_or_execution"
    )
    unfinished["immediate_prior"]["digest"] = None
    monkeypatch.setattr(subject, "declared", lambda: unfinished)
    monkeypatch.setattr(subject, "make_brain", forbidden)
    monkeypatch.setattr(ctx.sr, "revision", lambda: "1" * 40)
    for call in (
        subject.prior,
        lambda: subject.sources("capacity"),
        lambda: subject.run(
            "capacity", tmp_path / "receipt.json", tmp_path / "parameters", "1" * 40
        ),
    ):
        with pytest.raises(ValueError, match="unfinished actual H1 prior"):
            call()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("seed", [0, 1])
def test_original_case_draws_and_balanced_retention_are_exact(ctx, seed):
    p = subject.declared()
    schedule = ctx.task.draw(seed, p["task"]["rounds"], p["task"]["batch"])
    np.testing.assert_array_equal(schedule, ctx.base.context().task.draw(seed, 512, 16))
    amplitudes = subject.neutral_amplitudes(len(schedule))
    np.testing.assert_array_equal(amplitudes, np.tile([1.0, 0.5], 256))
    suffix = subject.retention_schedule(seed)
    np.testing.assert_array_equal(suffix, ctx.base.retention_schedule(seed))
    for rows in suffix:
        unique, counts = np.unique(rows, axis=0, return_counts=True)
        np.testing.assert_array_equal(unique, ctx.task.CASES)
        np.testing.assert_array_equal(counts, np.full(8, 2))
    np.testing.assert_array_equal(
        subject.neutral_amplitudes(len(suffix), offset=512), np.tile([1.0, 0.5], 128)
    )
    assert subject.joint_schedule_digest(schedule) != subject.joint_schedule_digest(
        schedule, offset=1
    )
    assert subject.joint_schedule_digest(suffix) != subject.joint_schedule_digest(
        suffix, offset=512
    )


@pytest.mark.parametrize("amplitude", [1.0, 0.5])
def test_actual_training_events_change_only_neutral_input_and_preserve_query(ctx, amplitude):
    rows = ctx.task.panel("all")
    original_prefix, original_query = ctx.task.events(rows)
    prefix, query = subject.training_events(rows, amplitude)
    np.testing.assert_array_equal(prefix[:2], original_prefix[:2])
    np.testing.assert_array_equal(prefix[2], amplitude * original_prefix[2])
    np.testing.assert_array_equal(query, original_query)
    assert np.count_nonzero(prefix[2]) == 8
    np.testing.assert_array_equal(prefix[2][:, 1], np.full(8, amplitude))
    np.testing.assert_array_equal(original_prefix[2][:, 1], np.ones(8))
    np.testing.assert_array_equal(ctx.task.targets(rows), rows[:, :2])


@pytest.mark.parametrize("amplitude", [1.0, 0.5])
def test_real_recall_witness_preserves_actual_amplitude_in_both_keys_and_values(ctx, amplitude):
    brain = subject.make_brain("h2", 0, "h2")
    rows = ctx.task.panel("all")
    prefix, _ = subject.training_events(rows, amplitude)
    # A supplied zero state permits checking normal witness/write arithmetic only;
    # it is not a claimed settled state or behavioral reading.
    state = SimpleNamespace(activation=np.zeros((8, brain.connectome.n)), device=None)
    for observation in prefix:
        brain.recall.witness(observation, state)
    expected = np.stack(prefix, axis=1)
    np.testing.assert_array_equal(brain.recall.values, expected)
    np.testing.assert_array_equal(brain.recall.keys, brain.recall.heard * expected)
    np.testing.assert_array_equal(brain.recall.held, np.ones((8, 3), dtype=bool))
    assert brain.recall.writes == 24
    assert brain.query_initialization["query_scale"] == 1


@pytest.mark.parametrize("offset", [0, 512])
@pytest.mark.parametrize("arm", ["exposed", "supplied_reads", "h2"])
def test_training_uses_actual_amplitude_and_same_captured_wrong_pair_boundary(
    ctx, monkeypatch, offset, arm
):
    history, lessons, prepared = [], [], []

    class InstrumentBrain:
        last_settlement = None

        def reset(self):
            history.append([])

        def act(self, observation, *, greedy):
            assert greedy
            history[-1].append(observation.copy())
            self.last_settlement = {
                "qualified": True,
                "steps": 1,
                "residual_checks": 1,
                "max_residual": 0.0,
            }
            return np.zeros((len(observation), 2), dtype=int)

    def prepare(brain, rows, selected_arm):
        assert selected_arm == arm
        _, query = ctx.task.events(rows, exposed=arm == "exposed")
        drive, memory = np.full((len(rows), 3), len(prepared)), object()
        prepared.append((drive, memory))
        return query, drive, memory

    schedule = ctx.task.draw(0, 4, 16)
    answer_index = iter(range(len(schedule)))

    def answer(brain, query, drive, work):
        tick = next(answer_index)
        assert drive is prepared[-1][0]
        result = ctx.task.targets(schedule[tick])
        result[0, 1] = 1 - result[0, 1]
        return result

    def teach(brain, drive, targets, selected, memory, work):
        assert drive is prepared[-1][0] and memory is prepared[-1][1]
        lessons.append((targets.copy(), selected.copy()))

    monkeypatch.setattr(ctx.admission, "prepare", prepare)
    monkeypatch.setattr(ctx.admission, "act_drive", answer)
    monkeypatch.setattr(ctx.admission, "teach", teach)
    result = subject.train(InstrumentBrain(), arm, schedule, float("inf"), offset=offset)
    assert result["completed"] and result["rounds"] == 4
    assert result["wrong_counts"] == [1] * 4
    subject.training_data_check(result, schedule, offset=offset)
    for tick, (observations, lesson) in enumerate(zip(history, lessons, strict=True)):
        assert len(observations) == 3
        prefix, _ = ctx.task.events(schedule[tick])
        np.testing.assert_array_equal(observations[:2], prefix[:2])
        np.testing.assert_array_equal(observations[2], prefix[2] * (1.0 if tick % 2 == 0 else 0.5))
        np.testing.assert_array_equal(lesson[0], ctx.task.targets(schedule[tick]))
        np.testing.assert_array_equal(lesson[1], np.arange(16) == 0)
    tampered = deepcopy(result)
    tampered["completed_neutral_amplitudes"][1] = 1.0
    with pytest.raises(ValueError, match="training schedule differs"):
        subject.training_data_check(tampered, schedule, offset=offset)


def test_research_snapshot_binds_all_parameters_optimizer_arrays_and_own_schema(
    ctx, monkeypatch, tmp_path
):
    brain = subject.make_brain("h2", 0, "h2")
    monkeypatch.setattr(
        subject, "sources", lambda stage: [("instrument.py", Path(subject.__file__))]
    )
    monkeypatch.setattr(ctx.sr, "revision", lambda: "2" * 40)
    descriptor = subject.snapshot(brain, tmp_path, stage="h2", arm="h2", seed=0)
    path = tmp_path / descriptor["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["archive_sha256"]
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            *subject.PARAMETERS,
            *("optimizer_" + k for k in subject.MOMENTS),
            "metadata",
        }
        assert descriptor["content_sha256"] == canonical_sha256(
            {key: archive[key].tolist() for key in archive.files}
        )
        metadata = json.loads(archive["metadata"].item())
        assert metadata == descriptor["metadata"]
        assert metadata["schema"] == "cadence-attention-neutral-variation-parameters/1"
        assert metadata["state_sha256"] == subject.state_digest(brain)
        assert metadata["updates"] == metadata["contrast_updates"] == 0


def test_synthetic_full_controller_capacity_h2_retention_and_schedule_verification(
    ctx, monkeypatch, tmp_path
):
    """Exercise real receipt/snapshot plumbing with explicitly fake neural boundaries."""
    from dataclasses import asdict

    from cadence.receipts import Receipt

    def phase_work(actions=1):
        work = ctx.sr.Work()
        work.actions, work.action_rows = actions, 8 * actions
        work.action_sweeps, work.action_row_sweeps = actions, 8 * actions
        work.action_residual_checks, work.action_row_residual_checks = actions, 8 * actions
        return work

    def fake_evaluate(brain, arm, initial, deadline, *, causal=False):
        del brain, arm, initial, deadline
        rows, readings = ctx.task.panel("all"), {}
        for condition in ctx.base.CONDITIONS if causal else ("intact",):
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
            readings[condition] = ctx.base.reading(
                answer, expected, phase_work(), "completed", attention=weights.tolist()
            )
        return {"prefix_work": asdict(phase_work(3)), "conditions": readings}

    def fake_train(brain, arm, schedule, deadline, *, offset=0):
        del arm, deadline
        brain.learner.updates += len(schedule)
        brain.learner.contrast_updates += len(schedule)
        return {
            "completed": True,
            "rounds": len(schedule),
            "wrong_counts": [0] * len(schedule),
            "stop": None,
            "work": asdict(ctx.sr.Work()),
            "global_round_offset": offset,
            "joint_schedule_sha256": subject.joint_schedule_digest(schedule, offset=offset),
            "completed_neutral_amplitudes": subject.neutral_amplitudes(
                len(schedule), offset=offset
            ).tolist(),
        }

    def fixture_sources(stage):
        paths = [
            ("fixture/controller.py", Path(subject.__file__)),
            ("fixture/protocol.json", subject.PROTOCOL),
        ]
        if stage == "h2":
            paths.append(("fixture/new-capacity.json", subject._CAPACITY))
        return paths

    fake_prior = Receipt.build("synthetic-prior/1", {"cumulative_host_seconds": 12.0})
    monkeypatch.setattr(subject, "prior", lambda: fake_prior)
    monkeypatch.setattr(subject, "sources", fixture_sources)
    monkeypatch.setattr(subject, "evaluate", fake_evaluate)
    monkeypatch.setattr(subject, "train", fake_train)
    monkeypatch.setattr(ctx.sr, "revision", lambda: "3" * 40)
    cap = tmp_path / "capacity.json"
    capacity = subject.run("capacity", cap, tmp_path / "capacity-arrays", "3" * 40)
    capacity.write(cap)
    assert capacity.body["capacity_admitted"]
    assert subject.verify(cap)[0]
    monkeypatch.setattr(subject, "_CAPACITY", cap)
    output = tmp_path / "h2.json"
    receipt = subject.run("h2", output, tmp_path / "h2-arrays", "3" * 40)
    receipt.write(output)
    assert subject.verify(output)[0]
    assert receipt.body["h2_admitted"] and receipt.body["distinct_useful_heads"]
    assert receipt.body["previous_host_seconds"] == capacity.body["cumulative_host_seconds"]
    assert receipt.body["prerequisites"]["capacity"] == capacity.digest
    assert receipt.body["prerequisites"]["completed_h1"] == fake_prior.digest
    for result in receipt.body["runs"]:
        assert result["retention"]["training"]["global_round_offset"] == 512
        assert result["retention"]["passed"]
    for phase in ("training", "retention"):
        changed = deepcopy(receipt.body)
        training = (
            changed["runs"][0]["training"]
            if phase == "training"
            else changed["runs"][0]["retention"]["training"]
        )
        training["completed_neutral_amplitudes"][0] = 0.5
        assert subject.check(changed)
    changed = deepcopy(receipt.body)
    changed["previous_host_seconds"] = 12.0
    assert subject.check(changed)


def test_h1_verifier_reconfigures_when_exact_declared_capacity_dependency_changes(ctx, monkeypatch):
    point = {"source_stage": "learned_h1", "capacity_file": "capacity-a.json"}
    calls = []

    def load(name, path):
        return SimpleNamespace(
            configure=lambda runtime, capacity: calls.append((runtime, capacity))
        )

    monkeypatch.setattr(subject, "finished_prior_spec", lambda: point)
    monkeypatch.setattr(ctx.base, "load", load)
    first = subject.h1_verifier()
    assert subject.h1_verifier() is first and len(calls) == 1
    point["capacity_file"] = "capacity-b.json"
    assert subject.h1_verifier() is not first and len(calls) == 2
    assert calls[-1][1].name == "capacity-b.json"


def test_positive_h1_capacity_cannot_close_the_required_learned_stage(ctx, monkeypatch, tmp_path):
    from cadence.receipts import Receipt

    p = subject.declared()
    old = ctx.runtime / "benchmarks/attention/results" / p["retained_negative"]["file"]
    results = tmp_path / "runtime/benchmarks/attention/results"
    results.mkdir(parents=True)
    (results / old.name).write_bytes(old.read_bytes())
    dependency = tmp_path / "dependency.txt"
    dependency.write_text("synthetic source boundary")
    monkeypatch.setattr(ctx, "runtime", tmp_path / "runtime")
    monkeypatch.setattr(ctx.base, "sources", lambda stage: [("fixture/dependency", dependency)])
    monkeypatch.setattr(ctx.base, "verify", lambda path: (True, "synthetic old verifier"))
    monkeypatch.setattr(
        subject,
        "h1_verifier",
        lambda: SimpleNamespace(
            sources=lambda stage: [("fixture/dependency", dependency)],
            verify=lambda path: (True, "synthetic H1 verifier"),
        ),
    )
    monkeypatch.setattr(subject, "declared", lambda: p)
    path = results / "synthetic-h1-capacity.json"
    for admitted in (True, False):
        receipt = Receipt.build(
            "synthetic-h1-capacity/1",
            {"stage": "capacity", "capacity_admitted": admitted, "cumulative_host_seconds": 5000.0},
        )
        receipt.write(path)
        p["immediate_prior"].update(
            status="finished_verified_h1_prior",
            file=path.name,
            kind=receipt.kind,
            digest=receipt.digest,
            archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            cumulative_host_seconds=5000.0,
            source_stage="capacity",
            capacity_file=None,
        )
        if admitted:
            with pytest.raises(ValueError, match="exact finished H1 prior differs"):
                subject.prior()
        else:
            assert subject.prior().digest == receipt.digest
