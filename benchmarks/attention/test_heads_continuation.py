"""Strict lineage and exact reset-suffix identity, without a behavioral campaign."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from cadence.receipts import canonical_json, canonical_sha256

SPEC = importlib.util.spec_from_file_location(
    "heads_continuation_tests", Path(__file__).with_name("heads_continuation.py"))
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def capture(brain):
    arrays = {name: getattr(brain.brain, name).copy() for name in subject.PARAMETERS}
    arrays.update({"optimizer_" + name: getattr(brain.learner, name).copy()
                   for name in subject.OPTIMIZER})
    return arrays, {"learning": brain.learner.config.to_dict(),
                    "updates": brain.learner.updates,
                    "contrast_updates": brain.learner.contrast_updates}


def test_two_plus_two_reset_rounds_equal_uninterrupted_training_exactly():
    protocol = deepcopy(subject.recipe())
    protocol["brain"]["modules"] = [4]
    first = subject.admission.make_brain("h2", 0, protocol)
    schedule = subject.task.draw(0, 4, 2)
    prefix = subject.admission.train(first, "h2", schedule[:2], time.monotonic() + 15)
    assert prefix["completed"] and not prefix["work"]["lesson_refusals"]
    arrays, metadata = capture(first)
    assert metadata["updates"] > 0 and metadata["contrast_updates"] > 0
    restored = subject.admission.make_brain("h2", 0, protocol)
    subject.restore_arrays(restored, arrays, metadata)
    assert subject.state_digest(restored) == subject.state_digest(first)
    # No RNG state is restored: the frozen recipe's actual acts are greedy and
    # each round resets its transient traces/slots before reading the next draw.
    restored.rng.random(11)
    restored.basal_ganglia.rng.random(13)
    uninterrupted = subject.admission.train(first, "h2", schedule[2:], time.monotonic() + 15)
    resumed = subject.admission.train(restored, "h2", schedule[2:], time.monotonic() + 15)
    assert uninterrupted == resumed and resumed["completed"]
    assert subject.state_digest(first) == subject.state_digest(restored)
    np.testing.assert_array_equal(first.basal_ganglia.state.v, restored.basal_ganglia.state.v)
    np.testing.assert_array_equal(first.recall.keys, restored.recall.keys)


@pytest.mark.parametrize("change", ["missing", "shape", "nonfinite", "negative_moment",
                                    "counters", "config"])
def test_invalid_research_state_is_refused_before_parameters_mutate(change):
    b = subject.admission.make_brain("h2", 0, subject.recipe())
    arrays, metadata = capture(b)
    before = subject.state_digest(b)
    if change == "missing":
        arrays.pop("optimizer_velocity")
    elif change == "shape":
        arrays["optimizer_velocity"] = arrays["optimizer_velocity"][:-1]
    elif change == "nonfinite":
        arrays["bias"][0] = float("nan")
    elif change == "negative_moment":
        arrays["optimizer_second_moment"][0] = -1
    elif change == "counters":
        metadata["contrast_updates"] = metadata["updates"] + 1
    else:
        metadata["learning"]["beta"] *= 2
    with pytest.raises((ValueError, KeyError)):
        subject.restore_arrays(b, arrays, metadata)
    assert subject.state_digest(b) == before


def test_research_restore_rejects_a_life_that_has_unrestored_durable_or_code_state():
    protocol = deepcopy(subject.recipe())
    protocol["brain"]["episodic"] = True
    b = subject.admission.make_brain("h2", 0, protocol)
    arrays, metadata = capture(b)
    with pytest.raises(ValueError, match="raw greedy reset recipe"):
        subject.restore_arrays(b, arrays, metadata)


def test_original_relevance_and_capacity_receipts_are_verified_without_new_dispatcher():
    parent = subject.parents()
    assert parent.digest == subject.declared()["parent_relevance_digest"]
    assert not parent.body["passed"] and len(parent.body["runs"]) == 10
    spec = subject.declared()
    assert spec["parent_relevance_host_seconds"] == parent.body["cumulative_host_seconds"]
    assert spec["previous_host_seconds"] == 2338.0763868783333
    diagnostic = subject.diagnostic(parent)
    assert spec["previous_host_seconds"] == (parent.body["cumulative_host_seconds"]
                                             + diagnostic.body["elapsed_host_seconds"])


def test_protocol_preserves_exact_suffixes_and_declares_only_additional_host_work():
    spec = subject.declared()
    assert [(p["seed"], p["completed_rounds"], p["remaining_rounds"])
            for p in spec["points"]] == [(0, 321, 191), (1, 289, 223)]
    assert spec["budget"]["additional_host_seconds"] == 1100
    assert spec["budget"]["training_host_seconds_per_call"] == 240
    paths = subject.sources()
    assert len(paths) == len(dict(paths)) and all(path.exists() for _, path in paths)
    assert "benchmarks/attention/results/heads-relevance-founder-extended.json" in dict(paths)
    assert "benchmarks/attention/results/heads-h1-boundary-diagnostic.json" in dict(paths)


def skeleton():
    parent = subject.parents()
    points = []
    for declaration in subject.declared()["points"]:
        old = subject.parent_point(parent, declaration["seed"])
        suffix = {"completed": False, "rounds": 0, "wrong_counts": [], "stop": "wall_budget",
                  "work": subject.admission.asdict(subject.sr.Work())}
        result = deepcopy(old)
        result["training"] = subject.joined_training(old["training"], suffix, 512)
        # The point remains the same negative measured panel; this fixture tests
        # source/cost/completion validation rather than inventing a positive score.
        metadata = {"schema": "fixture", "source": parent.source,
                    "source_revision": parent.body["source_revision"],
                    "learning": old["learning_config"], "parameter_sha256": old["final_parameters"]}
        custody = {"archive": old["parameter_snapshot"], "metadata": metadata,
                   "initial_parameters": old["initial_parameters"],
                   "initial_interface_sha256": old["initial_interface_sha256"],
                   "restored_parameters": old["final_parameters"],
                   "restored_state_sha256": declaration["restored_state_sha256"]}
        points.append({"seed": old["seed"], "suffix_training": suffix,
            "suffix_schedule_sha256": canonical_sha256(subject.task.draw(old["seed"], 512, 16)[
                declaration["completed_rounds"]:]), "restoration": custody,
            "elapsed_host_seconds": 0.0, "additional_work": {k: 0 for k in old["work"]},
            "result": result})
    body = {"protocol": subject.declared(), "parent_reference": parent.digest,
            "diagnostic_reference": subject.declared()["intervening_diagnostic"]["digest"],
            "previous_host_seconds": subject.declared()["previous_host_seconds"],
            "elapsed_host_seconds": 0.0,
            "cumulative_host_seconds": subject.declared()["previous_host_seconds"],
            "new_candidate_rounds_used": 0, "confirmation_seeds_used": [],
            "head_advantage_claimed": False, "points": points, "passed": False,
            "combined_relevance": subject.combined(parent, points, 0.0)}
    # This synthetic metadata cannot impersonate the frozen NPZ's metadata hash.
    return body


def test_invented_restoration_metadata_cannot_pass_a_rehashed_receipt():
    assert subject.check(skeleton()) == "restoration custody differs"


@pytest.fixture
def checked_negative(monkeypatch):
    body = skeleton()
    spec = deepcopy(body["protocol"])
    for declaration, point in zip(spec["points"], body["points"], strict=True):
        declaration["metadata_sha256"] = canonical_sha256(point["restoration"]["metadata"])
    body["protocol"] = spec
    monkeypatch.setattr(subject, "declared", lambda: spec)
    assert subject.check(body) is None
    return body


@pytest.mark.parametrize("change", ["score", "pass", "refusal", "suffix", "order", "prior_cost",
                                    "elapsed", "point_cost", "diagnostic", "comparators",
                                    "advantage", "candidate", "fresh"])
def test_strict_check_rejects_forged_gates_costs_or_comparators(checked_negative, change):
    body = deepcopy(checked_negative)
    if change == "score":
        body["points"][0]["result"]["metrics"]["primary_passed"] = True
    elif change == "pass":
        body["passed"] = True
    elif change == "refusal":
        body["points"][0]["suffix_training"]["work"]["action_refusals"] = -1
    elif change == "suffix":
        body["points"][0]["suffix_training"]["completed"] = True
    elif change == "order":
        body["points"].reverse()
    elif change == "prior_cost":
        body["previous_host_seconds"] = 0.0
    elif change == "elapsed":
        body["elapsed_host_seconds"] = float("nan")
    elif change == "point_cost":
        body["points"][0]["elapsed_host_seconds"] = 2.0
        body["points"][0]["result"]["elapsed_host_seconds"] += 2.0
    elif change == "diagnostic":
        body["diagnostic_reference"] = "0" * 64
    elif change == "comparators":
        body["combined_relevance"]["runs"][0]["passed"] = True
    elif change == "advantage":
        body["head_advantage_claimed"] = True
    elif change == "candidate":
        body["new_candidate_rounds_used"] = 1
    else:
        body["confirmation_seeds_used"] = [200]
    assert subject.check(body) is not None


def test_original_failed_comparators_are_preserved_by_combining_only_h2():
    parent = subject.parents()
    points = [{"seed": seed, "result": deepcopy(subject.parent_point(parent, seed))}
              for seed in (0, 1)]
    combined = subject.combined(parent, points, 0.25)
    assert [r for r in combined["runs"] if r["arm"] != "h2"] == [
        r for r in parent.body["runs"] if r["arm"] != "h2"]
    assert combined["candidate_rounds_used"] == 1 and not combined["passed"]
    assert combined["confirmation_seeds_used"] == []
    assert combined["elapsed_host_seconds"] == (parent.body["elapsed_host_seconds"]
        + subject.declared()["intervening_diagnostic"]["elapsed_host_seconds"] + 0.25)


def test_work_join_counts_every_attempt_and_takes_residual_maximum():
    original = {"action_sweeps": 100, "teaching_sweeps": 80, "max_action_residual": 0.002}
    suffix = {"action_sweeps": 40, "teaching_sweeps": 20, "max_action_residual": 0.003}
    assert subject.add_work(original, suffix) == {
        "action_sweeps": 140, "teaching_sweeps": 100, "max_action_residual": 0.003}
    with pytest.raises(ValueError, match="inventories"):
        subject.add_work(original, {"action_sweeps": 10})


def test_new_receipt_protocol_contains_no_binary_snapshot_or_local_path():
    spec = json.loads(subject.PROTOCOL.read_text())
    assert "/private/" not in canonical_json(spec)
    for point in spec["points"]:
        assert len(point["metadata_sha256"]) == len(point["restored_state_sha256"]) == 64


def test_standalone_verifier_rejects_another_receipt_kind(tmp_path):
    path = tmp_path / "wrong-kind.json"
    subject.Receipt.build("unrelated/1", {}).write(path)
    assert subject.verify(path) == (False, "not an attention continuation receipt")


def test_retained_completed_recipe_does_not_turn_into_an_acquisition_or_head_claim():
    path = subject.HERE / "results" / "heads-continuation-development.json"
    valid, message = subject.verify(path)
    assert valid, message
    receipt = subject.Receipt.read(path)
    assert receipt.digest == "12c384dae5992e3675baf2d74ec38d680fc90ada107ae99598036d146d065a40"
    body = receipt.body
    assert not body["passed"] and not body["head_advantage_claimed"]
    assert body["new_candidate_rounds_used"] == 0
    assert body["confirmation_seeds_used"] == []
    for point in body["points"]:
        result = point["result"]
        assert result["training"]["completed"] and result["training"]["rounds"] == 512
        assert result["training"]["stop"] is None
        assert result["training"]["work"]["action_refusals"] == 0
        assert result["training"]["work"]["lesson_refusals"] == 0
        assert result["evaluation"]["train"]["scores"]["exact_pair_accuracy"] == 8 / 48
        assert result["evaluation"]["held_out"]["scores"]["exact_pair_accuracy"] == 0
        assert result["metrics"]["required_causal_answers_qualified"]
        assert not result["metrics"]["acquired"]
        assert not result["metrics"]["causal_passed"]
        assert result["retention"]["status"] == "not_run_primary_failed"
