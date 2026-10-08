"""Guards of the finite recall chamber: protocol, arms, forks, seams and the receipt."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inputs = load_script("finite_horizon_inputs")
chamber = load_script("finite_horizon")


@pytest.fixture(autouse=True)
def quiet():
    warnings.simplefilter("ignore")


@pytest.fixture(scope="module")
def protocol():
    return chamber.load_protocol()


def test_the_protocol_declares_the_reviewed_recipe_and_fresh_founders(protocol):
    declared, digest = protocol
    assert digest == hashlib.sha256(chamber.PROTOCOL_PATH.read_bytes()).hexdigest()
    assert declared["schema"] == chamber.SCHEMA
    assert declared["seeds"]["founders"] == [301, 302, 303]
    brain = declared["brain"]
    assert (brain["trace_decay"], brain["trace_amplitude"]) == (0.8, 1.0)
    assert (brain["default_decay"], brain["default_amplitude"]) == (
        0.2,
        3.0,
    )  # the composed defaults
    assert declared["learning"]["qualified"] is True and declared["learning"]["nudged_steps"] == 128
    assert declared["arms"] == list(chamber.ARMS)
    assert declared["training"]["repeats"] == 16 and declared["evaluation"]["repeats"] == 24
    assert declared["caps"] == {"seconds": 900, "bytes": 160 * 1024 * 1024}
    assert declared["gates"]["minimum_horizon"] == 1


def test_the_arms_differ_only_in_their_declared_trace_genes(protocol):
    declared, _ = protocol
    vanished = chamber.brain_for("vanished", declared, 301)
    default = chamber.brain_for("default", declared, 301)
    history = chamber.brain_for("history", declared, 301)
    assert (vanished.working_memory.decay, vanished.working_memory.amplitude) == (0.8, 1.0)
    assert (default.working_memory.decay, default.working_memory.amplitude) == (0.2, 3.0)
    np.testing.assert_array_equal(vanished.brain.efficacy, history.brain.efficacy)
    np.testing.assert_array_equal(vanished.brain.efficacy, default.brain.efficacy)
    assert vanished.hippocampus is None and vanished.efference is None
    assert vanished.learner.config.qualified and vanished.learner.config.nudged_steps == 128


def test_the_trace_audit_accepts_the_source_recurrence_and_rejects_a_skipped_update():
    before = {"trace": np.zeros((2, 3)), "last": np.zeros((2, 3)), "cold": np.array([True, True])}
    h = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    after = {"trace": 0.2 * h, "last": h.copy(), "cold": np.array([False, False])}
    assert chamber.audit_trace(before, after, h, 0.8) <= 1e-12
    with pytest.raises(ValueError):
        chamber.audit_trace(before, {**after, "trace": 0.5 * h}, h, 0.8)
    with pytest.raises(ValueError):
        chamber.audit_trace(before, {**after, "cold": np.array([True, False])}, h, 0.8)
    after_default = {"trace": 0.8 * h, "last": h.copy(), "cold": np.array([False, False])}
    assert chamber.audit_trace(before, after_default, h, 0.2) <= 1e-12


def test_a_small_run_scores_forks_seams_and_timing_and_its_receipt_verifies(protocol, tmp_path):
    out = tmp_path / "run"
    code = chamber.main(["--out", str(out), "--seeds", "0", "--repeats", "1", "1"])
    assert code == 0
    valid, reason = chamber.verify(out)
    assert valid, reason
    body = json.loads((out / "summary.json").read_text())["body"]
    assert body["frozen_protocol"] is False and body["declaration"]["repeats"] == {
        "train": 1,
        "test": 1,
    }
    founder = body["founders"][0]
    assert set(founder["arms"]) == set(chamber.ARMS)
    vanished = founder["arms"]["vanished"]
    assert vanished["training"]["episodes"] == len(inputs.TRAIN_CONDITIONS)
    assert len(vanished["trials"]) == len(inputs.TEST_CONDITIONS)
    for trial in vanished["trials"]:
        assert set(trial["branches"]) == set(chamber.CONTROLS)
        assert len(trial["labels"]) == inputs.STREAMS and len(trial["random"]) == inputs.STREAMS
    assert set(vanished["seams"]) == {c.name for c in inputs.TEST_CONDITIONS}
    replacement = vanished["seams"]["replacement-1"]["seams"]
    assert set(replacement) == {"after_cue", "before_replacement", "before_query"}
    assert all(s["answers_equal"] and s["saved_arrays_equal"] for s in replacement.values())
    assert vanished["seams"]["clean-1"]["imagine_leaves_checkpoint"]
    assert vanished["seams"]["clean-1"]["refused_act_leaves_checkpoint"]
    assert vanished["timing"]["equal_arrays"] and vanished["work"]["trace_audits"] > 0
    assert vanished["work"]["trace_audit_max_error"] <= 1e-12
    scores = founder["scores"]
    assert set(scores) == set(chamber.ARMS)
    assert scores["random"]["clean-0"]["answer"]["planned"] == inputs.STREAMS
    assert scores["vanished"]["clean-1"]["shuffled_vs_transplanted"]["planned"] == inputs.STREAMS
    assert founder["history_initial_equal"] is True
    gates = founder["gates"]
    assert set(gates["conditions"]) == {c.name for c in inputs.TEST_CONDITIONS}
    assert -1 <= gates["horizon"] <= 2 and isinstance(gates["closure"], bool)
    # the receipt refuses edited scores
    stored = json.loads((out / "summary.json").read_text())
    stored["body"]["founders"][0]["gates"]["horizon"] = 2
    (out / "summary.json").write_text(json.dumps(stored))
    assert not chamber.verify(out)[0]


@pytest.mark.parametrize("bad", [["--seeds", "0", "0"], ["--repeats", "0", "1"]])
def test_invalid_cli_is_rejected_before_creating_an_attempt(tmp_path, bad):
    out = tmp_path / "never"
    with pytest.raises(SystemExit):
        chamber.main(["--out", str(out), *bad])
    assert not out.exists()
