"""Loop chamber guards: the pattern, the continuation, the controls, the gates and the receipt."""

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


chamber = load_script("loop_rhythm")


@pytest.fixture(autouse=True)
def quiet():
    warnings.simplefilter("ignore")


@pytest.fixture(scope="module")
def protocol():
    return chamber.load_protocol()


def test_the_protocol_declares_the_mechanism_the_contract_and_fresh_seeds(protocol):
    declared, digest = protocol
    assert digest == hashlib.sha256(chamber.PROTOCOL_PATH.read_bytes()).hexdigest()
    assert declared["schema"] == chamber.SCHEMA and declared["pattern"]["period"] == 4
    assert declared["copy"]["decay"] > 0 and declared["copy"]["amplitude"] > 0
    assert declared["ngram"]["order"] == declared["pattern"]["period"]
    assert not set(declared["seeds"]["development"]) & set(declared["seeds"]["confirmation"])
    assert declared["gates"]["share"] <= len(declared["seeds"]["confirmation"])


def test_the_pattern_and_its_continuation_keep_period_and_phase():
    rows = chamber.pattern(4, 12)
    assert rows.tolist() == [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    prime = chamber.pattern(4, 6, phase=1)
    assert prime.tolist() == [0, 1, 0, 0, 0, 1]
    assert chamber.continuation(4, prime, 6).tolist() == [0, 0, 0, 1, 0, 0]
    x = chamber.observation(chamber.ONSET)
    assert x.shape == (1, chamber.INPUTS) and x[0, 1] == 1.0 and x[0, 2] == 1.0


def test_the_ngram_control_continues_the_pattern_from_any_phase():
    table = chamber.NGram(4)
    table.teach(chamber.pattern(4, 256))
    for phase in range(4):
        prime = chamber.pattern(4, 16, phase)
        assert table.play(prime, 32) == chamber.continuation(4, prime, 32).tolist()


def test_scores_and_prime_dependence_read_the_plays():
    truth = chamber.pattern(4, 8)
    score = chamber.score_play([1, 0, 0, 0, 1, 0, None, 0], truth)
    assert score == {"onset_rate": 0.25, "agreement": 1.0, "refusals": 1}
    assert chamber.correlation([0, 0, 0, 0], [1, 0, 1, 0]) is None
    assert chamber.correlation([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_the_contract_leaves_the_heard_stream_alone_in_the_copy(protocol):
    declared, _ = protocol
    brain = chamber.make_brain(0, declared, "copy")
    work = chamber.Work()
    rows = chamber.pattern(4, 8)
    expected = None
    for t in range(len(rows) - 1):
        chamber.hear(brain, "copy", int(rows[t]), work)
        one_hot = np.eye(2)[[int(rows[t])]]
        expected = (
            one_hot * (1 - brain.efference.decay)
            if expected is None
            else brain.efference.decay * expected + (1 - brain.efference.decay) * one_hot
        )
        np.testing.assert_allclose(brain.efference.trace, expected)
    # kept: the brain's own answer is the newest entry of the copy
    answer = chamber.hear(brain, "copy", int(rows[-1]), work, keep=True)
    assert answer is not None
    decay = brain.efference.decay
    heard = decay * expected + (1 - decay) * np.eye(2)[[int(rows[-1])]]
    kept = decay * heard + (1 - decay) * np.eye(2)[[answer]]
    np.testing.assert_allclose(brain.efference.trace, kept)
    assert work.acts == len(rows) and work.lessons == 0


def test_a_right_answer_teaches_nothing_and_a_lesson_reads_what_the_act_read(protocol, monkeypatch):
    declared, _ = protocol
    brain = chamber.make_brain(0, declared, "copy")
    work = chamber.Work()
    rows = chamber.pattern(4, 12)
    seen: dict = {}
    real_act, real_teach = chamber.act, chamber.teach

    def spy_act(b, x, w):
        seen["act_input"] = chamber.memory_state(b)
        answer = real_act(b, x, w)
        seen["act_output"] = chamber.memory_state(b)
        return answer

    def spy_teach(b, x, label, w):
        seen["teach_input"] = chamber.memory_state(b)
        return real_teach(b, x, label, w)

    def same(a, b):
        assert set(a) == set(b)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    monkeypatch.setattr(chamber, "act", spy_act)
    monkeypatch.setattr(chamber, "teach", spy_teach)
    taught = 0
    for t in range(len(rows) - 1):
        seen.clear()
        given = work.lessons
        answer = chamber.hear(brain, "copy", int(rows[t]), work, label=int(rows[t + 1]))
        lesson = work.lessons - given
        assert (lesson == 0) == (answer == int(rows[t + 1]))
        if lesson:
            taught += 1
            # the lesson read the drive the act read
            same(seen["teach_input"]["working_memory"], seen["act_input"]["working_memory"])
            same(seen["teach_input"]["efference"], seen["act_input"]["efference"])
        after = chamber.memory_state(brain)
        # the trace stands as the act left it; the copy as the act read it (own command taken back)
        same(after["working_memory"], seen["act_output"]["working_memory"])
        same(after["efference"], seen["act_input"]["efference"])
    assert taught > 0 and work.lessons == taught and work.acts == len(rows) - 1


def test_the_arms_differ_only_as_declared(protocol):
    declared, _ = protocol
    copy, own, bare, frozen = (
        chamber.make_brain(0, declared, arm) for arm in ("copy", "own", "nocopy", "frozen")
    )
    assert copy.efference is not None and own.efference is not None and frozen.efference is not None
    assert bare.efference is None
    assert copy.efference.decay == declared["copy"]["decay"]
    np.testing.assert_array_equal(copy.brain.efficacy, own.brain.efficacy)
    np.testing.assert_array_equal(copy.brain.efficacy, frozen.brain.efficacy)
    assert copy.working_memory.amplitude == declared["recipe"]["trace_amplitude"]


def test_a_small_run_scores_every_arm_and_its_receipt_verifies(tmp_path):
    out = tmp_path / "run"
    assert chamber.main(["--out", str(out), "--seeds", "0", "--passes", "1"]) == 0
    valid, reason = chamber.verify(out)
    assert valid, reason
    body = json.loads((out / "summary.json").read_text())["body"]
    assert body["frozen_protocol"] is False and body["declaration"]["overrides"] == {"passes": 1}
    arms = {r["arm"]: r for r in body["runs"]}
    assert set(arms) == set(chamber.ARMS)
    period = body["protocol"]["pattern"]["period"]
    for run in arms.values():
        assert len(run["plays"]) == period and all(
            len(p) == body["protocol"]["play"]["rows"] for p in run["plays"]
        )
    assert arms["hold"]["mean_onset_rate"] == 0.0 and arms["hold"][
        "mean_agreement"
    ] == pytest.approx(1 - 1 / period)
    assert arms["ngram"]["mean_agreement"] == 1.0 and arms["ngram"]["prime_dependent"] is True
    assert "teaching" in arms["copy"] and "teaching" not in arms["frozen"]
    for arm in chamber.BRAIN_ARMS:
        assert 0.0 <= arms[arm]["watching"]["agreement"] <= 1.0
        assert arms[arm]["watching"]["rows"] + arms[arm]["watching"]["refusals"] == 255
    gates = body["gates"]
    assert set(gates["copy"]) >= {"fires", "follows", "prime_dependent", "learned", "mean_watching"}
    assert gates["ngram"]["prime_dependent"] == 1
    stored = json.loads((out / "summary.json").read_text())
    stored["body"]["gates"]["passed"] = True
    (out / "summary.json").write_text(json.dumps(stored))
    assert not chamber.verify(out)[0]


@pytest.mark.parametrize("bad", [["--seeds", "0", "0"], ["--arms", "copy", "teacher"]])
def test_invalid_cli_is_rejected_before_creating_an_attempt(tmp_path, bad):
    out = tmp_path / "never"
    with pytest.raises(SystemExit):
        chamber.main(["--out", str(out), *bad])
    assert not out.exists()
