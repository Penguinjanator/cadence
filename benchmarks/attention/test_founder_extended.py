"""A host-time extension retains the founder point, failed evidence and costs."""

from __future__ import annotations

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
    "heads_founder_extended_tests", Path(__file__).with_name("heads_relevance.py"))
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)
admission, task = subject.admission, subject.task


def protocol():
    return json.loads(admission.protocol_path(admission.FOUNDER_EXTENDED_PROTOCOL).read_text())


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
        "training": {"completed": True, "rounds": 512, "wrong_counts": [1] * 512,
                     "work": {"action_refusals": 0, "lesson_refusals": 0}},
        "initial": deepcopy(evaluation["all"]), "evaluation": evaluation,
        "initial_parameters": "a" * 64, "final_parameters": "b" * 64,
        "evaluation_parameters": "b" * 64, "parameters_preserved": True,
        "schedule_sha256": canonical_sha256(task.draw(seed, 512, 16)),
        "resolved": brain.describe(), "neuron_model": brain.brain.neuron_model.to_dict(),
        "learning_config": brain.learner.config.to_dict(),
        "numerical_policy": admission.numerical_settings(brain),
        "neurons": brain.connectome.n, "synapses": brain.connectome.synapses,
    }


@pytest.fixture
def fixture(monkeypatch):
    data = protocol()
    monkeypatch.setattr(admission, "PROTOCOL",
                        admission.protocol_path(admission.FOUNDER_EXTENDED_PROTOCOL))
    prior = SimpleNamespace(digest=data["prior"]["receipt_digest"], body={
        "cumulative_host_seconds": data["prior"]["receipt_cumulative_host_seconds"]})
    monkeypatch.setattr(admission, "founder_prior", lambda path, protocol: prior)
    body = {
        "protocol": data, "protocol_file": admission.FOUNDER_EXTENDED_PROTOCOL,
        "runs": [result(arm, seed, data) for arm in ("exposed", "supplied_reads")
                 for seed in data["development_seeds"]],
        "capacity_admitted": True,
        "decision": "capacity_admitted; selection_and_causal_gates_pending",
        "candidate_rounds_used": 0, "confirmation_seeds_used": [],
        "founder_defaults": admission.founder_defaults(),
        "budget_scope": "additional_founder_capacity_stage",
        "prior_reference": prior.digest, "previous_host_seconds": 756.0517235864615,
        "elapsed_host_seconds": 12.0, "cumulative_host_seconds": 768.0517235864615,
    }
    return SimpleNamespace(protocol=data, prior=prior, body=body)


def test_extension_changes_only_host_guards_and_evidence_declarations():
    base = json.loads(admission.protocol_path(admission.FOUNDER_PROTOCOL).read_text())
    extended = protocol()
    mutable = {"schema", "scope", "source", "budget", "prior", "relevance_stage", "host_extension"}
    assert {k: v for k, v in extended.items() if k not in mutable} == {
        k: v for k, v in base.items() if k not in mutable}
    for arm in ("exposed", "supplied_reads", *subject.ARMS):
        old = admission.make_brain(arm, 1, base)
        new = admission.make_brain(arm, 1, extended)
        assert old.brain.neuron_model == new.brain.neuron_model
        assert old.learner.config == new.learner.config
        assert old.describe() == new.describe()
        for name in ("efficacy", "bias", "log_gain"):
            np.testing.assert_array_equal(getattr(old.brain, name), getattr(new.brain, name))
    assert extended["budget"]["training_seconds_per_arm"] == 240
    assert extended["budget"]["development_total_host_seconds"] == 600
    assert extended["relevance_stage"]["training_seconds_per_arm"] == 240
    assert extended["relevance_stage"]["development_total_host_seconds"] == 3000


def test_extended_protocol_is_explicitly_declared_and_old_source_inventory_is_unchanged():
    assert admission.FOUNDER_PROTOCOLS == (
        "heads-founder-protocol.json", "heads-founder-extended-protocol.json")
    prior = Receipt.read(admission.HERE / "results" / "founder-capacity-development.json")
    base_sources = admission.sources(admission.protocol_path(admission.FOUNDER_PROTOCOL))
    assert [entry["path"] for entry in prior.source["files"]] == [name for name, _ in base_sources]
    extended = dict(admission.sources(admission.protocol_path(admission.FOUNDER_EXTENDED_PROTOCOL)))
    assert len(extended) == len(admission.sources(
        admission.protocol_path(admission.FOUNDER_EXTENDED_PROTOCOL)))
    assert "benchmarks/attention/heads-founder-protocol.json" in extended
    assert "benchmarks/attention/results/founder-capacity-development.json" in extended
    assert "benchmarks/attention/results/founder-capacity-extended.json" not in extended


def test_actual_negative_founder_prior_verifies_against_archived_source_bytes():
    data = protocol()
    receipt = admission.founder_prior(
        admission.HERE / "results" / data["prior"]["receipt_file"], data)
    assert receipt.kind == "cadence-attention-heads-admission/1"
    assert not receipt.body["capacity_admitted"]
    assert "passed" not in receipt.body
    assert admission.founder_previous(data, receipt) == 756.0517235864615
    assert [r["training"]["rounds"] for r in receipt.body["runs"]] == [512, 512, 419, 370]


def test_positive_or_wrong_prior_cannot_be_relabelled_as_the_required_failure(monkeypatch):
    data = protocol()
    prior = SimpleNamespace(
        kind=data["prior"]["receipt_kind"], digest=data["prior"]["receipt_digest"],
        body={"capacity_admitted": True, "protocol_file": admission.FOUNDER_PROTOCOL,
              "cumulative_host_seconds": data["prior"]["receipt_cumulative_host_seconds"]})
    monkeypatch.setattr(admission, "load", lambda *args: SimpleNamespace(
        verify=lambda path: (True, "verified")))
    monkeypatch.setattr(admission.Receipt, "read", lambda path: prior)
    with pytest.raises(ValueError, match="failed prior"):
        admission.founder_prior(Path("prior.json"), data)
    prior.body["capacity_admitted"] = False
    prior.body["protocol_file"] = admission.FOUNDER_EXTENDED_PROTOCOL
    with pytest.raises(ValueError, match="failed prior"):
        admission.founder_prior(Path("prior.json"), data)


def test_extended_capacity_has_a_new600_second_deadline_without_recharging_history(
    fixture, monkeypatch,
):
    monkeypatch.setattr(admission, "sources", lambda: [])
    monkeypatch.setattr(admission.time, "monotonic", lambda: 100.0)
    seen = []

    def trial(arm, seed, data, deadline):
        seen.append((arm, seed, deadline))
        return result(arm, seed, data)

    monkeypatch.setattr(admission, "trial", trial)
    body = admission.run(Path("prior.json")).body
    assert seen == [(arm, seed, 700.0) for arm in ("exposed", "supplied_reads") for seed in (0, 1)]
    assert body["previous_host_seconds"] == body["cumulative_host_seconds"] == 756.0517235864615
    assert body["prior_reference"] == fixture.prior.digest
    assert admission.check(body) is None
    with pytest.raises(ValueError, match="requires"):
        admission.run()


@pytest.mark.parametrize("change", ["diagnostic", "previous", "prior", "partial", "refusal",
                                    "sweep_cap", "held_unrun", "default_damping"])
def test_extension_cannot_recharge_or_waive_existing_gates(fixture, change):
    body = deepcopy(fixture.body)
    if change == "diagnostic":
        body["protocol"]["prior"]["failure_diagnostic_host_seconds"] = 94.9148939170409
    elif change == "previous":
        body["previous_host_seconds"] -= 501.3375700454424
    elif change == "prior":
        body["prior_reference"] = "f" * 64
    elif change == "partial":
        body["runs"][2]["training"]["rounds"] = 419
    elif change == "refusal":
        body["runs"][2]["training"]["work"]["action_refusals"] = 1
    elif change == "sweep_cap":
        body["runs"][2]["learning_config"]["free_steps"] = 4096
    elif change == "held_unrun":
        body["runs"][2]["evaluation"]["held_out"]["status"] = "unrun_budget"
    else:
        body["founder_defaults"]["learning_damping"] = 0
    assert admission.check(body) is not None


def test_relevance_uses_a_separate3000_seconds_and_keeps_every_cell(fixture, tmp_path, monkeypatch):
    capacity = Receipt.build("cadence-attention-heads-admission/1", fixture.body)
    retained = tmp_path / "results" / fixture.protocol["relevance_stage"]["capacity_receipt_file"]
    capacity.write(retained)
    monkeypatch.setattr(subject, "HERE", tmp_path)
    monkeypatch.setattr(subject, "verify_capacity", lambda path: capacity)
    monkeypatch.setattr(subject, "sources", lambda: [])
    clock = iter([0.0])
    monkeypatch.setattr(subject.time, "monotonic", lambda: next(clock, 3001.0))
    monkeypatch.setattr(subject, "trial", lambda *args, **kwargs:
                        pytest.fail("expired stage solved"))
    body = subject.run(retained).body
    assert len(body["runs"]) == 10
    assert all(r["status"] == "unrun_budget" for r in body["runs"])
    assert body["capacity_host_seconds"] == fixture.body["cumulative_host_seconds"]
    assert body["cumulative_host_seconds"] == fixture.body["cumulative_host_seconds"] + 3001.0
    assert body["previous_host_seconds"] == 756.0517235864615
    assert body["confirmation_seeds_used"] == [] and not body["passed"]
    assert subject.check(body) is None


def test_positive_relevance_prerequisite_uses_exact_extended_sources(
    fixture, tmp_path, monkeypatch,
):
    runtime = tmp_path / "runtime.py"
    runtime.write_text("runtime=1\n")
    monkeypatch.setattr(admission, "sources", lambda: [("runtime.py", runtime)])
    path = tmp_path / "capacity.json"
    Receipt.build("cadence-attention-heads-admission/1", fixture.body,
                  admission.sources()).write(path)
    assert subject.verify_capacity(path).body["capacity_admitted"]
    runtime.write_text("runtime=2\n")
    with pytest.raises(ValueError, match="source"):
        subject.verify_capacity(path)


def test_future_positive_capacity_is_bound_only_by_relevance_sources(fixture, monkeypatch):
    monkeypatch.setattr(admission, "sources", lambda: [])
    paths = dict(subject.sources())
    assert "benchmarks/attention/results/founder-capacity-extended.json" in paths
