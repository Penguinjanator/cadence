"""Read retained scientific decisions and custody without issuing any new solve."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cadence import NeuralGraph
from cadence.receipts import Receipt, canonical_json, canonical_sha256

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CAPACITY = ("heads-capacity-initial.json", "heads-capacity-corrected.json",
            "heads-capacity-longer.json", "heads-capacity-fixed-read.json")
SPEC = importlib.util.spec_from_file_location("heads_evidence_verifier", HERE / "verify.py")
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


@pytest.mark.parametrize("partial", [False, True])
def test_empty_boundary_dispatch_requires_the_complete_dependency_inventory(tmp_path, partial):
    sources = [("benchmarks/attention/heads.py", HERE / "heads.py")] if partial else []
    path = tmp_path / "unbound.json"
    Receipt.build("cadence-attention-empty-boundary-admission/1", {}, sources).write(path)
    valid, message = verifier.verify(path)
    assert not valid
    assert message == "empty-boundary source manifest paths differ"


def test_anchored_relevance_dispatch_rejects_a_missing_dependency_inventory(tmp_path):
    path = tmp_path / "unbound-relevance.json"
    Receipt.build("cadence-attention-heads-relevance/1",
                  {"protocol_file": "heads-anchored-protocol.json"}, []).write(path)
    valid, message = verifier.verify(path)
    assert not valid
    assert message == "anchored relevance source manifest paths differ"


@pytest.mark.parametrize("stage", ["admission", "relevance"])
@pytest.mark.parametrize("protocol", ["heads-founder-protocol.json",
                                      "heads-founder-extended-protocol.json"])
def test_founder_dispatch_requires_the_complete_dependency_inventory(tmp_path, stage, protocol):
    path = tmp_path / "unbound-founder.json"
    Receipt.build(f"cadence-attention-heads-{stage}/1",
                  {"protocol_file": protocol}, []).write(path)
    valid, message = verifier.verify(path)
    assert not valid
    assert message == "founder control source manifest paths differ"


def test_retained_empty_read_repair_verifies_without_admitting_learned_heads():
    path = RESULTS / "empty-boundary-development.json"
    valid, message = verifier.verify(path)
    assert valid, message
    body = Receipt.read(path).body
    assert body["masked_historical_reproduced"] is True
    assert body["occupied_query_overlap_identical"] is True
    assert [r["training"]["rounds"] for r in body["runs"]] == [42, 21, 140, 279]
    assert body["passed"] is False
    assert body["candidate_rounds_used"] == 0
    assert body["confirmation_seeds_used"] == []


def test_retained_founder_control_preserves_the_host_capped_failure():
    path = RESULTS / "founder-capacity-development.json"
    valid, message = verifier.verify(path)
    assert valid, message
    body = Receipt.read(path).body
    assert [r["training"]["rounds"] for r in body["runs"]] == [512, 512, 419, 370]
    assert all(not r["training"]["work"]["action_refusals"]
               and not r["training"]["work"]["lesson_refusals"] for r in body["runs"])
    assert [r["evaluation"]["train"]["scores"]["exact_pair_accuracy"]
            for r in body["runs"]] == [1.0, 1.0, 1.0, 0.875]
    assert body["capacity_admitted"] is False
    assert body["candidate_rounds_used"] == 0
    assert body["confirmation_seeds_used"] == []


def receipt(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def test_all_four_capacity_receipts_pass_the_public_source_and_arithmetic_dispatch() -> None:
    for name in CAPACITY:
        valid, message = verifier.verify(RESULTS / name)
        assert valid, (name, message)


def test_content_and_numerical_diagnostic_use_the_same_public_dispatch() -> None:
    for name in ("content-results-development.json", "heads-capacity-diagnostic.json"):
        valid, message = verifier.verify(RESULTS / name)
        assert valid, (name, message)


def test_positive_capacity_and_negative_learned_readers_preserve_their_separate_gates() -> None:
    capacity_name = "founder-capacity-extended.json"
    relevance_name = "heads-relevance-founder-extended.json"
    for name in (capacity_name, relevance_name):
        valid, message = verifier.verify(RESULTS / name)
        assert valid, (name, message)
    capacity, relevance = receipt(capacity_name), receipt(relevance_name)
    assert capacity["body"]["capacity_admitted"] is True
    body = relevance["body"]
    assert body["capacity_reference"] == capacity["digest"]
    assert body["passed"] is False and body["selected"] is None
    assert len(body["runs"]) == 10 and not any(run["passed"] for run in body["runs"])
    assert body["candidate_rounds_used"] == 1 and body["confirmation_seeds_used"] == []
    assert body["cumulative_host_seconds"] == (
        capacity["body"]["cumulative_host_seconds"] + body["elapsed_host_seconds"])


def test_offline_history_does_not_relax_the_current_source_prerequisite(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "current_relevance_gate", HERE / "heads_relevance.py")
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)
    declared = runtime.admission.protocol_path("heads-founder-extended-protocol.json")
    installed = runtime.admission.sources(declared)
    changed = tmp_path / "changed-heads.py"
    changed.write_bytes((HERE / "heads.py").read_bytes() + b"\n# another installed source\n")
    monkeypatch.setattr(runtime.admission, "sources", lambda: [
        (name, changed if name == "benchmarks/attention/heads.py" else path)
        for name, path in installed])

    def no_neural_phase(*args, **kwargs):
        pytest.fail("offline evidence verification attempted a neural phase")

    monkeypatch.setattr(NeuralGraph, "settle_batch", no_neural_phase)
    monkeypatch.setattr(NeuralGraph, "_equilibrate_local", no_neural_phase)
    capacity = RESULTS / "founder-capacity-extended.json"
    with pytest.raises(ValueError, match="source manifest differs"):
        runtime.verify_capacity(capacity)
    valid, message = verifier.verify(RESULTS / "heads-relevance-founder-extended.json")
    assert valid, message
    with pytest.raises(ValueError, match="source manifest differs"):
        runtime.verify_capacity(capacity)


def test_historical_parent_capacity_still_requires_its_actual_archived_bytes(tmp_path, monkeypatch):
    capacity = Receipt.read(RESULTS / "founder-capacity-extended.json")
    resolve = verifier._resolved_sources
    changed = tmp_path / "corrupt-snapshot.txt"
    changed.write_text("corrupt retained dependency\n")

    def corrupted(receipt):
        paths = resolve(receipt)
        if receipt.digest == capacity.digest:
            return [(name, changed if name == "benchmarks/attention/heads.py" else path)
                    for name, path in paths]
        return paths

    monkeypatch.setattr(verifier, "_resolved_sources", corrupted)
    valid, _ = verifier.verify(RESULTS / "heads-relevance-founder-extended.json")
    assert not valid


def test_a_redigested_relevance_receipt_cannot_substitute_another_capacity_reference(tmp_path):
    raw = deepcopy(receipt("heads-relevance-founder-extended.json"))
    raw["body"]["capacity_reference"] = "f" * 64
    raw["digest"] = canonical_sha256({name: raw[name] for name in ("kind", "body", "source")})
    changed = tmp_path / "rebound.json"
    changed.write_text(canonical_json(raw) + "\n")
    valid, _ = verifier.verify(changed)
    assert not valid


def test_prior_chain_charges_all_failed_controls_and_the_diagnostic_once() -> None:
    initial, corrected, longer, fixed = [receipt(name) for name in CAPACITY]
    for previous, following in zip((initial, corrected, longer),
                                   (corrected, longer, fixed), strict=True):
        assert following["body"]["prior_reference"] == previous["digest"]
    diagnostic = receipt("heads-capacity-diagnostic.json")["body"]
    charge = diagnostic["host_work"]["diagnostic_charged_seconds"]
    assert charge == pytest.approx(30.51993266819045, rel=0, abs=1e-12)
    assert corrected["body"]["previous_host_seconds"] == initial["body"]["elapsed_host_seconds"]
    corrected_total = (corrected["body"]["previous_host_seconds"]
                       + corrected["body"]["elapsed_host_seconds"])
    assert longer["body"]["previous_host_seconds"] == corrected_total + charge
    longer_total = (longer["body"]["previous_host_seconds"]
                    + longer["body"]["elapsed_host_seconds"])
    assert fixed["body"]["previous_host_seconds"] == longer_total
    total = fixed["body"]["previous_host_seconds"] + fixed["body"]["elapsed_host_seconds"]
    assert total == pytest.approx(304.1098535011988, rel=0, abs=1e-10)
    assert total < 600


def test_all_exposed_lives_acquire_but_every_supplied_control_refuses_before_completion() -> None:
    expected_rounds = ((30, 24), (30, 13), (83, 42), (42, 21))
    for name, refused_rounds in zip(CAPACITY, expected_rounds, strict=True):
        body = receipt(name)["body"]
        assert [(r["arm"], r["seed"]) for r in body["runs"]] == [
            (arm, seed) for arm in ("exposed", "supplied_reads") for seed in (0, 1)]
        for row in body["runs"][:2]:
            assert row["training"]["completed"] and row["training"]["rounds"] == 512
            assert row["training"]["work"]["action_refusals"] == 0
            assert row["training"]["work"]["lesson_refusals"] == 0
            scores = row["evaluation"]["train"]["scores"]
            assert scores["exact_pair_accuracy"] == scores["unequal_pair_accuracy"] == 1.0
            assert scores["per_slot_accuracy"] == scores["per_query_accuracy"] == [1.0, 1.0]
            assert row["acquired"]
        for row, rounds in zip(body["runs"][2:], refused_rounds, strict=True):
            assert not row["training"]["completed"] and row["training"]["rounds"] == rounds
            assert row["training"]["stop"] == "refused_prefix"
            assert row["training"]["work"]["action_refusals"] == 1
            assert not row["acquired"]
        assert not body["capacity_admitted"]
        assert body["decision"] == "supplied_reads_acquisition_failed"
        assert body["candidate_rounds_used"] == 0 and body["confirmation_seeds_used"] == []


def test_protocol_variants_preserve_the_original_acquisition_and_candidate_gates() -> None:
    protocols = [json.loads((HERE / name).read_text()) for name in (
        "heads-protocol.json", "heads-longer-protocol.json", "heads-fixed-read-protocol.json")]
    for protocol in protocols:
        assert protocol["gates"]["exposed_acquisition_floor"] == 0.95
        assert protocol["gates"]["content_use_floor"] == 0.95
        assert protocol["gates"]["held_out_exact_pair_floor"] == 0.90
        assert protocol["maximum_candidate_rounds"] == 2
        assert protocol["candidate_order"] == ["agreement", "linear"]
        assert protocol["budget"]["development_total_host_seconds"] == 600
    assert protocols[0]["brain"]["learning_free_steps"] == 1024
    assert protocols[1]["brain"]["learning_free_steps"] == 2048
    assert protocols[2]["brain"]["learning_free_steps"] == 2048
    assert protocols[2]["capacity_control"] == "held-address-agreement"
    assert receipt(CAPACITY[-1])["body"]["protocol_file"] == "heads-fixed-read-protocol.json"
