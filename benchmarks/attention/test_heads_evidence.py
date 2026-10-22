"""Read retained scientific decisions and custody without issuing any new solve."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from cadence.receipts import Receipt

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
