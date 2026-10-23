"""Independently recompute the retained acquisition repair and its confirmation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from cadence.receipts import Receipt

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("attention_repair_evidence", HERE / "verify.py")
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


@pytest.mark.parametrize("name", [
    "repair-development-precheck", "repair-development", "repair-confirmation",
])
def test_retained_repair_has_exact_sources_and_recomputed_decisions(name: str) -> None:
    path = HERE / "results" / (name + ".json")
    ok, message = verifier.verify(path)
    assert ok, message


def test_verifier_only_replay_preserves_every_parameter_answer_and_work_reading() -> None:
    first = Receipt.read(HERE / "results" / "repair-development-precheck.json")
    replay = Receipt.read(HERE / "results" / "repair-development.json")
    assert first.body["runs"] == replay.body["runs"]
    assert first.body["selected_recipe"] == replay.body["selected_recipe"] == "normalized-qualified"
    assert first.body["passed"] and replay.body["passed"]


def test_confirmation_uses_the_frozen_development_recipe_and_all_twenty_founders() -> None:
    development = Receipt.read(HERE / "results" / "repair-development.json")
    confirmation = Receipt.read(HERE / "results" / "repair-confirmation.json")
    body = confirmation.body
    assert body["development_reference"] == {
        "digest": development.digest, "selected_recipe": development.body["selected_recipe"],
    }
    assert body["protocol"] == development.body["protocol"]
    assert [r["seed"] for r in body["runs"]] == list(range(100, 120))
    for result in body["runs"]:
        assert result["policy"] == "wrong" and result["recipe"] == "normalized-qualified"
        training = result["training"]
        assert training["completed"] and training["rounds"] == 512
        assert training["work"]["action_refusals"] == training["work"]["lesson_refusals"] == 0
        scores = result["evaluation"]["branches"]["intact"]["scores"]
        assert scores["accuracy"] == scores["unequal_fact_accuracy"] == 1.0
        assert scores["per_query_accuracy"] == [1.0, 1.0]
        for phase in ("initial", "evaluation"):
            assert result[phase]["completed"] and result[phase]["original_preserved"]
            assert result[phase]["work"]["action_refusals"] == 0
    initial = [r["initial"]["branches"]["intact"]["scores"]["accuracy"] for r in body["runs"]]
    assert np.mean(initial) == pytest.approx(0.5125)
    assert np.mean([r["uniform_random"]["scores"]["accuracy"] for r in body["runs"]]) == 0.5
    # Include the original development BEFORE its verifier-only replay in actual host work.
    first = Receipt.read(HERE / "results" / "repair-development-precheck.json")
    elapsed = sum(r.body["elapsed_host_seconds"] for r in (first, development, confirmation))
    assert elapsed < body["protocol"]["total_seconds"]
