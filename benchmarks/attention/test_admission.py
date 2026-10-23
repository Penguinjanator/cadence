"""Retained failures remain failures; their source and decision custody is checked."""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cadence.receipts import Receipt

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("attention_verify", HERE / "verify.py")
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


@pytest.mark.parametrize("budget", [256, 1024])
def test_retained_admission_source_arithmetic_and_stop_decision(budget: int) -> None:
    path = HERE / "results" / f"admission-{budget}.json"
    ok, message = verifier.verify(path)
    assert ok, message
    body = Receipt.read(path).body
    assert not body["passed"] and not any(f["passed"] for f in body["founders"])
    assert body["candidate_rounds_used"] == 0 and not body["confirmation_seeds_used"]


def test_recomputed_gates_reject_optimistic_flags_and_altered_answers() -> None:
    body = Receipt.read(HERE / "results" / "admission-256.json").body
    changed = deepcopy(body)
    changed["passed"] = True
    assert verifier.check(changed) == "aggregate gate differs"
    changed = deepcopy(body)
    changed["founders"][1]["passed"] = True  # 100% answers after an incomplete training run
    assert verifier.check(changed) == "founder gate differs"
    changed = deepcopy(body)
    changed["founders"][0]["evaluation"]["branches"]["intact"]["answers"][0][0] ^= 1
    assert verifier.check(changed) == "behavioral scores differ"


def test_damping_readback_source_custody_and_qualification_flags() -> None:
    path = HERE / "results" / "damping-readback.json"
    receipt = Receipt.read(path)
    root = HERE.parents[1]
    sources = [(entry["path"], root / entry["path"]) for entry in receipt.source["files"]]
    ok, message = Receipt.verify(path, sources=sources)
    assert ok, message
    for founder in receipt.body["founders"]:
        qualified = max(founder["damped_residual"]) <= founder["tolerance"]
        assert founder["damped_qualified"] == qualified
        assert not qualified
        assert founder["damped_steps"] <= 1024
    assert not receipt.body["all_damped_boundaries_qualified"]
