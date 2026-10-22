"""Verify retained admission decisions, source bytes and receipt digests.

Historical source versions are content-addressed in results/sources. This
checks the actual old bytes rather than pretending the runner never changed.
It verifies custody and arithmetic; it does not rerun behavioral acquisition.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from cadence.receipts import Receipt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("attention_admission", HERE / "admission.py")
assert SPEC is not None and SPEC.loader is not None
admission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = admission
SPEC.loader.exec_module(admission)


def check(body: dict[str, Any]) -> str | None:
    protocol = body["protocol"]
    founders = body["founders"]
    if [f["seed"] for f in founders] != protocol["development_seeds"]:
        return "development founders differ"
    for founder in founders:
        training, evaluation = founder["training"], founder["evaluation"]
        if training["completed"] != (training["rounds"] == protocol["rounds"]):
            return "training completion differs"
        for branch in evaluation["branches"].values():
            if admission.relevance.scores(branch["answers"]) != branch["scores"]:
                return "behavioral scores differ"
        passed = training["completed"] and admission.relevance.acquisition_gate(
            evaluation, protocol["acquisition_floor"])
        if passed != founder["passed"]:
            return "founder gate differs"
    passed = all(f["passed"] for f in founders)
    if body["passed"] != passed:
        return "aggregate gate differs"
    expected = ("admitted_for_development_relevance_tests" if passed else
                "stop_behavioral_candidates; acquisition_failure_not_localized_to_retrieval")
    if body["decision"] != expected:
        return "admission decision differs"
    if body["candidate_rounds_used"] != 0 or body["confirmation_seeds_used"]:
        return "an admission receipt must not report candidate or confirmation work"
    return None


def _resolved_sources(receipt: Receipt) -> list[tuple[str, Path]]:
    """Resolve retained bytes without treating them as the currently installed model."""
    sources = []
    for entry in receipt.source["files"]:
        relative = PurePosixPath(entry["path"])
        digest = entry["sha256"]
        if (relative.is_absolute() or ".." in relative.parts
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid source manifest path or digest")
        snapshot = HERE / "results" / "sources" / (digest + ".txt")
        source = snapshot if snapshot.exists() else ROOT / relative
        sources.append((str(relative), source))
    return sources


def _historical_capacity(checker: Any, parent: Receipt):
    """Resolve only the positive prerequisite that this retained relevance receipt binds.

    The runtime checker still requires current source bytes. This dependency is
    injected only into one private offline checker, and its source hashes must
    be the same hashes recorded by the parent experiment itself.
    """
    parent_sources = {entry["path"]: entry["sha256"] for entry in parent.source["files"]}

    def verified(path: Path) -> Receipt:
        relative = str(path.resolve().relative_to(ROOT.resolve()))
        if parent_sources.get(relative) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("historical capacity file is not bound by its relevance receipt")
        capacity = Receipt.read(path)
        selector = capacity.body.get("protocol_file", "heads-protocol.json")
        if (capacity.kind != "cadence-attention-heads-admission/1"
                or selector != parent.body.get("protocol_file")
                or capacity.digest != parent.body["capacity_reference"]):
            raise ValueError("historical capacity identity or protocol differs")
        checker.admission.PROTOCOL = checker.admission.protocol_path(selector)
        expected = [name for name, _ in checker.admission.sources()]
        if [entry["path"] for entry in capacity.source["files"]] != expected:
            raise ValueError("historical capacity source inventory differs")
        if any(parent_sources.get(entry["path"]) != entry["sha256"]
               for entry in capacity.source["files"]):
            raise ValueError("historical capacity dependencies differ from its relevance receipt")
        valid, message = Receipt.verify(
            path, sources=_resolved_sources(capacity), check=checker.capacity_check)
        if not valid:
            raise ValueError("historical capacity prerequisite: " + message)
        return capacity

    return verified


def _historical_relevance_check(checker: Any, parent: Receipt):
    def arithmetic(body):
        current = checker.verify_capacity
        checker.verify_capacity = _historical_capacity(checker, parent)
        try:
            return checker.check(body)
        finally:
            checker.verify_capacity = current
    return arithmetic


def verify(path: Path) -> tuple[bool, str]:
    receipt = Receipt.read(path)
    try:
        sources = _resolved_sources(receipt)
    except ValueError as error:
        return False, str(error)
    arithmetic = check
    modules = {"cadence-attention-acquisition-repair/1": "repair_admission.py",
               "cadence-attention-content-control/1": "content_control.py",
               "cadence-attention-heads-admission/1": "heads_admission.py",
               "cadence-attention-heads-relevance/1": "heads_relevance.py",
               "cadence-attention-empty-boundary-admission/1": "empty_boundary_admission.py",
               "cadence-attention-heads-capacity-diagnostic/1": "diagnostic_verification.py"}
    if receipt.kind in modules:
        spec = importlib.util.spec_from_file_location(
            "attention_receipt_verifier", HERE / modules[receipt.kind])
        assert spec is not None and spec.loader is not None
        repair = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = repair
        spec.loader.exec_module(repair)
        if receipt.kind == "cadence-attention-empty-boundary-admission/1":
            expected_paths = [name for name, _ in repair.sources()]
            if [entry["path"] for entry in receipt.source["files"]] != expected_paths:
                return False, "empty-boundary source manifest paths differ"
        if receipt.kind in ("cadence-attention-heads-admission/1",
                            "cadence-attention-heads-relevance/1"):
            protocol_file = receipt.body.get("protocol_file", "heads-protocol.json")
            if protocol_file not in ("heads-protocol.json", "heads-longer-protocol.json",
                                     "heads-fixed-read-protocol.json",
                                     "heads-anchored-protocol.json",
                                     "heads-founder-protocol.json",
                                     "heads-founder-extended-protocol.json"):
                return False, "unknown heads protocol"
            owner = repair if receipt.kind.endswith("admission/1") else repair.admission
            owner.PROTOCOL = HERE / protocol_file
            if (protocol_file in ("heads-anchored-protocol.json", "heads-founder-protocol.json",
                                  "heads-founder-extended-protocol.json")
                    and [entry["path"] for entry in receipt.source["files"]]
                    != [name for name, _ in repair.sources()]):
                return False, ("anchored relevance source manifest paths differ"
                               if protocol_file == "heads-anchored-protocol.json" else
                               "founder control source manifest paths differ")
        arithmetic = repair.check
        if (receipt.kind == "cadence-attention-heads-relevance/1"
                and protocol_file in owner.FOUNDER_PROTOCOLS):
            arithmetic = _historical_relevance_check(repair, receipt)
    return Receipt.verify(path, sources=sources, check=arithmetic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    ok, message = verify(args.receipt)
    print(message)
    raise SystemExit(0 if ok else 1)
