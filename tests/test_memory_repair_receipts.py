"""Adversarial checks against selective reporting and false retention summaries."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "experiments/memory_repair"


def module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_content_receipt_cannot_drop_overflow_or_report_aliases_as_solved() -> None:
    check = module("content").check
    body = json.loads((HERE / "content.json").read_text())["body"]
    assert check(body) is None
    missing = copy.deepcopy(body)
    missing["rows"] = [r for r in missing["rows"] if r["condition"] != "overflow"]
    assert check(missing) is not None
    false = copy.deepcopy(body)
    row = next(r for r in false["rows"] if r["condition"] == "aliased")
    row["accuracy"], row["correct"] = 1.0, row["test_n"]
    assert check(false) is not None


def test_continual_receipt_cannot_hide_forgetting_or_extra_training() -> None:
    check = module("continual").check
    body = json.loads((HERE / "split.json").read_text())["body"]
    assert check(body) is None
    for metric in ("forgetting", "final_average"):
        false = copy.deepcopy(body)
        false["rows"][0][metric] += 0.1
        assert check(false) is not None
    false = copy.deepcopy(body)
    replay = next(r for r in false["rows"] if r["arm"] == "patch_replay")
    replay["extra_training_examples"] -= 32
    replay["training_examples"] -= 32
    assert check(false) is not None
    assert check(body | {"rows": body["rows"][:-1]}) is not None


def test_frozen_source_verifier_detects_archive_mutation(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(HERE))
    verify = module("verify").verify
    for name in ("content", "split_pilot", "split", "permuted"):
        assert verify(HERE / f"{name}.json")[0]
    changed = bytearray((HERE / "frozen_sources.zip").read_bytes())
    changed[len(changed) // 2] ^= 1
    bad = tmp_path / "changed.zip"
    bad.write_bytes(changed)
    assert not verify(HERE / "split.json", archive=bad)[0]
