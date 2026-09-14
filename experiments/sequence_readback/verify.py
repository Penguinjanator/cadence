"""Verify immutable source copies and recompute every reported test aggregate.

This verifier does not rerun training. Per-token losses and decisions are the
measured observables; tests/test_sequence.py separately checks causal write
ordering, no mutation during reads, and a hand-computed attention reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cadence.receipts import Receipt


def verify(path: Path, *, allow_partial: bool = False) -> dict:
    receipt = Receipt.read(path)
    body = receipt.body
    frozen = path.parent / "source"
    sources = [(e["path"], frozen / e["path"]) for e in receipt.source["files"]]
    ok, message = Receipt.verify(path, sources=sources)
    if not ok:
        raise ValueError(message)
    seeds = [row["seed"] for row in body["outcomes"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed")
    expected = body["design"]["seeds"]
    if seeds != expected[: len(seeds)] or (not allow_partial and seeds != expected):
        raise ValueError("scheduled seed outcomes are missing or reordered")
    summary = {"seeds": seeds, "patch": {}, "mlp": {}, "bigram": body["bigram"]}
    rows = []
    for row in body["outcomes"]:
        artifact = row["per_token_artifact"]
        file = path.parent / artifact["path"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("per-token artifact hash mismatch")
        with np.load(file, allow_pickle=False) as data:
            targets = data["test_targets"]
            for family in ("patch", "mlp"):
                if set(row[family]) != set(body["design"]["modes"]):
                    raise ValueError("missing architecture control")
                for mode, arm in row[family].items():
                    prefix = mode if family == "patch" else "mlp_" + mode
                    readings = [(prefix, arm["test"])]
                    if family == "patch":
                        if set(arm["cache"]) != {"raw", "legacy_stale", "centered"}:
                            raise ValueError("missing memory control")
                        for name, cache in arm["cache"].items():
                            readings.append((prefix + "_cache_" + name, cache["test"]))
                            candidate = min(cache["search"], key=lambda s: s["validation"]["bpc"])
                            if (
                                candidate["temperature"] != cache["temperature"]
                                or candidate["mixture"] != cache["mixture"]
                            ):
                                raise ValueError(
                                    "cache configuration was not selected on validation"
                                )
                            if candidate["validation"]["bpc"] != cache["validation_bpc"]:
                                raise ValueError("wrong selected validation metric")
                            for split in ("validation_selectivity", "test_selectivity"):
                                error = cache[split]["matched_cosine_max_abs_error"]
                                if error > 1e-10:
                                    raise ValueError("structure-matched conventional cache differs")
                    best = min(arm["history"], key=lambda h: h["validation"]["bpc"])
                    if best["epoch"] != arm["best_epoch"]:
                        raise ValueError("epoch was not selected on validation")
                    for key, reported in readings:
                        nll, hit = data[key + "_nll"], data[key + "_hit"]
                        if nll.shape != targets.shape or hit.shape != targets.shape:
                            raise ValueError("test rows differ across comparisons")
                        if not np.isfinite(nll).all() or (nll < 0).any() or hit.dtype != np.bool_:
                            raise ValueError("invalid per-token reading")
                        computed = {
                            "bpc": float(nll.mean()),
                            "accuracy": float(hit.mean()),
                            "tokens": int(nll.size),
                        }
                        if reported != computed:
                            raise ValueError(f"aggregate differs for {key}")
                        rows.append({"seed": row["seed"], "arm": family + "/" + key, **computed})
        for family in ("patch", "mlp"):
            for mode, arm in row[family].items():
                summary[family].setdefault(mode, []).append(arm["test"]["bpc"])
    summary["per_seed"] = rows
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = verify(args.receipt, allow_partial=args.allow_partial)
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
