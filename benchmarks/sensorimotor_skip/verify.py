"""Verify measured arithmetic and both frozen source bundles, without training."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import cadence as cd
from cadence.receipts import canonical_sha256


def source_manifest(archive: ZipFile) -> dict:
    files = [
        {"path": name, "sha256": hashlib.sha256(archive.read(name)).hexdigest()}
        for name in sorted(archive.namelist())
    ]
    return {"files": files, "manifest_sha256": canonical_sha256(files)}


def verify(path: Path) -> tuple[bool, str]:
    archived = path.parent / "sources.zip"
    receipt = cd.Receipt.read(path)
    with ZipFile(archived) as archive:
        if source_manifest(archive) != receipt.source:
            return False, "experiment source bundle differs"
        if (
            hashlib.sha256(archived.read_bytes()).hexdigest()
            != receipt.body["source_archive_sha256"]
        ):
            return False, "archive digest differs"
        namespace = {"__name__": "frozen_wiring_check"}
        exec(compile(archive.read("run.py"), "frozen:run.py", "exec"), namespace)
        ok, message = cd.Receipt.verify(path, check=namespace["check"])
        if not ok:
            return ok, message
        import json

        reference = json.loads(archive.read("reference/receipt.json"))
        payload = {key: reference[key] for key in ("kind", "body", "source")}
        if reference["digest"] != canonical_sha256(payload):
            return False, "reference receipt digest differs"
        with ZipFile(io.BytesIO(archive.read("reference/sources.zip"))) as original:
            if source_manifest(original) != reference["source"]:
                return False, "reference source bundle differs"
        if receipt.body["reference_receipt_digest"] != reference["digest"]:
            return False, "reference identity differs"
        indexed = {(row["arm"], row["seed"], row["length"]): row for row in receipt.body["rows"]}
        fields = ("training", "probabilities", "parameters", "accuracy", "stochastic_accuracy")
        expected = [
            {
                "arm": row["arm"],
                "seed": row["seed"],
                "length": row["length"],
                "exact_match": all(
                    indexed[row["arm"], row["seed"], row["length"]][key] == row[key]
                    for key in fields
                ),
            }
            for row in reference["body"]["rows"]
        ]
        if expected != receipt.body["reference_reproduction"]:
            return False, "reference reproduction comparison differs"
    return True, message + "; experiment and reference frozen sources agree"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    result, reason = verify(parser.parse_args().receipt)
    print(reason)
    raise SystemExit(0 if result else 1)
