"""Verify receipts against immutable archived source bytes after library changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import content
import continual

from cadence.receipts import Receipt, canonical_sha256

HERE = Path(__file__).resolve().parent


def verify(path: Path, archive: Path = HERE / "frozen_sources.zip") -> tuple[bool, str]:
    receipt = Receipt.read(path)
    check = content.check if receipt.kind == "cadence/content-repair/v1" else continual.check
    valid, message = Receipt.verify(path, check=check)
    if not valid:
        return valid, message
    manifest = json.loads((HERE / "frozen_sources.json").read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        return False, "frozen source archive digest differs"
    with zipfile.ZipFile(archive) as sources:
        names = sources.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest["files"]):
            return False, "frozen source inventory differs"
        for name in names:
            if hashlib.sha256(sources.read(name)).hexdigest() != manifest["files"][name]:
                return False, f"frozen source differs: {name}"
        entries = []
        for item in receipt.source["files"]:
            if item["path"] not in names:
                return False, f"receipt dependency is missing: {item['path']}"
            digest = hashlib.sha256(sources.read(item["path"])).hexdigest()
            entries.append({"path": item["path"], "sha256": digest})
    expected = {"files": entries, "manifest_sha256": canonical_sha256(entries)}
    if receipt.source != expected:
        return False, "receipt dependency hashes differ from frozen source bytes"
    return True, message + "; frozen executable sources agree"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.receipts or [
        HERE / (name + ".json") for name in ("content", "split_pilot", "split", "permuted")
    ]
    for path in paths:
        valid, message = verify(path)
        print(path.name + ": " + message)
        if not valid:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
