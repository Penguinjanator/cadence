"""Close an early six-file source snapshot over the complete immutable base package.

Never replaces a frozen file. Newer runs already copy the complete package in
run.sources(); this helper also binds an explicit archive manifest to the final
receipt. Use only once the schedule has finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from cadence.receipts import Receipt, canonical_json, canonical_sha256

ROOT = Path(__file__).resolve().parents[2]


def close_archive(receipt_path: Path, base: str = "9f859bf") -> Path:
    receipt = Receipt.read(receipt_path)
    if [row["seed"] for row in receipt.body["outcomes"]] != receipt.body["design"]["seeds"]:
        raise ValueError("cannot close an unfinished schedule")
    archive = receipt_path.parent / "source"
    manifest_path = receipt_path.parent / "archive_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("archive already closed")
    for entry in receipt.source["files"]:
        if hashlib.sha256((archive / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("a frozen source differs from the receipt")
    revision = subprocess.check_output(["git", "rev-parse", base], cwd=ROOT, text=True).strip()
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "src/cadence"], cwd=ROOT, text=True
    ).splitlines()
    for relative in paths:
        if not relative.endswith(".py"):
            continue
        target = archive / relative
        if not target.exists():
            content = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    files = [
        {
            "path": str(path.relative_to(archive)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(archive.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]
    body = {
        "kind": "cadence/sequence-source-closure/v1",
        "base_revision": revision,
        "receipt_digest": receipt.digest,
        "files": files,
    }
    manifest_path.write_text(canonical_json(body | {"digest": canonical_sha256(body)}) + "\n")
    return manifest_path


def verify_archive(receipt_path: Path) -> None:
    path = receipt_path.parent / "archive_manifest.json"
    body = json.loads(path.read_text())
    digest = body.pop("digest")
    if digest != canonical_sha256(body):
        raise ValueError("archive manifest digest mismatch")
    if body["receipt_digest"] != Receipt.read(receipt_path).digest:
        raise ValueError("archive is not bound to this final receipt")
    names = {entry["path"] for entry in body["files"]}
    for required in (
        "src/cadence/__init__.py",
        "src/cadence/circuits/__init__.py",
        "src/cadence/circuits/assembly.py",
        "src/cadence/circuits/deliberation.py",
    ):
        if required not in names:
            raise ValueError(f"archive lacks required package member {required}")
    for entry in body["files"]:
        content = (receipt_path.parent / "source" / entry["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("archive source hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        close_archive(args.receipt)
    verify_archive(args.receipt)
    print("complete source archive and final receipt agree")


if __name__ == "__main__":
    main()
