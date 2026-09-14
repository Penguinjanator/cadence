"""Verify all recorded outcomes against the immutable executed sources."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import cadence as cd
from cadence.receipts import canonical_sha256
from run import check


def verify():
    root = Path(__file__).resolve().parent / "results"
    path = root / "receipt.json"
    ok, message = cd.Receipt.verify(path, check=check)
    if not ok:
        return False, message
    receipt = cd.Receipt.read(path)
    with ZipFile(root / "sources.zip") as archive:
        entries = [{"path": p, "sha256": sha256(archive.read(p)).hexdigest()}
                   for p in sorted(archive.namelist())]
    source = {"files": entries, "manifest_sha256": canonical_sha256(entries)}
    if source != receipt.source:
        return False, "frozen source archive differs from receipt"
    return True, message + "; frozen source archive agrees"


if __name__ == "__main__":
    ok, reason = verify()
    print(reason)
    raise SystemExit(0 if ok else 1)
