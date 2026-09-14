"""Freeze executable experiment/library sources after a completed comparison.

Refuse replacement: a changed mechanism needs a new evidence bundle.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    archive = HERE / "frozen_sources.zip"
    if archive.exists():
        raise SystemExit(
            "frozen source archive already exists; do not overwrite historical sources"
        )
    paths = [
        *sorted((ROOT / "src/cadence").glob("*.py")),
        HERE / "content.py",
        HERE / "continual.py",
    ]
    manifest = {}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for path in paths:
            name = str(path.relative_to(ROOT))
            data = path.read_bytes()
            manifest[name] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            output.writestr(info, data)
    payload = {
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": manifest,
        "purpose": "historical reproducibility after library integration",
    }
    (HERE / "frozen_sources.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
