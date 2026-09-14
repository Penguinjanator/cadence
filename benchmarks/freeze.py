"""Content-addressed source bundles so later core edits cannot invalidate old experiments."""

from __future__ import annotations

import gzip
import io
import subprocess
import tarfile
from hashlib import sha256
from pathlib import Path

BASELINE_REVISION = "9f859bfd8f4df52aed5460dbd84306da02a61e34"
BASELINE_PATH = "benchmarks/baseline/brain-9f859bf.py"


def baseline_source(root: Path) -> str:
    frozen = root / BASELINE_PATH
    if frozen.exists():
        return frozen.read_text()
    return subprocess.check_output(
        ["git", "show", f"{BASELINE_REVISION}:src/cadence/brain.py"], cwd=root, text=True
    )


def source_bundle(root: Path) -> list[tuple[str, Path]]:
    """Freeze the cadence package, experiment scripts, and original baseline source."""
    paths = sorted((root / "src/cadence").rglob("*.py"))
    paths += sorted((root / "benchmarks").glob("*.py"))
    paths += [root / "pyproject.toml", root / "benchmarks/README.md"]
    files = {str(path.relative_to(root)): path.read_bytes() for path in paths}
    files[BASELINE_PATH] = baseline_source(root).encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    packed = gzip.compress(stream.getvalue(), mtime=0)
    digest = sha256(packed).hexdigest()
    relative = f"benchmarks/frozen/sources-{digest[:16]}.tar.gz"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != packed:
            raise ValueError("source bundle digest collision")
    else:
        path.write_bytes(packed)
    return [(relative, path)]
