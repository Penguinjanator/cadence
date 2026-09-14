from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/memory_repair"


@pytest.mark.parametrize("script", ["content", "continual", "drift"])
def test_experiments_cannot_overwrite_existing_results(tmp_path, script: str) -> None:
    out = tmp_path / "existing.json"
    out.write_text("existing scientific outcome\n")
    result = subprocess.run(
        [sys.executable, str(HERE / f"{script}.py"), "--out", str(out)],
        cwd=ROOT,
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "already exists" in result.stderr
    assert out.read_text() == "existing scientific outcome\n"


def test_content_specification_follows_output_path(tmp_path) -> None:
    out = tmp_path / "new_run/content.json"
    historical = (HERE / "content_spec.json").read_bytes()
    result = subprocess.run(
        [sys.executable, str(HERE / "content.py"), "--out", str(out)],
        cwd=ROOT,
        env=os.environ
        | {
            "PYTHONPATH": str(ROOT / "src"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.with_suffix(".spec.json").exists()
    assert (HERE / "content_spec.json").read_bytes() == historical


@pytest.mark.parametrize("archive", ["frozen_runtime.zip", "drift_sources.zip"])
def test_frozen_package_imports_with_recursive_circuits(tmp_path, archive: str) -> None:
    with zipfile.ZipFile(HERE / archive) as source:
        assert "src/cadence/circuits/__init__.py" in source.namelist()
        source.extractall(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import cadence; import cadence.circuits; "
            "assert cadence.ContentMemory; assert cadence.ReservoirReplay",
        ],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(tmp_path / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
