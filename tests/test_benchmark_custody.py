"""Frozen evidence must include subpackages and never overwrite older source bytes."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest


def test_source_bundle_includes_nested_modules_and_preserves_prior_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.freeze import BASELINE_PATH, source_bundle

    members = (
        "src/cadence/__init__.py",
        "src/cadence/circuits/__init__.py",
        "src/cadence/circuits/assembly.py",
        "benchmarks/runtime.py",
        "benchmarks/README.md",
        "pyproject.toml",
        BASELINE_PATH,
    )
    for name in members:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    first = source_bundle(tmp_path)[0][1]
    original = first.read_bytes()
    assert source_bundle(tmp_path)[0][1] == first
    with tarfile.open(first) as archive:
        assert set(archive.getnames()) == set(members)
    (tmp_path / "src/cadence/circuits/assembly.py").write_text("# new source\n")
    second = source_bundle(tmp_path)[0][1]
    assert second != first
    assert first.read_bytes() == original
