"""The saved scientific receipt must reject altered accounting and weaker work."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("device_energy_verify", ROOT / "verify.py")
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


@pytest.mark.parametrize("mode", ["pilot", "results"])
def test_saved_receipt(mode):
    VERIFY.verify(ROOT / mode / "receipt.json")


def test_phase_counts_and_reset_audit():
    VERIFY.verify_phases(ROOT / "phases/receipt.json", ROOT / "results/receipt.json")


@pytest.mark.parametrize(
    "mutation",
    ["count", "duplicate", "tolerance", "source", "gap", "joules", "idle", "occupancy", "duration"],
)
def test_mutated_measurement_is_rejected(mutation):
    path = ROOT / "results/receipt.json"
    body = copy.deepcopy(json.loads(path.read_text()))
    cell = body["cells"][0]
    row = cell["arms"]["baseline"]
    if mutation == "count":
        body["locked_counts"]["fixed_steps"] = 1
    elif mutation == "duplicate":
        body["cells"][1] = copy.deepcopy(cell)
    elif mutation == "tolerance":
        body["protocol"]["residual_tolerance"] = 1e-5
    elif mutation == "source":
        body["sources_sha256"] = "0" * 64
    elif mutation == "gap":
        cell["max_abs_state_parameter_difference"] = 1e-3
    elif mutation == "joules":
        row["work"]["raw_device_joules"] *= 0.9
    elif mutation == "idle":
        row["idle_corrected_device_joules"] *= 0.9
    elif mutation == "occupancy":
        row["work"]["occupancy_before"][0]["pid"] += 1
    else:
        row["work"]["seconds"] = 9
    with pytest.raises(AssertionError):
        VERIFY.verify(path, body)
