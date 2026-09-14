"""Verify source custody, numerical conformance, sensor arithmetic and the fixed schedule."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np

RUNTIME_SHA256 = "2bd9e9056d35055f25c455f2ccecf4d553ae9a77179b149bfedde0433e3575c7"
WORKLOADS = {"fixed_steps", "centered_updates", "residual_cold"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def same(actual, expected):
    assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def verify(path: Path, body: dict | None = None) -> dict:
    body = json.loads(path.read_text()) if body is None else body
    root = path.parent
    assert body["status"] == "complete" and not body["errors"]
    assert body["runtime_source_sha256"] == RUNTIME_SHA256
    source_bytes = (root / "sources.zip").read_bytes()
    assert sha(source_bytes) == body["sources_sha256"]
    pilot_mode = body["mode"] == "pilot"
    with ZipFile(io.BytesIO(source_bytes)) as source:
        assert sha(source.read("runtime_sources.tar.gz")) == RUNTIME_SHA256
        if not pilot_mode:
            pilot_bytes = source.read("pilot.json")
            assert sha(pilot_bytes) == body["pilot_sha256"]
            pilot = json.loads(pilot_bytes)
            assert pilot["status"] == "complete" and not pilot["errors"]
            assert pilot["locked_counts"] == body["locked_counts"]
            assert pilot["environment"]["gpu_uuid"] == body["environment"]["gpu_uuid"]
    seeds = {0} if pilot_mode else set(range(5))
    scheduled = {(s, w) for s in seeds for w in WORKLOADS}
    cells = body["cells"]
    assert len(cells) == len(scheduled)
    assert {(c["seed"], c["workload"]) for c in cells} == scheduled
    p = body["protocol"]
    assert p["width"] == 128 and p["batch"] == 64 and p["dtype"] == "float64"
    assert p["fixed_steps"] == 60 and p["updates"] == 5
    assert p["free_steps"] == 40 and p["nudged_steps"] == 20
    assert p["learning_tolerance"] is None and p["residual_tolerance"] == 1e-8
    assert p["residual_budget"] == 128 and p["residual_chunk"] == 4
    assert p["warmup_units"] == 3 and p["minimum_seconds"] == 10
    assert p["idle_seconds"] == 3 and p["target_seconds"] == 15
    assert p["snapshot_conformance_tolerance"] == 1e-11
    pid = body["environment"]["pid"]
    assert not body["environment"]["initial_occupancy"]
    summaries = {w: [] for w in WORKLOADS}
    for cell in cells:
        assert set(cell["arms"]) == {"baseline", "current"}
        vectors = {}
        for arm, row in cell["arms"].items():
            assert len(row["warmup_seconds"]) == 3
            data = (root / row["snapshot_file"]).read_bytes()
            assert sha(data) == row["snapshot_sha256"]
            vectors[arm] = np.load(io.BytesIO(data), allow_pickle=False)["values"]
            assert np.isfinite(vectors[arm]).all()
            if cell["workload"] == "residual_cold":
                assert row["host_residual_max"] <= 1e-8
                assert 0 <= row["steps"] <= 128
            elif cell["workload"] == "fixed_steps":
                assert row["steps"] == 60
            else:
                assert row["steps"] == 400
            if pilot_mode:
                assert len(row["unit_seconds"]) == 3
                same(row["median_unit_seconds"], float(np.median(row["unit_seconds"])))
                continue
            count = body["locked_counts"][cell["workload"]]
            assert isinstance(count, int) and count > 0
            work = row["work"]
            assert work["result"]["units"] == count
            assert work["result"]["last_unit_steps"] == row["steps"]
            assert row["meets_minimum_duration"] and work["seconds"] >= 10
            for interval in (row["idle_before"], work, row["idle_after"]):
                same(interval["seconds"], (interval["end_ns"] - interval["start_ns"]) / 1e9)
                assert interval["energy_after_mj"] > interval["energy_before_mj"]
                same(
                    interval["raw_device_joules"],
                    (interval["energy_after_mj"] - interval["energy_before_mj"]) / 1000,
                )
                for occupancy in (interval["occupancy_before"], interval["occupancy_after"]):
                    assert occupancy and all(x["pid"] == pid for x in occupancy)
                samples = [
                    s
                    for s in body["samples"]
                    if interval["start_ns"] <= s["time_ns"] <= interval["end_ns"]
                ]
                assert len(samples) >= interval["seconds"] * 2
                assert samples[0]["time_ns"] - interval["start_ns"] < 1e9
                assert interval["end_ns"] - samples[-1]["time_ns"] < 1e9
                for sample in samples:
                    assert all(
                        sample[k] is not None
                        for k in (
                            "energy_mj",
                            "power_mw",
                            "temperature_c",
                            "graphics_clock_mhz",
                            "memory_clock_mhz",
                        )
                    )
                assert all(
                    a["energy_mj"] <= b["energy_mj"]
                    for a, b in zip(samples, samples[1:], strict=False)
                )
                assert all(
                    b["time_ns"] - a["time_ns"] < 1e9
                    for a, b in zip(samples, samples[1:], strict=False)
                )
            assert row["idle_before"]["end_ns"] <= work["start_ns"]
            assert work["end_ns"] <= row["idle_after"]["start_ns"]
            for idle in (row["idle_before"], row["idle_after"]):
                assert idle["seconds"] >= 3
            idle_rate = float(
                np.mean(
                    [
                        x["raw_device_joules"] / x["seconds"]
                        for x in (row["idle_before"], row["idle_after"])
                    ]
                )
            )
            same(row["adjacent_idle_watts"], idle_rate)
            same(
                row["idle_corrected_device_joules"],
                work["raw_device_joules"] - idle_rate * work["seconds"],
            )
            same(row["seconds_per_unit"], work["seconds"] / count)
            same(row["raw_device_joules_per_unit"], work["raw_device_joules"] / count)
        gap = float(np.max(np.abs(vectors["baseline"] - vectors["current"])))
        same(cell["max_abs_state_parameter_difference"], gap)
        assert gap <= 1e-11 and cell["steps_match"]
        a, b = cell["arms"]["baseline"], cell["arms"]["current"]
        assert a["steps"] == b["steps"]
        if pilot_mode:
            expected = math.ceil(15 / min(x["median_unit_seconds"] for x in (a, b)))
            assert body["locked_counts"][cell["workload"]] == expected
        else:
            first, second = (a, b) if cell["seed"] % 2 == 0 else (b, a)
            assert first["idle_after"]["end_ns"] < second["idle_before"]["start_ns"]
            summaries[cell["workload"]].append(
                {
                    "seed": cell["seed"],
                    "baseline_seconds": a["seconds_per_unit"],
                    "current_seconds": b["seconds_per_unit"],
                    "baseline_joules": a["raw_device_joules_per_unit"],
                    "current_joules": b["raw_device_joules_per_unit"],
                    "baseline_corrected_joules": a["idle_corrected_device_joules"]
                    / body["locked_counts"][cell["workload"]],
                    "current_corrected_joules": b["idle_corrected_device_joules"]
                    / body["locked_counts"][cell["workload"]],
                    "time_ratio": a["seconds_per_unit"] / b["seconds_per_unit"],
                    "raw_energy_ratio": a["raw_device_joules_per_unit"]
                    / b["raw_device_joules_per_unit"],
                    "corrected_energy_ratio": a["idle_corrected_device_joules"]
                    / b["idle_corrected_device_joules"],
                }
            )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--phases", type=Path)
    args = parser.parse_args()
    summary = verify(args.receipt)
    if args.phases:
        verify_phases(args.phases, args.receipt)
    print(json.dumps({"verified": True, "summary": summary}, indent=2, sort_keys=True))


def verify_phases(path: Path, measurement_path: Path) -> None:
    body = json.loads(path.read_text())
    measured_bytes = measurement_path.read_bytes()
    measured = json.loads(measured_bytes)
    assert body["status"] == "complete" and not body["errors"]
    assert body["measurement_receipt_sha256"] == sha(measured_bytes)
    assert body["measurement_sources_sha256"] == measured["sources_sha256"]
    assert body["gpu_uuid"] == measured["environment"]["gpu_uuid"]
    source_bytes = (path.parent / "sources.zip").read_bytes()
    assert sha(source_bytes) == body["sources_sha256"]
    with ZipFile(io.BytesIO(source_bytes)) as source:
        assert sha(source.read("measurement_receipt.json")) == sha(measured_bytes)
        assert sha(source.read("measurement_sources.zip")) == measured["sources_sha256"]
    assert len(body["rows"]) == 10
    assert {(r["seed"], r["arm"]) for r in body["rows"]} == {
        (s, a) for s in range(5) for a in ("baseline", "current")
    }
    for key in ("occupancy_before", "occupancy_after"):
        assert all(p["pid"] == body["pid"] for p in body[key])
    assert body["occupancy_after"]
    for row in body["rows"]:
        assert len(row["repeats"]) == 2
        for repeat in row["repeats"]:
            assert repeat["phase_counts"] == [[40, 20, 20]] * 5
            assert repeat["actual_total_steps"] == 400
        assert 0 <= row["max_abs_repeat_difference"] <= 1e-11
        assert len(row["max_abs_differences_from_measurement"]) == 2
        assert all(0 <= x <= 1e-11 for x in row["max_abs_differences_from_measurement"])
        data = (measurement_path.parent / row["measurement_snapshot"]).read_bytes()
        assert sha(data) == row["measurement_snapshot_sha256"]


if __name__ == "__main__":
    main()
