"""Untimed CUDA check of fixed phase budgets and exact repeated-unit initialization."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(args):
    import numpy as np
    import pynvml
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    measured = json.loads((args.results / "receipt.json").read_text())
    assert measured["status"] == "complete"
    assert sha(args.results / "sources.zip") == measured["sources_sha256"]
    root = Path(__file__).parent
    assert sha(root / "measurement_sources.zip") == measured["sources_sha256"]
    assert sha(root / "measurement_receipt.json") == sha(args.results / "receipt.json")
    with ZipFile(root / "measurement_sources.zip") as archive:
        archive.extractall(root / "fixture")
    runtime = root / "runtime"
    runtime.mkdir()
    with tarfile.open(root / "fixture/runtime_sources.tar.gz") as archive:
        archive.extractall(runtime, filter="data")
    sys.path[:0] = [str(runtime / "src"), str(runtime)]
    import cadence as cd
    from benchmarks.runtime import baseline_functions

    spec = importlib.util.spec_from_file_location("energy_fixture", root / "fixture/run.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.cd, fixture.np, fixture.torch = cd, np, torch
    fixture.legacy = baseline_functions()
    meter = fixture.Meter(pynvml)
    assert meter.call("nvmlDeviceGetUUID") == measured["environment"]["gpu_uuid"]
    result = {
        "status": "running",
        "purpose": "untimed phase-count and reset validation",
        "sources_sha256": sha(args.out / "sources.zip"),
        "measurement_receipt_sha256": sha(args.results / "receipt.json"),
        "measurement_sources_sha256": measured["sources_sha256"],
        "gpu_uuid": meter.call("nvmlDeviceGetUUID"),
        "pid": os.getpid(),
        "occupancy_before": meter.processes(),
        "rows": [],
        "errors": [],
    }
    try:
        for seed in range(5):
            for arm in ("baseline", "current"):
                case = fixture.Case(seed, arm)
                samples = []
                vectors = []
                initial = case.initial_scale.clone()
                initial_bias = case.initial_bias.clone()
                for _ in range(2):
                    brain = case.brain._with_device_parameters(
                        case.initial_scale, case.initial_bias
                    )
                    learner = cd.Learner(brain, case.output, case.config)
                    phases = []
                    for _ in range(5):
                        states, _report = learner.step(case.drive, case.labels)
                        counts = [states.free.steps, states.nudged.steps, states.opposite.steps]
                        assert counts == [40, 20, 20]
                        phases.append(counts)
                    case.state, case.steps, case.final_brain = states.free, 400, learner.brain
                    torch.cuda.synchronize()
                    vectors.append(np.array(case.snapshot()["values"]))
                    samples.append(
                        {"phase_counts": phases, "actual_total_steps": sum(map(sum, phases))}
                    )
                    assert torch.equal(initial, case.initial_scale)
                    assert torch.equal(initial_bias, case.initial_bias)
                filename = f"snapshot-{seed}-centered_updates-{arm}.npz"
                reference = np.load(args.results / filename, allow_pickle=False)["values"]
                differences = [float(np.abs(x - reference).max()) for x in vectors]
                repeat_gap = float(np.abs(vectors[0] - vectors[1]).max())
                assert max(*differences, repeat_gap) <= 1e-11
                result["rows"].append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "repeats": samples,
                        "max_abs_differences_from_measurement": differences,
                        "max_abs_repeat_difference": repeat_gap,
                        "measurement_snapshot": filename,
                        "measurement_snapshot_sha256": sha(args.results / filename),
                    }
                )
        result["occupancy_after"] = meter.processes()
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["errors"].append(repr(exc))
        raise
    finally:
        (args.out / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "rows": len(result["rows"])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.results, args.out = args.results.resolve(), args.out.resolve()
    if args.worker:
        check(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    with ZipFile(args.out / "sources.zip", "x", compression=ZIP_DEFLATED) as archive:
        archive.write(Path(__file__), "check_phases.py")
        archive.write(args.results / "sources.zip", "measurement_sources.zip")
        archive.write(args.results / "receipt.json", "measurement_receipt.json")
    with tempfile.TemporaryDirectory(prefix="cadence-phase-audit-") as temporary:
        with ZipFile(args.out / "sources.zip") as archive:
            archive.extractall(temporary)
        subprocess.run(
            [
                sys.executable,
                str(Path(temporary) / "check_phases.py"),
                "--worker",
                "--results",
                str(args.results),
                "--out",
                str(args.out),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
