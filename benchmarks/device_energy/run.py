"""Source-bound isolated CUDA device-energy measurements, with a locked pilot budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from zipfile import ZIP_DEFLATED, ZipFile

RUNTIME_SHA256 = "2bd9e9056d35055f25c455f2ccecf4d553ae9a77179b149bfedde0433e3575c7"
WORKLOADS = ("fixed_steps", "centered_updates", "residual_cold")
ARMS = ("baseline", "current")
SEEDS = tuple(range(5))
TARGET_SECONDS = 15.0
MINIMUM_SECONDS = 10.0
IDLE_SECONDS = 3.0
SAMPLE_SECONDS = 0.2


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, body: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class Meter:
    def __init__(self, nvml):
        self.nvml = nvml
        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByIndex(0)
        self.samples: list[dict] = []
        self.errors: list[dict] = []
        self.label = "setup"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def call(self, name: str, *args):
        try:
            return getattr(self.nvml, name)(self.handle, *args)
        except Exception as exc:
            self.errors.append(
                {
                    "time_ns": time.monotonic_ns(),
                    "api": name,
                    "error": repr(exc),
                    "label": self.label,
                }
            )
            raise

    def energy(self) -> int:
        return int(self.call("nvmlDeviceGetTotalEnergyConsumption"))

    def processes(self) -> list[dict]:
        result = []
        for kind in ("Compute", "Graphics"):
            for p in self.call(f"nvmlDeviceGet{kind}RunningProcesses"):
                result.append(
                    {"kind": kind, "pid": p.pid, "used_gpu_memory_bytes": p.usedGpuMemory}
                )
        if any(p["pid"] != os.getpid() for p in result):
            raise RuntimeError(f"competing GPU processes: {result}")
        return result

    def sample(self):
        while not self.stop.is_set():
            row = {"time_ns": time.monotonic_ns(), "label": self.label}
            for key, function, args in (
                ("energy_mj", "nvmlDeviceGetTotalEnergyConsumption", ()),
                ("power_mw", "nvmlDeviceGetPowerUsage", ()),
                ("temperature_c", "nvmlDeviceGetTemperature", (0,)),
                ("graphics_clock_mhz", "nvmlDeviceGetClockInfo", (0,)),
                ("memory_clock_mhz", "nvmlDeviceGetClockInfo", (2,)),
            ):
                try:
                    row[key] = self.call(function, *args)
                except Exception:
                    row[key] = None
            self.samples.append(row)
            self.stop.wait(SAMPLE_SECONDS)

    def interval(self, label: str, operation) -> dict:
        self.label = label
        occupancy_before = self.processes()
        torch.cuda.synchronize()
        start_ns = time.monotonic_ns()
        before_mj = self.energy()
        result = operation()
        torch.cuda.synchronize()
        after_mj = self.energy()
        end_ns = time.monotonic_ns()
        occupancy_after = self.processes()
        if after_mj <= before_mj:
            raise RuntimeError("energy counter did not advance monotonically")
        seconds = (end_ns - start_ns) / 1e9
        return {
            "label": label,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "seconds": seconds,
            "energy_before_mj": before_mj,
            "energy_after_mj": after_mj,
            "raw_device_joules": (after_mj - before_mj) / 1000,
            "occupancy_before": occupancy_before,
            "occupancy_after": occupancy_after,
            "result": result,
        }


class Case:
    def __init__(self, seed: int, arm: str):
        graph = cd.layered(128, 128, 16, density=1, seed=seed)
        model = cd.learning_neuron_model()
        raw = cd.Brain(graph, model)
        scale = raw.efficacy / cd.row_mass(raw)
        self.brain = cd.Brain(graph, model, efficacy=scale, backend="torch", device="cuda")
        if arm == "baseline":
            self.brain.residual = MethodType(legacy["residual"], self.brain)
            for name in ("_activation", "_synaptic_input", "contrast_tensors"):
                setattr(self.brain._torch, name, MethodType(legacy[name], self.brain._torch))
        kernel = self.brain._torch
        self.initial_scale = kernel.scale.clone()
        self.initial_bias = kernel.bias_param.clone()
        self.output = graph.populations["output"]
        self.config = cd.LearnerConfig(free_steps=40, nudged_steps=20, tolerance=None)
        rng = np.random.default_rng(seed)
        self.drive = np.zeros((64, graph.n))
        self.drive[:, :128] = rng.uniform(0.1, 0.9, (64, 128))
        self.labels = rng.integers(0, 16, size=64)
        self.steps = None
        self.state = None
        self.final_brain = None

    def unit(self, workload: str):
        if workload == "fixed_steps":
            self.state = self.brain.settle_batch(self.drive, steps=60)
            self.steps = self.state.steps
            self.final_brain = self.brain
        elif workload == "residual_cold":
            result = self.brain.equilibrate(self.drive, budget=128, chunk=4, tolerance=1e-8)
            self.state, self.steps = result.state, result.steps
            self.final_brain = self.brain
        else:
            # Reset the five-update experiment for each unit. This device parameter
            # reset and learner initialization are included in the measured interval.
            brain = self.brain._with_device_parameters(self.initial_scale, self.initial_bias)
            learner = cd.Learner(brain, self.output, self.config)
            for _ in range(5):
                phases, _report = learner.step(self.drive, self.labels)
            self.state, self.steps, self.final_brain = phases.free, 400, learner.brain

    def units(self, workload: str, count: int):
        for _ in range(count):
            self.unit(workload)
        return {"units": count, "last_unit_steps": self.steps}

    def snapshot(self) -> dict:
        b, state = self.final_brain, self.state
        snapshot = np.concatenate((state.v.ravel(), state.activation.ravel(), b.efficacy, b.bias))
        residual = cd.Brain.residual(b, self.drive, state, on_device=False)
        return {
            "values": snapshot.tolist(),
            "steps": self.steps,
            "host_residual_max": float(residual.max()),
            "initial_scale_unchanged": bool(
                torch.equal(self.initial_scale, self.brain._torch.scale)
            )
            if self.steps != 400
            else None,
        }


def environment(meter: Meter) -> dict:
    n = meter.nvml
    return {
        "utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "nvml_python": subprocess.check_output(
            [sys.executable, "-m", "pip", "show", "nvidia-ml-py"], text=True
        ),
        "gpu_name": meter.call("nvmlDeviceGetName"),
        "gpu_uuid": meter.call("nvmlDeviceGetUUID"),
        "driver": n.nvmlSystemGetDriverVersion(),
        "power_limit_mw": meter.call("nvmlDeviceGetPowerManagementLimit"),
        "pid": os.getpid(),
        "dtype": "float64",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "initial_occupancy": meter.processes(),
        "cuda_synchronization": "both interval endpoints",
        "energy_scope": "GPU device only; excludes host CPU/memory/whole-system energy",
        "energy_api": "nvmlDeviceGetTotalEnergyConsumption (millijoules since driver reload)",
        "sensor_documentation": "https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html",
    }


def execute(args):
    global torch, np, cd, legacy
    import numpy as np
    import torch

    runtime = Path(__file__).parent / "runtime"
    runtime.mkdir()
    with tarfile.open(Path(__file__).parent / "runtime_sources.tar.gz") as archive:
        archive.extractall(runtime, filter="data")
    sys.path[:0] = [str(runtime / "src"), str(runtime)]
    import pynvml

    import cadence as cd
    from benchmarks.runtime import baseline_functions

    assert Path(cd.__file__).resolve().is_relative_to(runtime)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    legacy = baseline_functions()
    meter = Meter(pynvml)
    body = {
        "status": "running",
        "mode": args.worker,
        "runtime_source_sha256": RUNTIME_SHA256,
        "sources_sha256": digest(args.out / "sources.zip"),
        "environment": environment(meter),
        "protocol": {
            "seeds": list(SEEDS),
            "workloads": list(WORKLOADS),
            "arms": list(ARMS),
            "width": 128,
            "batch": 64,
            "dtype": "float64",
            "effective_row_mass": 1.0,
            "fixed_steps": 60,
            "updates": 5,
            "free_steps": 40,
            "nudged_steps": 20,
            "learning_tolerance": None,
            "residual_tolerance": 1e-8,
            "residual_budget": 128,
            "residual_chunk": 4,
            "warmup_units": 3,
            "target_seconds": TARGET_SECONDS,
            "minimum_seconds": MINIMUM_SECONDS,
            "idle_seconds": IDLE_SECONDS,
            "power_sample_seconds": SAMPLE_SECONDS,
            "arm_order": "alternates by seed",
            "interval_repetitions": 1,
            "snapshot_and_graph_setup_timed": False,
            "centered_unit_reset_timed": True,
            "centered_unit_reset": "restore device parameters; fresh learner; five updates",
            "idle_correction": "raw joules minus elapsed seconds times mean adjacent idle watts",
            "snapshot_conformance_tolerance": 1e-11,
        },
        "cells": [],
        "errors": meter.errors,
        "samples": meter.samples,
    }
    meter.thread.start()
    started = time.monotonic()
    try:
        if args.worker == "measure":
            pilot = json.loads((Path(__file__).parent / "pilot.json").read_text())
            assert pilot["status"] == "complete" and pilot["mode"] == "pilot"
            assert pilot["runtime_source_sha256"] == RUNTIME_SHA256
            assert pilot["environment"]["gpu_uuid"] == body["environment"]["gpu_uuid"]
            body["pilot_sha256"] = digest(Path(__file__).parent / "pilot.json")
            body["locked_counts"] = pilot["locked_counts"]
        for seed in [0] if args.worker == "pilot" else SEEDS:
            for workload in WORKLOADS:
                cell = {"seed": seed, "workload": workload, "arms": {}}
                snapshots = {}
                for arm in ARMS if seed % 2 == 0 else ARMS[::-1]:
                    if time.monotonic() - started > 35 * 60:
                        raise RuntimeError("bounded benchmark deadline exceeded")
                    case = Case(seed, arm)
                    meter.label = f"warmup/{seed}/{workload}/{arm}"
                    warm = []
                    for _ in range(3):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        case.unit(workload)
                        torch.cuda.synchronize()
                        warm.append(time.perf_counter() - t0)
                    if args.worker == "pilot":
                        durations = []
                        for _ in range(3):
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            case.unit(workload)
                            torch.cuda.synchronize()
                            durations.append(time.perf_counter() - t0)
                        row = {
                            "warmup_seconds": warm,
                            "unit_seconds": durations,
                            "median_unit_seconds": float(np.median(durations)),
                        }
                    else:
                        label = f"{seed}/{workload}/{arm}"
                        before = meter.interval(
                            label + "/idle_before", lambda: time.sleep(IDLE_SECONDS)
                        )
                        count = body["locked_counts"][workload]
                        work = meter.interval(
                            label + "/work",
                            lambda case=case, workload=workload, count=count: case.units(
                                workload, count
                            ),
                        )
                        after = meter.interval(
                            label + "/idle_after", lambda: time.sleep(IDLE_SECONDS)
                        )
                        idle_watts = float(
                            np.mean(
                                [x["raw_device_joules"] / x["seconds"] for x in (before, after)]
                            )
                        )
                        row = {
                            "warmup_seconds": warm,
                            "idle_before": before,
                            "work": work,
                            "idle_after": after,
                            "adjacent_idle_watts": idle_watts,
                            "idle_corrected_device_joules": work["raw_device_joules"]
                            - idle_watts * work["seconds"],
                            "seconds_per_unit": work["seconds"] / count,
                            "raw_device_joules_per_unit": work["raw_device_joules"] / count,
                            "meets_minimum_duration": work["seconds"] >= MINIMUM_SECONDS,
                        }
                        if not row["meets_minimum_duration"]:
                            body["errors"].append({"error": "short interval", "label": label})
                    snapshots[arm] = case.snapshot()
                    snapshot_file = args.out / f"snapshot-{seed}-{workload}-{arm}.npz"
                    np.savez_compressed(snapshot_file, values=np.array(snapshots[arm]["values"]))
                    row["snapshot_file"] = snapshot_file.name
                    row["snapshot_sha256"] = digest(snapshot_file)
                    row["steps"] = snapshots[arm]["steps"]
                    row["host_residual_max"] = snapshots[arm]["host_residual_max"]
                    cell["arms"][arm] = row
                    print(
                        json.dumps(
                            {
                                "seed": seed,
                                "workload": workload,
                                "arm": arm,
                                "seconds_per_unit": row.get(
                                    "seconds_per_unit", row.get("median_unit_seconds")
                                ),
                                "joules_per_unit": row.get("raw_device_joules_per_unit"),
                            }
                        ),
                        flush=True,
                    )
                gap = float(
                    np.max(
                        np.abs(
                            np.array(snapshots["baseline"]["values"])
                            - np.array(snapshots["current"]["values"])
                        )
                    )
                )
                cell["max_abs_state_parameter_difference"] = gap
                cell["steps_match"] = (
                    snapshots["baseline"]["steps"] == snapshots["current"]["steps"]
                )
                body["cells"].append(cell)
                if gap > 1e-11 or not cell["steps_match"]:
                    raise RuntimeError("numerical/work conformance failed")
                if (
                    workload == "residual_cold"
                    and max(x["host_residual_max"] for x in snapshots.values()) > 1e-8
                ):
                    raise RuntimeError("host residual tolerance failed")
                write(args.out / "receipt.json", body)
        if args.worker == "pilot":
            body["locked_counts"] = {
                c["workload"]: math.ceil(
                    TARGET_SECONDS / min(a["median_unit_seconds"] for a in c["arms"].values())
                )
                for c in body["cells"]
            }
            body["estimated_measurement_seconds_including_idle"] = 5 * sum(
                body["locked_counts"][c["workload"]]
                * sum(a["median_unit_seconds"] for a in c["arms"].values())
                + 4 * IDLE_SECONDS
                for c in body["cells"]
            )
        body["status"] = "complete" if not body["errors"] else "invalid"
    except Exception as exc:
        body["status"] = "failed"
        body["errors"].append({"error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        meter.stop.set()
        meter.thread.join(timeout=2)
        body["total_seconds"] = time.monotonic() - started
        write(args.out / "receipt.json", body)
    print(
        json.dumps(
            {
                "status": body["status"],
                "locked_counts": body.get("locked_counts"),
                "estimated_seconds": body.get("estimated_measurement_seconds_including_idle"),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-sources", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--mode", choices=("pilot", "measure"), default="pilot")
    parser.add_argument("--worker", choices=("pilot", "measure"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.worker:
        execute(args)
        return
    assert digest(args.runtime_sources) == RUNTIME_SHA256
    if args.mode == "measure" and args.pilot is None:
        parser.error("measurement requires a completed pilot receipt")
    args.out.mkdir(parents=True, exist_ok=False)
    sources = args.out / "sources.zip"
    with ZipFile(sources, "x", compression=ZIP_DEFLATED) as bundle:
        bundle.write(Path(__file__), "run.py")
        bundle.write(args.runtime_sources, "runtime_sources.tar.gz")
        if args.pilot:
            bundle.write(args.pilot, "pilot.json")
    with tempfile.TemporaryDirectory(prefix="cadence-energy-") as temporary:
        with ZipFile(sources) as bundle:
            bundle.extractall(temporary)
        result = subprocess.run(
            [
                sys.executable,
                str(Path(temporary) / "run.py"),
                "--worker",
                args.mode,
                "--out",
                str(args.out),
            ],
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            check=False,
        )
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
