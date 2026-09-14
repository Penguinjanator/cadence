"""Matched pre-change/current runtime measurements; no task-quality or energy claim.

See benchmarks/README.md for the CPU and accelerator commands.
The baseline functions are loaded from their immutable git revision. Both arms use the
same shapes, seeds, precision, warm starts, stopping tolerance, and centered update rule.
"""

from __future__ import annotations

import argparse
import ast
import cProfile
import importlib
import io
import platform
import pstats
import subprocess
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch

import cadence as cd

if __package__:
    from .freeze import BASELINE_REVISION, baseline_source, source_bundle
else:
    from freeze import BASELINE_REVISION, baseline_source, source_bundle

ROOT = Path(__file__).resolve().parents[1]


def baseline_functions() -> dict[str, object]:
    source = baseline_source(ROOT)
    tree = ast.parse(source)
    namespace = dict(vars(importlib.import_module("cadence.brain")))
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        names = (
            {"residual"}
            if node.name == "Brain"
            else (
                {"_activation", "_synaptic_input", "contrast_tensors"}
                if node.name == "_TorchKernel"
                else set()
            )
        )
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name in names:
                compiled = compile(
                    ast.Module(body=[member], type_ignores=[]), f"baseline:{member.name}", "exec"
                )
                exec(compiled, namespace)
                result[member.name] = namespace[member.name]
    assert len(result) == 4
    return result


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    legacy = baseline_functions()
    cells = []
    for seed in range(5):
        graph = cd.layered(args.width, args.width, 16, density=1, seed=seed)
        model = cd.learning_neuron_model()
        raw = cd.Brain(graph, model)
        scale = raw.efficacy / cd.row_mass(raw)
        rng = np.random.default_rng(seed)
        drive = np.zeros((args.batch, graph.n))
        drive[:, : args.width] = rng.uniform(0.1, 0.9, (args.batch, args.width))
        changed = drive.copy()
        changed[:, : args.width] += 1e-3
        labels = rng.integers(0, 16, size=args.batch)
        tolerance = 1e-6 if args.device == "mps" else 1e-8
        config = cd.LearnerConfig(free_steps=40, nudged_steps=20, tolerance=None)

        def build(arm: str, graph=graph, model=model, scale=scale) -> cd.Brain:
            brain = cd.Brain(graph, model, efficacy=scale, backend="torch", device=args.device)
            if arm == "baseline":
                brain.residual = MethodType(legacy["residual"], brain)
                for name in ("_activation", "_synaptic_input", "contrast_tensors"):
                    setattr(brain._torch, name, MethodType(legacy[name], brain._torch))
            return brain

        for workload in (
            "fixed_steps",
            "residual_cold",
            "residual_unchanged",
            "residual_small_change",
            "centered_updates",
        ):
            arms = {}
            snapshots = {}
            for arm in ["baseline", "current"] if seed % 2 == 0 else ["current", "baseline"]:
                elapsed = []
                for _repeat in range(args.warmups + args.repeats):
                    brain = build(arm)
                    learner = cd.Learner(brain, graph.populations["output"], config)
                    warm = (
                        brain.settle_batch(drive, steps=80)
                        if "unchanged" in workload or "small_change" in workload
                        else None
                    )
                    synchronize(args.device)
                    start = time.perf_counter()
                    if workload == "fixed_steps":
                        state = brain.settle_batch(drive, steps=60)
                        steps = state.steps
                    elif workload.startswith("residual_"):
                        result = brain.equilibrate(
                            changed if "small_change" in workload else drive,
                            state=warm,
                            budget=128,
                            chunk=args.chunk,
                            tolerance=tolerance,
                        )
                        state, steps = result.state, result.steps
                    else:
                        for _ in range(args.updates):
                            phases, _ = learner.step(drive, labels)
                        state, brain, steps = phases.free, learner.brain, args.updates * 80
                    synchronize(args.device)
                    elapsed.append(time.perf_counter() - start)
                # Snapshotting and the independent float64 residual are outside the timer.
                snapshot = np.concatenate(
                    (state.v.ravel(), state.activation.ravel(), brain.efficacy, brain.bias)
                )
                snapshots[arm] = snapshot
                residual = cd.Brain.residual(
                    brain, changed if "small_change" in workload else drive, state, on_device=False
                )
                arms[arm] = {
                    "seconds": elapsed[args.warmups :],
                    "median_seconds": float(np.median(elapsed[args.warmups :])),
                    "steps": steps,
                    "host_residual_max": float(residual.max()),
                    "warmup_seconds": elapsed[: args.warmups],
                }
            difference = float(np.abs(snapshots["current"] - snapshots["baseline"]).max())
            cells.append(
                {
                    "seed": seed,
                    "workload": workload,
                    "arms": arms,
                    "max_abs_state_parameter_difference": difference,
                    "tolerance": tolerance,
                    "baseline_over_current": arms["baseline"]["median_seconds"]
                    / arms["current"]["median_seconds"],
                }
            )
    profiles = {}
    for arm in ("baseline", "current"):
        profile = cProfile.Profile()
        brain = build(arm)
        profile.enable()
        for _ in range(20):
            brain.equilibrate(drive, tolerance=tolerance, chunk=args.chunk)
        synchronize(args.device)
        profile.disable()
        rendered = io.StringIO()
        pstats.Stats(profile, stream=rendered).sort_stats("cumulative").print_stats(20)
        profiles[arm] = rendered.getvalue()
    return {
        "baseline_revision": BASELINE_REVISION,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device_name": (
                torch.cuda.get_device_name(args.device)
                if args.device.startswith("cuda")
                else subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                ).strip()
                if platform.system() == "Darwin"
                else platform.processor()
            ),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": args.device,
            "dtype": "float32" if args.device == "mps" else "float64",
            "synchronization": "before/after each timed sample",
            "joules_measured": False,
        },
        "protocol": {
            "seeds": list(range(5)),
            "width": args.width,
            "batch": args.batch,
            "chunk": args.chunk,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "centered_updates": args.updates,
            "arms_alternate_order_by_seed": True,
            "setup_and_snapshotting_timed": False,
        },
        "cells": cells,
        "profiles": profiles,
    }


def verify(body: dict) -> str | None:
    limit = 3e-5 if body["environment"]["dtype"] == "float32" else 1e-11
    scheduled = {
        (s, w)
        for s in range(5)
        for w in (
            "fixed_steps",
            "residual_cold",
            "residual_unchanged",
            "residual_small_change",
            "centered_updates",
        )
    }
    if len(body["cells"]) != 25 or {(c["seed"], c["workload"]) for c in body["cells"]} != scheduled:
        return "missing scheduled cells"
    for cell in body["cells"]:
        a, b = cell["arms"]["baseline"], cell["arms"]["current"]
        if cell["max_abs_state_parameter_difference"] > limit or a["steps"] != b["steps"]:
            return "numerical or work conformance failed"
        if (
            cell["workload"].startswith("residual_")
            and max(a["host_residual_max"], b["host_residual_max"]) > cell["tolerance"]
        ):
            return "residual tolerance not met"
        if cell["workload"] == "residual_unchanged" and a["steps"] != 0:
            return "unchanged input used settling steps"
        for arm in (a, b):
            if not np.isclose(arm["median_seconds"], np.median(arm["seconds"]), rtol=1e-12):
                return "timing summary differs from samples"
        if not np.isclose(
            cell["baseline_over_current"], a["median_seconds"] / b["median_seconds"], rtol=1e-12
        ):
            return "speed ratio differs from samples"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = source_bundle(ROOT)
    body = run(args)
    cd.Receipt.build("matched-runtime-v1", body, sources).write(args.out)
    print(cd.Receipt.verify(args.out, sources=sources, check=verify))
    for workload in sorted({c["workload"] for c in body["cells"]}):
        selected = [c for c in body["cells"] if c["workload"] == workload]
        print(
            workload,
            "median baseline/current",
            np.median([c["baseline_over_current"] for c in selected]),
            "max difference",
            max(c["max_abs_state_parameter_difference"] for c in selected),
        )
    if verify(body):
        raise SystemExit(verify(body))


if __name__ == "__main__":
    main()
