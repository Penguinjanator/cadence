"""Frozen six-arm wiring comparison under the unchanged delayed-first-action protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import types
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

ARMS = ("trace", "no_trace", "legacy_quadratic", "tabular_score", "random_skip", "zero_skip")
SCHEDULE = {"seeds": list(range(5)), "lengths": [3, 9, 33], "episodes": 200, "batch": 32}


def freeze(integration: Path, destination: Path) -> None:
    """Read the integrated core once; subsequent execution imports only the archive."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=integration, text=True
    ).strip()
    files = {"run.py": Path(__file__).read_bytes()}
    for path in sorted((integration / "src/cadence").rglob("*.py")):
        files[str(path.relative_to(integration / "src"))] = path.read_bytes()
    files["policy_credit.py"] = (integration / "experiments/policy_credit/run.py").read_bytes()
    for name in ("receipt.json", "sources.zip"):
        files[f"reference/{name}"] = (
            integration / "experiments/policy_credit/results" / name
        ).read_bytes()
    files["provenance.json"] = json.dumps(
        {
            "integration_path": str(integration.resolve()),
            "integration_commit": commit,
            "captured_utc": datetime.now(UTC).isoformat(),
            "schedule": SCHEDULE,
            "purpose": "Wiring/capacity comparison, not learning-rule superiority",
        },
        sort_keys=True,
    ).encode()
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=integration, text=True).strip()
        != commit
    ):
        raise RuntimeError("integration revision changed during source capture")
    for path in sorted((integration / "src/cadence").rglob("*.py")):
        if files[str(path.relative_to(integration / "src"))] != path.read_bytes():
            raise RuntimeError("integrated core changed during source capture")
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)


def wire(protocol, seed: int, lam: float, legacy: bool, variant: str):
    """Add eight sensorimotor edges; the original edges and all hyperparameters agree."""
    cd = protocol.cd
    original = protocol.patch(seed, lam, legacy)
    if variant == "trace":
        return original
    old_brain = original.learner.brain
    graph = cd.layered(4, 8, 2, density=1.0, seed=seed, skip=True)
    scale = graph.sign.copy()
    skip = (graph.pre < 4) & (graph.post >= 12)
    assert int(skip.sum()) == 8
    if variant == "zero_skip":
        scale[skip] = 0.0
    elif variant != "random_skip":
        raise ValueError(variant)
    brain = cd.Brain(graph, old_brain.neuron_model, efficacy=scale)
    common = {
        (int(p), int(q)): float(w)
        for p, q, w in zip(
            old_brain.connectome.pre, old_brain.connectome.post, old_brain.weights, strict=True
        )
    }
    for p, q, w in zip(graph.pre, graph.post, brain.weights, strict=True):
        if (int(p), int(q)) in common:
            assert float(w) == common[int(p), int(q)]
    if variant == "zero_skip":
        np.testing.assert_array_equal(brain.dense(), old_brain.dense())
    learner = cd.Learner(brain, graph.populations["output"], original.learner.config)
    cls = protocol.LegacyActor if legacy else cd.ActorCritic
    return cls(learner, graph.populations["hidden"], original.config, seed=seed)


def initial_diagnostic(protocol, seed: int) -> dict:
    readings = {}
    for variant in ("trace", "random_skip", "zero_skip"):
        ac = wire(protocol, seed, 0.95, False, variant)
        state = ac.settle(protocol.drive(ac, np.array([0, 1])))
        readings[variant] = {
            "probabilities": ac.probabilities(state).tolist(),
            "parameters": ac.parameters(),
            "synapses": ac.learner.brain.connectome.synapses,
            "row_mass": protocol.cd.row_mass(ac.learner.brain),
        }
    np.testing.assert_array_equal(
        readings["trace"]["probabilities"], readings["zero_skip"]["probabilities"]
    )
    return {"seed": seed, "topologies": readings, "zero_skip_initial_predictions_exact": True}


def check(body: dict) -> str | None:
    if body["schedule"] != SCHEDULE:
        return "schedule differs from the declared experiment"
    expected = {(a, s, length) for a in ARMS for s in range(5) for length in (3, 9, 33)}
    rows = body["rows"]
    if len(rows) != len(expected) or {(r["arm"], r["seed"], r["length"]) for r in rows} != expected:
        return "missing or duplicate outcomes"
    for row in rows:
        if row["episodes"] != 200 or row["batch"] != 32 or len(row["training"]) != 200:
            return "training budget mismatch"
        if row["transitions"] != 200 * 32 * row["length"]:
            return "transition budget mismatch"
        p = np.asarray(row["probabilities"])
        if (
            p.shape != (2, 2)
            or not np.isfinite(p).all()
            or (p < 0).any()
            or not np.allclose(p.sum(1), 1)
        ):
            return "invalid probabilities"
        if row["accuracy"] != float((p.argmax(1) == [0, 1]).mean()):
            return "greedy accuracy mismatch"
        if abs(row["stochastic_accuracy"] - p[[0, 1], [0, 1]].mean()) > 1e-12:
            return "stochastic accuracy mismatch"
    if len(body["initial_states"]) != 5:
        return "missing initial-state checks"
    for state in body["initial_states"]:
        topologies = state["topologies"]
        if topologies["trace"]["probabilities"] != topologies["zero_skip"]["probabilities"]:
            return "zero skip did not preserve predictions"
    return None


def execute(output: Path, source_archive: Path) -> None:
    # Executed in the extracted snapshot, which owns both these modules.
    import policy_credit as protocol

    import cadence as cd

    frozen = Path(__file__).resolve().parent
    if Path(cd.__file__).resolve().parent != frozen / "cadence":
        raise RuntimeError("experiment did not import the frozen integrated core")
    body = {
        "schedule": SCHEDULE,
        "arms": list(ARMS),
        "provenance": json.loads((frozen / "provenance.json").read_text()),
        "source_archive_sha256": hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        "reference_receipt_digest": cd.Receipt.read(frozen / "reference/receipt.json").digest,
        "initial_states": [initial_diagnostic(protocol, seed) for seed in range(5)],
        "rows": [],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "boundary": (
            "Wiring/capacity, not learning-rule superiority. No tuning after results. "
            "CPU timings may share the host; no efficiency or joule claim."
        ),
    }
    for length in SCHEDULE["lengths"]:
        for seed in SCHEDULE["seeds"]:
            for arm in ARMS:
                if arm in ("random_skip", "zero_skip"):
                    # A private globals dictionary preserves the exact train() bytecode
                    # without changing the imported module or leaking a monkeypatch.
                    def factory(s, lam, legacy, variant=arm):
                        return wire(protocol, s, lam, legacy, variant)

                    train = types.FunctionType(
                        protocol.train.__code__, {**protocol.train.__globals__, "patch": factory}
                    )
                    row = train("trace", seed, length, 200, 32)
                    row["arm"] = arm
                else:
                    row = protocol.train(arm, seed, length, 200, 32)
                body["rows"].append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "training"}), flush=True)
    reference = cd.Receipt.read(frozen / "reference/receipt.json").body["rows"]
    indexed = {(r["arm"], r["seed"], r["length"]): r for r in body["rows"]}
    comparisons = []
    for old in reference:
        new = indexed[old["arm"], old["seed"], old["length"]]
        fields = ("training", "probabilities", "parameters", "accuracy", "stochastic_accuracy")
        comparisons.append(
            {
                "arm": old["arm"],
                "seed": old["seed"],
                "length": old["length"],
                "exact_match": all(old[k] == new[k] for k in fields),
            }
        )
    body["reference_reproduction"] = comparisons
    if check(body):
        raise ValueError(check(body))
    with ZipFile(source_archive) as archive:
        sources = [(name, frozen / name) for name in sorted(archive.namelist())]
    cd.Receipt.build("cadence/sensorimotor-skip/v1", body, sources=sources).write(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute:
        execute(args.output, args.archive)
        return
    if bool(args.integration) == bool(args.archive):
        raise ValueError("provide exactly one of --integration or --archive")
    output = args.output.resolve()
    archive = output.parent / "sources.zip"
    if output.exists() or archive.exists():
        raise ValueError(
            "receipts and source bundles are immutable; choose a fresh output directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.integration:
        freeze(args.integration.resolve(), archive)
    else:
        archive.write_bytes(args.archive.read_bytes())
    with tempfile.TemporaryDirectory(prefix="cadence-sensorimotor-") as directory:
        frozen = Path(directory)
        with ZipFile(archive) as packed:
            for name in packed.namelist():
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("unsafe archive member")
            packed.extractall(frozen)
        environment = dict(os.environ, PYTHONPATH=str(frozen))
        subprocess.run(
            [
                sys.executable,
                str(frozen / "run.py"),
                "--execute",
                "--archive",
                str(archive),
                "--output",
                str(output),
            ],
            cwd=frozen,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
