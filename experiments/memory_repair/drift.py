"""Boundary-informed diagnostic of head versus representation plasticity loss.

The resets below are causal probes, not task-free deployed interventions. After
30 ordinary tasks, each fork sees exactly the same unseen permutation and examples.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import continual
import numpy as np
import torch

import cadence as cd

HERE = Path(__file__).resolve().parent
SPEC = {
    "seeds": [201, 202, 203, 204, 205],
    "history_tasks": 30,
    "train_per_task": 2000,
    "test_per_task": 256,
    "arms": ["carried", "reset_head", "reset_representation", "reset_all"],
    "checkpoints_updates": [0, 16, 32, 63],
    "model": continual.SPEC["patch"],
    "inputs": 196,
    "hidden": 48,
    "batch": 32,
    "ridge": 0.001,
    "protocol": "30 no-replay tasks; unseen 31st permutation shared by all diagnostic forks",
    "information": "resets receive task boundary; no head mask or task ID at prediction",
    "data": "same 2000 training/256 test images within each seed; only permutations change",
    "readout_probe": "ridge on free hidden states; observed train labels, extra computation",
    "selection": "all arms and seeds fixed before execution; every outcome retained",
}


def sources() -> list[tuple[str, Path]]:
    return [(str(Path(__file__).relative_to(continual.ROOT)), Path(__file__)), *continual.sources()]


def masks(model: continual.Patch) -> tuple[np.ndarray, np.ndarray]:
    wiring = model.learner.brain.connectome
    output = np.asarray(wiring.populations["output"])
    return np.isin(wiring.pre, output) | np.isin(wiring.post, output), np.isin(
        np.arange(wiring.n), output
    )


def reset(model: continual.Patch, initial: continual.Patch, arm: str) -> continual.Patch:
    fork = copy.deepcopy(model)
    head_edges, head_neurons = masks(fork)
    if arm == "carried":
        return fork
    if arm == "reset_representation":
        head_edges, head_neurons = ~head_edges, ~head_neurons
    elif arm == "reset_all":
        head_edges[:], head_neurons[:] = True, True
    elif arm != "reset_head":
        raise ValueError(arm)
    brain, fresh = fork.learner.brain, initial.learner.brain
    efficacy, bias = brain.efficacy.copy(), brain.bias.copy()
    efficacy[head_edges], bias[head_neurons] = fresh.efficacy[head_edges], fresh.bias[head_neurons]
    fork.learner.brain = brain.with_parameters(efficacy=efficacy, bias=bias)
    # Base config has no momentum or RMS update; retain no accidental optimizer memory.
    for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias"):
        getattr(fork.learner, name).fill(0)
    return fork


def hidden(model: continual.Patch, x: np.ndarray) -> np.ndarray:
    indices = np.asarray(model.learner.brain.connectome.populations["hidden"])
    return np.concatenate(
        [
            model.learner.free(model.drive(x[i : i + 128])).activation[:, indices]
            for i in range(0, len(x), 128)
        ]
    )


def diagnostic(model: continual.Patch, x: np.ndarray) -> dict:
    activity = hidden(model, x)
    edges, neurons = masks(model)
    efficacy, bias = model.learner.brain.efficacy, model.learner.brain.bias
    return {
        "hidden_saturated_fraction": float(np.mean(activity >= 0.95)),
        "hidden_std_mean": float(np.std(activity, axis=0).mean()),
        "hidden_low_variance_fraction": float(np.mean(np.std(activity, axis=0) < 0.01)),
        "head_weight_abs_mean": float(np.abs(efficacy[edges]).mean()),
        "representation_weight_abs_mean": float(np.abs(efficacy[~edges]).mean()),
        "head_bias_mean": float(bias[neurons].mean()),
        "hidden_bias_mean": float(
            bias[np.asarray(model.learner.brain.connectome.populations["hidden"])].mean()
        ),
        "capped_weight_fraction": float(np.mean(np.abs(efficacy) >= 7.9999)),
    }


def ridge_accuracy(
    model: continual.Patch, x: np.ndarray, y: np.ndarray, xt: np.ndarray, yt: np.ndarray
) -> float:
    train, test = hidden(model, x), hidden(model, xt)
    train, test = (
        np.column_stack([train, np.ones(len(train))]),
        np.column_stack([test, np.ones(len(test))]),
    )
    regularizer = np.eye(train.shape[1]) * SPEC["ridge"]
    regularizer[-1, -1] = 0  # unpenalized intercept
    coefficient = np.linalg.solve(train.T @ train + regularizer, train.T @ np.eye(10)[y])
    return float(np.mean(np.argmax(test @ coefficient, axis=1) == yt))


def fit(model: continual.Patch, x: np.ndarray, y: np.ndarray) -> None:
    for start in range(0, len(x), SPEC["batch"]):
        model.step(x[start : start + SPEC["batch"]], y[start : start + SPEC["batch"]])


def trial(data: tuple, seed: int) -> dict:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    train_ids = rng.permutation(len(data[0]))[: SPEC["train_per_task"]]
    test_ids = rng.permutation(len(data[2]))[: SPEC["test_per_task"]]
    raw_x, y, raw_t, yt = (
        data[0][train_ids],
        data[1][train_ids],
        data[2][test_ids],
        data[3][test_ids],
    )
    permutations = [rng.permutation(SPEC["inputs"]) for _ in range(SPEC["history_tasks"] + 1)]
    initial = continual.Patch(seed)
    model = copy.deepcopy(initial)
    history = []
    for task, permutation in enumerate(permutations[:-1]):
        x, xt = raw_x[:, permutation], raw_t[:, permutation]
        fit(model, x, y)
        accuracy = model.accuracy(xt, yt)
        history.append({"task": task, "accuracy": accuracy, "diagnostic": diagnostic(model, xt)})
        print(seed, "history", task, round(accuracy, 4), flush=True)
    x, xt = raw_x[:, permutations[-1]], raw_t[:, permutations[-1]]
    rows = []
    for arm in SPEC["arms"]:
        fork = reset(model, initial, arm)
        curve = [{"updates": 0, "accuracy": fork.accuracy(xt, yt)}]
        before = diagnostic(fork, xt)
        for i, begin in enumerate(range(0, len(x), SPEC["batch"]), start=1):
            fork.step(x[begin : begin + SPEC["batch"]], y[begin : begin + SPEC["batch"]])
            if i in SPEC["checkpoints_updates"]:
                curve.append({"updates": i, "accuracy": fork.accuracy(xt, yt)})
        rows.append(
            {
                "arm": arm,
                "curve": curve,
                "before": before,
                "after": diagnostic(fork, xt),
                "ridge_accuracy": ridge_accuracy(fork, x, y, xt, yt),
                "old_task_accuracy": fork.accuracy(raw_t[:, permutations[0]], yt),
                "training_examples": len(x),
                "updates": i,
            }
        )
        print(
            seed,
            arm,
            "native",
            curve[-1]["accuracy"],
            "ridge",
            rows[-1]["ridge_accuracy"],
            flush=True,
        )
    return {
        "seed": seed,
        "history": history,
        "rows": rows,
        "permutations_sha256": continual.hashlib.sha256(
            np.asarray(permutations).tobytes()
        ).hexdigest(),
        "seconds": time.perf_counter() - start,
    }


def check(body: dict) -> str | None:
    if body["spec"] != SPEC:
        return "diagnostic specification differs"
    results = body["results"]
    if len(results) != len(SPEC["seeds"]) or {r["seed"] for r in results} != set(SPEC["seeds"]):
        return "incomplete seed grid"
    for result in results:
        if [r["task"] for r in result["history"]] != list(range(SPEC["history_tasks"])):
            return "incomplete history"
        if [r["arm"] for r in result["rows"]] != SPEC["arms"]:
            return "missing diagnostic arm"
        for row in result["rows"]:
            if [c["updates"] for c in row["curve"]] != SPEC["checkpoints_updates"]:
                return "missing adaptation checkpoint"
            if row["training_examples"] != SPEC["train_per_task"] or row["updates"] != 63:
                return "mismatched adaptation budget"
            if not all(0 <= r["accuracy"] <= 1 for r in row["curve"]):
                return "invalid accuracy"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "drift.json")
    parser.add_argument("--data", type=Path, default=HERE / "data/mnist.npz")
    args = parser.parse_args()
    spec_path = args.out.with_suffix(".spec.json")
    if args.out.exists() or spec_path.exists():
        raise SystemExit("output or specification already exists; choose a new --out")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=False)
    torch.set_num_threads(1)
    spec_path.write_text(cd.canonical_json(SPEC) + "\n")
    data = continual.load(args.data)
    results = []
    for seed in SPEC["seeds"]:
        results.append(trial(data, seed))
        body = {"spec": SPEC, "data_sha256": data[-1], "results": results}
        cd.Receipt.build("cadence/plasticity-diagnostic/v1", body, sources()).write(args.out)
    assert check(body) is None


if __name__ == "__main__":
    main()
