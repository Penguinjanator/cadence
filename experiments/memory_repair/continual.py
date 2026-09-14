"""Single-head replay and its plasticity cost, on a declared small MNIST protocol.

One pass through each task; 2x2 average-pooled images; no task identity at
inference. Replay and current-repeat see equally many rows at equally many
updates. All arms see the identical current examples and test examples.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import cadence as cd
from cadence.replay import ReservoirReplay

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ("patch_none", "patch_repeat", "patch_replay", "adam_none", "adam_repeat", "adam_replay")
SPEC = {
    "seeds": [101, 102, 103, 104, 105],
    "arms": ARMS,
    "inputs": 196,
    "hidden": 48,
    "outputs": 10,
    "batch": 32,
    "capacity": 256,
    "split_tasks": 5,
    "permuted_tasks": 10,
    "train_per_task": 2000,
    "test_per_task": 256,
    "pool": "2x2 average, 28x28 to 14x14",
    "patch": {
        "eta": 3.0,
        "eta_bias": 0.03,
        "beta": 0.1,
        "temperature": 0.1,
        "tolerance": 0.003,
        "free_steps": 100,
        "nudged_steps": 12,
    },
    "adam": {"lr": 0.001, "hidden_activation": "ReLU"},
    "replay": "sample previous reservoir before current write; <=32 prior rows per update",
    "repeat": "append <=32 current rows, count matched to replay; identical number of updates",
    "identity": "no task ID, head mask, class balance, or task-boundary operation for any arm",
    "selection": "fixed parameters before seeds 101:105; pilot seed99 retained separately",
    "data_url": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
}


def sources() -> list[tuple[str, Path]]:
    names = (
        "replay.py",
        "learning.py",
        "brain.py",
        "connectome.py",
        "neuron.py",
        "stream.py",
        "blocks.py",
        "recording.py",
        "fused.py",
    )
    paths = [Path(__file__), *[ROOT / "src/cadence" / name for name in names]]
    return [(str(p.relative_to(ROOT)), p) for p in paths if p.exists()]


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as data:
        x, y = data["x_train"], data["y_train"]
        xt, yt = data["x_test"], data["y_test"]

    def pool(a: np.ndarray) -> np.ndarray:
        return a.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, 196) / 255.0

    return pool(x), y, pool(xt), yt, digest


def make_tasks(data: tuple, seed: int, mode: str, per_task: int) -> list:
    x, y, xt, yt = data[:4]
    rng = np.random.default_rng(seed)
    count = SPEC["split_tasks"] if mode == "split" else SPEC["permuted_tasks"]
    tasks = []
    for task in range(count):
        if mode == "split":
            train_ids = np.flatnonzero(y // 2 == task)
            test_ids = np.flatnonzero(yt // 2 == task)
            permutation = np.arange(SPEC["inputs"])
        else:
            train_ids, test_ids = np.arange(len(y)), np.arange(len(yt))
            permutation = rng.permutation(SPEC["inputs"])
        train_ids = rng.permutation(train_ids)[:per_task]
        test_ids = rng.permutation(test_ids)[: SPEC["test_per_task"]]
        tasks.append(
            (x[train_ids][:, permutation], y[train_ids], xt[test_ids][:, permutation], yt[test_ids])
        )
    return tasks


class Patch:
    def __init__(self, seed: int) -> None:
        wiring = cd.layered(SPEC["inputs"], SPEC["hidden"], 10, density=1, seed=seed)
        self.learner = cd.Learner(
            cd.Brain(wiring, cd.learning_neuron_model(), backend="cpu"),
            wiring.populations["output"],
            cd.LearnerConfig(**SPEC["patch"]),
        )
        self.parameters = self.learner.parameters()
        self.steps = 0

    def drive(self, x: np.ndarray) -> np.ndarray:
        n = self.learner.brain.connectome.n
        return self.learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, n - x.shape[1]))))

    def step(self, x: np.ndarray, y: np.ndarray) -> None:
        state, _ = self.learner.step(self.drive(x), y)
        self.steps += state.free.steps + state.nudged.steps
        if state.opposite is not None:
            self.steps += state.opposite.steps

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.learner.accuracy(self.drive(x), y, batch=128)


class Adam:
    def __init__(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(SPEC["inputs"], SPEC["hidden"]), nn.ReLU(), nn.Linear(SPEC["hidden"], 10)
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=SPEC["adam"]["lr"])
        self.parameters = sum(p.numel() for p in self.net.parameters())
        self.steps = 0

    def step(self, x: np.ndarray, y: np.ndarray) -> None:
        self.optimizer.zero_grad()
        loss = nn.functional.cross_entropy(
            self.net(torch.tensor(x, dtype=torch.float32)), torch.tensor(y, dtype=torch.int64)
        )
        loss.backward()
        self.optimizer.step()

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        with torch.no_grad():
            guess = self.net(torch.tensor(x, dtype=torch.float32)).argmax(dim=1).numpy()
        return float(np.mean(guess == y))


def trial(tasks: list, seed: int, arm: str) -> dict:
    start = time.perf_counter()
    model = Patch(seed) if arm.startswith("patch") else Adam(seed)
    replay = ReservoirReplay(SPEC["capacity"], SPEC["inputs"], seed=seed)
    rows_seen = extra_seen = updates = 0
    matrix = []
    for task, (x, y, _, _) in enumerate(tasks):
        for begin in range(0, len(x), SPEC["batch"]):
            xb, yb = x[begin : begin + SPEC["batch"]], y[begin : begin + SPEC["batch"]]
            extra_n = min(len(xb), rows_seen, SPEC["capacity"])
            tx, ty = xb, yb
            if arm.endswith("replay"):
                rx, ry = replay.sample(len(xb))
                assert len(rx) == extra_n
                tx, ty = np.concatenate([xb, rx]), np.concatenate([yb, ry])
                extra_seen += len(rx)
            elif arm.endswith("repeat"):
                tx, ty = np.concatenate([xb, xb[:extra_n]]), np.concatenate([yb, yb[:extra_n]])
                extra_seen += extra_n
            model.step(tx, ty)
            if arm.endswith("replay"):
                replay.observe(xb, yb)
            rows_seen += len(xb)
            updates += 1
        matrix.append([model.accuracy(test_x, test_y) for _, _, test_x, test_y in tasks])
        print(
            arm,
            seed,
            task,
            "current",
            round(matrix[-1][task], 3),
            "all",
            round(float(np.mean(matrix[-1])), 3),
            flush=True,
        )
    matrix = np.array(matrix)
    current = np.diag(matrix)
    return {
        "seed": seed,
        "arm": arm,
        "matrix": matrix.tolist(),
        "final_average": float(matrix[-1].mean()),
        "current_average": float(current.mean()),
        "current_first3": float(current[:3].mean()),
        "current_last3": float(current[-3:].mean()),
        "forgetting": float(np.mean(current[:-1] - matrix[-1, :-1])),
        "observed_examples": rows_seen,
        "extra_training_examples": extra_seen,
        "training_examples": rows_seen + extra_seen,
        "updates": updates,
        "parameters": model.parameters,
        "settling_steps": model.steps,
        "memory_bytes": replay.to_dict()["mutable_bytes"] if arm.endswith("replay") else 0,
        "seconds": time.perf_counter() - start,
    }


def check(body: dict) -> str | None:
    spec, rows = body["spec"], body["rows"]
    expected = {(s, a) for s in spec["seeds"] for a in spec["arms"]}
    if len(rows) != len(expected) or {(r["seed"], r["arm"]) for r in rows} != expected:
        return "incomplete or duplicate seed/arm grid"
    for row in rows:
        matrix = np.asarray(row["matrix"])
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
            return "invalid accuracy matrix"
        current = matrix.diagonal()
        expected_metrics = {
            "final_average": matrix[-1].mean(),
            "current_average": current.mean(),
            "current_first3": current[:3].mean(),
            "current_last3": current[-3:].mean(),
            "forgetting": np.mean(current[:-1] - matrix[-1, :-1]),
        }
        if any(abs(row[k] - v) > 1e-12 for k, v in expected_metrics.items()):
            return "summary does not recompute"
        if row["observed_examples"] + row["extra_training_examples"] != row["training_examples"]:
            return "training examples do not add up"
    for seed in spec["seeds"]:
        same_seed = [r for r in rows if r["seed"] == seed]
        if len({r["updates"] for r in same_seed}) != 1:
            return "updates differ between arms"
        extra = [r["extra_training_examples"] for r in same_seed if not r["arm"].endswith("none")]
        if len(set(extra)) != 1:
            return "repeat/replay training budgets differ"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("split", "permuted"), default="split")
    parser.add_argument("--data", type=Path, default=HERE / "data/mnist.npz")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=sources(), check=check)
        print(message)
        raise SystemExit(0 if ok else 1)
    torch.set_num_threads(1)
    spec = SPEC | {"mode": args.mode, "seeds": [99] if args.pilot else SPEC["seeds"]}
    if args.pilot:
        spec = spec | {"train_per_task": 1000}
    out = args.out or HERE / (args.mode + ("_pilot" if args.pilot else "") + ".json")
    spec_path = out.with_suffix(".spec.json")
    if out.exists() or spec_path.exists():
        raise SystemExit("output or specification already exists; choose a new --out")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.touch(exist_ok=False)
    spec_path.write_text(cd.canonical_json(spec) + "\n")
    data = load(args.data)
    rows = []
    for seed in spec["seeds"]:
        tasks = make_tasks(data, seed, args.mode, spec["train_per_task"])
        for arm in ARMS:
            rows.append(trial(tasks, seed, arm))
            body = {
                "spec": spec,
                "rows": rows,
                "data_sha256": data[-1],
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                    "platform": platform.platform(),
                },
            }
            cd.Receipt.build("cadence/continual-memory-repair/v1", body, sources()).write(out)
    assert check(body) is None


if __name__ == "__main__":
    main()
