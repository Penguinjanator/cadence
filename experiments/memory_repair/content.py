"""Content retrieval without supplied addresses; retain capacity/aliasing controls."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import cadence as cd
from cadence.content_memory import ContentMemory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ("delta", "separated_delta", "nearest_fifo", "frozen_prototypes", "content")
CONDITIONS = ("clean", "noisy", "aliased", "overflow")
SPEC = {
    "seeds": [101, 102, 103, 104, 105],
    "arms": ARMS,
    "conditions": CONDITIONS,
    "inputs": 32,
    "capacity": 32,
    "identities": 16,
    "overflow_identities": 64,
    "observations_per_identity": 16,
    "queries_per_identity": 20,
    "noise": {"clean": 0.04, "noisy": 0.1, "aliased": 0.04, "overflow": 0.04},
    "match": 0.55,
    "key_rate": 0.2,
    "separator_expansion": 256,
    "separator_winners": 8,
    "protocol": "one shared memory; no task/slot/identity passed as cue; test values never written",
    "selection": "parameters fixed before 101:105; all conditions and arms retained",
}


def sources() -> list[tuple[str, Path]]:
    return [
        (str(p.relative_to(ROOT)), p)
        for p in [
            Path(__file__),
            ROOT / "src/cadence/content_memory.py",
            ROOT / "src/cadence/stream.py",
        ]
    ]


def trial(seed: int, condition: str, arm: str) -> dict:
    rng = np.random.default_rng(seed)
    n = SPEC["overflow_identities"] if condition == "overflow" else SPEC["identities"]
    d, cap, noise = SPEC["inputs"], SPEC["capacity"], SPEC["noise"][condition]
    centers = cd.FastSynapses._delta_unit(rng.standard_normal((n, d)))
    if condition == "aliased":
        centers[n // 2 :] = centers[: n // 2]
    labels = np.tile(np.arange(n), SPEC["observations_per_identity"])
    rng.shuffle(labels)
    train = centers[labels] + noise * rng.standard_normal((len(labels), d))
    test_y = np.repeat(np.arange(n), SPEC["queries_per_identity"])
    test = centers[test_y] + noise * rng.standard_normal((len(test_y), d))
    # Identical cues for paired aliases make the information boundary explicit.
    if condition == "aliased":
        test[len(test) // 2 :] = test[: len(test) // 2]
    values = np.eye(n)
    start = time.perf_counter()
    if arm in ("delta", "separated_delta"):
        sep = (
            cd.PatternSeparator(
                d, SPEC["separator_expansion"], SPEC["separator_winners"], seed=seed
            )
            if arm == "separated_delta"
            else None
        )
        memory = cd.FastSynapses(np.arange(d), np.arange(d, d + n), rule="delta", separator=sep)
        for x, y in zip(train, labels, strict=True):
            memory.observe(x[None], values[y : y + 1])
        predictions = np.concatenate([memory.recall(x[None]) for x in test])
        mutable = memory.strength.nbytes + memory.mass.nbytes
        fixed = 0 if sep is None else sep.projection.nbytes + sep.mean.nbytes
        info = {"mutable_bytes": mutable, "fixed_bytes": fixed, "writes": memory.writes}
    elif arm == "nearest_fifo":
        keys, stored = np.zeros((cap, d)), np.zeros((cap, n))
        for i, (x, y) in enumerate(zip(train, labels, strict=True)):
            keys[i % cap] = cd.FastSynapses._delta_unit(x[None])[0]
            stored[i % cap] = values[y]
        scores = cd.FastSynapses._delta_unit(test) @ keys.T
        predictions = stored[np.argmax(scores, axis=1)]
        predictions[np.max(scores, axis=1) < SPEC["match"]] = 0
        info = {
            "mutable_bytes": keys.nbytes + stored.nbytes,
            "fixed_bytes": 0,
            "writes": len(train),
        }
    else:
        memory = ContentMemory(
            d, n, cap, match=SPEC["match"], key_rate=SPEC["key_rate"] if arm == "content" else 0
        )
        memory.observe(train, values[labels])
        predictions = memory.recall(test)
        info = memory.to_dict() | {"fixed_bytes": 0}
    accepted = np.any(predictions != 0, axis=1)
    guesses = np.where(accepted, np.argmax(predictions, axis=1), -1)
    return {
        "seed": seed,
        "condition": condition,
        "arm": arm,
        "accuracy": float(np.mean(guesses == test_y)),
        "abstention": float(np.mean(~accepted)),
        "test_n": len(test_y),
        "correct": int(np.sum(guesses == test_y)),
        "memory": info,
        "seconds": time.perf_counter() - start,
    }


def check(body: dict) -> str | None:
    expected = {(s, c, a) for s in body["spec"]["seeds"] for c in CONDITIONS for a in ARMS}
    rows = body["rows"]
    if (
        len(rows) != len(expected)
        or {(r["seed"], r["condition"], r["arm"]) for r in rows} != expected
    ):
        return "incomplete or duplicate comparison grid"
    for row in rows:
        if abs(row["accuracy"] - row["correct"] / row["test_n"]) > 1e-12:
            return "accuracy does not recompute"
        if row["condition"] == "aliased" and row["accuracy"] > 0.5:
            return "identical aliased cues exceeded the information ceiling"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "content.json")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=sources(), check=check)
        print(message)
        raise SystemExit(0 if ok else 1)
    spec_path = args.out.with_suffix(".spec.json")
    if args.out.exists() or spec_path.exists():
        raise SystemExit("output or specification already exists; choose a new --out")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.touch(exist_ok=False)  # reserve this run's output without replacing another run
    spec_path.write_text(cd.canonical_json(SPEC) + "\n")
    rows = []
    for seed in SPEC["seeds"]:
        for condition in CONDITIONS:
            for arm in ARMS:
                row = trial(seed, condition, arm)
                rows.append(row)
                print(seed, condition, arm, round(row["accuracy"], 4), flush=True)
    body = {"spec": SPEC, "rows": rows}
    assert check(body) is None
    cd.Receipt.build("cadence/content-repair/v1", body, sources()).write(args.out)


if __name__ == "__main__":
    main()
