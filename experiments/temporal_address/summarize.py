"""Recompute all arm statistics after checking the original per-episode artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
from run import ARMS, HERE, cd, verify


def summary():
    paths = [HERE / "runs/main/receipt.json", HERE / "runs/pilot/receipt.json"]
    for path in paths:
        verify(path)
    main, pilot = [cd.Receipt.read(path) for path in paths]
    body = {
        "seeds": main.body["schedule"]["seeds"],
        "arms": {},
        "paired": {},
        "parent_digests": [main.digest, pilot.digest],
        "main_training_interactions": 0,
        "main_evaluation_interactions": 0,
        "main_measured_seconds": 0.0,
        "pilot_measured_seconds": 0.0,
    }
    values = {}
    for arm in ARMS:
        rows = [row for row in main.body["outcomes"] if row["arm"] == arm]
        v = np.array([row["mean_return"] for row in rows])
        values[arm] = v
        body["arms"][arm] = {
            "per_seed_returns": v.tolist(),
            "mean_return": float(v.mean()),
            "sample_sd": float(v.std(ddof=1)),
            "measured_seconds": sum(row["seconds"] for row in rows),
        }
        body["main_training_interactions"] += sum(row["training_steps"] for row in rows)
        body["main_evaluation_interactions"] += len(rows) * 100 * 51
        body["main_measured_seconds"] += body["arms"][arm]["measured_seconds"]
    body["pilot_measured_seconds"] = sum(row["seconds"] for row in pilot.body["outcomes"])
    for a, b in [
        ("patch_memory", "patch_masked"),
        ("patch_memory", "patch_no_direct"),
        ("patch_memory", "linear_memory"),
        ("linear_memory", "linear_masked"),
    ]:
        diff = values[a] - values[b]
        body["paired"][a + "_minus_" + b] = {
            "per_seed_difference": diff.tolist(),
            "mean_difference": float(diff.mean()),
            "sample_sd_difference": float(diff.std(ddof=1)),
        }
    sources = [(str(p.relative_to(HERE)), p) for p in [Path(__file__).resolve(), *paths]]
    return body, sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    body, sources = summary()
    path = HERE / "runs/summary/receipt.json"
    if args.verify:
        ok, reason = cd.Receipt.verify(
            path,
            sources=sources,
            check=lambda stored: None if stored == body else "summary differs",
        )
        if not ok:
            raise ValueError(reason)
        print(reason)
    else:
        if path.exists():
            raise FileExistsError("summary receipt already exists")
        cd.Receipt.build("cadence/temporal-address-summary/v1", body, sources).write(path)
        print(json.dumps(body, sort_keys=True))


if __name__ == "__main__":
    main()
