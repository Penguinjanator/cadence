"""Uniform-cache control using the same frozen checkpoints and prior-token budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from run import HERE, ROOT, Patch, load_data, metric, softmax, stream_windows

import cadence as cd

MIXTURES = [0.0, 0.05, 0.1, 0.2, 0.4]


def uniform(targets, vocabulary, capacity=128):
    counts = np.zeros((targets.shape[1], vocabulary))
    rows = np.arange(targets.shape[1])
    memories = []
    for t, target in enumerate(targets):
        memories.append(counts.copy() / max(min(t, capacity), 1))
        if t >= capacity:
            counts[rows, targets[t - capacity]] -= 1
        counts[rows, target] += 1
    return np.array(memories)


def verify(path):
    receipt = cd.Receipt.read(path)
    frozen = [(e["path"], path.parent / "source" / e["path"]) for e in receipt.source["files"]]
    ok, reason = cd.Receipt.verify(path, sources=frozen)
    if not ok:
        raise ValueError(reason)
    body = receipt.body
    if [r["seed"] for r in body["outcomes"]] != body["seeds"]:
        raise ValueError("missing scheduled uniform-cache outcome")
    alphabet = body["alphabet"]
    lookup = {c: i for i, c in enumerate(alphabet)}
    codes = np.array(
        [
            lookup[c]
            for c in (
                path.parent / "source/experiments/sequence_readback/data/test_decoder.txt"
            ).read_text()
        ]
    )
    _, targets, burn = stream_windows(codes, body["streams"], body["window"])
    memory = uniform(targets, len(alphabet))[burn:]
    for row in body["outcomes"]:
        if row["selected"] != min(row["search"], key=lambda x: x["validation_bpc"]):
            raise ValueError("uniform mixture not selected on validation")
        checkpoint = path.parent / row["checkpoint"]["path"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row["checkpoint"]["sha256"]:
            raise ValueError("reconstructed checkpoint changed")
        if row["validation_replay_absolute_error"] > 1e-10:
            raise ValueError("uniform control used a different checkpoint")
        artifact = path.parent / row["upstream"]["path"]
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != row["upstream"]["sha256"]:
            raise ValueError("upstream predictions changed")
        with np.load(artifact, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["targets"], targets[burn:])
            mixture = row["selected"]["mixture"]
            p = (1 - mixture) * data["base_probabilities"].astype(float) + mixture * memory
            if metric(p, targets[burn:])[0] != row["test"]:
                raise ValueError("uniform-cache test metric differs")
    return body


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("choose a fresh output directory")
    original = cd.Receipt.read(HERE / "runs/main/receipt.json")
    decoder = cd.Receipt.read(args.decoder / "receipt.json")
    if [r["seed"] for r in decoder.body["outcomes"]] != decoder.body["design"]["seeds"]:
        raise ValueError("decoder schedule must finish first")
    cfg = argparse.Namespace(**original.body["design"])
    data, metadata = load_data(cfg.train_chars, cfg.eval_chars)
    v = len(metadata["alphabet"])
    val_x, val_y, burn = stream_windows(data["validation"], cfg.eval_streams, cfg.window)
    memory = uniform(val_y, v)
    lookup = {c: i for i, c in enumerate(metadata["alphabet"])}
    test = np.array([lookup[c] for c in (HERE / "data/test_decoder.txt").read_text()])
    _, test_y, _ = stream_windows(test, cfg.eval_streams, cfg.window)
    test_memory = uniform(test_y, v)[burn:]
    args.output.mkdir(parents=True, exist_ok=True)
    files = [
        HERE / "uniform_control.py",
        HERE / "run.py",
        HERE / "data/manifest.json",
        HERE / "data/test_decoder.txt",
        HERE / "data/validation.txt",
        HERE / "data/train.txt",
        HERE / "data/test_confirmation.txt",
        HERE / "runs/main/receipt.json",
        args.decoder / "receipt.json",
    ]
    files += sorted((ROOT / "src/cadence").rglob("*.py"))
    frozen = []
    for source in files:
        relative = (
            str(source.relative_to(ROOT))
            if source.is_relative_to(ROOT)
            else "upstream/decoder_receipt.json"
        )
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        frozen.append((relative, target))
    body = {
        "seeds": decoder.body["design"]["seeds"],
        "alphabet": metadata["alphabet"],
        "streams": cfg.eval_streams,
        "window": cfg.window,
        "capacity": 128,
        "decoder_receipt_digest": decoder.digest,
        "boundary": "Uniform mean of the prior 128 tokens; no content key or retraining",
        "outcomes": [],
    }
    for prior in original.body["outcomes"]:
        seed = prior["seed"]
        net = Patch(v, cfg, seed, "echo")
        checkpoint = args.decoder / f"echo_seed{seed}.npz"
        net.learner = cd.Learner.load(checkpoint)
        logits, _ = net.run(val_x, val_y, train=False)
        p = softmax(logits / prior["patch"]["echo"]["temperature"])
        selected_echo = prior["patch"]["echo"]
        original_bpc = selected_echo["history"][selected_echo["best_epoch"] - 1]["validation"][
            "bpc"
        ]
        replay_error = abs(metric(p[burn:], val_y[burn:])[0]["bpc"] - original_bpc)
        if replay_error > 1e-10:
            raise ValueError("loaded checkpoint does not reproduce the selected validation result")
        mem = np.where(memory.sum(-1, keepdims=True) == 0, p, memory)
        search = [
            {
                "mixture": m,
                "validation_bpc": metric(((1 - m) * p + m * mem)[burn:], val_y[burn:])[0]["bpc"],
            }
            for m in MIXTURES
        ]
        selected = min(search, key=lambda x: x["validation_bpc"])
        artifact = args.decoder / f"seed{seed}.npz"
        with np.load(artifact, allow_pickle=False) as stored:
            mixed = (1 - selected["mixture"]) * stored["base_probabilities"].astype(
                float
            ) + selected["mixture"] * test_memory
            score = metric(mixed, test_y[burn:])[0]
        relative_artifact = os.path.relpath(artifact, args.output)
        row = {
            "seed": seed,
            "checkpoint": {
                "path": os.path.relpath(checkpoint, args.output),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            },
            "validation_replay_absolute_error": replay_error,
            "search": search,
            "selected": selected,
            "test": score,
            "upstream": {
                "path": relative_artifact,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            },
        }
        body["outcomes"].append(row)
        cd.Receipt.build("cadence/sequence-uniform-control/v1", body, frozen).write(
            args.output / "receipt.json"
        )
        print(json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder", type=Path, default=HERE / "runs/decoder")
    p.add_argument("--output", type=Path, default=HERE / "runs/decoder_uniform")
    p.add_argument("--verify", type=Path)
    args = p.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
