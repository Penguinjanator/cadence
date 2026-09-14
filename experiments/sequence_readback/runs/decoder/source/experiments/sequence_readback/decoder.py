"""Paired decoder follow-up: positive output current versus probability readback.

Reconstruct the five validation-selected Echo checkpoints from the completed
confirmation. Slow weights never learn during decoder evaluation. Raw cosine
keys, capacity, cache temperature and read-before-observe timing are identical.
The injected-current arm feeds its settled state back into Echo, as in R32.
This is an R32-style decoder comparison on fixed checkpoints, not a reproduction
of author training with an active notebook. A new disjoint test excerpt is used.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from run import HERE, ROOT, TEMPERATURES, Patch, load_data, metric, softmax, stream_windows

import cadence as cd
from cadence.sequence import SequenceCache

AMPLITUDES = [0.0, 0.1, 0.3, 1.0]
MIXTURES = [0.0, 0.05, 0.1, 0.2, 0.4]


def outputs(net, windows, targets, amplitude, *, second_phase=False):
    """The current target is used only in observe, after its output has been read."""
    streams = windows.shape[1]
    net.trace.reset(streams)
    net.learner.config = dataclasses.replace(net.learner.config, tolerance=1e-4)
    cache = SequenceCache(net.cfg.hidden, net.v, capacity=128, temperature=0.1, center_rate=0)
    cache.reset(streams)
    eye, warm = np.eye(net.v), None
    logits, memories = [], []
    for window, target in zip(windows, targets, strict=True):
        drive = np.zeros((streams, net.c.n))
        drive[np.arange(streams)[:, None], window + np.arange(net.cfg.window)[None, :] * net.v] = 1
        drive[:, net.context] = net.trace.trace
        probe = net.learner.free(drive, warm=warm)
        key = probe.activation[:, net.hidden]
        read = cache.read(key).value
        if amplitude or second_phase:
            drive[:, net.learner.output_index] += amplitude * read
            state = net.learner.free(drive, warm=probe)
        else:
            state = probe
        logits.append(state.activation[:, net.learner.output_index])
        memories.append(read)
        # R32 writes the probe key; its carried state follows the read-influenced phase.
        net.trace.observe(state.activation[:, net.hidden])
        cache.observe(key, eye[target])
        warm = state
    return np.array(logits), np.array(memories)


def sources(main_receipt):
    files = [
        HERE / "decoder.py",
        HERE / "run.py",
        main_receipt,
        HERE / "data/manifest.json",
        HERE / "data/decoder_test_manifest.json",
        HERE / "data/train.txt",
        HERE / "data/validation.txt",
        HERE / "data/test_decoder.txt",
    ]
    files += sorted((ROOT / "src/cadence").rglob("*.py"))
    return [(str(p.relative_to(ROOT)), p) for p in files]


def verify(path):
    receipt = cd.Receipt.read(path)
    frozen = [(e["path"], path.parent / "source" / e["path"]) for e in receipt.source["files"]]
    ok, reason = cd.Receipt.verify(path, sources=frozen)
    if not ok:
        raise ValueError(reason)
    body = receipt.body
    if [r["seed"] for r in body["outcomes"]] != body["design"]["seeds"]:
        raise ValueError("missing scheduled seed")
    for row in body["outcomes"]:
        if row["validation_replay_absolute_error"] > 1e-10:
            raise ValueError("checkpoint reconstruction differs from confirmation")
        artifact = path.parent / row["artifact"]["path"]
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != row["artifact"]["sha256"]:
            raise ValueError("artifact hash mismatch")
        for arm in ("injected_current", "probability_readback"):
            if row[arm]["selected"] != min(row[arm]["search"], key=lambda s: s["validation_bpc"]):
                raise ValueError("configuration is not the validation winner")
        with np.load(artifact, allow_pickle=False) as data:
            for arm in ("base", "injected_current", "probability_readback"):
                p, y = data[arm + "_probabilities"], data["targets"]
                if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1):
                    raise ValueError("invalid prediction distribution")
                score, _, _ = metric(p, y)
                if any(abs(score[key] - row[arm]["test"][key]) > 1e-12 for key in score):
                    raise ValueError("test aggregate differs from stored probabilities")
    return body


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("choose a fresh output directory")
    original = cd.Receipt.read(args.main_receipt)
    schedule = original.body["design"]
    if [r["seed"] for r in original.body["outcomes"]] != schedule["seeds"] or len(
        schedule["seeds"]
    ) < 5:
        raise ValueError("the original five-seed schedule must be complete first")
    cfg = argparse.Namespace(**schedule)
    data, metadata = load_data(cfg.train_chars, cfg.eval_chars)
    lookup = {c: i for i, c in enumerate(metadata["alphabet"])}
    test_meta = json.loads((HERE / "data/decoder_test_manifest.json").read_text())
    raw = (HERE / "data/test_decoder.txt").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == test_meta["sha256"]
    test = np.array([lookup[ch] for ch in raw.decode()], dtype=np.int64)
    train_x, train_y, _ = stream_windows(data["train"], cfg.batch, cfg.window)
    val_x, val_y, burn = stream_windows(data["validation"], cfg.eval_streams, cfg.window)
    test_x, test_y, test_burn = stream_windows(test, cfg.eval_streams, cfg.window)
    assert burn == test_burn
    args.output.mkdir(parents=True)
    frozen = []
    for relative, source in sources(args.main_receipt):
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        frozen.append((relative, target))
    body = {
        "design": {
            "seeds": schedule["seeds"],
            "source_receipt": original.digest,
            "amplitudes": AMPLITUDES,
            "mixtures": MIXTURES,
            "cache_temperature": 0.1,
            "capacity": 128,
            "test": test_meta,
        },
        "boundaries": [
            "Exploratory paired decoder comparison on validation-selected Echo checkpoints",
            "New disjoint test excerpt; all decoder settings selected on validation",
            "Both arms freeze slow weights and observe outcomes only after prediction",
            "R32-style positive-current injection; does not reproduce original notebook training",
            "Raw cosine keys in both arms; any gain is not attributed to centering",
            "Injection includes its second settling phase even at zero amplitude",
        ],
        "outcomes": [],
    }
    for previous in original.body["outcomes"]:
        seed, selected = previous["seed"], previous["patch"]["echo"]
        net = Patch(len(lookup), cfg, seed, "echo")
        start = time.perf_counter()
        for epoch in range(selected["best_epoch"]):
            net.run(train_x, train_y, train=True, epoch=epoch)
        reconstructed_seconds = time.perf_counter() - start
        net.learner.save(args.output / f"echo_seed{seed}.npz")
        val_logits, val_memory = outputs(net, val_x, val_y, amplitude=0)
        temperature = selected["temperature"]
        base_val = softmax(val_logits / temperature)
        original_bpc = selected["history"][selected["best_epoch"] - 1]["validation"]["bpc"]
        reconstructed_bpc = metric(base_val[burn:], val_y[burn:])[0]["bpc"]
        replay_error = abs(original_bpc - reconstructed_bpc)
        if replay_error > 1e-10:
            raise ValueError(f"checkpoint replay error for seed {seed}: {replay_error}")
        val_memory = np.where(val_memory.sum(-1, keepdims=True) == 0, base_val, val_memory)
        mixture_search = []
        for mixture in MIXTURES:
            p = (1 - mixture) * base_val[burn:] + mixture * val_memory[burn:]
            mixture_search.append(
                {"mixture": mixture, "validation_bpc": metric(p, val_y[burn:])[0]["bpc"]}
            )
        mixture_best = min(mixture_search, key=lambda r: r["validation_bpc"])
        injection_search = []
        for amplitude in AMPLITUDES:
            current_logits = outputs(net, val_x, val_y, amplitude, second_phase=True)[0]
            for t in TEMPERATURES:
                bpc = metric(softmax(current_logits[burn:] / t), val_y[burn:])[0]["bpc"]
                injection_search.append(
                    {"amplitude": amplitude, "temperature": t, "validation_bpc": bpc}
                )
        injection_best = min(injection_search, key=lambda r: r["validation_bpc"])
        test_logits, test_memory = outputs(net, test_x, test_y, amplitude=0)
        base_p = softmax(test_logits / temperature)
        test_memory = np.where(test_memory.sum(-1, keepdims=True) == 0, base_p, test_memory)
        mix = mixture_best["mixture"]
        probabilities = {
            "base": base_p[burn:],
            "probability_readback": ((1 - mix) * base_p + mix * test_memory)[burn:],
        }
        injected_logits = outputs(
            net, test_x, test_y, injection_best["amplitude"], second_phase=True
        )[0]
        probabilities["injected_current"] = softmax(
            injected_logits[burn:] / injection_best["temperature"]
        )
        probabilities = {arm: p.astype(np.float32) for arm, p in probabilities.items()}
        row = {
            "seed": seed,
            "reconstructed_epoch": selected["best_epoch"],
            "training_reconstruction_seconds": reconstructed_seconds,
            "validation_replay_absolute_error": replay_error,
            "base": {},
            "injected_current": {"selected": injection_best, "search": injection_search},
            "probability_readback": {"selected": mixture_best, "search": mixture_search},
        }
        arrays = {"targets": test_y[burn:]}
        for arm, p in probabilities.items():
            row[arm]["test"] = metric(p, test_y[burn:])[0]
            arrays[arm + "_probabilities"] = p
        artifact = args.output / f"seed{seed}.npz"
        np.savez_compressed(artifact, **arrays)
        row["artifact"] = {
            "path": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
        body["outcomes"].append(row)
        cd.Receipt.build("cadence/sequence-decoder/v1", body, frozen).write(
            args.output / "receipt.json"
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "results": {arm: row[arm]["test"] for arm in probabilities},
                    "injection": injection_best,
                    "mixture": mixture_best,
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-receipt", type=Path, default=HERE / "runs/main/receipt.json")
    parser.add_argument("--output", type=Path, default=HERE / "runs/decoder")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
