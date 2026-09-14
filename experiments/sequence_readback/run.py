"""Held-out next-character prediction: raw/normalized state and causal content reads.

Training and evaluation stream boundaries are shared by all neural arms. The
window MLP has the identical window, embedding width and hidden width. Recurrent
MLPs also receive the same-width detached trace. Independent cosine attention
is the exact structure-matched cache control.
Never uses test data to choose epochs, temperatures or mixture weights.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import cadence as cd
from cadence.sequence import BoundedTrace, SequenceCache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPERATURES = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
CACHE_TEMPERATURES = [0.02, 0.05, 0.1, 0.2]
MIXTURES = [0.0, 0.05, 0.1, 0.2, 0.4]


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)


def metric(p, y):
    shape = y.shape
    p = p.reshape(-1, p.shape[-1])
    y = y.reshape(-1)
    nll = -np.log2(np.maximum(p[np.arange(len(y)), y], 1e-12))
    correct = p.argmax(axis=1) == y
    return (
        {"bpc": float(nll.mean()), "accuracy": float(correct.mean()), "tokens": int(y.size)},
        nll.reshape(shape),
        correct.reshape(shape),
    )


def load_data(train_chars, eval_chars):
    meta = json.loads((HERE / "data/manifest.json").read_text())
    alphabet = meta["alphabet"]
    lookup = {ch: i for i, ch in enumerate(alphabet)}
    data = {}
    for split in ("train", "validation", "test"):
        filename = "test_confirmation.txt" if split == "test" else f"{split}.txt"
        raw = (HERE / "data" / filename).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == meta["splits"][split]["sha256"]
        count = train_chars if split == "train" else eval_chars
        data[split] = np.array([lookup[ch] for ch in raw.decode()[:count]], dtype=np.int64)
    return data, meta


def stream_windows(codes, streams, window, burn=16):
    rows = codes[: len(codes) // streams * streams].reshape(streams, -1)
    positions = range(window, rows.shape[1])
    windows = np.stack([rows[:, t - window : t] for t in positions])
    targets = np.stack([rows[:, t] for t in positions])
    return windows, targets, max(0, burn - window)


class Patch:
    def __init__(self, v, cfg, seed, mode):
        self.mode, self.v, self.cfg = mode, v, cfg
        if mode == "window":
            c, tie = cd.embedded(v, cfg.window, cfg.dim, cfg.hidden, v, seed=seed)
        else:
            c, tie = cd.stateful(v, cfg.window, cfg.dim, cfg.hidden, v, seed=seed)
        self.c = c
        self.context = list(c.populations.get("context", []))
        self.hidden = list(c.populations["hidden"])
        self.trace = BoundedTrace(cfg.hidden, radius=cfg.radius, center=True)
        self.learner = cd.Learner(
            cd.Brain(c, cd.learning_neuron_model(dt=1.0), dense_limit=32768),
            c.populations["output"],
            cd.LearnerConfig(
                eta=cfg.eta,
                eta_bias=0.03,
                beta=0.1,
                temperature=0.1,
                tolerance=0.003,
                free_steps=60,
                nudged_steps=12,
            ),
            tie_groups=tie,
        )

    def run(self, x, y, train, epoch=0):
        self.trace.reset(x.shape[1])
        if train:
            self.learner.config = dataclasses.replace(
                self.learner.config,
                eta=self.cfg.eta * 0.85**epoch,
                eta_bias=0.03 * 0.85**epoch,
                tolerance=0.003,
            )
        else:
            self.learner.config = dataclasses.replace(self.learner.config, tolerance=0.0001)
        outputs, features, norms, steps = [], [], [], []
        warm = None
        for window, target in zip(x, y, strict=True):
            drive = np.zeros((len(window), self.c.n))
            drive[
                np.arange(len(window))[:, None],
                window + np.arange(self.cfg.window)[None, :] * self.v,
            ] = 1
            if self.context:
                read = self.trace.read() if self.mode == "bounded" else self.trace.trace
                norms.append(float(np.linalg.norm(read, axis=1).mean()))
                drive[:, self.context] = read
            if train:
                learned, report = self.learner.step(drive, target, warm=warm)
                free = learned.free
                steps.append(report["free_steps"])
            else:
                free = self.learner.free(drive, warm=warm)
            warm = free
            h = free.activation[:, self.hidden]
            if self.context:
                self.trace.observe(h)
            if not train:
                outputs.append(free.activation[:, self.learner.output_index])
                features.append(h)
        if train:
            return {
                "mean_readback_norm": float(np.mean(norms)) if norms else 0.0,
                "mean_free_steps": float(np.mean(steps)),
            }
        return np.array(outputs), np.array(features)


class WindowMLP(nn.Module):
    def __init__(self, v, cfg, mode="window"):
        super().__init__()
        self.mode = mode
        self.context = nn.Linear(cfg.hidden, cfg.hidden, bias=False) if mode != "window" else None
        self.trace = BoundedTrace(cfg.hidden, radius=cfg.radius)
        self.embed = nn.Embedding(v, cfg.dim)
        self.hidden = nn.Linear(cfg.window * cfg.dim, cfg.hidden)
        self.out = nn.Linear(cfg.hidden, v)

    def forward(self, x):
        a = self.hidden(self.embed(x).flatten(1))
        if self.context is not None:
            read = self.trace.read() if self.mode == "bounded" else self.trace.trace
            a = a + self.context(torch.as_tensor(read, dtype=a.dtype))
        h = torch.relu(a)
        if self.context is not None:
            self.trace.observe(h.detach().numpy())
        return self.out(h)


def cache_read(features, targets, vocabulary, temperature, mode, capacity=128):
    """Score first and store the observed next token second, separately per stream."""
    cache = SequenceCache(
        features.shape[-1],
        vocabulary,
        capacity=capacity,
        temperature=temperature,
        center_rate=0.02 if mode != "raw" else 0.0,
    )
    cache.reset(features.shape[1])
    stale = np.zeros_like(cache.keys)
    out, entropy, maximum, counts = [], [], [], []
    eye = np.eye(vocabulary)
    reference_error = 0.0
    for x, y in zip(features, targets, strict=True):
        if mode == "legacy_stale":
            query = cache._unit(x - cache.mean)
            logit = (stale @ query[:, :, None])[:, :, 0] / temperature
            logit = np.where(cache.filled, logit, -1e300)
            weights = softmax(logit) * cache.filled
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
            out.append((weights[:, None, :] @ cache.records)[:, 0, :])
            entropy.append(-(weights * np.log(np.maximum(weights, 1e-300))).sum(axis=1))
            maximum.append(weights.max(axis=1))
            counts.append(cache.filled.sum(axis=1))
            stale[np.arange(len(x)), cache.head] = query
        else:
            read = cache.read(x)
            out.append(read.value)
            entropy.append(read.entropy)
            maximum.append(read.maximum_weight)
            counts.append(read.entries)
            # Separate scalar-stream implementation with identical information and budget.
            for row in range(len(x)):
                valid = np.flatnonzero(cache.filled[row])
                if not len(valid):
                    expected = np.zeros(vocabulary)
                else:
                    ks = cache.keys[row, valid] - cache.mean[row]
                    q = x[row] - cache.mean[row]
                    similarity = np.sum(ks * q, axis=1) / (
                        np.maximum(np.linalg.norm(ks, axis=1), 1e-12)
                        * max(np.linalg.norm(q), 1e-12)
                    )
                    attention = softmax(similarity / temperature)
                    expected = attention @ cache.records[row, valid]
                reference_error = max(
                    reference_error, float(np.max(np.abs(expected - read.value[row])))
                )
        cache.observe(x, eye[y])
    return np.array(out), {
        "mean_entropy": float(np.mean(entropy)),
        "mean_max_weight": float(np.mean(maximum)),
        "mean_entries": float(np.mean(counts)),
        "matched_cosine_max_abs_error": reference_error,
    }


def cache_comparison(val_features, test_features, val_y, test_y, val_p, test_p, burn):
    v = val_p.shape[-1]
    result, arrays = {}, {}
    for mode in ("raw", "legacy_stale", "centered"):
        best = None
        search = []
        for t in CACHE_TEMPERATURES:
            memory, diagnostic = cache_read(val_features, val_y, v, t, mode)
            absent = memory.sum(axis=-1, keepdims=True) == 0
            memory = np.where(absent, val_p, memory)
            for mix in MIXTURES:
                score, _, _ = metric((1 - mix) * val_p[burn:] + mix * memory[burn:], val_y[burn:])
                search.append(
                    {
                        "temperature": t,
                        "mixture": mix,
                        "validation": score,
                        "selectivity": diagnostic,
                    }
                )
                candidate = (score["bpc"], t, mix, diagnostic)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        _, temp, mix, val_diagnostic = best
        memory, test_diagnostic = cache_read(test_features, test_y, v, temp, mode)
        memory = np.where(memory.sum(axis=-1, keepdims=True) == 0, test_p, memory)
        p = (1 - mix) * test_p[burn:] + mix * memory[burn:]
        score, nll, hit = metric(p, test_y[burn:])
        result[mode] = {
            "validation_bpc": best[0],
            "search": search,
            "temperature": temp,
            "mixture": mix,
            "test": score,
            "validation_selectivity": val_diagnostic,
            "test_selectivity": test_diagnostic,
        }
        arrays[mode + "_nll"], arrays[mode + "_hit"] = nll, hit
    return result, arrays


def sources():
    # Copy the complete package recursively, including circuits and future subpackages.
    files = [HERE / "run.py", HERE / "data/manifest.json"]
    files += sorted((ROOT / "src/cadence").rglob("*.py"))
    files += [
        HERE / "data" / name for name in ("train.txt", "validation.txt", "test_confirmation.txt")
    ]
    return [(str(p.relative_to(ROOT)), p) for p in files]


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a fresh --output directory")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be distinct")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    data, metadata = load_data(args.train_chars, args.eval_chars)
    v = len(metadata["alphabet"])
    train_x, train_y, _ = stream_windows(data["train"], args.batch, args.window)
    val_x, val_y, burn = stream_windows(data["validation"], args.eval_streams, args.window)
    test_x, test_y, test_burn = stream_windows(data["test"], args.eval_streams, args.window)
    assert burn == test_burn
    counts = np.ones((v, v))
    np.add.at(counts, (data["train"][:-1], data["train"][1:]), 1)
    counts /= counts.sum(axis=1, keepdims=True)
    bigram, _, _ = metric(counts[test_x[burn:, :, -1]], test_y[burn:])
    body = {
        "design": vars(args) | {"output": str(args.output)},
        "data": metadata,
        "bigram": bigram,
        "outcomes": [],
        "boundaries": [
            "Small-book pilot, not an R31/R32 reproduction or general language result",
            "Held-out books and frozen slow weights; cache sees only past observed tokens",
            "Readback learns no semantic encoder or temporal credit",
            "Conventional cosine attention with identical keys is an equality control",
            "Every configuration selected on validation only, test scored after selection",
            "No additional hidden neurons or trainable parameters in bounded versus raw state",
        ],
    }
    cfg = args
    # Freeze the executed source bytes before the first comparison.
    frozen = []
    for relative, source in sources():
        copy = args.output / "source" / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(source.read_bytes())
        frozen.append((relative, copy))
    for seed in args.seeds:
        seed_data = {"seed": seed, "patch": {}}
        arrays = {"test_targets": test_y[burn:]}
        for mode in args.modes:
            net = Patch(v, cfg, seed, mode)
            start = time.perf_counter()
            history, best, saved = [], None, None
            for epoch in range(args.epochs):
                training = net.run(train_x, train_y, True, epoch)
                logits, features = net.run(val_x, val_y, False)
                temperature = min(
                    TEMPERATURES,
                    key=lambda t: metric(softmax(logits[burn:] / t), val_y[burn:])[0]["bpc"],
                )
                score, _, _ = metric(softmax(logits[burn:] / temperature), val_y[burn:])
                history.append(
                    {
                        "epoch": epoch + 1,
                        "validation": score,
                        "temperature": temperature,
                        **training,
                    }
                )
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "mode": mode,
                            **history[-1],
                            "elapsed": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
                if best is None or score["bpc"] < best:
                    best = score["bpc"]
                    saved = (
                        net.learner.brain.efficacy.copy(),
                        net.learner.brain.bias.copy(),
                        temperature,
                        epoch + 1,
                    )
            seconds = time.perf_counter() - start
            efficacy, bias, temperature, best_epoch = saved
            net.learner.brain = net.learner.brain.with_parameters(efficacy=efficacy, bias=bias)
            val_logits, val_features = net.run(val_x, val_y, False)
            test_logits, test_features = net.run(test_x, test_y, False)
            val_p, test_p = softmax(val_logits / temperature), softmax(test_logits / temperature)
            score, nll, hit = metric(test_p[burn:], test_y[burn:])
            arrays[mode + "_nll"], arrays[mode + "_hit"] = nll, hit
            caches, cache_arrays = cache_comparison(
                val_features, test_features, val_y, test_y, val_p, test_p, burn
            )
            for key, arr in cache_arrays.items():
                arrays[mode + "_cache_" + key] = arr
            seed_data["patch"][mode] = {
                "history": history,
                "best_epoch": best_epoch,
                "temperature": temperature,
                "test": score,
                "cache": caches,
                "train_seconds": seconds,
                "neurons": net.c.n,
                "trainable_parameters": int(net.learner.to_dict()["parameters"]),
                "test_hidden_mean": float(test_features.mean()),
                "test_hidden_std": float(test_features.std()),
                "test_hidden_saturated": float((test_features > 0.9).mean()),
            }
        seed_data["mlp"] = {}
        for mlp_mode in args.modes:
            torch.manual_seed(seed)
            mlp = WindowMLP(v, cfg, mlp_mode)
            optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
            x, y = torch.tensor(train_x), torch.tensor(train_y)
            vx, tx = torch.tensor(val_x), torch.tensor(test_x)
            best, saved, history = None, None, []
            start = time.perf_counter()
            for epoch in range(args.epochs):
                mlp.trace.reset(args.batch)
                for xx, yy in zip(x, y, strict=True):
                    optimizer.zero_grad()
                    nn.functional.cross_entropy(mlp(xx), yy).backward()
                    optimizer.step()
                mlp.trace.reset(args.eval_streams)
                with torch.no_grad():
                    logits = torch.stack([mlp(xx) for xx in vx]).numpy()
                temp = min(
                    [0.5, 0.7, 1.0, 1.5, 2.0],
                    key=lambda t: metric(softmax(logits[burn:] / t), val_y[burn:])[0]["bpc"],
                )
                score, _, _ = metric(softmax(logits[burn:] / temp), val_y[burn:])
                history.append({"epoch": epoch + 1, "validation": score, "temperature": temp})
                if best is None or score["bpc"] < best:
                    best, saved = (
                        score["bpc"],
                        ({k: p.clone() for k, p in mlp.state_dict().items()}, temp, epoch + 1),
                    )
            seconds = time.perf_counter() - start
            weights, temp, epoch = saved
            mlp.load_state_dict(weights)
            mlp.trace.reset(args.eval_streams)
            with torch.no_grad():
                p = softmax(torch.stack([mlp(xx) for xx in tx]).numpy() / temp)
            score, nll, hit = metric(p[burn:], test_y[burn:])
            arrays["mlp_" + mlp_mode + "_nll"], arrays["mlp_" + mlp_mode + "_hit"] = nll, hit
            seed_data["mlp"][mlp_mode] = {
                "history": history,
                "best_epoch": epoch,
                "test": score,
                "train_seconds": seconds,
                "parameters": sum(p.numel() for p in mlp.parameters()),
            }
        path = args.output / f"seed{seed}.npz"
        np.savez_compressed(path, **arrays)
        seed_data["per_token_artifact"] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        body["outcomes"].append(seed_data)
        cd.Receipt.build("cadence/sequence-readback/v1", body, frozen).write(
            args.output / "receipt.json"
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "summary": {k: d["test"] for k, d in seed_data["patch"].items()},
                    "mlp": {k: d["test"] for k, d in seed_data["mlp"].items()},
                }
            ),
            flush=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=HERE / "runs/main")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["window", "echo", "bounded"],
        default=["window", "echo", "bounded"],
    )
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--train-chars", type=int, default=60000)
    p.add_argument("--eval-chars", type=int, default=10000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval-streams", type=int, default=16)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--eta", type=float, default=1.5)
    p.add_argument("--radius", type=float, default=1.0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
