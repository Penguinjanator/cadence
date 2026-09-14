"""Generate the paired decoder comparison from independently verified input receipts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from decoder import verify as verify_decoder
from uniform_control import verify as verify_uniform

from cadence.receipts import Receipt, canonical_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def summarize():
    decoder_path = HERE / "runs/decoder/receipt.json"
    uniform_path = HERE / "runs/decoder_uniform/receipt.json"
    decoder = verify_decoder(decoder_path)
    uniform = verify_uniform(uniform_path)
    seeds = [r["seed"] for r in decoder["outcomes"]]
    if seeds != [r["seed"] for r in uniform["outcomes"]]:
        raise ValueError("paired seeds differ")
    values = {
        name: np.array([r[name]["test"]["bpc"] for r in decoder["outcomes"]])
        for name in ("base", "injected_current", "probability_readback")
    }
    values["uniform_readback"] = np.array([r["test"]["bpc"] for r in uniform["outcomes"]])
    stats = {
        name: {
            "per_seed_bpc": v.tolist(),
            "mean_bpc": float(v.mean()),
            "sample_sd_bpc": float(v.std(ddof=1)),
        }
        for name, v in values.items()
    }
    paired = {}
    for name in ("base", "injected_current", "uniform_readback"):
        diff = values[name] - values["probability_readback"]
        paired[name + "_minus_probability"] = {
            "per_seed_bpc_difference": diff.tolist(),
            "mean_bpc_difference": float(diff.mean()),
            "sample_sd_bpc_difference": float(diff.std(ddof=1)),
            "probability_readback_wins": int((diff > 0).sum()),
        }
    body = {
        "seeds": seeds,
        "test": decoder["design"]["test"],
        "tokens_per_seed": decoder["outcomes"][0]["base"]["test"]["tokens"],
        "arms": stats,
        "paired": paired,
        "decoder_receipt_digest": Receipt.read(decoder_path).digest,
        "uniform_receipt_digest": Receipt.read(uniform_path).digest,
        "boundary": "SD is across five initializations on one fixed excerpt, not across corpora",
    }
    files = [Path(__file__).resolve(), decoder_path, uniform_path]
    sources = [(str(p.relative_to(ROOT)), p) for p in files]
    return body, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    body, sources = summarize()
    path = HERE / "runs/decoder_comparison/receipt.json"
    if args.verify:
        ok, reason = Receipt.verify(
            path,
            sources=sources,
            check=lambda stored: None if stored == body else "summary differs",
        )
        if not ok:
            raise ValueError(reason)
        print(reason)
    else:
        if path.exists():
            raise FileExistsError("comparison receipt already exists")
        Receipt.build("cadence/sequence-decoder-comparison/v1", body, sources).write(path)
        print(canonical_json(body))


if __name__ == "__main__":
    main()
