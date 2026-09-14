"""False-green probes for the experiment receipt verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cadence.receipts import Receipt

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sequence_verifier", HERE / "experiments/sequence_readback/verify.py"
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def receipt(tmp_path: Path) -> Path:
    """A minimal valid diagnostic receipt; no accuracy claim is involved."""
    import hashlib

    arrays = tmp_path / "seed0.npz"
    np.savez_compressed(arrays, test_targets=np.zeros((1, 2), dtype=int))
    body = {
        "design": {"seeds": [0], "modes": []},
        "bigram": {},
        "outcomes": [
            {
                "seed": 0,
                "patch": {},
                "mlp": {},
                "per_token_artifact": {
                    "path": arrays.name,
                    "sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
                },
            }
        ],
    }
    return Receipt.build("cadence/sequence-readback/v1", body).write(tmp_path / "receipt.json")


def test_missing_scheduled_seed_fails_even_with_a_fresh_valid_digest(tmp_path) -> None:
    path = receipt(tmp_path)
    assert VERIFIER.verify(path)["seeds"] == [0]
    r = Receipt.read(path)
    r.body["design"]["seeds"] = [0, 1, 2, 3, 4]
    Receipt.build(r.kind, r.body).write(path)
    with pytest.raises(ValueError, match="missing"):
        VERIFIER.verify(path)


def test_token_artifact_mutation_fails(tmp_path) -> None:
    path = receipt(tmp_path)
    (tmp_path / "seed0.npz").write_bytes(b"fabricated")
    with pytest.raises(ValueError, match="hash mismatch"):
        VERIFIER.verify(path)


def test_duplicate_seed_fails_even_with_a_fresh_valid_digest(tmp_path) -> None:
    path = receipt(tmp_path)
    r = Receipt.read(path)
    r.body["outcomes"] *= 2
    Receipt.build(r.kind, r.body).write(path)
    with pytest.raises(ValueError, match="duplicate"):
        VERIFIER.verify(path)


def test_patch_evaluation_never_receives_the_current_or_future_target() -> None:
    """Mutation of the entire held-out target tape must leave slow features unchanged."""
    from argparse import Namespace

    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        "sequence_runner", HERE / "experiments/sequence_readback/run.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = Namespace(window=2, dim=2, hidden=4, radius=1.0, eta=1.5)
    patch = runner.Patch(3, cfg, seed=0, mode="bounded")
    rng = np.random.default_rng(0)
    windows = rng.integers(0, 3, size=(5, 2, 2))
    target = rng.integers(0, 3, size=(5, 2))
    before_weights = patch.learner.brain.efficacy.copy()
    before_bias = patch.learner.brain.bias.copy()
    original, features = patch.run(windows, target, train=False)
    mutated, mutated_features = patch.run(windows, (target + 1) % 3, train=False)
    np.testing.assert_array_equal(original, mutated)
    np.testing.assert_array_equal(features, mutated_features)
    np.testing.assert_array_equal(patch.learner.brain.efficacy, before_weights)
    np.testing.assert_array_equal(patch.learner.brain.bias, before_bias)
