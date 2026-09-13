"""CSR is a lowering of the owner-local sum, including parameter changes and fallbacks."""

from __future__ import annotations

import types

import numpy as np
import pytest

import cadence as cd
from cadence import sparse


def wiring():
    # Duplicate overlaps are legal in the raw Wiring constructor; isolated owners too.
    return cd.Wiring(
        9,
        np.array([0, 0, 2, 5, 5, 3]),
        np.array([1, 1, 4, 4, 6, 0]),
        np.array([2.0, 3.0, 1.0, 0.0, 4.0, 2.0]),
        np.array([1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
    )


def owner_sum(engine, activation, blocks=None):
    inbox = np.zeros_like(activation)
    for pre, post, weight in zip(
        engine.wiring.pre, engine.wiring.post, engine.weights, strict=True
    ):
        for row in range(len(activation)):
            inbox[row, post] += activation[row, pre] * weight
    return inbox


def test_csr_preserves_duplicates_signed_weights_and_isolated_owners():
    pytest.importorskip("scipy.sparse")
    engine = cd.Settlement(wiring(), cd.GradedRule(), dense_limit=0)
    state = np.random.default_rng(12).normal(size=(32, 9))
    assert engine.to_dict()["sparse_kernel"] is None
    np.testing.assert_allclose(engine._inbox(state), owner_sum(engine, state), atol=1e-14)
    assert engine.to_dict()["sparse_kernel"] == "scipy_csr"
    assert engine._csr is not None
    assert np.shares_memory(engine._csr[1].data, engine.weights)


@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("kind", ["free", "quadratic", "grouped_softmax"])
def test_changing_input_trajectories_match_owner_sum_with_mask_nudge_and_adaptation(batch, kind):
    rule = cd.GradedRule(
        dt=0.3, gain=0.2, leak=0.02, adaptation=cd.Adaptation(tau_steps=7, strength=0.25)
    )
    engine = cd.Settlement(wiring(), rule, dense_limit=0)
    reference = cd.Settlement(engine.wiring, rule, dense_limit=0)
    reference._inbox = types.MethodType(owner_sum, reference)
    rng = np.random.default_rng(33)
    keep = np.ones(9)
    keep[3], keep[7] = 0.0, 0.5
    nudge = None
    if kind != "free":
        mask = np.zeros(9)
        mask[[0, 1, 5, 6]] = 1
        target = rng.uniform(size=(batch, 9))
        groups = np.array([0, 0, -1, -1, -1, 1, 1, -1, -1])
        nudge = cd.Nudge(
            target,
            mask,
            0.13,
            softmax_temperature=0.2 if kind == "grouped_softmax" else None,
            groups=groups,
            weight=rng.uniform(-1, 1, batch),
        )
    actual = expected = None
    for _ in range(3):
        drive = rng.uniform(0, 3, size=(batch, 9))
        actual = engine.settle_batch(
            drive, state=actual, steps=17, mask=keep, nudge=nudge, trajectory=True
        )
        expected = reference.settle_batch(
            drive, state=expected, steps=17, mask=keep, nudge=nudge, trajectory=True
        )
        for field in ("v", "activation", "adaptation", "trajectory", "repair"):
            np.testing.assert_allclose(
                getattr(actual, field), getattr(expected, field), rtol=1e-12, atol=1e-13
            )
        assert actual.steps == expected.steps == 17


def test_new_parameters_cannot_reuse_stale_csr_weights_or_mutate_old_engine():
    pytest.importorskip("scipy.sparse")
    engine = cd.Settlement(wiring(), cd.GradedRule(), dense_limit=0)
    state = np.random.default_rng(73).normal(size=(3, 9))
    original = engine._inbox(state)
    original_weights = engine.weights.copy()
    changed = engine.with_parameters(edge_scale=-engine.edge_scale, bias=np.ones(9))
    assert changed._csr is None
    np.testing.assert_allclose(changed._inbox(state), -original, atol=1e-14)
    np.testing.assert_array_equal(engine._inbox(state), original)
    np.testing.assert_array_equal(engine.weights, original_weights)
    assert changed._csr[1] is not engine._csr[1]
    assert np.shares_memory(changed._csr[1].indices, engine._csr[1].indices)
    assert np.shares_memory(changed._csr[1].indptr, engine._csr[1].indptr)
    changed.edge_scale = engine.edge_scale * 2
    assert changed._csr is None
    np.testing.assert_allclose(changed._inbox(state), 2 * original, atol=1e-14)
    changed.weights[:] *= 0.5
    np.testing.assert_allclose(changed._inbox(state), original, atol=1e-14)
    gained = engine.with_parameters(log_gain=np.log(np.full(9, 3.0)))
    np.testing.assert_allclose(gained._inbox(state), 3 * original, atol=1e-14)


def test_numpy_only_fallback_still_runs_the_original_segmented_sum(monkeypatch):
    monkeypatch.setattr(sparse, "_csr_type", lambda: None)
    engine = cd.Settlement(wiring(), cd.GradedRule(), dense_limit=0)
    state = np.random.default_rng(15).normal(size=(3, 9))
    np.testing.assert_allclose(engine._inbox(state), owner_sum(engine, state), atol=1e-14)
    assert engine.to_dict()["sparse_kernel"] == "numpy_segmented"


@pytest.mark.parametrize("owners", [0, 8])
def test_empty_graph_has_no_sparse_dependency(owners, monkeypatch):
    def forbidden():
        raise AssertionError("empty graph should never import a sparse engine")

    monkeypatch.setattr(sparse, "_csr_type", forbidden)
    w = cd.Wiring.from_edges(owners, pre=[], post=[])
    engine = cd.Settlement(w, cd.GradedRule(), dense_limit=0)
    state = np.ones((3, owners))
    np.testing.assert_array_equal(engine._inbox(state), np.zeros_like(state))
    assert engine._csr is None


def test_wiring_still_rejects_self_overlaps_before_transport():
    with pytest.raises(ValueError, match="distinct owners"):
        cd.Wiring(2, np.array([0]), np.array([0]), np.ones(1), np.ones(1))
