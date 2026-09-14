"""Adversarial contracts for causal sequence readback."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.sequence import BoundedTrace, SequenceCache


def test_all_keys_share_the_current_center_after_distribution_shift() -> None:
    cache = SequenceCache(2, 2, center_rate=0.5, capacity=4, temperature=0.04)
    cache.reset(1)
    cache.observe([[10, 2]], [[1, 0]])
    cache.observe([[10, -2]], [[0, 1]])
    cache.observe([[30, 0]], [[0, 1]])
    result = cache.read([[10, 2]])
    assert result.value[0, 0] > 0.5
    # Independently recompute the read in ONE current coordinate system.
    raw = np.array([[10, 2], [10, -2], [30, 0]], dtype=float)
    centered = raw - cache.mean[0]
    query = np.array([10, 2]) - cache.mean[0]
    scores = centered @ query / (np.linalg.norm(centered, axis=1) * np.linalg.norm(query))
    weights = np.exp((scores - max(scores)) / 0.04)
    weights /= weights.sum()
    np.testing.assert_allclose(result.value[0], weights @ [[1, 0], [0, 1], [0, 1]])
    np.testing.assert_array_equal(cache.keys[0, :3], raw)


def test_centered_cosine_removes_a_shared_large_offset() -> None:
    cache = SequenceCache(2, 2, center_rate=0.5, temperature=0.1)
    cache.reset(1)
    cache.observe([[1000, 1]], [[1, 0]])
    cache.observe([[1000, -1]], [[0, 1]])
    assert cache.read([[1000, 1]]).value[0, 0] > 0.999
    raw = SequenceCache(2, 2, center_rate=0, temperature=0.1)
    raw.reset(1)
    raw.observe([[1000, 1]], [[1, 0]])
    raw.observe([[1000, -1]], [[0, 1]])
    assert raw.read([[1000, 1]]).maximum_weight[0] < 0.501


def test_query_does_not_mutate_any_memory_or_adapt_to_the_test_cue() -> None:
    cache = SequenceCache(3, 2)
    cache.reset(2)
    cache.observe([[1, 2, 3], [3, 2, 1]], [[1, 0], [0, 1]])
    before = {
        name: value.copy() for name, value in vars(cache).items() if isinstance(value, np.ndarray)
    }
    first = cache.read([[10, -20, 30], [-40, 50, 60]])
    again = cache.read([[10, -20, 30], [-40, 50, 60]])
    np.testing.assert_array_equal(first.value, again.value)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(cache, name), value)


def test_future_outcome_mutation_cannot_change_past_prediction() -> None:
    def predictions(outcomes: list[int]) -> np.ndarray:
        cache = SequenceCache(2, 2, center_rate=0)
        cache.reset(1)
        result = []
        for i, target in enumerate(outcomes):
            cue = [[1, i % 2]]
            result.append(cache.read(cue).value.copy())
            cache.observe(cue, np.eye(2)[[target]])
        return np.array(result)

    original = predictions([0, 1, 0, 1, 0])
    mutated = predictions([0, 1, 1, 0, 1])
    np.testing.assert_array_equal(original[:3], mutated[:3])
    assert not np.array_equal(original[3:], mutated[3:])


def test_streams_are_isolated_and_ring_evicts_only_the_oldest_record() -> None:
    cache = SequenceCache(2, 2, capacity=1, center_rate=0)
    cache.reset(2)
    cache.observe([[1, 0], [1, 0]], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(cache.read([[1, 0], [1, 0]]).value, [[1, 0], [0, 1]])
    cache.observe([[1, 0], [1, 0]], [[0.3, 0.7], [0.8, 0.2]])
    np.testing.assert_array_equal(cache.read([[1, 0], [1, 0]]).value, [[0.3, 0.7], [0.8, 0.2]])


def test_empty_and_zero_centered_features_are_finite() -> None:
    cache = SequenceCache(2, 1)
    cache.reset(1)
    out = cache.read([[0, 0]])
    np.testing.assert_array_equal(out.value, [[0]])
    np.testing.assert_array_equal(out.entropy, [0])
    cache.observe([[1, 1]], [[7]])
    np.testing.assert_array_equal(cache.read([[1, 1]]).value, [[7]])
    cache.reset(0)
    assert cache.read(np.zeros((0, 2))).value.shape == (0, 1)


@pytest.mark.parametrize("bad", [[[1, float("nan")]], [[1]], [[1, 2], [3, 4]]])
def test_bad_observation_does_not_partially_write_or_erase_stream_state(bad) -> None:
    cache = SequenceCache(2, 1)
    cache.reset(1)
    cache.observe([[1, 2]], [[1]])
    before = cache.keys.copy()
    with pytest.raises(ValueError):
        cache.observe(bad, [[2]])
    np.testing.assert_array_equal(cache.keys, before)
    with pytest.raises(ValueError):
        cache.observe([[3, 4]], [[float("inf")]])
    np.testing.assert_array_equal(cache.keys, before)


def test_bounded_trace_removes_common_component_without_increasing_weak_energy() -> None:
    trace = BoundedTrace(3, decay=0, radius=1)
    trace.reset(2)
    trace.observe([[100, 101, 99], [0, 0.1, -0.1]])
    result = trace.read()
    np.testing.assert_allclose(result.mean(axis=1), 0)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1, np.sqrt(0.02)])
    np.testing.assert_array_equal(trace.trace[0], [100, 101, 99])


def test_trace_is_owned_and_bounded_after_reused_input_mutation() -> None:
    trace = BoundedTrace(2, decay=0.5, radius=0.25, center=False)
    trace.reset(1)
    value = np.array([[1.0, -1.0]])
    trace.observe(value)
    value[:] = 100
    np.testing.assert_array_equal(trace.trace, [[0.5, -0.5]])
    assert np.linalg.norm(trace.read()) <= 0.25 + 1e-15


@pytest.mark.parametrize(
    "kwargs", [{"temperature": 0}, {"capacity": True}, {"center_rate": float("nan")}]
)
def test_bad_cache_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        SequenceCache(2, 2, **kwargs)


def test_subnormal_temperature_selects_without_nonfinite_probabilities() -> None:
    cache = SequenceCache(2, 2, center_rate=0, temperature=1e-320)
    cache.reset(1)
    cache.observe([[1, 0]], [[1, 0]])
    cache.observe([[0, 1]], [[0, 1]])
    with np.errstate(all="raise"):
        result = cache.read([[1, 0]])
    np.testing.assert_array_equal(result.value, [[1, 0]])
    assert np.isfinite(result.entropy).all()


def test_opposite_large_finite_features_have_finite_center_and_read() -> None:
    cache = SequenceCache(2, 2, center_rate=0.5, temperature=0.1)
    cache.reset(1)
    with np.errstate(all="raise"):
        cache.observe([[1e308, 0]], [[1, 0]])
        cache.observe([[-1e308, 0]], [[0, 1]])
        result = cache.read([[1e308, 0]])
    np.testing.assert_array_equal(cache.mean, [[0, 0]])
    assert result.value[0, 0] > 0.999
    assert np.isfinite(cache.keys).all()


def test_center_subtraction_and_convex_values_cannot_overflow() -> None:
    cache = SequenceCache(2, 2, center_rate=1, temperature=0.1)
    cache.reset(1)
    with np.errstate(all="raise"):
        cache.observe([[1e308, 0]], [[1e308, -1e308]])
        cache.observe([[-1e308, 0]], [[1e308, -1e308]])
        result = cache.read([[1e308, 0]])
    np.testing.assert_allclose(result.value, [[1e308, -1e308]])
    assert np.isfinite(result.value).all()


def test_tiny_nonzero_keys_retain_their_direction() -> None:
    cache = SequenceCache(2, 2, center_rate=0, temperature=0.01)
    cache.reset(1)
    cache.observe([[1e-310, 0]], [[1, 0]])
    cache.observe([[0, 1e-310]], [[0, 1]])
    assert cache.read([[1e-310, 0]]).value[0, 0] > 0.999


def test_large_finite_trace_has_finite_bounded_readback() -> None:
    trace = BoundedTrace(3, decay=0.5, radius=1, center=True)
    trace.reset(1)
    with np.errstate(all="raise"):
        trace.observe([[1e308, 1e308, -1e308]])
        result = trace.read()
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result), 1)
    np.testing.assert_allclose(result.sum(), 0, atol=1e-15)
