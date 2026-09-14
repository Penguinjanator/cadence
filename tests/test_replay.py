from __future__ import annotations

import numpy as np
import pytest

from cadence.replay import ReservoirReplay


def test_retains_prior_observations_only_and_owns_storage() -> None:
    replay = ReservoirReplay(4, 2, seed=3)
    assert replay.sample(4)[0].shape == (0, 2)
    x, y = np.eye(2), np.array([1, 2])
    replay.observe(x, y)
    x.fill(-1)
    y.fill(0)
    stored, labels = replay.sample(5)
    assert sorted(labels.tolist()) == [1, 2]
    assert np.array_equal(stored.sum(axis=0), [1, 1])
    stored.fill(100)
    assert (replay.features <= 1).all()


def test_admission_does_not_depend_on_sampling_and_reservoir_is_bounded() -> None:
    a, b = (ReservoirReplay(8, 1, seed=7) for _ in range(2))
    for i in range(100):
        x, y = np.array([[float(i)]]), np.array([i])
        a.observe(x, y)
        b.observe(x, y)
        b.sample(4)
    assert np.array_equal(a.features, b.features)
    assert np.array_equal(a.labels, b.labels)
    assert a.size == b.size == 8 and a.seen == b.seen == 100
    assert np.array_equal(a.features[:, 0], a.labels)
    assert a.to_dict()["mutable_bytes"] == 8 * 2 * 8


def test_reservoir_is_not_a_recent_window_or_class_oracle() -> None:
    # Inclusion frequencies across independent seeds reject a FIFO or biased admission.
    hits = np.zeros(40)
    for seed in range(400):
        replay = ReservoirReplay(8, 1, seed=seed)
        replay.observe(np.arange(40)[:, None], np.arange(40))
        hits[replay.labels] += 1
    assert hits.min() > 45 and hits.max() < 115
    assert abs(hits[:20].mean() - hits[20:].mean()) < 10


def test_invalid_batch_does_not_partially_admit() -> None:
    replay = ReservoirReplay(2, 1)
    for x, y in [
        (np.array([[1.0], [np.nan]]), np.array([1, 2])),
        (np.array([[1.0], [2.0]]), np.array([1.0, 2.0])),
        (np.array([[1.0], [2.0]]), np.array([1, -1])),
    ]:
        with pytest.raises(ValueError):
            replay.observe(x, y)
    assert replay.seen == replay.size == 0
    with pytest.raises(ValueError):
        replay.sample(True)
