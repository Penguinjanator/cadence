"""Explicit bounded experience rehearsal; labels and memory cost are never implicit.

Reactivation of stored experience motivates replay, but reservoir sampling is a
standard algorithm, not a proposed biological mechanism. This store can feed the
existing local Learner.step or a conventional optimizer on exactly the same data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stream import FastSynapses


@dataclass
class ReservoirReplay:
    """A uniform reservoir of past feature/label pairs, with no task identifiers.

    Call ``sample`` before ``observe`` to rehearse only prior experience. Each
    observed row has probability ``min(1, capacity / seen)`` of being retained.
    Separate admission/sampling RNGs keep storage identical when replay budgets
    differ. The caller owns learning, target availability, and stream isolation.
    Only scalar integer labels are supported; reward trajectories need their own
    transition/credit interface.
    """

    capacity: int
    inputs: int
    seed: int = 0
    features: np.ndarray = field(init=False, repr=False)
    labels: np.ndarray = field(init=False, repr=False)
    seen: int = field(init=False, default=0)
    size: int = field(init=False, default=0)
    sampled: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        for name in ("capacity", "inputs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        seeds = np.random.SeedSequence(self.seed).spawn(2)
        self._admission = np.random.default_rng(seeds[0])
        self._sampling = np.random.default_rng(seeds[1])
        self.features = np.zeros((self.capacity, self.inputs))
        self.labels = np.zeros(self.capacity, dtype=np.int64)

    def observe(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Offer actual observed examples, copying selected rows into owned storage."""
        features = FastSynapses._port(features, self.inputs, "features")
        labels = np.asarray(labels)
        if (
            labels.shape != (len(features),)
            or labels.dtype.kind not in "iu"
            or (labels < 0).any()
            or (labels > np.iinfo(np.int64).max).any()
        ):
            raise ValueError("labels must be nonnegative int64-compatible indices per row")
        for x, y in zip(features, labels, strict=True):
            self.seen += 1
            slot = (
                self.seen - 1
                if self.size < self.capacity
                else int(self._admission.integers(self.seen))
            )
            if slot < self.capacity:
                self.features[slot] = x
                self.labels[slot] = y
                self.size = min(self.size + 1, self.capacity)

    def sample(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """Return up to ``count`` distinct retained rows, as independently owned copies."""
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        count = min(count, self.size)
        indices = self._sampling.choice(self.size, size=count, replace=False)
        self.sampled += count
        return self.features[indices].copy(), self.labels[indices].copy()

    def to_dict(self) -> dict[str, int | str]:
        return {
            "kind": "uniform-reservoir-replay",
            "capacity": self.capacity,
            "inputs": self.inputs,
            "seed": self.seed,
            "seen": self.seen,
            "size": self.size,
            "sampled": self.sampled,
            "mutable_bytes": self.features.nbytes + self.labels.nbytes,
        }
