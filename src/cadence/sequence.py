"""Causal content readback for a sequence of observations.

The cache owns a bounded ring of *raw* feature/value pairs. All keys and the
current cue use the same running centre at read time. Keeping raw features is
essential: storing normalized deviations while their centre changes compares
vectors expressed in different coordinate systems. This module learns a mean
and records associations; it does not learn the feature encoder or an address.

Read before observing the outcome. A next-token application must score
``read(features)`` before calling ``observe(features, observed_next_token)``.
The returned value is a normalized mixture, suitable for mixing probabilities;
it should not automatically be added as a positive current to every output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SequenceRead:
    """Readback value and selectivity diagnostics, one row per stream."""

    value: np.ndarray
    entropy: np.ndarray
    maximum_weight: np.ndarray
    entries: np.ndarray


class SequenceCache:
    """A per-stream finite content cache with a shared centering frame.

    ``center_rate=0`` reproduces the uncentered cosine read. Otherwise, an
    exponential feature mean is updated only by ``observe``. Existing stored
    keys are centered again at every read, without changing their raw values.
    No label is used to make a key. A zero centered key contributes a cosine
    of zero, and an empty cache reads as zero with zero diagnostic values.

    Memory is O(batch * capacity * (features + values)). The implementation
    deliberately pays for recentering all slots and makes no speed advantage
    claim over an ordinary cosine-attention cache.
    """

    def __init__(
        self,
        features: int,
        values: int,
        *,
        capacity: int = 128,
        temperature: float = 0.1,
        center_rate: float = 0.02,
    ) -> None:
        for name, value in (("features", features), ("values", values), ("capacity", capacity)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(center_rate) or not 0 <= center_rate <= 1:
            raise ValueError("center_rate must lie in [0, 1]")
        self.features, self.values, self.capacity = features, values, capacity
        self.temperature, self.center_rate = float(temperature), float(center_rate)
        self.reset(0)

    def reset(self, batch: int) -> None:
        if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)) or batch < 0:
            raise ValueError("batch must be a nonnegative integer")
        self.keys = np.zeros((batch, self.capacity, self.features))
        self.records = np.zeros((batch, self.capacity, self.values))
        self.mean = np.zeros((batch, self.features))
        self.filled = np.zeros((batch, self.capacity), dtype=bool)
        self.head = np.zeros(batch, dtype=np.int64)
        self.observations = np.zeros(batch, dtype=np.int64)

    def _features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.features or not np.isfinite(x).all():
            raise ValueError(f"features must be a finite (batch, {self.features}) array")
        if x.shape[0] != len(self.head):
            raise ValueError("batch differs from cache; call reset(batch) explicitly")
        return x

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        return np.asarray(x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12))

    def read(self, features: np.ndarray) -> SequenceRead:
        """Query only; this never updates the centre, keys, values, or head."""
        x = self._features(features)
        unit_keys = self._unit(self.keys - self.mean[:, None, :])
        unit_cue = self._unit(x - self.mean)
        logits = (unit_keys @ unit_cue[:, :, None])[:, :, 0] / self.temperature
        # A finite sentinel also makes the entirely empty row well-defined.
        logits = np.where(self.filled, logits, -1e300)
        if not len(x):
            return SequenceRead(
                np.zeros((0, self.values)), np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
            )
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.where(self.filled, np.exp(logits), 0.0)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
        value = (weights[:, None, :] @ self.records)[:, 0, :]
        entropy = -(weights * np.log(np.maximum(weights, 1e-300))).sum(axis=1)
        return SequenceRead(value, entropy, weights.max(axis=1), self.filled.sum(axis=1))

    def observe(self, features: np.ndarray, values: np.ndarray) -> None:
        """Record an already observed association, then update the shared mean.

        Both arrays are validated before any state changes. Values can be
        arbitrary finite vectors; probability normalization is the caller's
        contract when using probability interpolation.
        """
        x = self._features(features)
        y = np.asarray(values, dtype=float)
        if y.shape != (len(x), self.values) or not np.isfinite(y).all():
            raise ValueError(f"values must be a finite (batch, {self.values}) array")
        rows = np.arange(len(x))
        self.keys[rows, self.head] = x
        self.records[rows, self.head] = y
        self.filled[rows, self.head] = True
        self.head = (self.head + 1) % self.capacity
        if self.center_rate:
            self.mean += self.center_rate * (x - self.mean)
            # The first sample initializes the mean, avoiding a long cold bias.
            self.mean[self.observations == 0] = x[self.observations == 0]
        self.observations += 1


class BoundedTrace:
    """A same-width leaky trace with bounded readback energy.

    A dense trace can overwhelm a sparse input even when each component is
    small. ``read`` rescales the whole trace to at most ``radius`` in L2 norm;
    it never amplifies weak traces. ``center=True`` removes the per-stream
    common scalar component at read time. This is an engineered normalization,
    not temporal credit assignment, and does not learn which history matters.
    """

    def __init__(
        self, width: int, *, decay: float = 0.5, radius: float = 1.0, center: bool = True
    ) -> None:
        if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1:
            raise ValueError("width must be a positive integer")
        if not np.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("decay must lie in [0, 1)")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")
        self.width, self.decay, self.radius, self.center = width, decay, radius, center
        self.reset(0)

    def reset(self, batch: int) -> None:
        if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)) or batch < 0:
            raise ValueError("batch must be a nonnegative integer")
        self.trace = np.zeros((batch, self.width))

    def read(self) -> np.ndarray:
        x = self.trace - self.trace.mean(axis=1, keepdims=True) if self.center else self.trace
        return x * np.minimum(
            1.0, self.radius / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        )

    def observe(self, value: np.ndarray) -> None:
        x = np.asarray(value, dtype=float)
        if x.shape != self.trace.shape or not np.isfinite(x).all():
            raise ValueError("value must be finite and have the current trace shape")
        self.trace = self.decay * self.trace + (1 - self.decay) * x
