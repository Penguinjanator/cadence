"""Bounded content retrieval with locally learned competitive prototypes.

The caller supplies observable features and values, never a slot or task identity.
Cosine competition selects a prototype, novelty recruits a slot, and a local delta
update adapts the winning prototype and its value. This learns prototypes in the
declared feature space, not an encoder or a policy for discovering relevant cues.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stream import FastSynapses


@dataclass
class ContentMemory:
    """A shared, fixed-capacity content store with novelty-gated winner selection.

    Matching cues update one prototype by ``key_rate`` and one value by
    ``value_rate``. Unmatched observations allocate a free slot or replace the
    least recently written slot. Reads below ``match`` abstain (return zero), as
    do zero cues. Reads never change state. Rows are observations of one shared
    memory, not independent streams; use one instance per isolated stream.

    Inputs need a stable feature coordinate system. No moving mean is applied:
    changing an encoder after storing keys can still invalidate retrieval. Two
    meanings with identical observable cues cannot be distinguished by this API.
    """

    inputs: int
    outputs: int
    capacity: int
    match: float = 0.75
    key_rate: float = 0.1
    value_rate: float = 1.0
    keys: np.ndarray = field(init=False, repr=False)
    values: np.ndarray = field(init=False, repr=False)
    last_write: np.ndarray = field(init=False, repr=False)
    size: int = field(init=False, default=0)
    writes: int = field(init=False, default=0)
    evictions: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        for name in ("inputs", "outputs", "capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.match) or not -1 <= self.match <= 1:
            raise ValueError("match must lie in [-1, 1]")
        for name in ("key_rate", "value_rate"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        self.keys = np.zeros((self.capacity, self.inputs))
        self.values = np.zeros((self.capacity, self.outputs))
        self.last_write = np.zeros(self.capacity, dtype=np.int64)

    def select(self, cue: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return winning slots and cosine scores; ``-1`` means no accepted match."""
        cue = FastSynapses._port(cue, self.inputs, "cue")
        slots = np.full(len(cue), -1, dtype=np.int64)
        scores = np.zeros(len(cue))
        if not self.size or not len(cue):
            return slots, scores
        unit = FastSynapses._delta_unit(cue)
        similarities = unit @ self.keys[: self.size].T
        winners = np.argmax(similarities, axis=1)
        scores = np.clip(similarities[np.arange(len(cue)), winners], -1.0, 1.0)
        accepted = (scores >= self.match) & np.any(cue != 0, axis=1)
        slots[accepted] = winners[accepted]
        return slots, scores

    def recall(self, cue: np.ndarray) -> np.ndarray:
        """Recall values from content only, with zero on a novel or absent cue."""
        slots, _ = self.select(cue)
        out = np.zeros((len(slots), self.outputs))
        accepted = slots >= 0
        out[accepted] = self.values[slots[accepted]]
        return out

    def observe(self, cue: np.ndarray, value: np.ndarray, write: np.ndarray | None = None) -> None:
        """Observe rows in order, after predicting; targets never select a slot.

        A zero cue is ignored. Validation precedes every mutation. A new slot
        stores the observed association exactly, irrespective of update rates;
        rates govern later adaptation of a matched record.
        """
        cue = FastSynapses._port(cue, self.inputs, "cue")
        value = FastSynapses._port(value, self.outputs, "value")
        if len(cue) != len(value):
            raise ValueError("cue and value batches must match")
        gate = np.ones(len(cue), dtype=bool) if write is None else np.asarray(write)
        if gate.shape != (len(cue),) or gate.dtype != np.bool_:
            raise ValueError("write must be a boolean vector matching the batch")
        unit = FastSynapses._delta_unit(cue)
        for row in np.flatnonzero(gate & np.any(cue != 0, axis=1)):
            selected, _ = self.select(unit[row : row + 1])
            slot = int(selected[0])
            if slot < 0:
                if self.size < self.capacity:
                    slot = self.size
                    self.size += 1
                else:
                    slot = int(np.argmin(self.last_write))
                    self.evictions += 1
                self.keys[slot] = unit[row]
                self.values[slot] = value[row]
            else:
                key = (1.0 - self.key_rate) * self.keys[slot] + self.key_rate * unit[row]
                # Opposite cues at match=-1 can cancel. Keep a meaningful direction.
                self.keys[slot] = (
                    FastSynapses._delta_unit(key[None])[0] if np.any(key) else unit[row]
                )
                self.values[slot] = (1.0 - self.value_rate) * self.values[
                    slot
                ] + self.value_rate * value[row]
            self.writes += 1
            self.last_write[slot] = self.writes

    def clear(self) -> None:
        """Erase all associations and counters without changing the declared capacity."""
        self.keys.fill(0)
        self.values.fill(0)
        self.last_write.fill(0)
        self.size = self.writes = self.evictions = 0

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "kind": "competitive-content",
            "inputs": self.inputs,
            "outputs": self.outputs,
            "capacity": self.capacity,
            "match": self.match,
            "key_rate": self.key_rate,
            "value_rate": self.value_rate,
            "size": self.size,
            "writes": self.writes,
            "evictions": self.evictions,
            "mutable_bytes": self.keys.nbytes + self.values.nbytes + self.last_write.nbytes,
        }
