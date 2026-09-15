"""Records: what followed each reading, kept where the reading lands.

A reading is the drive of the neurons a patch reads: sensory fields with their missing
flags, the goal, the action, the reads of declared stores. Each unit's running mean is
subtracted, so the code follows what deviates from the usual reading and a small cue can
move it. A fixed random projection maps the reading onto many cells; the most active cells
stay and the rest are inhibited. Each predicted field keeps a table with one record per
cell. The read of a reading is the sum of its active cells' records, weighted by their
activity; the witnessed outcome is written into exactly those records by the delta rule.
A reading touches few records, so one outcome is written almost exactly and a reading
elsewhere leaves it intact: records learn from one stream without replay.

Two codes come from one reading. The plain code serves the consequence fields. The valued
code first divides every declared input pathway by its running norm, so the goal and a
remembered cue have the same say as the senses; the valued fields (reward, terminal value)
read and write through it at their own rate.

The running mean settles: it is the plain average of the readings seen until one over their
count falls to the habituation rate, and follows at that rate after. A settled mean keeps the
code of a reading where its records were written; a mean that keeps moving would carry old
readings onto cells that were never written.

The projection is drawn from ``Mulberry32``, the generator of ``brain_scan.js``, so a page
rebuilds the same cells from the seed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["Mulberry32", "Records"]

M32 = 0xFFFFFFFF


class Mulberry32:
    """The 32-bit generator of ``brain_scan.js`` (``mulberry``): uniform draws in [0, 1)."""

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & M32

    def random(self) -> float:
        a = (self.state + 0x6D2B79F5) & M32
        self.state = a
        t = ((a ^ (a >> 15)) * (a | 1)) & M32
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296.0

    def batch(self, n: int) -> np.ndarray:
        """``n`` draws at once, equal to ``n`` calls of ``random``."""
        if n < 0:
            raise ValueError("n must be nonnegative")
        m32 = np.uint64(M32)
        step = np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x6D2B79F5)
        a = (np.uint64(self.state) + step) & m32
        t = ((a ^ (a >> np.uint64(15))) * (a | np.uint64(1))) & m32
        t = (t ^ ((t + (((t ^ (t >> np.uint64(7))) * (t | np.uint64(61))) & m32)) & m32)) & m32
        out: np.ndarray = ((t ^ (t >> np.uint64(14))) & m32).astype(np.float64) / 4294967296.0
        self.state = int((self.state + n * 0x6D2B79F5) & M32)
        return out

    def normals(self, n: int) -> np.ndarray:
        """``n`` standard normal draws by the Box-Muller transform of ``batch`` pairs."""
        pairs = (n + 1) // 2
        u = self.batch(2 * pairs)
        radius = np.sqrt(-2.0 * np.log(np.maximum(u[:pairs], 1e-12)))
        angle = 2.0 * np.pi * u[pairs:]
        out: np.ndarray = np.concatenate([radius * np.cos(angle), radius * np.sin(angle)])
        return out[:n]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _rate(name: str, value: float, *, upper: float) -> float:
    rate = float(value)
    if not np.isfinite(rate) or not 0.0 <= rate <= upper:
        raise ValueError(f"{name} lies in [0, {upper:g}]")
    return rate


class Records:
    """A records cortex over readings of ``inputs`` units.

    ``fields`` maps each predicted field to its width. Of ``cells`` code cells, ``active``
    stay per reading. ``rate`` is the write rate of the consequence fields; ``valued`` names
    the fields that read and write through the valued code, at ``valued_rate``.
    ``habituation`` is the slowest rate of each unit's running mean (0 subtracts nothing).
    ``pathways`` are index arrays into the reading whose running norms, at ``pathway_rate``,
    equalise their say in the valued code. ``bias`` scales the cells' fixed offsets. ``seed``
    starts the generator of the projection and the offsets.
    """

    def __init__(
        self,
        inputs: int,
        fields: Mapping[str, int],
        *,
        cells: int = 8000,
        active: int = 40,
        rate: float = 0.2,
        valued: Iterable[str] = (),
        valued_rate: float = 1.0,
        habituation: float = 0.002,
        bias: float = 0.3,
        pathways: Sequence[Sequence[int] | np.ndarray] = (),
        pathway_rate: float = 0.002,
        seed: int = 0,
    ) -> None:
        self.inputs = _positive_int("inputs", inputs)
        self.cells = _positive_int("cells", cells)
        self.active = _positive_int("active", active)
        if self.active > self.cells:
            raise ValueError("active must not exceed cells")
        if not fields:
            raise ValueError("records need at least one field")
        self.fields = {str(k): _positive_int(f"field {k!r}", w) for k, w in fields.items()}
        self.valued = frozenset(str(name) for name in valued)
        if not self.valued <= set(self.fields):
            missing = sorted(self.valued - set(self.fields))
            raise ValueError(f"valued fields {missing} are not fields")
        self.rate = _rate("rate", rate, upper=2.0)
        self.valued_rate = _rate("valued_rate", valued_rate, upper=2.0)
        self.habituation = _rate("habituation", habituation, upper=1.0)
        self.pathway_rate = _rate("pathway_rate", pathway_rate, upper=1.0)
        self.bias = float(bias)
        if not np.isfinite(self.bias) or self.bias < 0:
            raise ValueError("bias must be finite and nonnegative")
        self.seed = int(seed) & M32
        generator = Mulberry32(self.seed)
        draws = generator.normals(self.inputs * self.cells)
        self.projection = draws.reshape(self.inputs, self.cells) / np.sqrt(self.inputs)
        self.offset = generator.normals(self.cells) * self.bias
        self.pathways = [np.asarray(p, dtype=np.int64) for p in pathways if len(p)]
        for p in self.pathways:
            if p.ndim != 1 or p.min() < 0 or p.max() >= self.inputs:
                raise ValueError("a pathway is a one-dimensional index array into the reading")
        self.mean = np.zeros(self.inputs)
        self.seen = 0
        self.pathway_norm = np.ones(len(self.pathways))
        self.tables = {name: np.zeros((self.cells, width)) for name, width in self.fields.items()}
        self.writes = 0

    def _winners(self, x: np.ndarray) -> np.ndarray:
        drive = x @ self.projection + self.offset
        k = self.active
        out = np.zeros_like(drive)
        index = np.argpartition(-drive, k - 1, axis=1)[:, :k]
        rows = np.arange(drive.shape[0])[:, None]
        out[rows, index] = np.maximum(drive[rows, index], 0.0)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        code: np.ndarray = np.where(norm > 0, out / np.maximum(norm, 1e-12), out)
        return code

    def code(self, readings: np.ndarray, *, adapt: bool = False) -> np.ndarray:
        """The codes of ``(batch, inputs)`` readings: ``(2, batch, cells)``, plain then valued.

        ``adapt`` first moves the running mean and the pathway norms by these readings:
        witnessed readings adapt, imagined readings do not."""
        x = np.atleast_2d(np.asarray(readings, dtype=float))
        if x.ndim != 2 or x.shape[1] != self.inputs or not np.isfinite(x).all():
            raise ValueError(f"readings must be a finite (batch, {self.inputs}) array")
        if self.habituation > 0:
            if adapt:
                for row in x:
                    self.seen += 1
                    self.mean += max(self.habituation, 1.0 / self.seen) * (row - self.mean)
            x = x - self.mean
        plain = self._winners(x)
        if self.pathways:
            v = x.copy()
            for k, pathway in enumerate(self.pathways):
                norms = np.linalg.norm(v[:, pathway], axis=1)
                if adapt:
                    for value in norms:
                        self.pathway_norm[k] += self.pathway_rate * (value - self.pathway_norm[k])
                v[:, pathway] /= self.pathway_norm[k] + 1e-3
            valued = self._winners(v)
        else:
            valued = plain
        return np.stack([plain, valued])

    def _code_for(self, code: np.ndarray, name: str) -> np.ndarray:
        if code.shape[0] != 2 or code.shape[-1] != self.cells:
            raise ValueError(f"a code has shape (2, {self.cells}) or (2, batch, {self.cells})")
        selected: np.ndarray = code[1] if name in self.valued else code[0]
        return selected

    def read(self, code: np.ndarray) -> dict[str, np.ndarray]:
        """Each field's read: ``(width,)`` for one code, ``(batch, width)`` for a batch."""
        code = np.asarray(code, dtype=float)
        return {name: self._code_for(code, name) @ table for name, table in self.tables.items()}

    def write(
        self,
        code: np.ndarray,
        targets: Mapping[str, np.ndarray],
        known: Mapping[str, np.ndarray] | None = None,
    ) -> int:
        """Write the witnessed outcome of one reading.

        Each named field's records move toward its target by the delta rule, through the
        active cells only. ``known`` masks the entries of a target that were observed.
        Returns the number of fields written."""
        code = np.asarray(code, dtype=float)
        if code.shape != (2, self.cells):
            raise ValueError(f"write takes one code of shape (2, {self.cells})")
        written = 0
        for name, width in self.fields.items():
            if name not in targets:
                continue
            target = np.asarray(targets[name], dtype=float)
            if target.shape != (width,) or not np.isfinite(target).all():
                raise ValueError(f"the target of {name!r} must be a finite vector of width {width}")
            c = self._code_for(code, name)
            error = target - c @ self.tables[name]
            if known is not None and name in known:
                error = error * np.asarray(known[name], dtype=bool)
            rate = self.valued_rate if name in self.valued else self.rate
            self.tables[name] += rate * np.outer(c, error)
            written += 1
        self.writes += written
        return written

    def parameters(self) -> int:
        """Record entries, the learned state; the projection and the offsets are fixed."""
        return int(sum(table.size for table in self.tables.values()))

    def to_dict(self) -> dict[str, Any]:
        """The configuration that rebuilds the fixed cells (the tables are the learned state)."""
        return {
            "inputs": self.inputs,
            "fields": dict(self.fields),
            "cells": self.cells,
            "active": self.active,
            "rate": self.rate,
            "valued": sorted(self.valued),
            "valued_rate": self.valued_rate,
            "habituation": self.habituation,
            "bias": self.bias,
            "pathways": [p.tolist() for p in self.pathways],
            "pathway_rate": self.pathway_rate,
            "seed": self.seed,
        }
