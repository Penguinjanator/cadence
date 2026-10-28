"""The apical compartment: top-down input normalized to the bottom-up input it joins.

A cortical pyramidal neuron hears the area below it on its basal dendrites and the areas
above it on its apical tuft, and the tuft's effect on the cell is gain controlled: it
modulates and adds to what the basal input drives rather than replacing it (Larkum 2013;
divisive normalization, Carandini and Heeger 2012). In a brain whose regions are joined by
reciprocal synapse pairs, each region's top-down half is the other region's bottom-up half
run backwards, and a plain sum lets the loop through the pair take a region over as the
weights grow: the settle slows near the loop's critical gain, and the region carries an
echo of the answer instead of a representation of its input.

``Apical`` names the synapses that land on the apical compartment and the pools of neurons
that normalize them. In every sweep of a settle, for each row and each pool, it reads every
member's apical input ``u`` (the summed input of its apical synapses) and its basal input
``b`` (everything else the neuron receives: the other synapses, the stimulus, the bias, a
nudge), and the neuron receives ``b`` plus

    fraction * rms(b) * u / sqrt(rms(u)^2 + (floor * rms(b))^2),

the root mean squares taken over the pool. The top-down direction is kept; its magnitude
is at most ``fraction`` times the pool's basal drive whatever the weights grow to, and a
weak top-down input passes linearly with gain ``fraction / floor``. The fixed point is the
settle's: a nudge on the output moves the regions above and their apical input follows, so
every synapse's two-phase contrast is read as usual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Apical", "APICAL_FLOOR"]

# The founder semi-saturation of the normalization, in units of the pool's basal RMS: a
# top-down input as large as the basal input passes at 1/sqrt(2) of its saturated share.
APICAL_FLOOR = 1.0
_TINY = 1e-24  # under the square root: a pool with no input at all receives none


@dataclass(frozen=True, eq=False)
class Apical:
    """The synapses that land on the apical compartment, and how each pool normalizes them.

    ``synapses`` is a bool per synapse; ``groups`` an integer per neuron naming its pool
    (``-1``: no pool, so it may receive no apical synapse). ``fraction`` bounds the apical
    drive's RMS over a pool to that share of the pool's basal RMS; ``floor`` is the
    normalization's semi-saturation constant in units of the basal RMS.
    """

    synapses: np.ndarray
    groups: np.ndarray
    fraction: float
    floor: float = APICAL_FLOOR
    members: tuple[Any, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        synapses = np.asarray(self.synapses)
        groups = np.asarray(self.groups)
        if synapses.ndim != 1 or synapses.dtype != np.bool_:
            raise ValueError("apical synapses must be a boolean vector, one entry per synapse")
        if groups.ndim != 1 or not np.issubdtype(groups.dtype, np.integer) or (groups < -1).any():
            raise ValueError("apical groups must be an integer per neuron, -1 for none")
        for name in ("fraction", "floor"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"apical {name} must be a real number")
        if not np.isfinite(self.fraction) or self.fraction <= 0:
            raise ValueError("apical fraction must be finite and positive")
        if not np.isfinite(self.floor) or self.floor < 0:
            raise ValueError("apical floor must be finite and nonnegative")
        synapses = synapses.copy()
        groups = groups.astype(np.int64, copy=True)
        synapses.setflags(write=False)
        groups.setflags(write=False)
        object.__setattr__(self, "synapses", synapses)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "fraction", float(self.fraction))
        object.__setattr__(self, "floor", float(self.floor))
        members: list[Any] = []
        for pool in np.unique(groups[groups >= 0]):
            index = np.flatnonzero(groups == pool)
            if index[-1] - index[0] + 1 == len(index):  # one contiguous run: a view, no gather
                members.append(slice(int(index[0]), int(index[-1]) + 1))
            else:
                members.append(index)
        object.__setattr__(self, "members", tuple(members))

    def check(self, synapses: int, neurons: int, post: np.ndarray) -> None:
        """Refuse a spec that does not fit a connectome of these sizes and post neurons."""
        if self.synapses.shape != (synapses,) or self.groups.shape != (neurons,):
            raise ValueError("apical synapses and groups must match the connectome")
        if self.synapses.any() and (self.groups[np.asarray(post)[self.synapses]] < 0).any():
            raise ValueError("every apical synapse must land on a neuron of a pool")

    def drive(self, basal: np.ndarray, raw: np.ndarray) -> np.ndarray:
        """The normalized apical drive ``(batch, n)`` from the basal and raw apical inputs."""
        out = np.zeros_like(raw)
        fraction, floor2 = self.fraction, self.floor * self.floor
        for members in self.members:
            b = basal[:, members]
            u = raw[:, members]
            basal2 = np.mean(b * b, axis=1, keepdims=True)
            raw2 = np.mean(u * u, axis=1, keepdims=True)
            gain = fraction * np.sqrt(basal2) / np.sqrt(raw2 + floor2 * basal2 + _TINY)
            out[:, members] = gain * u
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "synapses": int(self.synapses.sum()),
            "pools": len(self.members),
            "fraction": self.fraction,
            "floor": self.floor,
        }
