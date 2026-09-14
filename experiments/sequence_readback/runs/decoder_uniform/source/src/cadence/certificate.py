"""The settling certificate: a contraction rate from the row mass and the neuron's slope.

One settling step of a brain without adaptation is ``v <- (1 - dt) v + dt (W act(v) + c)``.
With ``L`` a Lipschitz constant of the activation and ``rho`` the largest absolute incoming
effective weight sum of any neuron, the step contracts the sup norm by

    q = 1 - dt (1 - L rho)

whenever ``L rho < 1``. Then the equilibrium is unique, ``k`` steps from any start leave at
most ``q^k / (1 - q)`` times the first movement, the last observed movement bounds the
remaining error by ``movement / (1 - q)``, and a warm start after a stimulus change of size
``delta`` needs about ``log(dt delta / ((1 - q) tol)) / log(1 / q)`` steps. These statements
are proved for the library's activation in the flagship paper's Lean library.

The certificate applies to the free phase without adaptation. A nudge adds a term, and
adaptation adds a slow variable; neither is covered, and the residual (``Brain.residual``)
stays the only check for them. A brain that is not certified may settle in practice; the
certificate then says nothing, one way or the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .brain import Brain
    from .neuron import NeuronModel


def lipschitz_constant(model: NeuronModel) -> float:
    """Supremum of the activation's slope: ``(lambda / 4) max(1 / (1 - rho), leak / rho)``."""
    rest = model.rest_emission
    return float(model.slope / 4.0 * max(1.0 / (1.0 - rest), model.leak / rest))


def row_mass(brain: Brain) -> float:
    """The largest absolute incoming effective weight sum over all neurons."""
    incoming = np.bincount(
        brain.connectome.post, weights=np.abs(brain.weights), minlength=brain.connectome.n
    )
    return float(incoming.max()) if len(incoming) else 0.0


@dataclass(frozen=True)
class Certificate:
    """What the contraction argument certifies for a brain's free settling."""

    row_mass: float
    lipschitz: float
    dt: float
    adaptation: bool

    @property
    def rate(self) -> float:
        """The contraction constant ``1 - dt (1 - L rho)``; below one when certified."""
        return 1.0 - self.dt * (1.0 - self.lipschitz * self.row_mass)

    @property
    def certified(self) -> bool:
        return self.rate < 1.0 and not self.adaptation

    @property
    def mass_limit(self) -> float:
        """The row mass below which this neuron model is certified: ``1 / L``."""
        return 1.0 / self.lipschitz

    def error_bound(self, movement: np.ndarray | float) -> np.ndarray:
        """Remaining distance to the equilibrium from the last step's movement (sup norm).

        ``movement`` is the largest change of a potential in the last step, per row;
        the bound is ``movement / (1 - rate)``. Uncertified brains get ``inf``.
        """
        out = np.asarray(movement, dtype=float)
        if not self.certified:
            return np.full_like(out, np.inf)
        return np.asarray(out / (1.0 - self.rate))

    def apriori_bound(self, first_movement: np.ndarray | float, steps: int) -> np.ndarray:
        """Distance after ``steps`` settling steps, from the first step's movement."""
        out = np.asarray(first_movement, dtype=float)
        if not self.certified:
            return np.full_like(out, np.inf)
        return np.asarray(out * self.rate**steps / (1.0 - self.rate))

    def steps_for(self, change: float, tolerance: float) -> int:
        """Warm-start steps after a stimulus change of size ``change`` (sup norm) to be
        within ``tolerance`` of the new equilibrium. Zero for no change."""
        if not self.certified:
            raise ValueError("an uncertified brain has no step bound")
        if change <= 0.0:
            return 0
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        start = self.dt * change / (1.0 - self.rate)
        if start <= tolerance:
            return 0
        return int(math.ceil(math.log(start / tolerance) / math.log(1.0 / self.rate)))

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "row_mass": self.row_mass,
            "lipschitz": self.lipschitz,
            "dt": self.dt,
            "rate": self.rate,
            "certified": self.certified,
            "mass_limit": self.mass_limit,
            "adaptation": self.adaptation,
        }


def certificate(brain: Brain) -> Certificate:
    """The settling certificate of ``brain``'s free phase.

    Reads the effective weights, the neuron model's slope and leak, and whether adaptation
    is present. Ablation masks only remove synapses, so a certified brain stays certified
    under any mask.
    """
    model = brain.neuron_model
    return Certificate(
        row_mass=row_mass(brain),
        lipschitz=lipschitz_constant(model),
        dt=float(model.dt),
        adaptation=model.adaptation is not None,
    )
