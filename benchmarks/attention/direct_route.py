"""Route-only learned-code attention used by the retained component assay.

This is a benchmark mechanism, not a ``Brain.compose`` option.  A live route
``R`` first agrees with the memory selected by the already learned sensory
meaning ``h``.  A born-zero local map ``W`` turns that route into the answer
query during the same equilibrium::

    q = (1-g) h + g gamma R W
    E = g sigma E(R, h) + (1-g) eta ||R||^2 / 2 + E(A, q)

``A`` is the answer read, and each ``E(read, address)`` is the null-inclusive
softmax agreement energy used by the coupled-memory instrument.  There is no
separate query state or anchor: one head adds 64 live route coordinates.  All
forces below are negative derivatives of that single scalar energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _array(value: Any, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (ndim is not None and result.ndim != ndim) or not np.isfinite(result).all():
        raise ValueError("a complete finite float64 array is required")
    return result


@dataclass(frozen=True)
class DirectRoute:
    """Immutable one-head boundary for pure equation and credit checks."""

    keys: np.ndarray
    values: np.ndarray
    held: np.ndarray
    ages: np.ndarray
    meaning: np.ndarray
    weight: np.ndarray
    gate: np.ndarray
    amplitude: float = 3.0
    temperature: float = 0.25
    recency: float = 0.25
    null: float = 0.0
    gamma: float = 6.75
    route_stiffness: float = 36.0
    route_damping: float = 3.0

    def __post_init__(self) -> None:
        keys, values = _array(self.keys, ndim=3), _array(self.values, ndim=3)
        meaning, weight = _array(self.meaning, ndim=2), _array(self.weight, ndim=2)
        held, ages, gate = np.asarray(self.held), _array(self.ages), _array(self.gate)
        batch, slots, width = keys.shape
        if (values.shape != keys.shape or held.shape != (batch, slots)
                or held.dtype != np.bool_ or ages.shape not in ((slots,), (batch, slots))
                or meaning.shape != (batch, width) or weight.shape != (width, width)
                or gate.shape != (batch,) or ((gate < 0) | (gate > 1)).any()):
            raise ValueError("matching direct-route memory geometry is required")
        scalars = (self.amplitude, self.temperature, self.recency, self.null,
                   self.gamma, self.route_stiffness, self.route_damping)
        if (not all(type(value) is float and np.isfinite(value) for value in scalars)
                or self.amplitude < 0 or self.temperature <= 0 or self.recency < 0
                or self.gamma <= 0 or self.route_stiffness <= 0
                or self.route_damping <= 0):
            raise ValueError("finite positive direct-route settings are required")
        for name, value in (("keys", keys), ("values", values), ("held", held),
                            ("ages", ages), ("meaning", meaning), ("weight", weight),
                            ("gate", gate)):
            frozen = np.array(value, copy=True)
            frozen.flags.writeable = False
            object.__setattr__(self, name, frozen)

    @property
    def streams(self) -> int:
        return int(len(self.keys))

    @property
    def width(self) -> int:
        return int(self.keys.shape[2])

    def _agreement(self, read: np.ndarray, address: np.ndarray) -> dict[str, np.ndarray]:
        read, address = _array(read, ndim=2), _array(address, ndim=2)
        if read.shape != address.shape or read.shape != (self.streams, self.width):
            raise ValueError("read and address must match the route geometry")
        transformed = self.keys
        logits = (address[:, None, :] * transformed).sum(axis=2) / self.temperature
        logits -= self.recency * self.ages
        occupied = np.where(self.held, logits, -np.inf)
        present = self.held.any(axis=1)
        top = np.max(occupied, axis=1, initial=-np.inf)
        safe = np.where(present, top, 0.0)
        numerator = np.exp(occupied - safe[:, None])
        denominator = np.maximum(numerator.sum(axis=1), 1.0)
        conditional = numerator / denominator[:, None]
        partition = np.where(present, safe + np.log(denominator), -np.inf)
        delta = partition - self.null / self.temperature
        mass = np.exp(-np.logaddexp(0.0, -delta))
        null = np.exp(-np.logaddexp(0.0, delta))
        mean = (conditional[:, :, None] * self.values).sum(axis=1)
        error = read - mean
        squared = (error**2).sum(axis=1)
        alignment = (error[:, None, :] * self.values).sum(axis=2)
        centered = alignment - (conditional * alignment).sum(axis=1, keepdims=True)
        coefficient = self.amplitude * mass[:, None] * conditional * (
            centered - 0.5 * squared[:, None] * null[:, None])
        return {
            "energy": 0.5 * self.amplitude * mass * squared,
            "read_force": -self.amplitude * mass[:, None] * error,
            "address_force": (coefficient[:, :, None] * transformed).sum(axis=1)
                             / self.temperature,
            "weights": mass[:, None] * conditional,
        }

    def parts(self, answer: np.ndarray, route: np.ndarray) -> dict[str, Any]:
        answer, route = _array(answer, ndim=2), _array(route, ndim=2)
        expected = (self.streams, self.width)
        if answer.shape != expected or route.shape != expected:
            raise ValueError("answer and route must match the memory width")
        gate = self.gate[:, None]
        query = (1.0 - gate) * self.meaning + gate * self.gamma * (route @ self.weight)
        return {"answer": self._agreement(answer, query),
                "route": self._agreement(route, self.meaning),
                "query": query, "answer_state": answer, "route_state": route}

    def energy(self, answer: np.ndarray, route: np.ndarray) -> np.ndarray:
        parts = self.parts(answer, route)
        return (self.gate * self.route_stiffness * parts["route"]["energy"]
                + (1.0 - self.gate) * 0.5 * self.route_damping
                  * (parts["route_state"]**2).sum(axis=1)
                + parts["answer"]["energy"])

    def forces(self, answer: np.ndarray, route: np.ndarray) -> dict[str, np.ndarray]:
        parts = self.parts(answer, route)
        gate = self.gate[:, None]
        answer_address = parts["answer"]["address_force"]
        route_state = parts["route_state"]
        return {
            "answer": parts["answer"]["read_force"],
            "route": (gate * self.route_stiffness * parts["route"]["read_force"]
                      - (1.0 - gate) * self.route_damping * route_state
                      + gate * self.gamma * (answer_address @ self.weight.T)),
            "meaning": (gate * self.route_stiffness * parts["route"]["address_force"]
                        + (1.0 - gate) * answer_address),
            "weight": (self.gamma * self.gate[:, None, None]
                       * route_state[:, :, None] * answer_address[:, None, :]),
        }

    def weights(self, answer: np.ndarray, route: np.ndarray) -> np.ndarray:
        return self.parts(answer, route)["answer"]["weights"]


def centered_weight_gradient(boundary: DirectRoute, plus_answer: np.ndarray,
                             plus_route: np.ndarray, minus_answer: np.ndarray,
                             minus_route: np.ndarray, beta: float) -> np.ndarray:
    """Mean local equilibrium-propagation gradient for the born-zero ``W``."""
    if type(beta) is not float or not np.isfinite(beta) or beta <= 0:
        raise ValueError("a positive finite beta is required")
    plus = boundary.forces(plus_answer, plus_route)["weight"]
    minus = boundary.forces(minus_answer, minus_route)["weight"]
    result = ((plus - minus) / (2.0 * beta)).mean(axis=0)
    if not np.isfinite(result).all():
        raise ValueError("the complete local W contrast must be finite")
    return np.ascontiguousarray(result)
