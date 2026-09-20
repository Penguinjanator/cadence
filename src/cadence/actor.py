"""Fixed-model past repair and joint future-state/action repair.

This is the verified linear Gaussian/quadratic capacity component. Coordinates,
noise assumptions and preferences are supplied; no general memory or optimality
claim for arbitrary bodies follows. Planning never admits imagined observations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["BodyModel", "EquilibriumActor", "ActorPlan", "ActorReadback", "ObservationRecord"]


def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real array with shape {shape}")
    result = raw.astype(float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _spd(value: np.ndarray, name: str) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if not np.allclose(value, value.T, atol=1e-12, rtol=0):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(value)[0] <= 0:
        raise ValueError(f"{name} must be positive definite")


@dataclass(frozen=True, init=False)
class BodyModel:
    """Supplied position/displacement model; arrays are copied and read-only.

    Default Q=.04 GG.T + 1e-6 I is the audited stress-model assumption, not
    estimated sensor reliability. F/G may come from prior external learning.
    """

    F: np.ndarray
    G: np.ndarray
    Q: np.ndarray
    R: float

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        sensor_variance: float = 0.0,
        *,
        process_covariance: np.ndarray | None = None,
    ) -> None:
        f, g = _array(F, (2, 2), "F"), _array(G, (2,), "G")
        q = (
            0.04 * np.outer(g, g) + 1e-6 * np.eye(2)
            if process_covariance is None
            else _array(process_covariance, (2, 2), "Q")
        )
        if not np.isfinite(sensor_variance) or sensor_variance < 0:
            raise ValueError("sensor variance must be finite and nonnegative")
        _spd(q, "Q")
        for name, value in (("F", f), ("G", g), ("Q", q)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "R", float(sensor_variance))

    def snapshot(self) -> dict[str, Any]:
        return {"F": self.F.tolist(), "G": self.G.tolist(), "Q": self.Q.tolist(), "R": self.R}

    def fingerprint(self) -> str:
        arrays = {
            k: {
                "shape": list(getattr(self, k).shape),
                "dtype": getattr(self, k).dtype.str,
                "values": getattr(self, k).tolist(),
            }
            for k in ("F", "G", "Q")
        }
        model = {
            **arrays,
            "R": self.R,
            "R_dtype": np.asarray(self.R).dtype.str,
            "initial_mean": [0.0, 0.0],
            "initial_covariance": [[100.0, 0.0], [0.0, 100.0]],
        }
        return hashlib.sha256(
            json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()


@dataclass(frozen=True)
class ObservationRecord:
    """Caller-admitted actual reading; IDs enforce ordering, not authenticity."""

    identifier: int
    position: float


@dataclass(frozen=True)
class ActorReadback:
    record: ObservationRecord
    mean: np.ndarray
    covariance: np.ndarray
    residual: float
    minimum_pivot: float
    model_binding: str
    admitted: int
    marginalizations: int
    numeric_persistent_bytes: int


@dataclass(frozen=True)
class ActorPlan:
    boundary: np.ndarray
    covariance: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    readings: np.ndarray
    seams: np.ndarray
    cost_terms: dict[str, float]
    residual: float
    minimum_pivot: float
    message_bytes: int
    coefficient_bytes: int
    block_factorizations: int
    goal: float
    model_binding: str
    observation_id: int

    @property
    def action(self) -> float:
        """First unconstrained proposal; execute it externally, then admit reality."""
        return float(self.actions[0])

    @property
    def energy(self) -> float:
        return sum(self.cost_terms.values())


def _chain(
    diagonal: np.ndarray, lower: np.ndarray, information: np.ndarray
) -> tuple[np.ndarray, float, float, int]:
    """Exact SPD nearest-neighbor Schur messages, identical quadratic equations."""
    length, width = information.shape
    inverses: list[np.ndarray] = []
    messages: list[np.ndarray] = []
    minimum = np.inf
    if not all(np.isfinite(x).all() for x in (diagonal, lower, information)):
        raise ArithmeticError("nonfinite planning factors")
    for t in range(length):
        pivot, rhs = diagonal[t].copy(), information[t].copy()
        if t:
            cross = lower[t - 1] @ inverses[-1]
            pivot -= cross @ lower[t - 1].T
            rhs -= cross @ messages[-1]
        pivot = 0.5 * (pivot + pivot.T)
        eigenvalue = float(np.linalg.eigvalsh(pivot)[0])
        if not np.isfinite(eigenvalue) or eigenvalue <= 0:
            raise ArithmeticError("nonpositive local precision pivot")
        factor = np.linalg.cholesky(pivot)
        inverses.append(np.linalg.solve(factor.T, np.linalg.solve(factor, np.eye(width))))
        messages.append(rhs)
        minimum = min(minimum, eigenvalue)
    state = np.empty_like(information)
    for t in range(length - 1, -1, -1):
        rhs = messages[t].copy()
        if t + 1 < length:
            rhs -= lower[t].T @ state[t + 1]
        state[t] = inverses[t] @ rhs
    gradient = np.einsum("tij,tj->ti", diagonal, state) - information
    gradient[1:] += np.einsum("tij,tj->ti", lower, state[:-1])
    gradient[:-1] += np.einsum("tji,tj->ti", lower, state[1:])
    residual = float(np.max(np.abs(gradient)))
    if not np.isfinite(state).all() or not np.isfinite(residual) or residual > 1e-8:
        raise ArithmeticError("planning repair did not converge")
    return state, residual, minimum, sum(a.nbytes for a in inverses + messages)


class EquilibriumActor:
    """One-record Gaussian past boundary plus private quadratic action plans.

    Initial prior is mean zero, covariance100I. Before admitting a new reading,
    the old reading is conditioned exactly once and propagated through the
    executed action. The resulting incoming prior excludes the new likelihood.
    Goal changes and future repair never optimize the factual estimate.
    """

    def __init__(
        self,
        model: BodyModel,
        goal: float,
        *,
        horizon: int = 3,
        dynamics_precision: float = 1000.0,
        effort: float = 0.02,
    ) -> None:
        self.model = BodyModel(model.F, model.G, model.R, process_covariance=model.Q)
        self._binding = self.model.fingerprint()
        self.horizon = _positive_integer(horizon, "horizon")
        self.goal, self.q, self.effort = float(goal), float(dynamics_precision), float(effort)
        if (
            not np.isfinite([self.goal, self.q, self.effort]).all()
            or self.q <= 0
            or self.effort <= 0
        ):
            raise ValueError("finite goal and positive finite planning precisions required")
        self._precision = 0.01 * np.eye(2)
        self._information = np.zeros(2)
        self._record: ObservationRecord | None = None
        self._next = 0
        self._marginalizations = 0

    def _check(self) -> None:
        if self.model.fingerprint() != self._binding:
            raise ValueError("compressed past belongs to different model/noise parameters")
        if (
            not np.isfinite([self.goal, self.q, self.effort]).all()
            or self.q <= 0
            or self.effort <= 0
        ):
            raise ValueError("invalid goal/planning precisions")

    def _posterior(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        if self._record is None:
            raise ValueError("admit at least one actual reading before inference")
        position = self._record.position
        if self.model.R:
            precision, information = self._precision.copy(), self._information.copy()
            precision[0, 0] += 1 / self.model.R
            information[0] += position / self.model.R
            covariance = np.linalg.solve(precision, np.eye(2))
            mean = covariance @ information
            residual = float(np.max(np.abs(precision @ mean - information)))
            pivot = float(np.linalg.eigvalsh(precision)[0])
        else:
            pivot = float(self._precision[1, 1])
            mean = np.array(
                [position, (self._information[1] - self._precision[1, 0] * position) / pivot]
            )
            covariance = np.diag([0.0, 1 / pivot])
            residual = float(abs(self._precision[1] @ mean - self._information[1]))
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(covariance).all()
            or not np.isfinite(residual)
            or residual > 1e-7
            or pivot <= 0
        ):
            raise ArithmeticError("past repair did not converge")
        return mean, 0.5 * (covariance + covariance.T), residual, pivot

    def admit(
        self, position: float, *, identifier: int, executed_action: float | None = None
    ) -> ObservationRecord:
        """Admit ordered actual evidence; action is what was executed, not planned.

        No physical device is controlled here. The caller is responsible for
        measurement provenance and for reporting any externally clipped action.
        """
        self._check()
        identifier = _positive_integer(identifier, "identifier", 0)
        if identifier != self._next or not np.isfinite(position):
            raise ValueError("finite readings must be admitted exactly once in order")
        if self._record is None:
            if executed_action is not None:
                raise ValueError("first reading has no preceding action")
        elif executed_action is None or not np.isfinite(executed_action):
            raise ValueError("subsequent reading requires the actual executed action")
        precision, information = self._precision, self._information
        if self._record is not None:
            assert executed_action is not None
            # Frozen covariance-form prefix marginalization; no goal enters.
            covariance = np.linalg.solve(self._precision, np.eye(2))
            mean = covariance @ self._information
            gain = covariance[:, 0] / (covariance[0, 0] + self.model.R)
            mean += gain * (self._record.position - mean[0])
            transform = np.eye(2) - np.outer(gain, [1.0, 0.0])
            covariance = transform @ covariance @ transform.T + self.model.R * np.outer(gain, gain)
            mean = self.model.F @ mean + self.model.G * float(executed_action)
            covariance = self.model.F @ covariance @ self.model.F.T + self.model.Q
            covariance = 0.5 * (covariance + covariance.T)
            if not np.isfinite(covariance).all() or not np.isfinite(mean).all():
                raise ArithmeticError("nonfinite propagated past")
            _spd(covariance, "propagated covariance")
            precision = np.linalg.solve(covariance, np.eye(2))
            information = precision @ mean
            if not np.isfinite(information).all() or not np.isfinite(precision).all():
                raise ArithmeticError("nonfinite compressed past")
        # Commit only after all proposed message coefficients validate.
        self._precision, self._information = precision, information
        self._record = ObservationRecord(identifier, float(position))
        self._next += 1
        self._marginalizations += int(identifier > 0)
        return self._record

    def readback(self) -> ActorReadback:
        """Detached factual estimate; supplied future preferences cannot enter it."""
        self._check()
        mean, covariance, residual, pivot = self._posterior()
        assert self._record is not None
        return ActorReadback(
            self._record,
            mean,
            covariance,
            residual,
            pivot,
            self._binding,
            self._next,
            self._marginalizations,
            self.numeric_persistent_bytes(),
        )

    def plan(self, *, horizon: int | None = None, goal: float | None = None) -> ActorPlan:
        """Private joint state/action repair from a copied factual boundary.

        Override goal only for this branch. Returned actions are unconstrained;
        no hidden saturation or action teacher is applied. Covariance is exposed
        for diagnosis, not used as a chance constraint in this planner.
        """
        past = self.readback()
        length = _positive_integer(self.horizon if horizon is None else horizon, "horizon")
        preference = self.goal if goal is None else float(goal)
        if not np.isfinite(preference):
            raise ValueError("goal must be finite")
        f, g = self.model.F, self.model.G
        current = np.column_stack((np.eye(2), -g))
        previous = np.column_stack((-f, np.zeros(2)))
        diagonal = np.repeat((self.q * current.T @ current)[None], length, axis=0)
        diagonal[:, 2, 2] += self.effort
        diagonal[:-1] += self.q * previous.T @ previous
        diagonal[-1, :2, :2] += np.eye(2)
        lower = np.repeat((self.q * current.T @ previous)[None], length - 1, axis=0)
        information = np.zeros((length, 3))
        information[0] += self.q * current.T @ (f @ past.mean)
        information[-1, :2] += [preference, 0.0]
        state, residual, pivot, messages = _chain(diagonal, lower, information)
        states, actions = state[:, :2].copy(), state[:, 2].copy()
        seams = states - np.vstack((past.mean, states[:-1])) @ f.T - actions[:, None] * g
        terms = {
            "dynamics": 0.5 * self.q * float(np.square(seams).sum()),
            "terminal": 0.5 * float(np.square(states[-1] - [preference, 0.0]).sum()),
            "effort": 0.5 * self.effort * float(np.square(actions).sum()),
        }
        return ActorPlan(
            past.mean,
            past.covariance,
            states,
            actions,
            states[:, 0].copy(),
            seams,
            terms,
            residual,
            pivot,
            messages,
            diagonal.nbytes + lower.nbytes + information.nbytes,
            length,
            preference,
            self._binding,
            past.record.identifier,
        )

    def numeric_persistent_bytes(self) -> int:
        """Array/scalar/ID/hash payload; excludes Python and serialization overhead."""
        return (
            sum(
                x.nbytes
                for x in (
                    self.model.F,
                    self.model.G,
                    self.model.Q,
                    self._precision,
                    self._information,
                )
            )
            + 32
            + 7 * 8
            + (16 if self._record else 0)
        )

    def snapshot(self) -> dict[str, Any]:
        self._check()
        return {
            "format": "cadence-actor/1",
            "model": self.model.snapshot(),
            "binding": self._binding,
            "goal": self.goal,
            "horizon": self.horizon,
            "dynamics_precision": self.q,
            "effort": self.effort,
            "precision": self._precision.tolist(),
            "information": self._information.tolist(),
            "next_identifier": self._next,
            "marginalizations": self._marginalizations,
            "record": None
            if self._record is None
            else {"id": self._record.identifier, "position": self._record.position},
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, Any]) -> EquilibriumActor:
        """Validate and restore fixed-model history; never reinterpret a stale prefix."""
        try:
            if snapshot["format"] != "cadence-actor/1":
                raise ValueError("unsupported actor checkpoint")
            model = snapshot["model"]
            result = cls(
                BodyModel(model["F"], model["G"], model["R"], process_covariance=model["Q"]),
                snapshot["goal"],
                horizon=snapshot["horizon"],
                dynamics_precision=snapshot["dynamics_precision"],
                effort=snapshot["effort"],
            )
            if result._binding != snapshot["binding"]:
                raise ValueError("checkpoint model binding mismatch")
            result._precision = _array(snapshot["precision"], (2, 2), "precision")
            _spd(result._precision, "precision")
            result._information = _array(snapshot["information"], (2,), "information")
            result._next = _positive_integer(snapshot["next_identifier"], "next_identifier", 0)
            result._marginalizations = _positive_integer(
                snapshot["marginalizations"], "marginalizations", 0
            )
            record = snapshot["record"]
            if record is None:
                if (
                    result._next
                    or result._marginalizations
                    or not np.array_equal(result._precision, 0.01 * np.eye(2))
                    or np.any(result._information)
                ):
                    raise ValueError("invalid empty-history checkpoint")
            else:
                identifier = _positive_integer(record["id"], "record id", 0)
                if (
                    identifier != result._next - 1
                    or result._marginalizations != identifier
                    or not np.isfinite(record["position"])
                ):
                    raise ValueError("inconsistent evidence counters")
                result._record = ObservationRecord(identifier, float(record["position"]))
                result.readback()
            return result
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("invalid actor checkpoint") from error

    def save(self, path: str | Path) -> Path:
        from .checkpoint import _write

        return _write(
            {"meta": np.array(json.dumps(self.snapshot(), sort_keys=True, allow_nan=False))}, path
        )

    @classmethod
    def load(cls, path: str | Path) -> EquilibriumActor:
        with np.load(path, allow_pickle=False) as data:
            if data.files != ["meta"]:
                raise ValueError("invalid actor archive")
            return cls.restore(json.loads(str(data["meta"])))
