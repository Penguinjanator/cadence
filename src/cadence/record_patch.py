"""A temporal patch with a slow linear context and a fast record inside it.

The state transition is linear in the context and gated by the input; the
nonlinearity sits at the ports. The energy is quadratic in the context path,
so its detuned equilibria are unique and the centered detuning contrast equals
the adjoint (reverse-mode) gradient of the same loss. ``observe`` computes that
gradient by one backward scan; ``detune`` solves the two detuned equilibria and
returns the contrast, as the acceptance check of the identity. Records are read
at every moment through ordinary ports and written one-shot by the delta rule
after the observed path. Neither route differentiates through the record code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .records import Records

__all__ = ["RecordPatchNet", "RecordPath", "RecordObservation", "RecordContrast", "RecordReadback"]

_PARAMETERS = ("G", "g", "B", "b", "C", "c")
FORMAT = "cadence-record-patch/1"


@dataclass(frozen=True)
class RecordPath:
    """A detached free path; ``hidden`` is the bounded context (batch, time, hidden)."""

    hidden: np.ndarray
    output: np.ndarray
    gate: np.ndarray
    read: np.ndarray
    loss: float | None

    @property
    def final_state(self) -> np.ndarray:
        return self.hidden[:, -1].copy()


@dataclass(frozen=True)
class RecordObservation:
    """One teaching attempt: the prediction made before any write, and what changed."""

    updated: bool
    reason: str
    prediction: RecordPath
    delta: dict[str, np.ndarray] | None = None
    initial_loss: float | None = None
    final_loss: float | None = None
    accepted_rate: float | None = None
    replay_losses: tuple[float | None, ...] = ()
    replay_calls: int = 0
    writes: int = 0


@dataclass(frozen=True)
class RecordContrast:
    """The two detuned equilibria of the quadratic energy and their centered contrast."""

    contrast: dict[str, np.ndarray]
    plus_hidden: np.ndarray
    minus_hidden: np.ndarray
    plus_energy: float
    minus_energy: float
    iterations: tuple[int, int]
    residuals: tuple[float, float]
    converged: bool


@dataclass(frozen=True)
class RecordReadback:
    """Read-only self-observation; never automatically admitted as teaching data."""

    state: np.ndarray | None
    updates: int
    writes: int
    parameter_revision: int
    state_parameter_revision: int | None
    record_entries: int


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * x)))


class RecordPatchNet:
    """Learn observed streams with a gated linear context and one-shot records.

    For inputs ``u[t]``, context ``h[t]`` and outputs ``y[t]``::

        s[t] = g + G u[t]                       gate preactivation
        l[t] = sigmoid(s[t])                    per-channel retention in (0, 1)
        z[t] = tanh(B u[t] + b)                 the input port
        h[t] = l[t] * h[t-1] + (1 - l[t]) * z[t]
        m[t] = read(code([u[t], r * h[t]]))     the record read, a port value
        y[t] = C h[t] + c + m[t]

    ``r`` is the fixed per-channel unit of a context deviation. The context
    is a convex combination, so ``|h| <= max(|h[-1]|, 1)``. The
    seam residual ``h[t] - l[t] h[t-1] - (1 - l[t]) z[t]`` is linear in the
    context path: the energy is quadratic and the free path is its unique
    zero-defect normal form.

    A record holds what the slow readout got wrong at its reading: the write
    target is ``y_obs[t] - C h[t] - c``. A reading the slow parameters have
    learned leaves a record near zero, so the store holds only what the slow
    model does not yet know. Records are read with the tables as they stood
    when the call began and written after the path, so a prediction is a
    function of the parameters, the records and the boundary alone. To let a
    record written at one moment inform the next, observe in shorter paths.
    """

    def __init__(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
        *,
        seed: int = 0,
        output_precision: np.ndarray | None = None,
        cells: int = 4096,
        active: int = 32,
        record_rate: float = 0.5,
        habituation: float = 1e-5,
        record_bias: float = 0.3,
        slowest: float = 128.0,
    ) -> None:
        self.inputs = _integer("inputs", inputs, 1)
        self.hidden = _integer("hidden", hidden, 1)
        self.outputs = _integer("outputs", outputs, 1)
        self.set_output_precision(
            np.ones(self.outputs) if output_precision is None else output_precision
        )
        if not np.isfinite(slowest) or slowest < 2.0:
            raise ValueError("slowest must be a finite timescale >= 2")
        self.slowest = float(slowest)
        rng = np.random.default_rng(seed)
        # Retention timescales log-spaced from two moments to ``slowest`` moments.
        timescales = np.exp(np.linspace(np.log(2.0), np.log(self.slowest), self.hidden))
        retention = 1.0 - 1.0 / timescales
        self._g = np.log(retention / (1.0 - retention))
        # A channel averaging white drive of unit variance fluctuates with standard
        # deviation sqrt((1 - l) / (1 + l)); the record reads each channel in that unit,
        # so a slow channel's small deviations address records as well as a fast one's.
        self._scale = np.sqrt((1.0 + retention) / (1.0 - retention))
        self._G = np.zeros((self.hidden, self.inputs))
        self._B = rng.normal(size=(self.hidden, self.inputs)) / np.sqrt(self.inputs)
        self._b = np.zeros(self.hidden)
        self._C = rng.normal(size=(self.outputs, self.hidden)) / np.sqrt(self.hidden)
        self._c = np.zeros(self.outputs)
        self.records = Records(
            self.inputs + self.hidden,
            {"y": self.outputs},
            cells=cells,
            active=active,
            rate=record_rate,
            habituation=habituation,
            bias=record_bias,
            seed=seed,
        )
        self._state: np.ndarray | None = None
        self.updates = 0
        self._revision = 0
        self._state_revision: int | None = None

    # ----------------------------------------------------------------- state

    @property
    def state(self) -> np.ndarray | None:
        return None if self._state is None else self._state.copy()

    @property
    def output_precision(self) -> np.ndarray:
        """Detached positive per-output teaching weights, not learned parameters."""
        return self._output_precision.copy()

    def set_output_precision(self, precision: np.ndarray) -> None:
        """Atomically change the supplied loss geometry; the free path does not use it."""
        value = np.asarray(precision)
        if value.dtype.kind not in "fiu" or value.shape != (self.outputs,):
            raise ValueError("output_precision must be a real vector with one entry per output")
        value = value.astype(float, copy=True)
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("output_precision must be finite and strictly positive")
        self._output_precision = value

    def parameters(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, "_" + k).copy() for k in _PARAMETERS}

    def _shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "G": (self.hidden, self.inputs),
            "g": (self.hidden,),
            "B": (self.hidden, self.inputs),
            "b": (self.hidden,),
            "C": (self.outputs, self.hidden),
            "c": (self.outputs,),
        }

    def set_parameters(self, parameters: Mapping[str, np.ndarray]) -> None:
        """Explicitly replace all learned arrays after validating them atomically."""
        expected = self._shapes()
        if set(parameters) != set(expected):
            raise ValueError(f"parameters must contain exactly {', '.join(_PARAMETERS)}")
        values = {}
        for key, shape in expected.items():
            value = np.asarray(parameters[key])
            if value.dtype.kind not in "fiu" or value.shape != shape:
                raise ValueError(f"{key} must be a real array of shape {shape}")
            value = value.astype(float, copy=True)
            if not np.isfinite(value).all():
                raise ValueError("parameters must be finite")
            values[key] = value
        for key, value in values.items():
            setattr(self, "_" + key, value)
        self._revision += 1

    def _path(self, values: np.ndarray, ports: int, name: str) -> np.ndarray:
        raw = np.asarray(values)
        if (
            raw.dtype.kind not in "fiu"
            or raw.ndim != 3
            or not raw.shape[0]
            or not raw.shape[1]
            or raw.shape[2] != ports
        ):
            raise ValueError(f"{name} must be a real (batch, time, {ports}) array")
        result = raw.astype(float, copy=False)
        if not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite")
        return result

    def _boundary(self, batch: int, state: np.ndarray | None) -> np.ndarray:
        source = self._state if state is None else state
        if source is None:
            return np.zeros((batch, self.hidden))
        value = np.asarray(source)
        if value.dtype.kind not in "fiu" or value.shape != (batch, self.hidden):
            raise ValueError("state must match (batch, hidden); reset when streams change")
        value = value.astype(float, copy=True)
        if not np.isfinite(value).all():
            raise ValueError("state must be finite")
        return value

    def _teaching(self, inputs: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        path = self._path(inputs, self.inputs, "inputs")
        teaching = self._path(target, self.outputs, "target")
        if teaching.shape[:2] != path.shape[:2]:
            raise ValueError("target batch/time dimensions must match inputs")
        return path, teaching

    # --------------------------------------------------------------- forward

    def _readings(self, inputs: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """What the record reads at each moment: the input and the scaled context."""
        return np.concatenate((inputs, hidden * self._scale), axis=-1)

    def _forward(
        self, inputs: np.ndarray, boundary: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The free path: context, gate, port, code, read and output at every moment.

        The context does not depend on the reads, so the scan runs first and
        every moment's reading is coded in one batched projection."""
        batch, horizon, _ = inputs.shape
        gate = _sigmoid(self._g + inputs @ self._G.T)
        port = np.tanh(inputs @ self._B.T + self._b)
        hidden = np.empty((batch, horizon, self.hidden))
        previous = boundary
        for t in range(horizon):
            hidden[:, t] = gate[:, t] * previous + (1.0 - gate[:, t]) * port[:, t]
            previous = hidden[:, t]
        readings = self._readings(inputs, hidden).reshape(batch * horizon, -1)
        codes = self.records.code(readings, valued=False)[0].reshape(batch, horizon, -1)
        read = codes @ self.records.tables["y"]
        output = hidden @ self._C.T + self._c + read
        return hidden, gate, port, codes, read, output

    def _loss(self, output: np.ndarray, target: np.ndarray) -> float | None:
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(0.5 * np.mean(self._output_precision * (output - target) ** 2))
        return value if np.isfinite(value) else None

    def _free(
        self, inputs: np.ndarray, boundary: np.ndarray, target: np.ndarray | None
    ) -> tuple[RecordPath, np.ndarray, np.ndarray]:
        hidden, gate, port, codes, read, output = self._forward(inputs, boundary)
        loss = None if target is None else self._loss(output, target)
        return RecordPath(hidden, output, gate, read, loss), port, codes

    def _gradient(
        self,
        inputs: np.ndarray,
        boundary: np.ndarray,
        target: np.ndarray,
        path: RecordPath,
        port: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """The adjoint scan: dL/dtheta for L = 1/2 mean(w (y - target)^2)."""
        batch, horizon, _ = inputs.shape
        hidden, gate = path.hidden, path.gate
        d = self._output_precision * (path.output - target) / (batch * horizon * self.outputs)
        previous = np.concatenate((boundary[:, None], hidden[:, :-1]), axis=1)
        gh = d @ self._C
        carried = np.zeros((batch, self.hidden))
        for t in range(horizon - 1, -1, -1):
            gh[:, t] += carried
            carried = gate[:, t] * gh[:, t]
        gp = (1.0 - gate) * gh * (1.0 - port**2)
        gs = gh * (previous - port) * gate * (1.0 - gate)
        return {
            "G": np.einsum("bti,btj->ij", gs, inputs),
            "g": gs.sum(axis=(0, 1)),
            "B": np.einsum("bti,btj->ij", gp, inputs),
            "b": gp.sum(axis=(0, 1)),
            "C": np.einsum("bti,btj->ij", d, hidden),
            "c": d.sum(axis=(0, 1)),
        }

    # ------------------------------------------------------------ interface

    def imagine(self, inputs: np.ndarray, *, state: np.ndarray | None = None) -> RecordPath:
        """Free private rollout; parameters, records, state and readback do not change."""
        path = self._path(inputs, self.inputs, "inputs")
        boundary = self._boundary(len(path), state)
        return self._free(path, boundary, None)[0]

    def _carry(self, path: RecordPath) -> None:
        self._state = path.final_state
        self._state_revision = self._revision

    def advance(self, inputs: np.ndarray) -> RecordPath:
        """Advance live activity without learning or writing."""
        path = self.imagine(inputs)
        self._carry(path)
        return path

    def observe(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        *,
        rate: float = 1.0,
        backtrack: bool = False,
        write: bool = True,
    ) -> RecordObservation:
        """Learn one finite observed path and write its observed outputs into the records.

        The prediction uses the records as they stood at the start of the
        call. The slow parameters move against the adjoint gradient of the
        precision-weighted half mean squared error; with ``backtrack=True`` a
        step is admitted only after target-free replay from the original
        boundary lowers that loss, trying up to sixteen halved rates. The
        observed outputs are then written into the records of the readings
        that produced the prediction. ``write=False`` learns without writing.
        The final free context becomes the live state.
        """
        if not isinstance(backtrack, (bool, np.bool_)):
            raise ValueError("backtrack must be a boolean")
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("rate must be finite and nonnegative")
        path, teaching = self._teaching(inputs, target)
        boundary = self._boundary(len(path), None)
        prediction, port, _ = self._free(path, boundary, teaching)
        self._carry(prediction)
        if prediction.loss is None:
            return RecordObservation(False, "nonfinite_prediction", prediction)
        delta = self._gradient(path, boundary, teaching, prediction, port)
        writes = self._write(path, teaching, prediction.hidden) if write else 0
        result = self._admit(path, teaching, boundary, prediction, delta, rate, backtrack)
        return RecordObservation(
            result.updated,
            result.reason,
            prediction,
            delta,
            result.initial_loss,
            result.final_loss,
            result.accepted_rate,
            result.replay_losses,
            result.replay_calls,
            writes,
        )

    def _admit(
        self,
        path: np.ndarray,
        target: np.ndarray,
        boundary: np.ndarray,
        prediction: RecordPath,
        delta: dict[str, np.ndarray],
        rate: float,
        backtrack: bool,
    ) -> RecordObservation:
        initial = prediction.loss
        if not backtrack:
            with np.errstate(over="ignore", invalid="ignore"):
                proposed = {k: getattr(self, "_" + k) - rate * delta[k] for k in delta}
            if not all(np.isfinite(p).all() for p in proposed.values()):
                return RecordObservation(False, "nonfinite_update", prediction, delta, initial)
            self._commit(proposed)
            return RecordObservation(True, "updated", prediction, delta, initial)
        with np.errstate(over="ignore", invalid="ignore"):
            norm_squared = sum(float(np.sum(value * value)) for value in delta.values())
        losses: list[float | None] = []
        replays = 0
        if initial is not None and np.isfinite(norm_squared) and norm_squared > 0 and rate > 0:
            current = self.parameters()
            trial = RecordPatchNet.restore(self.snapshot())
            for index in range(16):
                step = rate * 0.5**index
                with np.errstate(over="ignore", invalid="ignore"):
                    proposed = {k: current[k] - step * delta[k] for k in delta}
                if not all(np.isfinite(value).all() for value in proposed.values()):
                    losses.append(None)
                    continue
                trial.set_parameters(proposed)
                loss = trial._free(path, boundary, target)[0].loss
                replays += 1
                losses.append(loss)
                floor = (
                    64
                    * np.finfo(float).eps
                    * max(abs(initial), abs(loss or 0.0), np.finfo(float).tiny)
                )
                if (
                    loss is not None
                    and loss < initial - floor
                    and loss <= initial - 1e-4 * step * norm_squared
                ):
                    self._commit(proposed)
                    return RecordObservation(
                        True,
                        "updated",
                        prediction,
                        delta,
                        initial,
                        loss,
                        step,
                        tuple(losses),
                        replays,
                    )
        return RecordObservation(
            False,
            "no_decreasing_parameter_step",
            prediction,
            delta,
            initial,
            initial,
            0.0,
            tuple(losses),
            replays,
        )

    def _commit(self, proposed: Mapping[str, np.ndarray]) -> None:
        for key, value in proposed.items():
            setattr(self, "_" + key, value)
        self.updates += 1
        self._revision += 1

    def _write(self, inputs: np.ndarray, target: np.ndarray, hidden: np.ndarray) -> int:
        """Write what the slow readout got wrong into the records of each reading, in order.

        Witnessed readings first move the running mean (imagined readings never
        do); the writes then land at the codes later readings will use. The
        residual is taken against the parameters that made the prediction."""
        batch, horizon, _ = inputs.shape
        readings = self._readings(inputs, hidden).reshape(batch * horizon, -1)
        self.records.witness(readings)
        codes = self.records.code(readings, valued=False)[0].reshape(batch, horizon, -1)
        residual = target - (hidden @ self._C.T + self._c)
        written = 0
        for t in range(horizon):
            for row in range(batch):
                code = np.stack((codes[row, t], codes[row, t]))
                written += self.records.write(code, {"y": residual[row, t]})
        return written

    def detune(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        *,
        beta: float = 1e-3,
        state: np.ndarray | None = None,
        tolerance: float = 1e-14,
        max_iterations: int = 10_000,
    ) -> RecordContrast:
        """Solve both detuned equilibria of the quadratic energy and return their contrast.

        With ``y`` eliminated, each detuned path minimizes a quadratic that is
        strictly convex for the tested detuning; conjugate gradients from the
        free path solve its normal equations to ``tolerance``. The contrast of
        the energy's parameter derivatives at the two equilibria, divided by
        ``2 beta``, is the learning signal of equilibrium detuning. It equals
        the adjoint gradient of ``observe`` up to terms of order ``beta^2``.
        Nothing here changes the network; this is the acceptance check of
        that identity.
        """
        path, teaching = self._teaching(inputs, target)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        boundary = self._boundary(len(path), state)
        hidden, gate, port, _, read, _ = self._forward(path, boundary)
        batch, horizon, _ = path.shape
        b = beta * self._output_precision / (horizon * self.outputs)
        if np.any(b >= 1.0):
            raise ValueError("beta*output_precision/(time*outputs) must be below one")
        q = read + self._c - teaching
        converged = True

        def seam(v: np.ndarray) -> np.ndarray:
            out = v.copy()
            out[:, 1:] -= gate[:, 1:] * v[:, :-1]
            return out

        def seam_transpose(w: np.ndarray) -> np.ndarray:
            out = w.copy()
            out[:, :-1] -= gate[:, 1:] * w[:, 1:]
            return out

        constant = seam(hidden)  # the free path has zero defect: S h = a exactly

        def solve(weight: np.ndarray) -> tuple[np.ndarray, int, float]:
            nonlocal converged

            def operator(v: np.ndarray) -> np.ndarray:
                return np.asarray(seam_transpose(seam(v)) + ((v @ self._C.T) * weight) @ self._C)

            rhs = seam_transpose(constant) - (q * weight) @ self._C
            x = hidden.copy()
            r = rhs - operator(x)
            p = r.copy()
            rr = float(np.sum(r * r))
            scale = max(float(np.sum(rhs * rhs)), np.finfo(float).tiny)
            iterations = 0
            while np.sqrt(rr / scale) > tolerance and iterations < max_iterations:
                ap = operator(p)
                curvature = float(np.sum(p * ap))
                if not curvature > 0.0:
                    converged = False
                    break
                alpha = rr / curvature
                x += alpha * p
                r -= alpha * ap
                new = float(np.sum(r * r))
                p = r + (new / rr) * p
                rr = new
                iterations += 1
            residual = float(np.sqrt(rr / scale))
            if residual > tolerance:
                converged = False
            return x, iterations, residual

        def derivatives(h: np.ndarray, sign: float) -> tuple[dict[str, np.ndarray], float]:
            bw = sign * b
            prev = np.concatenate((boundary[:, None], h[:, :-1]), axis=1)
            e = h - gate * prev - (1.0 - gate) * port
            estimate = h @ self._C.T + self._c + read
            y = (estimate + bw * teaching) / (1.0 + bw)
            r = y - estimate
            energy = float(
                0.5 * (np.sum(e * e) + np.sum(r * r)) / batch
                + 0.5 * sign * beta * np.mean(self._output_precision * (y - teaching) ** 2)
            )
            gp = -e * (1.0 - gate) * (1.0 - port**2)
            gs = -e * (prev - port) * gate * (1.0 - gate)
            grads = {
                "G": np.einsum("bti,btj->ij", gs, path) / batch,
                "g": gs.sum(axis=(0, 1)) / batch,
                "B": np.einsum("bti,btj->ij", gp, path) / batch,
                "b": gp.sum(axis=(0, 1)) / batch,
                "C": -np.einsum("bti,btj->ij", r, h) / batch,
                "c": -r.sum(axis=(0, 1)) / batch,
            }
            return grads, energy

        plus, plus_iterations, plus_residual = solve(b / (1.0 + b))
        minus, minus_iterations, minus_residual = solve(-b / (1.0 - b))
        plus_grads, plus_energy = derivatives(plus, 1.0)
        minus_grads, minus_energy = derivatives(minus, -1.0)
        contrast = {k: (plus_grads[k] - minus_grads[k]) / (2.0 * beta) for k in plus_grads}
        return RecordContrast(
            contrast,
            plus,
            minus,
            plus_energy,
            minus_energy,
            (plus_iterations, minus_iterations),
            (plus_residual, minus_residual),
            converged,
        )

    def reset(self) -> None:
        """Clear active context; retain learned parameters, records and counts."""
        self._state = None
        self._state_revision = None

    def readback(self) -> RecordReadback:
        return RecordReadback(
            self.state,
            self.updates,
            self.records.writes,
            self._revision,
            self._state_revision,
            self.records.parameters(),
        )

    # ------------------------------------------------------------ checkpoints

    def snapshot(self) -> dict[str, np.ndarray]:
        """Detached complete continuation state, without any retained training path."""
        meta = {
            "format": FORMAT,
            "inputs": self.inputs,
            "hidden": self.hidden,
            "outputs": self.outputs,
            "slowest": self.slowest,
            "records": self.records.to_dict(),
            "updates": self.updates,
            "parameter_revision": self._revision,
            "state_parameter_revision": self._state_revision,
        }
        result = {
            **self.parameters(),
            "output_precision": self.output_precision,
            "state": np.empty((0, self.hidden)) if self._state is None else self._state.copy(),
            "meta": np.array(json.dumps(meta, sort_keys=True, allow_nan=False)),
        }
        for key, value in self.records.state().items():
            result["records_" + key] = value
        return result

    @classmethod
    def restore(cls, snapshot: Mapping[str, np.ndarray]) -> RecordPatchNet:
        """Construct a fresh net from validated complete state; never load pickle."""
        try:
            meta = json.loads(str(snapshot["meta"]))
            if not isinstance(meta, dict) or meta.get("format") != FORMAT:
                raise ValueError("unsupported record-patch checkpoint")
            config: dict[str, Any] = dict(meta["records"])
            result = cls(
                meta["inputs"],
                meta["hidden"],
                meta["outputs"],
                seed=config["seed"],
                output_precision=snapshot["output_precision"],
                cells=config["cells"],
                active=config["active"],
                record_rate=config["rate"],
                habituation=config["habituation"],
                record_bias=config["bias"],
                slowest=meta["slowest"],
            )
            if result.records.to_dict() != config:
                raise ValueError("record configuration is not one this class constructs")
            result.set_parameters({k: snapshot[k] for k in _PARAMETERS})
            result.records.load_state(
                {
                    key[len("records_") :]: np.asarray(value)
                    for key, value in snapshot.items()
                    if key.startswith("records_")
                }
            )
            state = np.asarray(snapshot["state"])
            if state.ndim != 2 or state.shape[1] != result.hidden:
                raise ValueError("invalid saved state shape")
            result._state = None if not len(state) else result._boundary(len(state), state)
            result.updates = _integer("updates", meta["updates"], 0)
            result._revision = _integer("parameter_revision", meta["parameter_revision"], 0)
            revision = meta["state_parameter_revision"]
            result._state_revision = (
                None if revision is None else _integer("state_parameter_revision", revision, 0)
            )
            if (result._state is None) != (result._state_revision is None) or (
                result._state_revision is not None and result._state_revision > result._revision
            ):
                raise ValueError("invalid saved state revision")
            return result
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise ValueError("invalid record-patch checkpoint") from error

    def save(self, path: str | Path, *, compressed: bool = True) -> Path:
        from .checkpoint import _write

        return _write(self.snapshot(), path, compressed=compressed)

    @classmethod
    def load(cls, path: str | Path) -> RecordPatchNet:
        with np.load(path, allow_pickle=False) as arrays:
            return cls.restore({k: arrays[k] for k in arrays.files})
