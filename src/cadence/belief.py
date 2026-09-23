"""A belief patch: a learned transition under action, evidence repair by iteration, a store read
inside the repair, and a readout the store patches.

The record patch computes its context from the input of the moment; without an input there
is no context, so it cannot imagine on its own. This patch keeps a belief ``z`` that a learned
transition carries forward under the executed action, and lets the evidence of the moment
repair it through a few iterations of one nonlinear map that reads the belief, the encoded
observation, the transition's expectation and the record store together. Scene and action
meet inside that map. An imagined moment is the transition alone under a declared action,
with the store read at the expected belief and nothing observed; imagination never writes.

    p[t]    = g * z[t-1] + (1 - g) * tanh(T [z[t-1]; a[t-1]] + t_b),  g = sigmoid(G [z; a] + g_b)
    e[t]    = tanh(port(o[t]) + e_b)                                   the encoded evidence
    z(0)    = p[t]
    m(k)    = read(code([e[t], z(k)]))                                 the store, a coded residual
    z(k+1)      = (1 - alpha) z(k) + alpha tanh(F [z(k); e[t]; p[t]; m(k); 1] + f_b)
    y[t]        = C z(K) + c + decode(m(K))

Learning is one backward scan over a chunk through every iteration and the transition; the
record read is treated as given (no gradient reaches the store), as in the record patch. The
store holds the residual of the slow readout at the code of the final reading, written once
per observed moment. Everything private (``imagine``) leaves parameters, records, state and
counters unchanged and consumes no observation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ports import StructuredPort
from .records import Records

FORMAT = "cadence-belief/1"
_PARAMETERS = ("E", "e_b", "T", "t_b", "G", "g_b", "F", "f_b", "C", "c")


def _integer(name: str, value: Any, low: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < low
    ):
        raise ValueError(f"{name} must be an integer >= {low}")
    return int(value)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)))


@dataclass(frozen=True)
class BeliefPath:
    """A detached path: beliefs ``(batch, time, belief)``, the transition's expectations, the
    residual of the last repair iteration at each moment, and the outputs (slow readout plus
    the decoded store read)."""

    belief: np.ndarray
    expectation: np.ndarray
    residual: np.ndarray
    output: np.ndarray
    read: np.ndarray
    loss: float | None
    step: np.ndarray | None = None
    """The last repair iteration's move per belief unit, ``(batch, time, belief)``; its norm
    over the units is ``residual``. Where the belief moved under the evidence."""

    @property
    def slow_output(self) -> np.ndarray:
        return np.asarray(self.output - self.read)

    @property
    def final_state(self) -> np.ndarray:
        return self.belief[:, -1].copy()


@dataclass(frozen=True)
class BeliefObservation:
    updated: bool
    reason: str
    path: BeliefPath
    delta: dict[str, np.ndarray]
    initial_loss: float | None
    writes: int


class BeliefPatch:
    def __init__(
        self,
        observation: StructuredPort,
        actions: int,
        belief: int,
        outputs: int,
        *,
        iterations: int = 2,
        damping: float = 0.5,
        cells: int = 4096,
        active: int = 32,
        record_rate: float = 0.5,
        record_width: int = 64,
        habituation: float = 1e-5,
        record_bias: float = 0.3,
        output_precision: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        self.port = observation
        self.inputs = observation.inputs
        self.encoded = observation.outputs
        self.actions = _integer("actions", actions, 1)
        self.belief = _integer("belief", belief, 1)
        self.outputs = _integer("outputs", outputs, 1)
        self.iterations = _integer("iterations", iterations, 1)
        if not 0.0 < float(damping) <= 1.0:
            raise ValueError("damping lies in (0, 1]")
        self.damping = float(damping)
        self.record_width = _integer("record_width", record_width, 1)
        self.set_output_precision(
            np.ones(self.outputs) if output_precision is None else output_precision
        )
        rng = np.random.default_rng(seed)
        za = self.belief + self.actions
        self._E = np.concatenate([w.ravel() for w in observation.initial(rng, 1.0)])
        self._e_b = np.zeros(self.encoded)
        self._T = rng.normal(size=(self.belief, za)) / np.sqrt(za)
        self._t_b = np.zeros(self.belief)
        self._G = np.zeros((self.belief, za))
        self._g_b = np.full(self.belief, 1.0)  # retain most of the belief across a moment at birth
        fi = self.belief + self.encoded + self.belief + self.record_width + 1
        self._F = rng.normal(size=(self.belief, fi)) / np.sqrt(fi)
        self._f_b = np.zeros(self.belief)
        self._C = np.zeros((self.outputs, self.belief))
        self._c = np.zeros(self.outputs)
        signs = rng.integers(0, 2, (self.outputs, self.record_width))
        self._output_code = (2.0 * signs - 1.0) / np.sqrt(self.record_width)
        self.records = Records(
            self.encoded + self.belief,
            {"y": self.record_width},
            cells=cells,
            active=active,
            rate=record_rate,
            habituation=habituation,
            bias=record_bias,
            seed=seed,
        )
        self._input_norm = 1.0
        self._state: np.ndarray | None = None
        self.updates = 0

    # ------------------------------------------------------------------ parameters
    @property
    def state(self) -> np.ndarray | None:
        return None if self._state is None else self._state.copy()

    def set_output_precision(self, precision: np.ndarray) -> None:
        value = np.asarray(precision, dtype=float)
        if value.shape != (self.outputs,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("output_precision must be finite and positive, one per output")
        self._output_precision = value.copy()

    def parameters(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, "_" + k).copy() for k in _PARAMETERS}

    def set_parameters(self, parameters: Mapping[str, np.ndarray]) -> None:
        current = self.parameters()
        if set(parameters) != set(current):
            raise ValueError(f"parameters must contain exactly {', '.join(_PARAMETERS)}")
        values = {}
        for key, old in current.items():
            value = np.asarray(parameters[key], dtype=float)
            if value.shape != old.shape or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite with shape {old.shape}")
            values[key] = value.copy()
        for key, value in values.items():
            setattr(self, "_" + key, value)

    def _blocks(self) -> list[np.ndarray]:
        out, at = [], 0
        for b in self.port.blocks:
            shape = self.port.weight_shape(b)
            size = int(np.prod(shape))
            out.append(self._E[at : at + size].reshape(shape))
            at += size
        return out

    # ------------------------------------------------------------------ the moment
    def _encode(self, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pre = self.port.apply(o, self._blocks()) + self._e_b
        return np.tanh(pre), pre

    def _expect(self, z: np.ndarray, a: np.ndarray) -> dict[str, np.ndarray]:
        x = np.concatenate([z, a], axis=-1)
        g = _sigmoid(x @ self._G.T + self._g_b)
        cand = np.tanh(x @ self._T.T + self._t_b)
        return {"x": x, "g": g, "cand": cand, "p": g * z + (1.0 - g) * cand}

    def _reading(self, e: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.concatenate([e * (np.sqrt(self.encoded) / self._input_norm), z], axis=-1)

    def _read(self, e: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        code = self.records.code(self._reading(e, z), valued=False)[0]
        return code @ self.records.tables["y"], code

    def _repair(self, e: np.ndarray, p: np.ndarray, observed: bool) -> dict[str, Any]:
        """The iterations of one moment; with nothing observed the belief is the expectation."""
        n = len(p)
        zs, hs, us, ms = [p], [], [], []
        flag = np.full((n, 1), 1.0 if observed else 0.0)
        z = p
        if observed:
            for _ in range(self.iterations):
                m, _ = self._read(e, z)
                u = np.concatenate([z, e, p, m, flag], axis=-1)
                h = np.tanh(u @ self._F.T + self._f_b)
                z = (1.0 - self.damping) * z + self.damping * h
                zs.append(z)
                hs.append(h)
                us.append(u)
                ms.append(m)
        m, code = self._read(e, z)
        step = zs[-1] - zs[-2] if len(zs) > 1 else np.zeros_like(p)
        residual = np.linalg.norm(step, axis=-1)
        return {"z": z, "zs": zs, "hs": hs, "us": us, "read": m, "code": code, "residual": residual, "step": step}

    def _forward(
        self,
        observations: np.ndarray | None,
        actions: np.ndarray,
        boundary: np.ndarray,
        observed: np.ndarray,
    ) -> dict[str, Any]:
        n, t = actions.shape[:2]
        z = boundary
        record: dict[str, list[Any]] = {
            k: [] for k in ("expect", "e", "e_pre", "repair", "y", "read", "code")
        }
        for k in range(t):
            ex = self._expect(z, actions[:, k])
            if observations is not None and observed[k]:
                e, e_pre = self._encode(observations[:, k])
            else:
                e, e_pre = np.zeros((n, self.encoded)), np.zeros((n, self.encoded))
            rep = self._repair(e, ex["p"], bool(observed[k]))
            z = rep["z"]
            read = rep["read"] @ self._output_code.T
            y = z @ self._C.T + self._c + read
            for key, value in (
                ("expect", ex),
                ("e", e),
                ("e_pre", e_pre),
                ("repair", rep),
                ("y", y),
                ("read", read),
                ("code", rep["code"]),
            ):
                record[key].append(value)
        return record

    def _path(self, record: dict[str, list[Any]], target: np.ndarray | None) -> BeliefPath:
        belief = np.stack([r["z"] for r in record["repair"]], axis=1)
        expectation = np.stack([x["p"] for x in record["expect"]], axis=1)
        residual = np.stack([r["residual"] for r in record["repair"]], axis=1)
        step = np.stack([r["step"] for r in record["repair"]], axis=1)
        output = np.stack(record["y"], axis=1)
        read = np.stack(record["read"], axis=1)
        loss = None if target is None else self._loss(output - read, target)
        return BeliefPath(belief, expectation, residual, output, read, loss, step)

    def _loss(self, slow: np.ndarray, target: np.ndarray) -> float | None:
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(0.5 * np.mean(self._output_precision * (slow - target) ** 2))
        return value if np.isfinite(value) else None

    # ------------------------------------------------------------------ interface
    def _check(
        self, observations: Any, actions: Any, observed: Any
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        a = np.asarray(actions, dtype=float)
        if a.ndim != 3 or a.shape[2] != self.actions or not np.isfinite(a).all():
            raise ValueError(f"actions must be a finite (batch, time, {self.actions}) array")
        n, t = a.shape[:2]
        if observations is None:
            o = None
            mask = np.zeros(t, dtype=bool)
        else:
            o = np.asarray(observations, dtype=float)
            if o.shape != (n, t, self.inputs) or not np.isfinite(o).all():
                raise ValueError(
                    f"observations must be a finite (batch, time, {self.inputs}) array"
                )
            mask = (
                np.ones(t, dtype=bool)
                if observed is None
                else np.asarray(observed, dtype=bool).reshape(t)
            )
        return o, a, mask

    def _boundary(self, n: int, state: np.ndarray | None) -> np.ndarray:
        source = self._state if state is None else state
        if source is None:
            return np.zeros((n, self.belief))
        value = np.asarray(source, dtype=float)
        if value.shape != (n, self.belief) or not np.isfinite(value).all():
            raise ValueError("state must match (batch, belief); reset when streams change")
        return value.copy()

    def assimilate(
        self, observations: np.ndarray, actions: np.ndarray, observed: np.ndarray | None = None, *, state: np.ndarray | None = None
    ) -> BeliefPath:
        """Advance the belief through observed moments: the executed action, then the evidence.
        Nothing is learned or written; the final belief becomes the live state. ``state`` starts
        the moments from a given boundary instead of the live belief (a window that is replayed
        from the belief that was live at its first moment)."""
        o, a, mask = self._check(observations, actions, observed)
        record = self._forward(o, a, self._boundary(len(a), state), mask)
        path = self._path(record, None)
        self._state = path.final_state
        return path

    def imagine(self, actions: np.ndarray, *, state: np.ndarray | None = None) -> BeliefPath:
        """A private continuation under declared actions from the live belief (or ``state``):
        the transition alone, the store read at the expectation, no observation, no change."""
        _, a, mask = self._check(None, actions, None)
        record = self._forward(None, a, self._boundary(len(a), state), mask)
        return self._path(record, None)

    def observe(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        target: np.ndarray,
        *,
        observed: np.ndarray | None = None,
        rate: float = 1.0,
        write: bool = True,
        state: np.ndarray | None = None,
    ) -> BeliefObservation:
        """Learn one chunk of witnessed moments and write their outcomes into the store. ``state``
        starts the chunk from a given boundary instead of the live belief; the final belief under
        the chunk becomes the live state either way."""
        o, a, mask = self._check(observations, actions, observed)
        y = np.asarray(target, dtype=float)
        if y.shape != (*a.shape[:2], self.outputs) or not np.isfinite(y).all():
            raise ValueError("target must be a finite (batch, time, outputs) array")
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("rate must be finite and nonnegative")
        if o is None:
            raise ValueError("observe needs observations; imagine is the private continuation")
        boundary = self._boundary(len(a), state)
        record = self._forward(o, a, boundary, mask)
        path = self._path(record, y)
        self._state = path.final_state
        if path.loss is None:
            return BeliefObservation(False, "nonfinite_prediction", path, {}, None, 0)
        delta = self._adjoint(o, a, boundary, record, y)
        writes = self._write(record, y) if write else 0
        if rate > 0:
            proposed = {k: getattr(self, "_" + k) - rate * delta[k] for k in _PARAMETERS}
            if not all(np.isfinite(v).all() for v in proposed.values()):
                return BeliefObservation(False, "nonfinite_update", path, delta, path.loss, writes)
            for k, v in proposed.items():
                setattr(self, "_" + k, v)
            self.updates += 1
            return BeliefObservation(True, "updated", path, delta, path.loss, writes)
        return BeliefObservation(False, "no_step", path, delta, path.loss, writes)

    def reset(self) -> None:
        self._state = None

    # ------------------------------------------------------------------ learning
    def _adjoint(
        self,
        o: np.ndarray,
        a: np.ndarray,
        boundary: np.ndarray,
        record: dict[str, list[Any]],
        target: np.ndarray,
    ) -> dict[str, np.ndarray]:
        n, t = a.shape[:2]
        alpha = self.damping
        delta = {k: np.zeros_like(getattr(self, "_" + k)) for k in _PARAMETERS}
        block_grads = [np.zeros(self.port.weight_shape(b)) for b in self.port.blocks]
        dz_next = np.zeros((n, self.belief))  # gradient into z[t] from moment t+1
        for k in range(t - 1, -1, -1):
            rep, ex = record["repair"][k], record["expect"][k]
            z_prev = boundary if k == 0 else record["repair"][k - 1]["z"]
            slow = record["y"][k] - record["read"][k]
            dy = self._output_precision * (slow - target[:, k]) / (n * t * self.outputs)
            delta["C"] += dy.T @ rep["z"]
            delta["c"] += dy.sum(axis=0)
            dz = dy @ self._C + dz_next
            e = record["e"][k]
            de = np.zeros((n, self.encoded))
            dp = np.zeros((n, self.belief))
            for j in range(len(rep["hs"]) - 1, -1, -1):
                h, u = rep["hs"][j], rep["us"][j]
                dh = alpha * dz
                dz = (1.0 - alpha) * dz
                dpre = dh * (1.0 - h**2)
                delta["F"] += dpre.T @ u
                delta["f_b"] += dpre.sum(axis=0)
                du = dpre @ self._F
                dz = dz + du[:, : self.belief]
                de += du[:, self.belief : self.belief + self.encoded]
                dp += du[:, self.belief + self.encoded : 2 * self.belief + self.encoded]
            dp += dz  # z(0) = p
            if record["e_pre"][k] is not None and np.any(de):
                dpre_e = de * (1.0 - e**2)
                delta["e_b"] += dpre_e.sum(axis=0)
                for i, gb in enumerate(self.port.gradient(dpre_e, o[:, k])):
                    block_grads[i] += gb
            g, cand, x = ex["g"], ex["cand"], ex["x"]
            dz_prev = g * dp
            dg = dp * (z_prev - cand)
            dcand = dp * (1.0 - g)
            dpre_c = dcand * (1.0 - cand**2)
            delta["T"] += dpre_c.T @ x
            delta["t_b"] += dpre_c.sum(axis=0)
            dpre_g = dg * g * (1.0 - g)
            delta["G"] += dpre_g.T @ x
            delta["g_b"] += dpre_g.sum(axis=0)
            dx = dpre_c @ self._T + dpre_g @ self._G
            dz_next = dz_prev + dx[:, : self.belief]
        delta["E"] = np.concatenate([gb.ravel() for gb in block_grads])
        return delta

    def _write(self, record: dict[str, list[Any]], target: np.ndarray) -> int:
        """Write the slow readout's residual, coded, at the final reading of each observed
        moment."""
        written = 0
        t = len(record["y"])
        for k in range(t):
            rep = record["repair"][k]
            e = record["e"][k]
            if not np.any(e):
                continue  # nothing observed: nothing to write
            readings = self._reading(e, rep["z"])
            for value in np.linalg.norm(e, axis=1):
                self._input_norm += 0.01 * (max(float(value), 1e-6) - self._input_norm)
            self.records.witness(readings)
            codes = self.records.code(readings, valued=False)[0]
            residual = (target[:, k] - (record["y"][k] - record["read"][k])) @ self._output_code
            for row in range(len(codes)):
                written += self.records.write(
                    np.stack((codes[row], codes[row])), {"y": residual[row]}
                )
        return written

    # ------------------------------------------------------------------ custody
    def snapshot(self) -> dict[str, np.ndarray]:
        meta = {
            "format": FORMAT,
            "port": self.port.to_dict(),
            "actions": self.actions,
            "belief": self.belief,
            "outputs": self.outputs,
            "iterations": self.iterations,
            "damping": self.damping,
            "record_width": self.record_width,
            "records": self.records.to_dict(),
            "updates": self.updates,
        }
        out = {
            **self.parameters(),
            "output_precision": self._output_precision.copy(),
            "output_code": self._output_code.copy(),
            "input_norm": np.array(self._input_norm),
            "state": np.empty((0, self.belief)) if self._state is None else self._state.copy(),
            "meta": np.array(json.dumps(meta, sort_keys=True)),
        }
        for key, value in self.records.state().items():
            out["records_" + key] = value
        return out

    @classmethod
    def restore(cls, snapshot: Mapping[str, np.ndarray]) -> BeliefPatch:
        try:
            meta = json.loads(str(snapshot["meta"]))
            if meta.get("format") != FORMAT:
                raise ValueError("unsupported belief checkpoint")
            config = dict(meta["records"])
            result = cls(
                StructuredPort.from_dict(meta["port"]),
                meta["actions"],
                meta["belief"],
                meta["outputs"],
                iterations=meta["iterations"],
                damping=meta["damping"],
                cells=config["cells"],
                active=config["active"],
                record_rate=config["rate"],
                record_width=meta["record_width"],
                habituation=config["habituation"],
                record_bias=config["bias"],
                output_precision=snapshot["output_precision"],
                seed=config["seed"],
            )
            result.set_parameters({k: snapshot[k] for k in _PARAMETERS})
            result._output_code = np.asarray(snapshot["output_code"], dtype=float).copy()
            result._input_norm = float(np.asarray(snapshot["input_norm"]))
            result.records.load_state(
                {
                    k[len("records_") :]: np.asarray(v)
                    for k, v in snapshot.items()
                    if k.startswith("records_")
                }
            )
            state = np.asarray(snapshot["state"])
            result._state = None if not len(state) else state.astype(float).copy()
            result.updates = int(meta["updates"])
            return result
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("invalid belief checkpoint") from error

    def save(self, path: str | Path, *, compressed: bool = True) -> Path:
        from .checkpoint import _write

        return _write(self.snapshot(), path, compressed=compressed)

    @classmethod
    def load(cls, path: str | Path) -> BeliefPatch:
        with np.load(path, allow_pickle=False) as arrays:
            return cls.restore({k: arrays[k] for k in arrays.files})
