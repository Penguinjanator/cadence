"""A learned code of meaning: a region of the brain's own neurons settling to a sparse code.

``Brain.compose(..., learned_code=units)`` places a ``code`` region of that many neurons
between the sensory region and the first module. Its synapses are the brain's synapses,
read in every sweep as every other synapse is, and the region settles with the rest of the
brain under the brain's neuron model:

* the dictionary ``Phi`` (``inputs`` by ``units``, unit-length columns) is the one-way
  synapses from the sensory neurons: a code unit ``j`` is driven by ``Phi_j . s_sensory``;
* the dictionary's Gram matrix is the lateral inhibition among the code units: every ordered
  pair ``j != k`` carries ``-(Phi_j . Phi_k)``, so units whose columns overlap compete;
* the predictor ``W`` (``units`` by the prefrontal width) is the one-way synapses from the
  prefrontal region, the working trace of the association activity of the moment before,
  scaled by ``top_down``: the context's prediction of the code lowers the thresholds of the
  units it predicts;
* each unit's threshold is its bias, ``-t_j``.

A code unit therefore settles to ``v_j = Phi_j . (s_sensory - Phi s) + s_j + top_down p_j
- t_j`` with ``s_j = f(v_j)`` the neuron's rectified sigmoid and ``p = W s_prefrontal``: the
sparse state that reconstructs the sensory activation through the dictionary as far as the
thresholds allow, drawn toward what the context predicts. No synapse leads back into the
region, so its fixed point depends on the observation and the trace alone; the brain settles
it first, with the same solver and tolerance, addresses its memories on it, and settles the
whole graph warm from it.

Learning is local to each boundary and happens at every accepted moment, each row of the
batch one event, from the settled state alone (``a = max(s, 0)``, the rectified code;
``x'`` the sensory activation; ``z`` the prefrontal activation; ``p = W z``):

* the dictionary, Hebbian on the bottom-up residual, ``Phi += rate * sum_rows (x' - Phi a)
  a^T``, each column then scaled back to unit length (a unit's own synaptic scaling);
* the lateral weights, recomputed from the dictionary;
* the predictor, the delta rule on the top-down residual, ``W += sum_rows r_row (a - p)
  z^T`` with ``r_row = min(predictor, 1 / |z|^2)``, the row's rate bounded by its own
  input energy;
* the thresholds, homeostatic, ``t += threshold_rate * sum_rows ((a > 0) - target)``,
  floored at zero.

While the code fits, the residuals are small and nothing moves. The rates are genes
(``learned_code_<name>``); zero freezes a part. The memories key on the rectified settled
code at unit length (``Brain.observation_code``).

The code's fixed points are stable equilibria of the neuron equations (the lateral weights
are minus a Gram matrix, so the coupling's eigenvalues stay below the unit's slope), but the
brain's synchronous sweep at the model's step can oscillate around one when several units
with overlapping columns are active together: the sweep contracts only while ``dt * (1 -
mu_min)`` stays below 2, ``mu_min`` the smallest eigenvalue of the slope-weighted lateral
coupling. ``stability`` reports the halvings of the step the settled code needs, and the
brain integrates every phase at that step with its step budgets scaled to keep the same
time (``Brain`` re-steps its graph when the level changes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._configuration import real
from .brain import Brain as NeuralGraph
from .brain import BrainState, Equilibrium
from .connectome import Connectome
from .neuron import NeuronModel

__all__ = ["LearnedCode"]

# The dictionary's own stream of the brain's seed, apart from the streams the brain develops,
# acts and draws its fixed code from (``SeedSequence([seed, 7103])``).
_CODE_STREAM = 7109


def _unit_columns(values: np.ndarray) -> np.ndarray:
    """Each column scaled to unit length; a zero column stays zero."""
    norm = np.linalg.norm(values, axis=0, keepdims=True)
    return np.asarray(values / np.where(norm > 0.0, norm, 1.0))


def _unit_rows(values: np.ndarray) -> np.ndarray:
    """Each row scaled to unit length; a row without a positive entry stays zero."""
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return np.asarray(values / np.where(norm > 0.0, norm, 1.0))


@dataclass
class LearnedCode:
    """The learned code region's tables, genes, sub-network and rules; see the module note.

    ``inputs`` is the sensory width, ``units`` the region's, ``context`` the prefrontal
    width; ``model`` the brain's neuron model, under which the region settles. The genes are
    ``top_down`` (the prediction's weight on the units), ``rate`` (the dictionary's Hebbian
    rate per row), ``predictor`` (the delta rate per row), ``threshold`` (the initial
    threshold of every unit, a construction setting), ``threshold_rate`` and ``target`` (the
    share of rows a unit should be active in). Validation refuses a value outside its
    domain before anything is built.
    """

    inputs: int
    units: int
    context: int
    model: NeuronModel
    seed: int = 0
    top_down: float = 4.0
    rate: float = 0.125
    predictor: float = 0.03
    threshold: float = 0.1
    threshold_rate: float = 0.001
    target: float = 0.1
    place: str = "free"
    dictionary: np.ndarray = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)
    thresholds: np.ndarray = field(init=False, repr=False)

    GENES = ("top_down", "rate", "predictor", "threshold", "threshold_rate", "target", "place")
    RUNTIME = ("top_down", "rate", "predictor", "threshold_rate", "target", "place")
    CONSTRUCTION = ("threshold",)
    # what the place stores key on: the context-free code (``"free"``, the observation
    # settled alone, as ``Brain.observation_code`` gives it) or the moment's code in its
    # context (``"context"``, the key of the context records and the recall slots)
    PLACES = ("free", "context")
    # the sweep's contraction bound: ``dt * (1 - mu_min)`` is kept at or below this, so the
    # worst mode of the synchronous sweep contracts by at least a half per sweep
    STEP_MARGIN = 1.5

    @classmethod
    def founders(cls) -> dict[str, Any]:
        """The founder value of every gene."""
        return {
            "top_down": 4.0, "rate": 0.125, "predictor": 0.03, "threshold": 0.1,
            "threshold_rate": 0.001, "target": 0.1, "place": "free",
        }

    @staticmethod
    def setting(name: str, value: Any) -> Any:
        """One gene, validated and converted."""
        if name in ("top_down", "rate", "predictor", "threshold", "threshold_rate"):
            return real(name, value, low=0)
        if name == "target":
            number = real(name, value, low=0, high=1)
            if number <= 0 or number >= 1:
                raise ValueError("target must lie in (0, 1)")
            return number
        if name == "place":
            if not isinstance(value, str) or value not in LearnedCode.PLACES:
                raise ValueError(f"place is one of {LearnedCode.PLACES}")
            return value
        raise TypeError(f"unknown learned code setting: {name!r}")

    def __post_init__(self) -> None:
        for name in ("inputs", "units", "context"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        if self.inputs < 1 or self.context < 1:
            raise ValueError("inputs and context must be positive")
        if self.units < 2:
            raise ValueError("a learned code holds at least 2 units")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be a nonnegative integer")
        self.seed = int(self.seed)
        for name in self.GENES:
            setattr(self, name, self.setting(name, getattr(self, name)))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, _CODE_STREAM]))
        self.dictionary = _unit_columns(rng.standard_normal((self.inputs, self.units)))
        self.weights = np.zeros((self.units, self.context))
        self.thresholds = np.full(self.units, float(self.threshold))
        # the sub-network: the units with every ordered pair of lateral synapses
        pre, post = np.meshgrid(np.arange(self.units), np.arange(self.units), indexing="ij")
        apart = pre.ravel() != post.ravel()
        self._pre, self._post = pre.ravel()[apart], post.ravel()[apart]
        self.connectome = Connectome.from_synapses(
            self.units, pre=self._pre, post=self._post, sign=self.lateral()[self._pre, self._post],
            label=f"learned-code:{self.units}",
        )
        # the connectome sorts its synapses by (post, pre); the lateral weights are read in it
        self._pre, self._post = self.connectome.pre, self.connectome.post
        self.halvings = 0  # the step's halvings the settled code needs (``stability``)
        self.graph = NeuralGraph(
            self.connectome, self.model, efficacy=self.lateral_efficacy(), bias=-self.thresholds
        )
        self.work = {
            "settles": 0, "settle_rows": 0, "sweeps": 0, "capped": 0,
            "updates": 0, "update_rows": 0,
        }
        self.last: dict[str, Any] | None = None

    # -- the step

    def stability(self, v: np.ndarray) -> int:
        """The halvings of the model's step under which the synchronous sweep contracts
        around the fixed points at potentials ``v`` (``(batch, units)``): the least ``h``
        with ``dt / 2^h * (1 - mu_min) <= STEP_MARGIN``, ``mu_min`` the smallest eigenvalue
        over the rows of ``D^1/2 L D^1/2`` (``D`` the activation's slopes at ``v``, ``L``
        the lateral weights); 0 when the sweep contracts at the model's own step."""
        v = np.atleast_2d(np.asarray(v, dtype=float))
        root = np.sqrt(np.maximum(self.model.slope_at(v), 0.0))
        lateral = self.lateral()
        coupling = root[:, :, None] * lateral[None, :, :] * root[:, None, :]
        mu = float(np.linalg.eigvalsh(coupling)[:, 0].min())
        need = self.model.dt * (1.0 - mu) / self.STEP_MARGIN
        return 0 if need <= 1.0 else int(np.ceil(np.log2(need)))

    def restep(self, halvings: int) -> None:
        """The sub-network at the model's step halved ``halvings`` times."""
        if halvings == self.halvings:
            return
        self.halvings = int(halvings)
        self.graph = NeuralGraph(
            self.connectome,
            self.model.replace(dt=float(np.ldexp(self.model.dt, -self.halvings))),
            efficacy=self.lateral_efficacy(), bias=-self.thresholds,
        )

    # -- the tables as synapses

    def gram(self) -> np.ndarray:
        return np.asarray(self.dictionary.T @ self.dictionary)

    def lateral(self) -> np.ndarray:
        """The lateral weight from unit ``j`` to unit ``k``, ``-(Phi_j . Phi_k)``, as a
        ``(units, units)`` matrix whose diagonal is zero (a neuron has no synapse onto
        itself)."""
        out = -self.gram()
        np.fill_diagonal(out, 0.0)
        return np.asarray(out)

    def lateral_efficacy(self) -> np.ndarray:
        """The lateral weights in the sub-network's synapse order."""
        return np.asarray(self.lateral()[self._pre, self._post])

    def install(self) -> None:
        """The sub-network takes the tables as they are: the lateral weights and the
        thresholds as biases."""
        self.graph = self.graph.with_parameters(
            efficacy=self.lateral_efficacy(), bias=-self.thresholds
        )

    def take(self, dictionary: np.ndarray, weights: np.ndarray, thresholds: np.ndarray) -> None:
        """Replace the tables (a checkpoint's), validated, and install them."""
        dictionary = np.asarray(dictionary, dtype=float)
        weights = np.asarray(weights, dtype=float)
        thresholds = np.asarray(thresholds, dtype=float)
        if (
            dictionary.shape != (self.inputs, self.units)
            or weights.shape != (self.units, self.context)
            or thresholds.shape != (self.units,)
            or not np.isfinite(dictionary).all()
            or not np.isfinite(weights).all()
            or not np.isfinite(thresholds).all()
            or (thresholds < 0).any()
        ):
            raise ValueError("invalid learned code tables")
        norms = np.linalg.norm(dictionary, axis=0)
        if not np.allclose(norms, 1.0, rtol=0, atol=1e-9):
            raise ValueError("a learned code's dictionary has unit-length columns")
        self.dictionary, self.weights, self.thresholds = dictionary, weights, thresholds
        self.install()

    # -- the settle

    def stimulus(self, sensory: np.ndarray, context: np.ndarray | None) -> np.ndarray:
        """The drive of the code units, ``(batch, units)``: the dictionary's read of the
        sensory activation plus ``top_down`` times the predictor's read of the prefrontal
        activation; without a context the prediction is zero."""
        drive = np.asarray(sensory, dtype=float) @ self.dictionary
        if context is not None and self.top_down:
            drive = drive + self.top_down * self.prediction(context)
        return np.asarray(drive)

    def prediction(self, context: np.ndarray) -> np.ndarray:
        """``p = W z`` for the ``(batch, context)`` prefrontal activation."""
        return np.asarray(np.asarray(context, dtype=float) @ self.weights.T)

    def settle(
        self,
        stimulus: np.ndarray,
        warm: BrainState | None,
        *,
        budget: int,
        tolerance: float,
        damping: int = 3,
    ) -> Equilibrium:
        """The region settled under ``stimulus`` by the brain's solver policy: the fast
        repair at the model's step, then halved steps within the same ``budget``, to the
        residual ``tolerance``; counted as work."""
        phase = self.graph.equilibrate(
            stimulus, state=warm, budget=budget, tolerance=tolerance, damping=damping
        )
        self.work["settles"] += 1
        self.work["settle_rows"] += len(stimulus)
        self.work["sweeps"] += int(phase.steps)
        if not np.all(phase.qualified):
            self.work["capped"] += 1
        return phase

    def code_of(self, activation: np.ndarray) -> np.ndarray:
        """The code the memories key on: the rectified activation at unit length."""
        return _unit_rows(np.maximum(np.atleast_2d(np.asarray(activation, dtype=float)), 0.0))

    def residuals(
        self, sensory: np.ndarray, activation: np.ndarray, context: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """Per row: the bottom-up residual's squared norm over the sensory activation's,
        and the top-down residual's squared norm (zero without a context)."""
        x = np.atleast_2d(np.asarray(sensory, dtype=float))
        a = np.maximum(np.atleast_2d(np.asarray(activation, dtype=float)), 0.0)
        rx = x - a @ self.dictionary.T
        energy = np.maximum(np.sum(x * x, axis=1), 1e-12)
        top = np.zeros(len(x)) if context is None else np.sum((a - self.prediction(context)) ** 2,
                                                              axis=1)
        return {"bottom_up": np.sum(rx * rx, axis=1) / energy, "top_down": top}

    # -- learning

    def learn(
        self, sensory: np.ndarray, activation: np.ndarray, context: np.ndarray | None
    ) -> bool:
        """One step of the dictionary, the predictor and the thresholds from a settled
        moment, each row an event; returns whether any table moved (every rate zero, or an
        empty batch, moves nothing). The sub-network takes the new tables."""
        x = np.atleast_2d(np.asarray(sensory, dtype=float))
        a = np.maximum(np.atleast_2d(np.asarray(activation, dtype=float)), 0.0)
        n = len(x)
        if not n or not (self.rate or self.predictor or self.threshold_rate):
            return False
        moved = False
        if self.rate:
            # the columns of the units that fired move and are scaled back to unit length; a
            # silent unit's column stays as it is, bit for bit
            fired = (a > 0.0).any(axis=0)
            if fired.any():
                residual = x - a @ self.dictionary.T
                dictionary = self.dictionary.copy()
                dictionary[:, fired] = _unit_columns(
                    dictionary[:, fired] + self.rate * (residual.T @ a[:, fired])
                )
                if not np.isfinite(dictionary).all():
                    raise ValueError("the learned code's dictionary must remain finite")
                self.dictionary = dictionary
                moved = True
        if self.predictor and context is not None:
            z = np.atleast_2d(np.asarray(context, dtype=float))
            with np.errstate(divide="ignore"):
                bound = np.minimum(self.predictor, 1.0 / np.sum(z * z, axis=1))
            error = a - self.prediction(z)
            weights = self.weights + (error * bound[:, None]).T @ z
            if not np.isfinite(weights).all():
                raise ValueError("the learned code's predictor must remain finite")
            self.weights = weights
            moved = True
        if self.threshold_rate:
            active = (a > 0.0).astype(float) - self.target
            self.thresholds = np.maximum(
                self.thresholds + self.threshold_rate * active.sum(axis=0), 0.0
            )
            moved = True
        self.work["updates"] += 1
        self.work["update_rows"] += n
        if moved:
            self.install()
        return moved

    # -- reports

    def arrays(self) -> dict[str, np.ndarray]:
        """The tables by name, as a checkpoint holds them."""
        return {
            "dictionary": self.dictionary, "weights": self.weights, "thresholds": self.thresholds,
        }

    def parameters(self) -> int:
        return int(self.dictionary.size + self.weights.size + self.thresholds.size)

    def nbytes(self) -> int:
        return int(self.dictionary.nbytes + self.weights.nbytes + self.thresholds.nbytes)

    def genes(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.GENES}

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": self.inputs, "units": self.units, "context": self.context,
            "seed": self.seed, **self.genes(), "stepping": int(self.halvings),
            "work": dict(self.work),
        }
