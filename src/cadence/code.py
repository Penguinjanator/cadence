"""A learned code of meaning: a region of the brain's own neurons settling to a sparse code.

``Brain.compose(..., learned_code=units)`` places a ``code`` region of that many neurons
between the sensory region and the first module. Its synapses are the brain's synapses,
read in every sweep as every other synapse is, and the region settles with the rest of the
brain under the brain's neuron model:

* the dictionary ``Phi`` (``inputs`` by ``units``, unit-length columns) is the one-way
  synapses from the sensory neurons at the gain ``read = 1 / f(A)`` (``f`` the neuron's
  activation, ``A`` the model's stimulus amplitude): the region reads the sensory
  activation in units of one full input, ``x' = read * s_sensory``, so that one input at
  amplitude 1 drives the unit whose column matches it by one, and a code unit ``j`` is
  driven by ``Phi_j . x'``;
* the dictionary's Gram matrix is the lateral inhibition among the code units: every ordered
  pair ``j != k`` carries ``-(Phi_j . Phi_k)``, so units whose columns overlap compete;
* the predictor ``W`` (``units`` by the prefrontal width) is the one-way synapses from the
  prefrontal region, the working trace of the association activity of the moment before,
  scaled by ``top_down``: the context's prediction of the code, ``p = W s_prefrontal``;
* each unit's threshold is its bias, ``-t_j``;
* a standing boundary on the code units, the solver's quadratic anchor at gain ``top_down``
  toward zero, ``-top_down * s_j``, held through every phase: with the predictor's drive it
  is the pull ``top_down * (p_j - s_j)`` toward the prediction, the gradient of
  ``top_down / 2 * |s - p|^2``.

A code unit therefore settles to ``v_j = Phi_j . (x' - Phi s) + s_j + top_down (p_j -
s_j) - t_j`` with ``s_j = f(v_j)`` the neuron's rectified sigmoid: the sparse state that
reconstructs the sensory activation, as read, through the dictionary as far as the
thresholds allow,
drawn toward what the context predicts. No synapse leads back into the region, so its
fixed point depends on the observation and the trace alone; the brain settles it first,
with the same solver and tolerance, addresses its memories on it, and settles the whole
graph warm from it.

Learning is local to each boundary and happens at every accepted moment, each row of the
batch one event, from the moment's free settle alone: the lesson's nudged phases, the
outcome's eligibility phases, the imagined branches and the readings taken on copies enter
no table, as the contrast and the dopamine enter no synapse of the region
(``a = max(s, 0)``, the rectified code;
``x'`` the sensory activation as read; ``z`` the prefrontal activation; ``p = W z``):

* the dictionary, Hebbian on the bottom-up residual, ``Phi += rate * sum_rows (x' - Phi a)
  a^T``, each column then scaled back to unit length (a unit's own synaptic scaling);
* the lateral weights, recomputed from the dictionary;
* the predictor, the delta rule on the top-down residual normalized by the context's
  energy, ``W += sum_rows r_row (a - p) z^T`` with ``r_row = predictor / max(|z|^2,
  predictor^2)``: a row closes the share ``predictor`` of its own gap whatever the
  magnitude of its context (the prefrontal activation is small in the brain's units), a
  context fainter than ``|z| = predictor`` counted at that energy, so that no row commands a
  synaptic step above its gap;
* the thresholds, homeostatic, ``t += threshold_rate * sum_rows ((a_free > 0) - target)``
  with ``a_free`` the moment's context-free code (the region settled on the observation
  alone, the code ``Brain.observation_code`` gives): a unit's share is the share of the
  observations that drive it on their own, so what the context predicts may recruit a unit
  without raising its threshold, and the context-free code keeps the target share by
  construction; floored at zero, with a repair tied to the region's own disagreement: a
  row whose code leaves at least the share ``repair`` of its input's energy unexplained
  (``|x' - Phi a|^2
  >= repair * |x'|^2``; at the founder ``repair`` 1, a row whose code is silent or worse
  than nothing), in context or context-free (the place key), lowers the threshold of every
  unit silent in that row by ``threshold_rate * silence``, a silent unit in such a row
  counting as that many under-active rows, so no observation is left without a unit; a
  unit that fires, and a row the code reconstructs, are left to the share rule alone.
  There is no ceiling: a unit the world drives often has its threshold raised until its
  share is the target, and the rows it then leaves unexplained lower it again.

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
time (``Brain`` re-steps its graph when the level changes); the restoring term adds
``top_down`` times the slope to the coupling's diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._configuration import real
from .brain import Brain as NeuralGraph
from .brain import BrainState, Equilibrium, Nudge
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
    width; ``model`` the brain's neuron model, under which the region settles, and from
    which the read gain follows (``read``, one over the activation one input at amplitude
    1 gives a sensory neuron at rest). The genes are
    ``top_down`` (the prediction's weight on the units), ``rate`` (the dictionary's Hebbian
    rate per row), ``predictor`` (the share of a row's prediction gap closed per event),
    ``threshold`` (the initial threshold of every unit, a construction setting),
    ``threshold_rate`` and ``target`` (the share of rows a unit should be active in),
    ``repair`` (the share of a row's input energy its code may leave unexplained before the
    row lowers its silent units' thresholds) and ``silence`` (how many under-active rows a
    silent unit in such a row counts as), ``place`` (what the place stores key on).
    Validation refuses a value outside its domain before anything is built.
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
    repair: float = 1.0
    silence: float = 10.0
    place: str = "free"
    dictionary: np.ndarray = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)
    thresholds: np.ndarray = field(init=False, repr=False)
    read: float = field(init=False)

    GENES = (
        "top_down", "rate", "predictor", "threshold", "threshold_rate", "target", "repair",
        "silence", "place",
    )
    RUNTIME = (
        "top_down", "rate", "predictor", "threshold_rate", "target", "repair", "silence",
        "place",
    )
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
            "threshold_rate": 0.001, "target": 0.1, "repair": 1.0, "silence": 10.0,
            "place": "free",
        }

    @staticmethod
    def setting(name: str, value: Any) -> Any:
        """One gene, validated and converted."""
        if name in ("top_down", "rate", "threshold", "threshold_rate", "silence"):
            return real(name, value, low=0)
        if name == "predictor":
            return real(name, value, low=0, high=1)
        if name == "repair":
            number = real(name, value, low=0, high=1)
            if number <= 0:
                raise ValueError("repair must lie in (0, 1]")
            return number
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
        amplitude = np.array([self.model.stimulus_amplitude])
        full = float(np.asarray(self.model.activation(amplitude)).item())
        if not full > 0.0:
            raise ValueError(
                "a learned code needs a stimulus amplitude that drives a sensory neuron"
            )
        self.read = 1.0 / full
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
        over the rows of ``D^1/2 (L - top_down I) D^1/2`` (``D`` the activation's slopes at
        ``v``, ``L`` the lateral weights, ``top_down`` the standing boundary's gain); 0 when
        the sweep contracts at the model's own step."""
        v = np.atleast_2d(np.asarray(v, dtype=float))
        slopes = np.maximum(self.model.slope_at(v), 0.0)
        root = np.sqrt(slopes)
        lateral = self.lateral()
        coupling = root[:, :, None] * lateral[None, :, :] * root[:, None, :]
        # the standing boundary's restoring term, -top_down on every unit's own diagonal
        index = np.arange(self.units)
        coupling[:, index, index] -= self.top_down * slopes
        mu = float(np.linalg.eigvalsh(coupling)[:, 0].min())
        need = self.model.dt * (1.0 - mu) / self.STEP_MARGIN
        return 0 if need <= 1.0 else int(np.ceil(np.log2(need)))

    def standing(self) -> Nudge | None:
        """The region's standing boundary over its own units: the quadratic anchor toward
        zero at gain ``top_down`` on every unit, ``-top_down * s``; None at ``top_down`` 0."""
        if not self.top_down:
            return None
        zeros = np.zeros(self.units)
        return Nudge(
            zeros, zeros, 0.0, anchor=zeros, anchor_gain=np.full(self.units, self.top_down)
        )

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
        sensory activation at the region's gain plus ``top_down`` times the predictor's read
        of the prefrontal activation; without a context the prediction is zero."""
        drive = (self.read * np.asarray(sensory, dtype=float)) @ self.dictionary
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
            stimulus, state=warm, budget=budget, tolerance=tolerance, damping=damping,
            nudge=self.standing(),
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
        """Per row: the bottom-up residual's squared norm over the sensory activation's
        (both as read), and the top-down residual's squared norm (zero without a
        context)."""
        x = self.read * np.atleast_2d(np.asarray(sensory, dtype=float))
        a = np.maximum(np.atleast_2d(np.asarray(activation, dtype=float)), 0.0)
        rx = x - a @ self.dictionary.T
        energy = np.maximum(np.sum(x * x, axis=1), 1e-12)
        top = np.zeros(len(x)) if context is None else np.sum((a - self.prediction(context)) ** 2,
                                                              axis=1)
        return {"bottom_up": np.sum(rx * rx, axis=1) / energy, "top_down": top}

    # -- learning

    def learn(
        self, sensory: np.ndarray, activation: np.ndarray, context: np.ndarray | None,
        free: np.ndarray | None = None,
    ) -> bool:
        """One step of the dictionary, the predictor and the thresholds from a settled
        moment, each row an event; returns whether any table moved (every rate zero, or an
        empty batch, moves nothing). ``free`` is the moment's context-free code (the
        rectified activation of the region settled on the observation alone) where one was
        settled: the thresholds' share reads it, and the repair reads what it leaves
        unexplained beside the code in context; without it the code itself serves both.
        The sub-network takes the new tables."""
        x = self.read * np.atleast_2d(np.asarray(sensory, dtype=float))
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
            # the share of the row's gap, normalized by its context's energy, floored at
            # predictor^2; a silent context (z = 0) moves nothing
            rate = self.predictor / np.maximum(np.sum(z * z, axis=1), self.predictor**2)
            error = a - self.prediction(z)
            weights = self.weights + (error * rate[:, None]).T @ z
            if not np.isfinite(weights).all():
                raise ValueError("the learned code's predictor must remain finite")
            self.weights = weights
            moved = True
        if self.threshold_rate:
            own = a
            if free is not None:
                own = np.maximum(np.atleast_2d(np.asarray(free, dtype=float)), 0.0)
            # the share: the observations a unit fires for on their own, context-free
            active = (own > 0.0).astype(float) - self.target
            step = self.threshold_rate * active.sum(axis=0)
            if self.silence:
                # the repair: a row whose code leaves at least the share ``repair`` of its
                # input's energy unexplained, in context or context-free, lowers the
                # thresholds of the units silent in it
                step = step - self.threshold_rate * self.silence * self.unexplained(x, a)
                if free is not None:
                    step = step - self.threshold_rate * self.silence * self.unexplained(x, own)
            self.thresholds = np.maximum(self.thresholds + step, 0.0)
            moved = True
        self.work["updates"] += 1
        self.work["update_rows"] += n
        if moved:
            self.install()
        return moved

    def unexplained(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Per unit, how many rows of ``a`` (rectified codes of the read inputs ``x``)
        leave at least the share ``repair`` of their input's energy unexplained and have the
        unit silent; a row without input energy counts for no unit."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        a = np.atleast_2d(np.asarray(a, dtype=float))
        residual = x - a @ self.dictionary.T
        energy = np.sum(x * x, axis=1)
        left = np.sum(residual * residual, axis=1)
        repairing = (energy > 0.0) & (left >= self.repair * energy)
        return np.asarray((repairing[:, None] & ~(a > 0.0)).sum(axis=0), dtype=float)

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
