"""Optional task compositions: motor wiring, bounded futures and self-reading control.

These use the existing graded owner rule. They are engineered patterns, not
claims of biological universality, autonomous world-model learning or consciousness.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Generic, TypeVar

import numpy as np

from ..rules import GradedRule
from ..settle import SettledState, Settlement
from ..wiring import Wiring
from .coupling import couple

S = TypeVar("S")
A = TypeVar("A")


def sensor_motor(axes: int = 2) -> Wiring:
    """Sensory error ports followed by opposing motor pairs for each axis.

    A body reads ``max(0, positive) - max(0, negative)``. Body physics and sensor
    encodings are supplied by the application. Continue state to retain dynamics.
    """
    if isinstance(axes, bool) or not isinstance(axes, int) or axes < 1:
        raise ValueError("axes must be a positive integer")
    return Wiring.from_edges(
        3 * axes,
        pre=[i for i in range(axes) for _ in range(2)],
        post=[axes + i for i in range(2 * axes)],
        sign=[v for _ in range(axes) for v in (2.0, -2.0)],
        sets={
            "sensory": range(axes),
            "motor": range(axes, 3 * axes),
            "positive": range(axes, 3 * axes, 2),
            "negative": range(axes + 1, 3 * axes, 2),
        },
        label="sensory-motor",
    )


@dataclass(frozen=True)
class Future(Generic[S, A]):
    action: A
    score: float
    sequence: tuple[A, ...]
    state: S


@dataclass(frozen=True)
class Deliberation(Generic[S, A]):
    futures: tuple[Future[S, A], ...]
    nodes: int
    depth: int


def imagine(
    live: S,
    actions: Callable[[S], Sequence[A]],
    transition: Callable[[S, A], S],
    evaluate: Callable[[S], float],
    terminal: Callable[[S], bool],
    *,
    depth: int = 2,
    max_nodes: int = 10000,
    adversarial: bool = False,
) -> Deliberation[S, A]:
    """Compare isolated futures; alternate max/min when ``adversarial=True``.

    Values are always from the root actor's perspective. The transition receives
    its own deep copy. Evaluators/actions must be read-only and must not mutate
    external objects. No observations or training are fabricated. A hard budget
    failure raises rather than silently returning partially searched moves.
    """
    if (
        isinstance(depth, bool)
        or not isinstance(depth, int)
        or depth < 1
        or isinstance(max_nodes, bool)
        or not isinstance(max_nodes, int)
        or max_nodes < 1
    ):
        raise ValueError("depth and max_nodes must be positive integers")
    nodes = 0

    def visit(state: S, remaining: int, maximize: bool) -> tuple[float, tuple[A, ...], S]:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise ValueError("imagination node budget exceeded")
        choices = () if remaining == 0 or terminal(state) else actions(state)
        if not choices:
            value = float(evaluate(state))
            if not isfinite(value):
                raise ValueError("evaluator returned a nonfinite value")
            return value, (), state
        candidates = []
        for action in choices:
            after = transition(deepcopy(state), action)
            score, suffix, end = visit(after, remaining - 1, not maximize if adversarial else True)
            candidates.append((score, (action, *suffix), end))
        return (max if maximize else min)(candidates, key=lambda row: row[0])

    initial = deepcopy(live)
    if terminal(initial):
        return Deliberation((), 0, depth)
    futures = []
    for action in actions(initial):
        score, sequence, end = visit(
            transition(deepcopy(initial), action), depth - 1, not adversarial
        )
        futures.append(Future(action, score, (action, *sequence), end))
    return Deliberation(tuple(sorted(futures, key=lambda f: -f.score)), nodes, depth)


@dataclass(frozen=True)
class Readback:
    repair: float
    ambiguity: float
    pressure: float
    uncertainty: float
    request_more: bool
    state: SettledState


class ActivityMonitor:
    """Read the controller's activity and alternatives, then gate more deliberation.

    This six-owner monitor observes normalized activity change, option ambiguity,
    budget pressure and their integration. Its readout can request more work;
    an application must connect that output to its actual compute budget. The
    signal is a heuristic, not a calibrated probability or a consciousness test.
    """

    def __init__(self) -> None:
        self.engine = Settlement(
            Wiring.from_edges(
                6,
                pre=[0, 1, 2, 3, 4, 2],
                post=[4, 4, 4, 4, 5, 5],
                sign=[0.6, 0.8, -0.4, 0.4, 1.0, -1.0],
                sets={"readback": range(4), "integration": [4], "request": [5]},
                label="activity-monitor",
            ),
            GradedRule(gain=1, slope=2, threshold=0, leak=1, dt=0.25, clamp_amplitude=1),
        )
        self.state: SettledState | None = None
        self.previous: np.ndarray | None = None

    def read(self, activity: np.ndarray, scores: np.ndarray, *, pressure: float = 0) -> Readback:
        activity, scores = np.asarray(activity, float), np.asarray(scores, float)
        if (
            activity.ndim != 1
            or not activity.size
            or scores.ndim != 1
            or not scores.size
            or not np.isfinite(activity).all()
            or not np.isfinite(scores).all()
            or not isfinite(pressure)
            or not 0 <= pressure <= 1
        ):
            raise ValueError("finite vectors and pressure in [0, 1] required")
        repair = (
            0.0
            if self.previous is None or self.previous.shape != activity.shape
            else float(
                np.max(np.abs(activity - self.previous)) / max(1.0, float(np.max(np.abs(activity))))
            )
        )
        ordered = np.sort(scores)
        gap = float(ordered[-1] - ordered[-2]) if scores.size > 1 else float("inf")
        ambiguity = 1.0 / (1.0 + gap)
        drive = np.array([min(1, repair), ambiguity, pressure, float(scores.size > 1), 0, 0])
        self.state = self.engine.settle(drive, state=self.state, steps=40, tolerance=0)
        self.previous = activity.copy()
        uncertainty = float(self.state.activation[4])
        return Readback(
            repair,
            ambiguity,
            pressure,
            uncertainty,
            bool(self.state.activation[5] > 0.35 and pressure < 1),
            self.state,
        )


__all__ = [
    "ActivityMonitor",
    "Deliberation",
    "Future",
    "Readback",
    "couple",
    "imagine",
    "sensor_motor",
]
