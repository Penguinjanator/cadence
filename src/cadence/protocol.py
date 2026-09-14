"""Declared tests: named populations, stimuli, rows with predicates, and a shuffled control.

A protocol is data. It says which neurons a stimulus drives, which neurons a
readout reads, which few facts a model may be shown, and which facts are
held out and scored. Predicates carry their preconditions, so a dead net
cannot pass "no response after ablation" vacuously. The control is the
same protocol on a connectome whose postsynaptic endpoints were permuted with
every count and sign kept: if the connectome predicts the held-out facts and
the permuted connectome does not, the prediction came from the connectome.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .brain import Brain
from .connectome import Connectome

__all__ = ["Row", "Protocol", "evaluate_predicate", "shuffled", "select_gain", "PREDICATES"]

PREDICATES: dict[str, str] = {
    "active": "mean(readout) >= active_level",
    "inactive": "mean(readout) <= inactive_level",
    "reduced": "reference active; mean(readout) <= mean(reference) - margin",
    "retained": "reference active; mean(readout) >= active_level",
    "released": "reference inactive; mean(readout) >= mean(reference) + margin",
    "exceeds": "mean(readout) >= active_level and mean(readout) >= mean(other) + margin",
    "lateralized": (
        "max(readout, other) >= inactive_level and |mean(readout) - mean(other)| >= margin"
    ),
    "sparse": "fraction >= sparse_min; fraction(readout) <= sparse_max",
    "densified": "reference sparse; fraction(readout) >= fraction(reference) + densify_margin",
}


@dataclass(frozen=True, slots=True)
class Levels:
    active: float = 0.5
    inactive: float = 0.2
    margin: float = 0.15
    sparse_min: float = 0.005
    sparse_max: float = 0.2
    densify_margin: float = 0.05


def evaluate_predicate(
    predicate: str,
    value: Mapping[str, float],
    reference: Mapping[str, float],
    levels: Levels | None = None,
) -> bool:
    """``value`` is the row's reading, ``reference`` its comparison; both are {mean, fraction}."""
    L = levels if levels is not None else Levels()
    if predicate == "active":
        return value["mean"] >= L.active
    if predicate == "inactive":
        return value["mean"] <= L.inactive
    if predicate == "reduced":
        return reference["mean"] >= L.active and value["mean"] <= reference["mean"] - L.margin
    if predicate == "retained":
        return reference["mean"] >= L.active and value["mean"] >= L.active
    if predicate == "released":
        return reference["mean"] <= L.inactive and value["mean"] >= reference["mean"] + L.margin
    if predicate == "exceeds":
        return value["mean"] >= L.active and value["mean"] >= reference["mean"] + L.margin
    if predicate == "lateralized":
        return (
            max(value["mean"], reference["mean"]) >= L.inactive
            and abs(value["mean"] - reference["mean"]) >= L.margin
        )
    if predicate == "sparse":
        return L.sparse_min <= value["fraction"] <= L.sparse_max
    if predicate == "densified":
        return (L.sparse_min <= reference["fraction"] <= L.sparse_max) and value[
            "fraction"
        ] >= reference["fraction"] + L.densify_margin
    raise KeyError(predicate)


@dataclass(frozen=True, slots=True)
class Row:
    """One held-out fact: under ``stimulus`` less ``ablate``, ``readout`` meets ``predicate``."""

    id: str
    stimulus: str
    readout: str
    predicate: str
    reference: str = ""
    ablate: tuple[str, ...] = ()
    relative_to: str = (
        ""  # a stimulus (for reduced/released) or a readout (for exceeds/lateralized)
    )
    tier: str = "experiment"


@dataclass
class Protocol:
    stimuli: dict[str, tuple[str, ...]]  # stimulus name -> set names stimulated together
    rows: Sequence[Row]
    training: Sequence[
        tuple[str, str, str]
    ] = ()  # (stimulus, readout, predicate) the model may see
    levels: Levels = field(default_factory=Levels)
    steps: int = 60

    def neurons_for(self, connectome: Connectome, stimulus: str) -> tuple[int, ...]:
        return connectome.members(*self.stimuli[stimulus])

    def score(self, brain: Brain) -> dict[str, Any]:
        """Settle every needed (stimulus, ablation) pair once and score training facts and rows."""
        connectome = brain.connectome
        cache: dict[tuple[str, tuple[str, ...]], dict[str, dict[str, float]]] = {}
        readouts = sorted(
            {r.readout for r in self.rows}
            | {t[1] for t in self.training}
            | {r.relative_to for r in self.rows if r.relative_to in connectome.populations}
        )

        def readings(stimulus: str, ablate: tuple[str, ...] = ()) -> dict[str, dict[str, float]]:
            key = (stimulus, ablate)
            if key not in cache:
                mask = np.ones(connectome.n)
                if ablate:
                    mask[list(connectome.members(*ablate))] = 0.0
                state = brain.settle(
                    {i: 1.0 for i in self.neurons_for(connectome, stimulus)},
                    steps=self.steps,
                    mask=mask,
                )
                out = brain.readings(state, readouts)
                out["_all"] = {
                    "mean": float(state.activation.mean()),
                    "fraction": state.fraction_active(),
                }
                cache[key] = out
            return cache[key]

        training = []
        for stimulus, readout, predicate in self.training:
            value = readings(stimulus)[readout]
            training.append(
                {
                    "stimulus": stimulus,
                    "readout": readout,
                    "predicate": predicate,
                    "reading": value,
                    "passed": evaluate_predicate(predicate, value, value, self.levels),
                }
            )
        rows: list[dict[str, Any]] = []
        for row in self.rows:
            value = readings(row.stimulus, row.ablate)[row.readout]
            relative = row.relative_to in connectome.populations
            if row.predicate in ("exceeds", "lateralized") and relative:
                reference = readings(row.stimulus, row.ablate)[row.relative_to]
            elif row.relative_to in self.stimuli:
                reference = readings(row.relative_to)[row.readout]
            elif row.relative_to in connectome.populations:
                reference = readings(row.stimulus, row.ablate)[row.relative_to]
            elif row.relative_to:
                raise KeyError(f"unknown relative_to: {row.relative_to!r}")
            else:
                reference = readings(row.stimulus)[row.readout]
            rows.append(
                {
                    "id": row.id,
                    "stimulus": row.stimulus,
                    "ablate": list(row.ablate),
                    "readout": row.readout,
                    "predicate": row.predicate,
                    "tier": row.tier,
                    "reading": value,
                    "reference": reference,
                    "passed": evaluate_predicate(row.predicate, value, reference, self.levels),
                }
            )
        return {
            "training": training,
            "training_passed": sum(1 for t in training if t["passed"]),
            "rows": rows,
            "passed": sum(1 for r in rows if r["passed"]),
            "total": len(rows),
            "passed_by_tier": {
                tier: sum(1 for r in rows if r["passed"] and r["tier"] == tier)
                for tier in sorted({str(r["tier"]) for r in rows})
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "stimuli": {k: list(v) for k, v in self.stimuli.items()},
            "training": [list(t) for t in self.training],
            "rows": [
                {
                    "id": r.id,
                    "stimulus": r.stimulus,
                    "readout": r.readout,
                    "predicate": r.predicate,
                    "reference": r.reference,
                    "ablate": list(r.ablate),
                    "relative_to": r.relative_to,
                    "tier": r.tier,
                }
                for r in self.rows
            ],
            "levels": asdict(self.levels),
            "steps": self.steps,
            "predicates": PREDICATES,
        }


def shuffled(connectome: Connectome, seed: int, *, keep: np.ndarray | None = None) -> Connectome:
    """Permute postsynaptic endpoints; counts, signs, out-degrees, and named populations are kept.

    ``keep`` is a boolean mask over synapses that are left untouched, for
    example the bridges between two joined datasets.
    """
    rng = np.random.default_rng(seed)
    post = connectome.post.copy()
    movable = np.ones(connectome.synapses, dtype=bool) if keep is None else ~np.asarray(keep, bool)
    if movable.shape != (connectome.synapses,):
        raise ValueError("keep must have one entry per synapse")
    rows = np.flatnonzero(movable)
    post[rows] = post[rows][rng.permutation(len(rows))]
    # A permutation can land a synapse on its own neuron; swap those endpoints with random
    # movable rows until none is left, which keeps the multiset of endpoints intact.
    for _ in range(100):
        clash = rows[post[rows] == connectome.pre[rows]]
        if len(clash) == 0:
            break
        partner = rng.choice(rows, size=len(clash))
        # Index lists may overlap or repeat partners. Sequential swaps preserve the
        # endpoint multiset in those cases; simultaneous advanced assignment does not.
        for a, b in zip(clash, partner, strict=True):
            post[a], post[b] = post[b], post[a]
    else:
        raise ValueError("could not shuffle without autapses")
    return Connectome(
        connectome.n,
        connectome.pre,
        post,
        connectome.count,
        connectome.sign,
        connectome.populations,
        f"{connectome.label}:shuffled:{seed}",
    )


def select_gain(
    make_brain: Callable[[float], Brain],
    protocol: Protocol,
    grid: Sequence[float],
    *,
    sparsity_cap: float | None = 0.05,
) -> tuple[float, list[dict[str, Any]]]:
    """One global gain from the training facts alone: most facts passed, smallest gain on ties.

    A gain is admissible only while the net stays sparse under every
    training stimulus (at most ``sparsity_cap`` of neurons active). Runaway
    activity lights every readout and is not a fact about the connectome.
    Raises ``ValueError`` for an empty grid or when no gain is admissible.
    """
    if len(grid) == 0:
        raise ValueError("gain grid must not be empty")
    table = []
    best: tuple[int, float] | None = None
    for gain in grid:
        brain = make_brain(gain)
        connectome = brain.connectome
        passed = 0
        fraction = 0.0
        detail = {}
        for stimulus, readout, predicate in protocol.training:
            neurons = list(protocol.neurons_for(connectome, stimulus))
            state = brain.settle({i: 1.0 for i in neurons}, steps=protocol.steps)
            value = {
                "mean": state.mean(connectome.populations[readout]),
                "fraction": state.fraction_active(connectome.populations[readout]),
            }
            passed += int(evaluate_predicate(predicate, value, value, protocol.levels))
            fraction = max(fraction, state.fraction_active())
            detail[f"{stimulus}->{readout}"] = round(value["mean"], 4)
        admissible = sparsity_cap is None or fraction <= sparsity_cap
        table.append(
            {
                "gain": gain,
                "facts_passed": passed,
                "fraction_active": fraction,
                "admissible": admissible,
                "readings": detail,
            }
        )
        if admissible and (
            best is None or passed > best[0] or (passed == best[0] and gain < best[1])
        ):
            best = (passed, gain)
    if best is None:
        raise ValueError("no gain satisfies the sparsity cap")
    return best[1], table
