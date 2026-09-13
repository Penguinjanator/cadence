"""Cadence: machine learning by patch-net settlement.

A patch net is a set of owners, each holding one patch of state, joined by
declared overlaps. During settlement every owner repairs its own patch from
its inbox. Explicit memory patches retain observations between settlements.
Normalization and output softmax read their declared groups; conformance
checks the settlement transport rather than every auxiliary operation. Cadence gives
you the wiring, the owner rule,
the settlement engine on CPU or an accelerator, an owner-by-owner
reference engine with a message ledger to check the accelerated one
against, a protocol layer for declared held-out tests with a shuffled
control, and receipts that bind every result to the code and data that
produced it.

    >>> import cadence as cd
    >>> wiring = cd.Wiring.from_edges(4, pre=[0, 1, 2, 3], post=[1, 2, 3, 0], count=[120] * 4)
    >>> engine = cd.Settlement(wiring, cd.GradedRule(gain=0.03))
    >>> engine.settle(clamp={0: 3.0}, steps=60).activation.round(2)
    array([1., 1., 1., 1.])
"""

from __future__ import annotations

from .checkpoint import load, save
from .constitution import Constitution, Projection, Region, evolve, grow
from .custody import Source, fetch, manifest
from .learning import Learner, LearnerConfig, embedded, layered, learning_rule
from .plasticity import (
    ActorCritic,
    ActorCriticConfig,
    Bins,
    Valence,
)
from .protocol import Protocol, Row, evaluate_predicate, select_gain, shuffled
from .receipts import Receipt, canonical_json
from .reference import conformance
from .rules import Adaptation, GradedRule
from .settle import Nudge, SettledState, Settlement, available_backends
from .stream import Afterglow, Echo, FastSeams, Trace, stateful
from .wiring import Wiring

__all__ = [
    "ActorCritic",
    "ActorCriticConfig",
    "Valence",
    "Bins",
    "Adaptation",
    "Region",
    "Projection",
    "Constitution",
    "Echo",
    "Afterglow",
    "Trace",
    "FastSeams",
    "GradedRule",
    "Learner",
    "LearnerConfig",
    "Nudge",
    "Protocol",
    "Receipt",
    "Row",
    "Settlement",
    "SettledState",
    "Source",
    "Wiring",
    "available_backends",
    "canonical_json",
    "conformance",
    "embedded",
    "grow",
    "evolve",
    "evaluate_predicate",
    "fetch",
    "layered",
    "learning_rule",
    "load",
    "manifest",
    "save",
    "select_gain",
    "shuffled",
    "stateful",
]

__version__ = "0.8.1"
