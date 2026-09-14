"""Cadence: brains that learn by settling.

A brain is a network of neurons joined by declared synapses and organised into
regions. Each neuron holds a membrane potential and publishes an activation; on
every step it relaxes toward its synaptic input, stimulus and bias, and the whole
brain settles toward one equilibrium. Traces and fast synapses keep observations
between settling runs, and dopamine carries reward prediction errors to plastic
synapses. Normalization and output softmax read their declared populations;
conformance checks the synaptic transport rather than every auxiliary operation.
Cadence gives you the connectome, the neuron model, the brain on CPU or an
accelerator, a neuron-by-neuron reference with a transmission ledger to check the
accelerated one against, a protocol layer for declared held-out tests with a
shuffled control, and receipts that bind every result to the code and data that
produced it.

    >>> import cadence as cd
    >>> ring = dict(pre=[0, 1, 2, 3], post=[1, 2, 3, 0], count=[120] * 4)
    >>> brain = cd.Brain(cd.Connectome.from_synapses(4, **ring), cd.NeuronModel(gain=0.03))
    >>> brain.settle(stimulus={0: 3.0}, steps=60).activation.round(2)
    array([1., 1., 1., 1.])
"""

from __future__ import annotations

from . import regions
from .brain import Brain, BrainState, Equilibrium, Nudge, available_backends
from .checkpoint import load, save
from .connectome import Connectome
from .custody import Source, fetch, manifest
from .generic import GenericBrain
from .genome import Genome, Projection, develop, evolve
from .learning import Learner, LearnerConfig, embedded, layered, learning_neuron_model
from .neuron import Adaptation, NeuronModel
from .plasticity import (
    ActorCritic,
    ActorCriticConfig,
    Bins,
    Valence,
)
from .protocol import Protocol, Row, evaluate_predicate, select_gain, shuffled
from .receipts import Receipt, canonical_json
from .reference import conformance
from .regions import Region
from .stream import Afterglow, Echo, FastSynapses, Trace, stateful

__all__ = [
    "ActorCritic",
    "ActorCriticConfig",
    "Valence",
    "Bins",
    "Adaptation",
    "Region",
    "GenericBrain",
    "regions",
    "Projection",
    "Genome",
    "Echo",
    "Afterglow",
    "Trace",
    "FastSynapses",
    "NeuronModel",
    "Learner",
    "LearnerConfig",
    "Nudge",
    "Protocol",
    "Receipt",
    "Row",
    "Brain",
    "BrainState",
    "Equilibrium",
    "Source",
    "Connectome",
    "available_backends",
    "canonical_json",
    "conformance",
    "embedded",
    "develop",
    "evolve",
    "evaluate_predicate",
    "fetch",
    "layered",
    "learning_neuron_model",
    "load",
    "manifest",
    "save",
    "select_gain",
    "shuffled",
    "stateful",
]

__version__ = "0.9.0"

from . import legacy as _legacy  # noqa: E402  (the 0.8 names, deprecated)

_legacy.install()


def __getattr__(name: str) -> object:
    return _legacy.old_name(name)
