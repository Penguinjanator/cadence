"""Cadence: stateful neural systems that learn through ongoing experience.

Neural patches carry local state, exchange activity over declared synapses and
read back their responses. Memories retain observed associations; local updates
and reward eligibility repair future predictions and decisions. ``GenericBrain.step``
coordinates observation, feedback and action in one ongoing loop, without separate
training/inference modes. The application supplies its environment and update clocks;
settling or imagination alone does not teach synapses.

Raw ``Brain`` dynamics seek a fixed point or follow a transient; convergence must
be checked. Normalization and output softmax read declared groups. Protocols,
conformance and source-bound receipts check the stated numerical claims, not
human-like capability.

    >>> import cadence as cd
    >>> ring = dict(pre=[0, 1, 2, 3], post=[1, 2, 3, 0], count=[120] * 4)
    >>> brain = cd.Brain(cd.Connectome.from_synapses(4, **ring), cd.NeuronModel(gain=0.03))
    >>> brain.settle(stimulus={0: 3.0}, steps=60).activation.round(2)
    array([1., 1., 1., 1.])
"""

from __future__ import annotations

from . import regions
from .atlas import Atlas, atlas_of, brain_scan_script, build_atlas
from .brain import Brain, BrainState, Equilibrium, Nudge, available_backends
from .certificate import (
    Certificate,
    EPStructure,
    certificate,
    ep_structure,
    lipschitz_constant,
    row_mass,
)
from .checkpoint import load, save
from .connectome import Connectome
from .content_memory import ContentMemory
from .custody import Source, fetch, manifest
from .generic import GenericBrain
from .genome import Genome, Projection, develop, evolve
from .learning import Learner, LearnerConfig, embedded, layered, learning_neuron_model
from .memory import SynapticMemory
from .neuron import Adaptation, NeuronModel
from .plasticity import (
    ActorCritic,
    ActorCriticConfig,
    Bins,
    Valence,
)
from .protocol import Protocol, Row, evaluate_predicate, select_gain, shuffled
from .receipts import Receipt, canonical_json
from .recording import SettlementRecord, record_settlements
from .records import Mulberry32, Records
from .reference import conformance
from .regions import Region
from .replay import ReservoirReplay
from .stream import Afterglow, Echo, FastSynapses, PatternSeparator, Trace, stateful

__all__ = [
    "ContentMemory",
    "ReservoirReplay",
    "Atlas",
    "atlas_of",
    "build_atlas",
    "brain_scan_script",
    "Certificate",
    "EPStructure",
    "certificate",
    "ep_structure",
    "lipschitz_constant",
    "row_mass",
    "PatternSeparator",
    "Records",
    "Mulberry32",
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
    "SynapticMemory",
    "NeuronModel",
    "Learner",
    "LearnerConfig",
    "Nudge",
    "Protocol",
    "Receipt",
    "SettlementRecord",
    "record_settlements",
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
