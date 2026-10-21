"""Experimental brains built from local state, memory and repair.

Use ``Brain.compose`` for the default System 1 brain, adding ``observers`` for
optional System 2 feedback in the same neural graph. ``NeuralGraph`` exposes
lower-level neuronal dynamics. Temporal models provide planning and response
protection with their own contracts. See ``docs/contracts.md`` for their numerical
and learning boundaries.
"""

from __future__ import annotations

from . import regions
from .arousal import Arousal, ArousalConfig
from .brain import Brain as NeuralGraph
from .brain import BrainState, Equilibrium, Nudge, RefinementReport, available_backends
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
from .generic import Brain
from .genome import Genome, Projection, develop, evolve, genes
from .learning import (
    GraphHolder,
    Learner,
    LearnerConfig,
    LearningPhaseError,
    calibrate_bias,
    embedded,
    layered,
    learning_neuron_model,
    naive_efficacy,
    preflight,
    seam_report,
)
from .memory import SynapticMemory
from .neuron import Adaptation, NeuronModel
from .patch import PatchNet, PatchObservation
from .planning import TemporalPlan
from .plasticity import (
    ActorCritic,
    ActorCriticConfig,
    Bins,
    Valence,
)
from .protocol import Protocol, Row, evaluate_predicate, select_gain, shuffled
from .receipts import Receipt, canonical_json
from .recording import SettlementRecord, record_settlements
from .reference import conformance
from .regions import Region
from .stream import (
    Afterglow,
    Echo,
    Efference,
    FastSynapses,
    History,
    PatternSeparator,
    Recall,
    Trace,
    stateful,
)
from .temporal import TemporalObservation, TemporalPatchNet, TemporalPhase, TemporalReadback
from .temporal_memory import ConstraintReport, TemporalMemory

__all__ = [
    "Arousal",
    "ArousalConfig",
    "ConstraintReport",
    "TemporalMemory",
    "TemporalPlan",
    "TemporalPatchNet",
    "TemporalPhase",
    "TemporalObservation",
    "TemporalReadback",
    "PatchNet",
    "PatchObservation",
    "Certificate",
    "EPStructure",
    "certificate",
    "ep_structure",
    "lipschitz_constant",
    "row_mass",
    "PatternSeparator",
    "ActorCritic",
    "ActorCriticConfig",
    "Valence",
    "Bins",
    "Adaptation",
    "Region",
    "Brain",
    "regions",
    "Projection",
    "Genome",
    "Echo",
    "Afterglow",
    "Trace",
    "Efference",
    "History",
    "Recall",
    "FastSynapses",
    "SynapticMemory",
    "NeuronModel",
    "GraphHolder",
    "Learner",
    "LearnerConfig",
    "LearningPhaseError",
    "Nudge",
    "Protocol",
    "Receipt",
    "SettlementRecord",
    "record_settlements",
    "Row",
    "NeuralGraph",
    "BrainState",
    "Equilibrium",
    "RefinementReport",
    "Connectome",
    "available_backends",
    "canonical_json",
    "conformance",
    "embedded",
    "develop",
    "evolve",
    "genes",
    "evaluate_predicate",
    "layered",
    "learning_neuron_model",
    "calibrate_bias",
    "naive_efficacy",
    "preflight",
    "seam_report",
    "load",
    "save",
    "select_gain",
    "shuffled",
    "stateful",
]

__version__ = "0.84.3"
