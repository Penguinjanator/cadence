"""The names of Cadence 0.8, kept for one release as deprecated aliases.

Code written against 0.8 keeps running: every old top-level name, module path, keyword
argument, attribute and method resolves to its biological counterpart and emits a
``DeprecationWarning`` naming the replacement. Checkpoints of the first format load through
``cadence.checkpoint.load`` without this module.

====================================  ==========================================
0.8                                   0.9
====================================  ==========================================
``Wiring``, ``from_edges``, ``sets``  ``Connectome``, ``from_synapses``, ``populations``
``Settlement``, ``SettledState``      ``Brain``, ``BrainState``
``GradedRule``, ``learning_rule``     ``NeuronModel``, ``learning_neuron_model``
``clamp``, ``clamp_amplitude``        ``stimulus``, ``stimulus_amplitude``
``edge_scale``                        ``efficacy``
``trainable_overlaps/owners``         ``plastic_synapses/neurons``
``symmetric``                         ``reciprocal``
``repair``                            ``activity_change``
``FastSeams``                         ``FastSynapses``
``Constitution``, ``grow``            ``Genome``, ``develop``
``cadence.brains``, ``couple``        ``cadence.circuits``, ``assemble``
``sensor_motor``                      ``reflex_arc``
====================================  ==========================================
"""

from __future__ import annotations

import functools
import importlib
import importlib.abc
import importlib.machinery
import sys
import types
import warnings
from collections.abc import Callable
from typing import Any

__all__ = ["OLD_NAMES", "install"]

# old top-level or module-level name -> (module, new name)
OLD_NAMES: dict[str, tuple[str, str]] = {
    "Wiring": ("cadence.connectome", "Connectome"),
    "Settlement": ("cadence.brain", "Brain"),
    "SettledState": ("cadence.brain", "BrainState"),
    "GradedRule": ("cadence.neuron", "NeuronModel"),
    "FastSeams": ("cadence.stream", "FastSynapses"),
    "Constitution": ("cadence.genome", "Genome"),
    "grow": ("cadence.genome", "develop"),
    "learning_rule": ("cadence.learning", "learning_neuron_model"),
    "settle_owner_by_owner": ("cadence.reference", "settle_neuron_by_neuron"),
    "couple": ("cadence.circuits.assembly", "assemble"),
    "sensor_motor": ("cadence.circuits", "reflex_arc"),
}

_MODULES = {
    "cadence.wiring": "cadence.connectome",
    "cadence.rules": "cadence.neuron",
    "cadence.settle": "cadence.brain",
    "cadence.constitution": "cadence.genome",
    "cadence.brains": "cadence.circuits",
    "cadence.brains.coupling": "cadence.circuits.assembly",
}

# (module, class or function, method or None) -> {old keyword: new keyword}
_KEYWORDS: dict[tuple[str, str, str | None], dict[str, str]] = {
    ("cadence.connectome", "Connectome", "__init__"): {"sets": "populations"},
    ("cadence.connectome", "Connectome", "with_populations"): {},
    ("cadence.brain", "Brain", "__init__"): {
        "wiring": "connectome",
        "rule": "neuron_model",
        "edge_scale": "efficacy",
    },
    ("cadence.brain", "Brain", "settle"): {"clamp": "stimulus"},
    ("cadence.brain", "Brain", "stimulus_vector"): {"clamp": "stimulus"},
    ("cadence.brain", "Brain", "with_parameters"): {"edge_scale": "efficacy"},
    ("cadence.brain", "BrainState", "__init__"): {"repair": "activity_change"},
    ("cadence.neuron", "NeuronModel", "__init__"): {"clamp_amplitude": "stimulus_amplitude"},
    ("cadence.neuron", "NeuronModel", "replace"): {"clamp_amplitude": "stimulus_amplitude"},
    ("cadence.learning", "Learner", "__init__"): {
        "engine": "brain",
        "trainable_overlaps": "plastic_synapses",
        "trainable_owners": "plastic_neurons",
        "symmetric": "reciprocal",
    },
    ("cadence.stream", "Trace", "__init__"): {"wiring": "connectome"},
    ("cadence.stream", "Echo", "__init__"): {"wiring": "connectome"},
    ("cadence.stream", "Afterglow", "__init__"): {"wiring": "connectome"},
    ("cadence.genome", "Projection", "__init__"): {"symmetric": "reciprocal"},
    ("cadence.protocol", "Protocol", "score"): {"engine": "brain"},
    ("cadence.protocol", "Protocol", "neurons_for"): {"wiring": "connectome"},
    ("cadence.circuits", "Readback", "__init__"): {"repair": "activity_change"},
    ("cadence.reference", "Ledger", "__init__"): {
        "declared_overlaps": "declared_synapses",
        "deliveries": "transmissions",
    },
    ("cadence.reference", "conformance", None): {"engine": "brain", "clamp": "stimulus"},
    ("cadence.reference", "settle_neuron_by_neuron", None): {
        "wiring": "connectome",
        "rule": "neuron_model",
        "clamp": "stimulus",
        "edge_scale": "efficacy",
    },
    ("cadence.genome", "develop", None): {"constitution": "genome"},
    ("cadence.genome", "mutate", None): {"constitution": "genome"},
    ("cadence.genome", "evolve", None): {"constitution": "genome"},
    ("cadence.learning", "learning_neuron_model", None): {
        "clamp_amplitude": "stimulus_amplitude"
    },
    ("cadence.protocol", "select_gain", None): {"make_engine": "make_brain"},
    ("cadence.protocol", "shuffled", None): {"wiring": "connectome"},
    ("cadence.blocks", "layout", None): {"wiring": "connectome"},
    ("cadence.sparse", "transport", None): {"wiring": "connectome"},
    ("cadence.circuits.assembly", "assemble", None): {"bridges": "synapses"},
}

# (module, class) -> {old attribute: new attribute}
_ATTRIBUTES: dict[tuple[str, str], dict[str, str]] = {
    ("cadence.connectome", "Connectome"): {"sets": "populations", "edges": "synapses"},
    ("cadence.brain", "Brain"): {
        "wiring": "connectome",
        "rule": "neuron_model",
        "edge_scale": "efficacy",
    },
    ("cadence.brain", "BrainState"): {"repair": "activity_change"},
    ("cadence.neuron", "NeuronModel"): {"clamp_amplitude": "stimulus_amplitude"},
    ("cadence.learning", "Learner"): {
        "engine": "brain",
        "trainable_overlaps": "plastic_synapses",
        "trainable_owners": "plastic_neurons",
        "symmetric": "reciprocal",
    },
    ("cadence.stream", "Trace"): {"wiring": "connectome"},
    ("cadence.genome", "Projection"): {"symmetric": "reciprocal"},
    ("cadence.circuits", "Readback"): {"repair": "activity_change"},
    ("cadence.reference", "Ledger"): {
        "declared_overlaps": "declared_synapses",
        "deliveries": "transmissions",
    },
}

# (module, class) -> {old method: new method}
_METHODS: dict[tuple[str, str], dict[str, str]] = {
    ("cadence.connectome", "Connectome"): {"with_sets": "with_populations"},
    ("cadence.brain", "Brain"): {
        "clamp_vector": "stimulus_vector",
        "clamp_levels": "stimulus_levels",
    },
    ("cadence.stream", "Trace"): {"clamp": "stimulate"},
    ("cadence.stream", "FastSynapses"): {"clamp": "stimulate"},
    ("cadence.protocol", "Protocol"): {"clamp_for": "neurons_for"},
    ("cadence.blocks", "BlockTransport"): {"inbox": "synaptic_input"},
}


def _warn(old: str, new: str) -> None:
    warnings.warn(f"cadence: {old} is deprecated; use {new}", DeprecationWarning, stacklevel=3)


def _renaming(func: Callable[..., Any], renames: dict[str, str], where: str) -> Any:
    if not renames:
        return func

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if kwargs:
            for old, new in renames.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"{where}: pass {new}= or {old}=, not both")
                    _warn(f"{where}({old}=)", f"{new}=")
                    kwargs[new] = kwargs.pop(old)
        return func(*args, **kwargs)

    wrapper.__cadence_renames__ = renames  # type: ignore[attr-defined]
    return wrapper


def _alias_property(old: str, new: str, owner: str) -> property:
    def get(self: Any) -> Any:
        _warn(f"{owner}.{old}", f"{owner}.{new}")
        return getattr(self, new)

    def put(self: Any, value: Any) -> None:
        _warn(f"{owner}.{old}", f"{owner}.{new}")
        setattr(self, new, value)

    return property(get, put)


def _alias_method(old: str, new: str, owner: str) -> Callable[..., Any]:
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        _warn(f"{owner}.{old}()", f"{owner}.{new}()")
        return getattr(self, new)(*args, **kwargs)

    method.__name__ = old
    return method


class _OldModule(types.ModuleType):
    """An old module path: the new module's names, plus the old names with a warning."""

    def __init__(self, name: str, target: str) -> None:
        super().__init__(name)
        self.__dict__["_target"] = importlib.import_module(target)
        if name == "cadence.brains":
            self.__dict__["__path__"] = []

    def __getattr__(self, name: str) -> Any:
        target = self.__dict__["_target"]
        if hasattr(target, name):
            return getattr(target, name)
        if name in OLD_NAMES:
            module, new = OLD_NAMES[name]
            _warn(f"{self.__name__}.{name}", f"{module}.{new}")
            return getattr(importlib.import_module(module), new)
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")


class _OldModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in _MODULES:
            return None
        return importlib.machinery.ModuleSpec(
            fullname, self, is_package=fullname == "cadence.brains"
        )

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        _warn(f"module {spec.name}", _MODULES[spec.name])
        return _OldModule(spec.name, _MODULES[spec.name])

    def exec_module(self, module: types.ModuleType) -> None:
        return None


def old_name(name: str) -> Any:
    """An old top-level name of ``cadence``, with a deprecation warning; ``AttributeError``
    for anything else."""
    if name in OLD_NAMES:
        module, new = OLD_NAMES[name]
        _warn(f"cadence.{name}", f"cadence.{new}")
        return getattr(importlib.import_module(module), new)
    if name in {"wiring", "rules", "settle", "constitution", "brains"}:
        return importlib.import_module(f"cadence.{name}")
    raise AttributeError(f"module 'cadence' has no attribute {name!r}")


_installed = False


def install() -> None:
    """Attach the deprecated keywords, attributes and methods, and the old module paths."""
    global _installed
    if _installed:
        return
    _installed = True
    package = sys.modules["cadence"]
    for (module_name, owner, method), renames in _KEYWORDS.items():
        module = importlib.import_module(module_name)
        obj = getattr(module, owner)
        if method is None:
            wrapped = _renaming(obj, renames, owner)
            for namespace in (module, package, sys.modules.get("cadence.circuits")):
                if namespace is not None and getattr(namespace, owner, None) is obj:
                    setattr(namespace, owner, wrapped)
            continue
        original = obj.__dict__.get(method)
        if original is None:
            continue
        setattr(obj, method, _renaming(original, renames, f"{owner}.{method}"))
    for (module_name, owner), attributes in _ATTRIBUTES.items():
        cls = getattr(importlib.import_module(module_name), owner)
        for old, new in attributes.items():
            setattr(cls, old, _alias_property(old, new, owner))
    for (module_name, owner), methods in _METHODS.items():
        cls = getattr(importlib.import_module(module_name), owner)
        for old, new in methods.items():
            setattr(cls, old, _alias_method(old, new, owner))
    connectome = importlib.import_module("cadence.connectome").Connectome
    sets_renames = {"sets": "populations"}
    from_synapses = _renaming(connectome.from_synapses.__func__, sets_renames, "from_synapses")
    connectome.from_synapses = classmethod(from_synapses)

    def from_edges(cls: Any, *args: Any, **kwargs: Any) -> Any:
        _warn("Connectome.from_edges()", "Connectome.from_synapses()")
        return cls.from_synapses(*args, **kwargs)

    connectome.from_edges = classmethod(from_edges)
    sys.meta_path.insert(0, _OldModuleFinder())
