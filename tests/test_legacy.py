"""The 0.8 names resolve to their biological counterparts, each with a deprecation warning."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import cadence as cd


def ring() -> cd.Connectome:
    return cd.Connectome.from_synapses(
        4, pre=[0, 1, 2, 3], post=[1, 2, 3, 0], count=[120] * 4, populations={"a": [0]}
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("Wiring", "Connectome"),
        ("Settlement", "Brain"),
        ("SettledState", "BrainState"),
        ("GradedRule", "NeuronModel"),
        ("FastSeams", "FastSynapses"),
        ("Constitution", "Genome"),
        ("grow", "develop"),
        ("learning_rule", "learning_neuron_model"),
    ],
)
def test_old_top_level_names_are_the_new_objects(old: str, new: str) -> None:
    with pytest.warns(DeprecationWarning, match=new):
        assert getattr(cd, old) is getattr(cd, new)
    assert old not in cd.__all__
    with pytest.raises(AttributeError):
        cd.NoSuchName  # noqa: B018


def test_an_old_script_settles_to_the_same_numbers() -> None:
    with pytest.warns(DeprecationWarning):
        w = cd.Wiring.from_edges(
            4, pre=[0, 1, 2, 3], post=[1, 2, 3, 0], count=[120] * 4, sets={"a": [0]}
        )
        engine = cd.Settlement(
            wiring=w, rule=cd.GradedRule(gain=0.03, clamp_amplitude=3.0), edge_scale=w.sign
        )
        old = engine.settle(clamp={0: 1.0}, steps=60)
        assert w.sets == w.populations and w.edges == w.synapses == 4
        assert engine.wiring is w and engine.rule.clamp_amplitude == 3.0
        assert engine.edge_scale is engine.efficacy
        np.testing.assert_array_equal(old.repair, old.activity_change)
        np.testing.assert_array_equal(engine.clamp_vector([0]), engine.stimulus_vector([0]))
    new = cd.Brain(ring(), cd.NeuronModel(gain=0.03)).settle(stimulus={0: 1.0}, steps=60)
    np.testing.assert_array_equal(old.activation, new.activation)


def test_old_and_new_keyword_together_is_an_error() -> None:
    with pytest.raises(TypeError, match="not both"):
        cd.Brain(ring(), cd.NeuronModel()).settle(stimulus=[0], clamp=[0])


def test_old_module_paths_import_the_new_objects() -> None:
    for name in ("cadence.settle", "cadence.brains", "cadence.brains.coupling"):
        sys.modules.pop(name, None)
    with pytest.warns(DeprecationWarning, match="cadence.brain"):
        settle = importlib.import_module("cadence.settle")
    assert settle.Settlement is cd.Brain and settle.Nudge is cd.Nudge
    with pytest.warns(DeprecationWarning):
        coupling = importlib.import_module("cadence.brains.coupling")
        brains = importlib.import_module("cadence.brains")
        assert brains.sensor_motor is importlib.import_module("cadence.circuits").reflex_arc
    with pytest.warns(DeprecationWarning, match="synapses="):
        joined = coupling.couple({"x": ring(), "y": ring()}, bridges=[("x", 0, "y", 1, 0.5)])
    assert joined.synapses == 9


def test_learner_trace_and_projection_keywords_and_attributes() -> None:
    connectome = cd.layered(3, 4, 2, density=1.0, seed=0)
    brain = cd.Brain(connectome, cd.learning_neuron_model())
    with pytest.warns(DeprecationWarning):
        learner = cd.Learner(
            engine=brain,
            outputs=connectome.populations["output"],
            trainable_overlaps=np.ones(connectome.synapses, bool),
            symmetric=False,
        )
        assert learner.engine is learner.brain and learner.symmetric is False
        learner.symmetric = True
        assert learner.trainable_overlaps is learner.plastic_synapses
        projection = cd.Projection("a", "b", symmetric=False)
        assert projection.symmetric is False and projection.reciprocal is False
    assert learner.reciprocal is True


def test_a_first_format_checkpoint_loads(tmp_path: Path) -> None:
    connectome = cd.layered(3, 4, 2, density=1.0, seed=0)
    brain = cd.Brain(connectome, cd.learning_neuron_model())
    learner = cd.Learner(brain, connectome.populations["output"])
    learner.step(np.eye(3)[[0, 1]] * 0.5 @ np.eye(3, connectome.n), np.array([0, 1]))
    path = cd.save(learner, tmp_path / "new.npz")
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(str(arrays.pop("meta")))
    meta["format"] = "cadence-checkpoint/1"
    meta["sets"] = meta.pop("populations")
    meta["symmetric"] = meta.pop("reciprocal")
    model = meta.pop("neuron_model")
    model["clamp_amplitude"] = model.pop("stimulus_amplitude")
    meta["rule"] = model
    for old, new in (
        ("edge_scale", "efficacy"),
        ("trainable_overlaps", "plastic_synapses"),
        ("trainable_owners", "plastic_neurons"),
    ):
        arrays[old] = arrays.pop(new)
    first = tmp_path / "first.npz"
    np.savez_compressed(first, meta=np.array(json.dumps(meta)), **arrays)
    back = cd.load(first)
    drive = np.eye(3, connectome.n) * 0.5
    np.testing.assert_array_equal(back.predict(drive), learner.predict(drive))
    np.testing.assert_array_equal(back.brain.efficacy, learner.brain.efficacy)
