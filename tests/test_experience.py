"""Experience tutorials must depend on witnessed evidence and keep imagination read-only."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import cadence as cd


def example(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "examples" / (name + ".py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_experience_actions_depend_on_acquired_memories(seed):
    result = example("experience").run(seed)
    assert result["reached_object"] and result["single_sightings"] == 1
    assert result["choices"]["no_events"] is None
    assert result["choices"]["no_words"] is None
    assert result["choices"]["wrong_world"] != result["choices"]["full"]
    assert result["last_16_prediction_error"] < result["first_16_prediction_error"]


def test_search_cannot_know_unobserved_transitions_or_teach_itself():
    m = example("experience")
    model = cd.SynapticMemory(np.arange(6), np.arange(6, 9), decay=1)
    model.reset(1)
    assert m.route(model, 0, 2) is None
    model.observe(m.cue(1, 6), m.cue(2, 3))  # witnessed 0 --action 1--> 2
    fast, slow, writes = model.strength.copy(), model.consolidated.copy(), model.writes
    assert m.route(model, 0, 2) == (1,)
    np.testing.assert_array_equal(model.strength, fast)
    np.testing.assert_array_equal(model.consolidated, slow)
    assert model.writes == writes


def test_reward_example_adapts_without_restarting_its_brain():
    result = example("generic_brain").run()
    assert result["before_change"] > 0.9
    assert result["after_adapting"] > 0.9
    assert result["just_after_change"] < result["after_adapting"]
