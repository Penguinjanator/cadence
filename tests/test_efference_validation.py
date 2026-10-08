"""Efference boundaries stay silent until a command and retain valid saved custody."""

import json

import numpy as np
import pytest

import cadence as cd

DRIVE = np.array([[1.0, 0.0, 0.0, 0.0]])


def test_resting_bias_does_not_invent_a_command_before_the_first_action():
    options = dict(modules=(8,), seed=0, episodic=False, resting_bias=0.5)
    control = cd.Brain.compose(4, 2, **options)
    candidate = cd.Brain.compose(4, 2, efference_amplitude=1.0, **options)
    np.testing.assert_array_equal(
        control.act(DRIVE, greedy=True), candidate.act(DRIVE, greedy=True)
    )
    neurons = np.asarray(candidate.connectome.populations["efference"])
    np.testing.assert_array_equal(candidate.brain.bias[neurons], 0.0)
    np.testing.assert_array_equal(candidate.basal_ganglia.state.activation[:, neurons], 0.0)


@pytest.mark.parametrize(
    "tamper",
    [
        "redirected_target",
        "wrong_command_width",
        "negative_amplitude",
        "boolean_amplitude",
        "downgraded_with_arrays",
        "downgraded_without_arrays",
    ],
)
def test_saved_efference_cannot_change_ports_or_silently_lose_its_state(tmp_path, tamper):
    brain = cd.Brain.compose(4, 2, modules=(8,), episodic=False, efference_amplitude=1.0)
    brain.act(DRIVE, greedy=True)
    path = brain.save(tmp_path / "original")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    metadata = json.loads(str(arrays["generic"]))
    if tamper == "redirected_target":
        metadata["efference"]["target"] = "motor"
    elif tamper == "wrong_command_width":
        metadata["efference"].update(source="association", target="prefrontal")
        for name in ("trace", "last"):
            arrays["efference/" + name] = np.zeros((1, 8))
    elif tamper == "negative_amplitude":
        metadata["efference"]["amplitude"] = -1.0
    elif tamper == "boolean_amplitude":
        metadata["efference"]["amplitude"] = True
    else:
        metadata["format"] = "cadence-generic/2"
        del metadata["efference"]
        if tamper == "downgraded_without_arrays":
            for name in ("trace", "last", "cold"):
                del arrays["efference/" + name]
    arrays["generic"] = np.array(json.dumps(metadata))
    broken = tmp_path / "broken.npz"
    np.savez(broken, **arrays)
    with pytest.raises(ValueError, match="efference"):
        cd.Brain.load(broken)


def test_orphaned_efference_arrays_are_not_ignored(tmp_path):
    brain = cd.Brain.compose(4, 2, modules=(8,), episodic=False)
    path = brain.save(tmp_path / "original")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["efference/trace"] = np.zeros((0, 2))
    broken = tmp_path / "broken.npz"
    np.savez(broken, **arrays)
    with pytest.raises(ValueError, match="efference"):
        cd.Brain.load(broken)
