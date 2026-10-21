"""The learned code region: with ``learned_code=None`` every brain, checkpoint and life is the
composition before it; with ``learned_code=units`` a region of the brain's own neurons settles
to a sparse code of the observation, the memories key on it, and it learns every moment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence import brain as cadence_brain
from cadence.genome import Genome, Projection, develop
from cadence.regions import Region, cortex, motor_cortex, prefrontal_cortex

# The acts and the array contents below were recorded on the NumPy solver; the compiled dense
# kernel (numba) orders the arithmetic otherwise and gives other bits, so those pins are read
# on the NumPy solver alone (CADENCE_FUSED=0 selects it).
NUMPY_SOLVER = not cadence_brain._FUSED
BYTES_REASON = (
    "acts and array contents were recorded on the NumPy solver; this run uses the compiled "
    "dense kernel (CADENCE_FUSED=0 selects the NumPy loop)"
)
CUE = np.array([[1.0, 0.0, 0.5, 0.0]])
GO = np.array([[0.0, 0.0, 0.0, 1.0]])
FACT = np.array([[1.0, 0.0, 0.0, 1.0]])
NAME = np.array([[1.0, 0.0, 0.0, 0.0]])
STREAM = (CUE, GO, FACT, NAME)
INPUTS, ACTIONS, UNITS = 53, 4, 64


# -- the founders: byte-identical to the composition before the learned code


def checkpoint_digest(path: Path) -> str:
    """SHA-256 over every array of a checkpoint, by name, with the metadata JSON entries
    read back without the library version that wrote them."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            value = data[name]
            h.update(name.encode())
            if name in ("meta", "generic"):
                meta = json.loads(str(value))
                meta.pop("version", None)
                h.update(json.dumps(meta, sort_keys=True).encode())
                continue
            h.update(str(value.dtype).encode() + str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def names_digest(path: Path) -> tuple[str, str]:
    """The checkpoint's format, and its metadata keys and member names, which no backend
    changes."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
        names = json.dumps([sorted(meta), sorted(data.files)])
    return meta["format"], hashlib.sha256(names.encode()).hexdigest()


def parameter_digest(brain: cd.Brain) -> str:
    connectome, graph = brain.connectome, brain.brain
    h = hashlib.sha256()
    for values in (connectome.pre, connectome.post, connectome.count, connectome.sign,
                   graph.efficacy, graph.bias, graph.log_gain):
        h.update(np.ascontiguousarray(np.asarray(values)).tobytes())
    populations = {k: [int(i) for i in v] for k, v in connectome.populations.items()}
    h.update(json.dumps(populations).encode())
    return h.hexdigest()


def describe_digest(brain: cd.Brain) -> str:
    return hashlib.sha256(json.dumps(brain.describe(), sort_keys=True).encode()).hexdigest()


def stepped(brain: cd.Brain) -> list[int]:
    width = len(brain.sensory_index)
    rng = np.random.default_rng(21)
    acts = []
    for t in range(12):
        x = STREAM[t % 4] if width == 4 else (rng.random((1, width)) < 0.2).astype(float)
        if t == 0:
            action = brain.step(x)
        elif t % 3 == 0:
            action = brain.step(x, reward=[float(t % 2)], teacher=[t % 2])
        else:
            action = brain.step(x, reward=[float(t % 2) - 0.5], done=[t == 7])
        acts.append(int(np.asarray(action).ravel()[0]))
    acts.append(int(brain.act(x, greedy=True)[0]))
    return acts


def lived(brain: cd.Brain) -> list[int]:
    width = len(brain.sensory_index)
    rng = np.random.default_rng(22)
    frames = [STREAM[t % 4] if width == 4 else (rng.random((1, width)) < 0.2).astype(float)
              for t in range(14)]
    acts = []
    for t, x in enumerate(frames):
        reward = None if t == 0 else [float(t % 3 == 0)]
        done = None if t == 0 else [t == 9]
        acts.append(int(np.asarray(brain.live(x, reward=reward, done=done))[0]))
        if t == 5:
            brain.wait(frames[(t + 1) % 14])
    phases = brain.imagine(frames[:3])
    acts.extend(
        int(np.argmax(np.asarray(phase.state.activation)[0, brain.motor_index]))
        for phase in phases
    )
    return acts


def taught(brain: cd.Brain) -> list[int]:
    """Sixteen one-hot tokens, lessons judged by the act, as the binding chambers teach."""
    rng = np.random.default_rng(23)
    acts = []
    for t in range(10):
        x = np.eye(16)[rng.integers(16, size=4)]
        labels = rng.integers(8, size=4)
        if t % 2:
            brain.lesson(x, labels)
        acts.extend(int(a) for a in brain.act(x, greedy=True))
    return acts


# Recorded on the composition before the learned code (its parent commit): each founder's
# connectome and parameters, the acts of a short life, the description, and the format, the
# member and metadata names and the contents of every array of the checkpoint it saves.
FOUNDERS: dict[str, dict] = {
    "stepped": {
        "build": (4, 2, {"modules": (8,), "seed": 3}),
        "life": stepped,
        "parameters": "b81d75ef9bab5751c83c93bf64eb78e37b8eba039698948453ddc90b8bb4bee4",
        "acts": [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
        "format": "cadence-generic/2",
        "names": "5a728760433a9dea87769077dd838133cd49a8637125759627166a31d5cf16a8",
        "checkpoint": "03067e0fea6cf3e6a70d853e26355f17469035d66c097f42f1286b411a4ea5aa",
        "describe": "31e80db04583f2521aa45f841930ef549cf3d2718220d952d270597bc878c7f8",
    },
    "episodic": {
        "build": (4, 2, {"modules": (8,), "seed": 7, "memory_choice": True, "arousal": True}),
        "life": lived,
        "parameters": "aad357f3afceae3cfd5528b2a12c02ee7405eedd5ada8922cd1785cf66189164",
        "acts": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "format": "cadence-generic/6",
        "names": "2d5620ee6ddfa07ee9bdfbb174903829313bc90cb9ff24c09e9e530f7e2dd4be",
        "checkpoint": "2a2bdcd2f204f66b80c2c67d1af5d1c84dad39220a13caefc2a08bdc42986ac0",
        "describe": "3e5220b459d57a28dab8d72e5973a07a0d43f89b824071cdb4ff80f8956225ba",
    },
    "recalled": {
        "build": (4, 2, {"modules": (8,), "seed": 5, "arousal": True, "recall": True,
                         "history_decays": (0.2, 0.9), "efference_amplitude": 1.0,
                         "memory_choice": True}),
        "life": lived,
        "parameters": "7ccd7c7855a971743e898c29204bc3050d63d6ce0cde809cdca7fd7cadca5c4f",
        "acts": [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        "format": "cadence-generic/8",
        "names": "17b2b7a61fa822e150659b765cef59befb6219db6791a720d7ff79b32498636c",
        "checkpoint": "41f1e5283168a9f7969ae08f166ef1e3169f390edc59f3ce361bea31c5524321",
        "describe": "5424a8a43caa901e9164dfbc378da842dd811c96b478e17dc3d568b5ee6a44de",
    },
    "taught": {
        "build": (16, 8, {"modules": (64,), "lateral": 0.0, "seed": 1,
                          "learning_surprise": True, "recall": True}),
        "life": taught,
        "parameters": "1b72292d2b15754485932c2527b342353a6934c638e8e25f3da7d05e6b1e45ce",
        "acts": [4, 7, 1, 0, 0, 4, 4, 4, 0, 4, 4, 4, 0, 0, 7, 3, 0, 0, 7, 4, 0, 0, 7, 4, 1, 0, 0, 4, 3, 0, 0, 1, 0, 6, 0, 7, 0, 0, 0, 1],
        "format": "cadence-generic/8",
        "names": "a172f4a6e60b908b32b073312f524b39ac0283ad0ee1e21954d164a9c82ef000",
        "checkpoint": "a2b32980250fd9afa9bedaf63dac8cb549147addffab04507c85d76edf764ccf",
        "describe": "44c7b59da31a529ff39ae09c1dac85895dfe4d7b5c2b752fababb61b1a429c32",
    },
    "wide": {
        "build": (2_000, 3, {"modules": (8,), "seed": 13, "memory_choice": True,
                             "arousal": True}),
        "life": lived,
        "parameters": "63de06ff41d1403e4e706fbf3095c48d555682191afc24b659c5e7e90d28acf3",
        "acts": [2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "format": "cadence-generic/6",
        "names": "2d5620ee6ddfa07ee9bdfbb174903829313bc90cb9ff24c09e9e530f7e2dd4be",
        "checkpoint": "7b8bf9c566606464d87941f321b4aac2421a7c07b882959ae3bc160512efa90e",
        "describe": "cad22c16dd28ee758ef0f0a668b0e42282a94878745d3c9547490bbc5b5521fc",
    },
}


@pytest.mark.parametrize("name", sorted(FOUNDERS))
def test_a_founder_is_the_composition_before_the_learned_code(tmp_path, name):
    """On every backend: the connectome and parameters, the description, and the checkpoint's
    format, members and metadata keys after the life; no code region, no learned-code
    metadata, the memories keyed as before."""
    spec = FOUNDERS[name]
    inputs, actions, options = spec["build"]
    brain = cd.Brain.compose(inputs, actions, **options)
    assert brain.learned_code is None and "code" not in brain.connectome.populations
    assert brain.learner.plastic_synapses.all() and brain.learner.plastic_neurons.all()
    assert parameter_digest(brain) == spec["parameters"]
    spec["life"](brain)
    assert describe_digest(brain) == spec["describe"]
    assert "learned_code" not in brain.describe()
    assert not any(k.startswith("learned_code") for k in brain.describe()["genes"])
    path = brain.save(tmp_path / "founder.npz")
    assert names_digest(path) == (spec["format"], spec["names"])
    with np.load(path, allow_pickle=False) as data:
        assert not any("code" in member for member in data.files)


@pytest.mark.skipif(not NUMPY_SOLVER, reason=BYTES_REASON)
@pytest.mark.parametrize("name", sorted(FOUNDERS))
def test_the_founders_acts_and_checkpoint_bytes_are_the_recorded_ones(tmp_path, name):
    spec = FOUNDERS[name]
    inputs, actions, options = spec["build"]
    brain = cd.Brain.compose(inputs, actions, **options)
    assert spec["life"](brain) == spec["acts"]
    assert checkpoint_digest(brain.save(tmp_path / "founder.npz")) == spec["checkpoint"]


def test_learned_code_none_is_the_founder_and_the_width_is_as_before():
    assert cd.Brain.observation_width(53) == 53 and cd.Brain.observation_width(53, 24) == 24
    assert cd.Brain.observation_width(53, None, 16) == 16
    for value in (None, 0):
        brain = cd.Brain.compose(53, 4, modules=(8,), seed=2, learned_code=value)
        assert brain.learned_code is None and brain.code_width == 53
        assert "learned_code" not in brain.describe()["initialization"]["composition"]


# -- the region


def composed(seed: int = 0, units: int = UNITS, **options):
    return cd.Brain.compose(
        INPUTS, ACTIONS, modules=(64,), seed=seed, learning_surprise=True, learned_code=units,
        **options,
    )


def tokens(*indices: int) -> np.ndarray:
    return np.eye(INPUTS)[list(indices)]


def test_the_region_joins_the_connectome_with_its_three_synapse_groups():
    brain = composed()
    populations = brain.connectome.populations
    assert list(populations)[:3] == ["sensory", "code", "association"]
    assert len(populations["code"]) == UNITS and brain.code_width == UNITS
    code = brain.learned_code
    assert isinstance(code, cd.LearnedCode)
    assert code.inputs == INPUTS and code.units == UNITS and code.context == 64
    groups = brain._code_synapses
    assert len(groups["dictionary"][0]) == INPUTS * UNITS
    assert len(groups["lateral"][0]) == UNITS * (UNITS - 1)
    assert len(groups["predictor"][0]) == 64 * UNITS
    pre, post = np.asarray(brain.connectome.pre), np.asarray(brain.connectome.post)
    # the sensory region projects to the code region alone, and the code to the first module
    assert np.isin(post[np.isin(pre, brain.sensory_index)], brain.code_index).all()
    forward = np.isin(pre, brain.code_index) & np.isin(post, brain.association_index)
    assert forward.sum() == UNITS * 64
    # nothing leads back into the region, and the sensory and prefrontal neurons hear nothing
    assert not np.isin(post, np.concatenate([brain.sensory_index, brain.prefrontal_index])).any()
    into = np.isin(post, brain.code_index)
    assert np.isin(pre[into], np.concatenate(
        [brain.sensory_index, brain.code_index, brain.prefrontal_index])).all()
    # the tables are the synapses: the dictionary, the Gram as inhibition, the predictor
    efficacy = brain.brain.efficacy
    synapses, i, j = groups["dictionary"]
    np.testing.assert_array_equal(efficacy[synapses], code.read * code.dictionary[i, j])
    # the read gain: one input at amplitude 1 drives the unit whose column matches it by one
    model = brain.brain.neuron_model
    assert np.isclose(code.read * model.activation(np.array([model.stimulus_amplitude])), 1.0)
    synapses, j, k = groups["lateral"]
    np.testing.assert_allclose(efficacy[synapses], -(code.dictionary[:, j] * code.dictionary[:, k]).sum(0))
    synapses, i, j = groups["predictor"]
    np.testing.assert_array_equal(efficacy[synapses], 0.0)
    np.testing.assert_array_equal(brain.brain.bias[brain.code_index], -code.thresholds)
    np.testing.assert_allclose(np.linalg.norm(code.dictionary, axis=0), 1.0)
    # the lesson's contrast and the dopamine leave them alone
    plastic = brain.learner.plastic_synapses
    for synapses, _, _ in groups.values():
        assert not plastic[synapses].any()
    assert plastic.sum() == brain.connectome.synapses - sum(len(g[0]) for g in groups.values())
    assert not brain.learner.plastic_neurons[brain.code_index].any()
    assert brain.learner.plastic_neurons.sum() == brain.connectome.n - UNITS
    # the memories key on the code
    np.testing.assert_array_equal(brain.hippocampus.pre, np.arange(UNITS))
    described = brain.describe()
    assert described["observation_code"] == {
        "kind": "learned", "inputs": INPUTS, "width": UNITS, "units": UNITS, "fan_in": 0,
        "winners": 0, "cortex": "code", "separator": None,
    }
    assert described["learned_code"]["units"] == UNITS
    assert described["initialization"]["composition"]["learned_code"] == UNITS
    assert {k: v for k, v in described["genes"].items() if k.startswith("learned_code")} == {
        f"learned_code_{name}": value for name, value in cd.LearnedCode.founders().items()
    }
    assert brain.parameters() >= code.parameters()


def test_the_founders_are_the_declared_genes_and_bad_settings_are_refused():
    assert cd.LearnedCode.founders() == {
        "top_down": 4.0, "rate": 0.125, "predictor": 0.03, "threshold": 0.1,
        "threshold_rate": 0.001, "target": 0.1, "repair": 1.0, "silence": 10.0,
        "place": "free",
    }
    with pytest.raises(ValueError, match="repair must lie in"):
        composed(learned_code_repair=0.0)
    with pytest.raises(ValueError, match="repair"):
        composed(learned_code_repair=1.5)
    with pytest.raises(ValueError, match="at least 2 units"):
        composed(units=1)
    with pytest.raises(ValueError, match="learned_code must be an integer"):
        composed(units=2.5)
    with pytest.raises(ValueError, match="need learned_code=units"):
        cd.Brain.compose(53, 4, modules=(8,), learned_code_rate=0.1)
    with pytest.raises(TypeError, match="unknown brain setting"):
        composed(learned_code_speed=0.1)
    with pytest.raises(ValueError, match="joins identity and meaning"):
        composed(code=64)
    with pytest.raises(ValueError, match="a vector brain"):
        cd.Brain.compose((8, 8), 2, modules=(8,), learned_code=32)
    with pytest.raises(ValueError, match="target must lie in"):
        composed(learned_code_target=1.0)
    with pytest.raises(ValueError, match="rate must be finite and lie in"):
        composed(learned_code_rate=-0.1)
    with pytest.raises(ValueError, match="key on its code as it is"):
        composed(memory_choice=True, memory_expansion=512)
    brain = composed()
    with pytest.raises(ValueError, match="construction setting"):
        brain.retune(learned_code=32)
    with pytest.raises(ValueError, match="construction setting"):
        brain.retune(learned_code_threshold=0.2)
    with pytest.raises(ValueError, match="requires its existing learned code"):
        composed(units=None).retune(learned_code_rate=0.0)
    # a custom anatomy needs the region and its complete synapses
    regions = [Region("sensory", 8), cortex(6, lateral=-1.0, name="code"),
               Region("association", 8), prefrontal_cortex(8), motor_cortex(2)]
    projections = [Projection("sensory", "code", reciprocal=False),
                   Projection("code", "association", reciprocal=False),
                   Projection("association", "motor"),
                   Projection("prefrontal", "association", reciprocal=False)]
    graph = develop(Genome(tuple(regions), tuple(projections)), seed=1)
    with pytest.raises(ValueError, match="complete predictor synapses"):
        cd.Brain(graph, learned_code=6)
    with pytest.raises(ValueError, match="needs a population named 'code'"):
        cd.Brain(cd.Brain.compose(8, 2, modules=(8,)).connectome, learned_code=6)
    complete = develop(Genome(tuple(regions), (*projections, Projection(
        "prefrontal", "code", scale=0.0, reciprocal=False))), seed=1)
    custom = cd.Brain(complete, learned_code=6)
    assert custom.learned_code is not None and custom.code_width == 6
    with pytest.raises(ValueError, match="is not the learned code of 5 units"):
        cd.Brain(complete, learned_code=5)


def settled_code(brain: cd.Brain) -> np.ndarray:
    return np.atleast_2d(brain.basal_ganglia.state.activation)[:, brain.code_index]


def test_the_code_is_a_fixed_point_of_the_whole_settle_and_the_memories_key():
    """After an act the whole graph's residual holds on the code units; the free state's code
    is the phase the memories keyed on, within the tolerance; the key is the rectified code
    at unit length; the reconstruction error fell over the phase's sweeps."""
    brain = composed(recall=True)
    x = tokens(3, 7, 11)
    brain.act(x, greedy=True)
    report = brain.last_settlement
    assert report["qualified"] and report["code_steps"] > 0
    assert report["code_residual"] <= brain.learner.config.tolerance
    code = brain.learned_code
    s = settled_code(brain)
    key = code.code_of(s)
    assert (key >= 0).all() and np.allclose(np.linalg.norm(key, axis=1), 1.0)
    assert ((key > 0).sum(axis=1) >= 1).all() and ((key > 0).sum(axis=1) < UNITS // 2).all()
    np.testing.assert_allclose(brain.recall.keys[:, -1, :], key, atol=1e-3)
    np.testing.assert_allclose(brain.recall.values[:, -1, :], key, atol=1e-3)
    # the whole graph's equations hold on the code units at the settled state
    drive = brain.stimulus(x)
    state = brain.basal_ganglia.state
    graph = brain.brain
    total = graph._synaptic_input(np.atleast_2d(state.activation)) + drive + graph.bias
    if brain.learner.standing is not None:  # the code units' restoring term, every phase's
        total = total + brain.learner.standing.drive(np.atleast_2d(state.activation))
    residual = np.abs(total - np.atleast_2d(state.v))[:, brain.code_index].max()
    assert residual <= brain.learner.config.tolerance
    # the code phase's own fixed point: settled again under the same stimulus it stays
    phase = brain._settle_code(x)
    np.testing.assert_allclose(phase.state.activation, s, atol=1e-3)
    # the reconstruction error falls over the phase's sweeps, from rest to the fixed point
    last = code.last
    stimulus = code.stimulus(last["sensory"], last["context"])
    trajectory = code.graph.settle_batch(stimulus, steps=phase.steps or 32, trajectory=True)
    assert trajectory.trajectory is not None
    errors = [
        code.residuals(last["sensory"], a, None)["bottom_up"].mean()
        for a in (np.zeros_like(s), *trajectory.trajectory)
    ]
    assert errors[-1] < errors[0] and errors[-1] < 1.0
    # context-free: the public code reads the observation alone, in the same coordinates
    free = brain.observation_code(x)
    assert free.shape == key.shape and (free >= 0).all()


def test_the_region_learns_every_moment_on_its_residuals_and_homeostasis_moves_thresholds():
    brain = composed()
    code = brain.learned_code
    before = {name: value.copy() for name, value in code.arrays().items()}
    x = tokens(1, 2, 5, 9)
    brain.act(x, greedy=True)  # one accepted moment: one step
    assert code.work["updates"] == 1 and code.work["update_rows"] == 4
    last = code.last
    a = np.maximum(last["activation"], 0.0)
    active = (a > 0).any(axis=0)
    # the dictionary moved on the residual, in the columns of active units alone
    residual = code.read * last["sensory"] - a @ before["dictionary"].T
    expected = before["dictionary"] + code.rate * (residual.T @ a)
    expected /= np.linalg.norm(expected, axis=0, keepdims=True)
    # the step reads the sensory activation the settle holds, within the solver's tolerance
    # of its fixed point under a halved step
    np.testing.assert_allclose(code.dictionary, expected, atol=1e-6)
    np.testing.assert_array_equal(code.dictionary[:, ~active], before["dictionary"][:, ~active])
    assert not np.array_equal(code.dictionary[:, active], before["dictionary"][:, active])
    # the predictor moved by the delta rule on the top-down residual, bounded per row
    z = last["context"]
    rate = code.predictor / np.maximum(np.sum(z * z, axis=1), code.predictor**2)
    np.testing.assert_allclose(
        code.weights, before["weights"] + ((a - z @ before["weights"].T) * rate[:, None]).T @ z,
        atol=1e-12,
    )
    # the thresholds: a unit active in more rows than the target rises, a silent one falls;
    # the share reads the context-free code (the code itself at a stream's first moment,
    # within the solver's tolerance), and the repair what both codes leave unexplained
    free = last["free"]
    np.testing.assert_allclose(free, a, atol=1e-5)
    share = (free > 0).sum(axis=0) - 4 * code.target
    x_read = code.read * last["sensory"]
    repair = code.unexplained(x_read, a) + code.unexplained(x_read, free)
    np.testing.assert_allclose(
        code.thresholds,
        np.maximum(before["thresholds"] + code.threshold_rate * (share - code.silence * repair),
                   0.0),
    )
    assert (code.thresholds[~active] < before["thresholds"][~active]).all()
    # the graph took the tables
    efficacy = brain.brain.efficacy
    synapses, i, j = brain._code_synapses["dictionary"]
    np.testing.assert_array_equal(efficacy[synapses], code.read * code.dictionary[i, j])
    synapses, j, k = brain._code_synapses["lateral"]
    np.testing.assert_allclose(efficacy[synapses], code.lateral()[j, k])
    synapses, i, j = brain._code_synapses["predictor"]
    np.testing.assert_allclose(efficacy[synapses], code.top_down * code.weights[j, i])
    np.testing.assert_array_equal(brain.brain.bias[brain.code_index], -code.thresholds)
    # a moment the code already fits moves the dictionary little: the residual is the step
    for _ in range(40):
        brain.act(x, greedy=True)
    settled = {name: value.copy() for name, value in code.arrays().items()}
    brain.act(x, greedy=True)
    assert np.abs(code.dictionary - settled["dictionary"]).max() < 1e-2
    # zero rates freeze the tables
    brain.retune(learned_code_rate=0.0, learned_code_predictor=0.0,
                 learned_code_threshold_rate=0.0)
    frozen = {name: value.copy() for name, value in code.arrays().items()}
    brain.act(tokens(4, 4, 4, 4), greedy=True)
    for name, value in frozen.items():
        np.testing.assert_array_equal(getattr(code, name), value)
    # the thresholds never fall below zero
    low = composed(seed=3, learned_code_threshold=0.0, learned_code_threshold_rate=0.5)
    low.act(tokens(1), greedy=True)
    assert (low.learned_code.thresholds >= 0.0).all()


def test_the_lesson_and_the_outcome_leave_the_region_to_its_own_rule():
    brain = composed(recall=True)
    x = tokens(2, 6)
    brain.act(x, greedy=True)
    code = brain.learned_code
    tables = {name: value.copy() for name, value in code.arrays().items()}
    efficacy = brain.brain.efficacy.copy()
    report = brain.lesson(x, [1, 3])  # taught rows: the contrast moves the plastic synapses
    assert report["accepted_presentations"] > 0
    plastic = brain.learner.plastic_synapses
    assert (brain.brain.efficacy[plastic] != efficacy[plastic]).any()
    np.testing.assert_array_equal(brain.brain.efficacy[~plastic], efficacy[~plastic])
    for name, value in tables.items():
        np.testing.assert_array_equal(getattr(code, name), value)  # a lesson is no moment
    brain.step(x)  # a sampled act: one moment, then its outcome
    moved = {name: value.copy() for name, value in code.arrays().items()}
    assert not np.array_equal(moved["dictionary"], tables["dictionary"])
    efficacy = brain.brain.efficacy.copy()
    brain.learn(np.array([1.0, 0.0]), np.array([False, False]), x)
    np.testing.assert_array_equal(brain.brain.efficacy[~plastic], efficacy[~plastic])
    for name, value in moved.items():
        np.testing.assert_array_equal(getattr(code, name), value)


def test_the_key_of_a_decision_is_its_moments_code_and_the_history_hears_the_code():
    brain = composed(recall=True, history_decays=(0.5,), arousal=True, memory_choice=True)
    assert len(brain.connectome.populations["history"]) == UNITS
    x = tokens(5)
    brain.live(x)
    first = brain.learned_code.code_of(settled_code(brain))  # (1, units)
    key = brain.recall.keys[0, 0]  # the moment's key: the phase's code in the stream's context
    np.testing.assert_allclose(brain.history.trace, 0.5 * key[None, :], atol=1e-9)
    assert brain._lived[7] is not None
    # the keys are the phase's code; the whole settle moves the code units within tolerance
    np.testing.assert_allclose(brain._lived[7], first, atol=1e-3)
    np.testing.assert_allclose(key, first[0], atol=1e-3)
    brain.live(tokens(9), reward=[1.0])  # the outcome wakes the brain: recorded under the code
    assert brain.hippocampus.writes >= 1
    keyed = brain.hippocampus.recall(first)
    assert np.abs(keyed).max() > 0
    # a sampled act keeps the code it was keyed on until its outcome
    stepped = composed(seed=4)
    stepped.step(tokens(1, 2))
    assert stepped._moment_code is not None and stepped._moment_code.shape == (2, UNITS)
    stepped.step(tokens(3, 4), reward=[1.0, 0.0])
    assert stepped._moment_code is not None  # the next sampled act's


def test_a_checkpoint_carries_the_region_and_a_tampered_one_is_refused(tmp_path):
    brain = composed(recall=True, arousal=True, memory_choice=True, history_decays=(0.3,))
    x = [tokens(k) for k in (1, 4, 7)]
    for t in range(5):
        brain.live(x[t % 3], reward=None if t == 0 else [float(t % 2)])
    brain.wait(x[1])
    path = brain.save(tmp_path / "learned.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    meta = json.loads(str(arrays["generic"]))
    assert meta["format"] == "cadence-generic/10"
    assert meta["learned_code"]["units"] == UNITS and meta["composition"]["learned_code"] == UNITS
    assert meta["history"]["source"] == "code" and meta["recall"]["inputs"] == UNITS
    assert arrays["learned_code/dictionary"].shape == (INPUTS, UNITS)
    assert arrays["learned_code/weights"].shape == (UNITS, 64)
    assert arrays["learned_code/thresholds"].shape == (UNITS,)
    assert arrays["lived/code"].shape == (1, UNITS)
    twin = cd.Brain.load(path)
    assert twin.describe() == brain.describe()
    for name, value in brain.learned_code.arrays().items():
        np.testing.assert_array_equal(getattr(twin.learned_code, name), value)
    np.testing.assert_array_equal(twin.brain.efficacy, brain.brain.efficacy)
    assert checkpoint_digest(twin.save(tmp_path / "again.npz")) == checkpoint_digest(path)
    for t in range(3):
        np.testing.assert_array_equal(
            brain.live(x[t], reward=[0.5]), twin.live(x[t], reward=[0.5])
        )
    assert checkpoint_digest(brain.save(tmp_path / "a.npz")) == checkpoint_digest(
        twin.save(tmp_path / "b.npz")
    )
    # a sampled act awaiting its outcome saves the code it was keyed on
    sampled = composed(seed=2)
    sampled.step(tokens(2, 3))
    kept = sampled.save(tmp_path / "pending.npz")
    with np.load(kept, allow_pickle=False) as data:
        assert data["moment/code"].shape == (2, UNITS)
    loaded = cd.Brain.load(kept)
    np.testing.assert_array_equal(loaded._moment_code, sampled._moment_code)
    np.testing.assert_array_equal(
        loaded.step(tokens(5, 6), reward=[1.0, 0.0]), sampled.step(tokens(5, 6), reward=[1.0, 0.0])
    )

    def tampered(name: str, mutate) -> Path:
        copy = {k: v.copy() for k, v in arrays.items()}
        meta = json.loads(str(copy["generic"]))
        mutate(meta, copy)
        copy["generic"] = np.array(json.dumps(meta))
        np.savez(tmp_path / f"{name}.npz", **copy)
        return tmp_path / f"{name}.npz"

    def no_region(meta, _):
        del meta["learned_code"]

    def earlier_format(meta, _):
        meta["format"] = "cadence-generic/9"

    def other_units(meta, _):
        meta["learned_code"]["units"] = 32

    def unknown_field(meta, _):
        meta["learned_code"]["fan_in"] = 8

    def bad_gene(meta, _):
        meta["learned_code"]["target"] = 2.0

    def no_table(meta, copy):
        del copy["learned_code/weights"]

    def wrong_shape(meta, copy):
        copy["learned_code/thresholds"] = copy["learned_code/thresholds"][:-1]

    def negative_threshold(meta, copy):
        copy["learned_code/thresholds"][0] = -0.1

    def long_column(meta, copy):
        copy["learned_code/dictionary"][:, 0] *= 2.0

    def other_dictionary(meta, copy):  # unit columns that disagree with the saved synapses
        column = copy["learned_code/dictionary"][:, 0].copy()
        copy["learned_code/dictionary"][:, 0] = copy["learned_code/dictionary"][:, 1]
        copy["learned_code/dictionary"][:, 1] = column

    def lone_code(meta, copy):
        del meta["lived"]
        del copy["lived/observations"]
        del copy["lived/action"]

    for name, mutate, message in (
        ("no_region", no_region, "belongs to checkpoint format cadence-generic/10"),
        ("earlier_format", earlier_format, "belongs to checkpoint format cadence-generic/10"),
        ("other_units", other_units, "is not the saved learned code of 32 units"),
        ("unknown_field", unknown_field, "invalid saved learned code metadata"),
        ("bad_gene", bad_gene, "invalid saved learned code target"),
        ("no_table", no_table, "missing saved learned code table: weights"),
        ("wrong_shape", wrong_shape, "invalid saved learned code table: thresholds"),
        ("negative_threshold", negative_threshold, "thresholds are nonnegative"),
        ("long_column", long_column, "unit-length columns"),
        ("other_dictionary", other_dictionary, "disagree with the saved synapses"),
        ("lone_code", lone_code, "saved lived code belongs to a lived action"),
    ):
        with pytest.raises(ValueError, match=message):
            cd.Brain.load(tampered(name, mutate))


def test_a_code_phase_that_cannot_settle_refuses_the_moment_before_anything_changes():
    brain = composed(seed=5, learning_free_steps=1)
    x = tokens(1, 2)
    before = brain.basal_ganglia.state
    with pytest.raises(RuntimeError, match="code region did not settle"):
        brain.act(x, greedy=True)
    assert brain.basal_ganglia.state is before and brain._coded_moment is None
    assert brain.learned_code.work["capped"] == 1 and brain.learned_code.work["updates"] == 0
    assert brain.last_settlement is None


def test_independent_samples_settle_the_context_free_code():
    brain = composed(seed=6, units=24, episodic=False)
    x = np.eye(INPUTS)[:6]
    labels = np.arange(6) % ACTIONS
    assert brain.predict(x).shape == (6,)
    assert 0.0 <= brain.accuracy(x, labels) <= 1.0
    history = brain.fit(x, labels, epochs=2)
    assert len(history) == 2
    assert brain.learned_code.work["updates"] == 0  # no accepted moment: no step
    np.testing.assert_array_equal(brain.observation_code(x), brain.observation_code(x))


def test_the_brain_integrates_at_the_step_the_codes_competition_is_stable_under(tmp_path):
    """A dictionary whose columns overlap strongly makes the synchronous sweep oscillate
    around the code's fixed point at the model's step; the region reports the halvings the
    settled code needs, the brain re-steps its graph and scales the step budgets, the
    description keeps the configurations as given, and a checkpoint carries the level."""
    brain = composed(seed=8, units=24, recall=True)
    code = brain.learned_code
    assert brain._stepping == 0 and brain.learner.standing is not None
    # just above threshold (slope .5) the restoring term alone asks for the halved step at
    # gain 4; without the boundary the bound reads the lateral coupling alone
    near = np.full((1, 24), 0.01)
    with_boundary = code.stability(near)
    assert with_boundary >= 1
    brain.retune(learned_code_top_down=0.0)
    assert brain.learner.standing is None and code.stability(near) < with_boundary
    brain.retune(learned_code_top_down=4.0)
    assert brain.learner.standing is not None
    # columns made alike by hand: the units that fire together inhibit each other at full strength
    alike = code.dictionary.copy()
    alike[:, 1:8] = alike[:, [0]] + 0.05 * np.random.default_rng(1).standard_normal((INPUTS, 7))
    alike /= np.linalg.norm(alike, axis=0, keepdims=True)
    code.take(alike, code.weights, np.zeros(24))
    brain.learner.brain = brain._installed(brain.brain)
    x = tokens(0, 0)
    x[:, :] = alike[:, 0][None, :].clip(0)  # an observation the alike columns all explain
    brain.act(x, greedy=True)
    report = brain.last_settlement
    assert report["qualified"] and report["code_halvings"] >= 1
    level = brain._stepping
    assert level == report["code_halvings"] == code.halvings
    assert brain.brain.neuron_model.dt == 1.0 / 2**level == code.graph.neuron_model.dt
    base = brain._learning_base
    assert brain.learner.config.free_steps == base.free_steps << level
    assert brain.learner.config.nudged_steps == base.nudged_steps << level
    assert brain.basal_ganglia.config.eligibility_steps == 12 << level
    described = brain.describe()
    assert described["learning"]["free_steps"] == base.free_steps == 1024
    assert described["actor"]["eligibility_steps"] == 12
    assert described["learned_code"]["stepping"] == level
    # a lesson and an outcome run at the stepped budgets and leave the level's bookkeeping
    brain.lesson(x, [1, 2])
    brain.step(x)
    brain.step(x, reward=[1.0, 0.0])
    brain.retune(learning_eta=0.4, actor_eta=0.5)
    assert brain._learning_base.eta == 0.4 and brain.learner.config.eta == 0.4
    assert brain.learner.config.free_steps == 1024 << brain._stepping
    assert brain.basal_ganglia.config.eta == 0.5
    assert brain.basal_ganglia.config.eligibility_steps == 12 << brain._stepping
    path = brain.save(tmp_path / "stepped.npz")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
        saved_dt = json.loads(str(data["meta"]))["neuron_model"]["dt"]
    assert meta["learned_code"]["stepping"] == brain._stepping
    assert saved_dt == brain.brain.neuron_model.dt
    twin = cd.Brain.load(path)
    assert twin._stepping == brain._stepping and twin.describe() == brain.describe()
    assert twin.learner.config == brain.learner.config
    assert twin.basal_ganglia.config == brain.basal_ganglia.config
    np.testing.assert_array_equal(twin.act(x, greedy=True), brain.act(x, greedy=True))
    # a code that is stable again steps the brain back up: orthogonal columns and no
    # restoring term
    brain.retune(learned_code_top_down=0.0)
    code.take(np.linalg.qr(np.random.default_rng(2).standard_normal((INPUTS, 24)))[0],
              code.weights, np.full(24, 0.1))
    brain.learner.brain = brain._installed(brain.brain)
    brain.act(tokens(3, 4), greedy=True)
    assert brain._stepping == 0 and brain.brain.neuron_model.dt == 1.0
    assert brain.learner.config.free_steps == 1024
    # a saved stepping that disagrees with the saved step is refused
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    meta = json.loads(str(arrays["generic"]))
    meta["learned_code"]["stepping"] = 0
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "wrong.npz", **arrays)
    with pytest.raises(ValueError, match="stepping disagrees with the saved step"):
        cd.Brain.load(tmp_path / "wrong.npz")


def test_the_place_stores_key_on_the_context_free_code_and_the_others_on_the_moments():
    """The associative store and the episodic place records key on the context-free code,
    the observation settled alone; the recall slots and the context records on the moment's
    code in its context; ``learned_code_place="context"`` keys the place stores on the
    moment's code; the place phase is reported and costs nothing without a place store."""
    brain = composed(seed=9, recall=True, memory_choice=True, arousal=True)
    code = brain.learned_code
    assert code.place == "free"
    # a predictor with weight: at birth it is zero and the two keys coincide
    code.weights[:] = 0.2 * np.random.default_rng(3).random(code.weights.shape)
    code.install()
    brain.learner.brain = brain._installed(brain.brain)
    # the tables held still, so the moment's place key and a later context-free settle agree
    brain.retune(learned_code_rate=0.0, learned_code_predictor=0.0,
                 learned_code_threshold_rate=0.0)
    context, token = tokens(40), tokens(3)
    brain.live(context)
    before = brain._context.copy()
    brain.live(token, reward=[0.0])
    report = brain.last_settlement
    assert report["context_free_steps"] >= 0 and "code_steps" in report
    coded = brain._coded_moment
    assert coded is not None and coded[0] is not None
    key, place = coded[1], coded[3]
    np.testing.assert_allclose(place, brain.observation_code(token), atol=5e-3)
    assert not np.array_equal(key, place)  # the context predicted the token's units
    np.testing.assert_array_equal(brain.recall.keys[0, -1], key[0])  # the slot: in context
    np.testing.assert_array_equal(brain._lived[7], place)  # the decision's place key
    np.testing.assert_array_equal(brain._paths[0][-1][2], place[0])  # the path: the place key
    np.testing.assert_allclose(  # the context key: the moment's key with the history
        brain._decision[0, :UNITS], 0.5 * before[0, :UNITS] + 0.5 * key[0], atol=1e-12
    )
    brain.live(tokens(44), reward=[1.0])  # the outcome wakes the brain and is recorded
    assert brain.hippocampus.writes >= 1
    assert np.abs(brain.hippocampus.recall(place)).max() > 0
    assert np.abs(brain.hippocampus.recall(place)).max() >= np.abs(brain.hippocampus.recall(key)).max()
    # the same token in another context: the place key is the same, the moment's key differs
    brain.live(tokens(41), reward=[0.0])
    brain.live(token, reward=[0.0])
    again = brain._coded_moment
    np.testing.assert_allclose(again[3], brain.observation_code(token), atol=5e-3)
    np.testing.assert_allclose(again[3], place, atol=5e-3)
    assert not np.array_equal(again[1], key)
    # without a place store the context-free phase still runs for the thresholds' share,
    # and the place key is the moment's key; with the stores keyed in context the same
    bare = composed(seed=9, episodic=False)
    bare.act(token, greedy=True)
    assert bare.last_settlement["context_free_steps"] > 0
    assert bare._coded_moment[3] is bare._coded_moment[1]
    frozen = composed(seed=9, episodic=False, learned_code_threshold_rate=0.0)
    frozen.act(token, greedy=True)  # thresholds frozen, no place store: no second phase
    assert "context_free_steps" not in frozen.last_settlement
    in_context = composed(seed=9, learned_code_place="context", arousal=True)
    in_context.live(context)
    in_context.live(token, reward=[0.0])
    assert in_context._coded_moment[3] is in_context._coded_moment[1]
    assert "context_free_steps" in in_context.last_settlement  # the thresholds' share reads it
    in_context.retune(learned_code_place="free")
    in_context.live(tokens(44), reward=[0.0])
    assert "context_free_steps" in in_context.last_settlement
    with pytest.raises(ValueError, match="place is one of"):
        composed(learned_code_place="route")
    # a step, its outcome and a lesson under the split keys
    taught = composed(seed=10, memory_choice=True)
    taught.step(tokens(1, 2))
    np.testing.assert_allclose(taught._moment_code, taught.observation_code(tokens(1, 2)), atol=5e-3)
    taught.step(tokens(3, 4), reward=[1.0, 0.0])
    taught.act(tokens(5, 6), greedy=True)  # a greedy act replaces the pending one
    taught.lesson(tokens(5, 6), [1, 2])
    assert taught.hippocampus.writes >= 2


def test_a_long_stream_leaves_the_code_alive(tmp_path):
    """Six tokens heard for 2,000 moments by one stream, a lesson every tenth moment: the
    thresholds stay finite and nonnegative, every token keeps at least one active unit in
    context and context-free, and at least one unit is in use per token; a checkpoint with
    a negative threshold is refused."""
    brain = composed(seed=11, units=32, recall=True, learned_code_threshold_rate=0.01)
    code = brain.learned_code
    rng = np.random.default_rng(5)
    for t in range(2000):
        token = int(rng.integers(6))
        x = tokens(token)
        if t % 10 == 9:
            brain.lesson(x, [token % ACTIONS])
        brain.act(x, greedy=True)
    assert np.isfinite(code.thresholds).all() and (code.thresholds >= 0.0).all()
    used = np.zeros(32, bool)
    for token in range(6):
        x = tokens(token)
        brain.act(x, greedy=True)
        in_context = np.maximum(code.last["activation"], 0.0)
        free = brain.observation_code(x)
        assert (in_context > 0).sum() >= 1 and (free > 0).sum() >= 1
        used |= (free > 0)[0]
    assert used.sum() >= 6
    np.testing.assert_array_equal(brain.brain.bias[brain.code_index], -code.thresholds)
    path = brain.save(tmp_path / "alive.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["learned_code/thresholds"][0] = -0.1
    np.savez(tmp_path / "below.npz", **arrays)
    with pytest.raises(ValueError, match="thresholds are nonnegative"):
        cd.Brain.load(tmp_path / "below.npz")


def test_the_repair_lowers_the_silent_units_of_rows_the_code_leaves_unexplained():
    """At a repair level the newborn's rows exceed (its random dictionary explains little),
    one moment lowers the thresholds of the units silent in those rows by the silence term
    beyond the share rule, and leaves the units that fired to the share rule; at the
    founder level 1 the same moment, whose rows are not silent, moves nothing beyond it."""
    x = tokens(1, 2, 5, 9)
    strict = composed(seed=4)
    strict.act(x, greedy=True)
    code = strict.learned_code
    free = code.last["free"]
    share = (free > 0).sum(axis=0) - len(x) * code.target
    np.testing.assert_allclose(
        code.thresholds, np.maximum(code.threshold + code.threshold_rate * share, 0.0)
    )
    repairing = composed(seed=4, learned_code_repair=0.5)
    repairing.act(x, greedy=True)
    code = repairing.learned_code
    a = np.maximum(code.last["activation"], 0.0)
    left = code.residuals(code.last["sensory"], a, None)["bottom_up"]
    rows = left >= 0.5
    assert rows.any()
    free = code.last["free"]
    x_read = code.read * code.last["sensory"]
    repair = code.unexplained(x_read, a) + code.unexplained(x_read, free)
    np.testing.assert_array_equal(code.unexplained(x_read, a), (rows[:, None] & ~(a > 0)).sum(0))
    share = (free > 0).sum(axis=0) - len(x) * code.target
    expected = code.threshold + code.threshold_rate * (share - code.silence * repair)
    np.testing.assert_allclose(code.thresholds, np.maximum(expected, 0.0))
    fired = (a > 0).any(axis=0)
    assert (code.thresholds[~fired] < code.threshold).all()
    assert code.work["repairs"] == int(repair.sum()) > 0
    assert strict.learned_code.work["repairs"] == 0  # no row was silent at the strict level


def test_the_region_learns_from_the_moments_free_settle_alone():
    """With the working trace silenced (so that the lesson's synaptic changes reach no
    code), a stream of lessons and outcomes leaves every table of the region bit for bit as
    a stream of acts on the same observations leaves it: the nudged phases, the eligibility
    phases and the lesson's settle enter no table."""
    acting = composed(seed=3, recall=True, working_memory_amplitude=0.0)
    taught = composed(seed=3, recall=True, working_memory_amplitude=0.0)
    rng = np.random.default_rng(2)
    for t in range(12):
        x = tokens(*rng.integers(INPUTS, size=3))
        acting.act(x, greedy=True)
        taught.lesson(x, rng.integers(ACTIONS, size=3))
        taught.act(x, greedy=True)
        if t % 4 == 3:  # a sampled act and its outcome, the eligibility phases on taught
            x, following = (tokens(*rng.integers(INPUTS, size=3)) for _ in range(2))
            acting.step(x)
            taught.step(x)
            taught.learn(np.array([1.0, -1.0, 0.5]), np.array([False, False, True]), following)
    # the tables agree to the solver's tolerance (each brain's phases start warm from its
    # own stream's state and settle to the same fixed points within it); a step of any rule
    # from a lesson's phase would be of the order of the rates, a thousand times larger
    for name, value in acting.learned_code.arrays().items():
        np.testing.assert_allclose(getattr(taught.learned_code, name), value, atol=1e-6)
    assert taught.learned_code.work["updates"] == acting.learned_code.work["updates"] == 15


def test_the_thresholds_share_reads_the_context_free_code():
    """A prediction that recruits every unit in context raises no threshold: the share
    counts the units the observation drives on its own."""
    brain = composed(seed=5)
    code = brain.learned_code
    brain.act(tokens(2, 7), greedy=True)  # the trace now carries a moment
    code.weights[:] = 1.0  # every unit predicted from any context
    code.install()
    brain.learner.brain = brain._installed(brain.brain)
    before = code.thresholds.copy()
    brain.act(tokens(4, 9), greedy=True)
    a = np.maximum(code.last["activation"], 0.0)
    free = code.last["free"]
    assert (a > 0).sum() > (free > 0).sum()  # the context recruited units
    share = (free > 0).sum(axis=0) - 2 * code.target
    repair = code.unexplained(code.read * code.last["sensory"], a) + code.unexplained(
        code.read * code.last["sensory"], free
    )
    expected = before + code.threshold_rate * (share - code.silence * repair)
    np.testing.assert_allclose(code.thresholds, np.maximum(expected, 0.0))
    recruited = (a > 0).any(axis=0) & ~(free > 0).any(axis=0)
    assert recruited.any() and (code.thresholds[recruited] <= before[recruited]).all()
    assert code.work["repairs"] == int(repair.sum())


def test_retune_changes_the_rates_and_the_top_down_weight_in_the_same_life():
    brain = composed(seed=7)
    brain.act(tokens(3), greedy=True)
    brain.learned_code.weights[:] = 0.1
    brain.retune(learned_code_top_down=2.0, learned_code_target=0.2)
    code = brain.learned_code
    assert code.top_down == 2.0 and code.target == 0.2
    synapses, i, j = brain._code_synapses["predictor"]
    np.testing.assert_allclose(brain.brain.efficacy[synapses], 2.0 * code.weights[j, i])
    assert brain.describe()["genes"]["learned_code_top_down"] == 2.0
    with pytest.raises(ValueError, match="target must lie in"):
        brain.retune(learned_code_target=0.0)
