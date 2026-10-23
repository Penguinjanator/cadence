"""Independent contracts for the optional zero-content empty-memory agreement."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

SPEC = importlib.util.spec_from_file_location(
    "attention_empty_heads", Path(__file__).with_name("heads.py"))
assert SPEC is not None and SPEC.loader is not None
heads = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = heads
SPEC.loader.exec_module(heads)
coupled = heads.coupled


def boundary(*, empty_anchor=False, slots=3):
    rng = np.random.default_rng(70)
    held = np.array([[True, False, True], [False, False, False], [False, True, False]])
    return coupled.Memory(
        np.array([0, 1]), np.array([2, 3, 4]),
        rng.normal(size=(3, slots, 2)), rng.normal(size=(3, slots, 3)),
        held[:, :slots], np.arange(slots, dtype=float), amplitude=0.4,
        temperature=0.7, null=0.0, empty_anchor=empty_anchor)


@pytest.mark.parametrize("slots", [0, 3])
def test_empty_stream_reads_the_zero_content_null_with_exact_restoring_force(slots):
    memory = boundary(empty_anchor=True, slots=slots)
    state = np.random.default_rng(3).normal(size=(3, 6))
    empty = ~memory.held.any(axis=1)
    p = memory.weights(state)
    assert np.array_equal(p[empty], np.zeros_like(p[empty]))
    force = memory.drive(state)
    np.testing.assert_array_equal(force[empty][:, memory.query], np.zeros((empty.sum(), 2)))
    np.testing.assert_array_equal(force[empty][:, memory.read],
                                  -memory.amplitude * state[empty][:, memory.read])
    expected = 0.5 * memory.amplitude * np.square(state[empty][:, memory.read]).sum(axis=1)
    np.testing.assert_array_equal(memory.energy(state)[empty], expected)
    legacy = replace(memory, empty_anchor=False)
    np.testing.assert_array_equal(legacy.drive(state)[empty], np.zeros((empty.sum(), 6)))
    np.testing.assert_array_equal(legacy.energy(state)[empty], np.zeros(empty.sum()))


@pytest.mark.parametrize("slots", [0, 3])
def test_mixed_and_zero_slot_energy_derivatives_and_force_jacobian(slots):
    memory = boundary(empty_anchor=True, slots=slots)
    state = np.random.default_rng(17).normal(size=(3, 6))
    epsilon = 1e-5
    columns = []
    for column in range(state.shape[1]):
        delta = np.zeros_like(state)
        delta[:, column] = epsilon
        derivative = (memory.energy(state + delta) - memory.energy(state - delta)) / (2 * epsilon)
        np.testing.assert_allclose(memory.drive(state)[:, column], -derivative, atol=2e-9)
        columns.append((memory.drive(state + delta) - memory.drive(state - delta)) / (2 * epsilon))
    jacobian = np.stack(columns, axis=-1)
    np.testing.assert_allclose(jacobian, jacobian.swapaxes(1, 2), atol=2e-9)
    empty = ~memory.held.any(axis=1)
    for neuron in memory.read:
        np.testing.assert_allclose(jacobian[empty, neuron, neuron], -memory.amplitude, atol=1e-10)


def test_nonempty_streams_are_bit_identical_and_empty_read_at_rest_stays_zero():
    legacy = boundary()
    anchored = replace(legacy, empty_anchor=True)
    state = np.random.default_rng(29).normal(size=(3, 6))
    occupied = legacy.held.any(axis=1)
    np.testing.assert_array_equal(anchored.weights(state), legacy.weights(state))
    np.testing.assert_array_equal(anchored.energy(state)[occupied], legacy.energy(state)[occupied])
    np.testing.assert_array_equal(anchored.drive(state)[occupied], legacy.drive(state)[occupied])
    state[:] = 0.0
    np.testing.assert_array_equal(anchored.drive(state)[~occupied], np.zeros((1, 6)))
    np.testing.assert_array_equal(anchored.energy(state)[~occupied], np.zeros(1))


@pytest.mark.parametrize("value", [0, 1, None, "true", [True], np.array(True)])
def test_anchor_rejects_nonboolean_values_before_composition(value):
    with pytest.raises(ValueError, match="empty_anchor must be boolean"):
        boundary(empty_anchor=value)
    for cls in (coupled.AttentionBrain, heads.MultiHeadBrain):
        with pytest.raises(ValueError, match="empty_anchor must be boolean"):
            cls.compose(3, 2, modules=(4,), empty_anchor=value)


@pytest.mark.parametrize("mode", ["linear", "competitive"])
def test_other_equations_retain_the_legacy_empty_boundary_and_reject_the_anchor(mode):
    memory = replace(boundary(), mode=mode)
    state = np.random.default_rng(36).normal(size=(3, 6))
    np.testing.assert_array_equal(memory.drive(state)[1], np.zeros(6))
    with pytest.raises(ValueError, match="only for agreement"):
        replace(memory, empty_anchor=True)
    with pytest.raises(ValueError, match="only for agreement"):
        heads.MultiHeadBrain.compose(3, 2, heads=2, modules=(4,), mode=mode, empty_anchor=True)


def test_row_reordering_parameter_copies_and_head_lesions_preserve_the_boundary_choice():
    memory = boundary(empty_anchor=np.bool_(True))
    assert memory.empty_anchor is True
    state = np.random.default_rng(71).normal(size=(3, 6))
    subset = memory.take([1, 0])
    assert subset.empty_anchor is True
    np.testing.assert_array_equal(subset.drive(state[[1, 0]]), memory.drive(state)[[1, 0]])
    np.testing.assert_array_equal(subset.energy(state[[1, 0]]), memory.energy(state)[[1, 0]])
    graph = coupled.Graph(cd.Connectome.from_synapses(6, pre=[], post=[]),
                          cd.learning_neuron_model(dt=0.1))
    graph.memory = subset
    for changed in (graph.with_parameters(efficacy=graph.efficacy),
                    graph.with_parameters(log_gain=graph.log_gain),
                    graph.with_parameters(bias=graph.bias)):
        assert changed.memory is subset and changed.memory.empty_anchor
    second = replace(memory, query=np.array([6, 7]), read=np.array([8, 9, 10]))
    multi = heads.MultiMemory((memory, second))
    assert multi.empty_anchor and multi.take([1]).empty_anchor
    lesioned = heads.MultiMemory(tuple(replace(m, amplitude=0.0) for m in multi.memories))
    np.testing.assert_array_equal(lesioned.drive(np.zeros((3, 11)) + 0.4), np.zeros((3, 11)))
    with pytest.raises(ValueError, match="selector settings"):
        heads.MultiMemory((memory, replace(second, empty_anchor=False)))


def credit_graph(empty_anchor=True, *, occupied=False):
    graph = coupled.Graph(cd.Connectome.from_synapses(
        4, pre=[0, 2, 1, 2, 2, 3], post=[2, 0, 2, 1, 3, 2], sign=[0.12] * 6),
        cd.learning_neuron_model(dt=0.1), bias=np.full(4, 0.1))
    graph.memory = coupled.Memory(
        np.array([0]), np.array([1]), np.zeros((1, 1, 1)), np.zeros((1, 1, 1)),
        np.full((1, 1), occupied, dtype=bool), np.zeros(1),
        amplitude=0.4, empty_anchor=empty_anchor)
    return graph


def test_empty_anchor_keeps_read_live_and_local_contrast_matches_answer_derivative():
    graph = credit_graph()
    config = cd.LearnerConfig(beta=1e-4, nudge="quadratic", free_steps=1000,
                             nudged_steps=1000, tolerance=1e-12, qualified=True, damping=0)
    learner = cd.Learner(graph, [2, 3], config)
    drive = np.array([[0.4, 0.0, 0.1, 0.2]])
    target = np.array([[0.0, 0.0, 0.2, 0.7]])
    free = learner.free(drive)
    assert free.activation[0, 1] > 0  # finite read agreement is not a clamp
    plus = learner.nudged(drive, free, target)
    minus = learner.nudged(drive, free, target, sign=-1.0)
    contrast, _ = learner.contrast(free, plus, minus)
    for first, second in ((0, 2), (1, 2), (2, 3)):
        pair = ((graph.connectome.pre == first) & (graph.connectome.post == second)) | (
            (graph.connectome.pre == second) & (graph.connectome.post == first))

        def loss(delta, pair=pair):
            efficacy = graph.efficacy.copy()
            efficacy[pair] += delta
            state = graph.with_parameters(efficacy=efficacy).equilibrate(
                drive, budget=1000, tolerance=1e-12)
            assert state.qualified.all()
            return float(0.5 * np.square(state.state.activation[:, [2, 3]]
                                         - target[:, [2, 3]]).sum())

        numerical = -(loss(1e-5) - loss(-1e-5)) / 2e-5
        np.testing.assert_allclose(contrast[pair], numerical, rtol=3e-4, atol=1e-8)


def test_nonempty_free_and_teaching_phases_preserve_the_original_exact_equations():
    drive = np.array([[0.4, 0.0, 0.1, 0.2]])
    labels = np.array([1])
    phases = []
    for flag in (False, True):
        graph = credit_graph(flag, occupied=True)
        learner = cd.Learner(graph, [2, 3], cd.LearnerConfig(
            qualified=True, free_steps=512, nudged_steps=512, tolerance=1e-10, damping=0))
        free = learner.free(drive)
        target = learner.targets(labels)
        plus = learner.nudged(drive, free, target)
        minus = learner.nudged(drive, free, target, sign=-1.0)
        phases.append((free, plus, minus, learner.contrast(free, plus, minus)))
    for first, second in zip(phases[0][:3], phases[1][:3], strict=True):
        np.testing.assert_array_equal(first.v, second.v)
        np.testing.assert_array_equal(first.activation, second.activation)
        assert first.steps == second.steps
    for first, second in zip(phases[0][3], phases[1][3], strict=True):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("count,copies", [(1, 1), (2, 1), (1, 2)])
def test_composition_propagates_anchor_to_every_immutable_head(count, copies):
    brain = heads.MultiHeadBrain.compose(
        3, 2, heads=count, port_copies=copies, modules=(4,), episodic=False,
        empty_anchor=True, recall_next=False)
    brain.stimulus([[1, 0, 0]])
    assert brain.empty_anchor and brain.brain.memory.empty_anchor
    assert all(member.empty_anchor for member in brain.brain.memory.memories)
    assert brain.brain.memory.keys.shape[1] == 0


def test_one_head_remains_exactly_the_single_head_prototype_with_the_same_flag():
    options = dict(modules=(4,), episodic=False, recall_next=False, empty_anchor=True)
    original = coupled.AttentionBrain.compose(3, 2, seed=2, **options)
    candidate = heads.MultiHeadBrain.compose(3, 2, seed=2, **options)
    for observation in ([[1, 0, 0]], [[0, 1, 0]]):
        np.testing.assert_array_equal(original.act(observation), candidate.act(observation))
        np.testing.assert_array_equal(original.basal_ganglia.state.v,
                                      candidate.basal_ganglia.state.v)
        assert original.brain.memory.empty_anchor and candidate.brain.memory.empty_anchor


@pytest.mark.parametrize("flag", [False, True])
def test_reset_discards_the_previous_trial_memory_before_first_action_refusal(flag):
    brain = heads.MultiHeadBrain.compose(
        3, 2, heads=2, modules=(4,), episodic=False, empty_anchor=flag)
    brain.act([[1, 0, 0]])
    brain.act([[0, 1, 0]])
    graph = brain.brain
    assert graph.memory is not None and graph.memory.held.any()
    assert brain.head_attended is not None
    brain.reset()
    assert graph.memory is None
    assert brain.head_attended is None
    assert brain.recall.slots == 0 and brain.basal_ganglia.state is None
    brain.learner.config = replace(brain.learner.config, free_steps=0, tolerance=1e-12)
    with pytest.raises(RuntimeError, match="settle"):
        brain.act([[0, 0, 1]])
    assert graph.memory is None
    assert brain.head_attended is None
    assert brain.recall.slots == 0 and brain.basal_ganglia.state is None
    assert not brain.last_settlement["qualified"]


def test_refused_action_and_invalid_reset_preserve_the_current_boundary_and_life():
    brain = heads.MultiHeadBrain.compose(
        3, 2, heads=2, modules=(4,), episodic=False, empty_anchor=True)
    brain.act([[1, 0, 0]])
    brain.act([[0, 1, 0]])
    graph, memory, attended = brain.brain, brain.brain.memory, brain.head_attended
    free = brain.basal_ganglia.state
    trace = brain.working_memory.trace.copy()
    recalled = brain.recall.snapshot()
    reading = brain.recall.reading()
    with pytest.raises(ValueError, match="keep names"):
        brain.reset(keep=("invalid",))
    assert graph.memory is memory and brain.head_attended is attended
    brain.learner.config = replace(brain.learner.config, free_steps=0, tolerance=1e-12)
    with pytest.raises(RuntimeError, match="settle"):
        brain.act([[0, 0, 1]])
    assert graph.memory is memory and brain.head_attended is attended
    assert brain.basal_ganglia.state is free
    assert brain.recall.reading()[0] is reading[0] and brain.recall.reading()[1] is reading[1]
    np.testing.assert_array_equal(brain.working_memory.trace, trace)
    for name, value in recalled.items():
        np.testing.assert_array_equal(getattr(brain.recall, name), value)


@pytest.mark.parametrize("refused", [False, True])
def test_saved_teaching_retains_anchoring_through_row_take_and_parameter_replacement(
    monkeypatch, refused,
):
    brain = heads.MultiHeadBrain.compose(
        3, 2, heads=2, modules=(4,), episodic=False, recall_next=False,
        empty_anchor=True, learning_free_steps=256, learning_nudged_steps=256)
    brain.act([[1, 0, 0], [0, 1, 0]])
    drive = brain.stimulus([[0, 0, 1], [0, 0, 1]]).copy()
    graph, memory = brain.brain, brain.brain.memory
    phases = []
    original = brain.learner._qualified_phase

    def observed(*args, **kwargs):
        phases.append(brain.brain.memory)
        return original(*args, **kwargs)

    monkeypatch.setattr(brain.learner, "_qualified_phase", observed)
    if refused:
        brain.learner.config = replace(brain.learner.config, free_steps=0, tolerance=1e-12)
        with pytest.raises(cd.LearningPhaseError, match="free"):
            brain.teach_saved(drive, np.array([0, 1]), [1], memory)
        assert brain.brain is graph
    else:
        _, report = brain.teach_saved(drive, np.array([0, 1]), [1], memory)
        assert report["accepted_presentations"] == 1 and len(phases) == 3
        assert brain.brain is not graph
    assert phases and all(m is phases[0] and m.empty_anchor and m.streams == 1 for m in phases)
    assert graph.memory is brain.brain.memory is memory
    assert memory.empty_anchor and memory.streams == 2
