"""Sampling and local eligibility must use the same scalar or per-slot policy."""

import numpy as np
import pytest

import cadence as cd


def test_each_slot_temperature_reads_the_same_settled_motor_state():
    brain = cd.Brain.compose(2, 9, modules=(8,), slots=(2, 3, 4), seed=4)
    brain.act([[0.2, 0.8]], greedy=True)
    agent = brain.basal_ganglia
    state = agent.state
    temperatures = (0.2, 0.6, 1.0)
    probabilities = agent.probabilities(state, temperatures)
    assert probabilities.shape == (1, 3, 4)
    for index, (start, size, temperature) in enumerate(
        zip(brain.learner.slot_offsets, brain.learner.slot_sizes, temperatures, strict=True)
    ):
        motor = state.activation[0, brain.motor_index[start : start + size]]
        weights = np.exp((motor - motor.max()) / temperature)
        np.testing.assert_allclose(probabilities[0, index, :size], weights / weights.sum())
        assert not probabilities[0, index, size:].any()


@pytest.mark.parametrize("temperature", [0.6, (0.2, 0.6, 1.0)])
def test_taken_action_keeps_the_actual_sampling_policy_in_its_nudge(temperature, monkeypatch):
    brain = cd.Brain.compose(2, 9, modules=(8,), slots=(2, 3, 4), seed=4)
    x = np.array([[0.2, 0.8]])
    brain.act(x, greedy=True)
    agent = brain.basal_ganglia
    actual = np.broadcast_to(temperature, (3,))
    expected_temperature = np.full(agent.n, brain.learner.config.temperature)
    expected_mask = brain.learner.output_mask.copy()
    scale = min(brain.learner.config.temperature, min(actual))
    for start, size, value in zip(
        brain.learner.slot_offsets, brain.learner.slot_sizes, actual, strict=True
    ):
        indices = brain.motor_index[start : start + size]
        expected_temperature[indices] = value
        expected_mask[indices] = scale / value
    observed = []
    settle = brain.brain.settle_batch

    def witness(*args, **kwargs):
        nudge = kwargs.get("nudge")
        if nudge is not None:
            observed.append(nudge)
        return settle(*args, **kwargs)

    monkeypatch.setattr(brain.brain, "settle_batch", witness)
    action = brain.act(x, temperature=temperature)
    assert len(observed) == 2 and agent._pending is not None
    for nudge in observed:
        np.testing.assert_allclose(nudge.mask, expected_mask)
        actual_temperature = np.broadcast_to(nudge.softmax_temperature, (agent.n,))
        np.testing.assert_allclose(
            actual_temperature[brain.motor_index], expected_temperature[brain.motor_index]
        )
        np.testing.assert_array_equal(nudge.target, brain.learner.targets(action))


@pytest.mark.parametrize(
    "temperature",
    [(0.2,), (0.2, 0.4, 0.6), (0.2, 0.0), (0.2, np.inf), (True, False), ((0.2, 0.4),)],
)
def test_invalid_slot_temperatures_preserve_pending_action_and_randomness(temperature):
    brain = cd.Brain.compose(2, 6, modules=(8,), slots=2, seed=2)
    x = np.array([[0.2, 0.8]])
    brain.act(x)
    agent = brain.basal_ganglia
    before, pending, state = agent.rng.bit_generator.state, agent._pending, agent.state
    with pytest.raises(ValueError, match="temperature"):
        brain.act(x, temperature=temperature)
    assert agent.rng.bit_generator.state == before
    assert agent._pending is pending and agent.state is state


def test_population_code_samples_and_credits_per_dimension_temperatures(monkeypatch):
    bins = cd.Bins(dims=2, size=3)
    connectome = cd.layered(2, 4, 6, density=1.0, seed=0)
    graph = cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0))
    learner = cd.Learner(graph, connectome.populations["output"])
    agent = cd.ActorCritic(learner, connectome.populations["hidden"], population=bins)
    drive = graph.stimulus_levels(np.full((1, connectome.n), 0.25))
    agent.act(drive, greedy=True)
    temperatures = np.array([0.2, 0.8])
    probabilities = agent.probabilities(agent.state, temperatures)
    for slot, value in enumerate(temperatures):
        np.testing.assert_array_equal(
            probabilities[:, slot], agent.probabilities(agent.state, value)[:, slot]
        )
    nudges = []
    settle = graph.settle_batch

    def witness(*args, **kwargs):
        if kwargs.get("nudge") is not None:
            nudges.append(kwargs["nudge"])
        return settle(*args, **kwargs)

    monkeypatch.setattr(graph, "settle_batch", witness)
    action = agent.act(drive, temperature=temperatures)
    assert action.shape == (1, 2) and len(nudges) == 2
    for nudge in nudges:
        np.testing.assert_array_equal(
            nudge.softmax_temperature[learner.output_index], np.repeat(temperatures, 3)
        )
