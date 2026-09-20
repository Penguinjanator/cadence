"""Independent dense planning and sequential inference oracles."""

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cadence.actor import BodyModel, EquilibriumActor

F = np.array([[1.0, 1.0], [0.0, 1.0]])
G = np.array([1.0, 1.0])


def dense_plan(actor, boundary, horizon, goal):
    # Construct independent rectangular least-squares factors, not Schur blocks.
    rows, values = [], []
    for t in range(horizon):
        for k in range(2):
            row = np.zeros(3 * horizon)
            row[3 * t + k] = 1
            row[3 * t + 2] = -actor.model.G[k]
            value = actor.model.F[k] @ boundary if t == 0 else 0.0
            if t:
                row[3 * (t - 1) : 3 * (t - 1) + 2] = -actor.model.F[k]
            rows.append(np.sqrt(actor.q) * row)
            values.append(np.sqrt(actor.q) * value)
        row = np.zeros(3 * horizon)
        row[3 * t + 2] = np.sqrt(actor.effort)
        rows.append(row)
        values.append(0.0)
    for k, value in enumerate([goal, 0.0]):
        row = np.zeros(3 * horizon)
        row[3 * (horizon - 1) + k] = 1
        rows.append(row)
        values.append(value)
    return np.linalg.lstsq(np.array(rows), np.array(values), rcond=None)[0].reshape(horizon, 3)


@pytest.mark.parametrize("variance", [0.0, 0.0004, 0.0025])
def test_one_record_message_matches_all_prefix_filter_and_dense_future(variance):
    actor = EquilibriumActor(BodyModel(F, G, variance), 0.6)
    rng = np.random.default_rng(139)
    mean, covariance = np.zeros(2), 100 * np.eye(2)
    state = np.array([-0.15, 0.2])
    for identifier in range(24):
        action = None if identifier == 0 else float(rng.uniform(-0.3, 0.3))
        if action is not None:
            state = F @ state + G * action
            mean = F @ mean + G * action
            covariance = F @ covariance @ F.T + actor.model.Q
        observed = state[0] + rng.normal() * np.sqrt(variance)
        innovation = covariance[0, 0] + variance
        gain = covariance[:, 0] / innovation
        mean = mean + gain * (observed - mean[0])
        transform = np.eye(2) - np.outer(gain, [1.0, 0.0])
        covariance = transform @ covariance @ transform.T + variance * np.outer(gain, gain)
        actor.admit(observed, identifier=identifier, executed_action=action)
        past = actor.readback()
        assert_allclose(past.mean, mean, atol=5e-10)
        assert_allclose(past.covariance, covariance, atol=1e-11)
        snapshot = actor.snapshot()
        for goal in [-0.6, 0.6]:
            plan = actor.plan(goal=goal)
            expected = dense_plan(actor, mean, 3, goal)
            assert_allclose(plan.states, expected[:, :2], atol=5e-9)
            assert_allclose(plan.actions, expected[:, 2], atol=5e-9)
            assert plan.residual < 1e-8 and plan.minimum_pivot > 0
            assert plan.observation_id == identifier
            assert plan.energy >= 0
        assert snapshot == actor.snapshot()
        assert actor.numeric_persistent_bytes() == 232
    assert actor.readback().marginalizations == 23


def test_goal_changes_private_plan_not_measured_past_and_shadow_is_detached():
    actor = EquilibriumActor(BodyModel(F, G), 0.6)
    actor.admit(-0.1, identifier=0)
    actor.admit(0.1, identifier=1, executed_action=0.0)
    before = actor.snapshot()
    positive, negative = actor.plan(), actor.plan(goal=-0.6)
    assert abs(positive.action - negative.action) > 0.3
    assert_allclose(positive.boundary, negative.boundary, atol=0)
    assert_allclose(positive.covariance, negative.covariance, atol=0)
    assert actor.snapshot() == before
    positive.boundary[:] = 999
    positive.states[:] = 999
    actor.readback().mean[:] = 999
    snapshot = actor.snapshot()
    snapshot["precision"][0][0] = 999
    assert actor.snapshot() == before
    actor.goal = -0.6
    assert_allclose(actor.readback().mean, negative.boundary, atol=0)


def test_duplicate_evidence_and_stale_model_refused_atomically():
    actor = EquilibriumActor(BodyModel(F, G), 0.6)
    actor.admit(0.0, identifier=0)
    before = actor.snapshot()
    for kwargs in [
        dict(position=1.0, identifier=0),
        dict(position=1.0, identifier=2, executed_action=0.0),
        dict(position=np.nan, identifier=1, executed_action=0.0),
    ]:
        with pytest.raises(ValueError):
            actor.admit(**kwargs)
        assert actor.snapshot() == before
    # Mutation is normally blocked by read-only arrays. Deliberately bypass to
    # exercise binding; rejection must not mutate existing prefix afterwards.
    actor.model.F.flags.writeable = True
    actor.model.F[0, 0] += 0.1
    precision, information = actor._precision.copy(), actor._information.copy()
    for operation in [
        actor.readback,
        actor.plan,
        lambda: actor.admit(0.2, identifier=1, executed_action=0.0),
    ]:
        with pytest.raises(ValueError, match="different model"):
            operation()
        assert_allclose(actor._precision, precision, atol=0)
        assert_allclose(actor._information, information, atol=0)
        assert actor._next == 1


def test_checkpoint_restores_observer_planning_and_continuation(tmp_path):
    actor = EquilibriumActor(BodyModel(F, G, 0.0025), 0.6)
    for i in range(5):
        actor.admit(0.1 * i, identifier=i, executed_action=None if i == 0 else 0.02)
    restored = EquilibriumActor.load(actor.save(tmp_path / "actor"))
    assert restored.snapshot() == actor.snapshot()
    assert_allclose(restored.plan().actions, actor.plan().actions, atol=0)
    for current in [actor, restored]:
        current.admit(0.48, identifier=5, executed_action=-0.03)
    assert restored.snapshot() == actor.snapshot()
    corrupt = copy.deepcopy(actor.snapshot())
    corrupt["model"]["F"][0][0] += 0.1
    with pytest.raises(ValueError, match="binding"):
        EquilibriumActor.restore(corrupt)
    corrupt = copy.deepcopy(actor.snapshot())
    corrupt["record"]["id"] -= 1
    with pytest.raises(ValueError, match="counters"):
        EquilibriumActor.restore(corrupt)


def test_replanning_responds_to_actual_hidden_shock():
    def cost(feedback, shock):
        actor = EquilibriumActor(BodyModel(F, G), 0.6)
        actor.admit(0.0, identifier=0)
        actor.admit(0.0, identifier=1, executed_action=0.0)
        initial = actor.plan()
        state, effort = np.zeros(2), 0.0
        for t in range(3):
            action = actor.plan(horizon=3 - t).action if feedback else initial.actions[t]
            state = F @ state + G * (action + (shock if t == 0 else 0.0))
            effort += 0.02 * action**2
            actor.admit(state[0], identifier=t + 2, executed_action=float(action))
        return float((state[0] - 0.6) ** 2 + state[1] ** 2 + effort)

    assert cost(True, 0.25) < 0.03
    assert cost(True, 0.25) < 0.2 * cost(False, 0.25)


def test_invalid_initial_models_and_parameters():
    for variance in [-1.0, np.nan]:
        with pytest.raises(ValueError):
            BodyModel(F, G, variance)
    with pytest.raises(ValueError):
        BodyModel(F, G, process_covariance=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        EquilibriumActor(BodyModel(F, G), 0.6, horizon=True)
    actor = EquilibriumActor(BodyModel(F, G), 0.6)
    with pytest.raises(ValueError):
        actor.plan()
    with pytest.raises(ValueError):
        actor.admit(0.0, identifier=False)


def test_frozen_observer_and_actor_kernel_parity():
    import json
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures/actor_reference_v1.json").read_text())
    for case in fixture["cases"]:
        actor = EquilibriumActor(BodyModel(F, G, case["variance"]), 0.6)
        for identifier, (position, action) in enumerate(
            zip(case["positions"], case["actions"], strict=True)
        ):
            actor.admit(position, identifier=identifier, executed_action=action)
        assert_allclose(actor.readback().mean, case["mean"], atol=1e-11)
        assert_allclose(actor.readback().covariance, case["covariance"], atol=1e-12)
        assert_allclose(actor.plan().states, case["states"], atol=1e-10)
        assert_allclose(actor.plan().actions, case["planned_actions"], atol=1e-10)
