"""Independent ridge reference, split custody and non-geometric transfer gates."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "attention_representation_tests", Path(__file__).with_name("representation.py")
)
assert SPEC is not None and SPEC.loader is not None
representation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = representation
SPEC.loader.exec_module(representation)
task = representation.task


def code(rows: np.ndarray) -> np.ndarray:
    # Both names' values occur during training; their held-out combinations do not.
    # Rotate their additive one-hot coordinates and add an irrelevant offset.
    transform, _ = np.linalg.qr(np.random.default_rng(8).normal(size=(8, 8)))
    values = np.eye(4)[task.targets(rows)].reshape(len(rows), 8)
    return values @ transform + np.linspace(10, 20, 8)


def fixture() -> dict:
    train, held = task.panel("train"), task.panel("held_out")
    raw_train, raw_held = task.events(train)[1], task.events(held)[1]
    # A frozen state can encode the current query, without the witnessed values.
    projection = np.random.default_rng(19).normal(size=(task.INPUTS, 8))
    return {
        "train_features": code(train), "held_features": code(held),
        "frozen_train": raw_train @ projection,
        "frozen_held": raw_held @ projection,
        "swapped_features": code(task.swapped(held)),
        "erased_features": {"erase_A": np.zeros((len(held), 8)),
                            "erase_B": np.zeros((len(held), 8))},
    }


def test_ridge_matches_independent_augmented_least_squares_reference() -> None:
    x = np.random.default_rng(2).normal(size=(48, 7)) + 100
    targets = task.targets(task.panel("train"))
    probe = representation.fit(x, targets)
    y = np.eye(4)[targets].reshape(48, 8)
    centered = x - x.mean(axis=0)
    augmented_x = np.vstack([centered, np.sqrt(0.01) * np.eye(x.shape[1])])
    augmented_y = np.vstack([y - y.mean(axis=0), np.zeros((x.shape[1], 8))])
    expected = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    np.testing.assert_allclose(probe.weights.reshape(7, 8), expected, atol=1e-12)
    np.testing.assert_allclose(probe.feature_mean, x.mean(axis=0))
    np.testing.assert_allclose(probe.target_mean.reshape(8), y.mean(axis=0))


def test_fitted_probe_owns_readonly_arrays_and_prediction_never_updates_them() -> None:
    rows = task.panel("train")
    x = code(rows)
    probe = representation.fit(x, task.targets(rows))
    before = probe.record()
    x[:] = 1234
    for array in (probe.weights, probe.feature_mean, probe.target_mean):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0
    with pytest.raises(FrozenInstanceError):
        probe.training_rows = 99
    prediction = representation.predict(probe, code(task.panel("held_out")))
    np.testing.assert_array_equal(prediction, task.targets(task.panel("held_out")))
    assert probe.record() == before


def test_additive_representation_transfers_to_unseen_combinations_and_swapped_facts() -> None:
    data = fixture()
    result = representation.analyze(**data)
    assert result["passed"] and all(result["gates"].values())
    assert result["learned"]["probe"]["training_rows"] == 48
    for name in ("held_out", "swapped"):
        reading = result["learned"][name]
        assert len(reading["answers"]) == len(reading["decision_scores"]) == 16
        assert np.asarray(reading["decision_scores"]).shape == (16, 2, 4)
        assert reading["scores"]["exact_pair_accuracy"] == 1.0
    assert set(result["erasure_losses"]) == {"erase_A", "erase_B"}
    assert min(result["erasure_losses"].values()) >= 0.20
    assert min(result["held_out_gains"].values()) >= 0.20
    # The raw control has only cue/order, never dormant exposed-value inputs.
    assert not np.asarray(result["controls"]["raw_query"]["train"]["features"])[:, 5:].any()


def test_geometric_separation_without_held_out_content_transfer_fails() -> None:
    data = fixture()
    train, held = task.panel("train"), task.panel("held_out")
    # A large, distinct orthogonal state for each case can memorize all training
    # labels. Held-out cases occupy unused coordinates: distances do not transfer.
    full = np.eye(len(task.CASES)) * 1000
    data["train_features"] = full[~task.HELD]
    data["held_features"] = full[task.HELD]
    data["swapped_features"] = np.roll(full[task.HELD], 1, axis=0)
    data["erased_features"] = np.zeros((len(held), len(task.CASES)))
    result = representation.analyze(**data)
    assert len(train) == 48
    assert result["learned"]["train"]["scores"]["exact_pair_accuracy"] == 1.0
    assert not result["passed"] and not result["gates"]["held_out_transfer"]


def test_erasure_and_newborn_comparators_cannot_be_omitted_or_refitted() -> None:
    data = fixture()
    data["erased_features"] = data["held_features"].copy()
    no_erasure_loss = representation.analyze(**data)
    assert not no_erasure_loss["gates"]["relevant_erasure"]
    assert not no_erasure_loss["passed"]
    data = fixture()
    data["frozen_train"] = data["train_features"].copy()
    data["frozen_held"] = data["held_features"].copy()
    no_newborn_gain = representation.analyze(**data)
    assert not no_newborn_gain["gates"]["gain_over_frozen_newborn"]
    assert not no_newborn_gain["passed"]


@pytest.mark.parametrize("bad", [None, np.zeros((47, 8)), np.zeros((48, 0)),
                                  np.full((48, 8), np.nan), np.full((48, 8), np.inf)])
def test_analysis_never_drops_missing_or_nonfinite_training_rows(bad) -> None:
    data = fixture()
    data["train_features"] = bad
    with pytest.raises(ValueError):
        representation.analyze(**data)


@pytest.mark.parametrize("change", ["status", "refusal", "unanswered", "reorder", "missing"])
def test_recorded_refusal_or_wrong_panel_order_cannot_pass(change: str) -> None:
    data = fixture()
    rows = task.panel("held_out")
    recorded = {"association": data["held_features"], "rows": rows.tolist(),
                "status": "completed", "answers": task.targets(rows).tolist(),
                "work": {"action_refusals": 0}}
    if change == "status":
        recorded["status"] = "wall_budget"
    elif change == "refusal":
        recorded["work"]["action_refusals"] = 1
    elif change == "unanswered":
        recorded["answers"][0][1] = -1
    elif change == "reorder":
        recorded["rows"] = rows[::-1].tolist()
    elif change == "missing":
        recorded["association"] = None
    data["held_features"] = recorded
    with pytest.raises(ValueError):
        representation.analyze(**data)


@pytest.mark.parametrize("bad", [np.zeros((47, 2), dtype=int), np.full((48, 2), -1),
                                  np.full((48, 2), 4), np.zeros((48, 2), dtype=float)])
def test_fit_rejects_missing_refused_or_invalid_targets(bad) -> None:
    with pytest.raises(ValueError):
        representation.fit(np.zeros((48, 8)), bad)


def test_width_mismatch_empty_erasure_and_refused_frozen_control_are_rejected() -> None:
    data = fixture()
    data["swapped_features"] = np.zeros((16, 9))
    with pytest.raises(ValueError):
        representation.analyze(**data)
    data = fixture()
    data["erased_features"] = {}
    with pytest.raises(ValueError):
        representation.analyze(**data)
    data = fixture()
    data["frozen_held"] = {"association": data["frozen_held"], "status": "refused"}
    with pytest.raises(ValueError):
        representation.analyze(**data)
