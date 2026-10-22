"""Independent custody and decision checks for the bounded acquisition repair."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "attention_repair_test_subject", HERE / "repair_admission.py"
)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repair
SPEC.loader.exec_module(repair)


class TraceBrain:
    """An observable stream whose act changes both its trace and stimulus buffer."""

    def __init__(self, *, correct: bool = False, refuse: str | None = None) -> None:
        self.correct = correct
        self.refuse = refuse
        self.resets = 0
        self.updates = 0
        self.last_settlement: dict[str, Any] | None = None
        self.lessons: list[tuple[np.ndarray, np.ndarray]] = []
        self.prequery: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.learner = self
        self.context: np.ndarray | None = None
        self.buffer: np.ndarray | None = None
        self.ticks = 0
        self.query_reads = 0

    def reset(self) -> None:
        self.resets += 1
        self.context = None
        self.ticks = 0

    def stimulus(self, observations: np.ndarray) -> np.ndarray:
        batch = len(observations)
        if self.context is None:
            self.context = np.zeros(batch)
        if self.buffer is None:
            self.buffer = np.empty((batch, repair.sr.INPUTS + 1))
        self.buffer[:, :-1] = observations
        self.buffer[:, -1] = self.context
        if observations[:, 5:7].any():
            self.query_reads += 1
        return self.buffer

    def act(self, observations: np.ndarray, *, greedy: bool) -> np.ndarray:
        assert greedy
        drive = self.stimulus(observations)
        query = bool(observations[:, 5:7].any())
        refusal = self.refuse == ("query" if query else "prefix")
        self.last_settlement = {
            "qualified": not refusal, "steps": 1,
            "residual_checks": 1, "max_residual": 1.0 if refusal else 0.0,
        }
        if refusal:
            raise RuntimeError("deliberately refused component solve")
        if query:
            a = observations[:, 8].astype(np.int64)
            b = observations[:, 10].astype(np.int64)
            target = np.where(observations[:, 5] > 0, a, b)
            wrong = np.zeros(len(target), dtype=bool)
            if not self.correct:
                wrong[::3] = True
            self.prequery.append((drive.copy(), target.copy(), wrong.copy()))
            answer = target ^ wrong
        else:
            answer = np.zeros(len(observations), dtype=np.int64)
        self.ticks += 1
        assert self.context is not None and self.buffer is not None
        self.context += self.ticks * (observations @ np.arange(1, repair.sr.INPUTS + 1))
        # A caller retaining an alias, or asking for the stimulus after the act,
        # now sees the wrong event. The lesson must own a copy from before this.
        self.buffer.fill(-1234)
        return answer

    def step(self, drive: np.ndarray, target: np.ndarray) -> tuple[None, dict[str, float]]:
        self.lessons.append((drive.copy(), target.copy()))
        self.updates += 1
        return None, {
            "attempted_presentations": float(len(target)),
            "accepted_presentations": float(len(target)),
            "total_steps": 0.0, "total_row_sweeps": 0.0,
            "total_residual_checks": 0.0, "total_row_residual_checks": 0.0,
        }


def schedule() -> np.ndarray:
    return np.stack((np.arange(16)[::-1], np.roll(np.arange(16), 5), np.arange(16)))


@pytest.mark.parametrize("policy", ["every", "wrong"])
def test_training_teaches_selected_rows_of_the_complete_saved_prequery_drive(policy: str) -> None:
    brain = TraceBrain()
    result = repair.train(brain, schedule(), policy, deadline=float("inf"))
    assert result["completed"] and result["rounds"] == 3 and result["stop"] is None
    assert result["wrong_counts"] == [6, 6, 6]
    assert brain.resets == brain.updates == len(brain.lessons) == 3
    assert brain.query_reads == 6  # capture, then act; never a third read for teaching
    for (actual_drive, actual_target), (before, target, wrong) in zip(
        brain.lessons, brain.prequery, strict=True
    ):
        selected = np.ones(16, dtype=bool) if policy == "every" else wrong
        np.testing.assert_array_equal(actual_drive, before[selected])
        np.testing.assert_array_equal(actual_target, target[selected])
        assert (actual_drive[:, -1] > 0).all()  # all selected rows retain prefix context
    assert result["work"]["accepted_presentations"] == (48 if policy == "every" else 18)


@pytest.mark.parametrize("policy, expected_updates", [("wrong", 0), ("every", 3)])
def test_no_wrong_rounds_skip_only_the_wrong_policy_lesson(
    policy: str, expected_updates: int
) -> None:
    brain = TraceBrain(correct=True)
    result = repair.train(brain, schedule(), policy, deadline=float("inf"))
    assert result["completed"] and result["wrong_counts"] == [0, 0, 0]
    assert brain.updates == len(brain.lessons) == expected_updates
    assert result["work"]["lessons"] == expected_updates
    assert result["work"]["accepted_presentations"] == 16 * expected_updates
    assert result["work"]["actions"] == 12


@pytest.mark.parametrize("refuse", ["prefix", "query"])
def test_refused_answer_never_becomes_a_lesson_or_completed_round(refuse: str) -> None:
    brain = TraceBrain(refuse=refuse)
    result = repair.train(brain, schedule(), "wrong", deadline=float("inf"))
    assert not result["completed"] and result["rounds"] == 0
    assert result["stop"] == "refused_" + refuse
    assert result["work"]["action_refusals"] == 1
    assert brain.updates == 0 and result["work"]["lessons"] == 0


def test_expired_budget_and_invalid_policy_do_not_start_a_trial() -> None:
    brain = TraceBrain()
    result = repair.train(brain, schedule(), "wrong", deadline=-float("inf"))
    assert not result["completed"] and result["stop"] == "wall_budget"
    assert result["rounds"] == brain.resets == brain.updates == 0
    with pytest.raises(ValueError, match="policy"):
        repair.train(brain, schedule(), "unknown", deadline=float("inf"))
    assert brain.resets == 0


@pytest.fixture(autouse=True)
def frozen_small_protocol(tmp_path, monkeypatch) -> None:
    declared = json.loads((HERE / "repair-protocol.json").read_text())
    declared["rounds"] = 3
    frozen = tmp_path / "repair-protocol.json"
    frozen.write_text(json.dumps(declared))
    monkeypatch.setattr(repair, "PROTOCOL", frozen)


def protocol() -> dict[str, Any]:
    return json.loads(repair.PROTOCOL.read_text())


def evaluation(*, answered: bool = True) -> dict[str, Any]:
    answers = repair.sr.FACTS[:, :2].tolist() if answered else [[-1, -1]] * 8
    return {
        "completed": answered, "original_preserved": True,
        "branches": {
            name: {"answers": deepcopy(answers), "scores": repair.sr.scores(answers)}
            for name in ("intact", "erased")
        },
    }


def result(recipe: dict[str, Any], seed: int, policy: str, *, passed: bool) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence([31014, seed]))
    order = np.stack([rng.permutation(len(repair.sr.CASES)) for _ in range(protocol()["rounds"])])
    return {
        "recipe": recipe["name"], "seed": seed, "policy": policy,
        "resolved": {"genes": deepcopy(recipe["brain"]), "layout": {"inputs": 11}},
        "neuron_model": {"dt": recipe["dt"]},
        "initial_parameters": f"{seed:064x}",
        "schedule": {
            "cases": repair.sr.CASES.tolist(),
            "order_sha256": repair.canonical_sha256(order),
        },
        "initial": evaluation(), "evaluation": evaluation(),
        "training": {"completed": passed, "rounds": 3 if passed else 2},
        "passed": passed,
        "uniform_random": {
            "answers": [[0, 0]] * 8, "scores": repair.sr.scores([[0, 0]] * 8),
            "expected_accuracy": 0.5,
        },
    }


def body(stage: str = "development", selected: int | None = 0) -> dict[str, Any]:
    declared = protocol()
    recipes = declared["recipes"]
    chosen = None if selected is None else recipes[selected]["name"]
    if stage == "development":
        used = recipes if selected is None else recipes[:selected + 1]
        runs = [
            result(recipe, seed, policy, passed=recipe["name"] == chosen)
            for recipe in used for seed in declared["development_seeds"]
            for policy in ("every", "wrong")
        ]
    else:
        assert selected is not None
        runs = [result(recipes[selected], seed, "wrong", passed=True)
                for seed in declared["confirmation_seeds"]]
    return {
        "stage": stage, "protocol": declared, "runs": runs,
        "selected_recipe": chosen, "passed": chosen is not None,
    }


def test_trial_matches_initial_parameters_settings_and_schedule_across_policies(
    monkeypatch,
) -> None:
    declared = protocol()
    recipe = declared["recipes"][0]
    calls = []

    def capture(brain, order, policy, *, deadline, neutral_events):
        calls.append((order.copy(), policy, neutral_events))
        assert brain.brain.neuron_model.dt == recipe["dt"]
        assert brain.learner.config.qualified
        assert brain.learner.config.nudged_steps == brain.learner.config.free_steps == 1024
        return {"completed": True, "rounds": 3}

    monkeypatch.setattr(repair, "train", capture)
    monkeypatch.setattr(repair.sr, "evaluate", lambda *args, **kwargs: evaluation())
    runs = [repair.trial(recipe, 1, policy, declared, float("inf"), None)
            for policy in ("every", "wrong")]
    assert [call[1] for call in calls] == ["every", "wrong"]
    assert [call[2] for call in calls] == [declared["neutral_events"]] * 2
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    assert calls[0][0].shape == (3, 16)
    for key in ("initial_parameters", "resolved", "neuron_model", "schedule", "uniform_random"):
        assert runs[0][key] == runs[1][key]


@pytest.mark.parametrize("stage, selected", [
    ("development", 0), ("development", 1), ("development", None),
    ("confirmation", 0), ("confirmation", 1),
])
def test_recomputed_check_accepts_complete_declared_run_sets(
    stage: str, selected: int | None,
) -> None:
    assert repair.check(body(stage, selected)) is None


@pytest.mark.parametrize("branch", ["intact", "erased"])
def test_gate_rejects_unrun_answers_even_with_optimistic_completion(branch: str) -> None:
    measured = evaluation()
    measured["branches"][branch] = evaluation(answered=False)["branches"][branch]
    measured["completed"] = True
    assert not repair.gate({"completed": True, "rounds": 3}, measured, 0.95)


@pytest.mark.parametrize("field", ["training", "evaluation", "passed", "scores", "answers"])
def test_recomputed_check_refuses_optimistic_or_inconsistent_outcomes(field: str) -> None:
    measured = body()
    run = measured["runs"][1]
    if field == "training":
        run["training"]["rounds"] = 2
    elif field == "evaluation":
        run["evaluation"]["completed"] = False
    elif field == "passed":
        run["passed"] = False
    elif field == "scores":
        run["evaluation"]["branches"]["intact"]["scores"]["accuracy"] = 0.0
    else:
        run["evaluation"]["branches"]["intact"]["answers"][0][0] ^= 1
    assert repair.check(measured) is not None


@pytest.mark.parametrize("change", [
    "empty", "missing_every", "duplicate_every", "missing_wrong", "duplicate_wrong",
    "unknown_policy", "unknown_recipe", "skipped_primary", "continued_after_selection",
])
def test_development_check_requires_each_declared_founder_and_matched_arm_once(change: str) -> None:
    measured = body()
    if change == "empty":
        measured["runs"] = []
        measured["selected_recipe"], measured["passed"] = None, False
    elif change == "missing_every":
        measured["runs"] = [run for run in measured["runs"] if run["policy"] == "wrong"]
    elif change == "duplicate_every":
        measured["runs"].insert(0, deepcopy(measured["runs"][0]))
    elif change in ("missing_wrong", "duplicate_wrong"):
        if change == "missing_wrong":
            del measured["runs"][1]
        else:
            measured["runs"].insert(1, deepcopy(measured["runs"][1]))
        measured["selected_recipe"], measured["passed"] = None, False
    elif change in ("unknown_policy", "unknown_recipe"):
        extra = deepcopy(measured["runs"][0])
        extra["policy" if change == "unknown_policy" else "recipe"] = "undeclared"
        measured["runs"].append(extra)
    elif change == "skipped_primary":
        measured = body(selected=1)
        measured["runs"] = measured["runs"][4:]
    else:
        measured["runs"] += body(selected=1)["runs"][4:]
    assert repair.check(measured) is not None


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_arm", "wrong_recipe"])
def test_confirmation_check_rejects_missing_or_duplicate_founders_and_other_arms(
    change: str,
) -> None:
    measured = body("confirmation")
    if change == "missing":
        measured["runs"].pop()
    elif change == "duplicate":
        measured["runs"][1] = deepcopy(measured["runs"][0])
    elif change == "wrong_arm":
        measured["runs"][0]["policy"] = "every"
    else:
        measured["runs"][0]["recipe"] = "undeclared"
    assert repair.check(measured) is not None


@pytest.mark.parametrize("field", ["initial_parameters", "schedule", "resolved", "neuron_model"])
def test_development_verifier_refuses_unmatched_control_and_candidate_metadata(field: str) -> None:
    measured = body()
    run = measured["runs"][1]
    if field == "initial_parameters":
        run[field] = "f" * 64
    elif field == "schedule":
        run[field]["order_sha256"] = "f" * 64
    elif field == "resolved":
        run[field]["genes"]["teaching"] = "founder"
    else:
        run[field]["dt"] = 1.0
    assert repair.check(measured) is not None


def test_verifier_requires_both_evaluation_branches_and_a_known_stage() -> None:
    measured = body()
    del measured["runs"][1]["evaluation"]["branches"]["erased"]
    assert repair.check(measured) is not None
    measured = body("confirmation")
    measured["stage"] = "undeclared"
    assert repair.check(measured) is not None


@pytest.mark.parametrize("field", ["acquisition_floor", "rounds", "development_seeds", "recipes"])
def test_verifier_binds_the_exact_frozen_protocol(field: str) -> None:
    measured = body()
    if field == "acquisition_floor":
        measured["protocol"][field] = 0.5
    elif field == "rounds":
        measured["protocol"][field] = 2
    elif field == "development_seeds":
        measured["protocol"][field] = [0]
    else:
        measured["protocol"][field][0]["brain"]["teaching"] = "founder"
    assert repair.check(measured) == "protocol differs from the frozen declaration"


@pytest.mark.parametrize("branch", ["intact", "erased"])
def test_verifier_derives_completion_from_answers_in_both_branches(branch: str) -> None:
    measured = body()
    run = measured["runs"][1]
    run["evaluation"]["branches"][branch] = evaluation(answered=False)["branches"][branch]
    assert repair.check(measured) == "evaluation completion differs from recorded answers"
