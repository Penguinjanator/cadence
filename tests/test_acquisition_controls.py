"""The acquisition controls' surprise rule teaches the wrong rows from their rows of the complete
pre-query stimulus: the trace the prefix built is present, and the query's own settle never
advances it before the lesson."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "benchmarks" / "relevance"
sys.path.insert(0, str(HERE))

import acquisition_controls as ac  # noqa: E402
import selective_recall as sr  # noqa: E402

try:
    import cadence  # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    pytest.skip("Cadence is not importable", allow_module_level=True)

from cadence import Brain  # noqa: E402


@pytest.mark.parametrize("exposed", [False, True])
def test_surprise_lessons_read_the_pre_query_stimulus_of_their_rows(exposed: bool) -> None:
    brain = Brain.compose(sr.INPUTS, 2, seed=5, modules=[16], episodic=False)
    captured: list[tuple[np.ndarray, np.ndarray]] = []
    original = brain.learner.step

    def spy(drive: np.ndarray, labels: np.ndarray, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured.append((np.array(drive, copy=True), np.array(labels, copy=True)))
        return original(drive, labels, *args, **kwargs)

    brain.learner.step = spy  # type: ignore[method-assign]
    rng = np.random.default_rng(0)
    order = np.stack([rng.permutation(len(sr.CASES)) for _ in range(3)])
    glow = brain.working_memory.glow
    # replay the loop by hand for the first round to know the expected lesson
    rows = sr.CASES[order[0]]
    prefix, query, _ = sr.inputs(rows[:, :3], rows[:, 3], ac.NEUTRAL_EVENTS, exposed=exposed)
    twin = Brain.compose(sr.INPUTS, 2, seed=5, modules=[16], episodic=False)
    twin.reset()
    for x in prefix:
        twin.act(x, greedy=True)
    expected_drive = twin.stimulus(query)
    answer = twin.act(query, greedy=True)
    target = ac.targets("mux", rows, rows[:, 3])
    wrong = np.flatnonzero(np.asarray(answer).reshape(-1) != target)
    work = sr.Work()
    ac.train(brain, "mux", order[:1], work, "surprise", exposed)
    if len(wrong) == 0:
        assert captured == [] and work.lessons == 0
        return
    assert len(captured) == 1 and work.lessons == 1
    drive, labels = captured[0]
    assert drive.shape == (len(wrong), expected_drive.shape[1])
    assert np.array_equal(labels, target[wrong])
    assert np.allclose(drive, expected_drive[wrong])  # the rows of the complete stimulus
    assert np.abs(drive[:, glow]).max() > 0  # the context the prefix built is present
    assert not np.allclose(drive[:, glow], twin.stimulus(query)[wrong][:, glow])  # the query advanced the twin's trace


def test_every_rule_still_teaches_all_cases_then_acts() -> None:
    brain = Brain.compose(sr.INPUTS, 2, seed=3, modules=[16], episodic=False)
    calls: list[int] = []
    original = brain.learner.step

    def spy(drive, labels, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        calls.append(len(labels))
        return original(drive, labels, *args, **kwargs)

    brain.learner.step = spy  # type: ignore[method-assign]
    rng = np.random.default_rng(1)
    order = np.stack([rng.permutation(len(sr.CASES)) for _ in range(2)])
    work = sr.Work()
    trained = ac.train(brain, "and", order, work, "every")
    assert calls == [16, 16] and trained["lessons"] == 2 and work.lessons == 2
