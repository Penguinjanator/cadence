"""A brain saved between a lesson and its next act resumes there. ``Brain.lesson`` keeps the
free state the stream is in and clears the drive that state was settled for, so that the
next act settles afresh under the taught parameters; ``save`` writes the free state without
a drive, and through 0.84.2 ``load`` refused the checkpoint (``missing saved state:
actor/_drive``). The loaded brain has no drive either and continues as the saved brain does:
the same acts, lessons, outcomes and arrays. A calm ``live`` answer taught before its
outcome, and a ``step`` whose act refused after its lesson, leave the stream in the same
state."""

import warnings

import numpy as np
import pytest

import cadence as cd

EYE = np.eye(3)
RECIPES = {
    "founder": {},
    "surprise": {"learning_surprise": True},
    "episodic": {"memory_choice": True},
    "episodic_surprise": {"memory_choice": True, "learning_surprise": True},
    "recall": {"recall": True},
    "recall_surprise": {"recall": True, "learning_surprise": True},
    "recall_episodic_surprise": {
        "recall": True, "memory_choice": True, "learning_surprise": True,
        "recall_address_reward": 1.0,
    },
}


def rows(batch, *indices):
    """The one-hot observations of the first ``batch`` of ``indices``, one row per stream."""
    return EYE[list(indices[:batch])]


def saved_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def assert_same_checkpoint(first, second):
    a, b = saved_arrays(first), saved_arrays(second)
    assert set(a) == set(b)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def assert_same_free_state(first, second):
    for name in ("v", "activation", "adaptation"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)


def taught(recipe, batch):
    """A brain that acted, learned from the outcomes and was then taught without acting."""
    brain = cd.Brain.compose(3, 2, modules=(8,), seed=1, **recipe)
    brain.step(rows(batch, 0, 1))
    brain.step(rows(batch, 1, 2), reward=np.ones(batch))
    brain.learn(np.zeros(batch), np.zeros(batch, bool), rows(batch, 2, 0))  # nothing pending
    brain.lesson(rows(batch, 2, 0), np.array([1, 0][:batch]))
    return brain


def continuation(brain, batch):
    """A greedy act, a further lesson, a sampled act and its rewarded outcome: what each
    returns and reports, with the activity and parameters they leave."""
    agent = brain.basal_ganglia
    seen = [brain.act(rows(batch, 1, 2), greedy=True), agent.state.activation.copy()]
    seen.append(brain.lesson(rows(batch, 0, 1), np.array([0, 1][:batch])))
    seen.append(brain.step(rows(batch, 2, 1)))
    seen.append(brain.step(rows(batch, 0, 2), reward=np.array([1.0, -1.0][:batch])))
    seen.append(dict(brain.last_learning))
    seen += [agent.state.activation.copy(), brain.brain.efficacy.copy(), agent.w_critic.copy()]
    return seen


@pytest.mark.parametrize("recipe", [{}, {"learning_surprise": True}], ids=["founder", "surprise"])
def test_a_checkpoint_between_a_lesson_and_the_next_act_loads(tmp_path, recipe):
    """The defect of 0.84.2: a greedy act, a lesson and a save gave a checkpoint that
    ``load`` refused with ``missing saved state: actor/_drive``."""
    brain = cd.Brain.compose(3, 2, modules=(8,), seed=1, **recipe)
    brain.act(EYE[[0]], greedy=True)
    brain.lesson(EYE[[1]], np.array([1]))
    assert brain.basal_ganglia._drive is None and brain.basal_ganglia.state is not None
    copy = cd.Brain.load(brain.save(tmp_path / "taught"))
    assert copy.basal_ganglia._drive is None  # no cached drive: the next act settles afresh
    assert_same_free_state(copy.basal_ganglia.state, brain.basal_ganglia.state)
    np.testing.assert_array_equal(copy.act(EYE[[1]], greedy=True), brain.act(EYE[[1]], greedy=True))
    assert_same_free_state(copy.basal_ganglia.state, brain.basal_ganglia.state)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("recipe", list(RECIPES.values()), ids=list(RECIPES))
def test_a_brain_saved_after_a_lesson_continues_as_the_saved_one(tmp_path, recipe, batch):
    brain = taught(recipe, batch)
    path = brain.save(tmp_path / "taught")
    copy = cd.Brain.load(path)
    assert brain.basal_ganglia._drive is None and copy.basal_ganglia._drive is None
    assert_same_free_state(copy.basal_ganglia.state, brain.basal_ganglia.state)
    assert_same_checkpoint(path, copy.save(tmp_path / "copy"))  # the round trip is exact
    for kept, resumed in zip(continuation(brain, batch), continuation(copy, batch), strict=True):
        np.testing.assert_equal(resumed, kept)
    assert_same_checkpoint(brain.save(tmp_path / "kept"), copy.save(tmp_path / "resumed"))


def test_a_calm_answer_taught_before_its_outcome_resumes(tmp_path):
    """A lesson between a routine ``live`` answer and its outcome: both brains keep the issued
    action and its forecast, and take the outcome once."""
    brain = cd.Brain.compose(3, 2, modules=(8,), seed=1, arousal={"youth": 0})
    brain.live(EYE[[0]])
    for t in range(4):
        brain.live(EYE[[t % 3]], reward=[0.0])
    assert brain.last_arousal["mode"] == "routine" and brain.pending_feedback
    brain.lesson(EYE[[1]], np.array([1]))
    copy = cd.Brain.load(brain.save(tmp_path / "calm"))
    assert copy.basal_ganglia._drive is None
    assert copy.pending_feedback and copy.decision_id == brain.decision_id
    for moment, reward in enumerate((1.0, 0.0, -1.0, 0.0)):
        x = EYE[[moment % 3]]
        np.testing.assert_array_equal(copy.live(x, reward=[reward]), brain.live(x, reward=[reward]))
        np.testing.assert_equal(dict(copy.last_arousal), dict(brain.last_arousal))
    assert_same_checkpoint(brain.save(tmp_path / "kept"), copy.save(tmp_path / "resumed"))


def test_a_step_whose_act_refused_after_its_lesson_resumes(tmp_path):
    """``step(..., teacher=...)`` teaches, then acts. An act that refuses leaves the lesson
    learned and the stream as a lesson leaves it; its checkpoint resumes there."""
    brain = cd.Brain.compose(3, 2, modules=(8,), seed=1)
    brain.act(EYE[[0]], greedy=True)
    budget = brain.learner.config.free_steps
    brain.retune(learning_free_steps=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the finite lesson used its one sweep
        with pytest.raises(RuntimeError, match="did not settle"):
            brain.step(EYE[[1]], teacher=np.array([1]))
    assert brain.last_learning["demonstrations"] == 1.0 and brain.basal_ganglia._drive is None
    copy = cd.Brain.load(brain.save(tmp_path / "refused"))
    for life in (brain, copy):
        life.retune(learning_free_steps=budget)
    for kept, resumed in zip(continuation(brain, 1), continuation(copy, 1), strict=True):
        np.testing.assert_equal(resumed, kept)


def test_a_saved_drive_is_still_checked(tmp_path):
    """A drive saved with the free state has the free state's shape, or the checkpoint is
    refused."""
    brain = cd.Brain.compose(3, 2, modules=(8,), seed=1)
    brain.act(EYE[[0]], greedy=True)
    path = brain.save(tmp_path / "acted")
    arrays = saved_arrays(path)
    assert arrays["actor/_drive"].shape == (1, brain.connectome.n)
    arrays["actor/_drive"] = np.zeros((2, brain.connectome.n))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="invalid saved state: actor/_drive"):
        cd.Brain.load(path)
