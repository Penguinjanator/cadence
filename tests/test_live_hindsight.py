"""``arousal_hindsight``: the outcome that wakes a calm brain credits the calm moments before it.

A calm moment keeps no eligibility. With ``hindsight=k`` the brain holds its last ``k`` calm
transitions; an outcome that wakes it steps the actor and critic through them in order, the
waking outcome last. Calm moments that never wake the brain still change nothing, and the
founder ``hindsight=0`` is left out of genes and saves, so founders read as before the gene.
"""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd

OBS = np.array([[1.0, 0.0, 0.0]])


def calm_brain(hindsight: int, seed: int = 0) -> cd.Brain:
    return cd.Brain.compose(
        3, 2, modules=(12,), seed=seed, episodic=False,
        arousal=cd.ArousalConfig(youth=4, hindsight=hindsight),
    )


def parameters(brain: cd.Brain) -> dict[str, np.ndarray]:
    agent = brain.basal_ganglia
    return {
        "efficacy": np.asarray(brain.brain.efficacy).copy(),
        "bias": np.asarray(brain.brain.bias).copy(),
        "critic": np.asarray(agent.w_critic).copy(),
    }


def calm_life(brain: cd.Brain, moments: int = 60) -> None:
    """A steady income after youth: the brain settles into routine."""
    brain.live(OBS)
    for _ in range(moments):
        brain.live(OBS, reward=[0.1])
    assert brain.last_arousal["mode"] == "routine"


def test_the_founder_hindsight_is_left_out_and_a_positive_one_is_carried(tmp_path):
    founder, held = calm_brain(0), calm_brain(5)
    assert "arousal_hindsight" not in founder.describe()["genes"]
    assert held.describe()["genes"]["arousal_hindsight"] == 5
    assert founder.arousal.to_dict()["format"] == "cadence-arousal/4"
    assert held.arousal.to_dict()["format"] == "cadence-arousal/5"
    twin = cd.Brain.load(held.save(tmp_path / "held.npz"))
    assert twin.arousal.config.hindsight == 5
    saved = held.arousal.to_dict()
    with pytest.raises(ValueError, match="arousal format"):
        cd.Arousal.from_dict({**saved, "format": "cadence-arousal/4"})
    with pytest.raises(ValueError, match="arousal format"):
        cd.Arousal.from_dict({**saved, "config": {**saved["config"], "hindsight": 0}})
    with pytest.raises(ValueError, match="hindsight"):
        cd.ArousalConfig(hindsight=-1)
    with pytest.raises(ValueError, match="integer"):
        cd.Brain.compose(3, 2, modules=(8,), arousal=True, arousal_hindsight=2.5)


def test_calm_moments_change_nothing_until_an_outcome_wakes_the_brain():
    brain = calm_brain(4)
    calm_life(brain)
    before = parameters(brain)
    for _ in range(10):
        brain.live(OBS, reward=[0.1])
        assert brain.last_arousal["credited"] == 0
    for name, value in parameters(brain).items():
        np.testing.assert_array_equal(value, before[name], err_msg=name)


def test_a_waking_outcome_credits_the_held_calm_moments_and_only_with_hindsight():
    lives = {}
    for hindsight in (0, 4):
        brain = calm_brain(hindsight)
        calm_life(brain)
        before = parameters(brain)
        brain.live(OBS, reward=[-5.0])  # the routine act was harmful
        reading = brain.last_arousal
        assert reading["mode"] == "aroused"
        lives[hindsight] = (reading, before, parameters(brain), brain.last_learning)
    reading, before, after, _ = lives[0]
    assert reading["credited"] == 0 and not reading["learned"]
    for name in before:
        np.testing.assert_array_equal(after[name], before[name], err_msg=name)
    reading, before, after, learning = lives[4]
    assert reading["credited"] == 4 and reading["learned"]
    assert reading["learning_sweeps"] > 0
    assert not np.array_equal(after["efficacy"], before["efficacy"])
    assert not np.array_equal(after["critic"], before["critic"])
    assert learning["dopamine"] < 0  # the waking outcome was worse than forecast


def test_held_calm_moments_end_with_the_stream():
    brain = calm_brain(8)
    calm_life(brain)
    brain.reset()
    brain.live(OBS)
    brain.live(OBS, reward=[0.1])
    brain.live(OBS, reward=[-5.0])
    assert brain.last_arousal["credited"] <= 2  # only the moments lived since the reset


def test_a_lesson_on_the_moment_just_acted_on_warns_with_a_recall_memory():
    import warnings

    brain = cd.Brain.compose(3, 2, modules=(8,), seed=0, learned_code=4, attention=True)
    brain.act(OBS, greedy=True)
    with pytest.warns(RuntimeWarning, match="Teach before acting"):
        brain.lesson(OBS, [1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        other = np.array([[0.0, 1.0, 0.0]])
        brain.lesson(other, [0])  # the next moment, taught before its act
        brain.act(other, greedy=True)
        brain.reset()
        brain.lesson(OBS, [1])
        plain = cd.Brain.compose(3, 2, modules=(8,), seed=0)  # no recall memory: no warning
        plain.act(OBS, greedy=True)
        plain.lesson(OBS, [1])


def test_an_ended_waking_outcome_credits_the_sampled_actions_before_the_calm_run():
    brain = calm_brain(4)
    calm_life(brain)
    agent = brain.basal_ganglia
    agent.trace[...] = 1e-3  # credit left by earlier sampled actions
    seen = {}
    unfade = agent.unfade

    def spy(moments):
        unfade(moments)
        seen["trace"] = float(np.abs(agent.trace).max())

    agent.unfade = spy
    brain.live(OBS, reward=[-5.0], done=[True])
    assert brain.last_arousal["credited"] == 4
    assert seen["trace"] > 0  # the end did not clear the traces before the replay
    assert not np.abs(agent.trace).any()  # the replayed end cleared them


def test_a_refused_replay_leaves_no_held_action_pending(monkeypatch):
    brain = calm_brain(4)
    calm_life(brain)
    agent = brain.basal_ganglia
    calls = {"n": 0}
    learn = agent.learn

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("non-finite")
        return learn(*args, **kwargs)

    monkeypatch.setattr(agent, "learn", failing)
    with pytest.raises(ValueError, match="non-finite"):
        brain.live(OBS, reward=[-5.0])
    assert agent._pending is None and not brain.pending_feedback


def test_retuning_hindsight_drops_the_held_moments():
    brain = calm_brain(3)
    calm_life(brain)
    assert brain._calm
    brain.retune(arousal_hindsight=0)
    assert not brain._calm and brain._calm_issued is None
