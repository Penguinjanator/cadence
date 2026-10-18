"""The episodic memory's complete greedy policy.

With ``memory_choice`` a greedy act is the readout with the episodic vote. A lesson under
``learning_surprise`` judges a row by that act, ``live`` decides from it whether an issued action
was the brain's own best guess, and an imagined branch issues it to its efference copy and the
recall memory's predecessor key.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd


def episodic(*args, **options) -> cd.Brain:
    """``Brain.compose`` with the episodic memory."""
    return cd.Brain.compose(*args, **{"memory_choice": True, **options})


def one_hot(index: int, width: int) -> np.ndarray:
    x = np.zeros((1, width))
    x[0, index] = 1.0
    return x


def assert_same_checkpoint(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def cued(seed: int, **options) -> tuple[cd.Brain, dict[int, np.ndarray], np.ndarray]:
    """A brain shown each cue's answer once at the go moment after it. Its cortex learns
    nothing from the lessons (``learning_eta=0``): the records alone carry them."""
    brain = episodic(3, 2, modules=(8,), seed=seed, learning_eta=0, learning_eta_bias=0,
                     **options)
    cues, go = {0: one_hot(0, 3), 1: one_hot(1, 3)}, one_hot(2, 3)
    for cue, shown in cues.items():
        brain.reset()
        brain.step(shown, teacher=[cue])
        brain.step(go, reward=[0.0], teacher=[cue])
    brain.reset()
    return brain, cues, go


def misled(**options) -> tuple[cd.Brain, np.ndarray, int]:
    """A brain whose memory turns its cortex's answer into the other act: one demonstration of
    the other action is recorded, and the cortex learns nothing from it."""
    x = one_hot(0, 3)
    brain = episodic(3, 2, modules=(8,), seed=3, learning_surprise=True, learning_eta=0,
                     learning_eta_bias=0, **options)
    cortical = int(brain.act(x, greedy=True)[0])
    brain.reset()
    brain.step(x, teacher=[1 - cortical])
    brain.reset()
    return brain, x, cortical


# -- one calculation of the act a greedy brain issues


@pytest.mark.parametrize(
    "options,teacher",
    [
        ({"memory_choice": False}, [1]),
        ({}, [1]),
        ({"memory_context": 0.0}, [1]),
        ({"slots": 2}, [[1, 0]]),
        ({"slots": (1, 3)}, [[0, 2]]),
    ],
)
def test_the_complete_greedy_choice_is_the_act_a_greedy_brain_issues(options, teacher):
    brain = episodic(3, 4, modules=(8,), seed=2, **options)
    for t in range(3):
        brain.step(one_hot(t, 3), reward=None if t == 0 else [float(t == 2)], teacher=teacher)
    for t in range(6):
        x = one_hot(t % 3, 3)
        _, free = brain._settled(x)
        expected = brain._greedy(free, x, brain._decision_key(x))
        np.testing.assert_array_equal(brain.act(x, greedy=True), expected)


# -- the surprise lesson judges by the act the brain would issue


def test_a_surprise_lesson_corrects_an_act_the_memory_made_wrong(tmp_path):
    brain, x, cortical = misled()
    assert int(brain.act(x, greedy=True)[0]) == 1 - cortical  # the memory overrules the cortex
    brain.reset()
    saved = brain.save(tmp_path / "misled.npz")
    writes = brain.hippocampus.writes, brain.context_memory.writes
    report = brain.lesson(x, [cortical])  # the right answer, which the cortex already gives
    assert report["accepted_presentations"] == 1.0 and report["agreed_presentations"] == 0.0
    assert (brain.hippocampus.writes, brain.context_memory.writes) == (writes[0] + 1, writes[1] + 1)
    # the lesson every row gets without the rule corrects the memory alike
    control, resumed = cd.Brain.load(saved), cd.Brain.load(saved)
    control.retune(learning_surprise=False)
    control.lesson(x, [cortical])
    assert resumed.lesson(x, [cortical]) == report  # the saved brain takes the same lesson
    for other in (control, resumed):
        for name in ("hippocampus", "context_memory"):
            np.testing.assert_array_equal(getattr(other, name).strength,
                                          getattr(brain, name).strength)
    for subject in (brain, control, resumed):
        subject.reset()
        assert int(subject.act(x, greedy=True)[0]) == cortical


def test_an_act_the_memory_makes_right_is_no_surprise_and_gets_no_lesson():
    brain, x, cortical = misled()
    shown = 1 - cortical  # what the brain does, by its memory against its cortex
    brain.retune(learning_eta=0.5, learning_eta_bias=0.05)  # the cortex could learn again
    graph = brain.brain
    parameters = [np.array(getattr(graph, name), copy=True) for name in ("efficacy", "bias")]
    writes = brain.hippocampus.writes, brain.context_memory.writes
    report = brain.lesson(x, [shown])
    assert report["accepted_presentations"] == 0.0 and report["agreed_presentations"] == 1.0
    # consolidating the cortex under an answer the memory supports is not surprise teaching
    for name, before in zip(("efficacy", "bias"), parameters, strict=True):
        np.testing.assert_array_equal(getattr(brain.brain, name), before)
    assert (brain.hippocampus.writes, brain.context_memory.writes) == writes
    assert int(brain._choices(brain._settled(x)[1])[0]) == cortical


def test_a_surprise_lesson_agrees_exactly_where_the_brain_would_act_as_shown(tmp_path):
    brain, cues, go = cued(4)
    brain.retune(learning_surprise=True)
    decided = 0
    for cue, shown in cues.items():
        for label in (0, 1):
            brain.reset()
            brain.act(shown, greedy=True)  # the cue: the context of the decision at go
            copy = cd.Brain.load(brain.save(tmp_path / f"cue{cue}-{label}.npz"))
            act = int(copy.act(go, greedy=True)[0])
            decided += act != int(copy._choices(copy.basal_ganglia.state)[0])
            report = brain.lesson(go, [label])
            assert report["agreed_presentations"] == float(act == label)
            assert report["accepted_presentations"] == float(act != label)
    assert decided  # the vote, under the decision's own context, decides some of these acts


# -- an outcome of a greedy act the vote decided is the brain's own


def test_a_greedy_act_the_vote_decided_is_owned_and_its_outcome_can_surprise(tmp_path):
    x = one_hot(0, 2)
    brain = episodic(2, 2, modules=(8,), seed=0, arousal={"youth": 0})
    cortical = int(brain.act(x, greedy=True)[0])
    brain.reset()
    brain.step(x, teacher=[1 - cortical])
    brain.reset()
    assert int(brain.live(x)[0]) == 1 - cortical  # a calm, greedy act the vote decided
    assert brain._lived[5]
    saved = brain.save(tmp_path / "issued.npz")
    with np.load(saved, allow_pickle=False) as data:
        assert json.loads(str(data["generic"]))["lived"]["own"] is True
    copy = cd.Brain.load(saved)
    assert copy._lived[5]
    writes = brain.hippocampus.writes
    for subject in (brain, copy):
        for reward in [0.0] * 5 + [1.0]:  # as forecast five times, then far beyond it
            assert subject._lived[5]
            subject.live(x, reward=[reward])
        assert subject.arousal.outcomes == subject.arousal.records == 6
        reading = subject.last_arousal
        assert reading["mode"] == "aroused" and reading["surprise"] > 0 and reading["recorded"]
        assert subject.hippocampus.writes == writes + 1  # the outcome that woke it
    assert dict(copy.last_arousal) == dict(brain.last_arousal)
    for name in ("hippocampus", "context_memory"):
        np.testing.assert_array_equal(getattr(copy, name).strength, getattr(brain, name).strength)


def test_an_aroused_brain_owns_the_samples_of_its_complete_greedy_choice():
    x = one_hot(0, 2)
    # place records alone, and nothing learned or written while it lives: the vote stays
    brain = episodic(2, 2, modules=(8,), seed=0, memory_context=0.0, arousal={"youth": 1000},
                     learning_eta=0, learning_eta_bias=0, actor_eta=0, actor_eta_bias=0,
                     actor_eta_critic=0)
    cortical = int(brain.act(x, greedy=True)[0])
    brain.reset()
    brain.step(x, teacher=[1 - cortical])
    brain.reset()
    writes = brain.hippocampus.writes
    own, overruled = [], 0
    for t in range(30):
        action = int(brain.live(x, reward=None if t == 0 else [0.0])[0])
        assert brain.last_arousal["mode"] == "aroused"
        state = brain.basal_ganglia.state
        # the most probable action of the readout with the place records' vote
        greedy = int(np.argmax(brain.basal_ganglia.probabilities(state, bias=brain._vote(x, None))))
        overruled += greedy != int(brain._choices(state)[0])
        assert brain._lived[5] == (action == greedy)
        assert int(brain._greedy(state, x, None)[0]) == greedy
        own.append(brain._lived[5])
    assert brain.hippocampus.writes == writes
    assert overruled and True in own and False in own
    assert brain.arousal.outcomes == sum(own[:-1])  # the last action's outcome is still to come


# -- an imagined branch issues the act the brain would issue


def test_an_imagined_branch_issues_the_act_the_brain_would_issue(tmp_path, monkeypatch):
    brain, cues, go = cued(4, efference_amplitude=1.0)
    issued = []
    release = cd.Efference.issue

    def issue(self, command):
        issued.append(int(np.argmax(command[0])))
        release(self, command)

    monkeypatch.setattr(cd.Efference, "issue", issue)
    config = brain.learner.config
    before = brain.save(tmp_path / "before.npz")
    for shown in cues.values():
        sequence = [shown, go, go]
        issued.clear()
        phases = brain.imagine(sequence, budget=config.free_steps, tolerance=config.tolerance)
        imagined = list(issued)
        cortex = [int(np.argmax(phase.state.activation[0, brain.motor_index])) for phase in phases]
        live = cd.Brain.load(before)
        acts, states = [], []
        for x in sequence:
            acts.append(int(live.act(x, greedy=True)[0]))
            states.append(live.basal_ganglia.state.activation)
        # the vote, read under the branch's own temporal context, decides as it decides the acts
        assert imagined == acts and cortex != acts
        for phase, state in zip(phases, states, strict=True):
            # the private copy carries the act issued, so the branch settles as the stream does
            np.testing.assert_array_equal(phase.state.activation, state)
    assert_same_checkpoint(before, brain.save(tmp_path / "after.npz"))  # nothing real changed


def test_an_imagined_branch_gives_the_predecessor_key_the_act_the_brain_would_issue(
    tmp_path, monkeypatch
):
    brain, cues, _ = cued(0, recall=True, recall_key="predecessor")
    commands = []
    release = cd.Recall.witness

    def witness(self, observations, state, *, aroused=True, command=None):
        if self._private:
            commands.append(int(np.argmax(command[0])))
        release(self, observations, state, aroused=aroused, command=command)

    monkeypatch.setattr(cd.Recall, "witness", witness)
    before = brain.save(tmp_path / "before.npz")
    decided = 0
    for shown in cues.values():
        commands.clear()
        phase = brain.imagine([shown])[0]
        live = cd.Brain.load(before)
        act = int(live.act(shown, greedy=True)[0])
        assert commands == [act]
        decided += act != int(np.argmax(phase.state.activation[0, brain.motor_index]))
    assert decided
    assert_same_checkpoint(before, brain.save(tmp_path / "after.npz"))

