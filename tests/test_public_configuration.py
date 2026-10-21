"""Public configuration preserves the acquired life and the ownership of its outcomes."""

import json
from dataclasses import replace

import numpy as np
import pytest
from checkpoints import earlier, metadata

import cadence as cd

OBSERVATIONS = np.eye(3)


def small(**options):
    settings = dict(
        modules=(8,),
        seed=3,
        arousal=True,
        working_memory_amplitude=0.3,
        actor_eta=0.03,
        actor_eta_bias=0.003,
    )
    settings.update(options)
    return cd.Brain.compose(3, 2, **settings)


def archive(brain, tmp_path, name):
    with np.load(brain.save(tmp_path / name), allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def assert_archives_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


def acquire(brain):
    action = brain.live(OBSERVATIONS[[0]])
    for index in range(1, 9):
        reward = float(action[0] == (index - 1) % 2)
        action = brain.live(OBSERVATIONS[[index % 3]], reward=[reward])
    return action


def test_plain_compose_keeps_the_released_founders():
    brain = cd.Brain.compose(3, 2, modules=(8,))
    assert brain.arousal is None and brain.efference is None
    assert brain.learner.config.eta == 0.5
    assert brain.learner.config.eta_bias == 0.05
    assert brain.basal_ganglia.config.eta == 1.0
    assert brain.basal_ganglia.config.eta_bias == 0.05
    assert brain.working_memory.amplitude == 3.0
    assert brain.working_memory.decay == 0.0  # 0.2 before 0.82.0
    # the projection that reads the carried interpretation back: 12 before 0.82.0
    assert brain.describe()["layout"]["composition"]["working_memory_scale"] == 1.2


@pytest.mark.parametrize("arousal", [True, {}, {"need": 2.0}])
def test_arousal_shorthand_uses_semantic_limits_not_mutation_search_bounds(arousal):
    brain = small(arousal=arousal)
    expected = cd.ArousalConfig(**arousal) if isinstance(arousal, dict) else cd.ArousalConfig()
    assert brain.arousal.config == expected
    assert brain.live(OBSERVATIONS[[0]]).shape == (1,)


def test_public_overrides_keep_teaching_and_reward_rates_distinct():
    learning = cd.LearnerConfig(eta=0.02, eta_bias=0.002, momentum=0.6)
    reward = cd.ActorCriticConfig(eta=0.04, eta_bias=0.004, gamma=0.7)
    brain = small(
        learning=learning,
        reward=reward,
        temperature=0.4,
        learning_beta=0.15,
        actor_eta=0.01,
        actor_eta_bias=None,
        arousal_need=2.0,
    )
    assert brain.learner.config == replace(learning, temperature=0.4, beta=0.15)
    assert brain.basal_ganglia.config == replace(reward, eta=0.01, eta_bias=None)
    assert brain.arousal.config.need == 2.0
    assert learning.temperature == 0.2 and reward.eta == 0.04


def test_the_sensory_scale_changes_only_the_sensory_projection():
    control = cd.Brain.compose(3, 5, modules=(9, 6), slots=(2, 3), seed=11)
    scaled = cd.Brain.compose(3, 5, modules=(9, 6), slots=(2, 3), seed=11, scales={"sensory": 4})
    first, second = control.connectome, scaled.connectome
    for field in ("pre", "post", "count"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    incoming = np.isin(first.pre, control.sensory_index)
    np.testing.assert_array_equal(second.sign[incoming], 4 * first.sign[incoming])
    np.testing.assert_array_equal(second.sign[~incoming], first.sign[~incoming])
    np.testing.assert_array_equal(control.brain.bias, scaled.brain.bias)
    np.testing.assert_array_equal(
        scaled.brain.efficacy[incoming], 4 * control.brain.efficacy[incoming]
    )
    np.testing.assert_array_equal(
        scaled.brain.efficacy[~incoming], control.brain.efficacy[~incoming]
    )


def test_multi_module_composition_description_survives_sorted_checkpoint_metadata(tmp_path):
    brain = cd.Brain.compose(
        3, 5, modules=(9, 6), observers=(4, 3), slots=(2, 3), scales={"sensory": 2.0}
    )
    restored = cd.Brain.load(brain.save(tmp_path / "multi-module"))
    assert restored.describe() == brain.describe()
    assert restored.describe()["layout"]["composition"]["modules"] == [9, 6]


@pytest.mark.parametrize(
    "options",
    [
        {"eta": 0.01},
        {"actor_ette": 0.01},
        {"actor_eta": True},
        {"learning_beta": np.nan},
        {"learning_eta": 0.1},
        {"learning_normalize": 0.99},
        {"teaching": "fast"},
        {"teaching": "normalized", "learning": cd.LearnerConfig()},
        {"arousal_need": np.inf},
        {"learning_free_steps": 2.5},
        {"actor_center_scale": True},
        {"actor_dopamine_center": 0.0},
        {"temperature": 0.3, "learning_temperature": 0.4},
        {"arousal": {"need": 1.0}, "arousal_need": 2.0},
        {"sensory_scale": 4.0},
        {"scales": {"sensory": -1.0}},
        {"scales": {"sensory": True}},
        {"scales": {"cortex": 1.0}},
        {"scales": 1.0},
    ],
)
def test_constructor_refuses_unknown_conflicting_and_invalid_settings(options):
    with pytest.raises((ValueError, TypeError)):
        small(**options)


@pytest.mark.parametrize(
    "invalid",
    [
        {"actor_eta": -1.0},
        {"arousal_need": np.inf},
        {"learning_free_steps": True},
        {"memory_rate": 2.0},
        {"working_memory_decay": 1.0},
        {"working_memory_focus": -1.0},
        {"surprise_threshold": 0.1},
        {"sensory_scale": 4.0},
        {"scales": {"sensory": 4.0}},
        {"learning_eta": 0.1},
        {"teaching": "slow"},
        {"resting_bias": 0.1},
        {"reset_arousal": 1},
        {"learning_beta": 0.2},
    ],
)
def test_invalid_retune_is_atomic_with_acquired_state_and_pending_reward(tmp_path, invalid):
    brain = small()
    acquire(brain)
    before = archive(brain, tmp_path, "before")
    settings = {"temperature": 0.35, "consolidation": 0.25}
    settings.update(invalid)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        brain.retune(**settings)
    assert_archives_equal(before, archive(brain, tmp_path, "after"))
    assert brain.pending_feedback


def test_retune_matches_the_existing_stage_operation_and_preserves_pending_credit(tmp_path):
    brain = small(efference_amplitude=0.2)
    acquire(brain)
    path = brain.save(tmp_path / "acquired")
    legacy = cd.Brain.load(path)
    before = archive(brain, tmp_path, "before-stage")
    pending = brain.basal_ganglia._pending
    assert (
        brain.retune(
            arousal={"need": 0.0, "heat": 0.0},
            temperature=0.3,
            actor_eta=0.003,
            actor_eta_bias=0.0003,
        )
        is brain
    )
    brain.arousal.reset()
    assert brain.basal_ganglia._pending is pending
    after = archive(brain, tmp_path, "after-stage")
    for name in before:
        if name not in ("generic", "meta", "brain"):
            np.testing.assert_array_equal(before[name], after[name], err_msg=name)
    legacy.arousal.config = replace(legacy.arousal.config, need=0.0, heat=0.0)
    legacy.learner.config = replace(legacy.learner.config, temperature=0.3)
    legacy.basal_ganglia.config = replace(legacy.basal_ganglia.config, eta=0.003, eta_bias=0.0003)
    legacy.arousal.reset()
    restored = cd.Brain.load(brain.save(tmp_path / "retuned"))
    assert restored.describe() == brain.describe()
    for index, reward in enumerate((0.7, -0.2, 0.4, 0.0)):
        observation = OBSERVATIONS[[index % 3]]
        expected = legacy.live(observation, reward=[reward])
        np.testing.assert_array_equal(brain.live(observation, reward=[reward]), expected)
        np.testing.assert_array_equal(restored.live(observation, reward=[reward]), expected)
        np.testing.assert_array_equal(brain.brain.efficacy, legacy.brain.efficacy)
        np.testing.assert_array_equal(brain.brain.bias, legacy.brain.bias)
        np.testing.assert_array_equal(
            brain.hippocampus.consolidated, legacy.hippocampus.consolidated
        )
    assert_archives_equal(archive(brain, tmp_path, "live"), archive(restored, tmp_path, "resumed"))


def test_memory_and_trace_tuning_keeps_learned_contents(tmp_path):
    brain = small(efference_amplitude=0.2)
    acquire(brain)
    before = archive(brain, tmp_path, "before-memory")
    brain.retune(
        memory_decay=0.8,
        memory_rate=0.7,
        memory_amplitude=1.2,
        consolidation=0.2,
        working_memory_decay=0.4,
        working_memory_amplitude=0.2,
        efference_decay=0.4,
        efference_amplitude=0.1,
    )
    after = archive(brain, tmp_path, "after-memory")
    for name in before:
        if name not in ("generic", "meta", "brain"):
            np.testing.assert_array_equal(before[name], after[name], err_msg=name)
    restored = cd.Brain.load(brain.save(tmp_path / "memory-tuned"))
    assert brain.describe() == restored.describe()
    np.testing.assert_array_equal(
        brain.live(OBSERVATIONS[[2]], reward=[0.2]),
        restored.live(OBSERVATIONS[[2]], reward=[0.2]),
    )


@pytest.mark.parametrize(
    "options",
    [
        {"arousal_need": 1.0},
        {"arousal": {"need": 1.0}},
        {"reset_arousal": True},
        {"memory_amplitude": 2.0},
        {"consolidation": 0.2},
        {"efference_amplitude": 0.2},
    ],
)
def test_retune_cannot_create_missing_components(options):
    brain = cd.Brain.compose(3, 2, modules=(8,), episodic=False)
    with pytest.raises((ValueError, TypeError)):
        brain.retune(**options)


def test_description_is_detached_observational_and_survives_checkpoints(tmp_path):
    brain = small(scales={"sensory": 2.0})
    description = brain.describe()
    json.dumps(description, allow_nan=False)
    assert description["genes"]["actor_eta"] == 0.03
    assert description["genes"]["teaching"] == "founder"
    description["genes"]["actor_eta"] = 999
    assert brain.basal_ganglia.config.eta == 0.03
    acquire(brain)
    before = archive(brain, tmp_path, "before-description")
    brain.describe()
    assert_archives_equal(before, archive(brain, tmp_path, "after-description"))
    restored = cd.Brain.load(brain.save(tmp_path / "described"))
    assert brain.describe() == restored.describe()


def test_pending_feedback_covers_sampled_routine_and_consumed_outcomes():
    brain = small()
    assert not brain.pending_feedback
    brain.live(OBSERVATIONS[[0]])
    assert brain.pending_feedback
    brain.learn([0.2], [False], OBSERVATIONS[[1]])
    assert not brain.pending_feedback
    # Changing beta is safe only once its issued action's phases have been consumed.
    brain.retune(learning_beta=0.2)
    brain.reset()
    brain.retune(arousal={"youth": 0, "threshold": 10.0})
    brain.live(OBSERVATIONS[[0]])
    assert brain.last_arousal["mode"] == "routine" and brain.pending_feedback
    brain.act(OBSERVATIONS[[1]], greedy=True)
    assert not brain.pending_feedback
    brain.reset()
    assert not brain.pending_feedback


def test_the_trace_focus_of_0843_is_refused_by_name_and_read_at_zero(tmp_path):
    """The working trace's focus, a weighting of each neuron by its movement, is removed: the
    name is refused at compose and retune, and a checkpoint of 0.84.3 loads at the founder
    focus of zero and is refused above it, each refusal naming 0.84.3."""
    with pytest.raises(ValueError, match="working_memory_focus was removed.*0.84.3"):
        small(working_memory_focus=0.5)
    brain = small()
    with pytest.raises(ValueError, match="working_memory_focus was removed.*0.84.3"):
        brain.retune(working_memory_focus=0.0)
    assert "working_memory_focus" not in brain.describe()["genes"]
    path = brain.save(tmp_path / "trace.npz")
    with np.load(path, allow_pickle=False) as saved:
        arrays = earlier({name: saved[name].copy() for name in saved.files})
    meta = metadata(arrays)
    for focus, loads in ((0.0, True), (0.5, False)):
        meta["working_memory"]["focus"] = focus
        arrays["generic"] = np.array(json.dumps(meta))
        np.savez(path, **arrays)
        if loads:
            assert cd.Brain.load(path).working_memory.to_dict() == brain.working_memory.to_dict()
        else:
            with pytest.raises(ValueError, match="working_memory_focus=0.5.*0.84.3"):
                cd.Brain.load(path)


def test_the_resting_bias_of_0843_is_refused_by_name_and_its_record_loads(tmp_path):
    """The composed brain's resting bias, an initial bias of the processing regions, is
    removed: the name is refused at compose and retune with a message naming 0.84.3, and the
    value a checkpoint of 0.84.3 recorded is provenance its saved biases already hold."""
    with pytest.raises(ValueError, match="resting_bias was removed.*0.84.3"):
        small(resting_bias=0.5)
    brain = small()
    with pytest.raises(ValueError, match="resting_bias was removed.*0.84.3"):
        brain.retune(resting_bias=0.5)
    assert "resting_bias" not in brain.describe()["layout"]["composition"]
    path = brain.save(tmp_path / "recorded.npz")
    with np.load(path, allow_pickle=False) as saved:
        arrays = earlier({name: saved[name].copy() for name in saved.files})
    meta = metadata(arrays)
    meta["resting_bias"] = 0.5
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)
    np.testing.assert_array_equal(cd.Brain.load(path).brain.bias, brain.brain.bias)


def test_the_teaching_point_sets_the_lessons_steps_and_normalization_alone():
    founder = cd.Brain.compose(3, 2, modules=(8,), seed=1)
    named = cd.Brain.compose(3, 2, modules=(8,), seed=1, teaching="founder")
    assert named.learner.config == founder.learner.config  # the founder by its name
    normalized = cd.Brain.compose(3, 2, modules=(8,), seed=1, teaching="normalized")
    config = normalized.learner.config
    assert (config.eta, config.eta_bias, config.normalize, config.normalize_floor) == (
        0.003, 0.0003, 0.99, 1e-4
    )
    assert config.momentum == 0.9
    assert replace(config, eta=0.5, eta_bias=0.05, normalize=0.0, normalize_floor=1e-3) == (
        founder.learner.config
    )  # nothing else moves
    founder.retune(teaching="normalized")
    assert founder.learner.config == config
    founder.retune(teaching="founder", learning_surprise=True)
    assert founder.learner.config == replace(named.learner.config, surprise=True)
    with pytest.raises(ValueError, match="teaching"):
        cd.Brain.compose(3, 2, learning_eta=0.1)
    with pytest.raises(ValueError, match="one of them"):
        founder.retune(teaching="normalized", learning=cd.LearnerConfig())


def test_scales_name_the_projection_magnitudes_and_keep_their_founders():
    founder = cd.Brain.compose(3, 2, modules=(8,), seed=4, history_decays=(0.5,), recall=True)
    named = cd.Brain.compose(
        3, 2, modules=(8,), seed=4, history_decays=(0.5,), recall=True,
        scales={"sensory": 1.0, "trace": 1.2, "history": 1.2, "recall": 1.2},
    )
    np.testing.assert_array_equal(named.brain.efficacy, founder.brain.efficacy)
    composition = founder.describe()["layout"]["composition"]
    assert (
        composition["sensory_scale"], composition["working_memory_scale"],
        composition["history_scale"], composition["recall_scale"],
    ) == (1.0, 1.2, 1.2, 1.2)
    for old, key in (("sensory_scale", "sensory"), ("working_memory_scale", "trace"),
                     ("history_scale", "history"), ("recall_scale", "recall")):
        with pytest.raises(ValueError, match=f"scales={{'{key}'"):
            cd.Brain.compose(3, 2, **{old: 1.0})


def test_the_description_has_four_sections_and_names_every_gene_as_compose_takes_it():
    brain = cd.Brain.compose(
        4, 2, modules=(8,), seed=2, arousal=True, recall=True, memory_choice=True,
        history_decays=(0.5,), efference_amplitude=1.0, scales={"trace": 2.0},
    )
    described = brain.describe()
    assert list(described) == ["layout", "genes", "memories", "pending"]
    assert described["layout"]["composition"]["working_memory_scale"] == 2.0
    assert described["layout"]["code"]["kind"] == "observation"
    assert list(described["memories"]) == list(brain.memories)
    assert described["memories"]["episodic"] == brain.memories["episodic"].to_dict()
    assert described["pending"] is False
    genes = described["genes"]
    assert genes["teaching"] == "founder"
    assert not {"learning_eta", "learning_eta_bias", "learning_normalize",
                "learning_normalize_floor", "learning_nudge"} & set(genes)
    # every runtime gene goes back into retune as it is
    brain.retune(**{k: v for k, v in genes.items() if not k.startswith(
        ("recall_capacity", "recall_next", "history_decays"))})
    assert brain.describe()["genes"] == genes
    brain.retune(teaching="normalized")
    assert brain.describe()["genes"]["teaching"] == "normalized"
    brain.retune(learning=replace(brain.learner.config, eta=0.2))
    assert brain.describe()["genes"]["teaching"] == "custom"


def test_fit_keeps_the_arousal_of_a_life_and_retune_resets_nothing():
    brain = cd.Brain.compose(4, 2, modules=(8,), seed=5, arousal={"youth": 3})
    x = np.eye(4)[[0]]
    brain.live(x)
    for t in range(8):
        brain.live(np.eye(4)[[t % 4]], reward=[float(t % 2)])
    state = brain.arousal.to_dict()
    brain.fit(np.eye(4), np.array([0, 1, 0, 1]), epochs=1, batch=4)
    assert brain.arousal.to_dict() == state  # a control on independent samples
    brain.retune(arousal_heat=1.5)
    assert {k: v for k, v in brain.arousal.to_dict().items() if k != "config"} == {
        k: v for k, v in state.items() if k != "config"
    }
    with pytest.raises(ValueError, match="brain.arousal.reset"):
        brain.retune(reset_arousal=True)
