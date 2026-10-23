"""Optional initial numerical steps preserve equations, work and continuation."""

import json

import numpy as np
import pytest

import cadence as cd


def graph():
    pre, post = np.where(~np.eye(36, dtype=bool))
    c = cd.Connectome.from_synapses(36, pre=pre, post=post, sign=np.full(len(pre), -0.5))
    return cd.NeuralGraph(c, cd.learning_neuron_model(dt=1))


def legacy_schedule(brain, drive, *, budget, damping, tolerance):
    """The preceding default schedule, kept independently of the new copy hook."""
    if not damping:
        return brain._equilibrate_local(
            drive,
            budget=budget,
            chunk=32,
            tolerance=tolerance,
            state=None,
            mask=None,
            nudge=None,
        )
    attempts = min(damping + 1, max(1, budget))
    used = checks = comparisons = 0
    state = None
    for index in range(attempts):
        portion = (budget - used + attempts - index - 1) // (attempts - index)
        candidate = (
            brain
            if not index
            else cd.NeuralGraph(
                brain.connectome,
                brain.neuron_model.replace(dt=float(np.ldexp(1.0, -index))),
                efficacy=brain.efficacy,
                log_gain=brain.log_gain,
                bias=brain.bias,
                dense_limit=brain.dense_limit,
                layout=brain.layout,
                precision=brain.precision,
            )
        )
        phase = candidate._equilibrate_local(
            drive,
            budget=portion,
            chunk=32,
            tolerance=tolerance,
            state=state,
            mask=None,
            nudge=None,
            stop_if_stalled=index < attempts - 1,
        )
        used += phase.steps
        checks += phase.residual_checks
        comparisons += phase.stagnation_checks
        error = phase.residual
        if index:
            error = brain.residual(drive, phase.state)
            checks += 1
        result = cd.Equilibrium(
            phase.state,
            error,
            used,
            tolerance,
            residual_checks=checks,
            damping_halvings=index,
            stagnation_checks=comparisons,
        )
        if result.qualified.all() or used >= budget:
            return result
        state = phase.state
    return result


def same_state(first, second):
    for name in ("v", "activation", "adaptation"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


@pytest.mark.parametrize("damping,budget", [(0, 0), (0, 64), (3, 0), (3, 256)])
def test_default_zero_is_the_original_numerical_schedule(damping, budget):
    brain = graph()
    drive = np.full((2, 36), 0.2)
    expected = legacy_schedule(brain, drive, budget=budget, damping=damping, tolerance=3e-3)
    for actual in (
        brain.equilibrate(drive, budget=budget, damping=damping, tolerance=3e-3),
        brain.equilibrate(drive, budget=budget, damping=damping, tolerance=3e-3, initial_damping=0),
    ):
        same_state(actual.state, expected.state)
        np.testing.assert_array_equal(actual.residual, expected.residual)
        for name in ("steps", "residual_checks", "damping_halvings", "stagnation_checks"):
            assert getattr(actual, name) == getattr(expected, name)
        assert actual.initial_damping == 0
        assert actual.relative_damping_halvings == actual.damping_halvings


def test_remaining_attempts_preserve_warm_mask_nudge_and_count_original_checks(monkeypatch):
    brain, calls = graph(), []
    drive = np.ones((2, 36))
    warm = cd.BrainState(np.zeros_like(drive), np.zeros_like(drive), np.zeros_like(drive), 0)
    mask, nudge = np.ones(36), cd.Nudge(np.zeros_like(drive), np.ones(36), 0.1)

    def attempt(self, actual_drive, **options):
        calls.append((self.neuron_model.dt, options))
        assert actual_drive is drive and options["mask"] is mask and options["nudge"] is nudge
        assert options["state"] is warm
        return cd.Equilibrium(
            warm, np.zeros(2), options["budget"], 1e-9, residual_checks=2, stagnation_checks=1
        )

    monkeypatch.setattr(cd.NeuralGraph, "_equilibrate_local", attempt)
    # A candidate's local claim cannot qualify the unchanged original equations.
    monkeypatch.setattr(brain, "residual", lambda *a, **k: np.ones(2))
    actual = brain.equilibrate(
        drive,
        budget=9,
        damping=3,
        initial_damping=2,
        tolerance=1e-9,
        state=warm,
        mask=mask,
        nudge=nudge,
    )
    assert [(dt, k["budget"], k["stop_if_stalled"]) for dt, k in calls] == [
        (0.25, 5, True),
        (0.125, 4, False),
    ]
    assert actual.steps == 9 and actual.residual_checks == 6
    assert actual.stagnation_checks == 2 and not actual.qualified.any()
    assert (actual.initial_damping, actual.damping_halvings, actual.relative_damping_halvings) == (
        2,
        3,
        1,
    )
    assert brain.neuron_model.dt == 1


@pytest.mark.parametrize("value", [True, -1, 4, 1.5, np.nan])
def test_invalid_index_refuses_before_transport(monkeypatch, value):
    brain = graph()
    monkeypatch.setattr(brain, "_equilibrate_local", lambda *a, **k: pytest.fail("phase ran"))
    monkeypatch.setattr(brain, "_with_dt", lambda *a: pytest.fail("copy ran"))
    with pytest.raises(ValueError, match="initial_damping"):
        brain.equilibrate(np.ones((1, 36)), damping=3, initial_damping=value)
    with pytest.raises(ValueError, match="initial_damping"):
        cd.LearnerConfig(qualified=True, initial_damping=value)


@pytest.mark.parametrize("value", [np.uint64(0), np.uint64(2)])
def test_unsigned_graph_index_uses_host_integer_exponents(value):
    actual = graph().equilibrate(
        np.ones((1, 36)), budget=np.uint64(0), damping=np.uint64(3), initial_damping=value
    )
    assert actual.initial_damping == int(value)
    assert actual.damping_halvings == int(value)
    assert np.isfinite(actual.state.v).all()


def test_qualified_only_and_unsupported_backend_refuse_before_phase(monkeypatch):
    with pytest.raises(ValueError, match="qualified"):
        cd.LearnerConfig(initial_damping=1)
    brain = graph()
    monkeypatch.setattr(brain, "backend", "unsupported")
    with pytest.raises(ValueError, match="CPU float64"):
        brain.equilibrate(np.ones((1, 36)), damping=3, initial_damping=2)
    with pytest.raises(ValueError, match="CPU float64"):
        cd.Learner(brain, np.arange(36), cd.LearnerConfig(qualified=True, initial_damping=2))


def test_numerical_copy_uses_current_parameters_and_refuses_unowned_boundary():
    brain = graph().with_parameters(
        efficacy=np.full(graph().connectome.synapses, -0.1), bias=np.linspace(-0.01, 0.01, 36)
    )
    copy = brain._with_dt(0.25)
    assert copy is not brain and copy.connectome is brain.connectome
    assert copy.layout is brain.layout and copy.neuron_model.dt == 0.25
    for name in ("efficacy", "bias", "log_gain"):
        np.testing.assert_array_equal(getattr(copy, name), getattr(brain, name))

    class Boundary(cd.NeuralGraph):
        pass

    unsupported = Boundary(brain.connectome, brain.neuron_model)
    with pytest.raises(ValueError, match="boundary graph"):
        unsupported.equilibrate(np.ones((1, 36)), damping=3, initial_damping=2)

    class DroppedBoundary(Boundary):
        def _with_dt(self, dt):
            return brain._with_dt(dt)

    unsupported = DroppedBoundary(brain.connectome, brain.neuron_model)
    with pytest.raises(ValueError, match="boundary graph type"):
        unsupported.equilibrate(np.ones((1, 36)), damping=3, initial_damping=2)


def learner(*, budget=256):
    return cd.Learner(
        graph(),
        np.arange(36),
        cd.LearnerConfig(
            qualified=True,
            free_steps=budget,
            nudged_steps=budget,
            tolerance=3e-3,
            damping=3,
            initial_damping=2,
            momentum=0.9,
            normalize=0.9,
            eta=0.01,
            eta_bias=0.001,
        ),
    )


def snapshot(owner):
    return {
        name: getattr(
            owner.brain if name in ("efficacy", "bias", "log_gain") else owner, name
        ).copy()
        for name in (
            "efficacy",
            "bias",
            "log_gain",
            "velocity",
            "velocity_bias",
            "second_moment",
            "second_moment_bias",
        )
    }


def test_actual_native_three_phase_update_and_refusal_are_original_equation_qualified():
    owner, drive = learner(), np.full((2, 36), 0.2)
    model = owner.brain.neuron_model
    _, report = owner.step(drive, np.array([0, 1]))
    assert owner.updates == owner.contrast_updates == 1 and owner.brain.neuron_model is model
    for name in ("free", "nudged", "opposite"):
        assert report[name + "_residual"] <= 3e-3
        assert report[name + "_initial_damping"] == 2
        assert report[name + "_damping_halvings"] in (2, 3)
        assert report[name + "_relative_damping_halvings"] == report[name + "_damping_halvings"] - 2
    refused = learner(budget=0)
    before = snapshot(refused)
    with pytest.raises(cd.LearningPhaseError) as error:
        refused.step(drive, np.array([0, 1]))
    assert not error.value.phases["free"].qualified.any()
    assert error.value.report["free_initial_damping"] == 2
    assert refused.updates == refused.contrast_updates == 0
    for name, value in before.items():
        np.testing.assert_array_equal(snapshot(refused)[name], value)


def mutate(path, change):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    meta = json.loads(str(arrays["meta"]))
    change(meta)
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)


def test_learner_policy_archive_roundtrip_and_old_default_archive(tmp_path):
    owner, drive = learner(), np.full((2, 36), 0.2)
    owner.step(drive, np.array([0, 1]))
    path = owner.save(tmp_path / "policy.npz")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
    assert meta["format"] == "cadence-checkpoint/4"
    assert meta["numerical_policy"] == {"kind": "initial-damping/1", "initial_damping": 2}
    back = cd.Learner.load(path)
    assert back.config == owner.config and back.updates == owner.updates
    for name, value in snapshot(owner).items():
        np.testing.assert_array_equal(snapshot(back)[name], value)
    for item in (owner, back):
        item.step(drive, np.array([1, 0]))
    for name, value in snapshot(owner).items():
        np.testing.assert_array_equal(snapshot(back)[name], value)
    ordinary = cd.Learner(graph(), np.arange(36))
    assert ordinary.config.to_dict()["initial_damping"] == 0
    old = ordinary.save(tmp_path / "ordinary.npz")
    with np.load(old, allow_pickle=False) as data:
        ordinary_meta = json.loads(str(data["meta"]))
    assert ordinary_meta["format"] == "cadence-checkpoint/2"
    assert "initial_damping" not in ordinary_meta["config"]
    assert "numerical_policy" not in ordinary_meta
    mutate(old, lambda meta: meta["config"].pop("initial_damping", None))
    assert cd.Learner.load(old).config.initial_damping == 0
    mutate(old, lambda meta: meta["config"].update(nudge="unsupported-old-option"))
    replacement = cd.LearnerConfig(eta=0, eta_bias=0)
    assert cd.Learner.load(old, config=replacement).config is replacement


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.pop("numerical_policy"),
        lambda m: m["numerical_policy"].update(kind="unknown"),
        lambda m: m["config"].update(initial_damping=1),
        lambda m: m.update(format="cadence-checkpoint/2"),
        lambda m: m.update(format="cadence-checkpoint/3"),
    ],
)
def test_archive_policy_mismatch_or_removed_format_refuses(tmp_path, change):
    path = learner().save(tmp_path / "policy.npz")
    mutate(path, change)
    with pytest.raises(ValueError):
        cd.Learner.load(path)


def test_checkpoint_and_compose_reject_unsupported_policy_before_backend_creation(
    tmp_path, monkeypatch
):
    import cadence.checkpoint as checkpoint
    import cadence.generic as generic

    path = learner().save(tmp_path / "policy.npz")
    monkeypatch.setattr(checkpoint, "Brain", lambda *a, **k: pytest.fail("backend constructed"))
    with pytest.raises(ValueError, match="CPU float64"):
        checkpoint.load(path, backend="torch")
    with pytest.raises(ValueError, match="CPU float64"):
        checkpoint.load(path, precision="float32")
    # An ordinary archive with an explicit nondefault replacement must refuse too.
    ordinary = cd.Learner(graph(), np.arange(36)).save(tmp_path / "ordinary.npz")
    with pytest.raises(ValueError, match="CPU float64"):
        checkpoint.load(ordinary, backend="torch", config=learner().config)
    monkeypatch.setattr(generic, "NeuralGraph", lambda *a, **k: pytest.fail("backend constructed"))
    with pytest.raises(ValueError, match="CPU float64"):
        cd.Brain.compose(
            2,
            2,
            modules=(4,),
            backend="torch",
            learning_qualified=True,
            learning_nudged_steps=1024,
            learning_initial_damping=2,
        )


def test_composed_archive_keeps_optional_policy_and_saved_continuation(tmp_path):
    brain = cd.Brain.compose(
        2,
        2,
        modules=(4,),
        seed=3,
        learning_qualified=True,
        learning_free_steps=256,
        learning_nudged_steps=256,
        learning_initial_damping=2,
    )
    cue = np.array([[0.2, 0.8]])
    brain.lesson(cue, np.array([0]))
    brain.act(cue)
    assert brain.pending_feedback
    path = brain.save(tmp_path / "life.npz")
    with np.load(path, allow_pickle=False) as data:
        assert json.loads(str(data["meta"]))["format"] == "cadence-checkpoint/4"
        assert json.loads(str(data["brain"]))["format"] == "cadence-brain/1"
    back = cd.Brain.load(path)
    assert back.learner.config.initial_damping == 2
    assert back.learner.config == brain.learner.config
    for owner in (brain, back):
        owner.learn(np.array([0.4]), np.array([False]), cue)
        owner.lesson(cue, np.array([1]))
    for name, value in snapshot(brain.learner).items():
        np.testing.assert_array_equal(snapshot(back.learner)[name], value)
    assert np.array_equal(brain.act(cue, greedy=True), back.act(cue, greedy=True))


def test_default_archive_omits_only_derived_reports_and_load_restores_public_diagnostics(tmp_path):
    brain = cd.Brain.compose(2, 2, modules=(4,))
    report = {
        "free_damping_halvings": 2.0,
        "free_initial_damping": 0.0,
        "free_relative_damping_halvings": 2.0,
        "demonstration_nudged_damping_halvings": 1.0,
        "demonstration_nudged_initial_damping": 0.0,
        "demonstration_nudged_relative_damping_halvings": 1.0,
        "custom_initial_damping": 0.0,
    }
    brain.last_learning = report.copy()
    path = brain.save(tmp_path / "ordinary-report.npz")
    with np.load(path, allow_pickle=False) as data:
        saved = json.loads(str(data["brain"]))["last_learning"]
    assert saved == {
        "free_damping_halvings": 2.0,
        "demonstration_nudged_damping_halvings": 1.0,
        "custom_initial_damping": 0.0,
    }
    assert brain.last_learning == report
    resumed = cd.Brain.load(path)
    assert resumed.last_learning == report
    with np.load(resumed.save(tmp_path / "resaved.npz"), allow_pickle=False) as data:
        assert json.loads(str(data["brain"]))["last_learning"] == saved


@pytest.mark.parametrize("policy", [0, 2])
def test_archive_preserves_nonzero_or_nonredundant_report_state_and_explicit_policy(tmp_path, policy):
    brain = cd.Brain.compose(
        2, 2, modules=(4,), learning_qualified=True,
        learning_nudged_steps=1024, learning_initial_damping=policy,
    )
    report = {
        "free_damping_halvings": 2.0,
        "free_initial_damping": 2.0,
        "free_relative_damping_halvings": 0.0,
        "demonstration_nudged_damping_halvings": 1.0,
        "demonstration_nudged_initial_damping": 0.0,
        "demonstration_nudged_relative_damping_halvings": 2.0,
    }
    if policy:
        # Even redundant zero diagnostics remain fully explicit under format4.
        report.update(opposite_damping_halvings=1.0, opposite_initial_damping=0.0,
                      opposite_relative_damping_halvings=1.0)
    brain.last_learning = report.copy()
    path = brain.save(tmp_path / "actual-report.npz")
    with np.load(path, allow_pickle=False) as data:
        assert json.loads(str(data["brain"]))["last_learning"] == report
    assert brain.last_learning == report and cd.Brain.load(path).last_learning == report


def test_composed_gene_and_retune_keep_the_optional_index_integer():
    brain = cd.Brain.compose(
        2,
        2,
        modules=(4,),
        learning_qualified=True,
        learning_nudged_steps=1024,
        learning_initial_damping=np.int64(2),
    )
    assert brain.learner.config.initial_damping == 2
    assert type(brain.learner.config.initial_damping) is int
    assert brain.describe()["genes"]["learning_initial_damping"] == 2
    brain.retune(learning_initial_damping=1)
    assert brain.learner.config.initial_damping == 1
    brain.retune(learning_initial_damping=0, learning_qualified=False)
    assert brain.learner.config.initial_damping == 0 and not brain.learner.config.qualified
