"""Coupled life checkpoints retain native validation and exact pending custody.

These focused neural continuation tests are prepared separately from behavioral
campaigns; they must run only when the experiment coordinator releases compute.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cadence import Brain, NeuralGraph
from cadence.receipts import canonical_json

HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


heads = load_module("attention_checkpoint_heads_tests", "heads.py")
subject = load_module("attention_checkpoint_wrapper_tests", "attention_checkpoint.py")


def creature(*, count=2, copies=1, kind="heads", **options):
    cls = heads.MultiHeadBrain if kind == "heads" else heads.coupled.AttentionBrain
    extra = {"heads": count, "port_copies": copies} if kind == "heads" else {}
    return cls.compose(
        3, 2, modules=(4,), episodic=False, recall_next=False, empty_anchor=True,
        arousal=True, seed=4, learning_free_steps=1024, learning_nudged_steps=1024,
        **extra, **options)


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.array(saved[name], copy=True) for name in saved.files}


def rewrite(path, change):
    data = archive(path)
    outer = json.loads(str(data["brain"]))
    change(data, outer)
    data["brain"] = np.array(canonical_json(outer))
    with path.open("wb") as output:
        np.savez(output, **data)


def equal_archives(left, right):
    first, second = archive(left), archive(right)
    assert set(first) == set(second)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)


@pytest.mark.parametrize("kind,count,copies", [
    ("single", 1, 1), ("heads", 1, 1), ("heads", 2, 1), ("heads", 1, 2),
])
def test_fresh_composition_round_trip_keeps_exact_class_and_neural_configuration(
    tmp_path, kind, count, copies,
):
    brain = creature(kind=kind, count=count, copies=copies)
    saved = brain.save(tmp_path / "life")
    assert saved.name == "life.npz"
    restored = type(brain).load(saved)
    assert type(restored) is type(brain) and type(restored.brain) is brain._graph_type
    assert restored.mode == brain.mode and restored.empty_anchor is brain.empty_anchor
    assert restored.describe() == brain.describe()
    assert restored.learner.config == brain.learner.config
    assert restored.brain.neuron_model == brain.brain.neuron_model
    assert restored.brain.precision == brain.brain.precision
    for name in subject._LAYOUT:
        np.testing.assert_array_equal(getattr(restored.brain.layout, name),
                                      getattr(brain.brain.layout, name))
    assert restored.brain.memory is restored._pending_boundary is None
    assert restored.last_settlement is None
    equal_archives(saved, restored.save(tmp_path / "restored.npz"))


def test_ordinary_brain_loader_refuses_coupled_file_instead_of_flattening_it(tmp_path):
    brain = creature()
    saved = brain.save(tmp_path / "coupled.npz")
    with pytest.raises(ValueError, match="unsupported Brain checkpoint format"):
        Brain.load(saved)


def test_sampled_pending_waited_life_resumes_actual_feedback_and_rng_exactly(tmp_path):
    brain = creature()
    brain.live([[1., 0., 0.]])
    owner = brain.decision_id
    pending = brain._action_boundary
    brain.wait([[0., 1., 0.]])
    assert brain.pending_feedback and brain.decision_id == owner
    assert brain.brain.memory is not pending
    saved = brain.save(tmp_path / "awaiting.npz")
    restored = type(brain).load(saved)
    assert restored.pending_feedback and restored.decision_id == owner
    assert restored._pending_boundary is restored._action_boundary
    assert restored.brain.memory is not restored._action_boundary
    assert restored._pending_boundary.streams == 1
    assert restored._pending_boundary.empty_anchor
    for first, second in zip(brain._pending_boundary.memories,
                             restored._pending_boundary.memories, strict=True):
        for name in subject._BANK:
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
            assert not getattr(second, name).flags.writeable
    for left, right in ((brain, restored),):
        actual = left.live([[0., 0., 1.]], reward=[.75], decision_id=owner)
        resumed = right.live([[0., 0., 1.]], reward=[.75], decision_id=owner)
        np.testing.assert_array_equal(actual, resumed)
    equal_archives(brain.save(tmp_path / "actual.npz"),
                   restored.save(tmp_path / "continued.npz"))


def test_accepted_feedback_retains_its_distinct_captured_free_graph(tmp_path):
    brain = creature()
    brain.act([[1., 0., 0.]], greedy=False)
    before_feedback = brain.brain
    brain.learn([.75], [False], [[0., 1., 0.]], bootstrap=[0.])
    captured = brain.basal_ganglia._free_brain
    assert captured is before_feedback and captured is not brain.brain
    assert not brain.pending_feedback and brain.basal_ganglia.updates == 1
    restored = type(brain).load(brain.save(tmp_path / "accepted-feedback.npz"))
    old = restored.basal_ganglia._free_brain
    assert old is not restored.brain and type(old) is brain._graph_type
    for name in ("efficacy", "bias", "log_gain"):
        np.testing.assert_array_equal(getattr(old, name), getattr(captured, name))
    assert ((old.memory is restored.brain.memory)
            == (captured.memory is brain.brain.memory))
    for name in subject._BANK:
        np.testing.assert_array_equal(getattr(old.memory, name), getattr(captured.memory, name))
    actual = brain.act([[0., 0., 1.]], greedy=False)
    resumed = restored.act([[0., 0., 1.]], greedy=False)
    np.testing.assert_array_equal(actual, resumed)
    equal_archives(brain.save(tmp_path / "feedback-actual.npz"),
                   restored.save(tmp_path / "feedback-resumed.npz"))


def test_pending_graph_snapshot_survives_explicit_low_level_parameter_intervention(tmp_path):
    brain = creature()
    brain.act([[1., 0., 0.]], greedy=False)
    captured = brain.basal_ganglia._free_brain
    original = captured.efficacy.copy()
    original_bias = captured.bias.copy()
    # A low-level research intervention exercises the archive's graph identity
    # contract. Public lesson()/teach_saved() retain their pending-action guard.
    # This is not a supported teacher operation on an awaited action.
    changed = original.copy()
    changed[0] += .01
    brain.learner.brain = captured.with_parameters(efficacy=changed)
    assert brain.brain is not captured and brain.basal_ganglia._free_brain is captured
    assert (not np.array_equal(brain.brain.efficacy, original)
            or not np.array_equal(brain.brain.bias, original_bias))
    saved = brain.save(tmp_path / "pending-intervention.npz")
    restored = type(brain).load(saved)
    old = restored.basal_ganglia._free_brain
    assert old is not restored.brain and type(old) is brain._graph_type
    np.testing.assert_array_equal(old.efficacy, original)
    np.testing.assert_array_equal(old.bias, original_bias)
    assert old.memory is restored._pending_boundary
    assert old.memory is restored._action_boundary
    assert old.neuron_model == captured.neuron_model
    for name in subject._LAYOUT:
        np.testing.assert_array_equal(getattr(old.layout, name), getattr(captured.layout, name))
    first = brain.learn([.5], [False], [[0., 1., 0.]], bootstrap=[0.])
    second = restored.learn([.5], [False], [[0., 1., 0.]], bootstrap=[0.])
    assert first == second
    equal_archives(brain.save(tmp_path / "intervention-actual.npz"),
                   restored.save(tmp_path / "intervention-resumed.npz"))


def test_boundary_aliases_remain_immutable_and_distinct_from_later_slots(tmp_path):
    brain = creature()
    brain.act([[1., 0., 0.]], greedy=False)
    original = brain.brain.memory
    brain._forecast_boundary = original
    brain._pending_boundary = original
    brain._action_boundary = original
    restored = type(brain).load(brain.save(tmp_path / "aliases.npz"))
    bank = restored.brain.memory
    assert bank is restored._pending_boundary is restored._action_boundary
    assert bank is restored._forecast_boundary
    for name in subject._BANK:
        assert all(getattr(m, name) is getattr(bank, name) for m in bank.memories)
        assert not getattr(bank, name).flags.writeable
    before = bank.keys.copy()
    restored.stimulus([[0., 1., 0.]])
    np.testing.assert_array_equal(bank.keys, before)
    assert restored.brain.memory is not bank


def test_learned_code_pending_keys_and_tables_continue_without_reencoding_history(tmp_path):
    # The native learned-code graph uses dt1; use its existing bounded halving
    # policy instead of the raw prototype's explicit damping0 control.
    brain = creature(learned_code=2, learning_damping=3)
    assert brain.brain.neuron_model.dt == 1.0 and brain.learner.config.damping == 3
    brain.act([[1., 0., 0.]], greedy=False)
    keys = brain.recall.keys.copy()
    moment_code = brain._moment_code.copy()
    restored = type(brain).load(brain.save(tmp_path / "coded.npz"))
    np.testing.assert_array_equal(restored.recall.keys, keys)
    np.testing.assert_array_equal(restored._moment_code, moment_code)
    for name, value in brain.learned_code.arrays().items():
        np.testing.assert_array_equal(restored.learned_code.arrays()[name], value)
    actual = brain.learn([.25], [False], [[0., 1., 0.]], bootstrap=[0.])
    resumed = restored.learn([.25], [False], [[0., 1., 0.]], bootstrap=[0.])
    assert actual == resumed
    equal_archives(brain.save(tmp_path / "coded-actual.npz"),
                   restored.save(tmp_path / "coded-resumed.npz"))


def test_loaded_feedback_refusal_retains_same_pending_custody_and_retry_state(tmp_path):
    brain = creature()
    brain.act([[1., 0., 0.]], greedy=False)
    brain.learner.config = replace(brain.learner.config, free_steps=0, tolerance=1e-12)
    restored = type(brain).load(brain.save(tmp_path / "refusal.npz"))
    for instance in (brain, restored):
        boundary = instance._pending_boundary
        with pytest.raises(RuntimeError):
            instance.learn([.5], [False], [[0., 1., 0.]])
        assert instance.pending_feedback and instance._pending_boundary is boundary
    equal_archives(brain.save(tmp_path / "refused-actual.npz"),
                   restored.save(tmp_path / "refused-resumed.npz"))


@pytest.mark.parametrize("damage", ["missing_pending", "other_pending", "missing_action",
                                    "missing_graph", "streams", "stale_pending"])
def test_corrupt_action_custody_is_refused_at_load_before_feedback(tmp_path, damage):
    brain = creature()
    brain.act([[1., 0., 0.]], greedy=damage == "stale_pending")
    if damage != "stale_pending":
        brain.wait([[0., 1., 0.]])
    saved = brain.save(tmp_path / "action-custody.npz")

    def corrupt(data, outer):
        meta, base = outer["attention"], outer["base"]
        refs = meta["boundary_refs"]
        if damage == "missing_pending":
            refs["_pending_boundary"] = None
        elif damage == "other_pending":
            refs["_pending_boundary"] = meta["graphs"][0]["boundary"]
        elif damage == "missing_action":
            refs["_action_boundary"] = None
        elif damage == "missing_graph":
            # The valid current graph remains in the archive. Erasing only the
            # actor's graph identity must not quietly change its warm-cache use.
            meta["free_graph"], base["free_current"] = None, False
        elif damage == "streams":
            boundary = refs["_action_boundary"]
            for name in ("keys", "values", "held"):
                key = f"attention/boundary/{boundary}/{name}"
                data[key] = np.repeat(data[key], 2, axis=0)
        else:
            refs["_pending_boundary"] = refs["_action_boundary"]

    rewrite(saved, corrupt)
    with pytest.raises(ValueError, match="boundary|captured free"):
        type(brain).load(saved)


@pytest.mark.parametrize("damage", ["format", "base", "class", "heads", "anchor",
                                    "reference", "parameters", "layout", "extra"])
def test_corrupt_attention_metadata_or_graph_state_never_loads_as_plain_brain(tmp_path, damage):
    brain = creature()
    saved = brain.save(tmp_path / "corrupt.npz")

    def corrupt(data, outer):
        meta = outer["attention"]
        if damage == "format":
            outer["format"] = "cadence-attention-brain/99"
        elif damage == "base":
            outer["base"]["format"] = "cadence-brain/99"
        elif damage == "class":
            meta["class"] = "Brain"
        elif damage == "heads":
            meta["heads"] = True
        elif damage == "anchor":
            meta["empty_anchor"] = "true"
        elif damage == "reference":
            meta["boundary_refs"]["_action_boundary"] = 0
        elif damage == "parameters":
            data["attention/graph/0/efficacy"][0] += .1
        elif damage == "layout":
            data["attention/graph/0/layout/edge_index"][0] += 1
        else:
            data["attention/unused"] = np.zeros(1)

    rewrite(saved, corrupt)
    with pytest.raises(ValueError):
        type(brain).load(saved)


def test_native_optimizer_corruption_is_still_refused_by_native_loader(tmp_path):
    brain = creature()
    saved = brain.save(tmp_path / "optimizer.npz")
    rewrite(saved, lambda data, outer: data.update(velocity=np.zeros(1)))
    with pytest.raises(ValueError, match="optimizer"):
        type(brain).load(saved)


def test_unknown_subclass_and_plain_graph_are_refused_before_writing(tmp_path):
    class OtherBrain(heads.MultiHeadBrain):
        pass

    brain = creature()
    with pytest.raises(ValueError, match="supported benchmark class"):
        subject.load(tmp_path / "absent.npz", cls=OtherBrain)
    original = brain.brain
    brain.learner.brain = NeuralGraph(original.connectome, original.neuron_model)
    with pytest.raises(ValueError, match="exact coupled CPU graph"):
        brain.save(tmp_path / "plain.npz")
    assert not (tmp_path / "plain.npz").exists()


def test_incomplete_final_write_preserves_previous_checkpoint_and_cleans_temporary_files(
    tmp_path, monkeypatch,
):
    brain = creature()
    saved = subject.save(brain, tmp_path / "retained.npz")
    previous = saved.read_bytes()

    def refused(*args, **kwargs):
        raise OSError("simulated final archive failure")

    monkeypatch.setattr(subject, "_write", refused)
    with pytest.raises(OSError, match="simulated"):
        subject.save(brain, saved)
    assert saved.read_bytes() == previous
    assert not list(tmp_path.glob(".attention-checkpoint-*"))


@pytest.mark.parametrize("kwargs", [{"backend": "torch"}, {"device": "cpu"}])
def test_unsupported_checkpoint_backend_cannot_change_equations(tmp_path, kwargs):
    brain = creature()
    saved = brain.save(tmp_path / "cpu.npz")
    with pytest.raises(ValueError, match="CPU only"):
        type(brain).load(saved, **kwargs)
