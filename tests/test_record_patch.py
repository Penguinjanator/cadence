"""The record patch: an exact adjoint, its detuning identity, one-shot records and isolation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cadence import RecordPatchNet


def same_snapshot(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert_array_equal(left[key], right[key])


def independent_path(parameters, inputs, boundary, net=None, read=None):
    """A per-row loop of the documented equations.

    The record read is taken from ``net`` (its records and reading scale), from a
    fixed ``read`` array, or is zero."""
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    batch, horizon, _ = inputs.shape
    hidden = np.zeros((batch, horizon, parameters["B"].shape[0]))
    outputs = []
    for row in range(batch):
        h = boundary[row].copy()
        for t in range(horizon):
            u = inputs[row, t]
            gate = sigmoid(parameters["g"] + parameters["G"] @ u)
            z = np.tanh(parameters["B"] @ u + parameters["b"])
            h = gate * h + (1.0 - gate) * z
            hidden[row, t] = h
            m = np.zeros(parameters["c"].shape)
            if net is not None:
                code = net.records.code(np.concatenate((u * (np.sqrt(net.inputs) / net._input_norm), h * net._scale)))[0, 0]
                m = code @ net.records.tables["y"]
            elif read is not None:
                m = read[row, t]
            outputs.append(parameters["C"] @ h + parameters["c"] + m)
    return hidden, np.array(outputs).reshape(batch, horizon, -1)


def differences(function, point, step=1e-6):
    result = np.empty_like(point)
    for index in np.ndindex(point.shape):
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        result[index] = (function(plus) - function(minus)) / (2 * step)
    return result


def test_free_path_matches_the_documented_recurrence_and_is_bounded():
    rng = np.random.default_rng(3)
    net = RecordPatchNet(3, 5, 2, seed=11, cells=200, active=8)
    inputs = rng.normal(size=(2, 7, 3)) * 3.0
    boundary = rng.uniform(-1, 1, size=(2, 5))
    path = net.imagine(inputs, state=boundary)
    hidden, output = independent_path(net.parameters(), inputs, boundary)
    assert_allclose(path.hidden, hidden, atol=1e-12)
    assert_allclose(path.output, output, atol=1e-12)  # empty records read zero
    assert np.all(np.abs(path.hidden) < 1.0)
    assert np.all((path.gate > 0) & (path.gate < 1))


def test_records_change_the_output_through_the_read_port_only():
    rng = np.random.default_rng(4)
    net = RecordPatchNet(3, 5, 2, seed=11, cells=300, active=8, record_rate=1.0)
    inputs, target = rng.normal(size=(1, 4, 3)), rng.normal(size=(1, 4, 2))
    net.observe(inputs, target, rate=0.0)
    boundary = np.zeros((1, 5))
    path = net.imagine(inputs, state=boundary)
    hidden, output = independent_path(net.parameters(), inputs, boundary, net=net)
    assert_allclose(path.hidden, hidden, atol=1e-12)
    assert_allclose(path.output, output, atol=1e-12)
    assert np.abs(path.read).max() > 0.0


@pytest.mark.parametrize("precision", [np.ones(2), np.array([0.3, 2.5])])
def test_adjoint_gradient_matches_central_differences(precision):
    rng = np.random.default_rng(5)
    net = RecordPatchNet(2, 3, 2, seed=8, cells=64, active=4, output_precision=precision)
    inputs, target = rng.normal(size=(2, 4, 2)), rng.normal(size=(2, 4, 2))
    net.observe(inputs, target, rate=0.0)  # fill a few records so the read port is live
    net.reset()
    parameters = net.parameters()
    result = net.observe(inputs, target, rate=0.0, write=False)
    assert result.updated and result.delta is not None
    assert np.abs(result.prediction.read).max() > 0.0  # the read is live, and outside the loss

    def loss(name, value):
        _, output = independent_path({**parameters, name: value}, inputs, np.zeros((2, 3)))
        return 0.5 * np.mean(precision * (output - target) ** 2)

    for name, value in parameters.items():
        numerical = differences(lambda v, name=name: loss(name, v), value)
        assert_allclose(result.delta[name], numerical, atol=1e-8, rtol=1e-6)


def test_detuned_contrast_equals_the_adjoint_gradient():
    rng = np.random.default_rng(6)
    net = RecordPatchNet(3, 6, 2, seed=9, cells=128, active=6, output_precision=[1.0, 2.0])
    inputs, target = rng.normal(size=(2, 9, 3)), rng.normal(size=(2, 9, 2))
    net.observe(inputs, target, rate=0.0)
    net.reset()
    adjoint = net.observe(inputs, target, rate=0.0, write=False).delta
    net.reset()
    before = net.snapshot()
    contrast = net.detune(inputs, target, beta=1e-4)
    assert contrast.converged and contrast.plus_energy > 0 and contrast.minus_energy < 0
    for name, value in adjoint.items():
        error = np.linalg.norm(contrast.contrast[name] - value) / np.linalg.norm(value)
        assert error < 1e-6, (name, error)
    coarse = net.detune(inputs, target, beta=1e-2)
    fine_error = max(
        np.linalg.norm(contrast.contrast[k] - adjoint[k]) / np.linalg.norm(adjoint[k])
        for k in adjoint
    )
    coarse_error = max(
        np.linalg.norm(coarse.contrast[k] - adjoint[k]) / np.linalg.norm(adjoint[k])
        for k in adjoint
    )
    assert coarse_error > fine_error  # the contrast error shrinks with beta
    same_snapshot(before, net.snapshot())  # detuning changed nothing


def test_one_observation_is_recalled_from_the_records():
    rng = np.random.default_rng(7)
    net = RecordPatchNet(17, 8, 6, seed=2, cells=4096, active=32, record_rate=1.0)
    inputs = np.zeros((1, 16, 17))
    inputs[0, 0, 0] = 1.0  # one cue
    inputs[0, np.arange(16), 1 + np.arange(16)] = 1.0  # sixteen distinct heard events
    identities = rng.integers(6, size=16)
    target = np.eye(6)[identities][None]
    first = net.observe(inputs, target, rate=0.0)
    assert first.writes == 16 and first.initial_loss is not None
    recalled = net.imagine(inputs, state=np.zeros((1, 8)))
    assert np.array_equal(recalled.output[0].argmax(axis=1), identities)
    # Keys of neighbouring moments overlap through the shared context, so one
    # write is not exact; the delta rule converges over repeated observation.
    assert net._loss(recalled.output, target) < 0.3 * first.initial_loss
    second = net.observe(inputs, target, rate=0.0)
    third = net.observe(inputs, target, rate=0.0)
    fourth = net.observe(inputs, target, rate=0.0)
    assert second.prediction.loss < 0.3 * first.prediction.loss
    assert third.prediction.loss < second.prediction.loss
    assert fourth.prediction.loss < 0.05 * first.prediction.loss
    for name, value in net.parameters().items():  # the slow readout did not move at rate 0
        assert_array_equal(value, RecordPatchNet(17, 8, 6, seed=2).parameters()[name])


def test_readings_that_differ_only_by_context_are_separated_only_partly():
    """Same clock phase, different bar: the leaky context alone separates the keys
    partly. This records the measured limit; a heard stream supplies the rest."""
    rng = np.random.default_rng(7)
    net = RecordPatchNet(4, 8, 6, seed=2, cells=4096, active=32, record_rate=1.0)
    inputs = np.zeros((1, 16, 4))
    inputs[0, 0, 0] = 1.0
    inputs[0, np.arange(16), 1 + np.arange(16) % 3] = 1.0  # a three-position clock
    target = np.eye(6)[rng.integers(6, size=16)][None]
    identities = target[0].argmax(axis=1)
    for _ in range(3):
        net.observe(inputs, target, rate=0.0)
    recalled = net.imagine(inputs, state=np.zeros((1, 8)))
    correct = int((recalled.output[0].argmax(axis=1) == identities).sum())
    assert 5 <= correct <= 14


def test_the_slow_readout_learns_what_the_records_already_patch():
    """A second observation of the same path finds the record patching the error,
    yet the slow parameters keep learning it: their loss falls while the
    prediction's loss is already small."""
    rng = np.random.default_rng(12)
    net = RecordPatchNet(3, 6, 2, seed=3, cells=512, active=8, record_rate=1.0)
    inputs, target = rng.normal(size=(2, 8, 3)), rng.normal(size=(2, 8, 2))
    first = net.observe(inputs, target, rate=2.0, backtrack=True)
    net.reset()
    second = net.observe(inputs, target, rate=2.0, backtrack=True)
    net.reset()
    third = net.observe(inputs, target, rate=2.0, backtrack=True)
    assert second.prediction.loss < 0.7 * first.prediction.loss  # the record patched it
    assert third.prediction.loss < 0.5 * first.prediction.loss
    assert second.initial_loss < first.initial_loss  # and the slow readout still learned
    assert third.initial_loss < second.initial_loss
    assert second.updated and second.final_loss < second.initial_loss


def test_writes_can_be_withheld_and_learning_without_writes_still_updates():
    rng = np.random.default_rng(8)
    net = RecordPatchNet(2, 4, 1, seed=1, cells=64, active=4)
    inputs, target = rng.normal(size=(1, 5, 2)), rng.normal(size=(1, 5, 1))
    result = net.observe(inputs, target, rate=0.5, write=False)
    assert result.updated and result.writes == 0 and net.records.writes == 0
    assert net.updates == 1 and net.readback().writes == 0


def test_imagine_is_pure_and_observe_predicts_what_imagine_predicts():
    rng = np.random.default_rng(9)
    net = RecordPatchNet(3, 4, 2, seed=5, cells=128, active=8)
    inputs, target = rng.normal(size=(2, 6, 3)), rng.normal(size=(2, 6, 2))
    net.observe(inputs, target)
    net.reset()
    before = net.snapshot()
    branch = net.imagine(inputs)
    same_snapshot(before, net.snapshot())
    result = net.observe(inputs, target, rate=1.0, backtrack=True)
    assert_array_equal(result.prediction.output, branch.output)
    assert result.updated and result.accepted_rate is not None
    assert result.final_loss < result.initial_loss
    assert result.replay_calls >= 1 and net.state is not None
    assert_array_equal(net.state, result.prediction.final_state)
    # The carried context was computed before the accepted step, as in TemporalPatchNet.
    assert net.readback().state_parameter_revision == net.readback().parameter_revision - 1
    net.advance(inputs)
    assert net.readback().state_parameter_revision == net.readback().parameter_revision


def test_backtracking_rejects_a_step_that_cannot_lower_the_loss():
    net = RecordPatchNet(1, 2, 1, seed=3, cells=32, active=2)
    inputs, target = np.zeros((1, 3, 1)), np.zeros((1, 3, 1))
    net.set_parameters({**net.parameters(), "c": np.zeros(1)})
    result = net.observe(inputs, target, rate=1.0, backtrack=True)
    assert not result.updated and result.reason == "no_decreasing_parameter_step"
    assert result.replay_calls == 0 and net.updates == 0


def test_checkpoints_restore_records_state_and_counts_exactly(tmp_path):
    rng = np.random.default_rng(10)
    net = RecordPatchNet(3, 4, 2, seed=6, cells=128, active=8, output_precision=[2.0, 1.0])
    inputs, target = rng.normal(size=(2, 6, 3)), rng.normal(size=(2, 6, 2))
    net.observe(inputs, target, rate=0.3)
    net.advance(inputs)
    saved = net.save(tmp_path / "patch")
    restored = RecordPatchNet.load(saved)
    same_snapshot(net.snapshot(), restored.snapshot())
    assert restored.updates == 1 and restored.records.writes == 12
    assert_array_equal(restored.imagine(inputs).output, net.imagine(inputs).output)
    cold = RecordPatchNet.load(saved)
    cold.reset()
    assert cold.state is None and cold.readback().state_parameter_revision is None
    bad = net.snapshot()
    bad["meta"] = np.array('{"format": "cadence-temporal/1"}')
    with pytest.raises(ValueError):
        RecordPatchNet.restore(bad)
    missing = {k: v for k, v in net.snapshot().items() if k != "records_table_y"}
    with pytest.raises(ValueError):
        RecordPatchNet.restore(missing)


def test_invalid_data_changes_nothing():
    net = RecordPatchNet(2, 3, 1, seed=4, cells=32, active=2)
    before = net.snapshot()
    with pytest.raises(ValueError):
        net.observe(np.zeros((1, 2, 3)), np.zeros((1, 2, 1)))
    with pytest.raises(ValueError):
        net.observe(np.zeros((1, 2, 2)), np.full((1, 2, 1), np.nan))
    with pytest.raises(ValueError):
        net.observe(np.zeros((1, 2, 2)), np.zeros((1, 3, 1)))
    with pytest.raises(ValueError):
        net.observe(np.zeros((1, 2, 2)), np.zeros((1, 2, 1)), rate=-1.0)
    with pytest.raises(ValueError):
        net.imagine(np.zeros((1, 2, 2)), state=np.zeros((2, 3)))
    with pytest.raises(ValueError):
        net.set_parameters({**net.parameters(), "g": np.zeros(2)})
    with pytest.raises(ValueError):
        net.set_output_precision(np.array([0.0]))
    same_snapshot(before, net.snapshot())


def test_records_state_round_trip_and_witness_moves_the_mean():
    from cadence import Records

    rng = np.random.default_rng(11)
    records = Records(5, {"y": 2}, cells=64, active=4, seed=1)
    readings = rng.normal(size=(3, 5))
    twin = Records(5, {"y": 2}, cells=64, active=4, seed=1)
    twin.code(readings, adapt=True)
    records.witness(readings)
    assert_array_equal(records.mean, twin.mean) and records.seen == 3
    records.write(records.code(readings[0])[:, 0], {"y": np.array([1.0, -1.0])})
    other = Records(5, {"y": 2}, cells=64, active=4, seed=1)
    other.load_state(records.state())
    assert other.writes == 1 and other.seen == 3
    assert_array_equal(other.tables["y"], records.tables["y"])
    with pytest.raises(ValueError):
        other.load_state({k: v for k, v in records.state().items() if k != "mean"})


def test_snapshot_round_trip_carries_the_input_norm_and_record_state():
    rng = np.random.default_rng(12)
    net = RecordPatchNet(3, 5, 2, seed=8, cells=120, active=6, record_rate=0.05, record_averaging=True, record_homeostasis=0.05)
    inputs, target = rng.normal(size=(2, 9, 3)) * 2.0, rng.normal(size=(2, 9, 2))
    net.observe(inputs, target, rate=0.0)
    assert net._input_norm != 1.0
    assert net.records.count.sum() > 0 and np.abs(net.records.boost).max() > 0
    twin = RecordPatchNet.restore(net.snapshot())
    assert twin._input_norm == net._input_norm
    assert np.array_equal(twin.records.count, net.records.count)
    probe = rng.normal(size=(1, 4, 3))
    assert_allclose(twin.imagine(probe, state=np.zeros((1, 5))).output, net.imagine(probe, state=np.zeros((1, 5))).output, atol=1e-12)
