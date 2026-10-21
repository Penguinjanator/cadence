"""The readback types the public API hands back: each one obtained from its real producer,
checked to be the exported class, frozen where it is declared frozen, and consistent with the
inputs that produced it."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import cadence as cd


def _frozen(result: object, name: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, name, None)


# ---------------------------------------------------------------------------- temporal patch
def test_temporal_phase_observation_and_readback() -> None:
    rng = np.random.default_rng(4)
    net = cd.TemporalPatchNet(3, 6, 2, seed=5)
    inputs, target = rng.normal(size=(2, 4, 3)) * 0.2, rng.normal(size=(2, 4, 2)) * 0.2
    phase = net.imagine(inputs)
    assert isinstance(phase, cd.TemporalPhase)
    _frozen(phase, "hidden")
    assert phase.hidden.shape == (2, 4, 6) and phase.output.shape == (2, 4, 2)
    assert phase.converged and phase.reason

    result = net.observe(inputs, target, beta=0.01, rate=0.1)
    assert isinstance(result, cd.TemporalObservation)
    assert isinstance(result.free, cd.TemporalPhase)
    _frozen(result, "updated")
    assert result.updated == (result.reason == "updated")
    assert result.free.output.shape == target.shape

    readback = net.readback()
    assert isinstance(readback, cd.TemporalReadback)
    _frozen(readback, "state")
    assert readback.state is not None and readback.state.shape == (2, 6)
    assert readback.updates == int(result.updated)


def test_constraint_report() -> None:
    rng = np.random.default_rng(6)
    net = cd.TemporalPatchNet(4, 12, 3, seed=127)
    inputs = rng.normal(size=(1, 5, 4)) * 0.2
    report = cd.TemporalMemory().protect(net, inputs, state=np.zeros((1, 12)))
    assert isinstance(report, cd.ConstraintReport)
    _frozen(report, "bytes")
    assert set(report.ranks) == {"A", "B", "C"} and report.ranks["B"] <= 4
    assert report.bytes > 0 and report.maximum_residual < 1e-12


# ---------------------------------------------------------------------------- continuous brains
def test_patch_observation_equilibrium_and_ep_structure() -> None:
    p = cd.PatchNet.create(2, 5, 1, seed=3, density=1, tolerance=1e-7)
    structure = cd.ep_structure(p.brain)
    assert isinstance(structure, cd.EPStructure)
    _frozen(structure, "tolerance")
    assert structure.compatible and structure.to_dict()["compatible"] is True

    x = p.stimulus(np.array([[0.0, 1.0], [1.0, 0.0]]), amplitude=2)
    phase = p.settle(x)
    assert isinstance(phase, cd.Equilibrium)
    _frozen(phase, "steps")
    assert phase.residual.shape == (2,) and np.all(phase.converged)

    report = p.observe(x, np.array([[0.2], [0.8]]), source_id="one")
    assert isinstance(report, cd.PatchObservation)
    assert isinstance(report.free, cd.Equilibrium)
    _frozen(report, "updated")
    assert report.updated and report.reason == "updated"
    assert all(np.all(q.converged) for q in (report.free, report.plus, report.minus))


def test_equilibrium_from_brain_equilibrate() -> None:
    brain = cd.NeuralGraph(cd.layered(2, 3, 2, seed=4), cd.learning_neuron_model())
    drive = np.zeros((2, brain.connectome.n))
    drive[:, :2] = [[1.0, 0.5], [0.2, 0.8]]
    result = brain.equilibrate(drive, budget=256, tolerance=1e-6)
    assert isinstance(result, cd.Equilibrium)
    _frozen(result, "residual")
    assert result.residual.shape == (2,) and np.isfinite(result.residual).all()
    assert 0 < result.steps <= 256


def test_settlement_record() -> None:
    graph = cd.layered(2, 3, 2, seed=4)
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model(), backend="cpu", precision="float64")
    drive = np.zeros((2, graph.n))
    drive[:, :2] = [[1.0, 0.5], [0.2, 0.8]]
    records: list[cd.SettlementRecord] = []
    with cd.record_settlements(records.append, label="probe"):
        state = brain.settle_batch(drive, steps=5)
    (record,) = records
    assert isinstance(record, cd.SettlementRecord)
    _frozen(record, "label")
    assert record.label == "probe" and record.steps == state.steps == 5
    assert record.potential.shape == record.activation.shape == (6, 2, graph.n)
    np.testing.assert_array_equal(record.drive, drive)
