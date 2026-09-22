"""The belief patch: the adjoint against finite differences, imagination's contract, the store,
custody, and scene and action meeting inside the repair."""

from __future__ import annotations

import numpy as np

from cadence.belief import BeliefPatch
from cadence.ports import DenseBlock, MapBlock, StructuredPort


def _patch(seed: int = 0, iterations: int = 2) -> BeliefPatch:
    port = StructuredPort(
        1 * 5 * 5 + 3, [MapBlock(0, 1, 5, 5, 2, 3, 2), DenseBlock(25, 3, 4)], broadcast=(25, 3)
    )
    return BeliefPatch(
        port,
        actions=3,
        belief=6,
        outputs=4,
        iterations=iterations,
        damping=0.5,
        cells=64,
        active=4,
        record_width=8,
        seed=seed,
    )


def _data(rng, n=2, t=4, patch=None):
    patch = patch or _patch()
    o = rng.random((n, t, patch.inputs))
    a = rng.random((n, t, patch.actions))
    y = rng.normal(size=(n, t, patch.outputs))
    return o, a, y


def test_the_adjoint_matches_finite_differences_through_iterations_and_the_transition():
    rng = np.random.default_rng(0)
    patch = _patch()
    o, a, y = _data(rng, patch=patch)
    # a nonzero readout and a live boundary so every path carries gradient
    params = patch.parameters()
    params["C"] = rng.normal(size=params["C"].shape) * 0.3
    patch.set_parameters(params)
    patch.reset()
    patch.assimilate(o[:, :1], a[:, :1])
    boundary = patch.state
    result = patch.observe(o, a, y, rate=0.0, write=False)
    base = patch.parameters()

    def loss(p):
        m = BeliefPatch.restore(patch.snapshot())
        m.set_parameters(p)
        m._state = boundary.copy()
        rec = m._forward(o, a, m._boundary(len(a), None), np.ones(a.shape[1], dtype=bool))
        return m._path(rec, y).loss

    worst = 0.0
    for key in base:
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in base[key].shape)
            up = {k: v.copy() for k, v in base.items()}
            dn = {k: v.copy() for k, v in base.items()}
            up[key][idx] += 1e-6
            dn[key][idx] -= 1e-6
            fd = (loss(up) - loss(dn)) / 2e-6
            err = abs(fd - result.delta[key][idx]) / max(1.0, abs(fd))
            worst = max(worst, err)
            assert err < 1e-5, (key, fd, result.delta[key][idx])
    assert worst < 1e-5


def test_imagination_consumes_no_observation_and_changes_nothing():
    rng = np.random.default_rng(1)
    patch = _patch()
    o, a, y = _data(rng, patch=patch)
    patch.reset()
    patch.observe(o, a, y, rate=0.1)
    before = patch.snapshot()
    path = patch.imagine(rng.random((2, 5, patch.actions)))
    after = patch.snapshot()
    assert path.belief.shape == (2, 5, patch.belief) and np.isfinite(path.output).all()
    assert all(np.array_equal(before[k], after[k]) for k in before)
    assert np.allclose(path.residual, 0.0)  # nothing to repair without evidence


def test_one_write_moves_the_read_toward_the_outcome():
    rng = np.random.default_rng(2)
    patch = _patch()
    o, a, y = _data(rng, patch=patch)
    patch.reset()
    first = patch.observe(o, a, y, rate=0.0, write=True)
    patch.reset()
    again = patch.assimilate(o, a)
    assert first.writes == 8
    assert np.abs(again.output - y).mean() < np.abs(first.path.output - y).mean()


def test_custody_and_state():
    rng = np.random.default_rng(3)
    patch = _patch()
    o, a, y = _data(rng, patch=patch)
    patch.reset()
    patch.observe(o, a, y, rate=0.05)
    copy = BeliefPatch.restore(patch.snapshot())
    assert np.allclose(copy.state, patch.state)
    more = rng.random((2, 3, patch.actions))
    assert np.allclose(copy.imagine(more).output, patch.imagine(more).output)


def test_scene_and_action_meet_inside_the_repair():
    """The mixed difference of the belief over (observation, action) is nonzero: the repair map
    reads the encoded evidence and the transition's expectation together."""
    rng = np.random.default_rng(4)
    patch = _patch()
    params = patch.parameters()
    params["C"] = rng.normal(size=params["C"].shape)
    patch.set_parameters(params)
    o1, o2 = rng.random((1, 1, patch.inputs)), rng.random((1, 1, patch.inputs))
    a1, a2 = rng.random((1, 1, patch.actions)), rng.random((1, 1, patch.actions))

    def y(o, a):
        patch.reset()
        return patch.assimilate(o, a).slow_output[0, 0]

    mixed = np.abs(y(o1, a1) - y(o1, a2) - y(o2, a1) + y(o2, a2)).max()
    assert mixed > 1e-6


def test_the_torch_backend_matches_the_library_forward_and_gradient():
    torch = __import__("pytest").importorskip("torch")
    from cadence.belief_torch import TorchBelief

    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(5)
    patch = _patch()
    params = patch.parameters()
    params["C"] = rng.normal(size=params["C"].shape) * 0.3
    patch.set_parameters(params)
    twin = TorchBelief(
        patch.port,
        patch.actions,
        patch.belief,
        patch.outputs,
        iterations=patch.iterations,
        damping=patch.damping,
        record_width=patch.record_width,
    )
    twin.load(patch.parameters())
    o, a, y = _data(rng, patch=patch)
    patch.reset()
    result = patch.observe(o, a, y, rate=0.0, write=False)
    beliefs, outputs = twin(torch.as_tensor(o), torch.as_tensor(a))
    assert np.allclose(beliefs.detach().numpy(), result.path.belief, atol=1e-10)
    assert np.allclose(outputs.detach().numpy(), result.path.slow_output, atol=1e-10)
    loss = 0.5 * ((outputs - torch.as_tensor(y)) ** 2).mean()
    loss.backward()
    assert np.isclose(loss.item(), result.path.loss)
    assert np.allclose(twin.C.grad.numpy(), result.delta["C"], atol=1e-9)
    assert np.allclose(twin.F.grad.numpy(), result.delta["F"], atol=1e-9)
    assert np.allclose(twin.T.grad.numpy(), result.delta["T"], atol=1e-9)
    assert np.allclose(
        np.concatenate([w.grad.numpy().ravel() for w in twin.E.weights]),
        result.delta["E"],
        atol=1e-9,
    )
    again = BeliefPatch.restore(patch.snapshot())
    again.set_parameters(twin.export())
    again.reset()
    assert np.allclose(again.assimilate(o, a).slow_output, result.path.slow_output)
    imagined = (
        twin(None, torch.as_tensor(a), state=torch.as_tensor(result.path.final_state))[1]
        .detach()
        .numpy()
    )
    patch.reset()
    assert np.allclose(
        imagined, patch.imagine(a, state=result.path.final_state).slow_output, atol=1e-10
    )
    torch.set_default_dtype(torch.float32)
