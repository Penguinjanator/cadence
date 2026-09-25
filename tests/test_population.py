import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cadence import RecordPatchNet  # noqa: E402
from cadence.population import PopulationPatch  # noqa: E402


def _net(groups=None, seed=3):
    return RecordPatchNet(
        12, 8, 3, seed=seed, cells=64, active=4, record_rate=0.5, slowest=2.0, groups=groups
    )


@pytest.mark.parametrize("groups", [None, (3,)])
def test_prediction_parity_with_the_numpy_patch(groups) -> None:
    net = _net(groups)
    twin = PopulationPatch.from_patch(
        net, instances=2, streams=3, device="cpu", dtype=torch.float64
    )
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 12))
    out = twin.imagine(torch.as_tensor(x))
    for p in range(2):
        net.reset()
        ref = net.imagine(x[p][:, None, :])
        np.testing.assert_allclose(out["output"][p].numpy(), ref.output[:, 0], atol=1e-9)
        np.testing.assert_allclose(out["hidden"][p].numpy(), ref.hidden[:, -1], atol=1e-9)


@pytest.mark.parametrize("groups", [None, (3,)])
def test_slow_step_parity_with_the_adjoint(groups) -> None:
    net = _net(groups)
    twin = PopulationPatch.from_patch(
        net, instances=1, streams=1, device="cpu", dtype=torch.float64
    )
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 1, 12))
    t = rng.normal(size=(1, 1, 3))
    if groups is not None:
        t = np.eye(3)[rng.integers(3, size=(1, 1))]
    net.reset()
    net.observe(x, t, rate=0.3, backtrack=False, write=False)
    twin.observe(torch.as_tensor(x), torch.as_tensor(t), rate=0.3, write=False)
    ref = net.parameters()
    for name, key in (("G", "G"), ("g", "g"), ("B", "B"), ("b", "b"), ("C", "C"), ("c", "c")):
        np.testing.assert_allclose(
            twin.parameters()[key][0].numpy(), ref[name], atol=1e-9, err_msg=name
        )


def test_write_and_read_parity() -> None:
    net = _net()
    twin = PopulationPatch.from_patch(
        net, instances=1, streams=1, device="cpu", dtype=torch.float64
    )
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 1, 12))
    t = rng.normal(size=(1, 1, 3))
    net.reset()
    net.observe(x, t, rate=0.0, backtrack=False, write=True)
    twin.observe(torch.as_tensor(x), torch.as_tensor(t), rate=0.0, write=True)
    net.reset()
    ref = net.imagine(x).output[0, 0]
    out = twin.imagine(torch.as_tensor(x))["output"][0, 0].numpy()
    np.testing.assert_allclose(out, ref, atol=1e-9)
    slow = twin.imagine(torch.as_tensor(x))["slow"][0, 0].numpy()
    assert np.abs(out - t[0, 0]).max() < np.abs(slow - t[0, 0]).max()


def test_instances_and_streams_are_independent() -> None:
    twin = PopulationPatch(
        12,
        8,
        3,
        instances=3,
        streams=2,
        seed=0,
        cells=64,
        active=4,
        device="cpu",
        dtype=torch.float64,
    )
    rng = np.random.default_rng(3)
    x = torch.as_tensor(rng.normal(size=(3, 2, 12)))
    t = torch.as_tensor(rng.normal(size=(3, 2, 3)))
    before = twin.imagine(x)["output"].clone()
    twin.observe(x, t, rate=torch.tensor([0.5, 0.0, 0.5], dtype=torch.float64), write=False)
    after = twin.imagine(x)["output"]
    assert torch.allclose(before[1], after[1])  # rate zero: untouched
    assert not torch.allclose(before[0], after[0])
    # a write in stream 0 leaves stream 1's store alone
    twin.observe(x, t, rate=0.0, write=True)
    assert twin.tables[:, 0].abs().sum() > 0
    twin.tables[:, 1].zero_()
    out = twin.imagine(x)
    assert torch.allclose(out["read"][:, 1], torch.zeros_like(out["read"][:, 1]))


def test_inherit_copies_parents() -> None:
    twin = PopulationPatch(
        12,
        8,
        3,
        instances=4,
        streams=1,
        seed=0,
        cells=64,
        active=4,
        device="cpu",
        dtype=torch.float64,
    )
    rng = np.random.default_rng(4)
    x = torch.as_tensor(rng.normal(size=(4, 1, 12)))
    t = torch.as_tensor(rng.normal(size=(4, 1, 3)))
    twin.observe(x, t, rate=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64), write=False)
    twin.inherit(torch.tensor([0, 0, 0, 0]))
    out = twin.imagine(x[:1].expand(4, -1, -1))["output"]
    assert torch.allclose(out[0], out[3])
