import numpy as np
import pytest

import cadence as cd
from cadence.brains import couple, sensor_motor


def test_regions_exchange_feedback_in_one_equilibrium():
    eye = cd.Wiring.from_edges(1, pre=[], post=[])
    regions = {"vision": eye, "movement": sensor_motor(1)}
    bridges = [("vision", 0, "movement", 0, 0.5), ("movement", 1, "vision", 0, -0.2)]
    drive = np.array([0.6, 0.0, 0.0, 0.0])
    rule = cd.GradedRule(gain=1, slope=2, threshold=0, leak=1, dt=0.25, clamp_amplitude=1)
    results = []
    for links in [bridges, bridges[:1], []]:
        wiring = couple(regions, links)
        assert wiring.sets["movement"] == (1, 2, 3)
        assert wiring.sets["movement/motor"] == (2, 3)
        engine = cd.Settlement(wiring, rule)
        state = engine.settle(drive, steps=400, tolerance=0)
        assert engine.residual(drive, state)[0] < 1e-10
        results.append(state.activation)
    joint, forward, disconnected = results
    assert 0 < joint[2] < forward[2]
    assert joint[0] < forward[0]  # Motor readback changes the sensory equilibrium.
    np.testing.assert_array_equal(disconnected[1:], [0, 0, 0])
    # Equation residual is independent of a small integration step.
    v = np.arctanh(joint)
    assert abs(v[0] - (0.6 - 0.2 * joint[2])) < 1e-10
    assert abs(v[1] - 0.5 * joint[0]) < 1e-10


def test_couple_preserves_contact_counts_signs_and_named_ports():
    region = cd.Wiring.from_edges(2, pre=[0], post=[1], count=[3], sign=[-0.4], sets={"out": [1]})
    wiring = couple({"left": region, "right": region}, [("left", 1, "right", 0, 0.2)])
    edges = dict(
        zip(zip(wiring.pre, wiring.post, strict=True), wiring.count * wiring.sign, strict=True)
    )
    assert edges == {(0, 1): -1.2000000000000002, (2, 3): -1.2000000000000002, (1, 2): 0.2}
    assert wiring.sets["right/out"] == (3,)
    assert region.edges == 1 and region.sets["out"] == (1,)


@pytest.mark.parametrize(
    "bridge",
    [
        ("absent", 0, "a", 1, 1),
        ("a", -1, "a", 1, 1),
        ("a", 2, "a", 0, 1),
        ("a", True, "a", 0, 1),
        ("a", 0, "a", 0, 1),
        ("a", 0, "a", 1, float("nan")),
    ],
)
def test_couple_rejects_invalid_bridges(bridge):
    region = cd.Wiring.from_edges(2, pre=[], post=[])
    with pytest.raises(ValueError):
        couple({"a": region}, [bridge])


def test_couple_requires_unambiguous_region_names():
    with pytest.raises(ValueError):
        couple({})
    with pytest.raises(ValueError):
        couple({"a/b": cd.Wiring.from_edges(1, pre=[], post=[])})
