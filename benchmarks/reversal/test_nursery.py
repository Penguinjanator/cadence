"""Guards of the odour nursery: its frozen protocol, its world, its arms and its gates."""

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("odour_nursery", HERE / "odour_nursery.py")
nursery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nursery)


@pytest.fixture(scope="module")
def protocol():
    return json.loads((HERE / "protocol.json").read_text())


@pytest.fixture(scope="module")
def short(protocol):
    """The protocol with a short stay under each later rule, for quick lives."""
    return {**protocol, "after": 300}


def test_the_protocol_is_the_frozen_one(protocol):
    assert protocol["schema"] == "odour-nursery/1"
    assert protocol["exposures"] == [100, 300, 1000, 3000, 10000] and protocol["after"] == 600
    seeds = protocol["seeds"]
    assert not set(seeds["development"]) & set(seeds["confirmation"])
    assert seeds["confirmation"] == list(range(100, 110))
    assert protocol.get("payoff", "sugar") == "sugar" and protocol["reliability"] == 1.0
    assert cd.ArousalConfig(**protocol["arousal"]) == cd.ArousalConfig()  # the founders
    assert protocol["operating_point"] == {
        "modules": [32],
        "trace_amplitude": 0.3,
        "consolidation": 0.25,
        "actor_eta": 0.1,
    }
    gates = protocol["gates"]
    assert (gates["final"], gates["stable"], gates["share"], gates["lag_bound"]) == (
        0.9,
        0.95,
        0.9,
        150,
    )


def test_the_world_pays_the_sugar_odour_and_keeps_the_stable_pair():
    for sugar in (0, 1):
        for odour in range(4):
            best = nursery.optimal(odour, sugar)
            assert nursery.reward_of(odour, nursery.AVOID, sugar) == 0.0
            paid = nursery.reward_of(odour, nursery.APPROACH, sugar)
            assert paid == (1.0 if best == nursery.APPROACH else -1.0)
    assert [nursery.optimal(o, 0) for o in range(4)] == [1, 0, 1, 0]
    assert [nursery.optimal(o, 1) for o in range(4)] == [0, 1, 1, 0]


def test_the_live_arm_acquires_reverses_returns_and_keeps_the_stable_pair(short):
    life = nursery.run_life("live", 0, 300, short)
    for phase in life["phases"]:
        assert phase["final"] >= 0.9 and phase["lag"] is not None
        assert phase["stable"] == 1.0 and phase["aroused_late"] <= 0.2
    assert life["phases"][1]["end_greedy"] == [0, 1, 1, 0]
    assert life["phases"][2]["end_greedy"] == [1, 0, 1, 0]
    assert life["phases"][1]["first_try"] is not None and life["phases"][1]["flip"] is not None
    work = life["work"]
    assert work["routine"] > 2 * work["aroused"] > 0  # most of the life is routine
    assert nursery.gates([life], short)["300"]["reversed"] == 1.0


def test_the_controls_bracket_the_task(short):
    rows = {arm: nursery.run_life(arm, 0, 300, short) for arm in ("frozen", "random", "tabular")}
    assert rows["frozen"]["phases"][0]["final"] >= 0.9  # the same brain as live under rule A
    assert rows["frozen"]["phases"][1]["final"] < 0.7  # and no adaptation without outcomes
    assert all(0.3 < p["final"] < 0.7 for p in rows["random"]["phases"])
    assert rows["tabular"]["phases"][0]["final"] >= 0.85
    released = nursery.run_life("defaults", 0, 100, {**short, "after": 100})
    assert all(p["final"] < 0.75 for p in released["phases"])  # locked on one action


def test_a_life_saved_during_the_reversal_continues_identically(protocol, tmp_path):
    warnings.simplefilter("ignore")
    life = nursery.make_life("live", protocol, 1, protocol["arousal"])
    odours = np.random.default_rng(protocol["odour_seed"] + 1)
    odour = int(odours.integers(4))
    action, _ = life.act(odour, None)
    moments = [(0, 300), (1, 12)]  # rule A, then twelve trials into rule B
    for sugar, trials in moments:
        for _ in range(trials):
            reward = nursery.reward_of(odour, action, sugar)
            odour = int(odours.integers(4))
            action, _ = life.act(odour, reward)
    assert life.brain.arousal.aroused  # saved awake, in the middle of the repair
    twin = nursery.BrainLife(cd.Brain.load(life.brain.save(tmp_path / "life.npz")), use_live=True)
    theirs, mine, other = action, [], []
    for _ in range(80):
        reward = nursery.reward_of(odour, action, 1)
        reward_twin = nursery.reward_of(odour, theirs, 1)
        odour = int(odours.integers(4))
        action, _ = life.act(odour, reward)
        theirs, _ = twin.act(odour, reward_twin)
        mine.append(action)
        other.append(theirs)
    assert mine == other
    assert twin.brain.arousal.to_dict() == life.brain.arousal.to_dict()


def test_gates_pool_the_exposures_and_count_a_crash_as_a_failure(protocol):
    def life(exposure, finals, lag=20, stable=1.0, late=0.0):
        phases = [{"final": f, "lag": lag, "stable": stable, "aroused_late": late} for f in finals]
        return {"arm": "live", "seed": 0, "exposure": exposure, "phases": phases}

    good = [life(e, (1.0, 0.95, 0.9)) for e in (300, 1000, 3000) for _ in range(4)]
    report = nursery.gates(good, protocol)
    assert report["passed"] and report["pooled"]["lives"] == 12
    assert report["reversal_lag"] == {"300": 20.0, "1000": 20.0, "3000": 20.0}
    slow = good + [life(10000, (1.0, 1.0, 1.0), lag=400)]
    assert not nursery.gates(slow, protocol)["passed"]  # the lag bound holds per exposure
    stuck = good[:-2] + [life(3000, (1.0, 0.6, 1.0), lag=None)] * 2
    report = nursery.gates(stuck, protocol)
    assert not report["passed"] and report["pooled"]["reversed"] == pytest.approx(10 / 12)
    crashed = good + [{"arm": "live", "seed": 9, "exposure": 300, "error": "RuntimeError"}]
    report = nursery.gates(crashed, protocol)
    assert not report["passed"] and report["pooled"]["crashed"] == 1
    ungated = [life(100, (0.5, 0.5, 0.5))]
    assert "passed" not in nursery.gates(ungated, protocol)  # exposure 100 carries no gate
