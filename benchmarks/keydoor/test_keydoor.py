"""Guards of the key-door nursery: its protocol, its world, its arms, its custody and its gates."""

import gzip
import importlib.util
import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("key_door", HERE / "key_door.py")
keydoor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keydoor)


@pytest.fixture(scope="module")
def protocol():
    return json.loads((HERE / "protocol.json").read_text())


@pytest.fixture(scope="module")
def quick(protocol):
    """The protocol with few trips per rule, for quick lives."""
    return {**protocol, "episodes": 60, "probe_every": 20}


def test_the_protocol_is_the_frozen_one(protocol):
    assert protocol["schema"] == keydoor.SCHEMA == "key-door/1"
    assert protocol["delays"] == [2, 5, 10] and protocol["episodes"] == 500
    assert (protocol["length"], protocol["jitter"]) == (14, 1)
    assert (protocol["cost"], protocol["food"], protocol["truncation"]) == (0.25, 1.0, 0.05)
    seeds = protocol["seeds"]
    assert seeds["development"] == list(range(24))
    assert seeds["confirmation"] == list(range(700, 710))
    assert not set(seeds["development"]) & set(seeds["confirmation"])
    assert protocol["tabular"] == {"alpha": 0.5, "epsilon": 0.1, "gamma": 0.9, "lam": 0.8}
    founders = cd.ArousalConfig()
    assert cd.ArousalConfig(**protocol["arousal"]) == replace(founders, need=0.03)
    assert protocol["operating_point"] == {
        "modules": [32],
        "trace_amplitude": 0.3,
        "consolidation": 0.25,
        "eta": 0.1,
        "lam": 0.95,
        "gamma": 0.95,
        "eta_critic": 5.0,
    }
    gates = protocol["gates"]
    assert (gates["fed"], gates["wrong"], gates["aroused_late"], gates["share"]) == (
        0.9,
        0.5,
        0.2,
        0.9,
    )
    assert gates["delays"] == [2, 5]


def test_every_trip_has_the_same_length_and_the_declared_order(protocol):
    rng = np.random.default_rng(0)
    for delay in protocol["delays"]:
        for _ in range(20):
            cells = keydoor.corridor(delay, protocol["length"], protocol["jitter"], rng)
            assert len(cells) == protocol["length"]
            levers = cells.count(keydoor.LEVER)
            assert abs(levers - delay) <= protocol["jitter"]
            floors = cells.count(keydoor.FLOOR)
            assert cells == [keydoor.FLOOR] * floors + [keydoor.CHEST, keydoor.LAMP] + [
                keydoor.LEVER
            ] * levers + [keydoor.DOOR]
    x = keydoor.observe(keydoor.LAMP, True, True)
    assert x.shape == (1, 6) and x[0, keydoor.LAMP] == 1.0 and x[0, 5] == 1.0
    assert keydoor.observe(keydoor.LAMP, True, False).shape == (1, 5)


def test_the_world_pays_the_door_with_the_key_and_a_cut_trip_ends_with_done_clear(
    quick, monkeypatch
):
    """The tabular learner's life is deterministic and cheap: read the world back through
    what it was handed. ``done`` is set on the observation after a door and on no other,
    so a cut trip carries its forecast over into the next (a truncated bootstrap)."""
    made = keydoor.make_life
    handed = []

    def recording(arm, protocol, seed, genes):
        life = made(arm, protocol, seed, genes)
        act = life.act

        def logged(kind, holding, reward, done):
            handed.append((kind, reward, done))
            return act(kind, holding, reward, done)

        life.act = logged
        return life

    monkeypatch.setattr(keydoor, "make_life", recording)
    row = keydoor.run_life("tabular", 0, 2, {**quick, "episodes": 40, "truncation": 0.5})
    for phase in row["phases"]:
        assert 0.0 <= phase["fed"] <= phase["took"] <= 1.0  # food needs the key
        assert phase["cut"] > 0 and phase["episodes"] + phase["cut"] == 40
        assert sum(map(sum, phase["visits"])) == phase["moments"] < 40 * quick["length"]
        assert phase["wrong"] >= 0.0
    assert row["phases"][0]["keyed"] == "chest" and row["phases"][1]["keyed"] == "lamp"
    assert handed[0][1:] == (None, False) and all(r is not None for _, r, _ in handed[1:])
    for (kind, _, _), (_, _, done) in zip(handed, handed[1:], strict=False):
        assert done == (kind == keydoor.DOOR)
    assert sum(k == keydoor.DOOR for k, _, _ in handed) == 2 * 40 - sum(
        p["cut"] for p in row["phases"]
    )


def test_the_operating_point_reaches_the_brain(protocol):
    """Every setting the protocol declares is on the living brain; the actor's bias rate
    follows the composed rule unless the point names it."""
    point = protocol["operating_point"]
    life = keydoor.make_life("live", protocol, 0, protocol["arousal"])
    actor, learner = life.brain.basal_ganglia.config, life.brain.learner.config
    for name in type(actor).__slots__:
        if name in point:
            assert getattr(actor, name) == point[name], name
    if "eta" in point and "eta_bias" not in point:
        assert actor.eta_bias == point["eta"] / 10
    for name, value in point.get("learner", {}).items():
        assert getattr(learner, name) == value, name
    assert life.brain.arousal.config == cd.ArousalConfig(**protocol["arousal"])
    genes = protocol["arousal"]
    assert keydoor.make_life("blind", protocol, 0, genes).pouch is False
    assert keydoor.make_life("lambda-zero", protocol, 0, genes).brain.basal_ganglia.config.lam == 0
    assert keydoor.make_life("step", protocol, 0, genes).brain.arousal is None


def test_the_controls_bracket_the_task(quick):
    arms = ("random", "frozen", "step", "tabular", "yoked", "lambda-zero", "blind")
    rows = {arm: keydoor.run_life(arm, 0, 2, quick) for arm in arms}
    for arm, row in rows.items():
        assert "error" not in row and len(row["phases"]) == 2, arm
    for phase in rows["random"]["phases"]:
        assert 0.05 < phase["fed"] < 0.6 and phase["executed_probability"] == 0.5
    frozen = rows["frozen"]
    assert frozen["phases"][1]["executed_probability"] == 1.0  # greedy, with certainty
    assert frozen["phases"][1]["aroused"] == 0.0 and frozen["phases"][0]["aroused"] > 0.0
    assert frozen["work"]["learning_sweeps"] > 0  # it learned under rule A only
    step = rows["step"]["phases"][1]
    assert step["behaviour_interact"] == step["policy_interact"]  # it samples its policy
    assert step["executed_probability"] < 1.0 and step["aroused"] == 1.0
    table = rows["tabular"]
    assert table["work"]["brains"] == 0 and table["work"]["probes"] == 0
    assert rows["blind"]["phases"][0]["visits"] != rows["yoked"]["phases"][0]["visits"]


def test_the_probes_do_not_disturb_the_life_they_read(quick):
    executed = ("fed", "took", "wrong", "lag", "visits", "aroused", "behaviour_interact")
    measurement = ("probes", "probe_sweeps", "checkpoints", "latency_ms")
    unprobed = {**quick, "probe_every": 10_000}
    for arm in ("live", "step"):
        probed, plain = (keydoor.run_life(arm, 5, 2, p) for p in (quick, unprobed))
        for a, b in zip(probed["phases"], plain["phases"], strict=True):
            assert [a[key] for key in executed] == [b[key] for key in executed]
        lived = {k: v for k, v in probed["work"].items() if k not in measurement}
        assert lived == {k: v for k, v in plain["work"].items() if k not in measurement}
        assert probed["work"]["probes"] > plain["work"]["probes"]


def test_a_life_saved_in_the_delay_with_its_outcome_pending_continues_identically(
    protocol, tmp_path
):
    """Save between taking the key and reaching the door, the preceding action's outcome
    still to come: the twin lives on identically and takes that outcome once."""
    warnings.simplefilter("ignore")
    cost = protocol["cost"]

    def world(kind, holding, action):
        if action != keydoor.INTERACT:
            return holding, 0.0
        if kind == keydoor.CHEST and not holding:
            return True, 0.0
        if kind == keydoor.DOOR:
            return holding, 1.0 if holding else 0.0
        return holding, 0.0 if kind == keydoor.FLOOR else -cost

    life = keydoor.make_life("live", protocol, 2, protocol["arousal"])
    rng = np.random.default_rng(99)
    pending = (None, False)
    for _ in range(12):  # twelve trips under rule A
        holding = False
        cells = keydoor.corridor(5, protocol["length"], 0, rng)
        for i, kind in enumerate(cells):
            action, _ = life.act(kind, holding, *pending)
            holding, outcome = world(kind, holding, action)
            pending = (outcome, i == len(cells) - 1)
    cells = keydoor.corridor(5, protocol["length"], 0, rng)
    first_lever = cells.index(keydoor.LEVER)
    holding = False
    for kind in cells[:first_lever]:  # the thirteenth trip up to the lamp
        action, _ = life.act(kind, holding, *pending)
        holding, outcome = world(kind, holding, action)
        pending = (outcome, False)
    twin = keydoor.BrainLife(
        cd.Brain.load(life.brain.save(tmp_path / "life.npz")), use_live=True, pouch=True
    )
    mine, theirs = [], []
    for kind in cells[first_lever:] + keydoor.corridor(5, protocol["length"], 0, rng):
        a, _ = life.act(kind, holding, *pending)
        b, _ = twin.act(kind, holding, *pending)
        mine.append(a)
        theirs.append(b)
        holding, outcome = world(kind, holding, a)
        pending = (outcome, kind == keydoor.DOOR)
    assert mine == theirs
    assert twin.brain.arousal.to_dict() == life.brain.arousal.to_dict()


def test_a_refused_answer_is_charged_and_the_crashed_life_keeps_its_ledger(quick, monkeypatch):
    made = keydoor.make_life

    def strict(arm, protocol, seed, genes):
        life = made(arm, protocol, seed, genes)
        learner = life.brain.learner
        learner.config = replace(learner.config, free_steps=48, tolerance=1e-15)
        return life

    monkeypatch.setattr(keydoor, "make_life", strict)
    row = keydoor.run_life("live", 0, 2, {**quick, "episodes": 20})
    assert "error" in row and "did not settle" in row["error"] and row["completed_phases"] == 0
    assert row["work"]["refused_sweeps"] == 48 and row["work"]["brains"] == 1
    assert "phases" not in row and row["work"]["latency_ms"]["aroused"] is None


def test_gates_pool_the_gated_delays_and_count_a_crash_as_a_failure(protocol):
    def life(delay, fed, wrong=0.0, late=0.0):
        phases = [{"fed": f, "wrong": wrong, "aroused_late": late} for f in fed]
        return {"arm": "live", "seed": 0, "delay": delay, "phases": phases}

    good = [life(d, (1.0, 0.95)) for d in protocol["gates"]["delays"] for _ in range(5)]
    report = keydoor.gates(good, protocol)
    assert report["passed"] and report["pooled"]["lives"] == 10
    wasteful = good[:-2] + [life(2, (1.0, 1.0), wrong=2.0)] * 2
    report = keydoor.gates(wasteful, protocol)
    assert not report["passed"] and report["pooled"]["frugal"] == pytest.approx(0.8)
    one_restless = good[:-1] + [life(2, (1.0, 1.0), late=1.0)]
    assert keydoor.gates(one_restless, protocol)["passed"]  # one life in ten is within the share
    crashed = good + [{"arm": "live", "seed": 9, "delay": 2, "error": "RuntimeError"}]
    report = keydoor.gates(crashed, protocol)
    assert not report["passed"] and report["pooled"]["crashed"] == 1
    ungated = [life(10, (0.5, 0.5))]
    assert "passed" not in keydoor.gates(ungated, protocol)  # delay 10 carries no gate


def test_a_receipt_binds_its_sources_and_refuses_changed_rows(tmp_path, capsys):
    path = tmp_path / "receipt.json.gz"
    run = ["--arms", "tabular", "random", "--seeds", "0", "--delays", "2", "--episodes", "30"]
    run += ["--workers", "1", "--out", str(path)]
    assert keydoor.main(run) == 0
    agree = "canonical form, digest, sources, arithmetic agree"
    assert keydoor.verify(path, current=True) == (True, agree)
    assert keydoor.main(["--verify", str(path)]) == 0 and keydoor.main(["--report", str(path)]) == 0
    stored = json.loads(gzip.decompress(path.read_bytes()))
    body = stored["body"]
    assert stored["kind"] == keydoor.SCHEMA and not body["frozen_protocol"]  # trips overridden
    assert keydoor.read_receipt(path) == body
    files = [item["path"] for item in stored["source"]["files"]]
    assert files[0] == "key_door.py" and "cadence/arousal.py" in files

    def rewrite(content):
        path.write_bytes(gzip.compress((keydoor.canonical_json(content) + "\n").encode()))

    edited = json.loads(json.dumps(stored))
    edited["body"]["rows"][0]["phases"][0]["fed"] = 2.0
    rewrite(edited)
    assert keydoor.verify(path) == (False, "embedded digest does not verify")
    fewer = keydoor.Receipt.build(
        keydoor.SCHEMA, {**body, "rows": body["rows"][:-1]}, keydoor.sources()
    )
    rewrite(fewer.to_dict())
    assert keydoor.verify(path)[1] == "the rows are not the planned lives, each once and in order"
    claimed = keydoor.Receipt.build(
        keydoor.SCHEMA, {**body, "gates": {"passed": True}}, keydoor.sources()
    )
    rewrite(claimed.to_dict())
    assert keydoor.verify(path)[1] == "the stored gates do not follow from the rows"
    tampered = json.loads(json.dumps(body))
    tampered["rows"][0]["phases"][1]["fed"] = 1.5
    rewrite(keydoor.Receipt.build(keydoor.SCHEMA, tampered, keydoor.sources()).to_dict())
    assert keydoor.verify(path)[1] == "a recorded share is outside [0, 1]"
    capsys.readouterr()
