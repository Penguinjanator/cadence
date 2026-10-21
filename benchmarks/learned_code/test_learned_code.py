"""Contracts of the learned-code instrument: the code off is the released brain, the settle is
the energy's minimum, readings change nothing, and the GRU's gradient is its loss's."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


em = _load("embedding")
cr = em.cr


def _rounds(seed: int, count: int, *, asked: bool = True):
    held, answers = em.holdout(seed), em.answers_of(seed)
    rng = np.random.default_rng(np.random.SeedSequence([em.SCHEDULE_SEED, seed]))
    return [em.draw_round(rng, held, answers, asked=asked) for _ in range(count)]


def _saved(brain, path: Path) -> dict[str, np.ndarray]:
    brain.save(path)
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


@pytest.mark.parametrize("arm, code", [("raw", None), ("random", em.UNITS)])
def test_without_a_code_region_the_life_is_the_released_brain(tmp_path, arm, code):
    """With the code off a life drives ``Brain.compose`` as released: the same acts, the same
    lessons, and every saved array equal after a few rounds."""
    from cadence import Brain

    seed = 3
    life = em.Life(arm, seed)
    brain = Brain.compose(em.INPUTS, em.ACTIONS, modules=(em.WIDTH,), seed=seed, code=code,
                          learning_surprise=True)
    for seq, target in _rounds(seed, 4):
        life.reset()
        brain.reset()
        for t, x in enumerate(seq):
            if t == len(seq) - 1:
                life_out = life.teach(x, target)
                brain.lesson(x, target)
            else:
                life_out = life.hear(x)
            assert np.array_equal(life_out, brain.act(x, greedy=True))
    mine, released = _saved(life.brain, tmp_path / "a.npz"), _saved(brain, tmp_path / "b.npz")
    assert mine.keys() == released.keys()
    for name in mine:
        assert np.array_equal(mine[name], released[name]), name


def test_the_settle_minimizes_the_energy():
    """The settled code meets the energy's optimality conditions: an active unit's threshold
    equals its residual drive, a silent unit's is at least it."""
    rng = np.random.default_rng(0)
    settings = cr.CodeSettings(units=24, top_down=1.5, tolerance=1e-9, budget=20000)
    region = cr.CodeRegion(16, 8, settings, seed=1)
    region.thresholds = rng.uniform(0.02, 0.2, 24)
    x = rng.standard_normal((5, 16))
    p = np.abs(rng.standard_normal((5, 24))) * 0.3
    a, _ = region.settle(x, p)
    drive = (x - a @ region.phi.T) @ region.phi - settings.top_down * (a - p)
    active = a > 0
    assert active.any()
    assert np.allclose(drive[active], np.broadcast_to(region.thresholds, a.shape)[active],
                       atol=1e-6)
    assert np.all(drive[~active] <= np.broadcast_to(region.thresholds, a.shape)[~active]
                  + 1e-6)


def test_without_the_top_down_boundary_the_prediction_is_not_read():
    region = cr.CodeRegion(16, 8, cr.CodeSettings(units=24, top_down=0.0), seed=2)
    x = np.random.default_rng(1).standard_normal((3, 16))
    a0, _ = region.settle(x, np.zeros((3, 24)))
    a1, _ = region.settle(x, np.ones((3, 24)))
    assert np.array_equal(a0, a1)


def test_learning_moves_the_code_toward_its_input():
    """A fixed set of inputs: the bottom-up residual falls as the dictionary learns."""
    rng = np.random.default_rng(4)
    region = cr.CodeRegion(20, 4, cr.CodeSettings(units=32, top_down=0.0,
                                                  dictionary_rate=0.5), seed=4)
    x = np.eye(20)[rng.permutation(20)[:8]]
    zero = np.zeros((8, 32))
    first = region.residuals(x, region.settle(x, zero)[0], zero)["bottom_up"].mean()
    for _ in range(300):
        a, _ = region.settle(x, zero)
        region.learn(x, a, zero, None)
    last = region.residuals(x, region.settle(x, zero)[0], zero)["bottom_up"].mean()
    assert last < 0.5 * first


def test_readings_change_nothing(tmp_path):
    """Reading a code arm's life (panels, code report) leaves its code and brain as they
    were."""
    seed = 5
    life = em.Life("learned", seed)
    held, answers = em.holdout(seed), em.answers_of(seed)
    for seq, target in _rounds(seed, 3):
        em.live_round(life, seq, target)
    phi, w, thr = life.code.phi.copy(), life.code.w.copy(), life.code.thresholds.copy()
    before = _saved(life.brain, tmp_path / "a.npz")
    em.read_life(life, "learned", seed, held, answers)
    em.code_report(life, held, seed)
    assert np.array_equal(phi, life.code.phi) and np.array_equal(w, life.code.w)
    assert np.array_equal(thr, life.code.thresholds)
    after = _saved(life.brain, tmp_path / "b.npz")
    for name in before:
        assert np.array_equal(before[name], after[name]), name


def test_shuffled_contexts_keep_the_members_and_lose_the_groups():
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    shuffle = np.random.default_rng(8)
    plain = em.sentence(rng_a, 4000)
    mixed = em.sentence(rng_b, 4000, shuffle=shuffle)
    assert np.array_equal(plain[1], mixed[1])
    group = plain[1] // em.MEMBERS
    assert np.array_equal((plain[0] - em.PRE0) // em.PRE, group)
    agree = np.mean((mixed[0] - em.PRE0) // em.PRE == group)
    assert abs(agree - 1 / em.GROUPS) < 0.05


@pytest.mark.parametrize("predict", [False, True])
def test_the_gru_gradient_is_its_loss_gradient(predict):
    """The hand-written gradient through the round's events, with and without the next-token
    head, against central differences of the loss."""
    model = em.Recurrent(0, predict=predict, hidden=6)
    rng = np.random.default_rng(3)
    for value in model.parameters().values():
        value += 0.1 * rng.standard_normal(value.shape)
    seq = [np.eye(em.INPUTS)[rng.integers(em.INPUTS, size=3)] for _ in range(4)]
    target = rng.integers(em.ACTIONS, size=3)
    grads = model.gradients(seq, target)
    eps = 1e-6
    for name, value in model.parameters().items():
        for index in [tuple(rng.integers(s) for s in value.shape) for _ in range(4)]:
            keep = value[index]
            value[index] = keep + eps
            up = model.loss(seq, target)
            value[index] = keep - eps
            down = model.loss(seq, target)
            value[index] = keep
            assert np.isclose(grads[name][index], (up - down) / (2 * eps), atol=1e-6), name
