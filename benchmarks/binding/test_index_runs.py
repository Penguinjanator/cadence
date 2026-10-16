"""Contracts of the declared lives' camera and retina."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runs = _load("index_runs")
tb = runs.ir.tb


def test_the_camera_gives_each_token_its_own_block() -> None:
    camera = runs.Camera(3)
    assert camera.inputs == 12_824
    assert np.all(camera.patches.sum(axis=1) == runs.CAMERA["block"] ** 2)
    assert np.all(camera.patches.sum(axis=0) <= 1.0)  # blocks never overlap
    assert not camera.patches[:, camera.pixels:].any()  # the body inputs carry no token


def test_the_retina_codes_blocks_and_drops_lone_pixels() -> None:
    camera, retina = runs.Camera(3), runs.Retina()
    assert retina.units == 744 + runs.CAMERA["body"]
    event = np.zeros((2, tb.INPUTS))
    event[0, [0, 5]] = 1.0  # two tokens
    event[1, [0, 5, 9]] = 1.0  # the same two and a third
    frames = camera([event])[0]
    code = retina(frames)
    active = (code[:, : len(retina.index)] > 0).sum(axis=1)
    assert 40 <= active[0] < active[1] <= 128
    np.testing.assert_allclose(np.linalg.norm(code[:, : len(retina.index)], axis=1), 1.0)
    assert np.sum(code[0] * code[1]) > 0.7  # two of three blocks shared
    noise = (np.random.default_rng(3).random((1, camera.inputs)) < 0.01).astype(float)
    assert not retina(noise)[:, : len(retina.index)].any()
    np.testing.assert_array_equal(retina(noise)[:, len(retina.index):], noise[:, camera.pixels:])


def test_a_query_shares_its_facts_code_and_little_else() -> None:
    camera, retina = runs.Camera(4), runs.Retina()
    trials = tb.draw(np.random.default_rng(4), 32, tb.holdout(4))
    codes = [retina(f) for f in camera(tb.events(trials))]
    q = trials["query"]
    asked = np.where(q[:, None] == 0, codes[0], codes[1])
    other = np.where(q[:, None] == 0, codes[1], codes[0])
    assert np.sum(codes[3] * asked, axis=1).mean() > 0.35
    assert np.sum(codes[3] * other, axis=1).mean() < 0.1
