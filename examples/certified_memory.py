"""The settling certificate, and pattern separation for correlated memory keys.

Part one builds a small learning brain, reads its certificate (row mass, slope bound,
contraction rate), settles it, and compares the a posteriori bound with the true remaining
distance to the equilibrium. Part two writes sixteen correlated keys into a delta-rule
record with and without pattern separation and reads them back.
"""

from __future__ import annotations

import numpy as np

import cadence as cd


def part_one() -> None:
    connectome = cd.layered(8, 16, 4, density=1.0, seed=0)
    model = cd.learning_neuron_model()
    raw = cd.Brain(connectome, model)
    brain = cd.Brain(connectome, model, efficacy=raw.efficacy * (1.4 / cd.row_mass(raw)))
    cert = cd.certificate(brain)
    summary = {k: round(v, 4) if isinstance(v, float) else v for k, v in cert.to_dict().items()}
    print("certificate:", summary)
    rng = np.random.default_rng(0)
    inputs = np.asarray(connectome.populations["input"])
    drive = np.zeros((1, connectome.n))
    drive[:, inputs] = rng.uniform(0.0, 1.0, len(inputs))
    star = np.asarray(brain.settle_batch(drive, steps=1000, tolerance=None).v)
    state = None
    print("step  movement   bound      true distance")
    for k in range(1, 13):
        nxt = brain.settle_batch(drive, steps=1, state=state, tolerance=None)
        v_prev = np.zeros_like(star) if state is None else np.asarray(state.v)
        movement = float(np.max(np.abs(np.asarray(nxt.v) - v_prev)))
        distance = float(np.max(np.abs(v_prev - star)))
        bound = float(cert.error_bound(movement))
        print(f"{k - 1:4d}  {movement:9.2e}  {bound:9.2e}  {distance:9.2e}")
        assert distance <= float(cert.error_bound(movement)) + 1e-12
        state = nxt
    changed = drive.copy()
    changed[:, inputs] += rng.uniform(-0.1, 0.1, len(inputs))
    delta = float(np.max(np.abs(changed - drive)))
    budget = cert.steps_for(delta, 1e-6)
    star2 = np.asarray(brain.settle_batch(changed, steps=1000, tolerance=None).v)
    settled = brain.settle_batch(drive, steps=1000, tolerance=None)
    warm = brain.settle_batch(changed, steps=budget, state=settled, tolerance=None)
    error = float(np.max(np.abs(np.asarray(warm.v) - star2)))
    print(f"stimulus changed by {delta:.3f}: budget {budget} warm steps, error {error:.2e}")
    assert error <= 1e-6


def correlated_keys(
    rng: np.random.Generator, count: int, dim: int, cosine: float, shared: np.ndarray
) -> np.ndarray:
    keys = []
    for _ in range(count):
        own = rng.standard_normal(dim)
        own -= own @ shared * shared
        own /= np.linalg.norm(own)
        keys.append(np.sqrt(cosine) * shared + np.sqrt(1.0 - cosine) * own)
    return np.asarray(keys)


def accuracy(
    memory: cd.FastSynapses, keys: np.ndarray, values: np.ndarray, rng: np.random.Generator
) -> float:
    for t in rng.permutation(len(keys)):
        memory.observe(keys[t : t + 1], values[t : t + 1])
    reads = np.concatenate([memory.recall(keys[t : t + 1]) for t in range(len(keys))])
    return float(np.mean(reads.argmax(axis=1) == values.argmax(axis=1)))


def part_two() -> None:
    rng = np.random.default_rng(1)
    shared = rng.standard_normal(32)
    shared /= np.linalg.norm(shared)
    keys = correlated_keys(rng, 64, 32, cosine=0.9, shared=shared)
    values = np.eye(8)[rng.integers(0, 8, size=64)]
    plain = cd.FastSynapses(np.arange(32), np.arange(32, 40), rule="delta")
    sep = cd.PatternSeparator(inputs=32, expansion=1024, winners=8, seed=1, center=0.99)
    sep.habituate(correlated_keys(rng, 256, 32, cosine=0.9, shared=shared))
    separated = cd.FastSynapses(np.arange(32), np.arange(32, 40), rule="delta", separator=sep)
    a = accuracy(plain, keys, values, np.random.default_rng(2))
    b = accuracy(separated, keys, values, np.random.default_rng(2))
    print(f"64 keys in 32 dimensions at cosine 0.9: plain delta {a:.3f}, separated {b:.3f}")
    assert b >= 0.95 and b > a


if __name__ == "__main__":
    part_one()
    part_two()
