"""A generic brain learns to tell bars apart and to choose the rewarded action.

The brain is one connectome of standard regions: a visual cortex or a sensory region, an
association cortex and a motor cortex, with basal ganglia that learn from dopamine. The
same class learns from labels with ``fit`` and from reward with ``act`` and ``learn``.
"""

import numpy as np

import cadence as cd


def bars(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """6 by 6 noisy pictures of a vertical (label 0) or a horizontal (label 1) bar."""
    rng = np.random.default_rng(seed)
    pictures, labels = np.zeros((n, 6, 6)), rng.integers(0, 2, size=n)
    for k, at in enumerate(rng.integers(1, 5, size=n)):
        if labels[k] == 0:
            pictures[k, :, at] = 1.0
        else:
            pictures[k, at, :] = 1.0
    return np.clip(pictures + 0.2 * rng.standard_normal(pictures.shape), 0, 1), labels


def pictures() -> float:
    brain = cd.GenericBrain.build((6, 6), 2, seed=0)
    regions = [name for name in brain.connectome.populations if "/" not in name]
    print("regions:", ", ".join(regions), f"({brain.connectome.n} neurons)")
    train_x, train_y = bars(80, seed=0)
    brain.fit(train_x, train_y, epochs=30, batch=20)
    accuracy = brain.accuracy(*bars(200, seed=1))
    print(f"held-out bars read correctly: {accuracy:.2f}")
    return accuracy


def bandit() -> float:
    rng = np.random.default_rng(0)
    brain = cd.GenericBrain.build(4, 4, seed=0)

    def contexts(k: int) -> tuple[np.ndarray, np.ndarray]:
        c = rng.integers(0, 4, size=k)
        return np.eye(4)[c], c

    x, c = contexts(32)
    for _ in range(400):
        action = brain.act(x)
        reward = (action == c).astype(float)
        x, c = contexts(32)
        brain.learn(reward, np.ones(32, dtype=bool), x)
    test, answer = contexts(400)
    brain.reset()
    rate = float((brain.act(test, greedy=True) == answer).mean())
    print(f"rewarded action chosen in new situations: {rate:.2f}")
    return rate


if __name__ == "__main__":
    assert pictures() >= 0.85
    assert bandit() >= 0.9
