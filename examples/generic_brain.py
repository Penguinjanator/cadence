"""A generic brain learns to tell bars apart and to choose the rewarded action.

The brain is one connectome of standard regions: a visual cortex or a sensory region, an
association cortex and a motor cortex, with basal ganglia that learn from dopamine. The
same class can take demonstrations and reward in its ongoing ``step`` loop.
``fit`` below is a compact independent-picture benchmark, not a required life stage.
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
    action = brain.step(x)
    for _ in range(400):
        reward = (action == c).astype(float)
        x, c = contexts(32)
        action = brain.step(x, reward=reward, done=np.ones(32, dtype=bool))
    test, answer = contexts(400)
    brain.reset()
    rate = float((brain.act(test, greedy=True) == answer).mean())
    print(f"rewarded action chosen in new situations: {rate:.2f}")
    return rate


def lasting_memory() -> list[float]:
    """Clear transient state and compare ordinary, repeated and salient experiences."""
    result = []
    for repetitions, salience in ((1, 0.0), (40, 0.0), (1, 19.0)):
        memory = cd.SynapticMemory(np.arange(2), np.arange(2, 4))
        for _ in range(repetitions):
            memory.observe(
                np.array([[1., 0.]]), np.array([[1., 0.]]), salience=np.array([salience])
            )
        memory.reset(1)
        result.append(float(memory.recall(np.array([[1., 0.]]))[0, 0]))
    print("lasting association after clearing transient memory:", np.round(result, 4))
    return result


if __name__ == "__main__":
    assert pictures() >= 0.85
    assert bandit() >= 0.9
    retained = lasting_memory()
    assert retained[0] < 0.1 and retained[1] > 0.85 and retained[2] == 1
