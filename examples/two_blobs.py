"""A brain learns to tell two noisy patterns apart with the free/nudged rule.

Run:  python examples/two_blobs.py
"""

from __future__ import annotations

import numpy as np

import cadence as cd


def patterns(n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros((2 * n_per_class, 8))
    y = np.repeat([0, 1], n_per_class)
    x[:n_per_class, :4] = 1.0  # class 0 lights the left half; class 1 the right half
    x[n_per_class:, 4:] = 1.0
    return np.clip(x + 0.3 * rng.standard_normal(x.shape), 0.0, 1.0), y


x, y = patterns(60, seed=0)
test_x, test_y = patterns(30, seed=42)  # separate examples, never used for an update
rng = np.random.default_rng(1)

connectome = cd.layered(8, 16, 2, density=0.6, seed=1)  # sets: input, hidden, output
learner = cd.Learner(
    cd.Brain(connectome, cd.learning_neuron_model()),
    connectome.populations["output"],
    cd.LearnerConfig(eta=2.0),
)
drive = learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 8))))
test_drive = learner.brain.stimulus_levels(np.pad(test_x, ((0, 0), (0, connectome.n - 8))))

print(f"before learning: held-out accuracy {learner.accuracy(test_drive, test_y):.2f}")
for epoch in range(3):
    order = rng.permutation(len(y))
    for start in range(0, len(y), 20):
        idx = order[start : start + 20]
        _, report = learner.step(drive[idx], y[idx])
    print(
        f"epoch {epoch + 1}: held-out accuracy {learner.accuracy(test_drive, test_y):.2f}, "
        f"free phase {report['free_steps']:.0f} steps, nudged {report['nudged_steps']:.0f}"
    )
print(f"{learner.parameters()} parameters; the free phase never saw a label")
