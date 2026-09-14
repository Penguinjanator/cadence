"""A sensory region and opposing motors repair one shared state."""

import numpy as np

import cadence as cd
from cadence.brains import couple, sensor_motor


def run() -> float:
    retina = cd.Wiring.from_edges(1, pre=[], post=[])
    brain = couple(
        {"vision": retina, "movement": sensor_motor(1)},
        [
            ("vision", 0, "movement", 0, 0.5),
            ("movement", 1, "vision", 0, -0.1),
            ("movement", 2, "vision", 0, 0.1),
        ],
    )
    engine = cd.Settlement(
        brain,
        cd.GradedRule(gain=1, slope=2, threshold=0, leak=1, dt=0.25, clamp_amplitude=1),
    )
    drive = np.array([0.6, 0.0, 0.0, 0.0])
    state = engine.settle(drive, steps=400, tolerance=0)
    error = float(engine.residual(drive, state)[0])
    assert error < 1e-10
    print("Regions:", ", ".join(name for name in brain.sets if "/" not in name))
    print(f"Joint equation error: {error:.2e}")
    rates = np.maximum(0, state.activation[list(brain.sets["movement/motor"])])
    return float(rates[0] - rates[1])


if __name__ == "__main__":
    print("Motor command:", run())
