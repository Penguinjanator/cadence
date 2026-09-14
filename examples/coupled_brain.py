"""A sensory region and opposing motor neurons settle into one shared state."""

import numpy as np

import cadence as cd
from cadence.circuits import assemble, reflex_arc


def run() -> float:
    retina = cd.Connectome.from_synapses(1, pre=[], post=[])
    connectome = assemble(
        {"vision": retina, "movement": reflex_arc(1)},
        [
            ("vision", 0, "movement", 0, 0.5),
            ("movement", 1, "vision", 0, -0.1),
            ("movement", 2, "vision", 0, 0.1),
        ],
    )
    brain = cd.Brain(
        connectome,
        cd.NeuronModel(gain=1, slope=2, threshold=0, leak=1, dt=0.25, stimulus_amplitude=1),
    )
    drive = np.array([0.6, 0.0, 0.0, 0.0])
    result = brain.equilibrate(drive, budget=400, chunk=8, tolerance=1e-10)
    assert result.converged.all(), result.residual
    state = result.state
    error = float(result.residual[0])
    print("Regions:", ", ".join(name for name in connectome.populations if "/" not in name))
    print(f"Joint equation error: {error:.2e}")
    rates = np.maximum(0, state.activation[0, list(connectome.populations["movement/motor"])])
    return float(rates[0] - rates[1])


if __name__ == "__main__":
    print("Motor command:", run())
