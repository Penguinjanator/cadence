"""Two neurons that inhibit each other: a fixed point without adaptation, a rhythm with it.

Run:  python examples/half_center.py
"""

from __future__ import annotations

import cadence as cd

connectome = cd.Connectome.from_synapses(2, pre=[0, 1], post=[1, 0], count=[60, 60], sign=[-1, -1])
stimulus = {0: 1.0, 1: 0.95}  # both driven, one a little harder

still = cd.Brain(connectome, cd.NeuronModel(gain=0.03)).settle(stimulus, steps=400, trajectory=True)
rhythm = cd.Brain(
    connectome, cd.NeuronModel(gain=0.03, adaptation=cd.Adaptation(tau_steps=40, strength=2.0))
).settle(stimulus, steps=400, trajectory=True)

assert still.trajectory is not None and rhythm.trajectory is not None
print(
    "without adaptation, last 100 steps, std per neuron:",
    still.trajectory[-100:].std(axis=0).round(4),
)
print(
    "with adaptation,    last 200 steps, std per neuron:",
    rhythm.trajectory[-200:].std(axis=0).round(3),
)
for t in range(300, 400, 10):
    bar = "".join("#" if x > 0.5 else "." for x in rhythm.trajectory[t])
    print(f"step {t}: {bar}  {rhythm.trajectory[t].round(2)}")
