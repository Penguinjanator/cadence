"""The standard brain view: one page with the whole brain settling live.

Builds a small certified brain with a retinal sheet, records two settlings (from rest, then
warm from the first equilibrium after the stimulus changes), lays the whole connectome out
as an atlas, and writes one self-contained page that replays the recorded steps and can
detune and settle the same brain live in the browser. Open the printed path in a browser.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import cadence as cd


def main() -> Path:
    connectome = cd.layered(24, 40, 6, density=0.6, seed=0).with_populations(
        retina=range(24), association=range(24, 64), motor=range(64, 70)
    )
    model = cd.learning_neuron_model()
    raw = cd.Brain(connectome, model)
    brain = cd.Brain(connectome, model, efficacy=raw.efficacy * (1.5 / cd.row_mass(raw)))
    cert = cd.certificate(brain)
    rng = np.random.default_rng(0)
    drive = np.zeros((1, connectome.n))
    drive[0, :24] = rng.uniform(0.0, 1.0, 24)
    changed = drive.copy()
    changed[0, :24] += rng.uniform(-0.3, 0.3, 24)
    records: list[cd.SettlementRecord] = []
    with cd.record_settlements(records.append, label="demo"):
        first = brain.settle_batch(drive, steps=40, tolerance=None)
        brain.settle_batch(changed, steps=40, state=first, tolerance=None)
    atlas = cd.atlas_of(brain, shapes={"retina": (4, 6)})
    steps = np.concatenate([r.activation[:, 0, :] for r in records])
    potentials = np.concatenate([r.potential[:, 0, :] for r in records])
    frames = atlas.frames(steps, potentials)
    page = atlas.page(
        frames=frames,
        brain=brain,
        title="Cadence brain scan · a certified brain settles",
        note=(
            f"Row mass {cert.row_mass:.2f}, contraction rate {cert.rate:.3f}: certified. "
            "Recorded frames: a cold settling from rest, then a warm settling after the "
            "stimulus changed. Detune draws a new stimulus on the retina and settles live; "
            "the status line shows the certificate's error bound from the last movement."
        ),
    )
    out = Path(tempfile.gettempdir()) / "cadence_brain_scan_example.html"
    out.write_text(page, encoding="utf-8")
    print(f"atlas: {atlas.summary()}")
    size = out.stat().st_size // 1024
    print(f"recorded steps: {frames['steps']}; page written to {out} ({size} kB)")
    return out


if __name__ == "__main__":
    main()
