"""Finite-difference control for one-way projections from autonomous fixed inputs.

Five seeds, three beta values, converged phases, symmetric effective free/free weights,
no adaptation, positive potentials separated from the activation kink. This checks the
input projection parameter as well as the tied recurrent pair. It is not an audit of
historical trained checkpoints or a guarantee for arbitrary finite-beta learning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import cadence as cd

if __package__:
    from .freeze import source_bundle
else:
    from freeze import source_bundle


def measure(seed: int, beta: float, *, asymmetric: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    recurrent = float(rng.uniform(0.08, 0.18))
    graph = cd.Connectome.from_synapses(
        3,
        pre=[0, 0, 1, 2],
        post=[1, 2, 2, 1],
        count=[2, 3, 1, 1],
        sign=[0.1, 0.12, recurrent, recurrent * (3 if asymmetric else 1)],
    )
    brain = cd.Brain(graph, cd.learning_neuron_model(gain=1.3), bias=rng.uniform(0.2, 0.4, 3))
    learner = cd.Learner(
        brain, [1, 2], cd.LearnerConfig(beta=beta, nudge="cross_entropy", temperature=0.3)
    )
    drive = rng.uniform(0.2, 0.4, (4, 3))
    target = learner.targets(np.array([0, 1, 1, 0]))
    free = brain.equilibrate(drive, tolerance=1e-13, chunk=4)
    plus_nudge, minus_nudge = learner.nudge_for(target, beta), learner.nudge_for(target, -beta)
    plus = brain.equilibrate(drive, state=free.state, nudge=plus_nudge, tolerance=1e-13, chunk=4)
    minus = brain.equilibrate(drive, state=free.state, nudge=minus_nudge, tolerance=1e-13, chunk=4)
    contrast, _ = learner.contrast(free.state, plus.state, minus.state)

    def loss(efficacy: np.ndarray) -> float:
        candidate = brain.with_parameters(efficacy=efficacy)
        settled = candidate.equilibrate(drive, tolerance=1e-13, chunk=4)
        logits = settled.state.activation[:, [1, 2]] / learner.config.temperature
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return float(-(logp * target[:, [1, 2]]).sum(axis=1).mean())

    rows = []
    for source, destination in ((0, 1), (0, 2), (1, 2)):
        selected = (graph.pre == source) & (graph.post == destination)
        if source != 0:
            selected |= (graph.pre == destination) & (graph.post == source)
        edge = np.flatnonzero((graph.pre == source) & (graph.post == destination))[0]
        up, down = brain.efficacy.copy(), brain.efficacy.copy()
        up[selected] += 1e-5
        down[selected] -= 1e-5
        exact = -(loss(up) - loss(down)) / 2e-5
        # One contrast for a tied pair, or for one directed external-input parameter.
        estimate = float(contrast[edge] * brain._gain_pre[edge] / learner.config.temperature)
        rows.append(
            {
                "projection": [source, destination],
                "estimate": estimate,
                "negative_finite_difference": exact,
                "absolute_error": abs(estimate - exact),
            }
        )
    return {
        "seed": seed,
        "beta": beta,
        "asymmetric_control": asymmetric,
        "structure": cd.ep_structure(brain, fixed_inputs=[0]).to_dict(),
        "whole_matrix_structure": cd.ep_structure(brain).to_dict(),
        "contraction": cd.certificate(brain).to_dict(),
        "phase_residual_max": max(float(x.residual.max()) for x in (free, plus, minus)),
        "minimum_potential": min(float(x.state.v.min()) for x in (free, plus, minus)),
        "input_phase_difference": float(
            np.abs(plus.state.activation[:, 0] - minus.state.activation[:, 0]).max()
        ),
        "gradient_rows": rows,
    }


def verify(body: dict) -> str | None:
    if len(body["cells"]) != 20:
        return "missing cells"
    for cell in body["cells"]:
        if (
            cell["phase_residual_max"] > 1e-13
            or cell["minimum_potential"] <= 0.1
            or cell["input_phase_difference"] > 1e-12
        ):
            return "phase preconditions differ"
        if cell["structure"]["compatible"] == cell["asymmetric_control"]:
            return "structural control not distinguished"
        for row in cell["gradient_rows"]:
            if not np.isclose(
                row["absolute_error"],
                abs(row["estimate"] - row["negative_finite_difference"]),
                rtol=1e-12,
            ):
                return "gradient error arithmetic differs"
            if (
                not cell["asymmetric_control"]
                and cell["beta"] == 1e-3
                and row["absolute_error"] > 2e-6
            ):
                return "limiting gradient did not match"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = source_bundle(root)
    cells = [measure(seed, beta) for seed in range(5) for beta in (0.1, 0.01, 0.001)]
    cells += [measure(seed, 0.001, asymmetric=True) for seed in range(5)]
    body = {
        "seeds": list(range(5)),
        "betas": [0.1, 0.01, 0.001],
        "finite_difference_step": 1e-5,
        "residual_tolerance": 1e-13,
        "loss": "mean ordinary cross entropy; contrast converted by contact/gain divided by T",
        "primary_source": "https://arxiv.org/abs/1602.05179",
        "cells": cells,
    }
    cd.Receipt.build("ep-fixed-input-finite-difference-v1", body, sources).write(args.out)
    print(cd.Receipt.verify(args.out, sources=sources, check=verify))
    for beta in body["betas"]:
        print(
            "beta",
            beta,
            "maximum error",
            max(
                row["absolute_error"]
                for cell in cells
                if cell["beta"] == beta and not cell["asymmetric_control"]
                for row in cell["gradient_rows"]
            ),
        )
    print(
        "asymmetric control maximum errors",
        [
            max(r["absolute_error"] for r in c["gradient_rows"])
            for c in cells
            if c["asymmetric_control"]
        ],
    )
    if verify(body):
        raise SystemExit(verify(body))


if __name__ == "__main__":
    main()
