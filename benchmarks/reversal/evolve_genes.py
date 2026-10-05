"""Selection over the arousal genes on the odour nursery; the hand-set founders are the control.

Every constant of the arousal law is a gene (``ArousalConfig.space()``). This script lets
selection move them: ``cadence.evolve`` starts from the founders, scores each genome by
the share of optimal executed actions over whole lives of the ``live`` arm (acquisition,
reversal, return and the calm between them, so slow repair and needless exploration both
cost), and keeps the best. The winner and the founders are then scored on held-out seeds
that selection never saw. A gain on the selection seeds alone is not a result.

Run ``python benchmarks/reversal/evolve_genes.py --help``. The operating point, the world
and the exposures are the nursery protocol's; only the genes vary.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

sys.path.insert(0, str(Path(__file__).parent))
import odour_nursery as nursery  # noqa: E402

SELECTION_SEEDS = (0, 1, 2, 3)  # among the protocol's development seeds
HELD_OUT_SEEDS = tuple(range(200, 210))  # never used by selection or by the confirmation


def life_score(
    genes: dict[str, Any], seed: int, exposure: int, protocol: dict[str, Any]
) -> dict[str, float]:
    """One ``live`` life without probes: the optimal share of its executed actions over the
    whole life and over each later rule, and its aroused share."""
    warnings.simplefilter("ignore")
    life = nursery.make_life("live", protocol, seed, genes)
    odours = np.random.default_rng(protocol["odour_seed"] + seed)
    odour = int(odours.integers(nursery.ODOURS))
    action, aroused = life.act(odour, None)
    hits: list[list[int]] = []
    modes = []
    for sugar, length in ((0, exposure), (1, protocol["after"]), (0, protocol["after"])):
        phase = []
        for _ in range(length):
            phase.append(int(action == nursery.optimal(odour, sugar)))
            modes.append(aroused)
            reward = nursery.reward_of(odour, action, sugar)
            odour = int(odours.integers(nursery.ODOURS))
            action, aroused = life.act(odour, reward)
        hits.append(phase)
    return {
        "life": float(np.mean([h for phase in hits for h in phase])),
        "reversal": float(np.mean(hits[1])),
        "return": float(np.mean(hits[2])),
        "aroused": float(np.mean(modes)),
    }


def score(
    genes: dict[str, Any],
    seeds: tuple[int, ...],
    exposures: tuple[int, ...],
    protocol: dict[str, Any],
) -> dict[str, float]:
    try:
        cd.ArousalConfig(**genes)
    except ValueError:
        return {"life": 0.0, "reversal": 0.0, "return": 0.0, "aroused": 1.0}  # not a genome
    rows = [life_score(genes, seed, e, protocol) for seed in seeds for e in exposures]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def fitness(
    genes: dict[str, Any], seed: int, *, exposures: tuple[int, ...], protocol: dict[str, Any]
) -> float:
    """The mean optimal share over whole lives on the selection seeds. ``seed`` is the
    lineage's and is not used: every genome meets the same lives, so scores compare."""
    return score(genes, SELECTION_SEEDS, exposures, protocol)["life"]


def _held_out(job: tuple[str, dict[str, Any], tuple[int, ...], dict[str, Any]]) -> dict[str, Any]:
    name, genes, exposures, protocol = job
    return {"genome": name, "genes": genes, **score(genes, HELD_OUT_SEEDS, exposures, protocol)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--protocol", type=Path, default=nursery.PROTOCOL)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=9)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--exposures", nargs="*", type=int, default=[300, 2000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    exposures = tuple(args.exposures)
    founders = dict(protocol["arousal"])
    scorer = functools.partial(fitness, exposures=exposures, protocol=protocol)

    def progress(lineage: cd.genome.Lineage) -> None:
        row = lineage.generations[-1]
        print(
            f"generation {row['generation']}: best {row['best_fitness']:.4f} "
            f"mean {row['mean_fitness']:.4f}",
            flush=True,
        )

    with get_context("spawn").Pool(args.workers) as pool:
        lineage = cd.evolve(
            scorer,
            founders,
            mutate=cd.genes(cd.ArousalConfig.space(), rate=0.5),
            generations=args.generations,
            population=args.population,
            keep=args.keep,
            seed=args.seed,
            mapper=pool.map,
            report=progress,
        )
        jobs = [
            ("founders", founders, exposures, protocol),
            ("selected", dict(lineage.best), exposures, protocol),
        ]
        held_out = pool.map(_held_out, jobs)
        founders_fitness = pool.apply(scorer, (founders, 0))
    report = {
        "schema": "odour-nursery-genes/1",
        "cadence": cd.__version__,
        "exposures": list(exposures),
        "selection_seeds": list(SELECTION_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "founders": founders,
        "selected": lineage.best,
        "selection_fitness": {"founders": founders_fitness, "selected": lineage.best_fitness},
        "generations": lineage.generations,
        "held_out": held_out,
    }
    print(json.dumps({"selected": lineage.best, "held_out": held_out}, indent=1))
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
