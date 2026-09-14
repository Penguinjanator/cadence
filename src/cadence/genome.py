"""Where a connectome comes from: a genome, developed into a brain, selected over generations.

The connectomes in this library are designed (``layered``, ``embedded``, ``stateful``) or read
from a measured connectome, which evolution designed. An animal's specialised parts are
a mix: the genome lays down which regions exist, how large they are, which project to
which and with what sign and density; learning and development do the rest. This module is
the simplest version of that first step. A ``Genome`` is a handful of numbers per
region and per projection; ``develop`` turns it into a ``Connectome`` with named populations,
the same deterministic way every time for a given seed (development); ``mutate`` perturbs it; and
``evolve`` keeps, over generations, the genomes whose developed brains score best under a
fitness the caller supplies (a protocol score, a learning curve, a held-out accuracy).
Nothing here is a sixth primitive: the result is a connectome, and everything after it is
settling under the same neuron model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .connectome import Connectome
from .regions import Region

__all__ = ["Region", "Projection", "Genome", "develop", "mutate", "evolve"]


@dataclass(frozen=True)
class Projection:
    """Synapses from ``pre`` to a fraction ``density`` of ``post``.

    An end names a region, meaning its ``outputs`` (for ``pre``) or ``inputs`` (for ``post``)
    population and every neuron of a blank region, or one population as ``region/population``.
    """

    pre: str
    post: str
    density: float = 1.0
    sign: float = 0.0  # mean sign of the synapses: -1 all inhibitory, +1 all excitatory, 0 mixed
    scale: float = 1.0  # magnitude, fan-scaled
    count: float = 1.0
    reciprocal: bool = True  # also the reverse synapses, with the same weights: reciprocal pairs

    def __post_init__(self) -> None:
        if not self.pre or not self.post:
            raise ValueError("projection endpoints must name regions or region/population ports")
        if not 0 <= self.density <= 1 or not -1 <= self.sign <= 1:
            raise ValueError("density must lie in [0, 1] and sign in [-1, 1]")
        if not np.isfinite([self.scale, self.count]).all() or min(self.scale, self.count) < 0:
            raise ValueError("projection scale and count must be finite and nonnegative")


@dataclass(frozen=True)
class Genome:
    regions: tuple[Region, ...]
    projections: tuple[Projection, ...]
    label: str = "genome"

    def __post_init__(self) -> None:
        names = [r.name for r in self.regions]
        if not names or len(set(names)) != len(names):
            raise ValueError("a genome needs uniquely named regions")
        for projection in self.projections:
            for endpoint, side in ((projection.pre, "outputs"), (projection.post, "inputs")):
                name, _, population = endpoint.partition("/")
                if name not in names:
                    raise ValueError(f"unknown projection region: {name!r}")
                region = self.region(name)
                region.neurons(population or getattr(region, side))

    def region(self, name: str) -> Region:
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regions": [r.to_dict() for r in self.regions],
            "projections": [vars(p) for p in self.projections],
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], designed: Mapping[str, Region] | None = None) -> Genome:
        """The inverse of ``to_dict``: a genome read back from a lineage's record.

        A record keeps a designed region's circuit label and digest only; ``designed`` supplies
        the region itself by name, and its digest must match the record.
        """
        regions = []
        for r in d["regions"]:
            if "circuit" not in r:
                regions.append(Region(str(r["name"]), int(r["size"])))
                continue
            region = (designed or {}).get(str(r["name"]))
            if region is None or region.circuit is None:
                raise ValueError(f"designed region {r['name']!r} must be supplied by name")
            if (
                region.circuit.digest() != r["digest"]
                or region.name != r["name"]
                or region.size != r["size"]
                or region.inputs != r.get("inputs")
                or region.outputs != r.get("outputs")
            ):
                raise ValueError(f"designed region {r['name']!r} differs from the record")
            regions.append(region)
        return cls(
            regions=tuple(regions),
            projections=tuple(Projection(**p) for p in d["projections"]),
            label=str(d.get("label", "genome")),
        )


def develop(genome: Genome, seed: int = 0) -> Connectome:
    """Development: the genome as a connectome, region by region, deterministic in the seed.

    Regions are laid out in order. A designed region contributes its circuit's synapses and
    its populations as ``region/population``; every projection is then drawn between its two
    ends. The result is one connectome for one ``Brain``.
    """
    rng = np.random.default_rng(seed)
    starts: dict[str, int] = {}
    n = 0
    for r in genome.regions:
        starts[r.name] = n
        n += r.size
    pre: list[np.ndarray] = []
    post: list[np.ndarray] = []
    count: list[np.ndarray] = []
    sign: list[np.ndarray] = []
    populations: dict[str, Iterable[int]] = {}
    for r in genome.regions:
        s = starts[r.name]
        populations[r.name] = range(s, s + r.size)
        if r.circuit is None:
            continue
        populations.update(
            {f"{r.name}/{k}": [s + i for i in v] for k, v in r.circuit.populations.items()}
        )
        if r.circuit.synapses:
            pre.append(s + r.circuit.pre)
            post.append(s + r.circuit.post)
            count.append(r.circuit.count)
            sign.append(r.circuit.sign)

    def end(name: str, side: str) -> np.ndarray:
        region_name, _, population = name.partition("/")
        region = genome.region(region_name)
        local = region.neurons(population or getattr(region, side))
        return starts[region_name] + np.asarray(local, dtype=np.int64)

    for p in genome.projections:
        a, b = end(p.pre, "outputs"), end(p.post, "inputs")
        mask = rng.random((len(a), len(b))) < p.density
        i, j = np.nonzero(mask)
        magnitude = rng.uniform(0.0, 1.0, size=len(i)) * np.sqrt(6.0 / (len(a) + len(b))) * p.scale
        signs = np.where(rng.random(len(i)) < (1.0 + p.sign) / 2.0, 1.0, -1.0) * magnitude
        pre.append(a[i])
        post.append(b[j])
        count.append(np.full(len(i), p.count))
        sign.append(signs)
        if p.reciprocal:
            pre.append(b[j])
            post.append(a[i])
            count.append(np.full(len(i), p.count))
            sign.append(signs)
    if not pre:
        return Connectome.from_synapses(
            n, pre=[], post=[], populations=populations, label=genome.label
        )
    return Connectome.from_synapses(
        n,
        pre=np.concatenate(pre),
        post=np.concatenate(post),
        count=np.concatenate(count),
        sign=np.concatenate(sign),
        populations=populations,
        label=genome.label,
    )


def mutate(
    genome: Genome,
    rng: np.random.Generator,
    *,
    size_step: float = 0.25,
    fixed: tuple[str, ...] = (),
    tied: tuple[tuple[str, str], ...] = (),
) -> Genome:
    """One offspring: every blank region not in ``fixed`` may change size by about ``size_step`` of
    itself, every projection may change density, sign and scale a little; nothing is added or
    removed. A pair in ``tied`` keeps the second region the size of the first (a context
    range with one neuron per hidden neuron)."""
    if not np.isfinite(size_step) or size_step < 0:
        raise ValueError("size_step must be finite and nonnegative")
    for name in (*fixed, *(name for pair in tied for name in pair)):
        genome.region(name)
    regions = tuple(
        r
        if r.name in fixed or r.circuit is not None
        else replace(r, size=max(1, int(round(r.size * float(np.exp(rng.normal(0.0, size_step)))))))
        for r in genome.regions
    )
    for leader, follower in tied:
        size = next(r.size for r in regions if r.name == leader)
        regions = tuple(replace(r, size=size) if r.name == follower else r for r in regions)
    projections = tuple(
        replace(
            p,
            density=float(np.clip(p.density * np.exp(rng.normal(0.0, 0.2)), 0.01, 1.0)),
            sign=float(np.clip(p.sign + rng.normal(0.0, 0.2), -1.0, 1.0)),
            scale=float(np.clip(p.scale * np.exp(rng.normal(0.0, 0.2)), 0.05, 20.0)),
        )
        for p in genome.projections
    )
    return replace(genome, regions=regions, projections=projections)


class _Life:
    """One life as a picklable callable: grow the genome at its seed, score the connectome."""

    def __init__(self, fitness: Callable[[Connectome, int], float]) -> None:
        self.fitness = fitness

    def __call__(self, job: tuple[Genome, int]) -> float:
        genome, seed = job
        return float(self.fitness(develop(genome, seed=seed), seed))


@dataclass
class Lineage:
    """What selection did: the best genome of every generation and its fitness."""

    generations: list[dict[str, Any]] = field(default_factory=list)
    best: Genome | None = None
    best_fitness: float = -np.inf


def evolve(
    fitness: Callable[[Connectome, int], float],
    genome: Genome,
    *,
    generations: int = 10,
    population: int = 8,
    keep: int = 2,
    seed: int = 0,
    mapper: Callable[..., Iterable[float]] = map,
    report: Callable[[Lineage], None] | None = None,
    **mutation: Any,
) -> Lineage:
    """Selection over genomes: each generation grows ``population`` offspring of the
    ``keep`` best so far, scores each grown connectome with ``fitness(connectome, seed)``, and keeps
    the best. The fitness is the caller's: a protocol score, a learning curve, an accuracy.
    ``mapper`` runs a generation's lives: ``map`` one after another, a pool's ``map`` side
    by side (``fitness`` must then be picklable, so a module-level function). ``report``
    is called with the lineage so far after every generation, so a long run can be
    written out as it goes."""
    for name, value in (("generations", generations), ("population", population), ("keep", keep)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if keep > population:
        raise ValueError("keep cannot exceed population")
    rng = np.random.default_rng(seed)
    lineage = Lineage()
    parents = [genome]
    for g in range(generations):
        offspring = list(parents) if g == 0 else []
        while len(offspring) < population:
            offspring.append(mutate(parents[rng.integers(len(parents))], rng, **mutation))
        seeds = [seed + 1000 * g + k for k in range(len(offspring))]
        scores = mapper(_Life(fitness), zip(offspring, seeds, strict=True))
        scored = [
            (float(f), k, child) for k, (f, child) in enumerate(zip(scores, offspring, strict=True))
        ]
        if not all(np.isfinite(f) for f, _, _ in scored):
            raise ValueError("fitness must return finite scores")
        scored.sort(key=lambda s: -s[0])
        parents = [c for _, _, c in scored[:keep]]
        top = scored[0]
        lineage.generations.append(
            {
                "generation": g,
                "best_fitness": top[0],
                "best": top[2].to_dict(),
                "mean_fitness": float(np.mean([s[0] for s in scored])),
            }
        )
        if top[0] > lineage.best_fitness:
            lineage.best, lineage.best_fitness = top[2], top[0]
        if report is not None:
            report(lineage)
    return lineage
