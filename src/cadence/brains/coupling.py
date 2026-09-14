"""Assemble named regions into one wiring and one ordinary Settlement."""

from collections.abc import Mapping, Sequence

from ..wiring import Wiring


def couple(
    regions: Mapping[str, Wiring],
    bridges: Sequence[tuple[str, int, str, int, float]] = (),
) -> Wiring:
    """Merge regions plus directed bridges ``(source, i, target, j, weight)``.

    Region names identify complete owner sets; child sets become ``region/set``.
    Pass the returned wiring to a single Settlement, with one concatenated drive
    and state. Regions do not run independently. Convergence/optimality requires
    properties of the combined system, not merely each isolated region.
    """
    if not regions or any(not name or "/" in name for name in regions):
        raise ValueError("nonempty region names without '/' required")
    offsets: dict[str, int] = {}
    sets: dict[str, tuple[int, ...]] = {}
    pre: list[int] = []
    post: list[int] = []
    count: list[float] = []
    sign: list[float] = []
    n = 0
    for name, wiring in regions.items():
        offsets[name] = n
        sets[name] = tuple(range(n, n + wiring.n))
        sets.update(
            {f"{name}/{key}": tuple(n + i for i in ids) for key, ids in wiring.sets.items()}
        )
        pre.extend(n + int(i) for i in wiring.pre)
        post.extend(n + int(i) for i in wiring.post)
        count.extend(float(v) for v in wiring.count)
        sign.extend(float(v) for v in wiring.sign)
        n += wiring.n
    for a, i, b, j, weight in bridges:
        if a not in regions or b not in regions:
            raise ValueError("bridge names an unknown region")
        if (
            isinstance(i, bool)
            or not isinstance(i, int)
            or not 0 <= i < regions[a].n
            or isinstance(j, bool)
            or not isinstance(j, int)
            or not 0 <= j < regions[b].n
        ):
            raise ValueError("bridge owner outside its region")
        if offsets[a] + i == offsets[b] + j:
            raise ValueError("self bridges are not supported")
        pre.append(offsets[a] + i)
        post.append(offsets[b] + j)
        count.append(1.0)
        sign.append(weight)
    return Wiring.from_edges(
        n, pre=pre, post=post, count=count, sign=sign, sets=sets, label="coupled-brain"
    )
