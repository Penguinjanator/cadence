"""``ArousalConfig.need_scale``: a need in units of the reward scale the brain has learned.

Above zero, at every outcome the want is at least the unmet share of ``need_scale * scale``,
with the scale as it stood before that outcome entered it; the want and heat read afterwards
use the scale as it stands; youth, and a scale no outcome has entered, leave the founder need
of zero. At the founder 0 the law is unchanged: every brain and composed life pinned below
is the one the library before the gene gives (the acts and readings were recorded on main at
c7dc2f0, with the NumPy solver). The saved arousal states and checkpoints are pinned as this
version writes them: ``cadence-arousal/4`` carries every gene, and the record channel of
0.84.3, inert at its founder weight of zero, is gone with its readings; every other value
of those states and files is the one recorded on main.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cadence as cd
from cadence import brain as cadence_brain
from cadence.arousal import Arousal

NUMPY_SOLVER = not cadence_brain._FUSED
BYTES_REASON = (
    "acts and array contents were recorded on the NumPy solver; this run uses the compiled "
    "dense kernel (CADENCE_FUSED=0 selects the NumPy loop)"
)
OBS = np.array([[1.0]])  # the bandit's one constant observation
CUE = np.array([[1.0, 0.0, 0.5, 0.0]])
GO = np.array([[0.0, 0.0, 0.0, 1.0]])
FACT = np.array([[1.0, 0.0, 0.0, 1.0]])
NAME = np.array([[1.0, 0.0, 0.0, 0.0]])
STREAM = (CUE, GO, FACT, NAME)
# What the default reset forgets beside the level: the counts and the running statistics.
STATISTICS = ("outcomes", "rewards", "spreads", "square", "recent", "longrun", "usual")
Outcome = tuple[float, float, bool, bool, float | None]


class ScaledNeedReference(Arousal):
    """The scaled need as a subclass of the law, the instrument the gene reproduces: the need
    set at every reading to ``need_scale`` times the reward scale the arousal has learned,
    outside the config, with the fixed need at the founder zero."""

    need_scale: float = 0.0

    def _want(self, longrun: float, recent: float, scale: float) -> float:
        want = super()._want(longrun, recent, scale)
        k = float(self.need_scale)
        if k <= 0.0 or scale <= 0.0 or self.age < self.config.youth:
            return want
        need = k * scale
        unmet = 1.0 if recent <= 0.0 else (0.0 if recent >= need else (need - recent) / need)
        return max(want, unmet)


def with_reference(brain: cd.Brain, scale: float) -> cd.Brain:
    """The subclass installed in place on a composed brain's arousal, as the instrument did."""
    assert brain.arousal is not None and brain.arousal.config.need_scale == 0.0
    brain.arousal.__class__ = ScaledNeedReference
    brain.arousal.need_scale = scale  # type: ignore[attr-defined]
    return brain


def outcomes(count: int, seed: int) -> Iterator[Outcome]:
    """A fixed stream of outcomes: the error, the reward, whether the action was the brain's
    own, whether the outcome was learned from, and a fifth draw that keeps the stream's draws
    as they were recorded."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield (
            float(abs(rng.normal(0.0, 0.5))), float(rng.choice([0.0, 1.0, -1.0, 0.5])),
            bool(rng.random() < 0.8), bool(rng.random() < 0.5),
            None if rng.random() < 0.5 else float(abs(rng.normal(0.0, 0.5))),
        )


def unmet(need: float, recent: float) -> float:
    return 1.0 if recent <= 0.0 else (0.0 if recent >= need else (need - recent) / need)


def law_want(arousal: cd.Arousal, scale: float) -> float:
    """The law's own want at the arousal's readings, against ``scale``, at the founder need."""
    return 0.0 if scale <= 0.0 else float(
        np.clip((arousal.longrun - arousal.recent) / scale, 0.0, 1.0)
    )


def reading(brain: cd.Brain) -> dict[str, Any]:
    last = brain.last_arousal
    assert last is not None
    return dict(last)


def saved(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
        arrays = {name: data[name].copy() for name in data.files if name != "generic"}
    meta.pop("version", None)
    return meta, arrays


def assert_same_checkpoint(first: Path, second: Path, *, but: tuple[str, ...] = ()) -> None:
    (meta_a, arrays_a), (meta_b, arrays_b) = saved(first), saved(second)
    for name in but:
        meta_a.pop(name, None)
        meta_b.pop(name, None)
    assert meta_a == meta_b
    assert set(arrays_a) == set(arrays_b)
    for name in arrays_a:
        np.testing.assert_array_equal(arrays_a[name], arrays_b[name], err_msg=name)


def twin(brain: cd.Brain, path: Path) -> cd.Brain:
    return cd.Brain.load(brain.save(path))


def state_without_config(state: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in state.items() if k not in ("format", "config")}


def sparse_life(brain: cd.Brain, moments: int, start: int = 0) -> tuple[list[int], list[Any]]:
    """A world that pays rarely: 1.0 at every fourteenth moment, whatever the brain does, and
    nothing otherwise, so the income is far below the spread of the rewards."""
    acts = [int(brain.live(OBS)[0])]
    readings = [reading(brain)]
    for t in range(start, start + moments):
        acts.append(int(brain.live(OBS, reward=[1.0 if t % 14 == 0 else 0.0])[0]))
        readings.append(reading(brain))
    return acts, readings


def sparse_adult(scale: float, seed: int = 0, moments: int = 300, **options: Any) -> cd.Brain:
    brain = cd.Brain.compose(1, 2, modules=(16,), seed=seed, arousal=True,
                             arousal_need_scale=scale, **options)
    sparse_life(brain, moments)
    return brain


# -- the gene


def test_need_scale_is_a_gene_with_the_founder_zero_outside_the_declared_space() -> None:
    """The space names the ten genes of the law beside the need scale, so a seeded mutation of
    the founder draws as it does without the gene and copies the founder need scale to the
    child; a space that adds the gene can move it."""
    config = cd.ArousalConfig()
    assert config.need_scale == 0.0 and config.to_dict()["need_scale"] == 0.0
    assert cd.ArousalConfig(need_scale=0.1).to_dict()["need_scale"] == 0.1
    assert cd.ArousalConfig(**cd.ArousalConfig(need_scale=0.1).to_dict()).need_scale == 0.1
    space = cd.ArousalConfig.space()
    assert list(space) == [
        "threshold", "decay", "tolerance", "floor", "fast", "slow", "heat", "youth",
        "value_surprise", "need",
    ]
    assert set(config.to_dict()) == set(space) | {"need_scale"}
    before = {k: v for k, v in config.to_dict().items() if k != "need_scale"}
    child = cd.genes(space)(config.to_dict(), np.random.default_rng(1))
    assert child["need_scale"] == 0.0
    assert {k: v for k, v in child.items() if k != "need_scale"} == cd.genes(space)(
        before, np.random.default_rng(1)
    )
    assert cd.ArousalConfig(**child).need_scale == 0.0
    extended = {**space, "need_scale": ("linear", 0.02, 0.0, 1.0)}
    moved = [cd.genes(extended)(config.to_dict(), np.random.default_rng(seed))["need_scale"]
             for seed in range(20)]
    assert all(value >= 0.0 for value in moved) and any(value > 0.0 for value in moved)
    assert cd.ArousalConfig(**cd.genes(extended)(config.to_dict(), np.random.default_rng(3)))


@pytest.mark.parametrize("value", [-0.1, True, np.bool_(False), np.nan, np.inf, -np.inf, "0.1"])
def test_need_scale_rejects_invalid_genes(value: Any) -> None:
    with pytest.raises(ValueError, match="need_scale"):
        cd.ArousalConfig(need_scale=value)


# -- the law


@pytest.mark.parametrize("scale", [0.1, 0.5])
def test_the_gene_is_the_reference_subclass_outcome_for_outcome(scale: float) -> None:
    """The gene and the subclass give the same surprise and want at every outcome and the
    same want, heat, mode and level after it, through a reset; the founder differs."""
    gene = cd.Arousal(cd.ArousalConfig(youth=20, need_scale=scale))
    reference = ScaledNeedReference(cd.ArousalConfig(youth=20))
    reference.need_scale = scale
    founder = cd.Arousal(cd.ArousalConfig(youth=20))
    differed = 0
    for t, (error, reward, own, learned, _) in enumerate(outcomes(600, seed=int(10 * scale))):
        taken = [a.outcome(error, reward, own=own, learned=learned)
                 for a in (gene, reference, founder)]
        assert taken[0] == taken[1]
        assert (gene.want, gene.heat, gene.aroused, gene.level) == (
            reference.want, reference.heat, reference.aroused, reference.level
        )
        differed += taken[0] != taken[2]
        for arousal in (gene, reference, founder):
            arousal.lived(3, 1)
            if t == 300:
                arousal.reset()
    assert differed > 0
    state, other = gene.to_dict(), reference.to_dict()
    assert state["format"] == other["format"] == "cadence-arousal/4"
    assert state["config"]["need_scale"] == scale and other["config"]["need_scale"] == 0.0
    assert state_without_config(state) == state_without_config(other)


def test_an_outcome_reads_the_scale_before_it_enters_and_the_heat_the_scale_as_it_stands() -> None:
    """The want an outcome returns measures the recent reward against ``need_scale`` times
    the scale as it stood before that outcome entered it; the want and heat read afterwards
    use the scale as it stands; while no outcome has entered the scale the need is zero."""
    k = 0.5
    gene = cd.Arousal(cd.ArousalConfig(youth=0, need_scale=k, heat=2.0))
    assert gene.scale == 0.0 and gene.want == 0.0 and gene.heat == 1.0
    exceeded = 0
    for error, reward, own, learned, _ in outcomes(300, seed=3):
        before = gene.scale
        _, want = gene.outcome(error, reward, own=own, learned=learned)
        law = law_want(gene, before)
        expected = max(law, unmet(k * before, gene.recent)) if before > 0.0 else law
        assert want == expected
        after = gene.scale
        assert gene.want == (
            max(law_want(gene, after), unmet(k * after, gene.recent)) if after > 0.0
            else law_want(gene, after)
        )
        assert gene.heat == 1.0 + 2.0 * gene.want
        exceeded += before > 0.0 and want > law
    assert exceeded > 0


def test_youth_and_a_scale_no_outcome_has_entered_leave_the_founder_need() -> None:
    k = 0.5
    gene = cd.Arousal(cd.ArousalConfig(youth=30, need_scale=k))
    founder = cd.Arousal(cd.ArousalConfig(youth=30))
    stream = list(outcomes(30, seed=5))
    for error, reward, own, _, _ in stream:  # youth: every outcome enters the scale
        a = gene.outcome(error, reward, own=own, learned=True)
        b = founder.outcome(error, reward, own=own, learned=True)
        assert a == b and gene.want == founder.want and gene.scale == founder.scale
        gene.lived(1)
        founder.lived(1)
    assert gene.age == 30 == gene.config.youth and gene.scale > 0.0
    # an adult reset by default has an empty scale; calm outcomes that are not learned from
    # never enter it, so the scaled need stays at zero
    gene.reset()
    founder.reset()
    assert gene.scale == 0.0 and gene.spreads == 0
    for _, reward, _, _, _ in stream:
        assert gene.outcome(0.0, reward, learned=False) == founder.outcome(0.0, reward,
                                                                           learned=False)
        assert gene.want == founder.want == 0.0 and gene.scale == 0.0 and not gene.aroused
    # one learned outcome enters the scale; from then on the scaled need wants
    assert gene.outcome(0.0, 1.0, learned=True) == founder.outcome(0.0, 1.0, learned=True)
    assert gene.scale > 0.0
    larger = 0
    for _ in range(40):
        gene.outcome(0.0, 0.0, learned=True)
        founder.outcome(0.0, 0.0, learned=True)
        assert gene.want >= founder.want
        larger += gene.want > founder.want
    assert larger > 0 and gene.aroused


def test_a_need_and_a_need_scale_together_want_the_larger() -> None:
    both = cd.Arousal(cd.ArousalConfig(youth=0, need=0.4, need_scale=0.5))
    fixed = cd.Arousal(cd.ArousalConfig(youth=0, need=0.4))
    scaled = cd.Arousal(cd.ArousalConfig(youth=0, need_scale=0.5))
    for error, reward, own, _, _ in outcomes(200, seed=7):  # every outcome learned
        wants = [a.outcome(error, reward, own=own, learned=True)[1]
                 for a in (both, fixed, scaled)]
        assert wants[0] == max(wants[1], wants[2])
        assert both.want == max(fixed.want, scaled.want)


# -- the composed brain


def test_a_composed_brain_with_the_gene_lives_as_the_reference_subclass(tmp_path: Path) -> None:
    """In a world that pays rarely, the brain with the gene at .5 acts and reads at every
    moment as the brain with the subclass installed, and saves the same checkpoint but for the
    arousal's config; the founder calms and the gene keeps the brain wanting."""
    k = 0.5
    gene = cd.Brain.compose(1, 2, modules=(16,), seed=0, arousal=True, arousal_need_scale=k)
    reference = with_reference(cd.Brain.compose(1, 2, modules=(16,), seed=0, arousal=True), k)
    founder = cd.Brain.compose(1, 2, modules=(16,), seed=0, arousal=True)
    lives = {name: sparse_life(brain, 260) for name, brain in
             (("gene", gene), ("reference", reference), ("founder", founder))}
    assert lives["gene"] == lives["reference"]
    assert lives["gene"] != lives["founder"]
    assert gene.arousal is not None and reference.arousal is not None
    assert founder.arousal is not None
    assert founder.arousal.moments["routine"] > 0
    assert gene.arousal.moments["aroused"] > founder.arousal.moments["aroused"]
    assert_same_checkpoint(gene.save(tmp_path / "gene"), reference.save(tmp_path / "ref"),
                           but=("arousal",))
    state, other = gene.arousal.to_dict(), reference.arousal.to_dict()
    assert state["format"] == "cadence-arousal/4" and state["config"]["need_scale"] == k
    assert state_without_config(state) == state_without_config(other)


def test_the_scaled_need_survives_a_reset_that_keeps_the_arousal_and_a_default_reset_zeroes_it(
    tmp_path: Path,
) -> None:
    brain = sparse_adult(0.5)
    arousal = brain.arousal
    assert arousal is not None and arousal.age > arousal.config.youth
    assert arousal.scale > 0.0 and arousal.want > 0.0 and arousal.aroused
    want, scale, state = arousal.want, arousal.scale, arousal.to_dict()
    default = twin(brain, tmp_path / "default")
    restored = twin(brain, tmp_path / "restored")
    brain.reset(keep_arousal=True)
    default.reset()
    kept, left = brain.arousal, default.arousal
    assert kept is not None and left is not None
    assert kept.to_dict() == state and kept.scale == scale and kept.want == want
    assert kept.aroused and kept.config.need_scale == 0.5
    assert left.config.need_scale == 0.5 and left.age == kept.age
    assert left.scale == 0.0 and left.spreads == 0 and left.want == 0.0
    assert left.mode == "routine" and all(left.to_dict()[name] == 0 for name in STATISTICS)
    after_default = left.to_dict()
    default.live(OBS)
    default.live(OBS, reward=[0.0])
    assert reading(default)["want"] == 0.0
    # retune(reset_arousal=True) leaves the arousal as the default reset does
    retuned = twin(restored, tmp_path / "retuned")
    retuned.retune(reset_arousal=True)
    assert retuned.arousal is not None and retuned.arousal.config.need_scale == 0.5
    assert retuned.arousal.scale == 0.0 and retuned.arousal.want == 0.0
    assert retuned.arousal.age == kept.age and retuned.arousal.to_dict() == after_default
    # the kept reset continues as the documented restore of the state saved before it
    restored.reset()
    assert restored.arousal is not None
    restored.arousal = cd.Arousal.from_dict(state)
    lives = [sparse_life(b, 40, start=300) for b in (brain, restored)]
    assert lives[0] == lives[1] and lives[0][1][0]["mode"] == "aroused"
    assert_same_checkpoint(brain.save(tmp_path / "kept"), restored.save(tmp_path / "again"))


def test_a_saved_life_with_the_gene_continues_identically(tmp_path: Path) -> None:
    brain = sparse_adult(0.5, seed=1, moments=120)
    assert brain.basal_ganglia._pending is not None  # a sampled action awaits its outcome
    copy = twin(brain, tmp_path / "life")
    assert copy.arousal is not None and brain.arousal is not None
    assert copy.arousal.to_dict() == brain.arousal.to_dict()
    assert copy.arousal.config.need_scale == 0.5
    assert copy.arousal.to_dict()["format"] == "cadence-arousal/4"
    lives = []
    for b in (brain, copy):
        acts: list[int] = []
        readings = []
        for t in range(120, 160):
            acts.append(int(b.live(OBS, reward=[1.0 if t % 14 == 0 else 0.0])[0]))
            readings.append(reading(b))
        lives.append((acts, readings))
    assert lives[0] == lives[1]
    assert copy.arousal.to_dict() == brain.arousal.to_dict()
    assert_same_checkpoint(brain.save(tmp_path / "a"), copy.save(tmp_path / "b"))


def test_saves_under_earlier_formats_load_at_the_founder_and_a_scaled_save_needs_its_gene() -> None:
    """The saves of 0.84.3: ``cadence-arousal/2`` leaves out the founder need scale of zero,
    ``/3`` carries a positive one, and every one carries the record channel, removed since,
    at its founder weight of zero with its readings."""
    arousal = cd.Arousal(cd.ArousalConfig(youth=2))
    for error, reward, own, learned, _ in outcomes(12, seed=9):
        arousal.outcome(error, reward, own=own, learned=learned)
        arousal.lived(5, 2)
    current = arousal.to_dict()
    assert current["format"] == "cadence-arousal/4" and current["config"]["need_scale"] == 0.0
    founder = {k: v for k, v in current.items() if k != "config"}
    founder["config"] = {k: v for k, v in current["config"].items() if k != "need_scale"}
    founder["config"]["record_surprise"] = 0.0
    founder.update(format="cadence-arousal/2", records=4, usual_record=0.25)
    loaded = cd.Arousal.from_dict(founder)
    assert loaded.config.need_scale == 0.0 and loaded.to_dict() == current
    legacy = {k: v for k, v in founder.items() if k != "format"}
    legacy["config"] = {k: v for k, v in founder["config"].items() if k != "need"}
    restored = cd.Arousal.from_dict(legacy)
    assert restored.config.need == 0.0 and restored.config.need_scale == 0.0
    assert restored.to_dict() == current
    for earlier in (founder, {**founder, "format": "cadence-arousal/1"}, legacy):
        with pytest.raises(ValueError, match="cadence-arousal/3"):
            cd.Arousal.from_dict({**earlier, "config": {**earlier["config"], "need_scale": 0.1}})
    # a zero written under an earlier format, as ``ArousalConfig.to_dict()`` writes it, is the
    # founder and loads
    written = cd.Arousal.from_dict({**founder, "config": {**founder["config"], "need_scale": 0.0}})
    assert written.config.need_scale == 0.0 and written.to_dict() == current
    scaled = cd.Arousal.from_dict({**founder, "format": "cadence-arousal/3",
                                   "config": {**founder["config"], "need_scale": 0.1}})
    assert scaled.config.need_scale == 0.1
    assert scaled.to_dict()["format"] == "cadence-arousal/4"
    assert state_without_config(scaled.to_dict()) == state_without_config(current)
    with pytest.raises(ValueError, match="incomplete saved arousal config"):
        cd.Arousal.from_dict({**founder, "format": "cadence-arousal/3"})
    with pytest.raises(ValueError, match="incomplete saved arousal"):
        cd.Arousal.from_dict({k: v for k, v in scaled.to_dict().items() if k != "rewards"})
    # a scaled save that names the founder zero loads and saves as the founder
    zero = cd.Arousal.from_dict({**founder, "format": "cadence-arousal/3",
                                 "config": {**founder["config"], "need_scale": 0.0}})
    assert zero.to_dict() == current


def test_compose_retune_and_describe_carry_the_gene(tmp_path: Path) -> None:
    brain = cd.Brain.compose(3, 2, modules=(8,), seed=3, arousal=True, arousal_need_scale=0.2)
    assert brain.arousal is not None and brain.arousal.config.need_scale == 0.2
    described = brain.describe()
    assert described["genes"]["arousal_need_scale"] == 0.2
    assert described["arousal"]["need_scale"] == 0.2
    mapped = cd.Brain.compose(3, 2, modules=(8,), seed=3, arousal={"need_scale": 0.2})
    assert mapped.arousal is not None and mapped.arousal.config == brain.arousal.config
    assert_same_checkpoint(brain.save(tmp_path / "named"), mapped.save(tmp_path / "mapped"))
    for bad in (
        {"arousal": True, "arousal_need_scale": -0.1},
        {"arousal": True, "arousal_need_scale": np.inf},
        {"arousal": True, "arousal_need_scale": True},
        {"arousal_need_scale": 0.2},
        {"arousal": {"need_scale": 0.1}, "arousal_need_scale": 0.2},
    ):
        with pytest.raises((ValueError, TypeError)):
            cd.Brain.compose(3, 2, modules=(8,), seed=3, **bad)
    brain.live(np.eye(3)[[0]])
    for t in range(1, 12):
        brain.live(np.eye(3)[[t % 3]], reward=[float(t % 3 == 0)])
    state = brain.arousal.to_dict()
    brain.retune(arousal_need_scale=0.4)
    assert brain.arousal.config.need_scale == 0.4 and brain.pending_feedback
    assert state_without_config(brain.arousal.to_dict()) == state_without_config(state)
    assert brain.describe()["genes"]["arousal_need_scale"] == 0.4
    restored = cd.Brain.load(brain.save(tmp_path / "retuned"))
    assert restored.describe() == brain.describe()
    assert restored.arousal is not None and restored.arousal.to_dict() == brain.arousal.to_dict()
    without = cd.Brain.compose(3, 2, modules=(8,), seed=3)
    with pytest.raises(ValueError):
        without.retune(arousal_need_scale=0.2)


# -- at the founder every brain, checkpoint and composed life is the one main gives


def checkpoint_digest(path: Path) -> str:
    """SHA-256 over every array of a checkpoint, by name: dtype, shape and bytes, the
    metadata JSON entries read back without the library version that wrote them."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            value = data[name]
            h.update(name.encode())
            if name in ("meta", "generic"):
                meta = json.loads(str(value))
                meta.pop("version", None)
                h.update(json.dumps(meta, sort_keys=True).encode())
                continue
            h.update(str(value.dtype).encode() + str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def names(path: Path) -> tuple[str, str]:
    """The checkpoint's format and a digest of its metadata keys and member names."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
        listed = json.dumps([sorted(meta), sorted(data.files)])
    return str(meta["format"]), hashlib.sha256(listed.encode()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def bandit(brain: cd.Brain) -> tuple[str, list[dict[str, Any]]]:
    """150 moments of arm 0 paying, a default reset, 30 moments of arm 1 paying, a reset that
    keeps the arousal, 20 moments in which nothing pays; the acts and what each reset left."""
    assert brain.arousal is not None
    acts = [int(brain.live(OBS)[0])]
    for _ in range(150):
        acts.append(int(brain.live(OBS, reward=[float(acts[-1] == 0)])[0]))
    brain.reset()
    left = [brain.arousal.to_dict()]
    acts.append(int(brain.live(OBS)[0]))
    for _ in range(30):
        acts.append(int(brain.live(OBS, reward=[float(acts[-1] == 1)])[0]))
    brain.reset(keep_arousal=True)
    left.append(brain.arousal.to_dict())
    acts.append(int(brain.live(OBS)[0]))
    for _ in range(20):
        acts.append(int(brain.live(OBS, reward=[0.0])[0]))
    return "".join(map(str, acts)), left


def streams(brain: cd.Brain) -> tuple[str, list[dict[str, Any]]]:
    """Four streams of a composed life: live moments with a wait and episode ends, a default
    reset, a reset that keeps the memory, a reset that keeps the arousal."""
    assert brain.arousal is not None
    acts, left = [], []
    for start, length, kind in ((0, 14, None), (14, 8, "reset"), (22, 8, "memory"),
                                (30, 8, "arousal")):
        if kind == "reset":
            brain.reset()
        elif kind == "memory":
            brain.reset(keep_memory=True)
        elif kind == "arousal":
            brain.reset(keep_arousal=True)
        if start:
            left.append(brain.arousal.to_dict())
        for t in range(start, start + length):
            first = t == start
            reward = None if first else [float(t % 3 == 0)]
            done = None if first else [t % 7 == 6]
            acts.append(int(brain.live(STREAM[t % 4], reward=reward, done=done)[0]))
            if t == 5:
                brain.wait(GO)
    return "".join(map(str, acts)), left


Life = Callable[[cd.Brain], tuple[str, list[dict[str, Any]]]]
LIVES: dict[str, tuple[tuple[int, int, dict[str, Any]], Life]] = {
    "adult": ((1, 2, {"modules": (16,), "seed": 0, "arousal": True}), bandit),
    "needy": ((1, 2, {"modules": (16,), "seed": 2, "arousal": cd.ArousalConfig(need=0.05)}),
              bandit),
    "composed": ((4, 2, {"modules": (8,), "seed": 5,
                         "arousal": cd.ArousalConfig(youth=10, need=0.02), "recall": True,
                         "history_decays": (0.2, 0.9), "efference_amplitude": 1.0}), streams),
    "episodic": ((4, 2, {"modules": (8,), "seed": 7, "arousal": True, "memory_choice": True}),
                 streams),
}
# Recorded on main at c7dc2f0 (0.84.3 with ``reset(keep_arousal=True)``): each life's acts,
# what its resets left the arousal and the arousal at the end (digests of the saved states),
# and the format, the member and metadata names and the contents of every array of the
# checkpoint it saves. Where the library later removed a setting that was inert at its
# founder, the state, names and checkpoint digests were re-recorded after every other value
# matched the recording.
PINS: dict[str, dict[str, str]] = {
    "adult": {
        "acts": (
            "10000000000000000000000000110000000000000000000000000000000000000000000000000100"
            "00000000000100110100000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000"
        ),
        "format": "cadence-generic/3",
        "names": "0d66da3365c3dd3eb20a818ae910b0f42f38b2e957d8667100322412a62c03f9",
        "checkpoint": "69f2c400d2813f1436f2b15aa3deee0424b2b95b00b3494b91fd2d148f0ddbbd",
        "left": "9369e9b9ef6ea69c41ca7c9df69a3ccd1a65a94c677ea7a38287a8302da1ce9e",
        "arousal": "82930a7673513d9ce243d643a81e968e5e9d6bc52e526b78abe7356c5b31fb60",
    },
    "needy": {
        "acts": (
            "00100000000000001000000000000000000000000000000000000001000000000000000000001000"
            "00000000001000000000000000000000000000000000000000000000000000000000000000001111"
            "0111111111111111111111111111111111111111100"
        ),
        "format": "cadence-generic/3",
        "names": "29c25ae81099088aa4561925c4c990b36b4654a699d8e2f7e0371172d66627b6",
        "checkpoint": "7db0ba76256939a89d16183bb49e13824f55f48be3ea4b318121810785cc1242",
        "left": "5fe084619d8817251b331ed1f4a4ed9f13423af747216eff94e132d108af1ae6",
        "arousal": "609c15d6720de27c34cea0e220c670f9a949f5e8a3caef1fa5704501952d0ec0",
    },
    "composed": {
        "acts": "11110110010001100100010001000000000000",
        "format": "cadence-generic/8",
        "names": "0624617ac2a8160b04cfb478001ed82f9eb982d92ece84789974ae64108db965",
        "checkpoint": "993d1e2a5111ec9c23f1ca611f4c93a77eb42279ca14455d607332065ec151ed",
        "left": "8330ed860dd21843e2e489451ab3781b77bf825f67c413f1fe58b977e4cc38b5",
        "arousal": "d61a624596feedda53e9a3aee2ebde66dc0f820eb4d42431759425e3fbcb9fcc",
    },
    "episodic": {
        "acts": "11111101111111111111111111111111111110",
        "format": "cadence-generic/6",
        "names": "3d07f60219b802d1c6077f1190609cc7211b3a6ef50275beb84eb253f2bb6fca",
        "checkpoint": "68ed24048a551aa4e0d30cff22c64c9c5969c40fa6d2f1def7790ff439f1d322",
        "left": "75aef03fedbfcc5455e1febf4dcb7db64373faedc62d1288b23d73eca88a3ba0",
        "arousal": "cb50b5df1de334b9b8f16a0644bfd9a380fb5b620db0e54ed70523d1fa627d93",
    },
}


def _build(name: str) -> tuple[cd.Brain, Life]:
    (inputs, actions, options), life = LIVES[name]
    return cd.Brain.compose(inputs, actions, **options), life


@pytest.mark.parametrize("name", sorted(PINS))
def test_the_founder_saves_and_describes_as_on_main(tmp_path: Path, name: str) -> None:
    """On every backend: the checkpoint's format, members and metadata keys, and the saved
    arousal states, which carry the need scale at its founder zero."""
    brain, life = _build(name)
    _, left = life(brain)
    assert brain.arousal is not None
    path = brain.save(tmp_path / name)
    assert names(path) == (PINS[name]["format"], PINS[name]["names"])
    for state in (*left, brain.arousal.to_dict()):
        assert state["format"] == "cadence-arousal/4" and state["config"]["need_scale"] == 0.0
    assert brain.describe()["genes"]["arousal_need_scale"] == 0.0
    assert cd.Brain.load(path).describe() == brain.describe()


@pytest.mark.skipif(not NUMPY_SOLVER, reason=BYTES_REASON)
@pytest.mark.parametrize("name", sorted(PINS))
def test_the_founder_acts_arousal_states_and_checkpoint_bytes_are_those_of_main(
    tmp_path: Path, name: str
) -> None:
    brain, life = _build(name)
    acts, left = life(brain)
    assert brain.arousal is not None
    assert acts == PINS[name]["acts"]
    assert digest(left) == PINS[name]["left"]
    assert digest(brain.arousal.to_dict()) == PINS[name]["arousal"]
    assert checkpoint_digest(brain.save(tmp_path / name)) == PINS[name]["checkpoint"]


def law_readings() -> dict[str, tuple[list[list[Any]], dict[str, Any]]]:
    """The law alone, the founder and then a needy one, on one stream of outcomes with a
    reset in the middle: the surprise and want of every outcome, the want, heat and mode
    after it, and the state at the end."""
    rng = np.random.default_rng(20261021)
    out = {}
    for name, config in (("founder", cd.ArousalConfig(youth=5)),
                         ("needy", cd.ArousalConfig(youth=5, need=0.3))):
        arousal = cd.Arousal(config)
        readings: list[list[Any]] = []
        for t in range(120):
            reward = float(rng.choice([0.0, 1.0, -1.0, 0.5]))
            error = float(abs(rng.normal(0.0, 0.5)))
            own = bool(rng.random() < 0.8)
            learned = bool(rng.random() < 0.5)
            if rng.random() >= 0.5:  # the draws of the stream as it was recorded
                rng.normal(0.0, 0.5)
            surprise, want = arousal.outcome(error, reward, own=own, learned=learned)
            readings.append([surprise, want, arousal.want, arousal.heat, arousal.aroused])
            arousal.lived(int(rng.integers(1, 40)), int(rng.integers(0, 10)))
            if t == 60:
                arousal.reset()
        out[name] = (readings, arousal.to_dict())
    return out


# Recorded on main at c7dc2f0: the law's readings on the stream above; its saved state is
# the one of main in the layout of ``cadence-arousal/4``.
LAW: dict[str, tuple[str, str]] = {
    "founder": ("50a92666d40294ae132eeff72b6aa3c0e178c4eb981e6076ec324a5825b09c1b",
                "52bdfb98e436c58f3c63b6493a4107cdba336e320c94253d31b1fd88c1915ca3"),
    "needy": ("f41e6fd01cacac4128368ce8d3cbb6b0e5d785f15edd2ca501d023913ba82066",
              "90dcf4c0a162dd31d9878a76a47dbbdf4f9e35be35e51e9155f4ad46201c7705"),
}


def test_the_law_alone_reads_and_saves_as_on_main() -> None:
    for name, (readings, state) in law_readings().items():
        assert state["format"] == "cadence-arousal/4" and state["config"]["need_scale"] == 0.0
        assert (digest(readings), digest(state)) == LAW[name]
