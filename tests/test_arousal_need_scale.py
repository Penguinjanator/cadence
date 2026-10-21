"""``ArousalConfig.need_scale``: a need in units of the reward scale the brain has learned.

Above zero, at every outcome the want is at least the unmet share of ``need_scale * scale``,
with the scale as it stood before that outcome entered it; the want and heat read afterwards
use the scale as it stands; youth, and a scale no outcome has entered, leave the founder need
of zero. At the founder 0 the law is unchanged: every brain, checkpoint and composed life
pinned below is the one the library before the gene gives (the pins were recorded on main at
c7dc2f0, the acts and array contents with the NumPy solver), and a save carries the gene under
``cadence-arousal/3`` only when it is above zero.
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
STATISTICS = (
    "outcomes", "rewards", "spreads", "records",
    "square", "recent", "longrun", "usual", "usual_record",
)
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
    own, whether the outcome was learned from, and the record's error when it held one."""
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
    """The space names the eleven genes it named before, so a seeded mutation of the founder
    draws as it did and copies the founder need scale to the child; a space that adds the
    gene can move it."""
    config = cd.ArousalConfig()
    assert config.need_scale == 0.0 and config.to_dict()["need_scale"] == 0.0
    assert cd.ArousalConfig(need_scale=0.1).to_dict()["need_scale"] == 0.1
    assert cd.ArousalConfig(**cd.ArousalConfig(need_scale=0.1).to_dict()).need_scale == 0.1
    space = cd.ArousalConfig.space()
    assert list(space) == [
        "threshold", "decay", "tolerance", "floor", "fast", "slow", "heat", "youth",
        "value_surprise", "record_surprise", "need",
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
    for t, (error, reward, own, learned, record) in enumerate(outcomes(600, seed=int(10 * scale))):
        taken = [a.outcome(error, reward, own=own, learned=learned, record_error=record)
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
    assert state["format"] == "cadence-arousal/3" and state["config"]["need_scale"] == scale
    assert other["format"] == "cadence-arousal/2" and "need_scale" not in other["config"]
    assert state_without_config(state) == state_without_config(other)


def test_an_outcome_reads_the_scale_before_it_enters_and_the_heat_the_scale_as_it_stands() -> None:
    """The want an outcome returns measures the recent reward against ``need_scale`` times
    the scale as it stood before that outcome entered it; the want and heat read afterwards
    use the scale as it stands; while no outcome has entered the scale the need is zero."""
    k = 0.5
    gene = cd.Arousal(cd.ArousalConfig(youth=0, need_scale=k, heat=2.0))
    assert gene.scale == 0.0 and gene.want == 0.0 and gene.heat == 1.0
    exceeded = 0
    for error, reward, own, learned, record in outcomes(300, seed=3):
        before = gene.scale
        _, want = gene.outcome(error, reward, own=own, learned=learned, record_error=record)
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
    for error, reward, own, _, record in stream:  # youth: every outcome enters the scale
        a = gene.outcome(error, reward, own=own, learned=True, record_error=record)
        b = founder.outcome(error, reward, own=own, learned=True, record_error=record)
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
    for error, reward, own, _, record in outcomes(200, seed=7):  # every outcome learned
        wants = [a.outcome(error, reward, own=own, learned=True, record_error=record)[1]
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
    assert state["format"] == "cadence-arousal/3" and state["config"]["need_scale"] == k
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
    assert copy.arousal.to_dict()["format"] == "cadence-arousal/3"
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
    arousal = cd.Arousal(cd.ArousalConfig(youth=2))
    for error, reward, own, learned, record in outcomes(12, seed=9):
        arousal.outcome(error, reward, own=own, learned=learned, record_error=record)
        arousal.lived(5, 2)
    founder = arousal.to_dict()
    assert founder["format"] == "cadence-arousal/2" and "need_scale" not in founder["config"]
    loaded = cd.Arousal.from_dict(founder)
    assert loaded.config.need_scale == 0.0 and loaded.to_dict() == founder
    legacy = {k: v for k, v in founder.items() if k != "format"}
    legacy["config"] = {k: v for k, v in founder["config"].items() if k != "need"}
    restored = cd.Arousal.from_dict(legacy)
    assert restored.config.need == 0.0 and restored.config.need_scale == 0.0
    assert restored.to_dict() == founder
    for earlier in (founder, {**founder, "format": "cadence-arousal/1"}, legacy):
        with pytest.raises(ValueError, match="cadence-arousal/3"):
            cd.Arousal.from_dict({**earlier, "config": {**earlier["config"], "need_scale": 0.1}})
    # a zero written under an earlier format, as ``ArousalConfig.to_dict()`` writes it, is the
    # founder and loads
    written = cd.Arousal.from_dict({**founder, "config": {**founder["config"], "need_scale": 0.0}})
    assert written.config.need_scale == 0.0 and written.to_dict() == founder
    scaled = cd.Arousal.from_dict({**founder, "format": "cadence-arousal/3",
                                   "config": {**founder["config"], "need_scale": 0.1}})
    assert scaled.config.need_scale == 0.1
    assert scaled.to_dict()["format"] == "cadence-arousal/3"
    assert state_without_config(scaled.to_dict()) == state_without_config(founder)
    with pytest.raises(ValueError, match="incomplete saved arousal config"):
        cd.Arousal.from_dict({**founder, "format": "cadence-arousal/3"})
    with pytest.raises(ValueError, match="incomplete saved arousal"):
        cd.Arousal.from_dict({k: v for k, v in scaled.to_dict().items() if k != "rewards"})
    # a scaled save that names the founder zero loads and saves as the founder
    zero = cd.Arousal.from_dict({**founder, "format": "cadence-arousal/3",
                                 "config": {**founder["config"], "need_scale": 0.0}})
    assert zero.to_dict() == founder


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
# checkpoint it saves.
PINS: dict[str, dict[str, str]] = {
    "adult": {
        "acts": (
            "10000000000000000000000000110000000000000000000000000000000000000000000000000100"
            "00000000000100110100000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000"
        ),
        "format": "cadence-generic/3",
        "names": "a2556d56acd5de594cda87ec20272b09e5c26e3e8dc6370b6e39eae567c53f4e",
        "checkpoint": "9024bb2c4eb6048434ef9fe0252017dec2f0d1fd7c6d9a9a59195ba3775102a7",
        "left": "b9e6671408995fc6cbda05329f6d6b53c3902076c6f2e20ce75be85eedd3566d",
        "arousal": "b539907cc61d32ba90a85abf48af5d5d5187e5f49d496383a3075e01e7accc49",
    },
    "needy": {
        "acts": (
            "00100000000000001000000000000000000000000000000000000001000000000000000000001000"
            "00000000001000000000000000000000000000000000000000000000000000000000000000001111"
            "0111111111111111111111111111111111111111100"
        ),
        "format": "cadence-generic/3",
        "names": "b6a65dae0eaab25db0caf6cb4691b71a079d90080e67ef5cbd76ed0e7aaede79",
        "checkpoint": "b56e8c80eb36868a7e7a2369a457e305a018f2fda94db142b4d761b0141dd6de",
        "left": "e37b3e819f294da2aa27afd0ca1607ab9ac86f0d8f149fadc41c1a92b055d6a0",
        "arousal": "19f9a55109dc9efcdc4ed1ca7f83877b98a5c56d22570d27f220b7791438e2be",
    },
    "composed": {
        "acts": "11110110010001100100010001000000000000",
        "format": "cadence-generic/8",
        "names": "f68680bd8a9b32b381bdd3620503110329f7b2ebec08401a86188c413a7f49e0",
        "checkpoint": "f180f4f6c18b498ab0fc28ae0cab8d9e48904d24733da0b43fcce045a3a689d2",
        "left": "5e9130bb3328fe57973e96e478f3fd73333d76dc075ebb6a01e2002cbe93c1ff",
        "arousal": "6e43f6d3595d72185f4b93eac84edb54ebc6a7d600e7556f001b349d34611ab1",
    },
    "episodic": {
        "acts": "11111101111111111111111111111111111110",
        "format": "cadence-generic/6",
        "names": "2d5620ee6ddfa07ee9bdfbb174903829313bc90cb9ff24c09e9e530f7e2dd4be",
        "checkpoint": "66f31ba4612978a8b42b4126443452105a4f6718bdfa8ef4c15ca105e27c3af0",
        "left": "7f4cd57a6e07a2d0ec07cc955e523a82159199d1b9ed05351abfe2ab6ebf9e48",
        "arousal": "0ad681643bf6ceb7f339791bf26a885a736536c38b338f38d5fd8112f99c3a92",
    },
}


def _build(name: str) -> tuple[cd.Brain, Life]:
    (inputs, actions, options), life = LIVES[name]
    return cd.Brain.compose(inputs, actions, **options), life


@pytest.mark.parametrize("name", sorted(PINS))
def test_the_founder_saves_and_describes_as_on_main(tmp_path: Path, name: str) -> None:
    """On every backend: the checkpoint's format, members and metadata keys, and the saved
    arousal states, which carry no need scale and keep the format of the law without it."""
    brain, life = _build(name)
    _, left = life(brain)
    assert brain.arousal is not None
    path = brain.save(tmp_path / name)
    assert names(path) == (PINS[name]["format"], PINS[name]["names"])
    for state in (*left, brain.arousal.to_dict()):
        assert state["format"] == "cadence-arousal/2" and "need_scale" not in state["config"]
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
            record = None if rng.random() < 0.5 else float(abs(rng.normal(0.0, 0.5)))
            surprise, want = arousal.outcome(error, reward, own=own, learned=learned,
                                             record_error=record)
            readings.append([surprise, want, arousal.want, arousal.heat, arousal.aroused])
            arousal.lived(int(rng.integers(1, 40)), int(rng.integers(0, 10)))
            if t == 60:
                arousal.reset()
        out[name] = (readings, arousal.to_dict())
    return out


# Recorded on main at c7dc2f0: the law's readings on the stream above and its saved state.
LAW: dict[str, tuple[str, str]] = {
    "founder": ("50a92666d40294ae132eeff72b6aa3c0e178c4eb981e6076ec324a5825b09c1b",
                "11eb946b14f67a60d46faf742fffccb1df82edb0e71c622430806250334ab538"),
    "needy": ("f41e6fd01cacac4128368ce8d3cbb6b0e5d785f15edd2ca501d023913ba82066",
              "5a8494cf7ccb61bbfd678f0737425253ed587c3f34a7754e68f015b6eac4eea4"),
}


def test_the_law_alone_reads_and_saves_as_on_main() -> None:
    for name, (readings, state) in law_readings().items():
        assert state["format"] == "cadence-arousal/2" and "need_scale" not in state["config"]
        assert (digest(readings), digest(state)) == LAW[name]
