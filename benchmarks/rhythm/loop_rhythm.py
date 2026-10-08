"""The loop chamber of issue 140: a brain taught to predict the next event of a periodic
pattern plays it in the closed loop, hearing its own output, from a prime.

A brain that predicts well on held-out rows can collapse, when fed its own output, to one
trajectory that does not depend on the prime: on the C64 rows of 0.73.1 the loop held forever,
because a brain that cannot count rows since its last onset, hearing its own holds, has hold as
its only consistent answer. This chamber is the smallest version of that observation: one
voice, an onset every ``period`` rows, the heard event and a constant drive as the only input,
the next event as the only label.

The declared mechanism is the efference copy with a decay above zero (0.76.0): a decaying copy
of the brain's own last onset is a count of rows since it. The declared contract is that the
heard event enters the copy before the brain answers a row, in teaching, in watching and in
the prime of a play, and the brain's own command is taken back out of it, so that the copy
carries the heard stream alone; in the closed loop the brain's own command is the heard event
and stays. Without that contract a brain whose copy holds only its own greedy commands never
has an onset in it while it still emits holds, and the majority fixed point seals itself.
Teaching follows the routine rule of ``Brain.live``: a row answered right teaches nothing, a
wrong or refused answer is followed by one lesson on that row; a lesson on every row (the
watching recipe of the C64 lane) made the pattern come and go from pass to pass on the
development seeds. Arms on the same founder weights and the same rows:

- ``copy``      the copy at the protocol's decay under the declared contract;
- ``own``       the same copy, written only by the brain's own greedy commands (the simpler
                control of the contract);
- ``nocopy``    no copy: the working trace alone (the 0.73.1 setting);
- ``frozen``    the ``copy`` founder without lessons, played under the same contract (a beat
                from birth is not credited);
- ``ngram``     a table over the last ``period`` events, taught on the same rows: the matched
                conventional learner that counts explicitly;
- ``hold``      hold forever; ``random`` uniform random events.

Every brain arm is first watched: from reset it hears the taught rows once more with free
greedy acts and no lesson, and its agreement with the next event is the open-loop reading (the
agreement inside teaching is not that reading, since there every act follows a lesson on the
same row). Then every arm plays ``plays`` closed loops of ``rows`` rows from primes at every
phase of the pattern. Readings: onset rate against the truth's, agreement with the primed
pattern's own continuation, agreement with the other phases' continuations (prime dependence)
and the pairwise correlation of the plays across primes (the issue's own measure). Receipts
bind the chamber and the library. Run ``python benchmarks/rhythm/loop_rhythm.py --out DIR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd
from cadence import Brain, LearnerConfig, Receipt

SCHEMA = "loop-rhythm/1"
PROTOCOL_PATH = Path(__file__).with_name("protocol-loop.json")
ARMS = ("copy", "own", "nocopy", "frozen", "ngram", "hold", "random")
BRAIN_ARMS = ("copy", "own", "nocopy", "frozen")
CONTRACT_ARMS = ("copy", "frozen")  # the heard event enters the copy before the brain answers
HOLD, ONSET = 0, 1
INPUTS = 3  # the heard event (hold, onset) and a constant drive
REQUIRED = ("schema", "pattern", "teaching", "play", "recipe", "copy", "ngram", "seeds", "gates")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    missing = [key for key in REQUIRED if key not in protocol]
    if missing:
        raise ValueError(f"protocol lacks {missing}")
    if set(protocol["seeds"]["development"]) & set(protocol["seeds"]["confirmation"]):
        raise ValueError("confirmation seeds must be fresh")
    return protocol, hashlib.sha256(raw).hexdigest()


# -- the pattern


def pattern(period: int, rows: int, phase: int = 0) -> np.ndarray:
    """An onset every ``period`` rows, the first at row ``phase``."""
    return np.array([ONSET if (t - phase) % period == 0 else HOLD for t in range(rows)])


def observation(event: int) -> np.ndarray:
    x = np.zeros((1, INPUTS))
    x[0, 2] = 1.0
    x[0, event] = 1.0
    return x


def continuation(period: int, prime: np.ndarray, rows: int) -> np.ndarray:
    """What the primed pattern itself does next: the onsets keep their period and phase."""
    last_onset = int(np.flatnonzero(prime == ONSET)[-1])
    start = len(prime)
    return np.array(
        [ONSET if (t - last_onset) % period == 0 else HOLD for t in range(start, start + rows)]
    )


# -- brains


def make_brain(seed: int, protocol: dict, arm: str) -> Brain:
    recipe = protocol["recipe"]
    options: dict[str, Any] = {
        "episodic": False,
        "lateral": recipe["lateral"],
        "resting_bias": recipe["resting_bias"],
        "working_memory_amplitude": recipe["trace_amplitude"],
        "working_memory_decay": recipe["trace_decay"],
        "learning": LearnerConfig(**recipe["learning"]),
    }
    if arm in ("copy", "own", "frozen"):
        options["efference_amplitude"] = protocol["copy"]["amplitude"]
        options["efference_decay"] = protocol["copy"]["decay"]
    return Brain.compose(INPUTS, 2, modules=tuple(recipe["modules"]), seed=seed, **options)


class Work:
    def __init__(self) -> None:
        self.lessons = self.refused_lessons = self.acts = self.refused_acts = 0
        self.lesson_sweeps = self.act_sweeps = 0
        self.seconds = 0.0

    def summary(self) -> dict:
        return dict(vars(self))


def act(brain: Brain, x: np.ndarray, work: Work) -> int | None:
    start = time.perf_counter()
    work.acts += 1
    prior = brain.last_settlement
    try:
        answer = int(brain.act(x, greedy=True)[0])
    except RuntimeError:
        report = brain.last_settlement
        if report is None or report is prior or report["qualified"]:
            raise
        work.refused_acts += 1
        answer = None
    finally:
        work.seconds += time.perf_counter() - start
    work.act_sweeps += int(brain.last_settlement["steps"])
    return answer


def teach(brain: Brain, x: np.ndarray, label: int, work: Work) -> bool:
    from cadence.learning import LearningPhaseError

    start = time.perf_counter()
    work.lessons += 1
    try:
        _, report = brain.learner.step(brain.stimulus(x), np.array([label]))
    except LearningPhaseError as error:
        work.refused_lessons += 1
        report = error.report
        work.lesson_sweeps += int(report["total_steps"])
        return False
    finally:
        work.seconds += time.perf_counter() - start
    work.lesson_sweeps += int(report["total_steps"])
    return True


def memory_state(brain: Brain) -> dict[str, dict[str, np.ndarray]]:
    """The working trace and the copy as they stand: what the next settle will read."""
    state: dict[str, dict[str, np.ndarray]] = {}
    for name in ("working_memory", "efference"):
        part = getattr(brain, name)
        if part is not None:
            state[name] = {k: getattr(part, k).copy() for k in ("trace", "last", "cold")}
    return state


def restore_memory(brain: Brain, state: dict[str, dict[str, np.ndarray]]) -> None:
    for name, values in state.items():
        part = getattr(brain, name)
        for k, v in values.items():
            setattr(part, k, v.copy())


def hear(
    brain: Brain,
    arm: str,
    event: int,
    work: Work,
    label: int | None = None,
    keep: bool = False,
) -> int | None:
    """One row under the declared contract: the heard event enters the copy, the brain answers
    the row with a free greedy act and, when ``label`` is given and the answer is not the
    label, a lesson on the same drive follows (the trace and the copy are put back to what the
    act read for the lesson and left as the act left them after it): a right answer is routine
    and teaches nothing. Unless ``keep``, the brain's own command is then taken back out of
    the copy, so the copy carries the heard stream alone, one update per row. Under ``own`` the
    copy carries nothing but the brain's own commands."""
    echo = brain.efference if arm in CONTRACT_ARMS else None
    if echo is not None:
        if not len(echo.trace):
            echo.reset(1)
        echo.issue(np.eye(2)[[event]])
    x = observation(event)
    before = memory_state(brain)
    answer = act(brain, x, work)
    if label is not None and answer != label:
        after = memory_state(brain)
        restore_memory(brain, before)
        teach(brain, x, label, work)
        restore_memory(brain, after)
    if echo is not None and not keep:
        for k, v in before["efference"].items():
            setattr(echo, k, v)
    return answer


def teach_brain(brain: Brain, arm: str, rows: np.ndarray, passes: int, work: Work) -> dict:
    """The teaching contract: every row is heard and answered with a free greedy act, and only
    a wrong or refused answer is followed by a lesson on that row with the next event as the
    label. The agreement per pass is the share of right answers, each given before any lesson
    on its row (the online reading); ``watch_brain`` is the reading after teaching."""
    agreement, lessons = [], []
    for _ in range(passes):
        brain.reset()
        hits = total = 0
        given = work.lessons
        for t in range(len(rows) - 1):
            label = int(rows[t + 1])
            answer = hear(brain, arm, int(rows[t]), work, label=label)
            if answer is None:
                continue
            hits += int(answer == label)
            total += 1
        agreement.append(hits / max(1, total))
        lessons.append(work.lessons - given)
    return {
        "agreement_per_pass": agreement,
        "lessons_per_pass": lessons,
        "last_pass_agreement": agreement[-1] if agreement else None,
    }


def watch_brain(brain: Brain, arm: str, rows: np.ndarray, work: Work) -> dict:
    """The open-loop reading: from reset, the brain hears the rows once with free greedy acts
    and no lesson; agreement is the share of its answers equal to the next event, onset recall
    the share of onsets it announced."""
    brain.reset()
    hits = total = onsets = announced = 0
    for t in range(len(rows) - 1):
        label = int(rows[t + 1])
        answer = hear(brain, arm, int(rows[t]), work)
        if answer is None:
            continue
        total += 1
        hits += int(answer == label)
        if label == ONSET:
            onsets += 1
            announced += int(answer == ONSET)
    return {
        "agreement": hits / max(1, total),
        "onset_recall": announced / max(1, onsets),
        "rows": total,
        "refusals": len(rows) - 1 - total,
    }


def play_brain(
    brain: Brain, arm: str, prime: np.ndarray, rows: int, work: Work
) -> list[int | None]:
    """Hear the prime, then hear your own output for ``rows`` rows. The answer to the last
    prime row is the first event of the loop, and from there every heard event is the brain's
    own last answer (hold when it refused)."""
    brain.reset()
    answer: int | None = None
    last = len(prime) - 1
    for t, event in enumerate(prime):
        answer = hear(brain, arm, int(event), work, keep=t == last)
    out: list[int | None] = [answer]
    heard = HOLD if answer is None else answer
    for _ in range(rows - 1):
        answer = act(brain, observation(heard), work)
        out.append(answer)
        heard = HOLD if answer is None else answer
    return out


class NGram:
    """A table over the last ``order`` events, taught on the same rows; the matched learner
    that counts explicitly."""

    def __init__(self, order: int) -> None:
        self.order = order
        self.counts: dict[tuple[int, ...], np.ndarray] = {}

    def teach(self, rows: np.ndarray) -> None:
        history: list[int] = []
        for t in range(len(rows) - 1):
            history.append(int(rows[t]))  # the key ends with the current event, as in play
            key = tuple(history[-self.order :])
            self.counts.setdefault(key, np.zeros(2))[int(rows[t + 1])] += 1

    def answer(self, history: list[int]) -> int:
        key = tuple(history[-self.order :])
        counts = self.counts.get(key)
        return HOLD if counts is None else int(np.argmax(counts))

    def play(self, prime: np.ndarray, rows: int) -> list[int]:
        history = [int(e) for e in prime]
        out = []
        for _ in range(rows):
            nxt = self.answer(history)
            out.append(nxt)
            history.append(nxt)
        return out


# -- scoring


def score_play(out: list[int | None], truth: np.ndarray) -> dict:
    played = np.array([HOLD if v is None else v for v in out])
    return {
        "onset_rate": float(np.mean(played == ONSET)),
        "agreement": float(np.mean(played == truth)),
        "refusals": int(sum(v is None for v in out)),
    }


def correlation(a: list[int | None], b: list[int | None]) -> float | None:
    x = np.array([HOLD if v is None else v for v in a], dtype=float)
    y = np.array([HOLD if v is None else v for v in b], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_founder(seed: int, arm: str, protocol: dict, directory: Path) -> dict:
    directory.mkdir(parents=True)
    warnings.simplefilter("ignore")
    period = int(protocol["pattern"]["period"])
    rows = pattern(period, int(protocol["teaching"]["rows"]))
    prime_rows = int(protocol["play"]["prime"])
    play_rows = int(protocol["play"]["rows"])
    primes = [pattern(period, prime_rows, phase) for phase in range(period)]
    truths = [continuation(period, prime, play_rows) for prime in primes]
    work = Work()
    result: dict[str, Any] = {"arm": arm, "seed": seed}
    rng = np.random.default_rng(seed)
    if arm in BRAIN_ARMS:
        brain = make_brain(seed, protocol, arm)
        brain.save(directory / "initial.npz")
        if arm != "frozen":
            result["teaching"] = teach_brain(
                brain, arm, rows, int(protocol["teaching"]["passes"]), work
            )
            brain.save(directory / "taught.npz")
        result["watching"] = watch_brain(brain, arm, rows, work)
        plays = [play_brain(brain, arm, prime, play_rows, work) for prime in primes]
    elif arm == "ngram":
        table = NGram(int(protocol["ngram"]["order"]))
        table.teach(rows)
        plays = [table.play(prime, play_rows) for prime in primes]
    elif arm == "hold":
        plays = [[HOLD] * play_rows for _ in primes]
    else:
        plays = [[int(v) for v in rng.integers(0, 2, play_rows)] for _ in primes]
    result["plays"] = [[-1 if v is None else v for v in play] for play in plays]
    result["scores"] = [score_play(play, truth) for play, truth in zip(plays, truths, strict=True)]
    # prime dependence: a play should agree with its own prime's continuation more than with
    # the continuations of the other phases
    cross = [
        [
            float(np.mean(np.array([HOLD if v is None else v for v in play]) == other))
            for other in truths
        ]
        for play in plays
    ]
    result["cross_agreement"] = cross
    result["prime_dependent"] = all(
        row[i] > max(row[j] for j in range(len(row)) if j != i) for i, row in enumerate(cross)
    )
    pairs = [
        correlation(plays[i], plays[j]) for i in range(len(plays)) for j in range(i + 1, len(plays))
    ]
    present = [c for c in pairs if c is not None]
    result["cross_prime_correlation"] = float(np.mean(present)) if present else None
    result["mean_onset_rate"] = float(np.mean([s["onset_rate"] for s in result["scores"]]))
    result["mean_agreement"] = float(np.mean([s["agreement"] for s in result["scores"]]))
    result["refusals"] = int(sum(s["refusals"] for s in result["scores"]))
    result["work"] = work.summary()
    return result


def gates(rows: list[dict], protocol: dict) -> dict:
    g = protocol["gates"]
    period = int(protocol["pattern"]["period"])
    truth_rate = 1.0 / period
    hold_agreement = 1.0 - truth_rate
    out: dict[str, Any] = {}
    for arm in sorted({r["arm"] for r in rows}, key=ARMS.index):
        group = [r for r in rows if r["arm"] == arm]
        fires = sum(
            abs(r["mean_onset_rate"] - truth_rate) <= g["onset_tolerance"] * truth_rate
            for r in group
        )
        follows = sum(r["mean_agreement"] >= hold_agreement + g["agreement_margin"] for r in group)
        depends = sum(bool(r["prime_dependent"]) for r in group)
        out[arm] = {
            "founders": len(group),
            "fires": fires,
            "follows": follows,
            "prime_dependent": depends,
            "mean_onset_rate": float(np.mean([r["mean_onset_rate"] for r in group])),
            "mean_agreement": float(np.mean([r["mean_agreement"] for r in group])),
            "refusals": int(sum(r["refusals"] for r in group)),
        }
        if all("watching" in r for r in group):
            out[arm]["mean_watching"] = float(np.mean([r["watching"]["agreement"] for r in group]))
    if "copy" in out and "frozen" in out:
        frozen = {r["seed"]: r for r in rows if r["arm"] == "frozen"}
        learned = sum(
            abs(r["mean_onset_rate"] - truth_rate) <= g["onset_tolerance"] * truth_rate
            and r["mean_agreement"] >= hold_agreement + g["agreement_margin"]
            and bool(r["prime_dependent"])
            and not (
                r["seed"] in frozen
                and frozen[r["seed"]]["mean_agreement"] >= hold_agreement + g["agreement_margin"]
            )
            for r in rows
            if r["arm"] == "copy"
        )
        out["copy"]["learned"] = learned
        out["passed"] = learned >= g["share"] and out["copy"]["refusals"] == 0
    return out


def verify(directory: Path) -> tuple[bool, str]:
    try:
        path = directory / "summary.json"
        raw = json.loads(path.read_text())
        sources = [(item["path"], directory / item["path"]) for item in raw["source"]["files"]]

        def check(body: dict) -> str | None:
            for name, expected in body["artifacts"].items():
                if sha256(directory / name) != expected:
                    return f"artifact differs: {name}"
            if body["protocol_sha256"] != sha256(directory / "protocol.json"):
                return "protocol copy differs from the recorded hash"
            if gates(body["runs"], body["protocol"]) != body["gates"]:
                return "the stored gates do not follow from the runs"
            return None

        valid, reason = Receipt.verify(path, sources=sources, check=check)
        return valid, (
            "canonical form, digest, sources, artifact hashes and gates agree" if valid else reason
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, f"cannot verify artifact: {error}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out", type=Path)
    destination.add_argument("--verify", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--passes", type=int, default=None)
    args = parser.parse_args(argv)
    if args.verify is not None:
        valid, reason = verify(args.verify)
        print(json.dumps({"valid": valid, "reason": reason}))
        return 0 if valid else 1
    protocol, protocol_sha = load_protocol(args.protocol)
    overrides: dict[str, Any] = {}
    if args.passes is not None:
        protocol["teaching"]["passes"] = args.passes
        overrides["passes"] = args.passes
    seeds = list(protocol["seeds"]["confirmation"]) if args.seeds is None else list(args.seeds)
    if (
        len(seeds) != len(set(seeds))
        or any(s < 0 for s in seeds)
        or any(a not in ARMS for a in args.arms)
        or len(args.arms) != len(set(args.arms))
    ):
        parser.error("invalid seeds or arms")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_bytes(Path(args.protocol).read_bytes())
    began = time.perf_counter()
    declaration = {
        "schema": SCHEMA,
        "protocol_sha256": protocol_sha,
        "frozen_protocol": not overrides and args.seeds is None,
        "overrides": overrides,
        "seeds": seeds,
        "arms": list(args.arms),
        "python": sys.version,
        "platform": platform.platform(),
        "cadence_version": cd.__version__,
        "cadence_import": str(Path(cd.__file__).resolve()),
        "cpu_count": os.cpu_count(),
    }
    (args.out / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    sources, origins = [], []
    package = Path(cd.__file__).resolve().parent
    own = [Path(__file__).resolve()]
    for source in [*own, *sorted(package.rglob("*.py"))]:
        relative = (
            "source/" + source.name
            if source in own
            else "source/cadence/" + source.relative_to(package).as_posix()
        )
        target = args.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sources.append((relative, target))
        origins.append((source, target))
    runs = []
    for seed in seeds:
        for arm in args.arms:
            result = run_founder(seed, arm, protocol, args.out / f"seed{seed}-{arm}")
            runs.append(result)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": arm,
                        "watching": (
                            round(result["watching"]["agreement"], 3)
                            if "watching" in result
                            else None
                        ),
                        "onset_rate": round(result["mean_onset_rate"], 3),
                        "agreement": round(result["mean_agreement"], 3),
                        "prime_dependent": result["prime_dependent"],
                        "seconds": round(time.perf_counter() - began, 1),
                    }
                ),
                flush=True,
            )
    assert all(sha256(original) == sha256(copy) for original, copy in origins), (
        "source changed during the run; retain this incomplete attempt and rerun"
    )
    artifacts = {
        p.relative_to(args.out).as_posix(): sha256(p)
        for p in sorted(args.out.rglob("*"))
        if p.is_file()
        and "source" not in p.relative_to(args.out).parts
        and p.name != "summary.json"
    }
    summary = {
        "declaration": declaration,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "frozen_protocol": declaration["frozen_protocol"],
        "runs": runs,
        "gates": gates(runs, protocol),
        "seconds": time.perf_counter() - began,
        "artifacts": artifacts,
        "work_scope": (
            "Every lesson and free act of every arm and play is charged; sweeps are not joules."
        ),
    }
    Receipt.build(SCHEMA, summary, sources).write(args.out / "summary.json")
    valid, reason = verify(args.out)
    assert valid, reason
    print(
        json.dumps(
            {
                "verified": valid,
                "gates": summary["gates"].get("passed"),
                "summary": str(args.out / "summary.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
