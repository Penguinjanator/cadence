"""The finite continuing recall chamber of issue 84, roadmap row 02: can one continuing
``Brain.compose`` remember one of two observed tokens after its cue disappears, at a fixed
state size, through its working trace rather than a supplied answer or a warm numerical
guess? The reviewed protocol is ``FINITE_HORIZON_PROTOCOL.md``; the frozen inputs are built by
``finite_horizon_inputs.py`` before any brain runs; ``protocol-finite.json`` declares every
setting, the founders, the caps and the gates.

Arms, one continuing life each per founder, on the same frozen episodes:

- ``vanished``  the declared recipe: trace decay 0.8, amplitude 1, the trace's write fixed,
                its read learned at the QUERY lessons only; it never sees a history coordinate;
- ``default``   the same brain with the composed working-trace defaults (amplitude 3,
                decay 0.2): the simpler setting, retained as the control of the declared gene;
- ``history``   the external-history comparator: byte-equal initial arrays, the same lessons,
                and the actual observed payloads of the last four WRITE events appended at QUERY
                by the world; supplied assistance, counted separately;
- ``random``    the frozen uniform-random actions of every row.

At every evaluation query the vanished and default lives are saved and forked: intact (the
life continues), erased trace, shuffled trace (transplanted from the paired row with the
opposite value) and full reset. The supported horizon is the largest contiguous passed prefix
over 0, 1, 2 intervening events under the prewritten gates; nuisance, replacement, order,
continuation, purity and resource gates must also pass. A refusal is wrong; unrun planned rows
are wrong. Run ``python benchmarks/recall/finite_horizon.py --out DIR``; verify with
``--verify DIR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np  # noqa: E402  (after the thread environment is set)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finite_horizon_inputs as inputs  # noqa: E402

import cadence  # noqa: E402
from cadence import Brain, LearnerConfig, Receipt  # noqa: E402
from cadence.learning import LearningPhaseError  # noqa: E402

SCHEMA = "finite-recall/1"
PROTOCOL_PATH = Path(__file__).with_name("protocol-finite.json")
ARMS = ("vanished", "default", "history", "random")
CONTROLS = ("intact", "erased", "shuffled", "reset")
CONDITIONS = {c.name: c for c in inputs.TEST_CONDITIONS}
PREFIX = (("clean-0",), ("clean-1", "distractor-1"), ("clean-2", "distractor-2"))
NUISANCE = ("partial-1", "noise-1", "replacement-1", "order-latest-1")
REQUIRED = ("seeds", "brain", "learning", "arms", "training", "evaluation", "caps", "gates")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    missing = [key for key in REQUIRED if key not in protocol]
    if missing:
        raise ValueError(f"protocol lacks {missing}")
    return protocol, hashlib.sha256(raw).hexdigest()


# -- brains


def make_brain(protocol: dict, seed: int, *, decay: float, amplitude: float) -> Brain:
    genes = protocol["brain"]
    learning = LearnerConfig(**protocol["learning"])
    return Brain.compose(
        inputs.INPUTS,
        2,
        modules=tuple(genes["modules"]),
        lateral=genes["lateral"],
        seed=seed,
        learning=learning,
        episodic=False,
        working_memory_decay=decay,
        working_memory_amplitude=amplitude,
    )


def brain_for(arm: str, protocol: dict, seed: int) -> Brain:
    genes = protocol["brain"]
    if arm == "default":
        return make_brain(
            protocol, seed, decay=genes["default_decay"], amplitude=genes["default_amplitude"]
        )
    return make_brain(
        protocol, seed, decay=genes["trace_decay"], amplitude=genes["trace_amplitude"]
    )


# -- work


@dataclass
class Work:
    """Actual solver work, with the trace audit; sweeps are not joules."""

    action_attempts: int = 0
    action_refusals: int = 0
    action_sweeps: int = 0
    action_row_sweeps: int = 0
    action_residual_checks: int = 0
    teacher_attempts: int = 0
    teacher_refusals: int = 0
    teacher_presentations: int = 0
    teacher_sweeps: int = 0
    teacher_row_sweeps: int = 0
    trace_row_updates: int = 0
    trace_audits: int = 0
    trace_audit_max_error: float = 0.0
    checkpoints: int = 0
    calls_seconds: float = 0.0
    reports: list[dict] = field(default_factory=list, repr=False)

    def summary(self) -> dict:
        out = asdict(self)
        del out["reports"]
        return out

    def act(self, brain: Brain, x: np.ndarray, *, audit: bool = True) -> np.ndarray | None:
        """One greedy act of every row; a refused act is a missed answer. The trace's update
        is checked against the literal source recurrence of the protocol."""
        trace = brain.working_memory
        before = (
            None
            if trace is None or not audit
            else {name: getattr(trace, name).copy() for name in ("trace", "last", "cold")}
        )
        start = time.perf_counter()
        self.action_attempts += 1
        prior = brain.last_settlement
        try:
            answer = brain.act(x, greedy=True)
        except RuntimeError:
            report = brain.last_settlement
            if report is None or report is prior or report["qualified"]:
                raise
            self.action_refusals += 1
            answer = None
        finally:
            self.calls_seconds += time.perf_counter() - start
        report = dict(brain.last_settlement)
        self.reports.append({"operation": "act", **report})
        self.action_sweeps += int(report["steps"])
        self.action_row_sweeps += len(x) * int(report["steps"])
        self.action_residual_checks += int(report["residual_checks"])
        if answer is not None:
            self.trace_row_updates += len(x)
            if before is not None and len(before["trace"]) == len(x):
                state = brain.basal_ganglia.state
                assert state is not None and trace is not None
                h = np.atleast_2d(np.asarray(state.activation))[:, brain.association_index]
                after = {name: getattr(trace, name) for name in ("trace", "last", "cold")}
                error = audit_trace(before, after, h, trace.decay)
                self.trace_audits += 1
                self.trace_audit_max_error = max(self.trace_audit_max_error, error)
        return answer

    def teach(
        self, brain: Brain, x: np.ndarray, labels: np.ndarray, *, drive: np.ndarray | None = None
    ) -> bool:
        """One lesson on the rows of ``labels``; ``drive`` is the stimulus the rows' free act
        read when the lesson follows that act (the trace has moved on since)."""
        start = time.perf_counter()
        self.teacher_attempts += 1
        accepted = True
        try:
            _, report = brain.learner.step(brain.stimulus(x) if drive is None else drive, labels)
        except LearningPhaseError as error:
            self.teacher_refusals += 1
            report = error.report
            accepted = False
        finally:
            self.calls_seconds += time.perf_counter() - start
        self.reports.append({"operation": "teach", **report})
        self.teacher_presentations += int(report["attempted_presentations"])
        self.teacher_sweeps += int(report["total_steps"])
        self.teacher_row_sweeps += int(report["total_row_sweeps"])
        return accepted


def audit_trace(before: dict, after: dict, h: np.ndarray, decay: float) -> float:
    """An independent literal recurrence for the arm's declared decay, ``decay * trace +
    (1 - decay) * h``; the declared recipe's decay of 0.8 is also checked by the reviewed
    fixture's own function, which hard-codes that recipe."""
    expected = decay * np.asarray(before["trace"]) + (1.0 - decay) * np.asarray(h)
    error = float(np.max(np.abs(np.asarray(after["trace"]) - expected)))
    if not np.array_equal(after["last"], h) or np.asarray(after["cold"]).any():
        raise ValueError("last/cold do not record exactly one admitted real event")
    if not np.isfinite(error) or error > 1e-12:
        raise ValueError("trace does not match one declared accepted-event update")
    if decay == 0.8:
        error = max(error, inputs.check_trace_transition(before, after, h))
    return error


def same_arrays(first: Path, second: Path) -> bool:
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        return set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)


# -- the frozen episodes


def episode_arrays(frozen: dict[str, np.ndarray], phase: str, index: int) -> dict[str, np.ndarray]:
    return {
        name: frozen[f"{phase}/{index:04d}/{name}"]
        for name in (
            "observations",
            "labels",
            "token_values",
            "permutation",
            "uniform_actions",
            "timestamps",
            "irregular_timestamps",
        )
    }


def condition_name(frozen: dict[str, np.ndarray], phase: str, index: int) -> str:
    conditions = inputs.TRAIN_CONDITIONS if phase == "train" else inputs.TEST_CONDITIONS
    return conditions[int(frozen[f"{phase}/condition"][index])].name


# -- one arm's life


class Capped(RuntimeError):
    pass


def check_caps(began: float, directory: Path, caps: dict) -> None:
    if time.time() - began > caps["seconds"]:
        raise Capped("the worker's wall cap was reached")
    size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    if size > caps["bytes"]:
        raise Capped("the retained output cap was reached")


def train_arm(
    arm: str, brain: Brain, frozen: dict, protocol: dict, began: float, directory: Path
) -> dict:
    """The frozen training episodes: greedy acts and, at every QUERY, under the ``every``
    rule one lesson before the act (recall/1 and recall/2); under the ``surprise`` rule the
    act first and a lesson only on the rows it answered wrong, on the drive that act read, so
    that a right answer is routine and teaches nothing (recall/3)."""
    work = Work()
    count = int(len(frozen["train/condition"]))
    refused_life = False
    rule = str(protocol["training"].get("rule", "every"))
    if rule not in ("every", "surprise"):
        raise ValueError("training.rule must be 'every' or 'surprise'")
    for index in range(count):
        if index % 16 == 0:
            check_caps(began, directory, protocol["caps"])
        arrays = episode_arrays(frozen, "train", index)
        observations = arrays["observations"]
        if arm == "history":
            observations = inputs.append_observed_history(observations)
        for event, x in enumerate(observations):
            query = event == len(observations) - 1
            if query and rule == "every":
                work.teach(brain, x, arrays["labels"])
            drive = brain.stimulus(x) if query and rule == "surprise" else None
            answer = work.act(brain, x)
            if answer is None:
                refused_life = True
                break
            if drive is not None:
                wrong = np.asarray(answer) != np.asarray(arrays["labels"])
                if wrong.any():
                    work.teach(brain, x, np.asarray(arrays["labels"])[wrong], drive=drive[wrong])
        if refused_life:
            break
    return {
        "episodes": count,
        "completed": not refused_life,
        "rule": rule,
        "work": work.summary(),
    }


def fork_answers(checkpoint: Path, x: np.ndarray, permutation: np.ndarray, work: Work) -> dict:
    """Every control begins at the same complete pre-query state, with learning disabled."""
    out = {}
    for control in CONTROLS[1:]:
        branch = Brain.load(checkpoint)
        work.checkpoints += 1
        trace = branch.working_memory
        assert trace is not None
        if control == "erased":
            trace.reset(len(x))
        elif control == "shuffled":
            for name in ("trace", "last", "cold"):
                setattr(trace, name, getattr(trace, name)[permutation].copy())
        elif control == "reset":
            branch.reset()
        answer = work.act(branch, x, audit=False)
        out[control] = None if answer is None else answer.tolist()
    return out


def seam_checks(
    brain: Brain,
    observations: np.ndarray,
    cue_index: int,
    replacement_index: int | None,
    directory: Path,
    work: Work,
    protocol: dict,
) -> dict:
    """Saved seams of one episode: after the first cue, before a replacement when present, and
    before QUERY, each resumed by a clone that must execute the remaining observations with equal
    answers and equal complete saved arrays; private imagination and a refused act must leave a
    checkpoint unchanged; the brain never sees a timestamp."""
    from dataclasses import replace

    seams = {}
    saved: dict[str, Path] = {}
    stops = {"after_cue": cue_index + 1, "before_query": len(observations) - 1}
    if replacement_index is not None:
        stops["before_replacement"] = replacement_index
    answers: list[list[int] | None] = []
    for event, x in enumerate(observations):
        for name, stop in stops.items():
            if event == stop:
                saved[name] = brain.save(directory / f"seam-{name}.npz")
                work.checkpoints += 1
        answer = work.act(brain, x)
        answers.append(None if answer is None else answer.tolist())
    final = brain.save(directory / "seam-final.npz")
    work.checkpoints += 1
    for name, stop in stops.items():
        if name not in saved:
            continue
        clone = Brain.load(saved[name])
        clone_work = Work()
        clone_answers = [
            (lambda a: None if a is None else a.tolist())(clone_work.act(clone, x, audit=False))
            for x in observations[stop:]
        ]
        resumed = clone.save(directory / f"seam-{name}-resumed.npz")
        seams[name] = {
            "answers_equal": clone_answers == answers[stop:],
            "saved_arrays_equal": same_arrays(final, resumed),
            "work": clone_work.summary(),
        }
        work.checkpoints += 1
    # private imagination and a refused act from the pre-query seam leave it unchanged
    pre = saved["before_query"]
    twin = Brain.load(pre)
    twin.imagine([observations[-1], observations[-1]])
    after_imagine = twin.save(directory / "seam-after-imagine.npz")
    unchanged_imagine = same_arrays(pre, after_imagine)
    twin = Brain.load(pre)
    kept = twin.learner.config
    twin.learner.config = replace(kept, free_steps=0, tolerance=1e-15)
    changed = observations[-1].copy()
    changed[:, inputs.PAYLOAD] = 1.0  # a changed cue the brain never trained on
    refused = False
    try:
        twin.act(changed, greedy=True)
    except RuntimeError:
        refused = True
    twin.learner.config = kept
    after_refusal = twin.save(directory / "seam-after-refusal.npz")
    unchanged_refusal = same_arrays(pre, after_refusal)
    twin_answer = twin.act(observations[-1], greedy=True)
    continues = twin_answer.tolist() == answers[-1]
    work.checkpoints += 4
    for path in directory.glob("seam-*.npz"):
        path.unlink()
    return {
        "seams": seams,
        "imagine_leaves_checkpoint": unchanged_imagine,
        "refused_act_leaves_checkpoint": refused and unchanged_refusal,
        "continuation_after_refusal_equal": continues,
    }


def evaluate_arm(
    arm: str, brain: Brain, frozen: dict, protocol: dict, began: float, directory: Path
) -> dict:
    """The 24 frozen evaluation episodes per condition: forks at QUERY for the brain arms with a
    trace, the plain answer for the history arm; seams on the first episode of every condition."""
    work = Work()
    count = int(len(frozen["test/condition"]))
    trials: list[dict] = []
    seams: dict[str, dict] = {}
    completed = True
    seen: set[str] = set()
    for index in range(count):
        check_caps(began, directory, protocol["caps"])
        arrays = episode_arrays(frozen, "test", index)
        name = condition_name(frozen, "test", index)
        observations = arrays["observations"]
        if arm == "history":
            observations = inputs.append_observed_history(observations)
        trial: dict[str, Any] = {
            "condition": name,
            "index": index,
            "labels": arrays["labels"].tolist(),
            "transplanted": arrays["labels"][arrays["permutation"]].tolist(),
        }
        if arm in ("vanished", "default") and name not in seen:
            seen.add(name)
            # the seam episode runs on a loaded clone of the life so the life's own answer at
            # this episode is read from the life itself afterwards
            clone = Brain.load(brain.save(directory / "life-before-seams.npz"))
            work.checkpoints += 2
            writes = [i for i, x in enumerate(observations) if x[0, inputs.WRITE] == 1]
            replacement = writes[1] if CONDITIONS[name].opposite and len(writes) > 1 else None
            seams[name] = seam_checks(
                clone, observations, writes[0], replacement, directory, work, protocol
            )
            (directory / "life-before-seams.npz").unlink()
        answers: list[list[int] | None] = []
        for x in observations[:-1]:
            answer = work.act(brain, x)
            answers.append(None if answer is None else answer.tolist())
            if answer is None:
                completed = False
                break
        if not completed:
            trial["branches"] = (
                {control: None for control in CONTROLS} if arm != "history" else {"answer": None}
            )
            trials.append(trial)
            break
        query = observations[-1]
        if arm == "history":
            answer = work.act(brain, query)
            trial["branches"] = {"answer": None if answer is None else answer.tolist()}
        else:
            checkpoint = brain.save(directory / f"query-{index:04d}.npz")
            work.checkpoints += 1
            branches = fork_answers(checkpoint, query, arrays["permutation"], work)
            answer = work.act(brain, query)
            branches["intact"] = None if answer is None else answer.tolist()
            trial["branches"] = branches
            checkpoint.unlink()
        trial["random"] = arrays["uniform_actions"].tolist()
        trials.append(trial)
        if answer is None:
            completed = False
            break
    # the timestamp fork: identical events under two schedules must give equal arrays; the
    # brain never sees a timestamp, and this is recorded rather than assumed
    timing = None
    if arm in ("vanished", "default") and completed:
        index = next(i for i in range(count) if condition_name(frozen, "test", i) == "clean-2")
        arrays = episode_arrays(frozen, "test", index)
        observations = arrays["observations"]
        saves = []
        for schedule in ("timestamps", "irregular_timestamps"):
            clone = Brain.load(brain.save(directory / "timing-base.npz"))
            clone_work = Work()
            for x in observations:
                clone_work.act(clone, x, audit=False)
            saves.append(clone.save(directory / f"timing-{schedule}.npz"))
            work.checkpoints += 2
        timing = {
            "schedules": ["regular", "irregular"],
            "equal_arrays": same_arrays(*saves),
            "elapsed_regular": float(arrays["timestamps"][-1]),
            "elapsed_irregular": float(arrays["irregular_timestamps"][-1]),
        }
        for path in directory.glob("timing-*.npz"):
            path.unlink()
    brain.save(directory / "final.npz")
    work.checkpoints += 1
    return {
        "completed": completed,
        "trials": trials,
        "seams": seams,
        "timing": timing,
        "work": work.summary(),
    }


def run_random(frozen: dict) -> dict:
    trials = []
    for index in range(int(len(frozen["test/condition"]))):
        arrays = episode_arrays(frozen, "test", index)
        trials.append(
            {
                "condition": condition_name(frozen, "test", index),
                "index": index,
                "labels": arrays["labels"].tolist(),
                "branches": {"answer": arrays["uniform_actions"].tolist()},
            }
        )
    return {"completed": True, "trials": trials, "work": Work().summary()}


# -- scores and gates


def accuracy(answers: list[list[int] | None], labels: list[list[int]], planned: int) -> dict:
    correct = attempted = 0
    for a, lab in zip(answers, labels, strict=True):
        if a is None:
            continue
        attempted += len(lab)
        correct += int(sum(int(x == y) for x, y in zip(a, lab, strict=True)))
    return {
        "correct": correct,
        "attempted": attempted,
        "planned": planned,
        "accuracy": correct / planned if planned else None,
    }


def paired(
    answers: list[list[int] | None], labels: list[list[int]], planned_pairs: int
) -> float | None:
    both = 0
    for a, lab in zip(answers, labels, strict=True):
        if a is None:
            continue
        for i in range(0, len(lab), 2):
            both += int(a[i] == lab[i] and a[i + 1] == lab[i + 1])
    return both / planned_pairs if planned_pairs else None


def score_arm(result: dict, repeats: int) -> dict:
    """Per condition: accuracy over planned rows for every branch, paired accuracy for the
    intact answer, and for the shuffled branch the accuracy against the transplanted value."""
    out = {}
    for name in CONDITIONS:
        trials = [t for t in result["trials"] if t["condition"] == name]
        planned = repeats * inputs.STREAMS
        labels = [t["labels"] for t in trials]
        entry: dict[str, Any] = {"planned_rows": planned, "attempted_episodes": len(trials)}
        if trials and "answer" in trials[0]["branches"]:
            answers = [t["branches"]["answer"] for t in trials]
            entry["answer"] = accuracy(answers, labels, planned)
            entry["paired"] = paired(answers, labels, repeats * inputs.STREAMS // 2)
        else:
            for control in CONTROLS:
                answers = [t["branches"].get(control) for t in trials]
                entry[control] = accuracy(answers, labels, planned)
            entry["paired"] = paired(
                [t["branches"].get("intact") for t in trials], labels, repeats * inputs.STREAMS // 2
            )
            entry["shuffled_vs_transplanted"] = accuracy(
                [t["branches"].get("shuffled") for t in trials],
                [t["transplanted"] for t in trials],
                planned,
            )
        out[name] = entry
    return out


def gates_for(founder: dict, protocol: dict) -> dict:
    g = protocol["gates"]
    scores = founder["scores"]
    vanished, history, random = scores["vanished"], scores["history"], scores["random"]
    work = founder["arms"]["vanished"]["work"]

    def passes(name: str) -> dict:
        v, h, r = vanished[name], history[name], random[name]
        delay = CONDITIONS[name].delay
        checks = {
            "intact": v["intact"]["accuracy"] is not None
            and v["intact"]["accuracy"] >= g["intact"],
            "paired": v["paired"] is not None and v["paired"] >= g["paired"],
            "history": h["answer"]["accuracy"] is not None
            and h["answer"]["accuracy"] >= g["history"],
            "above_random": v["intact"]["accuracy"] is not None
            and v["intact"]["accuracy"] >= r["answer"]["accuracy"] + g["random_margin"],
            "complete": v["intact"]["attempted"] == v["intact"]["planned"]
            and v["intact"]["accuracy"] is not None,
        }
        if delay > 0:
            checks["erased"] = (
                v["erased"]["accuracy"] <= g["lesion"]
                and v["intact"]["accuracy"] >= v["erased"]["accuracy"] + g["lesion_margin"]
            )
            checks["reset"] = (
                v["reset"]["accuracy"] <= g["lesion"]
                and v["intact"]["accuracy"] >= v["reset"]["accuracy"] + g["lesion_margin"]
            )
            checks["shuffled"] = (
                v["shuffled_vs_transplanted"]["accuracy"] >= g["shuffled_transplanted"]
                and v["shuffled"]["accuracy"] <= g["shuffled_original"]
            )
        return {"passed": all(checks.values()), **checks}

    report = {name: passes(name) for name in CONDITIONS}
    horizon = -1
    for depth, names in enumerate(PREFIX):
        if all(report[name]["passed"] for name in names):
            horizon = depth
        else:
            break
    nuisance = all(report[name]["passed"] for name in NUISANCE)
    seams = founder["arms"]["vanished"].get("seams", {})
    custody = all(
        all(s["answers_equal"] and s["saved_arrays_equal"] for s in entry["seams"].values())
        and entry["imagine_leaves_checkpoint"]
        and entry["refused_act_leaves_checkpoint"]
        and entry["continuation_after_refusal_equal"]
        for entry in seams.values()
    ) and bool(seams)
    timing = founder["arms"]["vanished"].get("timing")
    audit = work["trace_audit_max_error"] <= g["trace_audit"] and work["trace_audits"] > 0
    refusals = (
        work["action_refusals"] == 0
        and founder["arms"]["vanished"]["training"]["work"]["action_refusals"] == 0
    )
    return {
        "conditions": report,
        "horizon": horizon,
        "nuisance": nuisance,
        "custody": custody,
        "timing": timing is not None and timing["equal_arrays"],
        "trace_audit": audit,
        "no_refusals": refusals,
        "capped": founder.get("capped", False),
        "closure": horizon >= g["minimum_horizon"]
        and nuisance
        and custody
        and (timing is not None and timing["equal_arrays"])
        and audit
        and refusals
        and not founder.get("capped", False),
    }


# -- a founder


def run_founder(seed: int, protocol: dict, directory: Path, repeats: dict) -> dict:
    directory.mkdir(parents=True)
    began = time.time()
    frozen = (
        inputs.freeze_episodes(directory / "episodes.npz", seed=seed)
        if repeats is None
        else freeze_small(directory / "episodes.npz", seed, repeats)
    )
    founder: dict[str, Any] = {"seed": seed, "arms": {}, "capped": False}
    try:
        for arm in protocol["arms"]:
            if arm == "random":
                founder["arms"][arm] = run_random(frozen)
                continue
            brain = brain_for(arm, protocol, seed)
            arm_dir = directory / arm
            arm_dir.mkdir()
            brain.save(arm_dir / "initial.npz")
            training = train_arm(arm, brain, frozen, protocol, began, arm_dir)
            brain.save(arm_dir / "trained.npz")
            evaluation = (
                evaluate_arm(arm, brain, frozen, protocol, began, arm_dir)
                if training["completed"]
                else {
                    "completed": False,
                    "trials": [],
                    "seams": {},
                    "timing": None,
                    "work": Work().summary(),
                }
            )
            founder["arms"][arm] = {"training": training, **evaluation}
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": arm,
                        "trained": training["completed"],
                        "evaluated": evaluation["completed"],
                        "seconds": round(time.time() - began, 1),
                    }
                ),
                flush=True,
            )
    except Capped as error:
        founder["capped"] = True
        founder["cap_reason"] = str(error)
    test_repeats = int(repeats["test"]) if repeats else 24
    founder["scores"] = {
        arm: score_arm(founder["arms"][arm], test_repeats)
        for arm in founder["arms"]
        if "trials" in founder["arms"][arm]
    }
    for arm in protocol["arms"]:
        founder["arms"].setdefault(
            arm, {"completed": False, "trials": [], "work": Work().summary()}
        )
        founder["scores"].setdefault(arm, score_arm(founder["arms"][arm], test_repeats))
    founder["gates"] = gates_for(founder, protocol)
    founder["seconds"] = round(time.time() - began, 1)
    # the initial arrays of the vanished and history brains are byte-equal
    founder["history_initial_equal"] = (
        same_arrays(directory / "vanished" / "initial.npz", directory / "history" / "initial.npz")
        if (directory / "history" / "initial.npz").exists()
        and (directory / "vanished" / "initial.npz").exists()
        else None
    )
    return founder


def freeze_small(path: Path, seed: int, repeats: dict) -> dict[str, np.ndarray]:
    """Development sizes: fewer repeats per condition, same construction and seeding."""
    arrays = {}
    for phase, count, conditions, salt in (
        ("train", int(repeats["train"]), inputs.TRAIN_CONDITIONS, 0),
        ("test", int(repeats["test"]), inputs.TEST_CONDITIONS, 1),
    ):
        rng = np.random.default_rng(np.random.SeedSequence([seed, salt]))
        order = np.tile(np.arange(len(conditions)), count)
        if phase == "train":
            rng.shuffle(order)
        arrays[phase + "/condition"] = order
        for index, selected in enumerate(order):
            episode = inputs.make_episode(rng, conditions[selected])
            for name in episode.__dataclass_fields__:
                arrays[f"{phase}/{index:04d}/{name}"] = getattr(episode, name)
    np.savez_compressed(path, **arrays)
    for array in arrays.values():
        array.flags.writeable = False
    return arrays


# -- custody


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
            for founder in body["founders"]:
                if gates_for(founder, body["protocol"]) != founder["gates"]:
                    return "the stored gates do not follow from the scores"
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
    parser.add_argument(
        "--repeats",
        type=int,
        nargs=2,
        metavar=("TRAIN", "TEST"),
        default=None,
        help="development sizes; marks the receipt not frozen",
    )
    args = parser.parse_args(argv)
    if args.verify is not None:
        valid, reason = verify(args.verify)
        print(json.dumps({"valid": valid, "reason": reason}))
        return 0 if valid else 1
    protocol, protocol_sha = load_protocol(args.protocol)
    seeds = list(protocol["seeds"]["founders"]) if args.seeds is None else list(args.seeds)
    repeats = None if args.repeats is None else {"train": args.repeats[0], "test": args.repeats[1]}
    if (
        len(seeds) != len(set(seeds))
        or any(s < 0 for s in seeds)
        or (repeats and min(repeats.values()) < 1)
    ):
        parser.error("invalid seeds or repeats")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_bytes(Path(args.protocol).read_bytes())
    began = time.perf_counter()
    declaration = {
        "schema": SCHEMA,
        "protocol_sha256": protocol_sha,
        "frozen_protocol": repeats is None and args.seeds is None,
        "repeats": repeats,
        "seeds": seeds,
        "python": sys.version,
        "platform": platform.platform(),
        "cadence_version": cadence.__version__,
        "cadence_import": str(Path(cadence.__file__).resolve()),
        "numpy": np.__version__,
        "float64_eps": float(np.finfo(np.float64).eps),
        "longdouble_eps": float(np.finfo(np.longdouble).eps),
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    (args.out / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    sources, origins = [], []
    package = Path(cadence.__file__).resolve().parent
    own = [Path(__file__).resolve(), Path(inputs.__file__).resolve()]
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
    founders = []
    for seed in seeds:
        founder = run_founder(seed, protocol, args.out / f"founder-{seed}", repeats)
        founders.append(founder)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "horizon": founder["gates"]["horizon"],
                    "nuisance": founder["gates"]["nuisance"],
                    "closure": founder["gates"]["closure"],
                    "capped": founder["capped"],
                    "seconds": founder["seconds"],
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
        "founders": founders,
        "seconds": time.perf_counter() - began,
        "artifacts": artifacts,
        "closure": all(f["gates"]["closure"] for f in founders) and bool(founders),
        "work_scope": (
            "Every teacher and free solve, fork, seam, imagined and refused phase and checkpoint "
            "is charged per arm; calls_seconds exclude checkpoint IO; sweeps are not joules."
        ),
    }
    Receipt.build(SCHEMA, summary, sources).write(args.out / "summary.json")
    valid, reason = verify(args.out)
    assert valid, reason
    print(
        json.dumps(
            {
                "verified": valid,
                "closure": summary["closure"],
                "summary": str(args.out / "summary.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
