"""The odour nursery: one continuing life through acquisition, reversal and return.

An animal meets one of four odours per trial and avoids (0) or approaches (1). Approaching
the sugar odour pays +1, approaching another odour costs 1, avoiding pays nothing. Two
odours are a reversal pair: the sugar sits at odour 0 (rule A), moves to odour 1 (rule
B) and returns (rule A again), unannounced. The other two are a stable pair, the
unrelated skill: odour 2 is always sugar, odour 3 never. This is the two-odour T-maze
of the reversal reports behind
[issue 88](https://github.com/muellerberndt/cadence/issues/88), with the stable pair added.

The life is one stream without resets. What the reports found is reproduced by the
``step`` arm: the longer rule A is lived, the harder the reversal, because the choice
that must change is the one the learner has stopped sampling. The ``live`` arm is the
same brain with arousal (``Brain.live``): routine while outcomes match its forecast,
exploring and learning when an outcome surprises it or life pays less than it used to.

Arms, all on the same odour sequence per seed:

- ``live``     ``Brain.compose`` with ``ArousalConfig`` at the protocol's operating point;
- ``step``     the same brain without arousal, ``step`` every moment (the simpler control);
- ``defaults`` ``Brain.compose`` defaults through ``step``;
- ``memory-only`` and ``graph-only``  the ``live`` brain with its actor's rates at zero,
  and without its associative memory: which part carries the adaptation;
- ``frozen``   the ``live`` brain after rule A, answering greedily without outcomes;
- ``replay``   the ``live`` brain re-living rule A after the change: it is paid what its
  action earned under rule A, the same work on old evidence;
- ``reset``    a newborn ``live`` brain at every rule change;
- ``tabular``  epsilon-greedy tabular Q-learning, the matched-information conventional
  online learner, its two settings selected on the development seeds;
- ``random``   uniform random actions.

Readings per rule: the share of optimal executed actions in the last 100 trials; the lag
(first trial from which the next 40 executed actions are at least 90% optimal); the
greedy choice and the policy's approach probability per odour of a saved and reloaded
copy every 25 trials; how often each odour was met and approached; the first executed
approach at the newly rewarded odour, the approaches executed there before the greedy
choice turned, and the trials between (too few contradicting witnesses, or a failure to
revise after them); whether the stable pair stayed right; the share of aroused moments
and the settling sweeps per moment and mode.

A run writes a ``cadence.Receipt`` bound to this file and every module of the library;
``--verify`` checks one. Nothing here is a claim about a body: the world is a table. Run
``python benchmarks/reversal/odour_nursery.py --help``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import warnings
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd
from cadence.receipts import Receipt, canonical_json, source_manifest

SCHEMA = "odour-nursery/2"
AVOID, APPROACH = 0, 1
ODOURS = 4
STABLE_SUGAR, STABLE_PLAIN = 2, 3
ARMS = (
    "live",
    "step",
    "defaults",
    "memory-only",
    "graph-only",
    "frozen",
    "replay",
    "reset",
    "tabular",
    "random",
)
PROTOCOL = Path(__file__).with_name("protocol.json")


def reward_of(odour: int, action: int, sugar: int, payoff: str = "sugar") -> float:
    """The outcome of an action. ``sugar`` is the nursery's payoff. ``cost`` is the same
    task as a body feels it: nothing for the right action and -1 for the wrong one, so a
    competent routine is paid nothing at all."""
    if payoff == "cost":
        return 0.0 if action == optimal(odour, sugar) else -1.0
    if action == AVOID:
        return 0.0
    if odour >= 2:
        return 1.0 if odour == STABLE_SUGAR else -1.0
    return 1.0 if odour == sugar else -1.0


def optimal(odour: int, sugar: int) -> int:
    if odour >= 2:
        return APPROACH if odour == STABLE_SUGAR else AVOID
    return APPROACH if odour == sugar else AVOID


# ------------------------------------------------------------------------------- learners


def make_brain(point: dict[str, Any], seed: int, genes: dict[str, Any] | None) -> cd.Brain:
    """A composed brain at an operating point; ``genes`` adds arousal, None leaves it out."""
    options: dict[str, Any] = {}
    if "trace_amplitude" in point:
        options["working_memory_amplitude"] = point["trace_amplitude"]
    if "trace_decay" in point:
        options["working_memory_decay"] = point["trace_decay"]
    if "consolidation" in point:
        options["consolidation"] = point["consolidation"]
    if not point.get("episodic", True):
        options["episodic"] = False
    if genes is not None:
        options["arousal"] = cd.ArousalConfig(**genes)
    brain = cd.Brain.compose(
        ODOURS, 2, modules=tuple(point.get("modules", (32,))), seed=seed, **options
    )
    if "actor_eta" in point:
        eta = float(point["actor_eta"])
        brain.basal_ganglia.config = replace(
            brain.basal_ganglia.config, eta=eta, eta_bias=eta / 10.0
        )
    return brain


def frozen_choices(brain: cd.Brain) -> tuple[list[int], list[float]]:
    """The greedy choice and the policy's approach probability per odour, each read on its
    own saved and reloaded copy; the live brain is never read for a measurement."""
    choices, approach = [], []
    with tempfile.TemporaryDirectory() as directory:
        path = brain.save(os.path.join(directory, "brain.npz"))
        for odour in range(ODOURS):
            copy = cd.Brain.load(path)
            choices.append(int(copy.act(np.eye(ODOURS)[[odour]], greedy=True)[0]))
            state = copy.basal_ganglia.state
            assert state is not None
            approach.append(float(copy.basal_ganglia.probabilities(state)[0, APPROACH]))
    return choices, approach


class BrainLife:
    """A composed brain living through ``live`` (with arousal) or ``step`` (without)."""

    def __init__(self, brain: cd.Brain, *, use_live: bool) -> None:
        self.brain = brain
        self.use_live = use_live
        self.frozen = False
        self.work = {"routine": 0, "aroused": 0, "sweeps_routine": 0, "sweeps_aroused": 0}
        self.learning_sweeps = 0

    def act(self, odour: int, reward: float | None) -> tuple[int, bool]:
        x = np.eye(ODOURS)[[odour]]
        brain = self.brain
        if self.frozen:
            return int(brain.act(x, greedy=True)[0]), False
        feedback = {} if reward is None else {"reward": [reward]}
        if self.use_live:
            action = int(brain.live(x, **feedback)[0])
            reading = brain.last_arousal
            assert reading is not None
            aroused = reading["mode"] == "aroused"
            mode = "aroused" if aroused else "routine"
            self.work[mode] += 1
            self.work["sweeps_" + mode] += int(reading["sweeps"])
            self.learning_sweeps += int(reading["learning_sweeps"])
            return action, aroused
        action = int(brain.step(x, **feedback)[0])
        settlement = brain.last_settlement
        assert settlement is not None
        self.work["aroused"] += 1
        self.work["sweeps_aroused"] += int(settlement["steps"])
        pending = brain.basal_ganglia._pending
        if pending is not None:
            self.learning_sweeps += int(pending[1].steps) + int(pending[2].steps)
        self.learning_sweeps += int(brain.last_learning.get("free_steps", 0))
        return action, True

    def probe(self) -> tuple[list[int], list[float]]:
        return frozen_choices(self.brain)


class Tabular:
    """Epsilon-greedy Q-learning over the odour table: the conventional online learner
    with the same information (the odour, its own action, the reward)."""

    def __init__(self, seed: int, *, alpha: float, epsilon: float) -> None:
        self.q = np.zeros((ODOURS, 2))
        self.alpha, self.epsilon = alpha, epsilon
        self.rng = np.random.default_rng(seed)
        self.last: tuple[int, int] | None = None
        self.frozen = False

    def act(self, odour: int, reward: float | None) -> tuple[int, bool]:
        if reward is not None and self.last is not None and not self.frozen:
            o, a = self.last
            self.q[o, a] += self.alpha * (reward - self.q[o, a])
        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(2))
        else:
            action = int(np.argmax(self.q[odour]))
        self.last = (odour, action)
        return action, True

    def probe(self) -> tuple[list[int], list[float]]:
        choices = [int(np.argmax(self.q[o])) for o in range(ODOURS)]
        return choices, [
            (1 - self.epsilon / 2) if c == APPROACH else self.epsilon / 2 for c in choices
        ]


class Random:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.frozen = False

    def act(self, odour: int, reward: float | None) -> tuple[int, bool]:
        return int(self.rng.integers(2)), True

    def probe(self) -> tuple[list[int], list[float]]:
        return [-1] * ODOURS, [0.5] * ODOURS


def make_life(arm: str, protocol: dict[str, Any], seed: int, genes: dict[str, Any]) -> Any:
    point = protocol["operating_point"]
    if arm in ("live", "frozen", "replay", "reset"):
        return BrainLife(make_brain(point, seed, genes), use_live=True)
    if arm == "memory-only":  # the actor's synapses and biases do not learn
        return BrainLife(make_brain({**point, "actor_eta": 0.0}, seed, genes), use_live=True)
    if arm == "graph-only":  # no associative memory: the graph's reward learning alone
        return BrainLife(make_brain({**point, "episodic": False}, seed, genes), use_live=True)
    if arm == "step":
        return BrainLife(make_brain(point, seed, None), use_live=False)
    if arm == "defaults":
        return BrainLife(
            make_brain({"modules": point.get("modules", (32,))}, seed, None), use_live=False
        )
    if arm == "tabular":
        table = protocol["tabular"]
        return Tabular(seed, alpha=table["alpha"], epsilon=table["epsilon"])
    if arm == "random":
        return Random(seed)
    raise ValueError(f"unknown arm {arm!r}")


# --------------------------------------------------------------------------------- one life


def run_life(
    arm: str,
    seed: int,
    exposure: int,
    protocol: dict[str, Any],
    genes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One arm, one seed, one pre-switch exposure: rule A for ``exposure`` trials, rule B,
    then rule A again. Returns the readings per rule and the life's work."""
    warnings.simplefilter("ignore")
    genes = {**protocol["arousal"], **(genes or {})}
    every, hold, floor = protocol["probe_every"], protocol["hold"], protocol["hold_floor"]
    rules = [(0, exposure), (1, protocol["after"]), (0, protocol["after"])]
    odours = np.random.default_rng(protocol["odour_seed"] + seed)
    # an unreliable world withholds the outcome of some approaches (they pay nothing)
    reliability = float(protocol.get("reliability", 1.0))
    payoff = protocol.get("payoff", "sugar")
    jitter = float(protocol.get("jitter", 0.0))  # measurement noise added to every reward
    luck = np.random.default_rng(protocol["odour_seed"] + 7919 + seed)
    life = make_life(arm, protocol, seed, genes)
    started = time.time()
    odour = int(odours.integers(ODOURS))
    action, aroused = life.act(odour, None)
    phases: list[dict[str, Any]] = []
    for index, (sugar, length) in enumerate(rules):
        if index and arm == "reset":
            # a newborn's seed meets no other generator of this life (odours, luck, brains)
            life = make_life(arm, protocol, 100_000 * index + seed, genes)
            action, aroused = life.act(odour, None)
        if index == 1 and arm == "frozen":
            life.frozen = True
        new_sugar = sugar if index else None
        hits, modes = [], []
        probes: list[tuple[int, list[int], list[float]]] = []
        visits, approaches = [0] * ODOURS, [0] * ODOURS
        tries: list[int] = []  # the trials of the executed approaches at the new sugar odour
        turned = None  # the approach whose outcome first left the greedy choice there turned
        for trial in range(length):
            hits.append(int(action == optimal(odour, sugar)))
            modes.append(aroused)
            visits[odour] += 1
            approaches[odour] += int(action == APPROACH)
            witnessed = odour == new_sugar and action == APPROACH
            if witnessed:
                tries.append(trial)
            if trial % every == 0:
                probes.append((trial, *life.probe()))
            # the replay arm keeps receiving the outcomes of rule A: no new evidence
            reward = reward_of(odour, action, 0 if arm == "replay" else sugar, payoff)
            if reliability < 1.0 and luck.random() >= reliability:
                reward = 0.0
            if jitter > 0.0:
                reward += float(luck.normal(0.0, jitter))
            odour = int(odours.integers(ODOURS))
            # A refused answer raises and the life is recorded as crashed: this small brain
            # is expected to settle every moment within its budget.
            action, aroused = life.act(odour, reward)
            # The outcome of an approach at the new sugar odour is taken: has the greedy
            # choice there turned? Read on a copy, for the first approaches of the rule.
            if witnessed and turned is None and len(tries) <= protocol["witness_probes"]:
                if life.probe()[0][new_sugar] == APPROACH:
                    turned = trial
        executed = np.asarray(hits)
        lag = next(
            (k for k in range(0, length - hold + 1) if executed[k : k + hold].mean() >= floor),
            None,
        )
        right = [all(c == optimal(o, sugar) for o, c in enumerate(p[1])) for p in probes]
        greedy_lag = next(
            (probes[k][0] for k in range(len(probes) - 1) if right[k] and right[k + 1]), None
        )
        # the greedy choice at the new sugar odour: first seen turned at a probe (`flip`),
        # and the approaches executed there until it turned (`witnesses`)
        flip = witnesses = None
        if new_sugar is not None:
            flip = next((p[0] for p in probes if p[1][new_sugar] == APPROACH), None)
            if turned is not None and (flip is None or turned < flip):
                witnesses = sum(t <= turned for t in tries)
            elif flip is not None:
                witnesses = sum(t < flip for t in tries)
        stable = [
            p[1][STABLE_SUGAR] == APPROACH and p[1][STABLE_PLAIN] == AVOID
            for p in probes
            if index or p[0] >= protocol["stable_after"]
        ]
        phases.append(
            {
                "sugar": sugar,
                "length": length,
                "final": float(executed[-min(100, length) :].mean()),
                "whole": float(executed.mean()),
                "lag": lag,
                "greedy_lag": greedy_lag,
                "first_try": tries[0] if tries else None,
                "flip": flip,
                "turned": turned,
                "witnesses": witnesses,
                "visits": visits,
                "approaches": approaches,
                "start_approach": [round(v, 4) for v in probes[0][2]],
                "stable": float(np.mean(stable)) if stable else None,
                "aroused": float(np.mean(modes)),
                "aroused_late": float(np.mean(modes[length // 2 :])),
                "end_greedy": probes[-1][1],
                "end_approach": [round(v, 4) for v in probes[-1][2]],
            }
        )
    result: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "exposure": exposure,
        "phases": phases,
        "seconds": round(time.time() - started, 2),
    }
    if isinstance(life, BrainLife):
        work = life.work
        result["work"] = {
            **work,
            "learning_sweeps": life.learning_sweeps,
            "sweeps_per_routine_moment": work["sweeps_routine"] / max(1, work["routine"]),
            "sweeps_per_aroused_moment": work["sweeps_aroused"] / max(1, work["aroused"]),
        }
    return result


def _job(args: tuple[str, int, int, dict[str, Any], dict[str, Any] | None]) -> dict[str, Any]:
    arm, seed, exposure, protocol, genes = args
    try:
        return run_life(arm, seed, exposure, protocol, genes)
    except Exception as error:  # a crashed life is a recorded outcome, not a missing row
        return {
            "arm": arm,
            "seed": seed,
            "exposure": exposure,
            "error": f"{type(error).__name__}: {error}"[:300],
        }


def run(
    protocol: dict[str, Any],
    *,
    arms: list[str],
    seeds: list[int],
    exposures: list[int],
    genes: dict[str, Any] | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    jobs = [
        (arm, seed, exposure, protocol, genes)
        for arm in arms
        for exposure in exposures
        for seed in seeds
    ]
    if workers <= 1:
        return [_job(job) for job in jobs]
    with get_context("spawn").Pool(workers) as pool:
        return pool.map(_job, jobs, chunksize=1)


# ----------------------------------------------------------------------------------- report


def gates(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    """The protocol's fixed gates over the ``live`` rows. Per exposure: the share of lives
    that end each rule at or above the floor, keep the stable pair and return to routine.
    ``pooled`` joins the exposures of 300 and more, and ``passed`` applies the protocol's
    share to it and the lag bound to each median reversal lag; a crashed life passes
    nothing."""
    g = protocol["gates"]
    tests = {
        "acquired": lambda r: r["phases"][0]["final"] >= g["final"],
        "reversed": lambda r: r["phases"][1]["final"] >= g["final"],
        "returned": lambda r: r["phases"][2]["final"] >= g["final"],
        "stable_kept": lambda r: all((p["stable"] or 0.0) >= g["stable"] for p in r["phases"]),
        "calm": lambda r: all(p["aroused_late"] <= g["aroused_late"] for p in r["phases"]),
    }

    def shares(lives: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"lives": len(lives)}
        out["crashed"] = sum("error" in r for r in lives)
        for name, test in tests.items():
            out[name] = float(np.mean(["error" not in r and bool(test(r)) for r in lives]))
        return out

    def median_lag(exposure: int) -> float | None:
        lags = [
            r["phases"][1]["lag"] for r in live if r["exposure"] == exposure and "error" not in r
        ]
        if not lags:
            return None
        # a life that never reversed lies beyond any bound
        return float(np.median([np.inf if v is None else v for v in lags]))

    live = [r for r in rows if r["arm"] == "live"]
    report: dict[str, Any] = {}
    for exposure in sorted({r["exposure"] for r in live}):
        report[str(exposure)] = shares([r for r in live if r["exposure"] == exposure])
    gated = [r for r in live if r["exposure"] >= 300]
    if gated:
        pooled = shares(gated)
        lags = {str(e): median_lag(e) for e in sorted({r["exposure"] for r in gated})}
        report["pooled"] = pooled
        report["reversal_lag"] = {
            k: (None if v is None or np.isinf(v) else v) for k, v in lags.items()
        }
        report["passed"] = bool(
            pooled["crashed"] == 0
            and all(pooled[name] >= g["share"] for name in tests)
            and all(v is not None and v <= g["lag_bound"] for v in lags.values())
        )
    return report


def summarize(rows: list[dict[str, Any]]) -> str:
    lines = []
    keys = sorted({(r["arm"], r["exposure"]) for r in rows}, key=lambda k: (ARMS.index(k[0]), k[1]))
    for arm, exposure in keys:
        group = [r for r in rows if r["arm"] == arm and r["exposure"] == exposure]
        good = [r for r in group if "error" not in r]
        if not good:
            lines.append(
                f"{arm:9s} {exposure:6d}  all {len(group)} lives crashed: {group[0]['error']}"
            )
            continue
        parts = []
        for index, name in enumerate(("A", "B", "A again")):
            finals = [r["phases"][index]["final"] for r in good]
            lags = [r["phases"][index]["lag"] for r in good]
            found = [v for v in lags if v is not None]
            median = f"{int(np.median(found)):4d}" if found else "   -"
            parts.append(
                f"{name}: {np.mean(finals):.2f} (min {min(finals):.2f}) lag {median} "
                f"[{len(found)}/{len(lags)}]"
            )
        stable = [p["stable"] for r in good for p in r["phases"] if p["stable"] is not None]
        aroused = np.mean([p["aroused"] for r in good for p in r["phases"]])
        tail = f"stable {min(stable):.2f} aroused {aroused:.2f}" if stable else ""
        crashed = f" crashed {len(group) - len(good)}" if len(good) < len(group) else ""
        lines.append(f"{arm:11s} {exposure:6d}  " + " | ".join(parts) + f" | {tail}{crashed}")
    return "\n".join(lines)


def markdown(report: dict[str, Any]) -> str:
    """The receipt's tables as Markdown, for the README: every number is read from the rows."""
    rows = report["rows"]
    exposures = sorted({r["exposure"] for r in rows})
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    # the first freeze's receipt predates the coverage and witness readings
    witnessed = all("witnesses" in r["phases"][1] for r in rows if "error" not in r)

    def lives(arm: str, exposure: int) -> list[dict[str, Any]]:
        return [
            r for r in rows if r["arm"] == arm and r["exposure"] == exposure and "error" not in r
        ]

    def table(title: str, cell: Any) -> list[str]:
        head = "| Arm | " + " | ".join(f"{e:,}" for e in exposures) + " |"
        lines = [f"### {title}", "", head, "| --- |" + " --- |" * len(exposures)]
        for arm in arms:
            cells = [cell(lives(arm, e)) if lives(arm, e) else "crashed" for e in exposures]
            lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
        return lines + [""]

    def final(index: int) -> Any:
        def cell(group: list[dict[str, Any]]) -> str:
            values = [r["phases"][index]["final"] for r in group]
            return f"{np.mean(values):.2f} ({min(values):.2f})"

        return cell

    def lag(index: int, reading: str = "lag") -> Any:
        def cell(group: list[dict[str, Any]]) -> str:
            values = [r["phases"][index][reading] for r in group]
            found = [v for v in values if v is not None]
            median = f"{int(np.median(found))}" if found else "none"
            return f"{median} ({len(found)}/{len(values)})"

        return cell

    def start(group: list[dict[str, Any]]) -> str:
        values = [r["phases"][1]["start_approach"][1] for r in group]
        return f"{np.median(values):.3f} ({min(values):.3f})"

    def unturned(group: list[dict[str, Any]]) -> str:
        left = [p for p in (r["phases"][1] for r in group) if p["witnesses"] is None]
        median = (
            f"{int(np.median([p['approaches'][p['sugar']] for p in left]))}" if left else "none"
        )
        return f"{median} ({len(left)}/{len(group)})"

    def stable(group: list[dict[str, Any]]) -> str:
        values = [p["stable"] for r in group for p in r["phases"] if p["stable"] is not None]
        return f"{np.mean(values):.2f} ({min(values):.2f})"

    out = [
        *table("Rule A: optimal share of the last 100 actions, mean (minimum)", final(0)),
        *table("Rule B: optimal share of the last 100 actions, mean (minimum)", final(1)),
        *table("Rule A again: optimal share of the last 100 actions, mean (minimum)", final(2)),
        *table("Reversal lag in trials, median (lives that reversed / lives)", lag(1)),
        *table("Return lag in trials, median (lives that returned / lives)", lag(2)),
        *table(
            "Greedy choices after the reversal: first of two consecutive probes with every "
            "odour right, median trial (lives / lives)",
            lag(1, "greedy_lag"),
        ),
        *table(
            "First executed approach at the new sugar odour after the reversal, median trial "
            "(lives / lives)",
            lag(1, "first_try"),
        ),
    ]
    if witnessed:
        out += [
            *table(
                "Policy's approach probability at the new sugar odour when the rule turns, "
                "median (minimum)",
                start,
            ),
            *table(
                "Approaches executed there until the greedy choice turned, median "
                "(lives whose choice turned / lives)",
                lag(1, "witnesses"),
            ),
            *table(
                "Lives whose greedy choice there never turned: approaches executed there "
                "under rule B, median (lives / lives)",
                unturned,
            ),
        ]
    out += table("Stable pair right at the probes, mean (minimum)", stable)
    head = (
        "| Exposure | first approach at the new sugar odour | from it to the turned greedy "
        "choice | "
        "aroused, whole life | aroused, second half of rule A | sweeps per routine moment | "
        "sweeps per aroused moment | learning sweeps per aroused moment |"
    )
    out += ["### The live arm: witnesses, arousal and work, medians over lives", "", head]
    out.append("| --- |" + " --- |" * 7)
    for exposure in exposures:
        group = lives("live", exposure)
        if not group:
            continue
        tries = [r["phases"][1]["first_try"] for r in group]
        flips = [
            min(v for v in (p.get("turned"), p["flip"]) if v is not None) - p["first_try"]
            for p in (r["phases"][1] for r in group)
            if p["flip"] is not None and p["first_try"] is not None
        ]
        tried = [v for v in tries if v is not None]
        work = [r["work"] for r in group]
        moments = [w["routine"] + w["aroused"] for w in work]
        out.append(
            f"| {exposure:,} | {int(np.median(tried)) if tried else 'none'} "
            f"({len(tried)}/{len(tries)}) | "
            f"{int(np.median(flips)) if flips else 'none'} ({len(flips)}/{len(tries)}) | "
            f"{np.median([w['aroused'] / m for w, m in zip(work, moments, strict=True)]):.3f} | "
            f"{np.median([r['phases'][0]['aroused_late'] for r in group]):.3f} | "
            f"{np.median([w['sweeps_per_routine_moment'] for w in work]):.1f} | "
            f"{np.median([w['sweeps_per_aroused_moment'] for w in work]):.1f} | "
            f"{np.median([w['learning_sweeps'] / max(1, w['aroused']) for w in work]):.1f} |"
        )
    out += ["", "### Gates", "", "```json", json.dumps(report["gates"], indent=1), "```"]
    return "\n".join(out)


def sources() -> list[tuple[str, Path]]:
    """The code a run depends on: this chamber and every module of the library."""
    package = Path(cd.__file__).resolve().parent
    return [
        ("odour_nursery.py", Path(__file__).resolve()),
        *(
            ("cadence/" + path.relative_to(package).as_posix(), path)
            for path in sorted(package.rglob("*.py"))
        ),
    ]


def read_receipt(path: Path) -> dict[str, Any]:
    """The body of a receipt from ``.json`` or ``.json.gz``. The receipt of the first
    confirmation run predates the source manifest and is its own body."""
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    stored: dict[str, Any] = json.loads(data)
    body: dict[str, Any] = stored["body"] if "body" in stored else stored
    return body


def write_receipt(path: Path, body: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Write a ``cadence.Receipt`` bound to the sources read when the run began; a ``.gz``
    name is compressed with a fixed header, so equal receipts give equal bytes."""
    receipt = Receipt.build(SCHEMA, body, sources())
    if receipt.source != manifest:
        raise RuntimeError("a source file changed during the run; discard it and run again")
    data = (canonical_json(receipt.to_dict()) + "\n").encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    path.write_bytes(data)


def planned(body: dict[str, Any]) -> list[tuple[str, int, int]]:
    """The lives a receipt declares, in the order they are run."""
    return [(a, e, s) for a in body["arms"] for e in body["exposures"] for s in body["seeds"]]


def verify(path: Path, *, current: bool = False, protocol: Path = PROTOCOL) -> tuple[bool, str]:
    """Check a receipt: canonical form and digest, one row for every planned life, and
    the gates recomputed from the rows. ``current`` also requires the source manifest and
    the protocol hash to be those of the files present now."""

    def check(body: dict[str, Any]) -> str | None:
        rows = body["rows"]
        if [(r["arm"], r["exposure"], r["seed"]) for r in rows] != planned(body):
            return "the rows are not the planned lives, each once and in order"
        if gates(rows, body["protocol"]) != body["gates"]:
            return "the stored gates do not follow from the rows"
        if current and body["protocol_sha256"] != hashlib.sha256(protocol.read_bytes()).hexdigest():
            return "the protocol file differs from the recorded hash"
        return None

    try:
        data = path.read_bytes()
        if path.suffix == ".gz":
            data = gzip.decompress(data)
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "receipt.json"
            plain.write_bytes(data)
            return Receipt.verify(plain, sources=sources() if current else None, check=check)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, f"cannot verify the receipt: {error}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS)
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=["development"],
        help="seed numbers, or the names of the protocol's seed sets",
    )
    parser.add_argument("--exposures", nargs="*", type=int, default=None)
    parser.add_argument("--genes", default=None, help="JSON overrides of the arousal genes")
    parser.add_argument(
        "--point",
        default=None,
        help="JSON overrides of the operating point; null leaves a setting at its default",
    )
    parser.add_argument("--reliability", type=float, default=None, help="override the world's")
    parser.add_argument("--payoff", choices=("sugar", "cost"), default=None)
    parser.add_argument("--jitter", type=float, default=None, help="reward noise, one sigma")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="print a receipt's tables")
    parser.add_argument("--verify", type=Path, default=None, help="check a receipt")
    parser.add_argument(
        "--current",
        action="store_true",
        help="with --verify: the receipt must also match the source files present now",
    )
    args = parser.parse_args(argv)
    if args.report is not None:
        print(markdown(read_receipt(args.report)))
        return 0
    if args.verify is not None:
        valid, reason = verify(args.verify, current=args.current)
        print(json.dumps({"verified": valid, "reason": reason}))
        return 0 if valid else 1
    manifest = source_manifest(sources())
    frozen = args.protocol.read_bytes()
    protocol = json.loads(frozen)
    if protocol.get("schema") != SCHEMA:
        parser.error(f"the protocol's schema is not {SCHEMA}")
    if args.reliability is not None:
        protocol["reliability"] = args.reliability
    if args.payoff is not None:
        protocol["payoff"] = args.payoff
    if args.jitter is not None:
        protocol["jitter"] = args.jitter
    if args.point:
        point = {**protocol["operating_point"], **json.loads(args.point)}
        # a null removes a setting: that part of the brain stays at its released default
        protocol["operating_point"] = {k: v for k, v in point.items() if v is not None}
    seeds: list[int] = []
    for value in args.seeds:
        seeds.extend(protocol["seeds"][value] if value in protocol["seeds"] else [int(value)])
    exposures = args.exposures or protocol["exposures"]
    genes = json.loads(args.genes) if args.genes else None
    rows = run(
        protocol,
        arms=args.arms,
        seeds=seeds,
        exposures=exposures,
        genes=genes,
        workers=args.workers,
    )
    print(summarize(rows))
    overridden = (
        args.genes
        or args.point
        or args.reliability is not None
        or args.payoff
        or args.jitter is not None
    )
    body = {
        "cadence": cd.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "protocol_sha256": hashlib.sha256(frozen).hexdigest(),
        # the settings are the frozen file's: no override, and the file is this chamber's
        "frozen_protocol": not overridden and frozen == PROTOCOL.read_bytes(),
        "protocol": protocol,
        "genes_override": genes,
        "arms": list(args.arms),
        "exposures": list(exposures),
        "seeds": seeds,
        "gates": gates(rows, protocol),
        "rows": rows,
    }
    print(json.dumps(body["gates"], indent=1))
    if args.out is not None:
        write_receipt(args.out, body, manifest)
        valid, reason = verify(args.out, current=True, protocol=args.protocol)
        print(json.dumps({"verified": valid, "reason": reason, "receipt": str(args.out)}))
        return 0 if valid else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
