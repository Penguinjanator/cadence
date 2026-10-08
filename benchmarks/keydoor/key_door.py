"""The key-door nursery: delayed credit in one continuing life.

A creature walks a corridor once per episode: empty floor, a chest, a lamp, ``D`` levers
and a door, met in that order; every trip has the same number of cells, so a longer delay
means fewer floor cells before the chest. At every cell it passes or interacts. Under rule
A the chest holds the key; under rule B the lamp does. Interacting at the door with the key
in the pouch pays +1 and ends the episode; interacting with a chest, lamp or lever that
supplies no key costs ``cost``; floor interactions, empty-door interactions and passing
pay nothing. Taking the key pays nothing by itself: its worth
arrives ``D + 2`` cells later from the chest or ``D + 1`` from the lamp, after irrelevant
choices at the levers, so credit cannot
follow the last action blindly. The creature sees the kind of cell it faces and, with the
pouch sense, whether it holds the key. The number of levers varies by one from episode to
episode: the delay varies in decision counts, with no physical-time interface. The outcome
of the door is delivered with the first
observation of the next episode, ``done`` set, as ``step`` and ``live`` define it. A share
``truncation`` of the trips is cut short before the door, the key lost with them; such a
trip ends with ``done`` clear, so the forecast carries over into the next trip (a truncated
bootstrap), where the door's end is terminal. The reversal chamber of issue 88 is this
chamber's model; this is the delayed key-door reward nursery of
[issue 111](https://github.com/muellerberndt/cadence/issues/111), roadmap row 07.

Arms, all on the same corridor sequence per seed:

- ``live``        ``Brain.compose`` with ``ArousalConfig`` at the protocol's operating point,
                  the simplest existing System 1;
- ``copy``        the ``live`` brain carrying the efference copy of its own last command at
                  the protocol's ``copy`` genes (0.76.0), a declared variant whose founder
                  value, zero, is the ``live`` arm;
- ``step``        the same brain without arousal, learning at every moment (the simpler control);
- ``lambda-zero`` the ``live`` brain with the eligibility decay ``lam`` at zero;
- ``yoked``       the ``live`` brain whose door outcome is paid at a random cell of the next
                  trip instead of at the door: its own earned rewards, with credit retimed;
- ``frozen``      the ``live`` brain after rule A, answering greedily without outcomes;
- ``blind``       the ``live`` brain without the pouch sense;
- ``recurrent``   an online recurrent actor-critic with the same information: the cell kind
                  and the pouch bit enter an Elman hidden layer carried across moments, read
                  by a softmax policy and a linear value, learned with eligibility traces;
- ``tabular``     epsilon-greedy Q(lambda) over (cell, pouch): the matched-information
                  conventional online learner, its settings selected on the development seeds;
- ``random``      uniform random actions.

``key-door/3`` lives through the protocol's ``rules`` in order, chest, lamp and the chest
again, so the return of the first contingency reads the retained skill; its lever count
varies by ``jitter`` cells from trip to trip, the irregular event time of the acceptance.

Readings per rule, over the trips that reached the door: the share of episodes that ended
with food, the share in which the key was taken, the wrong interactions per episode and the
share of door openings with the key, each over the last 50 episodes; the first 20-episode
window with 90% fed; the number of trips cut short; the greedy
choice and the probability of interacting, per cell and pouch state, of a saved and
reloaded copy every 25 episodes; the probability of interacting under the behaviour that
acted and under the base policy, per cell and pouch state, read from the living brain;
the share of aroused moments; and the work of the life. Receipts are ``cadence.Receipt``s
bound to this file and every module of the library. New receipts use ``key-door/3``;
``key-door/2`` receipts are the corrected two-rule instrument, and historical
``key-door/1`` receipts retain their original scheduling and metric limitations.
Run
``python benchmarks/keydoor/key_door.py --help``.
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
from cadence.receipts import Receipt, canonical_json, canonical_sha256, source_manifest

SCHEMA = "key-door/3"
LEGACY_SCHEMA = "key-door/1"
LEGACY_SCHEMAS = ("key-door/1", "key-door/2")
DEFAULT_RULES = ["chest", "lamp"]
PASS, INTERACT = 0, 1
FLOOR, CHEST, LAMP, LEVER, DOOR = range(5)
KINDS = ("floor", "chest", "lamp", "lever", "door")
ARMS = (
    "live",
    "copy",
    "step",
    "lambda-zero",
    "yoked",
    "frozen",
    "blind",
    "recurrent",
    "tabular",
    "random",
)
BRAINLESS = ("recurrent", "tabular", "random")
PROTOCOL = Path(__file__).with_name("protocol-3.json")
PROTOCOL_1 = Path(__file__).with_name("protocol.json")


def corridor(delay: int, length: int, jitter: int, rng: np.random.Generator) -> list[int]:
    """The cells of one trip: floor, chest, lamp, the levers and the door, ``length`` in all."""
    if delay < 0 or jitter < 0 or length < 3 + delay + jitter:
        raise ValueError("the corridor must fit the chest, lamp, door and all jittered levers")
    levers = max(0, delay + (int(rng.integers(-jitter, jitter + 1)) if jitter else 0))
    floors = max(0, length - 3 - levers)
    return [*([FLOOR] * floors), CHEST, LAMP, *([LEVER] * levers), DOOR]


def observe(kind: int, holding: bool, pouch: bool) -> np.ndarray:
    x = np.zeros((1, 6 if pouch else 5))
    x[0, kind] = 1.0
    if pouch:
        x[0, 5] = float(holding)
    return x


# ------------------------------------------------------------------------------- learners


def fresh_work() -> dict[str, int]:
    return {
        "routine": 0,
        "aroused": 0,
        "sweeps_routine": 0,
        "sweeps_aroused": 0,
        "learning_sweeps": 0,
        "probes": 0,
        "probe_sweeps": 0,
        "checkpoints": 0,
        "memory_reads": 0,
        "probe_memory_reads": 0,
        "memory_writes": 0,
        "brains": 0,
        "refused_sweeps": 0,
        "aborted_forecast_sweeps": 0,
    }


def make_brain(
    point: dict[str, Any], seed: int, genes: dict[str, Any] | None, pouch: bool
) -> cd.Brain:
    options: dict[str, Any] = {}
    if "trace_amplitude" in point:
        options["working_memory_amplitude"] = point["trace_amplitude"]
    if "trace_decay" in point:
        options["working_memory_decay"] = point["trace_decay"]
    if "consolidation" in point:
        options["consolidation"] = point["consolidation"]
    if point.get("efference_amplitude"):
        # the efference copy of the last command (0.76.0), the ``copy`` arm's genes; absent,
        # the brain is byte-identical to the key-door/2 brain
        options["efference_amplitude"] = point["efference_amplitude"]
        options["efference_decay"] = point.get("efference_decay", 0.0)
    if genes is not None:
        options["arousal"] = cd.ArousalConfig(**genes)
    brain = cd.Brain.compose(
        6 if pouch else 5, 2, modules=tuple(point.get("modules", (32,))), seed=seed, **options
    )
    names = type(brain.basal_ganglia.config).__slots__  # every actor gene the point names
    actor = {k: point[k] for k in names if k in point}
    if "eta" in actor and "eta_bias" not in actor:
        actor["eta_bias"] = actor["eta"] / 10.0  # the composed rule of issue 143
    if actor:
        brain.basal_ganglia.config = replace(brain.basal_ganglia.config, **actor)
    if point.get("learner"):  # settings of the settling learner, such as its temperature
        brain.learner.config = replace(brain.learner.config, **point["learner"])
    return brain


class BrainLife:
    """A composed brain living through ``live`` (with arousal) or ``step`` (without).
    ``last`` holds the probability of interacting under the behaviour that acted and under
    the base policy, and the probability of the executed action; ``work`` the ledger."""

    def __init__(self, brain: cd.Brain, *, use_live: bool, pouch: bool) -> None:
        self.brain = brain
        self.use_live = use_live
        self.pouch = pouch
        self.frozen = False
        self.work = fresh_work()
        self.work["brains"] = 1
        self.last = (0.5, 0.5, 1.0)
        self._count_memory(brain, "memory_reads")
        learn = brain.learn

        def counted_learn(*args: Any, **kwargs: Any) -> dict[str, float]:
            report = learn(*args, **kwargs)
            # Accepted feedback remains work even when the following answer refuses.
            self.work["learning_sweeps"] += int(report.get("free_steps", 0))
            return report

        brain.learn = counted_learn  # type: ignore[method-assign]
        forecast = brain._forecast
        self._forecast_sweeps = 0

        def counted_forecast(*args: Any, **kwargs: Any) -> Any:
            answer = forecast(*args, **kwargs)
            settlement = brain.last_settlement
            assert settlement is not None
            self._forecast_sweeps += int(settlement["steps"])
            return answer

        brain._forecast = counted_forecast  # type: ignore[method-assign]

    def _count_memory(self, brain: cd.Brain, counter: str) -> None:
        memory = brain.hippocampus
        if memory is not None:
            original = memory.recall

            def counted(*args: Any, **kwargs: Any) -> np.ndarray:
                self.work[counter] += 1
                return original(*args, **kwargs)

            memory.recall = counted  # type: ignore[method-assign]

    def _read(self, action: int, temperature: float | None) -> None:
        agent = self.brain.basal_ganglia
        state = agent.state
        assert state is not None
        policy = np.asarray(agent.probabilities(state))[0]
        if temperature is None:
            behaviour = np.zeros(2)
            behaviour[int(np.argmax(policy))] = 1.0
        else:
            behaviour = np.asarray(agent.probabilities(state, temperature))[0]
        self.last = (float(behaviour[INTERACT]), float(policy[INTERACT]), float(behaviour[action]))

    def act(self, kind: int, holding: bool, reward: float | None, done: bool) -> tuple[int, bool]:
        x = observe(kind, holding, self.pouch)
        brain = self.brain
        self._forecast_sweeps = 0
        try:
            if self.frozen:
                action = int(brain.act(x, greedy=True)[0])
                settlement = brain.last_settlement
                assert settlement is not None
                self.work["routine"] += 1
                self.work["sweeps_routine"] += int(settlement["steps"])
                self._read(action, None)
                return action, False
            feedback = {} if reward is None else {"reward": [reward], "done": [done]}
            if self.use_live:
                action = int(brain.live(x, **feedback)[0])
                reading = brain.last_arousal
                assert reading is not None
                aroused = reading["mode"] == "aroused"
                mode = "aroused" if aroused else "routine"
                self.work[mode] += 1
                self.work["sweeps_" + mode] += int(reading["sweeps"])
                self._forecast_sweeps = 0  # the successful reading already includes this work
                feedback_sweeps = int(brain.last_learning.get("free_steps", 0))
                self.work["learning_sweeps"] += int(reading["learning_sweeps"]) - feedback_sweeps
                self._read(action, reading["temperature"])
                return action, aroused
            action = int(brain.step(x, **feedback)[0])
            settlement = brain.last_settlement
            assert settlement is not None
            self.work["aroused"] += 1
            self.work["sweeps_aroused"] += int(settlement["steps"])
            pending = brain.basal_ganglia._pending
            if pending is not None:
                self.work["learning_sweeps"] += int(pending[1].steps) + int(pending[2].steps)
            self._read(action, brain.learner.config.temperature)
            return action, True
        except Exception:
            self.work["aborted_forecast_sweeps"] += self._forecast_sweeps
            self._charge_refusal(brain)
            raise
        finally:
            self._forecast_sweeps = 0

    def _charge_refusal(self, brain: cd.Brain) -> None:
        settlement = brain.last_settlement
        if settlement is not None and not settlement["qualified"]:
            self.work["refused_sweeps"] += int(settlement["steps"])

    def probe(self) -> tuple[list[list[int]], list[list[float]]]:
        """The greedy choice and the policy's probability of interacting per (pouch, cell),
        each read on its own saved and reloaded copy; the living brain is never read. A
        copy's refused answer is charged like the life's own."""
        choices: list[list[int]] = []
        interact: list[list[float]] = []
        with tempfile.TemporaryDirectory() as directory:
            path = self.brain.save(os.path.join(directory, "brain.npz"))
            self.work["checkpoints"] += 1
            for holding in (False, True):
                row_c, row_p = [], []
                for kind in range(5):
                    copy = cd.Brain.load(path)
                    self.work["checkpoints"] += 1
                    self._count_memory(copy, "probe_memory_reads")
                    try:
                        choice = copy.act(observe(kind, holding, self.pouch), greedy=True)
                    except Exception:
                        self._charge_refusal(copy)
                        raise
                    row_c.append(int(choice[0]))
                    state = copy.basal_ganglia.state
                    assert state is not None and copy.last_settlement is not None
                    row_p.append(float(copy.basal_ganglia.probabilities(state)[0, INTERACT]))
                    self.work["probe_sweeps"] += int(copy.last_settlement["steps"])
                choices.append(row_c)
                interact.append(row_p)
        self.work["probes"] += 1
        return choices, interact

    def ledger(self) -> dict[str, int]:
        memory = self.brain.hippocampus
        return {**self.work, "memory_writes": 0 if memory is None else int(memory.writes)}


class Recurrent:
    """An online recurrent actor-critic with the same information as the brain: the one-hot
    cell kind and the pouch bit enter an Elman hidden layer whose state carries across the
    moments of a trip and clears at the door, as the brain's working trace does; a softmax
    policy and a linear value read it. Learning is online at every moment, TD(lambda)
    eligibility traces over every weight with the gradient taken through the current
    moment's hidden state only, so nothing is stored or replayed. Its settings are selected
    on the development seeds like the tabular learner's. ``probe`` reads the policy from a
    cleared hidden state, so it is stateless and reported as such."""

    def __init__(
        self,
        seed: int,
        *,
        hidden: int = 16,
        alpha: float = 0.05,
        alpha_value: float = 0.1,
        gamma: float = 0.95,
        lam: float = 0.9,
        temperature: float = 1.0,
        clip: float = 5.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.alpha, self.alpha_value, self.gamma, self.lam = alpha, alpha_value, gamma, lam
        self.temperature = temperature
        self.clip = clip  # the largest norm of one moment's gradient; a declared guard
        self.clipped = 0
        self.Wx = rng.normal(0.0, 0.3, (hidden, 6))
        self.Wh = rng.normal(0.0, 0.3, (hidden, hidden)) / np.sqrt(hidden)
        self.b = np.zeros(hidden)
        self.Wa = np.zeros((2, hidden))
        self.ba = np.zeros(2)
        self.wv = np.zeros(hidden)
        self.bv = 0.0
        self.h = np.zeros(hidden)
        self.frozen = False
        self.last = (0.5, 0.5, 0.5)
        self.memo: dict[str, Any] | None = None
        self.actor_trace = {k: np.zeros_like(v) for k, v in self._actor().items()}
        self.value_trace = {k: np.zeros_like(v) for k, v in self._value().items()}

    def _actor(self) -> dict[str, np.ndarray]:
        return {"Wa": self.Wa, "ba": self.ba, "Wx": self.Wx, "Wh": self.Wh, "b": self.b}

    def _value(self) -> dict[str, np.ndarray]:
        return {
            "wv": self.wv,
            "bv": np.atleast_1d(self.bv),
            "Wx": self.Wx,
            "Wh": self.Wh,
            "b": self.b,
        }

    @staticmethod
    def _features(kind: int, holding: bool) -> np.ndarray:
        x = np.zeros(6)
        x[kind] = 1.0
        x[5] = float(holding)
        return x

    def _forward(self, x: np.ndarray, h_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        h = np.tanh(self.Wx @ x + self.Wh @ h_prev + self.b)
        logits = (self.Wa @ h + self.ba) / self.temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return h, probs, float(self.wv @ h + self.bv)

    def act(self, kind: int, holding: bool, reward: float | None, done: bool) -> tuple[int, bool]:
        x = self._features(kind, holding)
        h_prev = self.h
        h, probs, value = self._forward(x, h_prev)
        if reward is not None and self.memo is not None and not self.frozen:
            target = reward + (0.0 if done else self.gamma * value)
            delta = float(np.clip(target - self.memo["value"], -self.clip, self.clip))
            self.Wa += self.alpha * delta * self.actor_trace["Wa"]
            self.ba += self.alpha * delta * self.actor_trace["ba"]
            self.wv += self.alpha_value * delta * self.value_trace["wv"]
            self.bv += self.alpha_value * delta * float(self.value_trace["bv"][0])
            for name in ("Wx", "Wh", "b"):
                shared = getattr(self, name)
                shared += self.alpha * delta * self.actor_trace[name]
                shared += self.alpha_value * delta * self.value_trace[name]
            if done:
                for trace in (self.actor_trace, self.value_trace):
                    for v in trace.values():
                        v[:] = 0.0
                h_prev = np.zeros_like(self.h)
                h, probs, value = self._forward(x, h_prev)
        if self.frozen:
            action = int(np.argmax(probs))
        else:
            action = int(self.rng.random() < probs[INTERACT])
        # the eligibility of this decision: decayed traces plus the gradients through h,
        # each moment's gradient bounded in norm so a large step cannot run the weights away
        onehot = np.zeros(2)
        onehot[action] = 1.0
        dlogits = (onehot - probs) / self.temperature
        gz_actor = (self.Wa.T @ dlogits) * (1.0 - h * h)
        gz_value = self.wv * (1.0 - h * h)
        norm = float(
            np.sqrt(np.sum(dlogits**2) + np.sum(gz_actor**2) * (1.0 + x @ x + h_prev @ h_prev))
        )
        if norm > self.clip:
            dlogits, gz_actor = dlogits * (self.clip / norm), gz_actor * (self.clip / norm)
            self.clipped += 1
        decay = self.gamma * self.lam
        for trace in (self.actor_trace, self.value_trace):
            for v in trace.values():
                v *= decay
        self.actor_trace["Wa"] += np.outer(dlogits, h)
        self.actor_trace["ba"] += dlogits
        self.actor_trace["Wx"] += np.outer(gz_actor, x)
        self.actor_trace["Wh"] += np.outer(gz_actor, h_prev)
        self.actor_trace["b"] += gz_actor
        self.value_trace["wv"] += h
        self.value_trace["bv"] += 1.0
        self.value_trace["Wx"] += np.outer(gz_value, x)
        self.value_trace["Wh"] += np.outer(gz_value, h_prev)
        self.value_trace["b"] += gz_value
        self.h = h
        self.memo = {"value": value}
        p = float(probs[INTERACT])
        self.last = (p, p, p if action == INTERACT else 1.0 - p)
        return action, True

    def probe(self) -> tuple[list[list[int]], list[list[float]]]:
        choices, probs = [], []
        for holding in (False, True):
            row_c, row_p = [], []
            for kind in range(5):
                _, pi, _ = self._forward(self._features(kind, holding), np.zeros_like(self.h))
                row_c.append(int(np.argmax(pi)))
                row_p.append(round(float(pi[INTERACT]), 4))
            choices.append(row_c)
            probs.append(row_p)
        return choices, probs


class Tabular:
    """Epsilon-greedy Q(lambda) over (cell, pouch) with the same information as the brain:
    accumulating eligibility over the episode, decayed by gamma * lam at every step."""

    def __init__(
        self, seed: int, *, alpha: float, epsilon: float, gamma: float, lam: float
    ) -> None:
        self.q = np.zeros((2, 5, 2))  # pouch, cell, action
        self.trace = np.zeros_like(self.q)
        self.alpha, self.epsilon, self.gamma, self.lam = alpha, epsilon, gamma, lam
        self.rng = np.random.default_rng(seed)
        self.memo: tuple[int, int, int] | None = None
        self.frozen = False
        self.last = (0.5, 0.5, 0.5)

    def act(self, kind: int, holding: bool, reward: float | None, done: bool) -> tuple[int, bool]:
        h = int(holding)
        if reward is not None and self.memo is not None and not self.frozen:
            ph, pk, pa = self.memo
            target = reward + (0.0 if done else self.gamma * self.q[h, kind].max())
            self.trace *= self.gamma * self.lam
            self.trace[ph, pk, pa] += 1.0
            self.q += self.alpha * (target - self.q[ph, pk, pa]) * self.trace
            if done:
                self.trace[:] = 0.0
        greedy = int(np.argmax(self.q[h, kind]))
        action = int(self.rng.integers(2)) if self.rng.random() < self.epsilon else greedy
        self.memo = (h, kind, action)
        p = 1 - self.epsilon / 2 if greedy == INTERACT else self.epsilon / 2
        self.last = (p, p, p if action == INTERACT else 1 - p)
        return action, True

    def probe(self) -> tuple[list[list[int]], list[list[float]]]:
        choices = [[int(np.argmax(self.q[h, k])) for k in range(5)] for h in (0, 1)]
        return choices, [
            [(1 - self.epsilon / 2) if c == INTERACT else self.epsilon / 2 for c in row]
            for row in choices
        ]


class Random:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.frozen = False
        self.last = (0.5, 0.5, 0.5)

    def act(self, kind: int, holding: bool, reward: float | None, done: bool) -> tuple[int, bool]:
        return int(self.rng.integers(2)), True

    def probe(self) -> tuple[list[list[int]], list[list[float]]]:
        return [[-1] * 5, [-1] * 5], [[0.5] * 5, [0.5] * 5]


def make_life(arm: str, protocol: dict[str, Any], seed: int, genes: dict[str, Any]) -> Any:
    point = protocol["operating_point"]
    if arm in ("live", "yoked", "frozen"):
        return BrainLife(make_brain(point, seed, genes, True), use_live=True, pouch=True)
    if arm == "copy":
        carried = {**point, **{k: v for k, v in protocol["copy"].items() if k != "note"}}
        return BrainLife(make_brain(carried, seed, genes, True), use_live=True, pouch=True)
    if arm == "lambda-zero":
        return BrainLife(
            make_brain({**point, "lam": 0.0}, seed, genes, True), use_live=True, pouch=True
        )
    if arm == "recurrent":
        settings = {k: v for k, v in protocol["recurrent"].items() if k != "note"}
        return Recurrent(seed, **settings)
    if arm == "blind":
        return BrainLife(make_brain(point, seed, genes, False), use_live=True, pouch=False)
    if arm == "step":
        return BrainLife(make_brain(point, seed, None, True), use_live=False, pouch=True)
    if arm == "tabular":
        table = protocol["tabular"]
        return Tabular(
            seed,
            alpha=table["alpha"],
            epsilon=table["epsilon"],
            gamma=table["gamma"],
            lam=table["lam"],
        )
    if arm == "random":
        return Random(seed)
    raise ValueError(f"unknown arm {arm!r}")


# --------------------------------------------------------------------------------- one life


def run_life(
    arm: str,
    seed: int,
    delay: int,
    protocol: dict[str, Any],
    genes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One arm, one seed, one delay: rule A for ``episodes`` episodes, then rule B. Returns
    the readings per rule and the life's work; a life whose answer refuses is returned with
    its error and the work done until then."""
    warnings.simplefilter("ignore")
    genes = {**protocol["arousal"], **(genes or {})}
    cost, food = float(protocol["cost"]), float(protocol.get("food", 1.0))
    truncation = float(protocol.get("truncation", 0.0))
    jitter, every = int(protocol["jitter"]), int(protocol["probe_every"])
    length = int(protocol["length"])
    window, floor = int(protocol["window"]), float(protocol["window_floor"])
    cells_rng = np.random.default_rng(protocol["corridor_seed"] + seed)
    # Exogenous events must not depend on how many rewards an arm earns or relocates.
    cuts_rng = np.random.default_rng(protocol["corridor_seed"] + 7919 + seed)
    food_rng = np.random.default_rng(protocol["corridor_seed"] + 15401 + seed)
    yoked_rng = np.random.default_rng(protocol["corridor_seed"] + 23719 + seed)
    started = time.time()
    life = make_life(arm, protocol, seed, genes)
    latency: dict[str, list[float]] = {"routine": [], "aroused": []}
    phases: list[dict[str, Any]] = []
    result: dict[str, Any] = {"arm": arm, "seed": seed, "delay": delay}

    def ledger() -> dict[str, Any]:
        work = dict(life.ledger()) if isinstance(life, BrainLife) else fresh_work()
        work["sweeps_per_routine_moment"] = work["sweeps_routine"] / max(1, work["routine"])
        work["sweeps_per_aroused_moment"] = work["sweeps_aroused"] / max(1, work["aroused"])
        work["latency_ms"] = {
            mode: [round(1000.0 * float(v), 3) for v in np.percentile(times, (50, 90, 100))]
            if times
            else None
            for mode, times in latency.items()
        }
        return work

    pending: tuple[float, bool] | None = None  # the preceding action's outcome
    bank = 0.0  # the yoked control's door outcome, paid at a random cell of the next episode
    rules = [KINDS.index(name) for name in protocol.get("rules", DEFAULT_RULES)]
    try:
        for index, keyed in enumerate(rules):
            if index and arm == "frozen":
                life.frozen = True
            fed, took, wrong, opened, modes = [], [], [], [], []
            behaviour = np.zeros((2, 5))
            policy = np.zeros((2, 5))
            visits = np.zeros((2, 5), dtype=int)
            executed = 0.0
            moments = cuts = 0
            probes: list[tuple[int, list[list[int]], list[list[float]]]] = []
            for episode in range(int(protocol["episodes"])):
                if episode % every == 0:
                    probes.append((episode, *life.probe()))
                cells = corridor(delay, length, jitter, cells_rng)
                cut = cuts_rng.random() < truncation
                available = food_rng.random() < food  # one draw per trip, for every arm
                if cut:  # the trip ends somewhere before the door, the key lost with it
                    cells = cells[: int(cuts_rng.integers(1, len(cells)))]
                holding = False
                got = take = 0
                wrongs = 0
                opening = None
                # the yoked control pays its bank at a random cell before the door; a trip cut to
                # one cell pays at that cell
                paid_at = int(yoked_rng.integers(max(1, len(cells) - 1))) if arm == "yoked" else -1
                for i, kind in enumerate(cells):
                    began = time.perf_counter()
                    reward, done = (None, False) if pending is None else pending
                    action, aroused = life.act(kind, holding, reward, done)
                    latency["aroused" if aroused else "routine"].append(time.perf_counter() - began)
                    modes.append(aroused)
                    moments += 1
                    h = int(holding)
                    visits[h, kind] += 1
                    behaviour[h, kind] += life.last[0]
                    policy[h, kind] += life.last[1]
                    executed += life.last[2]
                    outcome = 0.0
                    if kind == DOOR and holding:
                        opening = int(action == INTERACT)
                    if action == INTERACT:
                        if kind == keyed and not holding:
                            holding, take = True, 1
                        elif kind == DOOR:
                            if holding and available:
                                outcome, got = 1.0, 1
                        elif kind != FLOOR:  # the floor has nothing to interact with
                            outcome, wrongs = -cost, wrongs + 1
                    if arm == "yoked":
                        # the door's outcome is banked and paid at a random cell of the next
                        # episode, retiming credit for the creature's own earned reward
                        if kind == DOOR:
                            bank, outcome = outcome, 0.0
                        if i == paid_at:
                            outcome, bank = outcome + bank, 0.0
                    pending = (outcome, i == len(cells) - 1 and not cut)
                if cut:
                    cuts += 1
                    continue
                fed.append(got)
                took.append(take)
                wrong.append(wrongs)
                opened.append(opening)
            probes.append((int(protocol["episodes"]), *life.probe()))
            lag = next(
                (
                    k
                    for k in range(0, len(fed) - window + 1)
                    if np.mean(fed[k : k + window]) >= floor
                ),
                None,
            )
            n = min(50, len(fed))
            openings = [v for v in opened[-n:] if v is not None] if n else []
            phases.append(
                {
                    "keyed": KINDS[keyed],
                    "episodes": len(fed),
                    "cut": cuts,
                    "moments": moments,
                    "fed": float(np.mean(fed[-n:])) if n else None,
                    "fed_whole": float(np.mean(fed)) if fed else None,
                    "took": float(np.mean(took[-n:])) if n else None,
                    "wrong": float(np.mean(wrong[-n:])) if n else None,
                    "opened": float(np.mean(openings)) if openings else None,
                    "lag": lag,
                    "behaviour_interact": [
                        [round(b / v, 4) if v else None for b, v in zip(brow, vrow, strict=True)]
                        for brow, vrow in zip(behaviour, visits, strict=True)
                    ],
                    "policy_interact": [
                        [round(b / v, 4) if v else None for b, v in zip(brow, vrow, strict=True)]
                        for brow, vrow in zip(policy, visits, strict=True)
                    ],
                    "visits": visits.tolist(),
                    "executed_probability": round(executed / max(1, moments), 4),
                    "aroused": float(np.mean(modes)),
                    "aroused_late": float(np.mean(modes[len(modes) // 2 :])),
                    "start_greedy": probes[0][1],
                    "end_greedy": probes[-1][1],
                    "end_interact": [[round(v, 4) for v in row] for row in probes[-1][2]],
                }
            )
    except Exception as error:  # a crashed life is a recorded outcome, with its work
        result["error"] = f"{type(error).__name__}: {error}"[:300]
        result["completed_phases"] = len(phases)
    else:
        result["phases"] = phases
        result["pending_outcome"] = (
            None if pending is None else {"reward": pending[0], "done": pending[1]}
        )
        result["yoked_bank"] = bank
    result["seconds"] = round(time.time() - started, 2)
    result["work"] = ledger()
    return result


def _job(args: tuple[str, int, int, dict[str, Any], dict[str, Any] | None]) -> dict[str, Any]:
    arm, seed, delay, protocol, genes = args
    try:
        return run_life(arm, seed, delay, protocol, genes)
    except Exception as error:  # a life that could not start is a recorded outcome too
        return {
            "arm": arm,
            "seed": seed,
            "delay": delay,
            "error": f"{type(error).__name__}: {error}"[:300],
            "completed_phases": 0,
        }


def run(
    protocol: dict[str, Any],
    *,
    arms: list[str],
    seeds: list[int],
    delays: list[int],
    genes: dict[str, Any] | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    validate_plan(protocol, arms, seeds, delays)
    jobs = [
        (arm, seed, delay, protocol, genes) for arm in arms for delay in delays for seed in seeds
    ]
    if workers <= 1:
        return [_job(job) for job in jobs]
    with get_context("spawn").Pool(workers) as pool:
        return pool.map(_job, jobs, chunksize=1)


# ----------------------------------------------------------------------------------- report


def gates(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    """The protocol's gates over the ``live`` rows, per delay and pooled over the gated
    delays: the share of lives fed at the end of each rule, taking the key, wasting few
    interactions and calm again; a crashed life passes nothing."""
    g = protocol["gates"]
    tests = {
        "acquired": lambda r: (
            r["phases"][0]["fed"] is not None and r["phases"][0]["fed"] >= g["fed"]
        ),
        "adapted": lambda r: (
            r["phases"][1]["fed"] is not None and r["phases"][1]["fed"] >= g["fed"]
        ),
        "frugal": lambda r: all(
            p["wrong"] is not None and p["wrong"] <= g["wrong"] for p in r["phases"]
        ),
        "calm": lambda r: all(p["aroused_late"] <= g["aroused_late"] for p in r["phases"]),
    }
    if len(protocol.get("rules", DEFAULT_RULES)) > 2:
        # the first contingency returns: fed again, and found within the declared lag
        tests["retained"] = lambda r: (
            len(r["phases"]) > 2
            and r["phases"][2]["fed"] is not None
            and r["phases"][2]["fed"] >= g["fed"]
            and r["phases"][2]["lag"] is not None
            and r["phases"][2]["lag"] <= g["retained_lag"]
        )

    def shares(lives: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"lives": len(lives), "crashed": sum("error" in r for r in lives)}
        for name, test in tests.items():
            out[name] = float(np.mean(["error" not in r and bool(test(r)) for r in lives]))
        return out

    live = [r for r in rows if r["arm"] == "live"]
    report: dict[str, Any] = {}
    for delay in sorted({r["delay"] for r in live}):
        report[str(delay)] = shares([r for r in live if r["delay"] == delay])
    gated = [r for r in live if r["delay"] in g["delays"]]
    if gated:
        pooled = shares(gated)
        report["pooled"] = pooled
        report["passed"] = bool(
            pooled["crashed"] == 0 and all(pooled[name] >= g["share"] for name in tests)
        )
    return report


def summarize(rows: list[dict[str, Any]]) -> str:
    lines = []
    keys = sorted({(r["arm"], r["delay"]) for r in rows}, key=lambda k: (ARMS.index(k[0]), k[1]))
    for arm, delay in keys:
        group = [r for r in rows if r["arm"] == arm and r["delay"] == delay]
        good = [r for r in group if "error" not in r]
        if not good:
            lines.append(
                f"{arm:12s} D={delay:<3d} all {len(group)} lives crashed: {group[0]['error']}"
            )
            continue
        parts = []
        names = ("A", "B", "A'")[: len(good[0]["phases"])]
        for index, name in enumerate(names):
            fed = [r["phases"][index]["fed"] for r in good]
            took = [r["phases"][index]["took"] for r in good]
            wrong = [r["phases"][index]["wrong"] for r in good]
            fed, took, wrong = (
                [v for v in values if v is not None] for values in (fed, took, wrong)
            )
            if not fed:
                parts.append(f"{name}: no completed trips")
                continue
            lags = [r["phases"][index]["lag"] for r in good]
            found = [v for v in lags if v is not None]
            parts.append(
                f"{name}: fed {np.mean(fed):.2f} (min {min(fed):.2f}) took {np.mean(took):.2f} "
                f"wrong {np.mean(wrong):.2f} lag {int(np.median(found)) if found else '-':>4} "
                f"[{len(found)}/{len(lags)}]"
            )
        aroused = np.mean([p["aroused"] for r in good for p in r["phases"]])
        late = np.mean([p["aroused_late"] for r in good for p in r["phases"]])
        crashed = f" crashed {len(group) - len(good)}" if len(good) < len(group) else ""
        lines.append(
            f"{arm:12s} D={delay:<3d} "
            + " | ".join(parts)
            + f" | aroused {aroused:.2f} late {late:.2f}{crashed}"
        )
    return "\n".join(lines)


def markdown(report: dict[str, Any]) -> str:
    """The receipt's tables as Markdown; every number is read from the rows."""
    rows = report["rows"]
    delays = sorted({r["delay"] for r in rows})
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]

    def lives(arm: str, delay: int) -> list[dict[str, Any]]:
        return [r for r in rows if r["arm"] == arm and r["delay"] == delay and "error" not in r]

    def table(title: str, cell: Any) -> list[str]:
        head = "| Arm | " + " | ".join(f"D={d}" for d in delays) + " |"
        lines = [f"### {title}", "", head, "| --- |" + " --- |" * len(delays)]
        for arm in arms:
            cells = [cell(lives(arm, d)) if lives(arm, d) else "crashed" for d in delays]
            lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
        return lines + [""]

    def share(index: int, key: str) -> Any:
        def cell(group: list[dict[str, Any]]) -> str:
            values = [r["phases"][index][key] for r in group]
            values = [v for v in values if v is not None]
            if not values:
                return "none"
            return f"{np.mean(values):.2f} ({min(values):.2f})"

        return cell

    def lag(index: int) -> Any:
        def cell(group: list[dict[str, Any]]) -> str:
            values = [r["phases"][index]["lag"] for r in group]
            found = [v for v in values if v is not None]
            median = f"{int(np.median(found))}" if found else "none"
            return f"{median} ({len(found)}/{len(values)})"

        return cell

    def behaviour(index: int, holding: int, kind: int) -> Any:
        def cell(group: list[dict[str, Any]]) -> str:
            values = [r["phases"][index]["behaviour_interact"][holding][kind] for r in group]
            found = [v for v in values if v is not None]
            return f"{np.median(found):.3f} ({min(found):.3f})" if found else "none"

        return cell

    out = [
        *table(
            "Rule A, chest holds the key: episodes fed in the last 50, mean (minimum)",
            share(0, "fed"),
        ),
        *table(
            "Rule B, lamp holds the key: episodes fed in the last 50, mean (minimum)",
            share(1, "fed"),
        ),
        *table("Rule A: key taken in the last 50 episodes, mean (minimum)", share(0, "took")),
        *table("Rule B: key taken in the last 50 episodes, mean (minimum)", share(1, "took")),
        *table(
            "Rule A: wrong interactions per episode in the last 50, mean (minimum)",
            share(0, "wrong"),
        ),
        *table(
            "Rule B: wrong interactions per episode in the last 50, mean (minimum)",
            share(1, "wrong"),
        ),
        *table("Rule A: first 20-episode window 90% fed, median episode (lives / lives)", lag(0)),
        *table("Rule B: first 20-episode window 90% fed, median episode (lives / lives)", lag(1)),
        *table(
            "Rule A behaviour: probability of taking at the chest without the key, "
            "median (minimum)",
            behaviour(0, 0, CHEST),
        ),
        *table(
            "Rule B behaviour: probability of taking at the lamp without the key, median (minimum)",
            behaviour(1, 0, LAMP),
        ),
        *table(
            "Rule B behaviour: probability of interacting at a lever with the key, "
            "median (minimum)",
            behaviour(1, 1, LEVER),
        ),
    ]
    if any(len(r["phases"]) > 2 for r in rows if "error" not in r):
        out += [
            *table(
                "Rule A again, the key back in the chest: episodes fed in the last 50, "
                "mean (minimum)",
                share(2, "fed"),
            ),
            *table(
                "Rule A again: first 20-episode window 90% fed, median episode (lives / lives)",
                lag(2),
            ),
            *table(
                "Rule A again: wrong interactions per episode in the last 50, mean (minimum)",
                share(2, "wrong"),
            ),
        ]
    head = (
        "| Delay | aroused, whole life | aroused, second half of rule A | sweeps per routine "
        "moment | per aroused moment | learning sweeps | probe sweeps | memory reads | "
        "memory writes | routine ms (median, p90) | aroused ms (median, p90) |"
    )
    out += ["### The live arm: arousal and work, medians over lives", "", head]
    out.append("| --- |" + " --- |" * 10)

    def med(work: list[dict[str, Any]], key: str) -> str:
        return f"{np.median([w[key] for w in work]):.0f}"

    def ms(work: list[dict[str, Any]], mode: str) -> str:
        times = [w["latency_ms"][mode] for w in work if w["latency_ms"][mode]]
        if not times:
            return "none"
        return f"{np.median([t[0] for t in times]):.2f}, {np.median([t[1] for t in times]):.2f}"

    for delay in delays:
        group = lives("live", delay)
        if not group:
            continue
        work = [r["work"] for r in group]
        share = np.median([w["aroused"] / max(1, w["aroused"] + w["routine"]) for w in work])
        out.append(
            f"| {delay} | {share:.3f} | "
            f"{np.median([r['phases'][0]['aroused_late'] for r in group]):.3f} | "
            f"{np.median([w['sweeps_per_routine_moment'] for w in work]):.1f} | "
            f"{np.median([w['sweeps_per_aroused_moment'] for w in work]):.1f} | "
            f"{med(work, 'learning_sweeps')} | {med(work, 'probe_sweeps')} | "
            f"{med(work, 'memory_reads')} | {med(work, 'memory_writes')} | "
            f"{ms(work, 'routine')} | {ms(work, 'aroused')} |"
        )
    out += ["", "### Gates", "", "```json", json.dumps(report["gates"], indent=1), "```"]
    return "\n".join(out)


# --------------------------------------------------------------------------------- receipts


def sources() -> list[tuple[str, Path]]:
    package = Path(cd.__file__).resolve().parent
    return [
        ("key_door.py", Path(__file__).resolve()),
        *(
            ("cadence/" + p.relative_to(package).as_posix(), p)
            for p in sorted(package.rglob("*.py"))
        ),
    ]


def read_receipt(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    stored: dict[str, Any] = json.loads(data)
    body: dict[str, Any] = stored["body"]
    return body


def write_receipt(path: Path, body: dict[str, Any], manifest: dict[str, Any]) -> None:
    receipt = Receipt.build(SCHEMA, body, sources())
    if receipt.source != manifest:
        raise RuntimeError("a source file changed during the run; discard it and run again")
    data = (canonical_json(receipt.to_dict()) + "\n").encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    path.write_bytes(data)


def planned(body: dict[str, Any]) -> list[tuple[str, int, int]]:
    return [(a, d, s) for a in body["arms"] for d in body["delays"] for s in body["seeds"]]


def validate_plan(
    protocol: dict[str, Any], arms: list[str], seeds: list[int], delays: list[int]
) -> None:
    """Reject empty or duplicate plans and world settings without their declared meaning."""
    for name, values in (("arms", arms), ("seeds", seeds), ("delays", delays)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and unique")
    if any(a not in ARMS for a in arms):
        raise ValueError("unknown arm in the plan")
    if any(type(v) is not int or v < 0 for v in [*seeds, *delays]):
        raise ValueError("seeds and delays must be nonnegative integers")
    for name in ("episodes", "probe_every", "window", "length"):
        if type(protocol[name]) is not int or protocol[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(protocol["jitter"]) is not int or protocol["jitter"] < 0:
        raise ValueError("jitter must be a nonnegative integer")
    if max(delays) + protocol["jitter"] + 3 > protocol["length"]:
        raise ValueError("the planned delays do not fit the fixed corridor length")
    for name in ("food", "truncation", "window_floor"):
        value = protocol.get(name, 1.0 if name == "food" else 0.0)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if not np.isfinite(protocol["cost"]) or protocol["cost"] < 0:
        raise ValueError("cost must be finite and nonnegative")


def verify(path: Path, *, current: bool = False, protocol: Path = PROTOCOL) -> tuple[bool, str]:
    """Canonical form and digest, one row per planned life, the gates recomputed, the
    protocol text against its hash; ``current`` also requires the present sources."""

    def check(body: dict[str, Any]) -> str | None:
        kinds = (SCHEMA, *LEGACY_SCHEMAS)
        if stored["kind"] not in kinds or body["protocol"]["schema"] not in kinds:
            return "the receipt has the wrong kind"
        rules = body["protocol"].get("rules", DEFAULT_RULES)
        source = stored["source"]
        files = source["files"]
        if source["manifest_sha256"] != canonical_sha256(files):
            return "the embedded source manifest digest does not verify"
        paths = [entry["path"] for entry in files]
        if (
            not paths
            or paths[0] != "key_door.py"
            or len(set(paths)) != len(paths)
            or not {"cadence/generic.py", "cadence/arousal.py", "cadence/receipts.py"} <= set(paths)
            or any(
                not isinstance(entry["sha256"], str)
                or len(entry["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in entry["sha256"])
                for entry in files
            )
        ):
            return "the embedded source manifest is incomplete or malformed"
        validate_plan(body["protocol"], body["arms"], body["seeds"], body["delays"])
        rows = body["rows"]
        if [(r["arm"], r["delay"], r["seed"]) for r in rows] != planned(body):
            return "the rows are not the planned lives, each once and in order"
        for row in rows:
            work = row.get("work")
            if work is not None:
                new_counters = {"probe_memory_reads", "aborted_forecast_sweeps"}
                counters = set(fresh_work()) - new_counters
                if stored["kind"] != LEGACY_SCHEMA:
                    counters.update(new_counters)
                if any(type(work[k]) is not int or work[k] < 0 for k in counters):
                    return "work counters must be nonnegative integers"
                for mode in ("routine", "aroused"):
                    if work["sweeps_per_" + mode + "_moment"] != work["sweeps_" + mode] / max(
                        1, work[mode]
                    ):
                        return "the sweeps per moment do not follow from the work counters"
            if "error" in row:
                if not row["error"] or row["completed_phases"] not in range(len(rules)):
                    return "a crashed life has invalid completion accounting"
                continue
            if [p["keyed"] for p in row["phases"]] != rules:
                return "a completed life must contain the protocol's rules, once and in order"
            for phase in row["phases"]:
                for k in ("episodes", "cut", "moments"):
                    if type(phase[k]) is not int or phase[k] < 0:
                        return "phase counts must be nonnegative integers"
                shares = (
                    "fed",
                    "fed_whole",
                    "took",
                    "opened",
                    "aroused",
                    "aroused_late",
                    "executed_probability",
                )
                if not all(phase[k] is None or 0 <= phase[k] <= 1 for k in shares):
                    return "a recorded share is outside [0, 1]"
                if any(
                    phase[k] is None for k in ("aroused", "aroused_late", "executed_probability")
                ):
                    return "a completed phase is missing moment readings"
                if phase["episodes"]:
                    if any(phase[k] is None for k in ("fed", "fed_whole", "took", "wrong")):
                        return "a completed trip is missing outcome readings"
                    if (
                        phase["fed"] > phase["took"]
                        or not 0 <= phase["wrong"] <= body["protocol"]["length"] - 2
                    ):
                        return "the outcomes contradict the key or wrong-interaction bounds"
                elif any(
                    phase[k] is not None
                    for k in ("fed", "fed_whole", "took", "wrong", "opened", "lag")
                ):
                    return "a phase without completed trips has outcome readings"
                visits = phase["visits"]
                if (
                    len(visits) != 2
                    or any(len(v) != 5 for v in visits)
                    or any(type(v) is not int or v < 0 for vs in visits for v in vs)
                ):
                    return "visits must be a nonnegative integer 2 by 5 table"
                if sum(map(sum, phase["visits"])) != phase["moments"]:
                    return "visits do not agree with the moments lived"
                if sum(v[DOOR] for v in visits) != phase["episodes"]:
                    return "door visits do not agree with completed trips"
                if phase["episodes"] + phase["cut"] != body["protocol"]["episodes"]:
                    return "the trips that reached the door and the cut trips do not add up"
                if phase["lag"] is not None and (
                    type(phase["lag"]) is not int
                    or not 0 <= phase["lag"] <= phase["episodes"] - body["protocol"]["window"]
                ):
                    return "the lag is not a completed-trip window index"
            if work is None or work["brains"] != int(row["arm"] not in BRAINLESS):
                return "the work does not identify the arm's brain"
            if work["brains"] and work["routine"] + work["aroused"] != sum(
                p["moments"] for p in row["phases"]
            ):
                return "the work moments do not agree with the phases"
            if stored["kind"] != LEGACY_SCHEMA:
                pending = row["pending_outcome"]
                if (
                    type(pending["done"]) is not bool
                    or not -body["protocol"]["cost"] <= pending["reward"] <= 1
                ):
                    return "the pending outcome is outside the world's range"
                if row["yoked_bank"] not in (0.0, 1.0) or (
                    row["arm"] != "yoked" and row["yoked_bank"] != 0
                ):
                    return "the yoked bank is invalid"
        if gates(rows, body["protocol"]) != body["gates"]:
            return "the stored gates do not follow from the rows"
        text = body["protocol_source"]
        if hashlib.sha256(text.encode()).hexdigest() != body["protocol_sha256"]:
            return "the embedded protocol source differs from its recorded hash"
        if body["frozen_protocol"] and (
            body["protocol"] != json.loads(text)
            or body["genes_override"] is not None
            or body["protocol"]["schema"] != stored["kind"]
        ):
            return "the frozen settings differ from the recorded protocol"
        if current and body["protocol_sha256"] != hashlib.sha256(protocol.read_bytes()).hexdigest():
            return "the protocol file differs from the recorded hash"
        return None

    try:
        data = path.read_bytes()
        if path.suffix == ".gz":
            data = gzip.decompress(data)
        stored = json.loads(data)
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "receipt.json"
            plain.write_bytes(data)
            valid, reason = Receipt.verify(
                plain, sources=sources() if current else None, check=check
            )
            if valid and stored["kind"] == LEGACY_SCHEMA:
                reason += (
                    "; legacy key-door/1: policy-dependent schedules and historical metric limits"
                )
            elif valid and stored["kind"] != SCHEMA:
                reason += "; key-door/2: the corrected two-rule instrument"
            return valid, reason
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        EOFError,
    ) as error:
        return False, f"cannot verify the receipt: {error}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument(
        "--seeds", nargs="+", default=["development"], help="seed numbers or set names"
    )
    parser.add_argument("--delays", nargs="+", type=int, default=None)
    parser.add_argument("--genes", default=None, help="JSON overrides of the arousal genes")
    parser.add_argument(
        "--point", default=None, help="JSON overrides of the operating point; null removes"
    )
    parser.add_argument(
        "--cost", type=float, default=None, help="override the wrong-interaction cost"
    )
    parser.add_argument(
        "--food", type=float, default=None, help="override the chance of food at the door"
    )
    parser.add_argument("--episodes", type=int, default=None, help="override the episodes per rule")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="print a receipt's tables")
    parser.add_argument("--verify", type=Path, default=None, help="check a receipt")
    parser.add_argument(
        "--current", action="store_true", help="with --verify: against the present sources"
    )
    args = parser.parse_args(argv)
    if args.report is not None:
        valid, reason = verify(args.report)
        if not valid:
            print(json.dumps({"verified": False, "reason": reason}))
            return 1
        print(markdown(read_receipt(args.report)))
        return 0
    if args.verify is not None:
        valid, reason = verify(args.verify, current=args.current)
        print(json.dumps({"verified": valid, "reason": reason}))
        return 0 if valid else 1
    manifest = source_manifest(sources())
    frozen = args.protocol.read_bytes()
    protocol = json.loads(frozen)
    if protocol.get("schema") not in (SCHEMA, *LEGACY_SCHEMAS):
        parser.error(f"the protocol's schema is not {SCHEMA} or one of {LEGACY_SCHEMAS}")
    if protocol.get("schema") == SCHEMA and "recurrent" not in protocol:
        parser.error("a key-door/3 protocol declares the recurrent learner's settings")
    overridden = False
    for name in ("cost", "food", "episodes"):
        if getattr(args, name) is not None:
            protocol[name] = getattr(args, name)
            overridden = True
    if args.point:
        point = {**protocol["operating_point"], **json.loads(args.point)}
        protocol["operating_point"] = {k: v for k, v in point.items() if v is not None}
        overridden = True
    genes = json.loads(args.genes) if args.genes else None
    overridden = overridden or genes is not None
    seeds: list[int] = []
    for value in args.seeds:
        seeds.extend(protocol["seeds"][value] if value in protocol["seeds"] else [int(value)])
    delays = args.delays or protocol["delays"]
    try:
        validate_plan(protocol, args.arms, seeds, delays)
    except (ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    rows = run(
        protocol, arms=args.arms, seeds=seeds, delays=delays, genes=genes, workers=args.workers
    )
    print(summarize(rows))
    body = {
        "cadence": cd.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "protocol_sha256": hashlib.sha256(frozen).hexdigest(),
        "protocol_source": frozen.decode(),
        "frozen_protocol": not overridden
        and protocol["schema"] == SCHEMA
        and frozen == PROTOCOL.read_bytes(),
        "protocol": protocol,
        "genes_override": genes,
        "arms": list(args.arms),
        "delays": list(delays),
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
