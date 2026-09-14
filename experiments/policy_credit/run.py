"""Five-seed delayed reward controls; no claim of a general temporal-credit solution.

Only the first action matters. Reward arrives after a sequence of blank
observations and irrelevant sampled actions. All arms get identical contexts,
delay, reward, and transition budgets. The tabular score-rule control receives
the same one-hot observation; it has fewer parameters and no settling dynamics.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

import cadence as cd

ARMS = ("trace", "no_trace", "legacy_quadratic", "tabular_score")


class LegacyActor(cd.ActorCritic):
    """The old discrete actor inherited the imitation loss."""

    def _nudged_groups(self, drive, free, target, beta):
        return self.learner.nudged(drive, free, target, sign=beta / self.learner.config.beta)


def patch(seed, lam, legacy):
    graph = cd.layered(4, 8, 2, density=1.0, seed=seed)
    learner = cd.Learner(
        cd.Brain(graph, cd.learning_neuron_model(dt=1.0)),
        graph.populations["output"],
        cd.LearnerConfig(nudge="quadratic", beta=.01, free_steps=80,
                         nudged_steps=80, tolerance=1e-6, temperature=.2),
    )
    cls = LegacyActor if legacy else cd.ActorCritic
    ac = cls(learner, graph.populations["hidden"],
             cd.ActorCriticConfig(gamma=1., lam=lam, eta=1., eta_bias=.05,
                                  eta_critic=0., dopamine_cap=0.), seed=seed)
    return ac


def drive(ac, context):
    levels = np.zeros((len(context), ac.n))
    for j in (0, 1):
        levels[np.arange(len(context)), 2 * context + j] = 1.
    return ac.learner.brain.stimulus_levels(levels)


def train(arm, seed, length, episodes, batch):
    started = time.perf_counter()
    data = np.random.default_rng(10_000 + seed)
    curve = []
    ac = None if arm == "tabular_score" else patch(seed, 0. if arm == "no_trace" else .95,
                                                 arm == "legacy_quadratic")
    # A linear categorical policy on the same four observed channels plus bias.
    rng = np.random.default_rng(seed)
    weights = np.zeros((5, 2))
    for _ in range(episodes):
        context = data.integers(0, 2, batch)
        first = None
        trace = np.zeros((batch, 5, 2))
        if ac is not None:
            ac.reset()
            d = drive(ac, context)
            blank = np.zeros_like(d)
        for moment in range(length):
            if ac is not None:
                chosen = ac.act(d if moment == 0 else blank)
            else:
                x = np.zeros((batch, 5))
                x[:, -1] = 1.
                if moment == 0:
                    x[np.arange(batch), 2 * context] = 1.
                    x[np.arange(batch), 2 * context + 1] = 1.
                logits = x @ weights
                p = np.exp(logits - logits.max(axis=1, keepdims=True))
                p /= p.sum(axis=1, keepdims=True)
                chosen = (rng.random(batch) > p[:, 0]).astype(int)
                score = np.eye(2)[chosen] - p
                trace = .95 * trace + x[:, :, None] * score[:, None, :]
            if moment == 0:
                first = chosen.copy()
            terminal = moment == length - 1
            reward = (first == context).astype(float) if terminal else np.zeros(batch)
            if ac is not None:
                ac.learn(reward, np.full(batch, terminal), blank)
            else:
                weights += .2 * (reward[:, None, None] * trace).mean(axis=0)
        curve.append(float(reward.mean()))
    # Exact evaluation of both possible fresh contexts, including stochastic accuracy.
    if ac is not None:
        ac.reset()
        state = ac.settle(drive(ac, np.array([0, 1])))
        probabilities = ac.probabilities(state)
        parameters = ac.parameters()
    else:
        x = np.array([[1., 1., 0., 0., 1.], [0., 0., 1., 1., 1.]])
        logits = x @ weights
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        parameters = weights.size
    return {"arm": arm, "seed": seed, "length": length, "training": curve,
            "episodes": episodes, "batch": batch, "transitions": episodes * batch * length,
            "seconds": time.perf_counter() - started, "parameters": parameters,
            "probabilities": probabilities.tolist(),
            "accuracy": float((probabilities.argmax(axis=1) == [0, 1]).mean()),
            "stochastic_accuracy": float(probabilities[[0, 1], [0, 1]].mean())}


def check(body):
    s = body["schedule"]
    expected = {(a, seed, length) for a in ARMS for seed in s["seeds"] for length in s["lengths"]}
    rows = body["rows"]
    if len(rows) != len(expected) or {(r["arm"], r["seed"], r["length"]) for r in rows} != expected:
        return "missing or duplicate outcomes"
    for r in rows:
        if len(r["training"]) != s["episodes"] or r["transitions"] != s["episodes"] * s["batch"] * r["length"]:
            return "wrong interaction budget"
        p = np.asarray(r["probabilities"])
        if p.shape != (2, 2) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1.):
            return "invalid probabilities"
        if r["accuracy"] != float((p.argmax(1) == [0, 1]).mean()) or abs(r["stochastic_accuracy"] - p[[0, 1], [0, 1]].mean()) > 1e-12:
            return "summary mismatch"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--lengths", default="3,9")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    frozen = args.output.parent / "sources"
    if args.verify:
        sources = [(str(p.relative_to(frozen)), p) for p in sorted(frozen.rglob("*.py"))]
        ok, reason = cd.Receipt.verify(args.output, sources=sources, check=check)
        print(reason)
        return 0 if ok else 1
    if args.output.exists() or frozen.exists():
        raise ValueError("use a new output directory; receipts and sources are immutable")
    frozen.mkdir(parents=True)
    shutil.copy2(__file__, frozen / "run.py")
    shutil.copytree(Path(cd.__file__).resolve().parent, frozen / "cadence",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.nbc", "*.nbi"))
    sources = [(str(p.relative_to(frozen)), p) for p in sorted(frozen.rglob("*.py"))]
    schedule = {"seeds": list(map(int, args.seeds.split(","))),
                "lengths": list(map(int, args.lengths.split(","))),
                "episodes": args.episodes, "batch": 32}
    body = {"schedule": schedule, "rows": [], "python": platform.python_version(),
            "platform": platform.platform(), "numpy": np.__version__, "joules": None,
            "boundary": "Toy delayed credit, not POPGym or console performance. No learned address or teacher. CPU timings may share the host with other audits; not an efficiency claim."}
    for length in schedule["lengths"]:
        for seed in schedule["seeds"]:
            for arm in ARMS:
                row = train(arm, seed, length, args.episodes, schedule["batch"])
                body["rows"].append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "training"}), flush=True)
    error = check(body)
    if error:
        raise ValueError(error)
    cd.Receipt.build("cadence/policy-credit/v1", body, sources=sources).write(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
