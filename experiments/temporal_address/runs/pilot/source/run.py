"""Actual POPGym Easy task with a causal three-stage sensory delay, no clock input."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "core/src"))
sys.path.insert(0, str(HERE / "environment"))

import numpy as np
from popgym.envs.repeat_previous import RepeatPreviousEasy

import cadence as cd

ARMS = ["patch_memory", "patch_masked", "patch_no_direct", "linear_memory", "linear_masked"]


class SensoryDelay:
    """Three internal one-step stages. Read first; write the current observation later."""

    def __init__(self, batch, lag=3):
        self.state = np.zeros((batch, lag, 4))

    def features(self, observation, masked=False):
        old = self.state[:, -1].copy()
        if masked:
            old[:] = 0
        return np.concatenate([np.eye(4)[observation], old], axis=1)

    def observe(self, observation):
        self.state[:, 1:] = self.state[:, :-1].copy()
        self.state[:, 0] = np.eye(4)[observation]


class Patch:
    def __init__(self, seed, direct=True):
        c = cd.layered(8, 32, 4, density=1, seed=seed, skip=direct, skip_init=0)
        self.c = c
        learner = cd.Learner(
            cd.Brain(c, cd.learning_neuron_model(dt=1.0), dense_limit=8192),
            c.populations["output"],
            cd.LearnerConfig(
                beta=0.01, free_steps=50, nudged_steps=50, tolerance=1e-5, temperature=0.2
            ),
        )
        self.ac = cd.ActorCritic(
            learner,
            c.populations["hidden"],
            cd.ActorCriticConfig(
                gamma=0, lam=0, eta=0.02, eta_bias=0.002, eta_critic=0.1, critic_signal="td"
            ),
            seed=seed,
        )

    def reset(self):
        self.ac.reset()

    def act(self, features, greedy):
        self.drive = np.zeros((len(features), self.c.n))
        self.drive[:, self.c.populations["input"]] = features
        # Start a real decision at this drive; no hidden-state memory is carried.
        self.ac.reset()
        return self.ac.act(self.drive, greedy=greedy)

    def learn(self, reward, done):
        # Immediate reward: gamma=0. Rest is a dummy next drive with no bootstrap effect.
        self.ac.learn(reward, done, np.zeros_like(self.drive))

    def state(self):
        return {
            "efficacy": self.ac.learner.brain.efficacy,
            "bias": self.ac.learner.brain.bias,
            "critic": self.ac.w_critic,
            "critic_bias": np.array(self.ac.b_critic),
        }

    def description(self):
        return {
            "parameters": self.ac.parameters(),
            "learner": self.ac.learner.config.to_dict(),
            "actor": self.ac.config.to_dict(),
            "connectome": self.c.digest(),
        }


class Linear:
    """Ordinary softmax score policy, trained only from sampled-action rewards."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.w = self.rng.normal(0, 0.01, (8, 4))
        self.bias = np.zeros(4)
        self.critic = np.zeros(9)

    def reset(self):
        pass

    def act(self, features, greedy):
        self.features = features
        z = features @ self.w + self.bias
        p = np.exp(z - z.max(axis=1, keepdims=True))
        self.p = p / p.sum(axis=1, keepdims=True)
        self.action = (
            self.p.argmax(axis=1)
            if greedy
            else np.array([self.rng.choice(4, p=row) for row in self.p])
        )
        return self.action

    def learn(self, reward, done):
        x = np.column_stack([self.features, np.ones(len(reward))])
        delta = reward - x @ self.critic
        score = np.eye(4)[self.action] - self.p
        weighted = np.clip(delta, -1, 1)[:, None] * score
        self.w += 0.02 * self.features.T @ weighted / len(reward)
        self.bias += 0.002 * weighted.mean(axis=0)
        self.critic += 0.1 * (delta[:, None] * x / (1 + (x * x).sum(axis=1, keepdims=True))).mean(
            axis=0
        )

    def state(self):
        return {"weights": self.w, "bias": self.bias, "critic": self.critic}

    def description(self):
        return {
            "parameters": 45,
            "actor_rate": 0.02,
            "bias_rate": 0.002,
            "critic_rate": 0.1,
            "gamma": 0,
            "gradient": "sampled-action categorical log-policy score",
        }


def episode(agent, seed, batch, masked, train):
    envs = [RepeatPreviousEasy() for _ in range(batch)]
    obs = np.array([env.reset(seed=seed + i)[0] for i, env in enumerate(envs)])
    delay = SensoryDelay(batch)
    agent.reset()
    tapes = {name: [] for name in ("observations", "actions", "rewards", "memory")}
    while True:
        features = delay.features(obs, masked)
        action = agent.act(features, greedy=not train)
        transitions = [env.step(int(a)) for env, a in zip(envs, action, strict=True)]
        next_obs = np.array([row[0] for row in transitions])
        reward = np.array([row[1] for row in transitions])
        done = np.array([row[2] or row[3] for row in transitions])
        assert done.all() or not done.any()
        if train:
            agent.learn(reward * 48, done)
        for name, value in (
            ("observations", obs),
            ("actions", action),
            ("rewards", reward),
            ("memory", features[:, 4:]),
        ):
            tapes[name].append(value.copy())
        # The current observation becomes past state only after its decision and reward.
        delay.observe(obs)
        obs = next_obs
        if done.all():
            break
    for env in envs:
        env.close()
    tape = {name: np.array(value) for name, value in tapes.items()}
    assert len(tape["rewards"]) == 51
    return tape


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(body, directory):
    outcomes = body["outcomes"]
    expected = {(s, a) for s in body["schedule"]["seeds"] for a in ARMS}
    if {(r["seed"], r["arm"]) for r in outcomes} != expected or len(outcomes) != len(expected):
        raise ValueError("missing scheduled outcomes")
    for row in outcomes:
        for artifact in row["artifacts"]:
            if digest(directory / artifact["name"]) != artifact["sha256"]:
                raise ValueError("artifact changed")
        with np.load(directory / row["artifacts"][0]["name"]) as data:
            reward = data["rewards"]
            if reward.shape != (51, 100):
                raise ValueError("wrong original task episode shape")
            expected_reward = np.where(
                data["actions"][3:] == data["observations"][:-3], 1 / 48, -1 / 48
            )
            np.testing.assert_array_equal(reward[:3], 0)
            np.testing.assert_array_equal(reward[3:], expected_reward)
            if row["arm"].endswith("masked"):
                np.testing.assert_array_equal(data["memory"], 0)
            else:
                np.testing.assert_array_equal(
                    data["memory"][3:], np.eye(4)[data["observations"][:-3]]
                )
            np.testing.assert_allclose(reward.sum(axis=0), row["returns"], atol=0, rtol=0)
            if float(reward.sum(axis=0).mean()) != row["mean_return"]:
                raise ValueError("return arithmetic changed")
        if len(row["training_returns"]) != body["schedule"]["episodes"]:
            raise ValueError("missing training episodes")


def verify(path):
    receipt = cd.Receipt.read(path)
    sources = [(e["path"], path.parent / "source" / e["path"]) for e in receipt.source["files"]]
    ok, message = cd.Receipt.verify(path, sources=sources)
    if not ok:
        raise ValueError(message)
    check(receipt.body, path.parent)
    print(message + "; episode rewards, causal memory and all outcomes agree")


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    files = [HERE / "run.py", HERE / "versions.json"]
    files += sorted((HERE / "core").rglob("*.py")) + sorted((HERE / "environment").rglob("*.py"))
    files += [p for p in sorted((HERE / "original").rglob("*")) if p.is_file()]
    frozen = []
    for p in files:
        name = str(p.relative_to(HERE))
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
        frozen.append((name, target))
    body = {
        "core_commit": "2de0383",
        "schedule": {
            "seeds": args.seeds,
            "episodes": args.episodes,
            "batch": 32,
            "evaluation_episodes": 100,
        },
        "task": {
            "class": "RepeatPreviousEasy",
            "k": 4,
            "lag": 3,
            "episode_steps": 51,
            "rewarded_steps": 48,
            "suits": 4,
            "reward_scale_for_learning": 48,
        },
        "boundary": "Designed internal three-stage sensory delay, no clock or target input. Immediate reward. No PPO superiority claim.",
        "outcomes": [],
    }
    for seed in args.seeds:
        for arm in ARMS:
            agent = (
                Linear(seed) if arm.startswith("linear") else Patch(seed, arm != "patch_no_direct")
            )
            start = time.perf_counter()
            curve = []
            for episode_index in range(args.episodes):
                tape = episode(
                    agent, seed * 100000 + episode_index * 32, 32, arm.endswith("masked"), True
                )
                curve.append(float(tape["rewards"].sum(axis=0).mean()))
            held = episode(agent, 1000000 + seed * 1000, 100, arm.endswith("masked"), False)
            returns = held["rewards"].sum(axis=0)
            artifacts = []
            for suffix, values in (("evaluation", held), ("parameters", agent.state())):
                p = args.output / f"{arm}_seed{seed}_{suffix}.npz"
                np.savez_compressed(p, **values)
                artifacts.append({"name": p.name, "sha256": digest(p)})
            row = {
                "seed": seed,
                "arm": arm,
                "description": agent.description(),
                "training_returns": curve,
                "returns": returns.tolist(),
                "mean_return": float(returns.mean()),
                "training_steps": args.episodes * 32 * 51,
                "seconds": time.perf_counter() - start,
                "artifacts": artifacts,
            }
            body["outcomes"].append(row)
            cd.Receipt.build("cadence/temporal-address/v1", body, frozen).write(
                args.output / "receipt.json"
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "arm": arm,
                        "return": row["mean_return"],
                        "seconds": row["seconds"],
                    }
                ),
                flush=True,
            )
    check(body, args.output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--output", type=Path, default=HERE / "runs/main")
    p.add_argument("--verify", type=Path)
    args = p.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    if args.verify:
        verify(args.verify)
    else:
        run(args)


if __name__ == "__main__":
    main()
