"""A bounded diagnosis of T1's observation-time contract, not a new RL benchmark.

Schedule: seeds 0,1,2; ten vector episodes of 32 environments per arm; 100
fresh evaluation episodes. The corrected ring, original ring and masked-memory
control share the same allocated network and immediate-reward learning rule.
Twenty oracle episodes per level test the installed environment independently.
All scheduled outcomes are retained, including unsuccessful learning.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import cadence as cd
from popgym.envs.repeat_previous import RepeatPrevious

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("t1", HERE.parent / "t1_repeat_previous/run.py")
t1 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(t1)
CORE = Path(cd.__file__).resolve().parent
SOURCES = [("t1_alignment/run.py", Path(__file__).resolve()),
           ("t1_repeat_previous/run.py", HERE.parent / "t1_repeat_previous/run.py"),
           ("popgym/repeat_previous.py", Path(inspect.getfile(RepeatPrevious)))]
SOURCES += [(f"cadence/{p.name}", p) for p in sorted(CORE.glob("*.py"))]


def oracle(level: str, lag: int, episodes: int = 20) -> dict:
    returns = []
    for seed in range(episodes):
        env = t1.make_env(level)
        observation, _ = env.reset(seed=seed)
        seen, total = [observation], 0.0
        while True:
            action = seen[-1 - lag] if len(seen) > lag else 0
            observation, reward, terminated, truncated, _ = env.step(action)
            total += reward
            seen.append(observation)
            if terminated or truncated:
                break
        env.close()
        returns.append(total)
    return {"level": level, "lag": lag, "returns": returns, "mean_return": float(np.mean(returns))}


def episode(agent, envs, arm: str, learn: bool) -> dict:
    observation = envs.reset()
    agent.reset(len(observation))
    agent.ac.reset()
    total = np.zeros(len(observation))
    memory_hits = memory_trials = 0
    history = []

    def drive(obs, tick):
        out = agent.drive(obs, tick)
        if arm == "masked":
            out[:, agent.memory] = 0
        return out

    d = drive(observation, envs.t)
    done = False
    while not done:
        history.append(observation.copy())
        if len(history) >= agent.k:
            expected = history[-agent.k]
            memory_hits += int((d[:, agent.memory].argmax(axis=1) == expected).sum())
            memory_trials += len(observation)
        state = agent.ac.settle(d)
        agent.after(state, observation)
        action = agent.ac.act(d, greedy=not learn)
        observation, reward, done = envs.step(action)
        d = drive(observation, envs.t)
        if learn:
            # Reward arrives immediately for this decision. Gamma=lambda=0 removes
            # unrelated long-horizon credit from this address-alignment diagnosis.
            scale = envs.envs[0].deck.num_cards - agent.k
            agent.ac.learn(reward * scale, np.full(len(observation), done), d)
        total += reward
    return {"return": float(total.mean()), "memory_target_accuracy": memory_hits / memory_trials}


def trained(arm: str, seed: int, episodes: int) -> dict:
    # Every arm allocates four beat owners. The corrected ring leaves one unused,
    # keeping initialization, parameter count, current observation and capacity equal.
    agent = t1.PatchAgent("fast", 4, seed, memory_lag=4)
    agent.memory_lag = 3 if arm != "original" else 4
    agent.learner.config = cd.LearnerConfig(beta=0.01, free_steps=50,
                                         nudged_steps=50, tolerance=1e-5)
    agent.ac.config = cd.ActorCriticConfig(gamma=0.0, lam=0.0, eta=0.02,
                                         eta_bias=0.002, eta_critic=0.1)
    envs = t1.Envs("easy", 32, seed * 100_000)
    started = time.perf_counter()
    curve = [episode(agent, envs, arm, True) for _ in range(episodes)]
    held_out = episode(agent, t1.Envs("easy", 100, 1_000_000 + seed * 1000), arm, False)
    print(arm, seed, held_out, flush=True)
    return {"arm": arm, "seed": seed, "episodes": episodes, "env_steps": episodes * 32 * 51,
            "seconds": time.perf_counter() - started, "parameters": agent.parameters(),
            "lag": agent.memory_lag, "actor": agent.ac.config.to_dict(),
            "learner": agent.learner.config.to_dict(), "training": curve, "held_out": held_out}


def check(body: dict) -> str | None:
    for row in body["oracles"]:
        if abs(np.mean(row["returns"]) - row["mean_return"]) > 1e-12:
            return "oracle summary mismatch"
        if row["lag"] == t1.LEVELS[row["level"]] - 1 and not np.allclose(row["returns"], 1.0):
            return "the installed environment no longer has the measured lag"
    scheduled = {(arm, seed) for arm in ("corrected", "original", "masked") for seed in body["schedule"]["seeds"]}
    observed = {(row["arm"], row["seed"]) for row in body["trained"]}
    if scheduled != observed or len(body["trained"]) != len(scheduled):
        return "scheduled outcomes missing or duplicated"
    for row in body["trained"]:
        if len(row["training"]) != body["schedule"]["episodes"]:
            return "incomplete training schedule"
        if not -1 - 1e-12 <= row["held_out"]["return"] <= 1 + 1e-12:
            return "invalid return"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=HERE / "receipt.json")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=SOURCES, check=check)
        print(message)
        return 0 if ok else 1
    seeds = [int(s) for s in args.seeds.split(",")]
    body = {"schedule": {"seeds": seeds, "episodes": args.episodes, "batch": 32, "evaluation_episodes": 100},
            "oracles": [oracle(level, k + offset) for level, k in t1.LEVELS.items() for offset in (0, -1)],
            "trained": [trained(arm, seed, args.episodes) for seed in seeds for arm in ("corrected", "original", "masked")],
            "boundary": "The memory write and clock are designed; only the reward-to-action map learns. This is an alignment diagnosis, not learned memory, delayed reward credit, or general superiority."}
    error = check(body)
    if error:
        raise ValueError(error)
    cd.Receipt.build("cadence-paper/t1-alignment/v1", body, sources=SOURCES).write(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
