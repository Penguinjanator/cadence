"""T1: POPGym's RepeatPrevious, from reward. Match the environment's scored card.

The first environment of tier 1 in MIND_TESTS.md: a memory task with a published suite
behind it (Morad et al. 2023). A card's suit arrives each step; the action must be the
suit at hand[-k] before the next deal (k = 4, 32, 64 for Easy, Medium and Hard),
which is k - 1 observations before the current one in the installed environment. The reward is
plus or minus one over the steps that count, so an episode's return lies in [-1, 1]. The
brain learns from reward alone with the three-factor rule (`ActorCritic`), as the cart-pole
gate did. Its memory is the n-back mechanism: the agent's own step count as a clock of
period k - 1, one owner per beat, into fast seams (`FastSeams`, replacing) onto a memory range
of four owners, read before the write, so the beat's slot still holds the suit from k - 1
steps ago when the beat returns. Arms: the clock and the fast seams; the carried context
alone (`Echo`); nothing (a memoryless net). Baselines: PPO with a memoryless MLP, and PPO
with a window of the last k + 1 observations (memory handed to it). The receipt: the mean
episode return over held-out episodes for every arm, and the steps to learn.

Run:  ../../.venv_suites/bin/python run.py --level easy --pilot
      ../../.venv_suites/bin/python run.py --level easy
      ../../.venv_suites/bin/python run.py --verify receipt_easy.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import cadence as cd
from cadence.stream import columns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.ppo import MLPPolicy, PPOConfig, count_parameters, train_ppo  # noqa: E402

SOURCES = [
    ("experiments/t1_repeat_previous/run.py", Path(__file__).resolve()),
    ("experiments/common/ppo.py", HERE.parent / "common" / "ppo.py"),
    ("cadence/plasticity.py", Path(cd.__file__).resolve().parent / "plasticity.py"),
    ("cadence/stream.py", Path(cd.__file__).resolve().parent / "stream.py"),
    ("cadence/learning.py", Path(cd.__file__).resolve().parent / "learning.py"),
]
LEVELS = {"easy": 4, "medium": 32, "hard": 64}
SUITS = 4
ENVS = 32
HIDDEN = 32
EVAL_EPISODES = 100
REWARD_SCALE = 10.0  # the per-step reward is 1/(52 - k); scaled so the critic's error is of order one
ACTOR = cd.ActorCriticConfig(gamma=0.99, lam=0.95, eta=0.001, eta_bias=0.0001, eta_critic=0.5, normalize=0.999, momentum=0.9)
CONFIG = cd.LearnerConfig(beta=0.1, eta=0.5, nudge="quadratic", tolerance=1e-3, free_steps=40, nudged_steps=12)


def make_env(level: str) -> Any:
    from popgym.envs.repeat_previous import RepeatPreviousEasy, RepeatPreviousHard, RepeatPreviousMedium

    return {"easy": RepeatPreviousEasy, "medium": RepeatPreviousMedium, "hard": RepeatPreviousHard}[level]()


class Envs:
    """``k`` environments in step; every episode has the same length, so they reset together."""

    def __init__(self, level: str, k: int, seed: int) -> None:
        self.envs = [make_env(level) for _ in range(k)]
        self.seed = seed
        self.obs = np.zeros(k, dtype=np.int64)
        self.t = 0

    def reset(self) -> np.ndarray:
        self.obs = np.array([e.reset(seed=self.seed + i)[0] for i, e in enumerate(self.envs)])
        self.seed += len(self.envs)
        self.t = 0
        return self.obs

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        obs, rew, done = [], [], False
        for e, a in zip(self.envs, actions, strict=True):
            o, r, terminated, truncated, _ = e.step(int(a))
            obs.append(o)
            rew.append(r)
            done = done or terminated or truncated
        self.obs = np.array(obs)
        self.t += 1
        return self.obs, np.array(rew), done


class PatchAgent:
    """Suit, beat and memory owners into a hidden range; four action owners; the critic on the hidden owners."""

    def __init__(self, arm: str, k: int, seed: int, echo: float = 0.5, *, memory_lag: int | None = None) -> None:
        self.arm, self.k = arm, k
        self.memory_lag = k - 1 if memory_lag is None else memory_lag
        if self.memory_lag < 1:
            raise ValueError("the memory lag must be positive")
        beats = self.memory_lag if arm == "fast" else 0
        memory = SUITS if arm == "fast" else 0
        inputs = SUITS + beats + memory
        if arm == "context":
            self.wiring, tie = cd.stateful(inputs, 1, inputs, HIDDEN, SUITS, seed=seed)
        else:
            self.wiring = cd.layered(inputs, HIDDEN, SUITS, density=1.0, seed=seed)
            tie = None
        engine = cd.Settlement(self.wiring, cd.learning_rule(dt=1.0), dense_limit=8192)
        self.learner = cd.Learner(engine, self.wiring.sets["output"], CONFIG, tie_groups=tie)
        self.ac = cd.ActorCritic(self.learner, self.wiring.sets["hidden"], ACTOR, seed=seed)
        self.input = np.asarray(self.wiring.sets["input"])
        self.suit = self.input[:SUITS]
        self.beat = self.input[SUITS : SUITS + beats]
        self.memory = self.input[SUITS + beats : SUITS + beats + memory]
        self.fast = cd.FastSeams(self.beat, self.memory, replace=True) if arm == "fast" else None
        self.echo = cd.Echo(self.wiring, decay=echo) if arm == "context" else None

    def reset(self, batch: int) -> None:
        if self.fast is not None:
            self.fast.reset(batch)
        if self.echo is not None:
            self.echo.reset(batch)

    def drive(self, obs: np.ndarray, t: int) -> np.ndarray:
        d = np.zeros((len(obs), self.wiring.n))
        rows = np.arange(len(obs))
        amplitude = self.ac.learner.engine.rule.clamp_amplitude
        d[rows, self.suit[obs]] = amplitude
        if self.fast is not None:
            d[:, self.beat[t % self.memory_lag]] = amplitude
            self.fast.clamp(d, inplace=True)  # read before the write
        if self.echo is not None:
            if len(self.echo.trace) != len(obs):
                self.echo.reset(len(obs))
            d[:, columns(self.echo.context)] = self.echo.amplitude * self.echo.trace
        return d

    def after(self, state: cd.SettledState, obs: np.ndarray) -> None:
        if self.fast is not None:
            post = np.zeros((len(obs), SUITS))
            post[np.arange(len(obs)), obs] = 1.0
            self.fast.update(state, np.ones(len(obs), dtype=bool), post=post)
        if self.echo is not None:
            self.echo.update(state)

    def parameters(self) -> int:
        return int(self.ac.parameters())


def run_episode(agent: PatchAgent, envs: Envs, learn: bool) -> tuple[float, float]:
    """One episode for every environment in step: the mean return and the mean steps per decision."""
    obs = envs.reset()
    agent.reset(len(obs))
    agent.ac.reset()
    total = np.zeros(len(obs))
    steps: list[int] = []
    done = False
    d = agent.drive(obs, envs.t)
    while not done:
        state = agent.ac.settle(d)  # cached for act; the write needs the settled state
        steps.append(int(state.steps))
        agent.after(state, obs)
        actions = agent.ac.act(d, greedy=not learn)
        obs, reward, done = envs.step(np.asarray(actions))
        d = agent.drive(obs, envs.t)
        if learn:
            agent.ac.learn(reward * REWARD_SCALE, np.full(len(obs), done), d)
        total += reward
    return float(total.mean()), float(np.mean(steps))


def train_patch(arm: str, level: str, seed: int, episodes: int, log: list[dict[str, float]]) -> tuple[PatchAgent, float]:
    k = LEVELS[level]
    agent = PatchAgent(arm, k, seed)
    envs = Envs(level, ENVS, seed)
    t0 = time.perf_counter()
    for episode in range(episodes):
        ret, steps = run_episode(agent, envs, learn=True)
        log.append({"episode": episode, "return": ret, "steps_per_decision": steps})
        if episode % 20 == 0:
            print(f"[{arm} {level}] episode {episode}/{episodes}: return {ret:+.3f}, {steps:.0f} steps per decision ({time.perf_counter() - t0:.0f}s)", flush=True)
    return agent, time.perf_counter() - t0


def evaluate_patch(agent: PatchAgent, level: str, seed: int) -> dict[str, float]:
    envs = Envs(level, EVAL_EPISODES, seed)
    ret, steps = run_episode(agent, envs, learn=False)
    return {"return": ret, "steps_per_decision": steps, "episodes": EVAL_EPISODES}


class WindowEnv(gym.Env):
    """A gym wrapper for PPO: the observation is the last ``window`` suits as one-hots (a memory handed over)."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, level: str, window: int) -> None:
        super().__init__()
        self.env = make_env(level)
        self.window = window
        self.history = np.zeros(window, dtype=np.int64)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (window * SUITS,), dtype=np.float32)
        self.action_space = self.env.action_space

    def _obs(self) -> np.ndarray:
        out = np.zeros((self.window, SUITS), dtype=np.float32)
        out[np.arange(self.window), self.history] = 1.0
        return out.reshape(-1)

    def reset(self, seed: int | None = None, options: Any = None) -> tuple[np.ndarray, dict]:  # type: ignore[override]
        super().reset(seed=seed)
        o, info = self.env.reset(seed=seed)
        self.history[:] = 0
        self.history[-1] = o
        return self._obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:  # type: ignore[override]
        o, r, term, trunc, info = self.env.step(int(action))
        self.history = np.roll(self.history, -1)
        self.history[-1] = o
        return self._obs(), float(r), term, trunc, info

    def close(self) -> None:
        pass


def evaluate_ppo(policy: torch.nn.Module, level: str, window: int, seed: int) -> dict[str, float]:
    returns = []
    for ep in range(EVAL_EPISODES):
        env = WindowEnv(level, window)
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                logits, _ = policy(torch.tensor(obs[None], dtype=torch.float32))
            action = int(logits.argmax(dim=1)[0])
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        returns.append(total)
    return {"return": float(np.mean(returns)), "episodes": EVAL_EPISODES}


def run_ppo(level: str, window: int, seed: int, total_steps: int) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    policy = MLPPolicy(window * SUITS, SUITS, HIDDEN, continuous=False)
    cfg = PPOConfig(total_steps=total_steps, seed=seed)
    t0 = time.perf_counter()
    train_ppo(lambda: WindowEnv(level, window), policy, cfg, continuous=False)
    return policy, {"seconds": time.perf_counter() - t0, "parameters": count_parameters(policy), "window": window}


def run(args: argparse.Namespace) -> dict[str, Any]:
    level, k = args.level, LEVELS[args.level]
    arms: dict[str, Any] = {}
    for arm in args.arms.split(","):
        log: list[dict[str, float]] = []
        agent, seconds = train_patch(arm, level, args.seed, args.episodes, log)
        ev = evaluate_patch(agent, level, args.seed + 10_000)
        arms[arm] = {"parameters": agent.parameters(), "seconds": seconds, "episodes": args.episodes, "training_return_last_20": float(np.mean([r["return"] for r in log[-20:]])), **ev, "log": log[:: max(1, len(log) // 50)]}
        print(f"arm {arm}: held-out return {ev['return']:+.3f}, {ev['steps_per_decision']:.0f} steps per decision, {agent.parameters():,} parameters ({seconds:.0f}s)", flush=True)
    baselines = []
    env = make_env(level)
    episode_length = env.max_episode_length
    env.close()
    ppo_steps = args.episodes * ENVS * episode_length
    for name, window in (("PPO, memoryless MLP", 1), (f"PPO, MLP over a window of {k + 1}", k + 1)):
        policy, info = run_ppo(level, window, args.seed, ppo_steps)
        ev = evaluate_ppo(policy, level, window, args.seed + 10_000)
        baselines.append({"model": name, **info, **ev, "env_steps": ppo_steps})
        print(f"baseline {name}: held-out return {ev['return']:+.3f} ({info['parameters']} parameters, {info['seconds']:.0f}s)", flush=True)
    body = {
        "task": {"environment": f"popgym RepeatPrevious {level}", "k": k, "observation_lag": k - 1, "episode_length": episode_length, "return_range": [-1, 1], "eval_episodes": EVAL_EPISODES},
        "brain": {"hidden": HIDDEN, "envs_in_step": ENVS, "actor": ACTOR.to_dict(), "learner": CONFIG.to_dict(), "reward_scale": REWARD_SCALE, "clock": "the agent's own step count modulo (k - 1), one owner per beat", "memory": "FastSeams from the beats to a memory range, replacing, read before the write"},
        "schedule": {"episodes": args.episodes, "seed": args.seed},
        "arms": arms, "baselines": baselines,
        "boundary": {"learning": "three-factor rule from reward alone; no label", "memory_enters_as_a_clamp": True, "held_out_episodes_with_fresh_seeds": True},
    }
    receipt = cd.Receipt.build("cadence-paper/t1-repeat-previous/v2", body, sources=SOURCES)
    receipt.write(args.output)
    print(f"receipt {args.output} ({receipt.digest[:16]}...)")
    return body


def check(body: dict) -> str | None:
    for name, arm in body["arms"].items():
        if not -1.0 - 1e-12 <= arm["return"] <= 1.0 + 1e-12:
            return f"{name}: the return is out of range"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", choices=list(LEVELS), default="easy")
    parser.add_argument("--arms", default="fast,context,none")
    parser.add_argument("--episodes", type=int, default=300, help="training episodes, each of 32 environments in step")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=SOURCES, check=check)
        print(message)
        return 0 if ok else 1
    if args.pilot:
        args.episodes = 10
    if args.output is None:
        args.output = HERE / f"receipt_{args.level}_aligned.json"
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
