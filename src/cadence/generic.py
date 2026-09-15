"""A generic brain for simple tasks: senses, cortex, a motor choice, dopamine and memory.

``GenericBrain`` composes the standard regions into one connectome that settles as a whole:

* a sensory region: a blank population for a vector observation, or a ``visual_cortex`` for
  an image;
* an association ``cortex`` that every sensory region projects to;
* a ``motor_cortex`` with one neuron per action, connected to the association cortex by
  reciprocal synapse pairs so that a nudge on the choice reaches the cortex;
* a basal-ganglia analogue, an external ``ActorCritic`` whose linear critic reads the
  association cortex and whose
  dopamine prediction error moves every plastic synapse through its eligibility trace;
* optionally a ``prefrontal_cortex`` driven by a ``Trace`` of the association cortex, a
  working memory of the moments before;
* a default hippocampal-memory analogue, ``SynapticMemory`` from sensory to motor
  neurons, recording outcomes quickly and consolidating them through repetition/salience.

``step`` runs one ongoing perceive/feedback/act loop; demonstrations and rewards are
signals in that loop, without a train/eval mode. The lower-level ``fit``, ``act``
and ``learn`` operations expose individual mechanisms for controlled experiments.
This composition does not include a learned world model, hierarchical goals or language.
Its connectome comes from a ``Genome``, so ``evolve`` can select its region sizes/densities, and a
designed region can replace any of them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .brain import Backend, Brain, BrainState
from .connectome import Connectome
from .genome import Genome, Projection, develop
from .learning import Learner, LearnerConfig, learning_neuron_model
from .memory import SynapticMemory
from .plasticity import ActorCritic, ActorCriticConfig
from .regions import Region, motor_cortex, prefrontal_cortex, visual_cortex
from .stream import FastSynapses, Trace

__all__ = ["GenericBrain"]

_REWARD_ARRAYS = (
    "w_critic",
    "trace",
    "trace_bias",
    "trace_critic",
    "second_moment",
    "second_moment_bias",
    "velocity",
    "velocity_bias",
    "salience",
    "_drive",
)


def _learning() -> LearnerConfig:
    # bias-corrected momentum and a moderate step keep a visual pathway from collapsing
    return LearnerConfig(
        beta=0.1, eta=0.5, temperature=0.2, tolerance=3e-3, nudged_steps=12, momentum=0.9
    )


def _reward() -> ActorCriticConfig:
    return ActorCriticConfig(gamma=0.9, lam=0.8, eta=1.0, eta_critic=0.3)


class GenericBrain:
    """Senses, an association cortex, a motor cortex, basal ganglia and consolidating memory.

    ``connectome`` needs populations ``sensory`` (or ``visual/input`` for an image),
    ``association`` and ``motor``, and ``prefrontal`` for a working memory; ``genome`` builds
    one. Use ``GenericBrain.build`` for the default layout.
    """

    def __init__(
        self,
        connectome: Connectome,
        *,
        episodic: bool = True,
        consolidation: float = 0.05,
        working_memory_decay: float = 0.2,
        working_memory_amplitude: float = 3.0,
        learning: LearnerConfig | None = None,
        reward: ActorCriticConfig | None = None,
        seed: int = 0,
        backend: Backend = "cpu",
        device: str | None = None,
    ) -> None:
        populations = connectome.populations
        for name in ("association", "motor"):
            if name not in populations:
                raise ValueError(f"a generic brain needs a population named {name!r}")
        sensory = "sensory" if "sensory" in populations else "visual/input"
        if sensory not in populations:
            raise ValueError("a generic brain needs a 'sensory' or 'visual/input' population")
        self.connectome = connectome
        self.sensory_index = np.asarray(populations[sensory], dtype=np.int64)
        self.motor_index = np.asarray(populations["motor"], dtype=np.int64)
        self.association_index = np.asarray(populations["association"], dtype=np.int64)
        brain = Brain(connectome, learning_neuron_model(dt=1.0), backend=backend, device=device)
        self.learner = Learner(brain, list(self.motor_index), learning or _learning())
        self.basal_ganglia = ActorCritic(
            self.learner, list(self.association_index), reward or _reward(), seed=seed
        )
        self.working_memory: Trace | None = None
        if "prefrontal" in populations:
            self.working_memory = Trace(
                connectome,
                decay=working_memory_decay,
                amplitude=working_memory_amplitude,
                source="association",
                target="prefrontal",
            )
        self.hippocampus: FastSynapses | None = None
        if episodic:
            self.hippocampus = SynapticMemory(
                self.sensory_index, self.motor_index, consolidation=consolidation
            )
        self.rng = np.random.default_rng(seed)
        self._moment: tuple[np.ndarray, np.ndarray] | None = None
        self._prepared: np.ndarray | None = None  # the observations ``learn`` settled
        self.last_learning: dict[str, float] = {}

    @property
    def brain(self) -> Brain:
        """The current learned dynamics; learning can replace this Brain instance."""
        return self.learner.brain

    # -- construction

    @staticmethod
    def genome(
        inputs: int | Sequence[int],
        actions: int,
        *,
        hidden: int = 64,
        density: float = 1.0,
        lateral: float = -0.5,
        working_memory: bool = False,
        memory_scale: float = 12.0,
        features: int = 8,
        field: int = 3,
        seed: int = 0,
    ) -> Genome:
        """The default layout: ``inputs`` is a vector length or an image shape
        ``(height, width)`` or ``(height, width, channels)``."""
        if isinstance(inputs, (int, np.integer)):
            sensory = Region("sensory", inputs)
            forward = Projection("sensory", "association", density=density, reciprocal=False)
        else:
            shape = tuple(inputs)
            if len(shape) not in (2, 3):
                raise ValueError(
                    "image inputs must be (height, width) or (height, width, channels)"
                )
            height, width, *rest = shape
            sensory = visual_cortex(
                height,
                width,
                channels=rest[0] if rest else 1,
                features=features,
                field=field,
                seed=seed,
            )
            forward = Projection("visual", "association", density=density)
        regions = [sensory, Region("association", hidden), motor_cortex(actions, lateral=lateral)]
        projections = [forward, Projection("association", "motor")]
        if working_memory:
            regions.insert(1, prefrontal_cortex(hidden))
            # the trace of a settled cortex is small; a strong projection lets it steer the next
            # moment
            projections.append(
                Projection("prefrontal", "association", scale=memory_scale, reciprocal=False)
            )
        return Genome(tuple(regions), tuple(projections), label="generic-brain")

    @classmethod
    def build(
        cls,
        inputs: int | Sequence[int],
        actions: int,
        *,
        hidden: int = 64,
        density: float = 1.0,
        lateral: float = -0.5,
        working_memory: bool = False,
        memory_scale: float = 12.0,
        episodic: bool = True,
        features: int = 8,
        field: int = 3,
        seed: int = 0,
        **options: Any,
    ) -> GenericBrain:
        """Develop the default genome and wrap it; ``options`` go to the constructor."""
        genome = cls.genome(
            inputs,
            actions,
            hidden=hidden,
            density=density,
            lateral=lateral,
            working_memory=working_memory,
            memory_scale=memory_scale,
            features=features,
            field=field,
            seed=seed,
        )
        return cls(develop(genome, seed=seed), episodic=episodic, seed=seed, **options)

    # -- stimulus

    def _observations(self, observations: Any) -> np.ndarray:
        x = np.asarray(observations, dtype=float)
        if x.ndim < 2 or not x.shape[0] or not np.isfinite(x).all():
            raise ValueError(
                "observations must be a nonempty finite batch; use [observation] for one"
            )
        x = x.reshape(len(x), -1)
        if x.shape[1] != len(self.sensory_index):
            raise ValueError(
                f"observations need {len(self.sensory_index)} values per row, got {x.shape[1]}"
            )
        return x

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        """The drive of a batch of observations: the sensory neurons stimulated, plus the
        working memory and the hippocampal recall when ``memory`` is set."""
        x = self._observations(observations)
        drive = np.zeros((len(x), self.connectome.n))
        drive[:, self.sensory_index] = x * self.brain.neuron_model.stimulus_amplitude
        if memory and self.working_memory is not None:
            drive = self.working_memory.stimulate(drive)
        if memory and self.hippocampus is not None:
            drive = self.hippocampus.stimulate(drive)
        return drive

    # -- learning from labels

    def fit(
        self,
        observations: Any,
        labels: Any,
        *,
        epochs: int = 30,
        batch: int = 32,
    ) -> list[float]:
        """Supervised learning on independent samples; the training accuracy after each epoch."""
        drive = self.stimulus(observations, memory=False)
        labels = self.learner._labels(np.asarray(labels))
        if len(labels) != len(drive):
            raise ValueError("observations and labels must have the same batch size")
        for name, value, minimum in (("epochs", epochs, 0), ("batch", batch, 1)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        # Supervision changes the parameters under any previously cached decision.
        self.reset()
        history = []
        for _ in range(epochs):
            order = self.rng.permutation(len(labels))
            for start in range(0, len(labels), batch):
                rows = order[start : start + batch]
                self.learner.step(drive[rows], labels[rows])
            history.append(self.learner.accuracy(drive, labels))
        return history

    def predict(self, observations: Any) -> np.ndarray:
        """The most active motor neuron for each independent observation."""
        return self.learner.predict(self.stimulus(observations, memory=False))

    def accuracy(self, observations: Any, labels: Any) -> float:
        return self.learner.accuracy(self.stimulus(observations, memory=False), np.asarray(labels))

    # -- ongoing interaction

    def step(
        self,
        observations: Any,
        *,
        reward: Any = None,
        done: Any = None,
        teacher: Any = None,
        salience: Any = None,
        bootstrap: Any = None,
    ) -> np.ndarray:
        """One moment: sense, incorporate real feedback, and choose the next action.

        Reward/done describe the preceding action; an omitted reward means no
        reward event (zero). ``teacher`` labels the current observation. The first
        call has no previous action to reward. Later calls retain the batch's row
        identities; call ``reset`` before starting different streams. Salience is
        nonnegative, per row, and defaults to absolute reward for consolidation.
        There is no training/inference switch. For a frozen measurement use a
        separate instance's ``predict``/greedy ``act``.
        """
        x = self._observations(observations)
        batch = len(x)
        r = np.zeros(batch) if reward is None else np.asarray(reward, dtype=float)
        ended = np.zeros(batch, bool) if done is None else np.asarray(done)
        if r.shape != (batch,) or not np.isfinite(r).all():
            raise ValueError("reward must be a finite vector matching the observation batch")
        if ended.shape != (batch,) or ended.dtype != np.bool_:
            raise ValueError("done must be a boolean vector matching the observation batch")
        importance = SynapticMemory.salience_vector(
            np.abs(r) if salience is None else salience, batch
        )
        labels = None if teacher is None else self.learner._labels(np.asarray(teacher))
        if labels is not None and len(labels) != batch:
            raise ValueError("teacher must match the observation batch")
        if bootstrap is not None:
            bootstrap = np.asarray(bootstrap, dtype=float)
            if bootstrap.shape != (batch,) or not np.isfinite(bootstrap).all():
                raise ValueError("bootstrap must be a finite vector matching the batch")
        pending = self.basal_ganglia._pending is not None
        if not pending and (reward is not None or done is not None or bootstrap is not None):
            raise RuntimeError("feedback needs a preceding action; start with step(observations)")
        # All feedback and demonstrations are checked before changing any state.
        if pending:
            self.last_learning = self.learn(r, ended, x, bootstrap=bootstrap, salience=importance)
        else:
            self.last_learning = {}
        if labels is not None:
            drive = self.stimulus(x)
            self.learner.step(drive, labels)
            # A demonstration teaches the slow policy, not an invented reward value.
            self._prepared = None
            self.basal_ganglia._drive = None
            self.last_learning["demonstrations"] = float(batch)
        return self.act(x)

    # -- lower-level interaction operations

    def act(self, observations: Any, *, greedy: bool = False) -> np.ndarray:
        """Settle on the observations of a batch of streams and choose an action per stream."""
        x = self._observations(observations)
        drive = self.stimulus(x)
        if self._prepared is None or not np.array_equal(self._prepared, x):
            self.basal_ganglia.settle(drive)  # continue each stream from its last state
        self._prepared = None
        action = self.basal_ganglia.act(drive, greedy=greedy)
        state = self.basal_ganglia.state
        if self.working_memory is not None and state is not None:
            self.working_memory.update(state)
        self._moment = None if greedy else (x.copy(), action.copy())
        return action

    def learn(
        self,
        reward: Any,
        done: Any,
        next_observations: Any,
        *,
        bootstrap: Any = None,
        salience: Any = None,
    ) -> dict[str, float]:
        """Dopamine from the reward of the last action; ``done`` rows start a new episode.

        The hippocampus records the reward of the chosen action for the situation it was
        chosen in, and the next choice in that situation reads the record.
        """
        following = self._observations(next_observations)
        # Preflight without reading or resetting memories: invalid input must not write an episode.
        reward, done, _, bootstrap = self.basal_ganglia._validated_transition(
            reward, done, self.stimulus(following, memory=False), bootstrap
        )
        importance = SynapticMemory.salience_vector(
            np.abs(reward) if salience is None else salience, len(following)
        )
        if self.hippocampus is not None and self._moment is not None:
            keys, action = self._moment
            if isinstance(self.hippocampus, SynapticMemory):
                target = np.zeros((len(action), len(self.motor_index)))
                target[np.arange(len(action)), action] = reward
                observed = np.zeros(target.shape, bool)
                observed[np.arange(len(action)), action] = True
                self.hippocampus.observe(keys, target, salience=importance, value_mask=observed)
            else:
                target = self.hippocampus.recall(keys)
                target[np.arange(len(action)), action] = reward
                self.hippocampus.observe(keys, target)
        if self.working_memory is not None and done.any():
            self.working_memory.reset(len(done), rows=np.flatnonzero(done))
        report = self.basal_ganglia.learn(reward, done, self.stimulus(following), bootstrap)
        self._moment = None
        self._prepared = following.copy()
        return report

    def reset(self) -> None:
        """Start every stream afresh; hippocampal records are kept."""
        self.basal_ganglia.reset()
        if self.working_memory is not None:
            self.working_memory.reset(0)
        self._moment = None
        self._prepared = None
        self.last_learning = {}

    def parameters(self) -> int:
        """Actor/critic and consolidated weights; per-stream transient storage is additional."""
        return self.basal_ganglia.parameters() + (
            self.hippocampus.consolidated.size
            if isinstance(self.hippocampus, SynapticMemory)
            else 0
        )

    def save(self, path: str | Path) -> Path:
        """Save the complete composition, including an action awaiting its real outcome.

        Includes critic, optimizer, random generators, working and episodic memories,
        eligibility, the current free phase and any pending action's nudged states.
        ``Learner.save`` remains available for the slow learned response alone.
        """
        from .checkpoint import _learner_data, _write
        from .receipts import canonical_json

        agent = self.basal_ganglia
        data = _learner_data(self.learner)
        metadata: dict[str, Any] = {
            "format": "cadence-generic/2",
            "reward": agent.config.to_dict(),
            "rng": self.rng.bit_generator.state,
            "actor_rng": agent.rng.bit_generator.state,
            "updates": agent.updates,
            "b_critic": agent.b_critic,
            "working_memory": None
            if self.working_memory is None
            else self.working_memory.to_dict(),
            "hippocampus": None if self.hippocampus is None else self.hippocampus.to_dict(),
            "free_steps": None if agent.state is None else agent.state.steps,
            "pending": agent._pending is not None,
            "last_learning": self.last_learning,
        }
        if agent._pending is not None:
            kind, plus, minus, value = agent._pending
            if kind != "states":
                raise ValueError("GenericBrain checkpoints require its standard action states")
            data["pending/value"] = value
            for name, phase in (("plus", plus), ("minus", minus)):
                metadata["pending_" + name + "_steps"] = phase.steps
                for field in ("v", "activation", "adaptation"):
                    data[f"pending/{name}/{field}"] = np.asarray(getattr(phase, field))
            if self._moment is not None:
                data["moment/observations"], data["moment/action"] = self._moment
        for name in _REWARD_ARRAYS:
            value = getattr(agent, name)
            if value is not None:
                data["actor/" + name] = np.asarray(value)
        if agent._trace_device is not None:
            for name, tensor in zip(("trace", "trace_bias"), agent._trace_device, strict=True):
                data["actor/" + name] = tensor.detach().cpu().double().numpy()
        for name in ("mean", "var"):
            data["valence/" + name] = np.asarray(getattr(agent.valence, name))
        if agent.state is not None:
            for name in ("v", "activation", "adaptation"):
                data["free/" + name] = np.asarray(getattr(agent.state, name))
        if self._prepared is not None:
            data["prepared"] = self._prepared
        if self.working_memory is not None:
            for name in ("trace", "last", "cold"):
                data["working/" + name] = getattr(self.working_memory, name)
        if self.hippocampus is not None:
            for name in ("pre", "post", "strength", "mass"):
                data["episodic/" + name] = getattr(self.hippocampus, name)
            if isinstance(self.hippocampus, SynapticMemory):
                data["episodic/consolidated"] = self.hippocampus.consolidated
        data["generic"] = np.array(canonical_json(metadata))
        return _write(data, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        backend: Backend = "cpu",
        device: str | None = None,
        precision: str | None = None,
    ) -> GenericBrain:
        """Resume a complete saved brain, defaulting to portable CPU inference and learning.

        Keep the saved batch row identities, or call ``reset`` for new episodes.
        Cross-backend results are subject to floating-point differences.
        """
        import json

        with np.load(path, allow_pickle=False) as data:
            if "generic" not in data:
                raise ValueError("not a GenericBrain checkpoint; use Learner.load for a learner")
            meta = json.loads(str(data["generic"]))
            if meta.get("format") not in ("cadence-generic/1", "cadence-generic/2"):
                raise ValueError("unsupported GenericBrain checkpoint format")
            learner = Learner.load(path, backend=backend, device=device, precision=precision)
            result = cls(
                learner.brain.connectome, episodic=False, reward=ActorCriticConfig(**meta["reward"])
            )
            result.learner = learner
            agent = result.basal_ganglia
            agent.learner = learner
            for name in _REWARD_ARRAYS:
                if "actor/" + name in data:
                    value = data["actor/" + name].copy()
                    if not np.isfinite(value).all():
                        raise ValueError(f"nonfinite saved actor state: {name}")
                    setattr(agent, name, value)
            agent.updates, agent.b_critic = int(meta["updates"]), float(meta["b_critic"])
            result.rng.bit_generator.state = meta["rng"]
            agent.rng.bit_generator.state = meta["actor_rng"]
            agent.valence.mean = data["valence/mean"].copy()
            agent.valence.var = data["valence/var"].copy()
            if meta["free_steps"] is not None:
                agent._free = BrainState(
                    data["free/v"].copy(),
                    data["free/activation"].copy(),
                    data["free/adaptation"].copy(),
                    int(meta["free_steps"]),
                )
            result._prepared = data["prepared"].copy() if "prepared" in data else None
            result.last_learning = meta.get("last_learning", {})
            if meta.get("pending", False):
                phases = []
                for name in ("plus", "minus"):
                    phases.append(
                        BrainState(
                            data[f"pending/{name}/v"].copy(),
                            data[f"pending/{name}/activation"].copy(),
                            data[f"pending/{name}/adaptation"].copy(),
                            int(meta["pending_" + name + "_steps"]),
                        )
                    )
                agent._pending = ("states", phases[0], phases[1], data["pending/value"].copy())
                if "moment/observations" in data:
                    result._moment = (
                        data["moment/observations"].copy(),
                        data["moment/action"].copy(),
                    )
            working = meta["working_memory"]
            result.working_memory = None
            if working is not None:
                result.working_memory = Trace(
                    result.connectome,
                    **{
                        name: working[name]
                        for name in ("decay", "amplitude", "focus", "source", "target")
                    },
                )
                for name in ("trace", "last", "cold"):
                    setattr(result.working_memory, name, data["working/" + name].copy())
            episodic = meta["hippocampus"]
            if episodic is not None:
                memory_class = (
                    SynapticMemory if episodic.get("kind") == "consolidating" else FastSynapses
                )
                memory_options = (
                    {"consolidation": episodic["consolidation"]}
                    if memory_class is SynapticMemory
                    else {}
                )
                result.hippocampus = memory_class(
                    data["episodic/pre"],
                    data["episodic/post"],
                    **{
                        name: episodic[name]
                        for name in (
                            "decay",
                            "rate",
                            "amplitude",
                            "normalize",
                            "replace",
                            "rule",
                            "writes",
                        )
                    },
                    **memory_options,
                )
                result.hippocampus.strength = data["episodic/strength"].copy()
                result.hippocampus.mass = data["episodic/mass"].copy()
                if isinstance(result.hippocampus, SynapticMemory):
                    weights = data["episodic/consolidated"].copy()
                    if (
                        weights.shape != result.hippocampus.consolidated.shape
                        or not np.isfinite(weights).all()
                    ):
                        raise ValueError("invalid consolidated synaptic weights")
                    result.hippocampus.consolidated = weights
        return result
