"""Carried state as a stimulus: a brain that carries its last equilibrium into the next one.

If settling has a unique attracting fixed point, fully settling the next input
erases the starting state. Multiple attractors and interrupted settling can retain
history, but are not a reliable addressed store. Here a range of *context* neurons,
one per hidden neuron, is stimulated to a leaky trace of the hidden neurons' own activation at
the previous inputs:

    c <- decay * c + (1 - decay) * h

so the state reverberates and fades over about ``1 / (1 - decay)`` inputs rather than
vanishing at once. The context neurons hear nothing (they are a source range of the block
transport, so their product is computed once per settling run) and their synapses into the
hidden neurons learn under the same free/nudged rule as every other synapse. Credit does not
flow back through time: the trace is a stimulus the rule sees, not a path it differentiates.
``stateful`` builds the connectome, ``Echo`` keeps the trace for a batch of streams.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass, field, fields
from typing import Any, NamedTuple

import numpy as np

from ._configuration import real
from .brain import Brain as NeuralGraph
from .brain import BrainState
from .connectome import Connectome
from .learning import LearnedState, embedded

__all__ = [
    "Trace", "Afterglow", "stateful", "Echo", "FastSynapses", "History", "Recall", "columns"
]


def stateful(
    vocabulary: int,
    positions: int,
    dim: int,
    hidden: int,
    outputs: int,
    *,
    seed: int = 0,
    init: float = 1.0,
    context_init: float = 1.0,
) -> tuple[Connectome, np.ndarray]:
    """``embedded`` plus a context range of ``hidden`` neurons connected densely to the hidden ones.

    Neurons: ``positions`` blocks of ``vocabulary`` one-hot inputs, then ``hidden`` context
    neurons, then the embedding, hidden and output neurons of ``embedded``. Sets: ``input``,
    ``context``, ``embedding``, ``hidden``, ``output``. Returns the connectome and the tie groups
    of the embedding (pass to ``Learner(tie_groups=...)``).
    """
    base, base_tie = embedded(vocabulary, positions, dim, hidden, outputs, seed=seed, init=init)
    n_in = positions * vocabulary
    c0 = n_in  # the context range sits right after the inputs
    shift = hidden  # every non-input neuron of the base moves up by the context range

    def moved(index: np.ndarray) -> np.ndarray:
        return np.where(index >= n_in, index + shift, index)

    rng = np.random.default_rng(seed + 1)
    hidden_members = np.asarray(base.populations["hidden"]) + shift
    ctx, hid = np.meshgrid(np.arange(hidden), hidden_members, indexing="ij")
    magnitude = rng.uniform(0.0, 1.0, size=hidden * hidden) * np.sqrt(6.0 / (2 * hidden))
    signs = rng.choice([-1.0, 1.0], size=hidden * hidden) * magnitude * context_init
    pre = np.concatenate([moved(base.pre), c0 + ctx.ravel()])
    post = np.concatenate([moved(base.post), hid.ravel()])
    count = np.concatenate([base.count, np.ones(hidden * hidden)])
    sign = np.concatenate([base.sign, signs])
    tie = np.concatenate([base_tie, np.full(hidden * hidden, -1)])
    populations: dict[str, Iterable[int]] = {
        "input": range(0, n_in),
        "context": range(c0, c0 + hidden),
    }
    populations.update(
        {k: [int(i) + shift for i in v] for k, v in base.populations.items() if k != "input"}
    )
    connectome = Connectome.from_synapses(
        base.n + hidden,
        pre=pre,
        post=post,
        count=count,
        sign=sign,
        populations=populations,
        label=f"stateful:{positions}x{vocabulary}->{dim}->{hidden}(+{hidden} context)->{outputs}",
    )
    key = {(int(a), int(b)): int(g) for a, b, g in zip(pre, post, tie, strict=True)}
    groups = np.array(
        [key[(int(a), int(b))] for a, b in zip(connectome.pre, connectome.post, strict=True)],
        dtype=np.int64,
    )
    return connectome, groups


def columns(index: np.ndarray) -> slice | np.ndarray:
    """A slice when the neurons are one contiguous range, else the index array.

    Reading or writing the columns of a wide batch through a slice is a strided pass;
    through an index array it is a gather that comes back Fortran-ordered (and a scatter
    on write), which costs tens of times more on a batch of thousands of rows.
    """
    if len(index) and np.array_equal(index, np.arange(index[0], index[0] + len(index))):
        return slice(int(index[0]), int(index[0]) + len(index))
    return np.asarray(index, dtype=np.int64)


@dataclass
class Trace:
    """A quantity decaying across moments, kept per stream, entering the next settling run
    as a stimulus: the memory of the moment before.

    Over a range's activation it is the trace of that range (the Echo of the interpretation
    at ``focus`` 0); weighted by how much each neuron moved since the last moment (its share
    of the row's mean movement, to the power ``focus``) it is brightest where the moment
    changed, and of the input neurons it is an afterimage of the picture itself, the memory
    that reads a cue against a static background (``tests/test_child.py``). What had
    to change while settling stays lit; what stood still fades.
    """

    connectome: Connectome
    decay: float = 0.5
    amplitude: float = 1.0  # the stimulus amplitude of the neuron model
    focus: float = 0.0
    source: str = "hidden"  # the range traced
    # the range the trace enters as a stimulus, one neuron per source neuron
    target: str = "context"
    trace: np.ndarray = field(init=False)
    last: np.ndarray = field(init=False)
    cold: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.decay < 1:
            raise ValueError("decay lies in [0, 1)")
        if not np.isfinite([self.amplitude, self.focus]).all() or self.focus < 0:
            raise ValueError("amplitude and focus must be finite; focus must be nonnegative")
        self.glow = np.asarray(self.connectome.populations[self.target], dtype=np.int64)
        self.hidden = np.asarray(self.connectome.populations[self.source], dtype=np.int64)
        if len(self.glow) != len(self.hidden):
            raise ValueError(f"one {self.target} neuron per {self.source} neuron")
        self._glow_columns = columns(self.glow)
        self._hidden_columns = columns(self.hidden)
        self.reset(0)

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Fresh for every stream, or for ``rows`` only (streams that begin again)."""
        if rows is None or len(self.trace) != batch:
            self.trace = np.zeros((batch, len(self.hidden)))
            self.last = np.zeros((batch, len(self.hidden)))
            self.cold = np.ones(batch, dtype=bool)
        else:
            self.trace[rows] = 0.0
            self.last[rows] = 0.0
            self.cold[rows] = True

    def keep(self, rows: np.ndarray) -> None:
        """Keep the traces of ``rows`` only (streams that ended are dropped)."""
        self.trace, self.last, self.cold = self.trace[rows], self.last[rows], self.cold[rows]

    def stimulate(self, drive: np.ndarray) -> np.ndarray:
        """Read the trace into a drive copy; another batch size reads zero context."""
        out = np.array(drive, dtype=float)
        out[:, self._glow_columns] = (
            self.amplitude * self.trace if len(self.trace) == len(out) else 0.0
        )
        return out

    def update(self, state: BrainState) -> None:
        """After the free phase: the trace decays toward the source neurons' activation, each
        weighted by its movement since the last moment when ``focus`` is above zero; a cold
        stream's first moment weighs one."""
        h = self._source_activation(state)
        if len(self.trace) != len(h):
            self.reset(len(h))
        if self.focus:
            moved = np.abs(h - self.last)
            weight = (moved / (moved.mean(axis=1, keepdims=True) + 1e-9)) ** self.focus
            weight[self.cold] = 1.0
            self.trace = self.decay * self.trace + (1.0 - self.decay) * weight * h
        else:
            self.trace = self.decay * self.trace + (1.0 - self.decay) * h
        self.last = h.copy()  # own the previous moment even if the caller reuses state storage
        self.cold[:] = False

    def _source_activation(self, state: BrainState) -> np.ndarray:
        """The source range's activation as a host array. A state that rests on a torch device
        is sliced there and only the slice comes to the host: for an afterimage of a retina
        of thousands of neurons that is the trace's whole cost."""
        device = getattr(state, "device", None)
        s = device.get("s") if isinstance(device, dict) else None
        if s is not None and hasattr(s, "device") and hasattr(s, "cpu"):
            cols = self._hidden_columns
            if isinstance(cols, slice):
                part = s[:, cols]
            else:
                import torch

                part = s[:, torch.as_tensor(np.asarray(cols), device=s.device)]
            return np.ascontiguousarray(part.detach().to("cpu").numpy().astype(float))
        return np.ascontiguousarray(np.atleast_2d(state.activation)[:, self._hidden_columns])

    def ringing(self, floor: float = 0.1) -> np.ndarray:
        """Each source neuron's trace magnitude over the row's mean magnitude, plus ``floor``;
        one for every other neuron: a salience for the eligibility of the synapses
        out of them, so that what is still ringing is what a signal writes through
        (``ActorCritic.salience``)."""
        if not np.isfinite(floor) or floor < 0:
            raise ValueError("floor must be finite and nonnegative")
        out = np.ones((len(self.trace), self.connectome.n))
        magnitude = np.abs(self.trace)
        mean = magnitude.mean(axis=1, keepdims=True) + 1e-9
        out[:, self._hidden_columns] = floor + magnitude / mean
        return out

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "decay": self.decay,
            "amplitude": self.amplitude,
            "focus": self.focus,
            "source": self.source,
            "target": self.target,
            "neurons": int(len(self.glow)),
        }


@dataclass
class Echo(Trace):
    """The Trace of the hidden equilibria at focus 0, into the ``context`` range."""

    def to_dict(self) -> dict[str, float | int | str]:
        return {"decay": self.decay, "amplitude": self.amplitude, "context": int(len(self.glow))}


@dataclass
class Afterglow(Trace):
    """The Trace with the afterglow's defaults: focused, into the ``afterglow`` range."""

    focus: float = 1.0
    target: str = "afterglow"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "decay": self.decay,
            "amplitude": self.amplitude,
            "focus": self.focus,
            "source": self.source,
            "afterglow": int(len(self.glow)),
        }


@dataclass
class Efference(Trace):
    """The corollary discharge: a Trace written from the command the brain issued, not from
    a settled activation.

    One ``efference`` neuron per motor neuron carries the fading one-hot of the actions the
    stream executed, and the next moment reads it through the plastic projection from the
    ``efference`` range into the association region. The working trace copies the settled
    state before the decision, so under identical observations it carries the executed
    action only through whatever margin the motor competition left, and nothing of a
    sampled choice; this trace carries the command itself. The write is fixed, as the
    working trace's is; what the brain does with its own last command is learned.
    """

    source: str = "motor"
    target: str = "efference"

    def update(self, state: BrainState) -> None:
        raise TypeError("the efference copy is written by the issued command; use issue")

    def issue(self, command: np.ndarray) -> None:
        """After the decision: the trace decays toward the ``(batch, motor)`` command, one
        row per stream with ones at the motor neurons of the executed action. ``focus``
        does not apply: a command is not a settled activation with a movement to weigh."""
        command = np.asarray(command, dtype=float)
        if command.ndim != 2 or command.shape[1] != len(self.hidden):
            raise ValueError("a command is one row of motor values per stream")
        if not np.isfinite(command).all():
            raise ValueError("a command must be finite")
        if len(self.trace) != len(command):
            self.reset(len(command))
        self.trace = self.decay * self.trace + (1.0 - self.decay) * command
        self.last = command.copy()
        self.cold[:] = False


@dataclass
class History:
    """The observations as heard, carried at several timescales in a region of their own.

    One block of ``target`` neurons per decay, each block one neuron per ``source`` neuron.
    After each act the block of decay ``d`` moves toward the observation the brain acted on,
    ``h <- d * h + (1 - d) * x``: the values the ``source`` neurons were stimulated with,
    before any settling. The next moment reads ``amplitude * h`` as the stimulus of the
    block's own neurons, so at a decision the history holds the past and the sensory drive
    holds the present. A fast block keeps the last moments apart, a slow one keeps an
    average of many. The write is fixed, as the working trace's is; the association region
    reads the blocks through a plastic projection, so what the past means is learned. This
    is the pattern of the efference copy applied to the senses: in a closed loop the brain
    hears its own output as its next observation.
    """

    connectome: Connectome
    decays: tuple[float, ...]
    amplitude: float = 3.0
    source: str = "sensory"  # the range heard
    target: str = "history"  # one block of source width per decay
    trace: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        decays = tuple(self.decays)
        if not decays or any(
            isinstance(d, (bool, np.bool_))
            or not isinstance(d, (int, float, np.integer, np.floating))
            or not 0 <= d < 1
            for d in decays
        ):
            raise ValueError("decays are one or more real values in [0, 1)")
        self.decays = tuple(float(d) for d in decays)
        if (
            isinstance(self.amplitude, (bool, np.bool_))
            or not isinstance(self.amplitude, (int, float, np.integer, np.floating))
            or not np.isfinite(self.amplitude)
            or self.amplitude < 0
        ):
            raise ValueError("amplitude must be a finite nonnegative real scalar")
        self.amplitude = float(self.amplitude)
        self.glow = np.asarray(self.connectome.populations[self.target], dtype=np.int64)
        self.width = len(self.connectome.populations[self.source])
        if len(self.glow) != len(self.decays) * self.width:
            raise ValueError(f"one block of {self.source} width per decay in {self.target}")
        self._glow_columns = columns(self.glow)
        # the decay of every history neuron: block by block
        self._decay = np.repeat(np.asarray(self.decays), self.width)
        self.reset(0)

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Silence for every stream, or for ``rows`` only (streams that begin again)."""
        if rows is None or len(self.trace) != batch:
            self.trace = np.zeros((batch, len(self.glow)))
        else:
            self.trace[rows] = 0.0

    def keep(self, rows: np.ndarray) -> None:
        """Keep the histories of ``rows`` only (streams that ended are dropped)."""
        self.trace = self.trace[rows]

    def stimulate(self, drive: np.ndarray) -> np.ndarray:
        """Read the history into a drive copy; another batch size reads silence."""
        out = np.array(drive, dtype=float)
        out[:, self._glow_columns] = (
            self.amplitude * self.trace if len(self.trace) == len(out) else 0.0
        )
        return out

    def hear(self, observations: np.ndarray) -> None:
        """After the act: every block decays toward the ``(batch, source)`` observations the
        streams acted on; another batch size begins the streams in silence first."""
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.width:
            raise ValueError(f"observations are one row of {self.width} values per stream")
        if not np.isfinite(x).all():
            raise ValueError("observations must be finite")
        if len(self.trace) != len(x):
            self.reset(len(x))
        heard = np.tile(x, (1, len(self.decays)))
        self.trace = self._decay * self.trace + (1.0 - self._decay) * heard

    def to_dict(self) -> dict[str, Any]:
        return {
            "decays": list(self.decays),
            "amplitude": self.amplitude,
            "source": self.source,
            "target": self.target,
            "neurons": int(len(self.glow)),
        }


class _Moment(NamedTuple):
    """The present moment of every stream as the recall memory addresses it."""

    observed: np.ndarray  # the observations as given, to recognise the moment at its write
    event: np.ndarray  # the observations as the memory keys them: themselves, or their code
    cortex: np.ndarray  # the settled association activity of the moment before
    address: np.ndarray  # the address q, computed before the settle
    key: np.ndarray  # the fixed part of the address, the slot's key once it is written


@dataclass
class Recall:
    """The index memory: each stream's recent moments as slots, addressed before each settle.

    Each moment the brain accepts (an act, a moment ``wait`` senses) is written after its
    settle as one slot per stream (``witness``). The slot's key is ``heard * x``: ``x`` is
    the observation as heard or, with ``code`` above zero, its sparse code, and with
    ``context`` the fading history of the events before it is added. The slot's value is
    the event ``x`` (``value="event"``) or the settled activity of the association region
    (``value="cortex"``), and the slot before it learns its next: the value of the moment
    that followed it, zero until that moment comes. Before each settle the present moment's
    address

        q = heard * x + x W_x + c W_c + b

    (``c`` the settled association activity of the moment before) weighs the stream's
    slots, ``p_i = softmax_i(q . K_i / temperature - recency * age_i)``, and ``stimulate``
    drives the ``target`` region at ``amplitude * sum_i p_i V_i`` and, with ``next``, the
    ``next`` region at ``amplitude * sum_i p_i N_i``. Both regions are sources whose one-way
    plastic synapses into the association region learn by the settle's own contrast, as the
    sensory ones do. With ``W_x``, ``W_c`` and ``b`` at zero the address is the event heard:
    a moment that shares an event with a stored one finds it before any lesson.

    ``learn`` is the address lesson. After a lesson the association's movement between the
    nudged phases, read back through each source region's synapses, is the direction in
    which a recall would have moved the cortex the way the answer needs; each slot scores
    its value and next against it, the moments the address attended to that would have
    answered are named by ``softmax(score / address_temperature + address_prior * log p)``,
    and the map takes the least-mean-squares step ``address * (K* - q)`` on ``[x, c, 1]`` at
    the taught rows, ``K*`` the named moments' keys. ``capacity`` bounds the slots per
    stream, the oldest dropped first. ``reset`` empties streams, and an empty stream reads
    silence. The address of a moment cannot depend on what that moment recalls: a second
    recall takes a second moment.
    """

    connectome: Connectome
    amplitude: float = 3.0
    temperature: float = 0.25
    heard: float = 1.0
    recency: float = 0.25
    capacity: int = 64
    value: str = "event"
    next: bool = True
    address: float = 0.01
    address_temperature: float = 0.1
    address_prior: float = 1.0
    context: float = 0.0
    context_decay: float = 0.5
    code: int = 0
    code_fan_in: int = 32
    code_winners: int = 24
    seed: int = 0
    source: str = "sensory"  # the observations' range
    target: str = "recall"  # the recall region; the next region is ``next``
    writes: int = 0  # slots written over the memory's life, one per stream and moment
    keys: np.ndarray = field(init=False, repr=False)
    values: np.ndarray = field(init=False, repr=False)
    nexts: np.ndarray = field(init=False, repr=False)
    held: np.ndarray = field(init=False, repr=False)
    previous: np.ndarray = field(init=False, repr=False)
    history: np.ndarray = field(init=False, repr=False)
    w_x: np.ndarray = field(init=False, repr=False)
    w_c: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)

    VALUES = ("event", "cortex")
    # the settings ``Brain.retune`` may change in a life, and those that shape the memory
    RUNTIME = (
        "amplitude", "temperature", "heard", "recency", "address", "address_temperature",
        "address_prior", "context", "context_decay",
    )
    CONSTRUCTION = ("capacity", "value", "next", "code", "code_fan_in", "code_winners")

    @classmethod
    def founders(cls) -> dict[str, Any]:
        """The founder value of every setting: the genes of ``Brain.compose(..., recall=True)``."""
        defaults = {item.name: item.default for item in fields(cls)}
        return {name: defaults[name] for name in (*cls.RUNTIME, *cls.CONSTRUCTION)}

    @staticmethod
    def setting(name: str, value: Any) -> Any:
        """One setting, validated and converted; ``name`` is a field of the memory."""
        if name in ("amplitude", "recency", "address", "address_prior", "context"):
            return real(name, value, low=0)
        if name in ("temperature", "address_temperature"):
            number = real(name, value, low=0)
            if number <= 0:
                raise ValueError(f"{name} must be positive")
            return number
        if name == "heard":
            return real(name, value)
        if name == "context_decay":
            number = real(name, value, low=0)
            if number >= 1:
                raise ValueError("context_decay must lie in [0, 1)")
            return number
        if name in ("capacity", "code", "code_fan_in", "code_winners", "seed"):
            low = 0 if name in ("code", "seed") else 1
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < low
            ):
                raise ValueError(f"{name} must be an integer of at least {low}")
            return int(value)
        if name == "value":
            if not isinstance(value, str) or value not in Recall.VALUES:
                raise ValueError(f"value is one of {Recall.VALUES}")
            return value
        if name == "next":
            if not isinstance(value, bool):
                raise ValueError("next must be boolean")
            return value
        raise TypeError(f"unknown recall setting: {name!r}")

    def __post_init__(self) -> None:
        for name in (*self.RUNTIME, *self.CONSTRUCTION, "seed"):
            setattr(self, name, self.setting(name, getattr(self, name)))
        populations = self.connectome.populations
        regions = (self.source, "association", self.target, *(("next",) if self.next else ()))
        for name in regions:
            if name not in populations:
                raise ValueError(f"a recall memory needs a population named {name!r}")
        self.observed: np.ndarray = np.asarray(populations[self.source], dtype=np.int64)
        self.association: np.ndarray = np.asarray(populations["association"], dtype=np.int64)
        self.recall_neurons: np.ndarray = np.asarray(populations[self.target], dtype=np.int64)
        self.next_neurons: np.ndarray = (
            np.asarray(populations["next"], dtype=np.int64) if self.next else np.zeros(0, np.int64)
        )
        inputs = len(self.observed)
        if self.code and (self.code_fan_in > inputs or self.code_winners > self.code):
            raise ValueError("a code needs code_fan_in <= inputs and code_winners <= code")
        self.key_width = self.code or inputs
        self.value_width = self.key_width if self.value == "event" else len(self.association)
        if len(self.recall_neurons) != self.value_width or (
            self.next and len(self.next_neurons) != self.value_width
        ):
            raise ValueError(
                f"the {self.target} and next regions need one neuron per value coordinate "
                f"({self.value_width})"
            )
        self.code_index: np.ndarray = np.zeros((0, self.code_fan_in), dtype=np.int64)
        self.code_weight: np.ndarray = np.zeros((0, self.code_fan_in))
        if self.code:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 7103]))
            self.code_index = np.stack(
                [np.sort(rng.choice(inputs, self.code_fan_in, replace=False))
                 for _ in range(self.code)]
            ).astype(np.int64)
            self.code_weight = rng.standard_normal((self.code, self.code_fan_in)) / np.sqrt(
                self.code_fan_in
            )
        width = self.key_width
        self.w_x = np.zeros((width, width))
        self.w_c = np.zeros((len(self.association), width))
        self.b = np.zeros(width)
        self._recall_columns = columns(self.recall_neurons)
        self._next_columns = columns(self.next_neurons)
        self._association_columns = columns(self.association)
        self._recall_synapses = self._synapses(self.recall_neurons)
        self._next_synapses = self._synapses(self.next_neurons)
        self.last: _Moment | None = None
        self._private = False  # an imagined branch reads the slots and writes none
        self.reset(0)

    # -- layout and per-stream state

    @property
    def neurons(self) -> int:
        """The neurons the memory drives: the recall region, and the next region with ``next``."""
        return int(len(self.recall_neurons) + len(self.next_neurons))

    @property
    def streams(self) -> int:
        return int(len(self.keys))

    @property
    def slots(self) -> int:
        """The slots every stream holds (a stream that began since holds fewer, ``held``)."""
        return int(self.keys.shape[1])

    def ages(self) -> np.ndarray:
        """Moments since each slot was written: ``slots - 1`` for the oldest, 0 the latest."""
        return np.arange(self.slots - 1, -1, -1, dtype=float)

    def _empty(self, batch: int) -> None:
        self.keys = np.zeros((batch, 0, self.key_width))
        self.values = np.zeros((batch, 0, self.value_width))
        self.nexts = np.zeros((batch, 0, self.value_width))
        self.held = np.zeros((batch, 0), dtype=bool)
        self.previous = np.zeros((batch, len(self.association)))
        self.history = np.zeros((0, self.key_width))
        self.last = None

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Empty every stream, or the streams of ``rows`` only (streams that begin again): their
        slots, the cortex of the moment before and the history. The address map is learned
        and stays."""
        if rows is None or self.streams != batch:
            self._empty(batch)
            return
        chosen = np.zeros(batch, dtype=bool)
        chosen[np.asarray(rows)] = True
        if chosen.all():
            self._empty(batch)
            return
        if not chosen.any():
            return
        keep = ~chosen
        # new arrays, never writes into the old ones: a caller may hold them to restore
        held = self.held & keep[:, None]
        keys = np.where(keep[:, None, None], self.keys, 0.0)
        values = np.where(keep[:, None, None], self.values, 0.0)
        nexts = np.where(keep[:, None, None], self.nexts, 0.0)
        # slots no stream holds any more are dropped from the oldest end
        lead = held.any(axis=0)
        first = int(np.argmax(lead)) if lead.any() else held.shape[1]
        self.keys, self.values, self.nexts, self.held = (
            a[:, first:].copy() for a in (keys, values, nexts, held)
        )
        self.previous = np.where(keep[:, None], self.previous, 0.0)
        if len(self.history) == batch:
            self.history = np.where(keep[:, None], self.history, 0.0)
        self.last = None

    def keep(self, rows: np.ndarray) -> None:
        """Keep the memories of ``rows`` only (streams that ended are dropped)."""
        self.keys, self.values, self.nexts, self.held = (
            a[rows] for a in (self.keys, self.values, self.nexts, self.held)
        )
        self.previous = self.previous[rows]
        if len(self.history):
            self.history = self.history[rows]
        self.last = None

    def snapshot(self) -> dict[str, Any]:
        """The per-stream state, to restore after a refused operation; every write replaces
        these arrays rather than changing them."""
        return {
            name: getattr(self, name)
            for name in ("keys", "values", "nexts", "held", "previous", "history")
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Return to a ``snapshot``; the next moment is addressed afresh."""
        for name, value in snapshot.items():
            setattr(self, name, value)
        self.last = None

    def imagined(self) -> Recall:
        """A private copy for an imagined branch: it reads the live slots and writes none; the
        cortex of the moment before and the history follow the branch."""
        branch = copy(self)
        branch.previous = self.previous.copy()
        branch.history = self.history.copy()
        branch.last = None
        branch._private = True
        return branch

    # -- the moment's address and read

    def _port(self, observations: np.ndarray) -> np.ndarray:
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.observed):
            raise ValueError(f"observations are one row of {len(self.observed)} values per stream")
        if not np.isfinite(x).all():
            raise ValueError("observations must be finite")
        return x

    def coded(self, observations: np.ndarray) -> np.ndarray:
        """The ``(batch, inputs)`` observations as the memory keys them: themselves, or with
        ``code`` their sparse code, ``code`` neurons each reading ``code_fan_in`` inputs with
        fixed random weights of variance ``1 / code_fan_in``, the ``code_winners`` largest
        positive sums kept and the code scaled to unit length. The code's storage and work
        follow its neurons and fan-in, whatever the number of inputs."""
        x = np.asarray(observations, dtype=float)
        if not self.code:
            return x
        drive = np.einsum("buf,uf->bu", x[:, self.code_index], self.code_weight)
        top = np.argpartition(-drive, self.code_winners - 1, axis=1)[:, : self.code_winners]
        rows = np.arange(len(x))[:, None]
        out = np.zeros_like(drive)
        out[rows, top] = np.maximum(drive[rows, top], 0.0)
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return np.asarray(out / np.maximum(norm, 1e-12))

    def _moment(self, observed: np.ndarray) -> _Moment:
        """The address of the present moment of each stream, from the cortex of the moment
        before and, with ``context``, the history of the events before it."""
        x = self.coded(observed)
        n = len(x)
        c = (
            self.previous
            if len(self.previous) == n
            else np.zeros((n, len(self.association)))
        )
        key = self.heard * x
        if self.context:
            h = self.history if len(self.history) == n else np.zeros_like(x)
            key = key + self.context * h
        q = key + x @ self.w_x + c @ self.w_c + self.b
        return _Moment(observed, x, c, q, key)

    def weights(self, address: np.ndarray) -> np.ndarray | None:
        """``p`` over each stream's slots for the ``(batch, key)`` addresses, or None when the
        streams hold no slot or are another batch."""
        if not self.slots or self.streams != len(address):
            return None
        logits = np.einsum("bk,bmk->bm", address, self.keys) / self.temperature
        if self.recency:
            logits = logits - self.recency * self.ages()[None, :]
        if self.held.all():
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            return np.asarray(p / p.sum(axis=1, keepdims=True))
        # a stream that began since holds fewer slots; one that holds none reads silence
        logits = np.where(self.held, logits, -np.inf)
        empty = ~self.held.any(axis=1)
        top = np.where(empty, 0.0, logits.max(axis=1))
        p = np.exp(logits - top[:, None])
        total = p.sum(axis=1, keepdims=True)
        return np.asarray(p / np.where(total > 0, total, 1.0))

    def stimulate(self, drive: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """Address the present moment of every stream and read its slots into a drive copy:
        ``amplitude * sum_i p_i V_i`` on the recall region and, with ``next``,
        ``amplitude * sum_i p_i N_i`` on the next region. Streams without slots, or of
        another batch, read silence. The moment is kept for its write and its lesson."""
        moment = self._moment(self._port(observations))
        self.last = moment
        out = np.array(drive, dtype=float)
        p = self.weights(moment.address)
        if p is not None:
            out[:, self._recall_columns] = self.amplitude * np.einsum("bm,bmc->bc", p, self.values)
            if self.next:
                out[:, self._next_columns] = self.amplitude * np.einsum(
                    "bm,bmc->bc", p, self.nexts
                )
        return out

    # -- the write after the settle

    def activity(self, state: BrainState) -> np.ndarray:
        """The association region's settled activation as a host array, ``(batch,
        association)``. A state on a torch device is sliced there, and only the slice comes
        to the host."""
        device = getattr(state, "device", None)
        s = device.get("s") if isinstance(device, dict) else None
        cols = self._association_columns
        if s is not None and hasattr(s, "device") and hasattr(s, "cpu"):
            if isinstance(cols, slice):
                part = s[:, cols]
            else:
                import torch

                part = s[:, torch.as_tensor(np.asarray(cols), device=s.device)]
            return np.ascontiguousarray(part.detach().to("cpu").numpy().astype(float))
        settled = np.atleast_2d(np.asarray(state.activation, dtype=float))
        return np.asarray(settled[:, cols], dtype=float).copy()

    def witness(self, observations: np.ndarray, state: BrainState) -> None:
        """After the settle: the ``(batch, source)`` observations the streams acted on join
        their slots under the moment's key, the slot before learns them as its next, and the
        settled cortex of ``state`` and the history move on to this moment. Another batch
        size begins the streams with empty memories first. An imagined branch moves the
        cortex and the history only."""
        x = self._port(observations)
        moment = self.last
        if moment is None or moment.observed.shape != x.shape or not np.array_equal(
            moment.observed, x
        ):
            moment = self._moment(x)
        cortex = self.activity(state)
        if len(cortex) != len(x):
            raise ValueError("the settled state needs one row per stream")
        if not self._private:
            self._observe(moment.key, moment.event if self.value == "event" else cortex)
            self.writes += len(x)
        self.previous = cortex
        if self.context:
            h = self.history if len(self.history) == len(x) else np.zeros_like(moment.event)
            self.history = self.context_decay * h + moment.event
        self.last = None

    def _observe(self, key: np.ndarray, value: np.ndarray) -> None:
        """Write one moment per stream: the latest slot learns what followed it, a new slot
        holds this moment; beyond ``capacity`` the oldest slot is dropped."""
        if len(key) != self.streams:
            self._empty(len(key))
        nexts = self.nexts
        if self.slots:
            nexts = nexts.copy()
            if self.held.all():
                nexts[:, -1] = value
            else:
                rows = self.held[:, -1]
                nexts[rows, -1] = value[rows]
        keys = np.concatenate([self.keys, key[:, None, :]], axis=1)
        values = np.concatenate([self.values, value[:, None, :]], axis=1)
        nexts = np.concatenate([nexts, np.zeros_like(value)[:, None, :]], axis=1)
        held = np.concatenate([self.held, np.ones((len(key), 1), dtype=bool)], axis=1)
        if keys.shape[1] > self.capacity:
            drop = keys.shape[1] - self.capacity
            keys, values, nexts, held = (a[:, drop:].copy() for a in (keys, values, nexts, held))
        self.keys, self.values, self.nexts, self.held = keys, values, nexts, held

    # -- the address lesson

    def _synapses(self, region: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The synapses from ``region`` into the association region: their indices, and the
        source and association coordinate of each."""
        connectome = self.connectome
        pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
        local_source = -np.ones(connectome.n, dtype=np.int64)
        local_source[region] = np.arange(len(region))
        local_cortex = -np.ones(connectome.n, dtype=np.int64)
        local_cortex[self.association] = np.arange(len(self.association))
        synapses = np.flatnonzero((local_source[pre] >= 0) & (local_cortex[post] >= 0))
        return synapses, local_source[pre[synapses]], local_cortex[post[synapses]]

    def readback(self, graph: NeuralGraph, region: str, movement: np.ndarray) -> np.ndarray:
        """``movement @ W.T`` for the one-way synapses of the ``"recall"`` or ``"next"``
        region into the association region: per source neuron, the sum over its synapses of
        effective weight times the movement of its association neuron."""
        synapses, rows, cols = self._recall_synapses if region == "recall" else self._next_synapses
        width = len(self.recall_neurons) if region == "recall" else len(self.next_neurons)
        w = np.zeros((width, len(self.association)))
        np.add.at(w, (rows, cols), graph._gain_pre[synapses] * graph.efficacy[synapses])
        return np.asarray(movement @ w.T)

    def learn(self, graph: NeuralGraph, learned: LearnedState) -> dict[str, float]:
        """The address lesson after a lesson whose phases are ``learned``, at its taught rows,
        through the synapses of ``graph`` (the learner's brain after the lesson's update).
        Reads the moment the lesson's drive addressed (``stimulate``); returns how many rows
        moved and the mean absolute error of their addresses."""
        quiet = {"address_rows": 0.0, "address_error": 0.0}
        moment = self.last
        if (
            not self.address
            or moment is None
            or learned.free is None
            or learned.nudged is None
            or learned.nudged is learned.free
        ):
            return quiet
        taught = learned.taught
        rows = (
            np.arange(len(moment.address))
            if taught is None
            else np.flatnonzero(np.asarray(taught, dtype=bool))
        )
        if not self.slots or not len(rows) or self.streams != len(moment.address):
            return quiet
        plus = np.atleast_2d(learned.nudged.activation)
        minus = np.atleast_2d(
            learned.free.activation if learned.opposite is None else learned.opposite.activation
        )
        movement = (plus - minus)[:, self.association]
        d_value = self.readback(graph, "recall", movement)
        d_next = self.readback(graph, "next", movement) if self.next else None
        return self._step(moment, rows, d_value, d_next)

    def _step(
        self, moment: _Moment, rows: np.ndarray, d_value: np.ndarray, d_next: np.ndarray | None
    ) -> dict[str, float]:
        """The least-mean-squares step of the address map toward the slots ``d`` names."""
        quiet = {"address_rows": 0.0, "address_error": 0.0}
        values, nexts, keys = self.values[rows], self.nexts[rows], self.keys[rows]
        held = self.held[rows]
        masked = not held.all()
        score = np.einsum("bc,bmc->bm", d_value, values)
        norm = np.sum(d_value * d_value, axis=1)
        if d_next is not None:
            score += np.einsum("bc,bmc->bm", d_next, nexts)
            norm += np.sum(d_next * d_next, axis=1)
        norm = np.sqrt(norm)
        moving = norm > 1e-12
        if masked:
            moving &= held.any(axis=1)  # a stream that began since and holds no slot
        if not moving.any():
            return quiet
        logits = score[moving] / norm[moving, None] / self.address_temperature
        if masked:
            logits = np.where(held[moving], logits, -np.inf)
        if self.address_prior:
            # among the moments the address attended to: the attention is the prior
            q_ = moment.address[rows][moving]
            attention = np.einsum("bk,bmk->bm", q_, keys[moving]) / self.temperature
            if self.recency:
                attention = attention - self.recency * self.ages()[None, :]
            if masked:
                attention = np.where(held[moving], attention, -np.inf)
            attention -= attention.max(axis=1, keepdims=True)
            attention -= np.log(np.exp(attention).sum(axis=1, keepdims=True))
            logits = logits + self.address_prior * attention
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        target = np.einsum("bm,bmk->bk", p, keys[moving])
        x, c, q = (a[rows][moving] for a in (moment.event, moment.cortex, moment.address))
        error = target - q
        n = len(error)
        rate = self.address
        self.w_x += rate * x.T @ error / n
        self.w_c += rate * c.T @ error / n
        self.b += rate * error.mean(axis=0)
        return {"address_rows": float(n), "address_error": float(np.abs(error).mean())}

    def to_dict(self) -> dict[str, Any]:
        return {
            **{name: getattr(self, name) for name in (*self.RUNTIME, *self.CONSTRUCTION)},
            "seed": self.seed,
            "source": self.source,
            "target": self.target,
            "neurons": self.neurons,
            "writes": int(self.writes),
        }


@dataclass
class PatternSeparator:
    """Pattern separation: expand a key into a wider random code and keep the strongest winners.

    ``inputs`` key neurons project through a fixed random matrix onto ``expansion`` code
    neurons; the ``winners`` largest positive entries stay and the rest are zero. Expansion
    can reduce overlap between correlated keys; it does not guarantee separate codes.
    Codes with disjoint winners do not interfere: a delta write changes another key's read
    in proportion to their dot product. ``center`` > 0 keeps a running mean of the keys
    seen by ``observe`` (forgetting factor ``center``) and subtracts it first. Updating that
    mean can move the code of a previously stored key.
    """

    inputs: int
    expansion: int
    winners: int
    seed: int = 0
    center: float = 0.0
    projection: np.ndarray = field(init=False, repr=False)
    mean: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("inputs", "expansion", "winners"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.winners > self.expansion:
            raise ValueError("winners must not exceed the expansion")
        if not 0 <= self.center < 1:
            raise ValueError("center lies in [0, 1)")
        rng = np.random.default_rng(self.seed)
        self.projection = rng.standard_normal((self.inputs, self.expansion)) / np.sqrt(self.inputs)
        self.mean = np.zeros(self.inputs)

    def habituate(self, keys: np.ndarray) -> None:
        """Initialize the running mean from ``keys`` before storing records.

        With positive ``center``, later learning calls continue updating this mean;
        initialization does not freeze the codes used to address stored records.
        """
        x = np.asarray(keys, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.inputs or not np.isfinite(x).all() or not len(x):
            raise ValueError(f"keys must be a finite nonempty (batch, {self.inputs}) array")
        self.mean = (x / len(x)).sum(axis=0)

    def code(self, key: np.ndarray, learn: bool = False) -> np.ndarray:
        """The sparse code of ``(batch, inputs)`` keys: ``(batch, expansion)``, ``winners``
        nonzero entries per row at most. ``learn`` updates the running mean (writes only)."""
        x = np.asarray(key, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.inputs or not np.isfinite(x).all():
            raise ValueError(f"key must be a finite (batch, {self.inputs}) array")
        mean = self.mean
        if self.center > 0.0:
            if learn and len(x):
                mean = self.center * mean + (1.0 - self.center) * (x / len(x)).sum(axis=0)
            with np.errstate(over="ignore", invalid="ignore"):
                x = x - mean
        with np.errstate(over="ignore", invalid="ignore"):
            drive = np.maximum(x @ self.projection, 0.0)
        if not np.isfinite(mean).all() or not np.isfinite(drive).all():
            raise ValueError("pattern separation overflowed; scale the input keys")
        if learn:
            self.mean = mean
        if self.winners >= self.expansion:
            return np.asarray(drive)
        keep = np.argpartition(-drive, self.winners - 1, axis=1)[:, : self.winners]
        out: np.ndarray = np.zeros_like(drive)
        np.put_along_axis(out, keep, np.take_along_axis(drive, keep, axis=1), axis=1)
        return np.asarray(out)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "inputs": self.inputs,
            "expansion": self.expansion,
            "winners": self.winners,
            "seed": self.seed,
            "center": self.center,
        }


@dataclass
class FastSynapses:
    """A bounded associative memory between two ranges, one matrix per stream.

    ``rule="hebb"`` adds an outer product, preserving the original behaviour.
    ``rule="delta"`` reads before writing: for a unit key ``k``,
    ``M += rate * outer(k, value - k @ M)``. Repeated evidence then stops changing
    a correct prediction. A rate-one write corrects this key's unscaled prediction; other
    nonorthogonal keys can interfere. This is the established delta/LMS rule,
    not a guarantee of unlimited capacity or learned memory addressing.

    ``observe``/``recall`` operate directly on key/value ports. ``update``/``read``
    adapt the same operations to settled states and full drives. Key normalisation
    reads the whole key vector; a synapse uses its key coordinate and the post neuron's
    prediction error. Strengths decay once per observation/update, never on reads.
    """

    pre: np.ndarray
    post: np.ndarray
    decay: float = 1.0
    rate: float = 1.0
    amplitude: float = 1.0
    normalize: bool = False  # unit keys and cue, the read divided by the decayed count of writes
    replace: bool = False  # a write clears what its active pre neurons held (a slot; one-hot keys)
    rule: str = "hebb"  # "delta": unit keys, residual writes, no count-averaged read
    separator: PatternSeparator | None = None  # pattern separation of the keys before use
    strength: np.ndarray = field(init=False)
    mass: np.ndarray = field(init=False)  # (batch,) the decayed count of writes, for ``normalize``
    writes: int = 0

    def __post_init__(self) -> None:
        for name in ("pre", "post"):
            raw = np.asarray(getattr(self, name))
            if (
                raw.dtype.kind not in "iuf"
                or not np.isfinite(raw).all()
                or (raw != np.floor(raw)).any()
            ):
                raise ValueError(f"{name} must contain integer neuron indices")
        self.pre = np.asarray(self.pre, dtype=np.int64)
        self.post = np.asarray(self.post, dtype=np.int64)
        for name, neurons in (("pre", self.pre), ("post", self.post)):
            if neurons.ndim != 1 or not len(neurons) or np.any(neurons < 0):
                raise ValueError(f"{name} must be a nonempty vector of neuron indices")
            if len(np.unique(neurons)) != len(neurons):
                raise ValueError(f"{name} neuron indices must be unique")
        if not 0 <= self.decay <= 1:
            raise ValueError("decay lies in [0, 1]")
        if not np.isfinite(self.rate) or self.rate < 0 or not np.isfinite(self.amplitude):
            raise ValueError("rate must be finite and nonnegative; amplitude must be finite")
        if self.rule not in ("hebb", "delta"):
            raise ValueError("rule must be 'hebb' or 'delta'")
        if self.rule == "delta":
            if self.rate > 1:
                raise ValueError("delta rate lies in [0, 1]")
            if self.normalize or self.replace:
                raise ValueError(
                    "delta already normalises keys; normalize/replace are Hebbian modes"
                )
        if self.separator is not None and self.separator.inputs != len(self.pre):
            raise ValueError("the separator's inputs must equal the number of pre neurons")
        self._pre_columns = columns(self.pre)
        self._post_columns = columns(self.post)
        self.strength = np.zeros((0, self.key_width, len(self.post)))
        self.mass = np.zeros(0)

    @property
    def key_width(self) -> int:
        """Rows of the strength matrix: the code width with a separator, else the pre count."""
        return self.separator.expansion if self.separator is not None else int(len(self.pre))

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Clear all streams, or only the selected streams at episode boundaries."""
        if rows is None or len(self.strength) != batch:
            self.strength = np.zeros((batch, self.key_width, len(self.post)))
            self.mass = np.zeros(batch)
        else:
            self.strength[rows] = 0.0
            self.mass[rows] = 0.0

    def keep(self, rows: np.ndarray) -> None:
        self.strength = self.strength[rows]
        self.mass = self.mass[rows]

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        # Preserve the original Hebbian normalisation, including its small-norm floor.
        return np.asarray(x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12))

    @staticmethod
    def _delta_unit(x: np.ndarray) -> np.ndarray:
        # Scale first so even very small/large finite keys have a well-defined direction.
        scale = np.max(np.abs(x), axis=-1, keepdims=True)
        scaled = x / np.where(scale > 0, scale, 1.0)
        norm = np.linalg.norm(scaled, axis=-1, keepdims=True)
        return np.asarray(scaled / np.where(norm > 0, norm, 1.0))

    @staticmethod
    def _port(values: np.ndarray, width: int, name: str) -> np.ndarray:
        out = np.asarray(values, dtype=float)
        if out.ndim != 2 or out.shape[1] != width or not np.isfinite(out).all():
            raise ValueError(f"{name} must be a finite (batch, {width}) array")
        return out

    def recall(self, key: np.ndarray) -> np.ndarray:
        """Read ``(batch, pre)`` keys as ``(batch, post)`` values without changing memory.

        A different batch size reads the baseline without erasing live streams. Delta uses
        unit keys and does not divide by the number of writes. ``amplitude``
        scales the returned drive, not the residual used when learning a value.
        """
        cue = self._port(key, len(self.pre), "key")
        same_batch = len(self.strength) == len(cue)
        strength = self.strength if same_batch else self._baseline()[None, :, :]
        if self.separator is not None:
            cue = self.separator.code(cue)
        if self.rule == "delta":
            cue = self._delta_unit(cue)
        elif self.normalize:
            cue = self._unit(cue)
        out = (cue[:, None, :] @ strength)[:, 0, :]
        if self.normalize:
            mass = self.mass if same_batch else np.zeros(len(cue))
            out = out / np.maximum(mass, 1e-12)[:, None]
        return np.asarray(self.amplitude * out)

    def _baseline(self) -> np.ndarray:
        """The unobserved stream's readback, without changing any stored stream."""
        return np.zeros((self.key_width, len(self.post)))

    def observe(self, key: np.ndarray, value: np.ndarray, write: np.ndarray | None = None) -> None:
        """Decay once, then write selected rows from the key and observed-value ports.

        Inputs are ``(batch, pre)`` and ``(batch, post)``. ``write=None`` writes
        every row; a boolean mask gates individual streams. A zero key cannot
        write a delta association. Validation happens before any state change.
        For prediction tasks call ``recall`` before revealing the new value.
        """
        key = self._port(key, len(self.pre), "key")
        value = self._port(value, len(self.post), "value")
        batch = len(key)
        if len(value) != batch:
            raise ValueError("key and value batches must match")
        gate = np.ones(batch, dtype=bool) if write is None else np.asarray(write)
        if gate.shape != (batch,) or gate.dtype != np.bool_:
            raise ValueError("write must be a boolean vector with one entry per stream")
        same_batch = len(self.strength) == batch
        strength = (
            self.strength * self.decay
            if same_batch
            else np.zeros((batch, self.key_width, len(self.post)))
        )
        mass = self.mass * self.decay if same_batch else np.zeros(batch)
        if self.rule == "delta":
            gate = gate & np.any(key != 0.0, axis=1)
        rows = np.flatnonzero(gate)
        if not len(rows):
            self.strength, self.mass = strength, mass
            return
        a, b = key[rows], value[rows]
        previous_mean = None if self.separator is None else self.separator.mean
        if self.separator is not None:
            a = self.separator.code(a, learn=True)
        if self.rule == "delta":
            a = self._delta_unit(a)
        elif self.normalize:
            a = self._unit(a)
        with np.errstate(over="ignore", invalid="ignore"):
            if self.rule == "delta":
                b = b - (a[:, None, :] @ strength[rows])[:, 0, :]
            elif self.replace:
                strength[rows] *= (a <= 0.0)[:, :, None]
            strength[rows] += self.rate * a[:, :, None] * b[:, None, :]
            mass[rows] += self.rate
        if not np.isfinite(strength).all() or not np.isfinite(mass).all():
            if self.separator is not None and previous_mean is not None:
                self.separator.mean = previous_mean
            raise ValueError("synaptic update overflowed; scale the observed keys and values")
        self.strength, self.mass = strength, mass
        self.writes += len(rows)

    def read(self, drive: np.ndarray) -> np.ndarray:
        """The post neurons' drive from the pre range's stimulus in ``drive``: ``(batch, post)``.

        Normalized Hebbian mode uses unit keys and cue, then divides the read by the
        decayed count of writes. Cosine weights can be signed, so this need not be a
        convex average. ``rate`` and ``amplitude`` also scale the returned values.
        """
        return self.recall(drive[:, self._pre_columns])

    def stimulate(self, drive: np.ndarray, inplace: bool = False) -> np.ndarray:
        """Add the read to the post columns of ``drive`` (a copy, unless ``inplace``)."""
        out = drive if inplace else np.array(drive, dtype=float)
        out[:, self._post_columns] += self.read(out)
        return out

    def update(
        self,
        state: BrainState,
        write: np.ndarray | None = None,
        post: np.ndarray | None = None,
    ) -> None:
        """After settling: strengths fade; the rows in ``write`` add their outer product.

        ``post`` replaces the post neurons' activations for the write, ``(batch, post)``: what
        actually followed (the next symbols read) rather than what the net settled on.
        """
        s = np.atleast_2d(state.activation)
        self.observe(
            s[:, self._pre_columns],
            s[:, self._post_columns] if post is None else post,
            np.zeros(len(s), dtype=bool) if write is None else write,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre": int(len(self.pre)),
            "post": int(len(self.post)),
            "decay": self.decay,
            "rate": self.rate,
            "amplitude": self.amplitude,
            "normalize": self.normalize,
            "replace": self.replace,
            "rule": self.rule,
            "separator": None if self.separator is None else self.separator.to_dict(),
            "writes": self.writes,
        }
