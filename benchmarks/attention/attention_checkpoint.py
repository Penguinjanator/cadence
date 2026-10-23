"""Versioned benchmark checkpoints for the coupled attention composition.

The native Brain archive owns the ordinary life state and validation. This
wrapper adds immutable memory boundaries and captured graph identities. A
distinct outer format makes ordinary Brain.load refuse rather than discard
the coupled equations. No executable objects or import names are read from
the archive; the caller supplies the explicit benchmark class.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from cadence import Brain, NeuralGraph
from cadence.blocks import Layout
from cadence.checkpoint import _write
from cadence.generic import BRAIN_FORMAT
from cadence.neuron import Adaptation, NeuronModel
from cadence.receipts import canonical_json

FORMAT = "cadence-attention-brain/1"
_CACHES = ("_pending_boundary", "_action_boundary", "_forecast_boundary")
_LAYOUT = ("starts", "pair_pre", "pair_post", "offset", "edge_index")
_BANK = ("keys", "values", "held", "ages")
_SETTINGS = ("amplitude", "temperature", "recency", "null", "mode", "empty_anchor")


def _integer(value: Any, minimum: int, name: str) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum):
        raise ValueError(f"invalid attention checkpoint {name}")
    return int(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"invalid attention checkpoint {name}")
    return bool(value)


def _class(cls: type[Brain]) -> None:
    if (not isinstance(cls, type) or not issubclass(cls, Brain)
            or cls.__name__ not in ("AttentionBrain", "MultiHeadBrain")
            or not isinstance(getattr(cls, "_graph_type", None), type)
            or not issubclass(cls._graph_type, NeuralGraph)
            or not isinstance(getattr(cls, "_memory_type", None), type)):
        raise ValueError("attention checkpoints require an explicit supported benchmark class")


def _index(value: Any, n: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != 1 or array.dtype.kind not in "iu" or (array < 0).any()
            or (array >= n).any() or len(np.unique(array)) != len(array)):
        raise ValueError(f"invalid attention checkpoint {name}")
    return array.astype(np.int64)


def _ports(brain: Brain, heads: int, copies: int) -> tuple[tuple, tuple]:
    recall = brain.recall
    if recall is None or recall.window != 1:
        raise ValueError("attention checkpoints require a one-event recall composition")
    populations = brain.connectome.populations
    queries, reads = [], []
    for head in range(heads):
        query = populations["query" if head == 0 else f"query_{head}"]
        if head:
            read = populations[f"read_{head}"]
        else:
            read = np.concatenate((recall.recall_neurons, recall.next_neurons))
            if copies > 1:
                read = np.concatenate((read, populations["read_0_extra"]))
        query = _index(query, brain.connectome.n, "query ports")
        read = _index(read, brain.connectome.n, "read ports")
        if (len(query) != copies * recall.address_width
                or len(read) != copies * recall.value_width * (1 + int(recall.next))):
            raise ValueError("attention checkpoint port widths disagree with recall")
        queries.append(query)
        reads.append(read)
    extra = {f"query_{head}" for head in range(1, heads)}
    if {name for name in populations if name.startswith("query_")} != extra:
        raise ValueError("attention checkpoint head count disagrees with the anatomy")
    extra_reads = {f"read_{head}" for head in range(1, heads)}
    if copies > 1:
        extra_reads.add("read_0_extra")
    if {name for name in populations if name.startswith("read_")} != extra_reads:
        raise ValueError("attention checkpoint read count disagrees with the anatomy")
    all_ports = np.concatenate((*queries, *reads))
    if len(np.unique(all_ports)) != len(all_ports):
        raise ValueError("attention checkpoint ports must be disjoint")
    return tuple(queries), tuple(reads)


def _same_connectome(first, second) -> bool:
    return (first.n == second.n and first.label == second.label
            and dict(first.populations) == dict(second.populations)
            and all(np.array_equal(getattr(first, name), getattr(second, name))
                    for name in ("pre", "post", "count", "sign")))


def _validate_custody(brain: Brain) -> None:
    """A saved action keeps its own bank even after a later waited moment."""
    actor = brain.basal_ganglia
    state, graph = actor.state, actor._free_brain
    if (state is None) != (graph is None):
        raise ValueError("attention checkpoint captured free graph and state disagree")
    pending = actor._pending is not None
    bank = getattr(brain, "_pending_boundary", None)
    action = getattr(brain, "_action_boundary", None)
    if pending != (bank is not None) or (pending and bank is not action):
        raise ValueError("attention checkpoint pending action boundary custody disagrees")
    if action is not None and (
        state is None or getattr(action, "streams", None) != len(np.atleast_2d(state.v))
    ):
        raise ValueError("attention checkpoint action boundary streams disagree with its state")


def _validated_layout(saved: dict[str, np.ndarray], connectome) -> Layout:
    arrays = {}
    for name in _LAYOUT:
        array = np.asarray(saved[name])
        if (array.ndim != 1 or array.dtype.kind not in "iu" or (array < 0).any()
                or (array > np.iinfo(np.int64).max).any()):
            raise ValueError("invalid attention checkpoint layout indices")
        arrays[name] = array.astype(np.int64)
    starts, pre, post, offset, edges = (arrays[name] for name in _LAYOUT)
    ranges, pairs = len(starts) - 1, len(pre)
    if (ranges < 1 or starts[0] != 0 or starts[-1] != connectome.n
            or (np.diff(starts) <= 0).any() or post.shape != pre.shape
            or (pre >= ranges).any() or (post >= ranges).any()
            or len(set(zip(pre.tolist(), post.tolist(), strict=True))) != pairs
            or offset.shape != (pairs + 1,) or offset[0] != 0
            or edges.shape != (connectome.synapses,)):
        raise ValueError("invalid attention checkpoint layout shape")
    width = np.diff(starts)
    expected_offset = np.r_[0, np.cumsum(width[pre] * width[post])]
    if not np.array_equal(offset, expected_offset):
        raise ValueError("attention checkpoint layout offsets disagree")
    lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(pre, post, strict=True))}
    rp = np.searchsorted(starts, connectome.pre, side="right") - 1
    rq = np.searchsorted(starts, connectome.post, side="right") - 1
    expected = []
    for edge, (a, b) in enumerate(zip(rp, rq, strict=True)):
        if (int(a), int(b)) not in lookup:
            raise ValueError("attention checkpoint layout omits a synaptic block")
        block = lookup[(int(a), int(b))]
        expected.append(offset[block] + (connectome.pre[edge] - starts[a]) * width[b]
                        + connectome.post[edge] - starts[b])
    if not np.array_equal(edges, expected):
        raise ValueError("attention checkpoint layout disagrees with synapses")
    return Layout(**arrays)


def _capture(brain: Brain) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cls = type(brain)
    _class(cls)
    _validate_custody(brain)
    heads = _integer(getattr(brain, "heads", 1), 1, "heads")
    copies = _integer(getattr(brain, "port_copies", 1), 1, "port copies")
    if cls.__name__ == "AttentionBrain" and (heads != 1 or copies != 1):
        raise ValueError("a single attention brain has one original port pair")
    mode = brain.mode
    anchor = _boolean(brain.empty_anchor, "empty anchor")
    if mode not in ("agreement", "linear", "competitive") or (anchor and mode != "agreement"):
        raise ValueError("invalid attention checkpoint memory equation")
    queries, reads = _ports(brain, heads, copies)
    if hasattr(brain, "head_queries") and any(
        not np.array_equal(a, b) for a, b in zip(brain.head_queries, queries, strict=True)
    ):
        raise ValueError("attention query descriptors disagree with the anatomy")
    if hasattr(brain, "head_reads") and any(
        not np.array_equal(a, b) for a, b in zip(brain.head_reads, reads, strict=True)
    ):
        raise ValueError("attention read descriptors disagree with the anatomy")
    arrays = {}
    boundaries, boundary_ids = [], {}

    def boundary(value):
        if value is None:
            return None
        if id(value) in boundary_ids:
            return boundary_ids[id(value)]
        multi = getattr(cls, "_multi_memory_type", None)
        if type(value) is cls._memory_type:
            kind, members = "single", (value,)
        elif multi is not None and type(value) is multi:
            kind, members = "multi", value.memories
        else:
            raise ValueError("unsupported attention checkpoint memory boundary type")
        if len(members) != heads:
            raise ValueError("attention checkpoint boundary has another head count")
        number = len(boundaries)
        boundary_ids[id(value)] = number
        descriptors = []
        for member in members:
            matches = [i for i, (q, r) in enumerate(zip(queries, reads, strict=True))
                       if np.array_equal(q, member.query) and np.array_equal(r, member.read)]
            if len(matches) != 1:
                raise ValueError("attention checkpoint boundary ports disagree")
            descriptors.append({"port": matches[0],
                                **{name: getattr(member, name) for name in _SETTINGS}})
            if any(not np.array_equal(getattr(members[0], name), getattr(member, name))
                   for name in _BANK):
                raise ValueError("attention heads must share their witnessed boundary")
        if len({member["port"] for member in descriptors}) != heads:
            raise ValueError("attention boundary repeats a physical head")
        for name in _BANK:
            arrays[f"attention/boundary/{number}/{name}"] = np.array(
                getattr(members[0], name), copy=True)
        boundaries.append({"kind": kind, "members": descriptors})
        return number

    graph_ids, graphs = {}, []

    def graph(value):
        if value is None:
            return None
        if id(value) in graph_ids:
            return graph_ids[id(value)]
        if type(value) is not cls._graph_type or value.backend != "cpu":
            raise ValueError("attention checkpoints require the exact coupled CPU graph")
        if not _same_connectome(brain.connectome, value.connectome):
            raise ValueError("captured attention graphs must share the saved anatomy")
        number = len(graphs)
        graph_ids[id(value)] = number
        layout = {name: getattr(value.layout, name) for name in _LAYOUT}
        _validated_layout(layout, value.connectome)
        for name in ("efficacy", "log_gain", "bias"):
            parameter = np.asarray(getattr(value, name))
            size = value.connectome.synapses if name == "efficacy" else value.connectome.n
            if parameter.shape != (size,) or not np.isfinite(parameter).all():
                raise ValueError("invalid captured attention graph parameters")
            arrays[f"attention/graph/{number}/{name}"] = np.array(parameter, copy=True)
        for name in _LAYOUT:
            arrays[f"attention/graph/{number}/layout/{name}"] = np.array(layout[name], copy=True)
        graphs.append({"neuron_model": value.neuron_model.to_dict(),
            "backend": value.backend, "precision": value.precision,
            "dense_limit": _integer(value.dense_limit, 0, "dense limit"),
            "boundary": boundary(value.memory)})
        return number

    current = graph(brain.brain)
    free = graph(brain.basal_ganglia._free_brain)
    if current != 0 or (brain.basal_ganglia.state is None and free is not None):
        raise ValueError("invalid captured attention free graph")
    references = {name: boundary(getattr(brain, name, None)) for name in _CACHES}
    metadata = {"class": cls.__name__, "mode": mode, "heads": heads,
        "port_copies": copies, "empty_anchor": anchor, "graphs": graphs,
        "free_graph": free, "boundaries": boundaries, "boundary_refs": references}
    return metadata, arrays


def save(brain: Brain, path: str | Path) -> Path:
    """Atomically save one coupled life; an incomplete write preserves its predecessor."""
    metadata, extra = _capture(brain)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".attention-checkpoint-", dir=path.parent) as folder:
        native = Brain.save(brain, Path(folder) / "native.npz")
        with np.load(native, allow_pickle=False) as archive:
            data = {name: np.array(archive[name], copy=True) for name in archive.files}
        base = json.loads(str(data["brain"]))
        if base.get("format") != BRAIN_FORMAT:
            raise ValueError("unsupported native Brain checkpoint format")
        if base["free_current"] is not (metadata["free_graph"] == 0):
            raise ValueError("native and coupled free-graph identity disagree")
        data.update(extra)
        data["brain"] = np.array(canonical_json(
            {"format": FORMAT, "base": base, "attention": metadata}))
        return _write(data, path)


def _schema(meta, cls):
    fields = {"class", "mode", "heads", "port_copies", "empty_anchor", "graphs",
              "free_graph", "boundaries", "boundary_refs"}
    if not isinstance(meta, dict) or set(meta) != fields or meta["class"] != cls.__name__:
        raise ValueError("unsupported or incomplete attention checkpoint class metadata")
    heads = _integer(meta["heads"], 1, "heads")
    copies = _integer(meta["port_copies"], 1, "port copies")
    anchor = _boolean(meta["empty_anchor"], "empty anchor")
    if (meta["mode"] not in ("agreement", "linear", "competitive")
            or (anchor and meta["mode"] != "agreement")
            or (cls.__name__ == "AttentionBrain" and (heads != 1 or copies != 1))
            or not isinstance(meta["boundary_refs"], dict)
            or set(meta["boundary_refs"]) != set(_CACHES)):
        raise ValueError("invalid attention checkpoint composition")
    free = meta["free_graph"]
    if free is not None:
        free = _integer(free, 0, "free graph")
    if (free not in (None, 0, 1) or not isinstance(meta["graphs"], list)
            or len(meta["graphs"]) != (2 if free == 1 else 1)
            or not isinstance(meta["boundaries"], list)):
        raise ValueError("invalid attention checkpoint captured graph references")
    return heads, copies, anchor


def load(path: str | Path, *, cls: type[Brain], backend: str = "cpu",
         device: str | None = None, precision: str | None = None) -> Brain:
    """Resume the explicit coupled class after unchanged native life validation.

    Only CPU is supported by this benchmark. Observational diagnostics follow
    native Brain.load and start cleared; all continuation-owned boundaries and
    captured graphs are restored, including a pre-update action's own graph.
    """
    _class(cls)
    if backend != "cpu" or device is not None:
        raise ValueError("attention checkpoint continuation supports CPU only")
    try:
        with np.load(path, allow_pickle=False) as archive:
            data = {name: np.array(archive[name], copy=True) for name in archive.files}
        outer = json.loads(str(data["brain"]))
        if (not isinstance(outer, dict) or set(outer) != {"format", "base", "attention"}
                or outer["format"] != FORMAT):
            raise ValueError("unsupported attention checkpoint format")
        base, meta = outer["base"], outer["attention"]
        if not isinstance(base, dict) or base.get("format") != BRAIN_FORMAT:
            raise ValueError("unsupported native Brain checkpoint format")
        heads, copies, anchor = _schema(meta, cls)
        if base["free_current"] is not (meta["free_graph"] == 0):
            raise ValueError("native and coupled free-graph identity disagree")
        if base["free_steps"] is None and meta["free_graph"] is not None:
            raise ValueError("a captured free graph requires a saved free phase")
        native_data = {name: value for name, value in data.items()
                       if not name.startswith("attention/")}
        native_data["brain"] = np.array(canonical_json(base))
        with tempfile.TemporaryDirectory(prefix="attention-checkpoint-load-") as folder:
            native = _write(native_data, Path(folder) / "native.npz")
            result = Brain.load.__func__(cls, native, backend=backend, precision=precision)
        queries, reads = _ports(result, heads, copies)
        expected = set()

        def array(name):
            expected.add(name)
            return data[name]

        memories = []
        for number, descriptor in enumerate(meta["boundaries"]):
            if (not isinstance(descriptor, dict) or set(descriptor) != {"kind", "members"}
                    or not isinstance(descriptor["members"], list)
                    or len(descriptor["members"]) != heads):
                raise ValueError("invalid attention checkpoint boundary descriptor")
            bank = {name: array(f"attention/boundary/{number}/{name}") for name in _BANK}
            if bank["held"].dtype != np.bool_:
                raise ValueError("attention held masks must be saved as booleans")
            members, used = [], []
            for member in descriptor["members"]:
                if not isinstance(member, dict) or set(member) != {"port", *_SETTINGS}:
                    raise ValueError("invalid attention checkpoint head descriptor")
                port = _integer(member["port"], 0, "boundary port")
                if port >= heads or port in used:
                    raise ValueError("invalid or duplicate attention boundary port")
                used.append(port)
                members.append(cls._memory_type(
                    query=queries[port], read=reads[port], **bank,
                    **{name: member[name] for name in _SETTINGS}))
            if descriptor["kind"] == "single" and heads == 1:
                value = members[0]
            elif descriptor["kind"] == "multi" and hasattr(cls, "_multi_memory_type"):
                value = cls._multi_memory_type(tuple(members))
            else:
                raise ValueError("unsupported attention checkpoint boundary type")
            memories.append(value)

        used_memories = set()

        def memory(reference):
            if reference is None:
                return None
            reference = _integer(reference, 0, "boundary reference")
            if reference >= len(memories):
                raise ValueError("attention checkpoint boundary reference is missing")
            used_memories.add(reference)
            return memories[reference]

        graphs = []
        native_graph = result.brain
        for number, descriptor in enumerate(meta["graphs"]):
            fields = {"neuron_model", "backend", "precision", "dense_limit", "boundary"}
            if not isinstance(descriptor, dict) or set(descriptor) != fields:
                raise ValueError("invalid attention checkpoint graph descriptor")
            if descriptor["backend"] != "cpu" or descriptor["precision"] not in (
                None, "float64", "float32",
            ):
                raise ValueError("unsupported attention checkpoint backend or precision")
            dense = _integer(descriptor["dense_limit"], 0, "dense limit")
            layout = _validated_layout({name: array(f"attention/graph/{number}/layout/{name}")
                                        for name in _LAYOUT}, result.connectome)
            model = dict(descriptor["neuron_model"])
            adaptation = model.pop("adaptation", None)
            rule = NeuronModel(**model, adaptation=Adaptation(**adaptation) if adaptation else None)
            parameters = {name: array(f"attention/graph/{number}/{name}")
                          for name in ("efficacy", "log_gain", "bias")}
            if number == 0 and (rule != native_graph.neuron_model
                    or dense != native_graph.dense_limit
                    or descriptor["precision"] != native_graph.precision
                    or any(not np.array_equal(value, getattr(native_graph, name))
                           for name, value in parameters.items())):
                raise ValueError("coupled current graph disagrees with the native saved learner")
            graph = cls._graph_type(
                result.connectome, rule, backend="cpu", layout=layout,
                dense_limit=dense, precision=descriptor["precision"], **parameters)
            graph.memory = memory(descriptor["boundary"])
            graphs.append(graph)
        if {name for name in data if name.startswith("attention/")} != expected:
            raise ValueError("missing or unrecognized attention checkpoint arrays")
        result.learner.brain = graphs[0]
        result.basal_ganglia._free_brain = (None if meta["free_graph"] is None
                                            else graphs[meta["free_graph"]])
        result.mode, result.empty_anchor = meta["mode"], anchor
        if cls.__name__ == "MultiHeadBrain":
            result.heads, result.port_copies = heads, copies
            for array in (*queries, *reads):
                array.flags.writeable = False
            result.head_queries, result.head_reads = queries, reads
            result.head_attended = None
        for name in _CACHES:
            setattr(result, name, memory(meta["boundary_refs"][name]))
        if used_memories != set(range(len(memories))):
            raise ValueError("unreferenced attention checkpoint boundary")
        _validate_custody(result)
        return result
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("invalid or incomplete attention checkpoint") from error
