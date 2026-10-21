"""The memories of a composed brain, under one contract.

A composed brain's memories are the members of ``brain.memories``: an ordered, read-only
mapping from a name to the memory, present memories only, in the order the brain reads and
writes them (``working_memory``, ``efference``, ``history``, ``recall``, ``hippocampus``,
``episodic``). Every member keeps one contract (``Memory``):

- ``stimulus(drive, moment)``: the drive of a moment before the settle with what the memory
  adds to it; a memory that adds nothing returns the drive. ``Moment`` gives the moment's
  observations, their code, the key the associative store reads under and whether the
  stores read into the drive at all.
- ``branch()``: a private copy for an imagined branch: it reads the live state and writes
  none of it.
- ``begin(keep=False)``: the streams begin afresh. What the memory keeps across streams
  stays: the associative records, and with ``keep`` the recall memory's slots.
- ``reset(batch, rows=None)`` and ``keep(rows)``: fresh streams, all of them or ``rows``;
  the state of ``rows`` kept and the other streams dropped.
- ``to_dict()`` and ``arrays()``: its settings and counts, and the arrays a checkpoint keeps
  of it (under ``memories/<name>/``).

How a memory is written after the settle follows its own inputs: the working trace moves
toward the settled state, the efference copy takes the issued command, the heard history the
observation, the recall memory witnesses the moment, the stores record outcomes. The brain
writes them in the order of the list.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, NamedTuple, Protocol

import numpy as np

__all__ = ["Memories", "Memory", "Moment"]


class Moment(NamedTuple):
    """What a memory reads of a moment before its settle."""

    observations: np.ndarray  # the observations, one row per stream
    code: np.ndarray  # their code: the observations themselves without a code
    place: np.ndarray | None  # the key the associative store reads; None reads the drive
    stores: bool = True  # whether the stores read into the drive (not under an episodic vote)


class Memory(Protocol):
    """The contract every member of ``brain.memories`` keeps."""

    def stimulus(self, drive: np.ndarray, moment: Moment) -> np.ndarray: ...

    def branch(self) -> Memory: ...

    def begin(self, keep: bool = False) -> None: ...

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None: ...

    def keep(self, rows: np.ndarray) -> None: ...

    def to_dict(self) -> dict[str, Any]: ...

    def arrays(self) -> dict[str, np.ndarray]: ...


class Memories(Mapping[str, Any]):
    """The present memories of a brain by name, in the order the brain reads them."""

    ORDER = ("working_memory", "efference", "history", "recall", "hippocampus", "episodic")

    def __init__(self, members: Mapping[str, Any]) -> None:
        unknown = set(members) - set(self.ORDER)
        if unknown:
            raise ValueError(f"unknown memory: {sorted(unknown)[0]!r}")
        self._members = {
            name: members[name]
            for name in self.ORDER
            if name in members and members[name] is not None
        }

    def __getitem__(self, name: str) -> Any:
        return self._members[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def __repr__(self) -> str:
        return f"Memories({', '.join(self._members)})"

    def stimulus(self, drive: np.ndarray, moment: Moment) -> np.ndarray:
        """The drive with what every member adds, in order."""
        for memory in self._members.values():
            drive = memory.stimulus(drive, moment)
        return drive
