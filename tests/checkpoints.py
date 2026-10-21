"""A brain checkpoint in the layout the reader validates, for the tests that read one.

``Brain.save`` writes ``cadence-brain/1``: one group ``memories/<name>/...`` per memory and the
brain's metadata in the ``brain`` member. ``Brain.load`` validates a file in the layout of
the formats of 0.84.3 (``generic`` metadata, each memory's settings at its top level and its
arrays under the prefixes of those formats, the format number of the latest component), and
reads a file of ``cadence-brain/1`` through that layout. ``earlier`` gives a test that layout:
a test that reads a saved brain, or corrupts one and writes it back, does so in the layout
the reader validates, and the file it writes is read as a file of 0.84.3.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from cadence.generic import _earlier_layout


def earlier(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """A checkpoint's members in the layout the reader validates; a file of that layout
    already is returned as it is."""
    members = {name: arrays[name] for name in arrays}
    if "brain" not in members:
        return members
    meta, data = _earlier_layout(json.loads(str(members["brain"])), members)
    return {**data, "generic": np.array(json.dumps(meta))}


def metadata(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """The brain's metadata in the layout the reader validates."""
    return json.loads(str(earlier(arrays)["generic"]))
