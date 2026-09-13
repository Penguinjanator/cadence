"""Optional CSR transport without a batch-by-overlap message array.

Rows are receiving owners and columns are publishing owners. The wiring already
orders overlaps by (post, pre), so its topology needs no sorting or dense matrix.
SciPy is part of ``cadence-net[fast]``; without it settlement keeps its NumPy sum.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .wiring import Wiring


@lru_cache(maxsize=1)
def _csr_type() -> Any:
    try:
        from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
    except ImportError:
        return None
    return csr_matrix


def transport(wiring: Wiring, weights: np.ndarray) -> Any:
    """A sparse matrix sharing current weights and cached immutable topology, or None.

    Each settlement owns its matrix wrapper: replacing parameters on a copied
    settlement must not replace the old settlement's data. In-place changes to a
    weights array remain visible through SciPy's shared data view.
    """
    constructor = _csr_type()
    if constructor is None:
        return None
    topology = wiring.__dict__.get("_csr_topology")
    if topology is None:
        dtype = np.int32 if max(wiring.n, wiring.edges) <= np.iinfo(np.int32).max else np.int64
        indices = wiring.pre.astype(dtype, copy=False)
        indptr = np.empty(wiring.n + 1, dtype=dtype)
        indptr[0] = 0
        np.cumsum(np.bincount(wiring.post, minlength=wiring.n), out=indptr[1:])
        topology = indices, indptr
        wiring.__dict__["_csr_topology"] = topology
    indices, indptr = topology
    return constructor((weights, indices, indptr), shape=(wiring.n, wiring.n), copy=False)
