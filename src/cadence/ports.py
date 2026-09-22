"""Input ports of a record patch: how the inputs reach the context channels.

The dense port is the record patch as published: every context channel reads every input
through its own row. A structured port reads a declared layout of the inputs through
blocks: a *map* block is a tied local kernel over a grid of the input (a convolution, the
same kernel at every position, so a thing is the same thing wherever it appears), whose
output is a grid of channels laid out as retinotopic maps; a *dense* block is a plain
matrix over a slice of the inputs. The patch's equations do not change: the port is the
linear map ``u -> B u`` (and ``u -> G u`` for the gate), and the adjoint scan needs its
transpose and its parameter gradient, which every block supplies. Blocks are the genes a
genome sets: which slice, what grid, how many channels, what kernel, what stride.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class MapBlock:
    """A tied local kernel over a ``(channels_in, height, width)`` grid of the inputs."""

    start: int  # first input of the grid (channel-major, row-major)
    channels_in: int
    height: int
    width: int
    channels_out: int
    kernel: int
    stride: int = 1

    @property
    def inputs(self) -> int:
        return self.channels_in * self.height * self.width

    @property
    def out_height(self) -> int:
        return (self.height - self.kernel) // self.stride + 1

    @property
    def out_width(self) -> int:
        return (self.width - self.kernel) // self.stride + 1

    @property
    def outputs(self) -> int:
        return self.channels_out * self.out_height * self.out_width

    @property
    def weights(self) -> tuple[int, ...]:
        return (self.channels_out, self.channels_in, self.kernel, self.kernel)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "map", "start": self.start, "channels_in": self.channels_in, "height": self.height, "width": self.width, "channels_out": self.channels_out, "kernel": self.kernel, "stride": self.stride}


@dataclass(frozen=True)
class DenseBlock:
    """A plain matrix over a slice of the inputs."""

    start: int
    inputs: int
    outputs: int

    @property
    def weights(self) -> tuple[int, ...]:
        return (self.outputs, self.inputs)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "dense", "start": self.start, "inputs": self.inputs, "outputs": self.outputs}


Block = MapBlock | DenseBlock


def block_from_dict(d: dict[str, Any]) -> Block:
    d = dict(d)
    kind = d.pop("kind")
    return MapBlock(**d) if kind == "map" else DenseBlock(**d)


class StructuredPort:
    """``B u`` as a sum of blocks, each reading its slice of ``u`` into its slice of the output.

    ``weights`` is a list of arrays, one per block, in the block order. The port is linear
    in ``u`` and in each block's weights; ``apply``, ``transpose`` and ``gradient`` are the
    three operations the adjoint scan needs."""

    def __init__(self, inputs: int, blocks: list[Block]) -> None:
        self.inputs = int(inputs)
        self.blocks = list(blocks)
        for b in self.blocks:
            if b.start < 0 or b.start + b.inputs > self.inputs:
                raise ValueError("a block reads outside the inputs")
        self.outputs = int(sum(b.outputs for b in self.blocks))
        self._offsets = np.cumsum([0] + [b.outputs for b in self.blocks])

    def initial(self, rng: np.random.Generator, scale: float = 1.0) -> list[np.ndarray]:
        """Fan-in scaled draws per block (zero with ``scale`` 0, for a gate port)."""
        out = []
        for b in self.blocks:
            fan = b.inputs if isinstance(b, DenseBlock) else b.channels_in * b.kernel * b.kernel
            out.append(rng.normal(size=b.weights) * (scale / np.sqrt(fan)) if scale else np.zeros(b.weights))
        return out

    # ------------------------------------------------------------------ the three maps
    def apply(self, u: np.ndarray, weights: list[np.ndarray]) -> np.ndarray:
        """``(..., inputs) -> (..., outputs)``."""
        lead = u.shape[:-1]
        parts = []
        for b, w in zip(self.blocks, weights, strict=True):
            x = u[..., b.start : b.start + b.inputs]
            if isinstance(b, DenseBlock):
                parts.append(x @ w.T)
            else:
                parts.append(self._conv(x.reshape(*lead, b.channels_in, b.height, b.width), b, w).reshape(*lead, -1))
        return np.concatenate(parts, axis=-1)

    def transpose(self, v: np.ndarray, weights: list[np.ndarray]) -> np.ndarray:
        """``(..., outputs) -> (..., inputs)``: the gradient with respect to the inputs."""
        lead = v.shape[:-1]
        out = np.zeros((*lead, self.inputs))
        for k, (b, w) in enumerate(zip(self.blocks, weights, strict=True)):
            y = v[..., self._offsets[k] : self._offsets[k + 1]]
            if isinstance(b, DenseBlock):
                out[..., b.start : b.start + b.inputs] += y @ w
            else:
                out[..., b.start : b.start + b.inputs] += self._conv_transpose(y.reshape(*lead, b.channels_out, b.out_height, b.out_width), b, w).reshape(*lead, -1)
        return out

    def gradient(self, v: np.ndarray, u: np.ndarray) -> list[np.ndarray]:
        """The gradient of ``sum(v * (B u))`` with respect to each block's weights, summed over
        every leading axis (batch and time)."""
        n = int(np.prod(u.shape[:-1])) if u.ndim > 1 else 1
        uf, vf = u.reshape(n, -1), v.reshape(n, -1)
        out = []
        for k, b in enumerate(self.blocks):
            y = vf[:, self._offsets[k] : self._offsets[k + 1]]
            x = uf[:, b.start : b.start + b.inputs]
            if isinstance(b, DenseBlock):
                out.append(y.T @ x)
            else:
                patches = self._patches(x.reshape(n, b.channels_in, b.height, b.width), b)  # (n, oh, ow, cin, k, k)
                yy = y.reshape(n, b.channels_out, b.out_height, b.out_width)
                out.append(np.einsum("nohw,nhwikl->oikl", yy, patches))
        return out

    # ------------------------------------------------------------------ convolution
    @staticmethod
    def _patches(x: np.ndarray, b: MapBlock) -> np.ndarray:
        """``(..., cin, H, W) -> (..., oh, ow, cin, k, k)`` windows at the stride."""
        windows = sliding_window_view(x, (b.kernel, b.kernel), axis=(-2, -1))  # (..., cin, H-k+1, W-k+1, k, k)
        windows = windows[..., :: b.stride, :: b.stride, :, :]
        return np.moveaxis(windows, -5, -3)  # (..., oh, ow, cin, k, k)

    def _conv(self, x: np.ndarray, b: MapBlock, w: np.ndarray) -> np.ndarray:
        patches = self._patches(x, b)
        return np.einsum("...hwikl,oikl->...ohw", patches, w)

    @staticmethod
    def _conv_transpose(y: np.ndarray, b: MapBlock, w: np.ndarray) -> np.ndarray:
        """Scatter each output's kernel back onto the input grid (the correlation's adjoint)."""
        lead = y.shape[:-3]
        out = np.zeros((*lead, b.channels_in, b.height, b.width))
        contributions = np.einsum("...ohw,oikl->...hwikl", y, w)  # (..., oh, ow, cin, k, k)
        for i in range(b.kernel):
            for j in range(b.kernel):
                rows = slice(i, i + b.stride * b.out_height, b.stride)
                cols = slice(j, j + b.stride * b.out_width, b.stride)
                out[..., :, rows, cols] += np.moveaxis(contributions[..., i, j], -1, -3)
        return out

    # ------------------------------------------------------------------ custody
    def to_dict(self) -> dict[str, Any]:
        return {"inputs": self.inputs, "blocks": [b.to_dict() for b in self.blocks]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StructuredPort:
        return cls(int(d["inputs"]), [block_from_dict(b) for b in d["blocks"]])

    def dense_matrix(self, weights: list[np.ndarray]) -> np.ndarray:
        """The equivalent ``(outputs, inputs)`` matrix, for tests and small ports."""
        eye = np.eye(self.inputs)
        return self.apply(eye, weights).T
