"""Record patches in lockstep on a device: a population of instances, each with many
streams, every settle one batched product.

``PopulationPatch`` is the torch twin of ``RecordPatchNet`` for one-moment paths from rest
(a reading settled once, the way a decision or a school batch reads a patch). It holds
``instances`` independent parameter sets and, for each instance, ``streams`` independent
record stores, so a whole population of brains, each in many worlds, settles, reads,
writes and takes its slow step in a handful of tensor operations per moment. Nothing
crosses an instance, a stream or a moment: the slow step of each instance is the adjoint of
its own one-moment loss (autograd computes it; the NumPy patch's adjoint is the reference
and a parity test holds both to rounding), the records are the delta rule at the code, and
no gradient reaches the store or another patch. ``from_patch`` copies a NumPy patch's
parameters, projection, offsets and running statistics into every instance, which is how
the parity is checked and how a schooled patch is broadcast into a population.

What batches: instances, streams and, through several ``PopulationPatch`` objects, the
patches of a brain. What does not: the moments of one stream, which a temporal patch
settles in order. Memory: the tables are ``instances * streams * cells * outputs`` values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .record_patch import RecordPatchNet


def device_of(name: str | None = None) -> torch.device:
    """The named device, or the best one present: cuda, then mps, then cpu."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PopulationPatch:
    def __init__(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
        *,
        instances: int,
        streams: int,
        seed: int = 0,
        cells: int = 4096,
        active: int = 32,
        record_rate: float = 0.5,
        habituation: float = 1e-5,
        record_bias: float = 0.3,
        slowest: float = 2.0,
        groups: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if groups is not None and (len(groups) != 1 or groups[0] != outputs):
            raise ValueError("groups: one softmax over every output (groups=(outputs,)) or None")
        self.inputs, self.hidden, self.outputs = int(inputs), int(hidden), int(outputs)
        self.P, self.B = int(instances), int(streams)
        self.cells, self.active = int(cells), int(active)
        self.record_rate = float(record_rate)
        self.habituation = float(habituation)
        self.groups: tuple[int, ...] | None = groups
        self.dev = device_of(device) if not isinstance(device, torch.device) else device
        self.dtype = dtype
        # the same construction as the NumPy patch: a fresh one, copied into every instance
        seed_net = RecordPatchNet(
            inputs,
            hidden,
            outputs,
            seed=seed,
            cells=cells,
            active=active,
            record_rate=record_rate,
            habituation=habituation,
            record_bias=record_bias,
            slowest=slowest,
            groups=groups,
        )
        self._load(seed_net)
        self.settles = 0
        self.writes = 0

    # --- construction ------------------------------------------------------------------------

    def _t(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=self.dtype, device=self.dev)

    def _load(self, net: RecordPatchNet) -> None:
        P, B = self.P, self.B
        params = net.parameters()
        self.G = self._t(params["G"]).expand(P, -1, -1).clone()
        self.g = self._t(params["g"]).expand(P, -1).clone()
        self.Bm = self._t(params["B"]).expand(P, -1, -1).clone()
        self.b = self._t(params["b"]).expand(P, -1).clone()
        self.C = self._t(params["C"]).expand(P, -1, -1).clone()
        self.c = self._t(params["c"]).expand(P, -1).clone()
        self.scale = self._t(net._scale)  # the unit of a channel's fluctuation
        self.projection = self._t(net.records.projection)  # (inputs + hidden, cells)
        self.offset = self._t(net.records.offset)
        width = self.inputs + self.hidden
        self.mean = self._t(net.records.mean).expand(P, B, -1).clone()
        seen = float(net.records.seen)
        self.seen = torch.full((P, B), seen, dtype=self.dtype, device=self.dev)
        norm = float(net._input_norm)
        self.input_norm = torch.full((P, B), norm, dtype=self.dtype, device=self.dev)
        self.tables = self._t(net.records.tables["y"]).expand(P, B, -1, -1).clone()
        assert self.mean.shape[-1] == width

    @classmethod
    def from_patch(
        cls,
        net: RecordPatchNet,
        *,
        instances: int,
        streams: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> PopulationPatch:
        """Every instance a copy of ``net``: its parameters, its projection and offsets, its
        running statistics and its tables (one copy per stream)."""
        plain = net.record_writes == "sequential" and net._output_code is None
        if not plain or net.records.averaging:
            raise ValueError("from_patch takes a sequential patch, no output code, no averaging")
        twin = cls.__new__(cls)
        twin.inputs, twin.hidden, twin.outputs = net.inputs, net.hidden, net.outputs
        twin.P, twin.B = int(instances), int(streams)
        twin.cells, twin.active = net.records.cells, net.records.active
        twin.record_rate = float(net.records.rate)
        twin.habituation = float(net.records.habituation)
        twin.groups = tuple(net.groups) if net.groups is not None else None
        twin.dev = device_of(device) if not isinstance(device, torch.device) else device
        twin.dtype = dtype
        twin._load(net)
        twin.settles = 0
        twin.writes = 0
        return twin

    def parameters(self) -> dict[str, torch.Tensor]:
        return {"G": self.G, "g": self.g, "B": self.Bm, "b": self.b, "C": self.C, "c": self.c}

    # --- one moment from rest -----------------------------------------------------------------

    def _context(self, u: torch.Tensor) -> torch.Tensor:
        """(P, B, hidden) from rest: ``h = (1 - l) tanh(B u + b)`` with ``l = sigmoid(g + G u)``."""
        z = torch.tanh(torch.einsum("phn,pbn->pbh", self.Bm, u) + self.b[:, None, :])
        gate = torch.sigmoid(torch.einsum("phn,pbn->pbh", self.G, u) + self.g[:, None, :])
        return (1.0 - gate) * z

    def _slow(self, h: torch.Tensor) -> torch.Tensor:
        drive = torch.einsum("poh,pbh->pbo", self.C, h) + self.c[:, None, :]
        if self.groups is None:
            return drive
        return torch.softmax(drive, dim=-1)

    def _readings(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """What the records read: the input and the scaled context, unit variance per unit."""
        scaled = u * (np.sqrt(self.inputs) / self.input_norm)[..., None]
        return torch.cat([scaled, h * self.scale], dim=-1)

    def code(self, readings: torch.Tensor) -> torch.Tensor:
        """The k-winner code (P, B, cells) of readings (P, B, inputs + hidden)."""
        x = readings - self.mean if self.habituation > 0 else readings
        drive = torch.einsum("pbq,qc->pbc", x, self.projection) + self.offset
        top = torch.topk(drive, self.active, dim=-1)
        values = torch.clamp(top.values, min=0.0)
        norm = values.norm(dim=-1, keepdim=True)
        values = torch.where(norm > 0, values / torch.clamp(norm, min=1e-12), values)
        out = torch.zeros_like(drive)
        out.scatter_(-1, top.indices, values)
        return out

    def read(self, code: torch.Tensor) -> torch.Tensor:
        return torch.einsum("pbc,pbco->pbo", code, self.tables)

    def imagine(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predictions for readings (P, B, inputs): the slow readout, the record read, their
        sum, the context and the code; private, from rest, no learning."""
        u = self._as(inputs)
        with torch.no_grad():
            h = self._context(u)
            slow = self._slow(h)
            code = self.code(self._readings(u, h))
            read = self.read(code)
        self.settles += 1
        return {"output": slow + read, "slow": slow, "read": read, "hidden": h, "code": code}

    def _as(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=self.dtype, device=self.dev)
        if t.shape != (self.P, self.B, self.inputs) and t.shape[-1] != self.inputs:
            raise ValueError(f"inputs must be (instances, streams, {self.inputs})")
        return t.reshape(self.P, self.B, -1)

    # --- learning -------------------------------------------------------------------------------

    def loss(self, slow: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """The slow loss per instance, the mean over its streams: half the mean squared error
        over the outputs for a linear readout (the NumPy patch's convention), the
        cross-entropy for a categorical port."""
        if self.groups is None:
            return 0.5 * ((slow - target) ** 2).mean(dim=-1).mean(dim=-1)
        return -(target * torch.log(torch.clamp(slow, min=1e-12))).sum(dim=-1).mean(dim=-1)

    def observe(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        *,
        rate: float | torch.Tensor = 1.0,
        write: bool = True,
    ) -> dict[str, torch.Tensor]:
        """One moment observed in every stream: the slow parameters of every instance move
        against the adjoint of their own slow loss by ``rate`` (a float or a tensor of
        ``(instances,)``), and the records of every stream take what the readout got wrong,
        against the parameters that made the prediction."""
        u = self._as(inputs)
        t = torch.as_tensor(target, dtype=self.dtype, device=self.dev).reshape(
            self.P, self.B, self.outputs
        )
        names = ("G", "g", "Bm", "b", "C", "c")
        leaves = []
        for name in names:
            v = getattr(self, name).detach().requires_grad_(True)
            setattr(self, name, v)
            leaves.append(v)
        h = self._context(u)
        slow = self._slow(h)
        per_instance = self.loss(slow, t)
        grads = torch.autograd.grad(per_instance.sum(), leaves)
        step = torch.as_tensor(rate, dtype=self.dtype, device=self.dev).reshape(-1)
        if step.numel() == 1:
            step = step.expand(self.P)
        with torch.no_grad():
            residual = t - slow
            code = None
            if write:
                code = self._write(u, h.detach(), residual.detach())
            for name, leaf, grad in zip(names, leaves, grads, strict=True):
                shape = (self.P,) + (1,) * (leaf.dim() - 1)
                setattr(self, name, (leaf - step.reshape(shape) * grad).detach())
        self.settles += 1
        return {
            "loss": per_instance.detach(),
            "slow": slow.detach(),
            "hidden": h.detach(),
            "residual": residual.detach(),
            "code": code if code is not None else torch.zeros(0),
        }

    def _write(self, u: torch.Tensor, h: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """The witnessed reading moves the running statistics; the code then takes the residual
        by the delta rule through its active cells."""
        norms = u.norm(dim=-1)
        self.input_norm = self.input_norm + 0.01 * (torch.clamp(norms, min=1e-6) - self.input_norm)
        readings = self._readings(u, h)
        if self.habituation > 0:
            self.seen = self.seen + 1.0
            step = torch.clamp(1.0 / self.seen, min=self.habituation)[..., None]
            self.mean = self.mean + step * (readings - self.mean)
        code = self.code(readings)
        held = self.read(code)
        move = (residual - held)[..., None, :]
        self.tables = self.tables + self.record_rate * code[..., :, None] * move
        self.writes += 1
        return code

    def clear_records(self) -> None:
        self.tables.zero_()

    # --- population operations -------------------------------------------------------------------

    def inherit(
        self, parents: torch.Tensor, *, sigma: float = 0.0, generator: torch.Generator | None = None
    ) -> None:
        """Every instance becomes a copy of instance ``parents[p]``, with a normal step of
        ``sigma`` times each tensor's spread on the slow parameters when ``sigma > 0``; the
        stores and statistics restart."""
        idx = torch.as_tensor(parents, device=self.dev, dtype=torch.long)
        for name in ("G", "g", "Bm", "b", "C", "c"):
            v = getattr(self, name).detach()[idx].clone()
            if sigma > 0:
                noise = torch.randn(v.shape, generator=generator, dtype=v.dtype).to(self.dev)
                v = v + noise * sigma * torch.clamp(v.std(), min=1e-3)
            setattr(self, name, v)
        self.tables.zero_()
        self.mean.zero_()
        self.seen.zero_()
        self.input_norm.fill_(1.0)

    def state(self) -> dict[str, Any]:
        return {
            "parameters": {k: v.detach().cpu().numpy() for k, v in self.parameters().items()},
            "tables": self.tables.cpu().numpy(),
            "mean": self.mean.cpu().numpy(),
            "seen": self.seen.cpu().numpy(),
            "input_norm": self.input_norm.cpu().numpy(),
            "settles": self.settles,
            "writes": self.writes,
        }

    def memory_bytes(self) -> int:
        size = self.tables.element_size()
        counted = sum(v.numel() for v in self.parameters().values())
        return int((counted + self.tables.numel() + self.mean.numel()) * size)
