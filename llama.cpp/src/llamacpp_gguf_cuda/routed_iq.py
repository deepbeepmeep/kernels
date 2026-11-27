"""Packed routed expert projections with device-resident addresses and routing.

The pointer table has one int64 gate, up and down address per expert. A zero
address represents an expert computed elsewhere. All allocations follow the
caller's stream. Only the model's SwiGLU with separate gate/up soft caps uses
these entry points.
"""
import ctypes
from functools import cache
import os
from pathlib import Path

import numpy as np
import torch

GATE_TYPES = (16, 17, 18, 19, 22, 29)
DOWN_TYPES = (20, 42)


@cache
def _library():
    lib = ctypes.CDLL(str(Path(__file__).with_name("routed_iq.dll" if os.name == "nt" else "librouted_iq.so")))
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    lib.iq_pack.argtypes = [pointer, pointer, integer, integer, pointer]
    lib.iq_rows.argtypes = [pointer] * 6 + [integer] * 6 + [ctypes.c_float] * 2 + [pointer]
    lib.iq_grouped_gate.argtypes = [pointer] * 6 + [integer] * 6 + [ctypes.c_float] * 2 + [pointer]
    return lib


def require_native():
    """Load the wheel's routed expert library, reporting a loader error immediately."""
    _library()


def _check(status):
    if status:
        raise RuntimeError(f"Routed IQ kernel launch failed ({status})")


def _pack(x, stream):
    if x.dtype not in (torch.float32, torch.bfloat16):
        x = x.float()
    x = x.contiguous()
    blocks = x.numel() // 32
    packed = torch.empty((blocks, 36), dtype=torch.uint8, device=x.device)
    if not blocks:
        return packed
    _check(_library().iq_pack(x.data_ptr(), packed.data_ptr(), blocks, x.dtype == torch.bfloat16, stream))
    return packed


@cache
def _widths(device, grouped):
    prop = torch.cuda.get_device_properties(device)
    if grouped:
        # Consecutive pairs of an expert keep the weight rows hot. Two reduced
        # rows per wave lower the number of resident waves at the Inkling widths.
        return 16, 16
    if torch.version.hip:
        return 32, 16
    # A full warp for the wider gate on Ampere+, two independent rows per warp
    # for the narrow down projection. Turing uses smaller reductions for both.
    return (32 if prop.major >= 8 else 16), 16


class _Pass:
    def __init__(self, x, ids, probs, intermediate, types, caps, mid, zero, grouped=False, use_matrix=False):
        if x.device.type != "cuda" or x.ndim != 2 or ids.ndim != 2 or x.shape[0] != ids.shape[0] or ids.shape != probs.shape:
            raise ValueError("Expected GPU activations [tokens, hidden] and matching routing IDs and probabilities [tokens, topk]")
        if len(caps) != 2 or min(caps) <= 0:
            raise ValueError("Gate and up caps must be positive")
        self.tokens, self.hidden = x.shape
        self.topk = ids.shape[1]
        self.count = self.tokens * self.topk
        self.intermediate, self.types, self.caps = intermediate, types, caps
        if types[0] != types[1] or types[0] not in GATE_TYPES or types[2] not in DOWN_TYPES:
            raise ValueError(f"Unsupported routed IQ formats {types}")
        if self.hidden % 256 or intermediate % (32 if types[2] == 20 else 64):
            raise ValueError("Routed IQ rows must contain whole quantization blocks")
        self.ids = ids.to(device=x.device, dtype=torch.int32).contiguous()
        self.probs = probs.to(device=x.device, dtype=torch.float32).contiguous()
        self.stream = torch.cuda.current_stream(x.device).cuda_stream
        self.x = None if use_matrix else _pack(x, self.stream)
        self.mid = torch.empty((self.count, intermediate), device=x.device, dtype=torch.float32) if mid is None else mid
        if self.mid.shape != (self.count, intermediate) or self.mid.dtype != torch.float32 or self.mid.device != x.device or not self.mid.is_contiguous():
            raise ValueError("Expected contiguous FP32 intermediate scratch on the activation device")
        self.parts = (torch.zeros if zero else torch.empty)((self.count, self.hidden), device=x.device, dtype=torch.float32)
        self.widths = _widths(x.device, grouped)

    def project(self, pointers, pairs=None, tiles=None, first=0):
        if pointers.dtype != torch.int64 or pointers.device != self.mid.device or not pointers.is_contiguous() or pointers.numel() % 3:
            raise ValueError("Expected contiguous device int64 addresses in gate, up and down order")
        count = self.count if pairs is None else pairs.numel()
        if not count:
            return
        ids, probs, table = self.ids.data_ptr(), self.probs.data_ptr(), pointers.data_ptr()
        ptr = pairs.data_ptr() if pairs is not None else 0
        gate, _, down = self.types
        common = (ids, ptr, table, probs)
        if tiles is None:
            _check(_library().iq_rows(self.x.data_ptr(), *common, self.mid.data_ptr(), count, self.hidden, self.intermediate, self.topk, gate, self.widths[0], *self.caps, self.stream))
        else:
            _check(_library().iq_grouped_gate(self.half_input.data_ptr(), ids, ptr, tiles.data_ptr(), table, self.mid.data_ptr(),
                                              first, len(tiles), self.hidden, self.intermediate, self.topk, gate, *self.caps, self.stream))
        mid = _pack(self.mid[:count], self.stream)
        _check(_library().iq_rows(mid.data_ptr(), *common, self.parts.data_ptr(), count, self.intermediate, self.hidden, self.topk, down, self.widths[1], 0., 0., self.stream))

    def finish(self, out=None):
        return torch.sum(self.parts.view(self.tokens, self.topk, self.hidden), dim=1, out=out)


def forward(x, ids, probabilities, pointers, intermediate, types, caps, out=None, mid=None):
    with torch.cuda.device(x.device):
        job = _Pass(x, ids, probabilities, intermediate, types, caps, mid, False)
        job.project(pointers)
        return job.finish(out)


class GroupedPass(_Pass):
    """Run only the routed pairs whose expert weights are available in each stream slot."""
    def __init__(self, x, ids, probabilities, intermediate, types, caps, order, zero=False):
        flat = ids.contiguous().numpy().reshape(-1)
        # A nearly empty matrix tile loses to packed dots. At the production
        # widths on SM120 the crossover is 16-32 routed pairs per expert.
        wave_matrix = bool(torch.version.hip) or torch.cuda.get_device_capability(x.device)[0] >= 7
        use_matrix = wave_matrix and intermediate % 32 == 0 and flat.size >= 24 * max(1, len(order))
        super().__init__(x, ids, probabilities, intermediate, types, caps, None, zero or (flat < 0).any(),
                         grouped=True, use_matrix=use_matrix)
        sorted_pairs = np.argsort(flat, kind="stable")
        first = np.searchsorted(flat[sorted_pairs], np.arange(int(flat.max(initial=-1)) + 2))
        groups = [sorted_pairs[first[e]:first[e + 1]] if e < len(first) - 1 else sorted_pairs[:0] for e in order]
        self.first = np.cumsum([0] + [len(group) for group in groups])
        self.tiles = None
        if use_matrix:
            batches = [[(offset, min(16, end - offset)) for offset in range(int(begin), int(end), 16)] for begin, end in zip(self.first, self.first[1:])]
            self.tile_offsets = np.cumsum([0] + [len(batch) for batch in batches])
            self.tiles = torch.tensor([tile for batch in batches for tile in batch], dtype=torch.int32, device=x.device).reshape(-1, 2)
            self.half_input = x.to(dtype=torch.float16).contiguous()
        ordered = np.concatenate(groups) if groups else np.empty(0, dtype=np.int32)
        self.pairs = torch.from_numpy(ordered.astype(np.int32, copy=False)).to(x.device)

    def run(self, pointers, start, stop):
        begin, end = int(self.first[start]), int(self.first[stop])
        tiles = None if self.tiles is None else self.tiles[self.tile_offsets[start]:self.tile_offsets[stop]]
        self.project(pointers, self.pairs[begin:end], tiles, begin)
