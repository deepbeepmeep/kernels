# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE-WanGP.txt)
"""Candidates of an output head from a one-bit copy of it, see _head_candidates.cu: the rows worth an exact score. A drafter's head reads
its whole vocabulary at each draft (Qwen3.8 27B: 98,304 rows of Q6_K, 412 MB, 244 us on an RTX 5090); the signs of its weights and a scale
per row (61 MB) are enough to keep the head's best token among 1,024 rows in 99.9 % of 6,000 real cases; the rows left out hold 0.4 % of
the head's probability on average."""
import ctypes as ct
import functools
from pathlib import Path

import torch

from ._cuda_jit import available, compile_ptx, Kernel

_DTYPES = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
MAX_FEATURES, MAX_GROUP_ROWS = 8192, 1024  # of the kernel's shared memory


@functools.cache
def eligible(device_index):
    return torch.version.hip is None and available() and torch.cuda.get_device_capability(device_index)[0] >= 5


@functools.cache
def _kernels(device, xtype, keep, threads):
    major, minor = torch.cuda.get_device_capability(device)
    with torch.cuda.device(device):
        ptx = compile_ptx(f'#define XTYPE {_DTYPES[xtype]}\n#define KEEP {keep}\n#define THREADS {threads}\n' + Path(__file__).with_suffix('.cu').read_text(), major * 10 + minor)
        return Kernel(ptx, 'head_quantize'), Kernel(ptx, 'head_candidates')


def sign_bits(weight):
    """The one-bit copy of dense rows [N, K] (K a multiple of 128): (bits int32 [N, K / 32], bit j of word b set where weight 32 b + j is
    not negative; scale float32 [N], the row's mean magnitude)."""
    rows, columns = weight.shape
    if columns % 128:
        raise ValueError('The one-bit copy takes rows of a multiple of 128 values.')
    powers = (1 << torch.arange(32, dtype=torch.int64, device=weight.device))
    bits = ((weight.reshape(rows, -1, 32) >= 0).to(torch.int64) * powers).sum(dim=2)
    return torch.where(bits >= 1 << 31, bits - (1 << 32), bits).to(torch.int32), weight.float().abs().mean(dim=1)


def group_bits(bits, groups):
    """The bits of sign_bits, rows in their final order, laid out for the kernel (same shape): within each of `groups` runs of consecutive
    rows, part j (16 bytes, 128 weights) of every row follows part j of the row before, so that a group's rows are read as contiguous runs."""
    rows, words = bits.shape
    if rows % groups or words % 4:
        raise ValueError('The one-bit copy is grouped by whole groups of rows of a multiple of 128 values.')
    return bits.view(groups, rows // groups, words // 4, 4).permute(0, 2, 1, 3).contiguous().view(rows, words)


@torch.library.custom_op('llamacpp_gguf::head_candidates', mutates_args=())
def candidates(input: torch.Tensor, bits: torch.Tensor, row_scale: torch.Tensor, ids: torch.Tensor, groups: int, keep: int) -> torch.Tensor:
    """input [rows, K]; row_scale: the scales of sign_bits for the head's rows in any order, bits: their bits through group_bits(bits,
    groups); ids int32 [N]: the output index of each row. The rows are taken by groups of N / groups consecutive ones: int32
    [rows, groups * keep], the ids of each group's `keep` best rows."""
    x = input.reshape(-1, input.shape[-1])
    if x.stride(1) != 1:
        x = x.contiguous()
    rows, columns = x.shape
    count, group_rows = bits.shape[0], bits.shape[0] // groups
    if (columns % 128 or columns > MAX_FEATURES or bits.shape[1] * 32 != columns or count % groups or not keep <= group_rows <= MAX_GROUP_ROWS or bits.dtype != torch.int32
            or row_scale.dtype != torch.float32 or ids.dtype != torch.int32 or not (bits.is_contiguous() and row_scale.is_contiguous() and ids.is_contiguous())
            or row_scale.numel() != count or ids.numel() != count or x.dtype not in _DTYPES):
        raise ValueError('head_candidates: unsupported shapes or types')
    out = torch.empty((rows, groups * keep), dtype=torch.int32, device=x.device)
    if rows:
        planes = torch.empty((rows, columns // 64 * 6), dtype=torch.int64, device=x.device)  # per 64 values: 3 magnitude bits of the positive and of the negative ones
        d = torch.empty((rows, 2 * (columns // 64)), dtype=torch.float32, device=x.device)
        with torch.cuda.device(x.device):
            threads = min(512, -(-group_rows // 32) * 32)  # a row per thread up to 512 rows a group
            quantize, select = _kernels(x.device.index, x.dtype, keep, threads)
            quantize.launch(((columns // 64 + 7) // 8, rows), 256, [ct.c_void_p(x.data_ptr()), ct.c_longlong(x.stride(0)), ct.c_void_p(planes.data_ptr()), ct.c_void_p(d.data_ptr()), ct.c_int(columns)])
            select.launch((groups, rows), threads, [ct.c_void_p(bits.data_ptr()), ct.c_void_p(row_scale.data_ptr()), ct.c_void_p(ids.data_ptr()), ct.c_void_p(planes.data_ptr()), ct.c_void_p(d.data_ptr()),
                                                ct.c_void_p(out.data_ptr()), ct.c_int(columns), ct.c_int(group_rows)])
    return out


@candidates.register_fake
def _candidates_fake(input, bits, row_scale, ids, groups, keep):
    return input.new_empty((input.reshape(-1, input.shape[-1]).shape[0], groups * keep), dtype=torch.int32)
