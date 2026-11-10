# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE-WanGP.txt)
"""IQ3_S and IQ2_S linear for 1-8 activation rows, see _iq_rows.cu: each weight block is decoded once for all rows, where llama.cpp's
MMVQ decodes it once per row, so that the rows of a speculative verification cost little more than one (RTX 5090, weights read
cache-cold, IQ3_S on 1 / 5 / 8 rows: a 34816 x 5120 weight 52 / 71 / 99 us with MMVQ, 55 / 59 us on 5 / 8 rows with the tensor-core
variant; a 5120 x 17408 weight 27 / 39 / 52 us, then 31 / 32 us). Its activation scales stay in FP32 and IQ2_S's half scales are exact:
no less accurate than MMVQ. Which variant serves a shape and a row count is a measured decision (set_decision, as the callers do for
the Q4_K tensor-core path); without one, what was measured on compute capability 12.0."""
import ctypes as ct
import functools
import os
from pathlib import Path
from typing import List, Optional

import torch

from ._cuda_jit import available, compile_ptx, Kernel
from ._tuning import LaunchTuner

QTYPES = {'IQ3_S': (3, 110), 'IQ2_S': (2, 82)}  # kernel variant, bytes of a 256-value block
_DTYPES = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_CONFIGS = ((2, 4, 1), (1, 4, 1), (2, 2, 1), (4, 4, 1), (1, 8, 1), (2, 8, 1), (4, 2, 1))  # DP4A kernel: (output features per warp, warps per block)
_MMA = {'mma': (0, 1, 1), 'mma2': (0, 2, 1), 'mma2x2': (0, 2, 2)}  # the tensor-core kernel (blocks of 4 warps): (0, staging buffers, 16-row tiles of a block)
REVISION = 4  # of the kernels: callers key their measured decisions with it
VARIANTS = ('dp4a', *_MMA)
decisions = {}  # (qtype, rows, out features, in features) -> a variant or '' (the installed kernel)
_TUNER = LaunchTuner('wangp_iq_rows', 2, 'GGUF IQ Rows')
MAX_ROWS = 8  # above, llama.cpp's MMQ
DEFAULT_ROWS = {'IQ3_S': 3, 'IQ2_S': 4}  # compute capability 12.0 without a measured decision: tensor cores from this many rows
enabled = os.environ.get('LLAMACPP_GGUF_IQ_ROWS', '1').strip().lower() not in ('0', 'off', 'false')


@functools.cache
def _architecture(device_index):
    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + minor


@functools.cache
def eligible(device_index):
    return torch.version.hip is None and available() and _architecture(device_index) >= 61  # DP4A


def variants(device_index):
    """The kernel variants of this GPU: DP4A from compute capability 6.1, INT8 tensor cores with cp.async from 8.0 ('mma2' reads the
    next weight blocks during the products, 'mma2x2' also shares the activations between two tiles of rows: which one is faster
    depends on the format, the weight's shape and the rows)."""
    if not eligible(device_index):
        return ()
    return VARIANTS if _architecture(device_index) >= 80 else VARIANTS[:1]


def set_decision(qtype_name, rows, out_features, in_features, variant):
    """The variant of a shape and row count ('' for the installed kernel), or None to go back to the default."""
    if qtype_name not in QTYPES or variant not in (None, '', *VARIANTS):
        raise ValueError(f'IQ rows decisions apply to {tuple(QTYPES)} with {VARIANTS}, got {qtype_name!r}, {variant!r}')
    key = (qtype_name, int(rows), int(out_features), int(in_features))
    if variant is None:
        decisions.pop(key, None)
    else:
        decisions[key] = variant


def variant(qtype_name, tensor_shape, rows, device_index):
    """The variant that serves this call, '' for the installed kernel."""
    if not enabled or qtype_name not in QTYPES or not 1 <= rows <= MAX_ROWS or tensor_shape[1] % 256 or not eligible(device_index):
        return ''
    decided = decisions.get((qtype_name, rows, int(tensor_shape[0]), int(tensor_shape[1])))
    if decided is not None:
        return decided if decided in variants(device_index) else ''
    if _architecture(device_index) != 120:  # elsewhere the installed kernel, until a caller measures
        return ''
    return 'mma2' if rows >= DEFAULT_ROWS[qtype_name] else ''


def kernel_source(qtype_name, rows, xtype, dtype, silu, config):
    outputs, warps, tiles = config if config[0] else (1, 1, config[2])
    constants = dict(QTYPE=QTYPES[qtype_name][0], NCOLS=rows, R=outputs, WARPS=warps, XTYPE=_DTYPES[xtype], DTYPE=_DTYPES[dtype], SILU=int(silu),
                     MMA=int(not config[0]), MSTAGES=1 if config[0] else config[1], MTILES=tiles)
    return ''.join(f'#define {key} {value}\n' for key, value in constants.items()) + Path(__file__).with_suffix('.cu').read_text()


@functools.cache
def _kernels(device, qtype_name, rows, xtype, dtype, silu, config):
    with torch.cuda.device(device):
        ptx = compile_ptx(kernel_source(qtype_name, rows, xtype, dtype, silu, config), _architecture(device))
        return Kernel(ptx, 'iq_quantize'), Kernel(ptx, 'iq_mm' if config[0] else 'iq_mma')


def _launch(qtype_name, x, raw, q, d, out, n, k, silu, config):
    rows = x.shape[0]
    quantize, mm = _kernels(x.device.index, qtype_name, rows, x.dtype, out.dtype, silu, config)
    quantize.launch(((k // 32 + 7) // 8, rows), 256, [ct.c_void_p(x.data_ptr()), ct.c_longlong(x.stride(0)), ct.c_void_p(q.data_ptr()),
                                                    ct.c_void_p(d.data_ptr()), ct.c_int(k)])
    outputs, warps = config[:2] if config[0] else (16 * config[2], 1)
    mm.launch(((n + outputs * warps - 1) // (outputs * warps), 1), warps * 32 if config[0] else 128,
              [ct.c_void_p(raw.data_ptr()), ct.c_longlong(raw.numel() * raw.element_size()), ct.c_void_p(q.data_ptr()), ct.c_void_p(d.data_ptr()),
               ct.c_void_p(out.data_ptr()), ct.c_int(n), ct.c_int(k)])


@torch.library.custom_op('llamacpp_gguf::iq_rows_linear', mutates_args=())
def iq_linear(input: torch.Tensor, raw: torch.Tensor, qtype_name: str, shape: List[int], bias: Optional[torch.Tensor], dtype: torch.dtype, silu_mul: bool, variant: str) -> torch.Tensor:
    """input: [..., K] activations, or [..., 2 K] gate then value of a SwiGLU when silu_mul; raw: the GGUF blocks of the [N, K] weight;
    variant: one of VARIANTS (the tensor-core ones from compute capability 8.0)."""
    n, k = shape
    x = input.reshape(-1, 2 * k if silu_mul else k)
    if x.dtype not in _DTYPES:
        x = x.to(dtype)
    if x.stride(1) != 1:
        x = x.contiguous()
    rows = x.shape[0]
    if raw.numel() * raw.element_size() < n * (k // 256) * QTYPES[qtype_name][1]:
        raise ValueError('GGUF weight is smaller than its logical shape')
    output = torch.empty((rows, n), dtype=dtype, device=x.device)
    if rows:
        q = torch.empty((rows, k), dtype=torch.int8, device=x.device)
        d = torch.empty((rows, k // 32), dtype=torch.float32, device=x.device)
        with torch.cuda.device(x.device):
            def scratch():
                sx, sq, sd, sy = x.clone(), torch.empty_like(q), torch.empty_like(d), torch.empty_like(output)
                return lambda config: _launch(qtype_name, sx, raw, sq, sd, sy, n, k, silu_mul, config)
            if variant in _MMA and raw.data_ptr() % 2 == 0:  # its 16-bit loads need blocks at even addresses, as files store them
                config = _MMA[variant]
            else:
                config = _TUNER.config('iq_mm', (qtype_name, n, k, rows, str(x.dtype), str(dtype), silu_mul), x.device, _CONFIGS, scratch, True)
            _launch(qtype_name, x, raw, q, d, output, n, k, silu_mul, config)
        if bias is not None:
            output += bias.to(dtype)
    return output.reshape(*input.shape[:-1], n)


@iq_linear.register_fake
def _iq_linear_fake(input, raw, qtype_name, shape, bias, dtype, silu_mul, variant):
    return input.new_empty((*input.shape[:-1], shape[0]), dtype=dtype)
