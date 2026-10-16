# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE-WanGP.txt)
"""Packed SM75 GGUF, using FP16 tensor cores with FP32 accumulation."""
import ctypes as ct
import functools
from pathlib import Path
from typing import List, Optional
import torch
from ._cuda_jit import compile_ptx, Kernel
from ._tuning import LaunchTuner
from ._packed import validate

_QTYPES = ('Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0')
_DTYPES = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_CONFIGS = ((32, 64, 32), (64, 64, 32), (64, 128, 32), (32, 64, 64))
_TUNER = LaunchTuner('wangp_gguf_turing', 6, 'GGUF Turing')


def kernel_source(n, k, qtype, xtype, dtype, bias, config):
    bm, bn, bk = config
    constants = dict(N=n, K=k, QTYPE=_QTYPES.index(qtype), XTYPE=_DTYPES[xtype], DTYPE=_DTYPES[dtype],
                     HAS_BIAS=int(bias), BM=bm, BN=bn, BK=bk)
    return ''.join(f'#define {key} {value}\n' for key, value in constants.items()) + Path(__file__).with_suffix('.cu').read_text()


@functools.cache
def _kernel(device, n, k, qtype, xtype, dtype, bias, config):
    source = kernel_source(n, k, qtype, xtype, dtype, bias, config)
    with torch.cuda.device(device):
        return Kernel(compile_ptx(source, 75), 'packed_mm')


def _launch(x, raw, bias, out, n, k, qtype, config):
    kernel = _kernel(x.device.index, n, k, qtype, x.dtype, out.dtype, bias is not None, config)
    bm, bn, _ = config
    values = [ct.c_void_p(t.data_ptr()) for t in (x, raw, out if bias is None else bias, out)]
    values += [ct.c_int(x.shape[0]), ct.c_longlong(x.stride(0)), ct.c_longlong(x.stride(1))]
    wm, wn = 32 if bm >= 64 else 16, 32 if bn >= 64 else 16
    kernel.launch(((x.shape[0] + bm - 1) // bm, (n + bn - 1) // bn), bm // wm * (bn // wn) * 32, values)


@torch.library.custom_op('llamacpp_gguf::turing_flat_linear', mutates_args=())
def flat_linear(input: torch.Tensor, raw: torch.Tensor, shape: List[int], qtype_name: str,
                bias: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    bias = validate(raw, qtype_name, shape, input, bias, dtype)
    n, k = shape
    x = input.reshape(-1, k)
    if x.dtype not in _DTYPES:
        x = x.to(dtype)
    output = torch.empty((x.shape[0], n), dtype=dtype, device=x.device)
    if x.shape[0]:
        if bias is not None:
            bias = bias.to(dtype).contiguous()
        with torch.cuda.device(x.device):
            def scratch():
                sx = x[:min(256, x.shape[0])].clone()
                sy = output.new_empty((sx.shape[0], n))
                return lambda config: _launch(sx, raw, bias, sy, n, k, qtype_name, config)
            identity = (n, k, qtype_name, x.dtype, dtype, bias is not None)
            config = _TUNER.config('gguf_turing_mm', identity, x.device, _CONFIGS, scratch, x.shape[0] >= 16)
            _launch(x, raw, bias, output, n, k, qtype_name, config)
    return output.reshape(*input.shape[:-1], n)


@flat_linear.register_fake
def _flat_linear_fake(input, raw, shape, qtype_name, bias, dtype):
    return input.new_empty((*input.shape[:-1], shape[0]), dtype=dtype)
