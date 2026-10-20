# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE-WanGP.txt)
"""PTQ1_0 linear for 1-7 activation rows on SM75 (Turing), see _ptq1_turing.cu: llama.cpp's PTQ1 x Q8_1 arithmetic, each weight block
decoded once for all rows. llama.cpp's MMVQ costs a full pass per row there (it is compute-bound on the trit decoding), which made the
rows of a speculative verification as expensive as decoding each token alone."""
import ctypes as ct
import functools
from pathlib import Path
from typing import List, Optional

import torch

from ._cuda_jit import compile_ptx, Kernel
from ._tuning import LaunchTuner

_DTYPES = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_CONFIGS = ((2, 8), (1, 8), (2, 4), (4, 4), (1, 16), (2, 16))  # (output features per warp, warps per block)
_TUNER = LaunchTuner('wangp_ptq1_turing', 1, 'PTQ1 Turing')
MAX_ROWS = 7  # llama.cpp uses MMVQ for PTQ1_0 on SM75 up to 7 rows, MMQ above (other activation quantization)


@functools.cache
def eligible(device_index):
    return torch.version.hip is None and torch.cuda.get_device_capability(device_index) == (7, 5)


def kernel_source(rows, xtype, dtype, bias, config):
    outputs, warps = config
    constants = dict(NCOLS=rows, R=outputs, WARPS=warps, XTYPE=_DTYPES[xtype], DTYPE=_DTYPES[dtype], HAS_BIAS=int(bias))
    return ''.join(f'#define {key} {value}\n' for key, value in constants.items()) + Path(__file__).with_suffix('.cu').read_text()


@functools.cache
def _kernels(device, rows, xtype, dtype, bias, config):
    with torch.cuda.device(device):
        ptx = compile_ptx(kernel_source(rows, xtype, dtype, bias, config), 75)
        return Kernel(ptx, 'ptq1_quantize'), Kernel(ptx, 'ptq1_mm')


def _launch(x, raw, bias, q, d, out, n, k, config):
    rows = x.shape[0]
    quantize, mm = _kernels(x.device.index, rows, x.dtype, out.dtype, bias is not None, config)
    quantize.launch(((k + 255) // 256, rows), 256, [ct.c_void_p(x.data_ptr()), ct.c_longlong(x.stride(0)), ct.c_void_p(q.data_ptr()),
                                                    ct.c_void_p(d.data_ptr()), ct.c_int(k)])
    outputs, warps = config
    mm.launch(((n + outputs * warps - 1) // (outputs * warps), 1), warps * 32,
              [ct.c_void_p(t.data_ptr()) for t in (raw, q, d, out if bias is None else bias, out)] + [ct.c_int(n), ct.c_int(k)])


@torch.library.custom_op('llamacpp_gguf::turing_ptq1_linear', mutates_args=())
def ptq1_linear(input: torch.Tensor, raw: torch.Tensor, shape: List[int], bias: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    n, k = shape
    x = input.reshape(-1, k)
    if x.dtype not in _DTYPES:
        x = x.to(dtype)
    if x.stride(1) != 1:
        x = x.contiguous()
    rows = x.shape[0]
    output = torch.empty((rows, n), dtype=dtype, device=x.device)
    if rows:
        if bias is not None:
            bias = bias.to(dtype).contiguous()
        q = torch.empty((rows, k), dtype=torch.int8, device=x.device)
        d = torch.empty((rows, k // 32), dtype=torch.float32, device=x.device)
        with torch.cuda.device(x.device):
            def scratch():
                sx, sq, sd, sy = x.clone(), torch.empty_like(q), torch.empty_like(d), torch.empty_like(output)
                return lambda config: _launch(sx, raw, bias, sq, sd, sy, n, k, config)
            config = _TUNER.config('ptq1_mm', (n, k, rows, str(x.dtype), str(dtype), bias is not None), x.device, _CONFIGS, scratch, True)
            _launch(x, raw, bias, q, d, output, n, k, config)
    return output.reshape(*input.shape[:-1], n)


@ptq1_linear.register_fake
def _ptq1_linear_fake(input, raw, shape, bias, dtype):
    return input.new_empty((*input.shape[:-1], shape[0]), dtype=dtype)
