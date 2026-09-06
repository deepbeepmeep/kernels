"""Precompiled SM120 async attention, with no runtime Triton dependency."""
from functools import lru_cache
import json
from pathlib import Path

import torch
from . import _attention


_ROOT = Path(__file__).parent / 'kernels' / ('sm120_cu' + torch.version.cuda.split('.')[0])
_MANIFEST = json.loads((_ROOT / 'manifest.json').read_text(encoding='utf-8'))


@lru_cache(maxsize=None)
def _kernel(device, name, dtype):
    key = name + ('_bf16' if dtype == torch.bfloat16 else '_fp16')
    spec = _MANIFEST['kernels'][key]
    with torch.cuda.device(device):
        return _attention.load_sm120_kernel((_ROOT / (key + '.cubin')).read_bytes(), spec['function'], spec['shared'])


def q8_paged_prefill(q, k_cache, v_cache, k_scale, v_scale, context, softmax_scale):
    output = torch.empty_like(q)
    _attention.sm120_prefill(_kernel(q.device.index, 'q8_prefill_async', q.dtype), q, k_cache, v_cache, k_scale, v_scale, context.block_tables, context.cu_seqlens_q, context.cu_seqlens_k, output, softmax_scale)
    return output


def q8_grouped_partials(q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, partial, maximum, denominator, splits, softmax_scale):
    _attention.sm120_grouped(_kernel(q.device.index, 'q8_grouped_async', q.dtype), q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, partial, maximum, denominator, splits, softmax_scale)
