"""Packed multiplication for logical rows that cross GGUF block boundaries."""
import functools

import torch


FORMATS = {
    'Q2_K': (256, 84), 'Q3_K': (256, 110), 'Q4_K': (256, 144),
    'Q5_K': (256, 176), 'Q6_K': (256, 210),
    'Q4_0': (32, 18), 'Q4_1': (32, 20), 'Q5_0': (32, 22),
    'Q5_1': (32, 24), 'Q8_0': (32, 34),
}
DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@functools.cache
def implementation(device_index):
    if torch.version.hip is not None:
        return None
    capability = torch.cuda.get_device_capability(device_index)
    if capability == (7, 5):
        from ._packed_turing import flat_linear
        return flat_linear
    if capability not in ((8, 6), (8, 9), (12, 0)):
        return None
    try:
        from ._packed_triton import flat_linear
    except ImportError:
        return None
    return flat_linear


def supports(qtype_name, device, output_dtype):
    device = torch.device(device)
    return (device.type == 'cuda' and qtype_name in FORMATS and output_dtype in DTYPES
            and implementation(device.index) is not None)


def linear(raw, qtype_name, shape, x, bias, dtype):
    if not supports(qtype_name, x.device, dtype):
        raise NotImplementedError(f'Packed GGUF rows crossing blocks are unavailable for {qtype_name}, {x.device}, {dtype}.')
    return implementation(x.device.index)(x, raw, list(shape), qtype_name, bias, dtype)


def validate(raw, qtype_name, shape, x, bias, dtype):
    # Run inside the custom-op boundary: tensor-subclass dispatch has already
    # unwrapped the call, so metadata checks do not redispatch for every field.
    n, k = shape
    block, size = FORMATS[qtype_name]
    if (n <= 0 or k <= 0 or n * k % block or raw.numel() != n * k // block * size):
        raise ValueError('GGUF packed byte count must match the complete logical matrix.')
    if raw.dtype != torch.uint8 or not raw.is_contiguous() or raw.device != x.device:
        raise ValueError('GGUF packed bytes must be contiguous uint8 on the input device.')
    if x.ndim == 0 or x.shape[-1] != k or not x.is_floating_point():
        raise ValueError('GGUF linear input must be floating point with the declared input width.')
    if bias is not None:
        if bias.ndim != 1 or bias.numel() != n:
            raise ValueError('GGUF linear bias must have one value per output feature.')
        if bias.device != x.device or bias.dtype != dtype:
            bias = bias.to(device=x.device, dtype=dtype)
    return bias
