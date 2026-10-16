"""Independent GGUF fixtures for the public crossing-row linear API."""
import numpy as np
import torch
import gguf
import llamacpp_gguf_cuda as kernels


def _packed(name, rows=8, columns=256):
    qtype = gguf.GGMLQuantizationType[name]
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    raw = np.random.default_rng(321).integers(0, 256, size=(rows * columns // block_size, type_size), dtype=np.uint8)
    offsets = [-4, -2] if name == 'Q2_K' else [-2] if name in ('Q3_K', 'Q6_K') else [0, 2] if name in ('Q4_K', 'Q5_K', 'Q4_1', 'Q5_1') else [0]
    for offset in offsets:
        offset %= type_size
        raw[:, offset:offset + 2] = np.frombuffer(np.float16(.025).tobytes(), dtype=np.uint8)
    reference = gguf.quants.dequantize(raw, qtype).reshape(rows, columns)
    return raw, qtype, torch.from_numpy(reference)


class PackedWeight:
    def __init__(self, raw, qtype, shape, dtype):
        self._data, self._tensor_type, self.shape, self.dtype = raw, qtype, shape, dtype

    def linear(self, x, bias=None):
        return kernels.linear(self._data, self._tensor_type.name, self.shape, x, bias, self.dtype)


def _weight(name, n, k, dtype):
    raw, qtype, dense = _packed(name, n, k)
    return PackedWeight(torch.from_numpy(raw).flatten().cuda(), qtype, (n, k), dtype), dense.cuda()
