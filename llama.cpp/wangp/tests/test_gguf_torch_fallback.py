from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shared.qtypes import gguf as handler


gguf = pytest.importorskip('gguf')
QTYPES = ('Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'IQ2_S', 'IQ3_S', 'IQ1_S', 'IQ2_XS', 'IQ2_XXS', 'IQ3_XXS', 'IQ4_NL', 'IQ4_XS')


def _packed(name, rows=8, columns=256):
    qtype = gguf.GGMLQuantizationType[name]
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    raw = np.random.default_rng(321).integers(0, 256, size=(rows * columns // block_size, type_size), dtype=np.uint8)
    offsets = [-4, -2] if name == 'Q2_K' else [-2] if name in ('Q3_K', 'Q6_K') else [0, 2] if name in ('Q4_K', 'Q5_K', 'Q4_1', 'Q5_1') else [0]
    for offset in offsets:
        offset %= type_size
        raw[:, offset:offset + 2] = np.float16(.025).tobytes()[0], np.float16(.025).tobytes()[1]
    reference = gguf.quants.dequantize(raw, qtype).reshape(rows, columns)
    return raw, qtype, torch.from_numpy(reference)


@pytest.mark.parametrize('name', QTYPES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_dequantization_matches_gguf_float32_then_cast(name, dtype):
    raw, qtype, expected = _packed(name)
    actual = handler._gguf_dequantize_tensor(torch.from_numpy(raw), qtype, expected.shape, dtype)
    torch.testing.assert_close(actual, expected.to(dtype), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('name', QTYPES)
@torch.inference_mode()
def test_missing_native_linear_embedding_and_graph_replay(name, monkeypatch):
    monkeypatch.setattr(handler, '_gguf_cuda_module', lambda: None)
    monkeypatch.setattr(handler, '_GGUF_DEQUANTIZE_BLOCK_CHUNK', 5)
    raw, qtype, expected = _packed(name)
    raw = torch.from_numpy(raw).flatten().cuda()
    weight = handler.GGUFWeightTensor.create(raw, expected.shape, expected.stride(), torch.bfloat16, tensor_type=qtype, tensor_shape=expected.shape)
    inputs = torch.randn(3, expected.shape[1], device='cuda', dtype=torch.bfloat16)
    indices = torch.tensor([3, 3, 0], device='cuda')
    dense = expected.cuda().bfloat16()

    def run():
        return weight.linear(inputs), weight.embedding(indices)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        linear, embedding = run()
    for ids in ([3, 3, 0], [7, 1, 2]):
        indices.copy_(torch.tensor(ids, device='cuda'))
        inputs.normal_()
        graph.replay()
        torch.testing.assert_close(linear, torch.nn.functional.linear(inputs, dense), atol=0, rtol=0)
        torch.testing.assert_close(embedding, dense[indices], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('module', [SimpleNamespace(), SimpleNamespace(may_support_linear_qtype_name=lambda name: True)])
def test_older_native_package_missing_linear(module, monkeypatch):
    monkeypatch.setattr(handler, '_gguf_cuda_module', lambda: module)
    raw, qtype, expected = _packed('Q4_K')
    weight = handler.GGUFWeightTensor.create(torch.from_numpy(raw).flatten().cuda(), expected.shape, expected.stride(), torch.bfloat16, tensor_type=qtype, tensor_shape=expected.shape)
    inputs = torch.randn(1, expected.shape[1], device='cuda', dtype=torch.bfloat16)
    torch.testing.assert_close(weight.linear(inputs), torch.nn.functional.linear(inputs, expected.cuda().bfloat16()), atol=0, rtol=0)


@pytest.mark.parametrize('name', ['Q4_K', 'Q6_K'])
def test_mtp_partial_head_fallback_slices_packed_rows(name, monkeypatch):
    monkeypatch.setattr(handler, '_gguf_cuda_module', lambda: None)
    raw, qtype, expected = _packed(name)
    weight = handler.GGUFWeightTensor.create(torch.from_numpy(raw).reshape(expected.shape[0], -1), expected.shape, expected.stride(), torch.bfloat16, tensor_type=qtype, tensor_shape=expected.shape)
    source = torch.nn.Module()
    source.register_parameter('weight', torch.nn.Parameter(weight, requires_grad=False))
    head = handler.GGUFFirstRowsLinear(source, 3)
    inputs = torch.randn(2, expected.shape[1], dtype=torch.bfloat16)
    torch.testing.assert_close(head(inputs), torch.nn.functional.linear(inputs, expected[:3].bfloat16()), atol=0, rtol=0)
