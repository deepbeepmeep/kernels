"""The precompiled SM120 grouped path must preserve values."""
import hashlib

import pytest
import torch

from shared.llm_engines.nanovllm.layers import attention
from test_q8_paged_attention import _case


native = getattr(attention.llamacpp_gguf_cuda, 'sm120', None)
pytestmark = pytest.mark.skipif(native is None or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason='SM120 and GGUF wheel with compiled async attention required')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('queries,heads,kv_heads', [(1, 24, 4), (3, 24, 4), (5, 16, 4), (9, 8, 2)])
@torch.inference_mode()
def test_native_grouped_matches_shared(monkeypatch, dtype, queries, heads, kv_heads):
    q, k, v, ks, vs, context = _case(dtype, 256, [queries, queries], [531, 991], heads, kv_heads)
    run = lambda: attention._q8_grouped_attention(q, k, v, ks, vs, context.block_tables, context.context_lens, .0625)
    monkeypatch.setattr(attention, '_SM120_Q8', None)
    expected = run()
    monkeypatch.setattr(attention, '_SM120_Q8', native)
    actual = run()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    q.mul_(.5)
    context.context_lens.copy_(torch.tensor([257, 769], dtype=torch.int32, device='cuda'))
    context.block_tables[:, :2] = context.block_tables[:, :2].flip(1)
    graph.replay()
    monkeypatch.setattr(attention, '_SM120_Q8', None)
    torch.testing.assert_close(output, run(), atol=0, rtol=0)


def test_exactly_four_binaries_cover_runtime_shapes():
    assert set(native._MANIFEST['kernels']) == {'q8_prefill_async_fp16', 'q8_prefill_async_bf16', 'q8_grouped_async_fp16', 'q8_grouped_async_bf16'}
    for key, metadata in native._MANIFEST['kernels'].items():
        assert metadata['signature']['H_Q'] == metadata['signature']['H_KV'] == metadata['signature']['PAGE'] == 'i32'
        assert metadata['scratch_pointer_arguments'] == 2
        assert hashlib.sha256((native._ROOT / (key + '.cubin')).read_bytes()).hexdigest() == metadata['sha256']
