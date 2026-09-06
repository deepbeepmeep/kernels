from types import SimpleNamespace

import pytest
import torch

from shared.llm_engines.nanovllm import vllm_support
from shared.llm_engines.nanovllm.layers import activation, attention, sampler
from test_q8_paged_attention import _case, _reference


def _forbidden(*args, **kwargs):
    raise AssertionError('Optional GPU backend called in PyTorch mode')


@pytest.mark.parametrize('mode', ['legacy', 'cg'])
def test_explicit_pytorch_modes_do_not_probe_triton(mode, monkeypatch):
    monkeypatch.setattr(vllm_support, '_is_mps_available', lambda: False)
    monkeypatch.setattr(vllm_support, 'probe_vllm_runtime', _forbidden)
    assert vllm_support.resolve_lm_decoder_engine(mode, ['cg', 'vllm']) == mode


@pytest.mark.parametrize('module', [None, SimpleNamespace(), SimpleNamespace(q8_paged_attention_format=lambda: 'q8_0_fp16_scales_v1'), SimpleNamespace(q8_paged_attention=_forbidden), SimpleNamespace(q8_paged_attention=_forbidden, q8_paged_attention_format=lambda: 'old')])
def test_older_q8_kernel_package_is_optional(module):
    assert attention._get_q8_paged_attention(module) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_pytorch_silu_and_repetition_do_not_launch_triton(monkeypatch):
    import triton
    monkeypatch.setattr(triton.runtime.JITFunction, 'run', _forbidden)
    source = torch.randn(3, 66, device='cuda', dtype=torch.bfloat16)
    expected = torch.nn.functional.silu(source[..., :33]) * source[..., 33:]
    actual = activation.SiluAndMul(use_triton=False).forward_list([source])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    scores = torch.linspace(-5, 5, 64, device='cuda')
    expected = scores.clone()
    ids = [1, 17, 19, 23, 47, 52, 55, 56, 57, 58, 59, 60]
    expected[ids] = torch.where(expected[ids] < 0, expected[ids] * 1.05, expected[ids] / 1.05)
    buffer = torch.zeros(64, device='cuda', dtype=torch.long)
    buffer[:2] = torch.tensor(ids[:2], device='cuda')
    sampler.apply_sparse_repetition_penalty_(scores, buffer, 2, ids[2:4], ids[4:], 1.05, use_triton=False)
    torch.testing.assert_close(scores, expected)


@pytest.mark.skipif(not torch.cuda.is_available() or attention.triton is None, reason='CUDA and Triton required for reference')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_pytorch_q8_storage_preserves_fp32_quantization(dtype):
    torch.manual_seed(982)
    k, v = torch.randn(2, 17, 4, 256, device='cuda', dtype=dtype).unbind()
    slots = torch.arange(17, device='cuda', dtype=torch.int32)
    results = []
    for use_triton in (True, False):
        kc = torch.zeros(1, 256, 4, 256, device='cuda', dtype=torch.int8)
        vc = torch.zeros_like(kc)
        ks = torch.zeros(1, 256, 4, 8, device='cuda', dtype=torch.float16)
        vs = torch.zeros_like(ks)
        attention.store_kvcache(k, v, kc, vc, slots, ks, vs, use_triton_kv_cache=use_triton)
        results.append((kc, vc, ks, vs))
    for index, source in enumerate((k, v)):
        blocks = source.float().reshape(17, 4, 8, 32)
        scales = blocks.abs().amax(-1).div(127).clamp_min(1e-8)
        ratios = blocks / scales.unsqueeze(-1)
        expected = ratios.round().clamp(-127, 127).to(torch.int8).reshape_as(source)
        actual = results[1][index][0, :17]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        native = results[0][index][0, :17]
        # Triton's approximate division can choose the other side of an FP32 half tie.
        mismatch = native != actual
        assert (native.int() - actual.int()).abs().max() <= 1
        assert torch.all(((ratios.reshape_as(source)[mismatch].abs() % 1) - .5).abs() <= 2e-5)
        torch.testing.assert_close(results[0][index + 2], results[1][index + 2], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('phase', ['decode', 'verify', 'prefix'])
@torch.inference_mode()
def test_q8_pytorch_fallback_without_native_or_triton(phase, monkeypatch):
    import triton
    monkeypatch.setattr(triton.runtime.JITFunction, 'run', _forbidden)
    monkeypatch.setattr(attention, '_Q8_PAGED_ATTENTION', None)
    count = 1 if phase == 'decode' else 3
    q, kc, vc, ks, vs, context = _case(torch.bfloat16, 64, [count], [257])
    context.is_prefill = phase != 'decode'
    context.speculative_verify = phase == 'verify'
    context.max_seqlen_q, context.max_seqlen_k = count, 257
    context.slot_mapping = context.block_tables[0, 0] * 256 + torch.arange(count, device='cuda', dtype=torch.int32)
    layer = attention.Attention(24, 64, 64 ** -.5, 4)
    layer.use_triton_kv_cache = False
    layer.flash_attn_varlen_func = layer.flash_attn_with_kvcache = None
    layer.k_cache, layer.v_cache, layer.k_scale, layer.v_scale = kc, vc, ks, vs
    monkeypatch.setattr(attention, 'get_context', lambda: context)
    k, v = torch.randn(2, count, 4, 64, device='cuda', dtype=q.dtype).unbind()
    actual = layer(q, k, v)
    if phase != 'prefix':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = layer(q, k, v)
        graph.replay()
    expected = _reference(q, kc, vc, ks, vs, context)
    torch.testing.assert_close(actual.reshape_as(q), expected, atol=.005, rtol=.03)
