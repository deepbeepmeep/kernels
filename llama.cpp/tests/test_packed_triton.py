"""Packed row-boundary decode, numerical precision, allocation and graph checks."""
from pathlib import Path
import pytest
import torch
import triton
import triton.language as tl
from llamacpp_gguf_cuda import _packed as dispatch
import gguf
import llamacpp_gguf_cuda as api
from llamacpp_gguf_cuda._packed_triton import _weight_tile, _weight_tile_16, flat_linear, _TUNER
from packed_fixtures import _weight, _packed

QTYPES = ['Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0']

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@triton.jit
def _decode_test(W, Y, N: tl.constexpr, K: tl.constexpr, QTYPE: tl.constexpr):
    n = tl.program_id(0) * 16 + tl.arange(0, 16)
    k = tl.program_id(1) * 128 + tl.arange(0, 128)
    if K % 16 == 0:
        w = _weight_tile_16(W, n, tl.program_id(1) * 8 + tl.arange(0, 8), N, K, QTYPE)
    else:
        w = _weight_tile(W, n[:, None] * K + k[None, :], (n[:, None] < N) & (k[None, :] < K), QTYPE)
    tl.store(Y + n[:, None] * K + k[None, :], w, (n[:, None] < N) & (k[None, :] < K))


@pytest.mark.parametrize('qtype', QTYPES)
@pytest.mark.parametrize('n,k', [(16, 1136), (16, 4304), (16, 80), (256, 31)])
@torch.inference_mode()
def test_packed_decode_is_exact(qtype, n, k):
    raw, _, dense = _packed(qtype, n, k)
    raw = torch.from_numpy(raw).cuda()
    actual = torch.empty((n, k), device='cuda', dtype=torch.float32)
    _decode_test[(triton.cdiv(n, 16), triton.cdiv(k, 128))](raw, actual, n, k, qtype, enable_fp_fusion=False)
    torch.testing.assert_close(actual.cpu(), dense, atol=0, rtol=0)


@pytest.mark.parametrize('qtype', QTYPES)
@pytest.mark.parametrize('n,k', [(16, 1136), (16, 4304), (16, 80), (256, 31)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_packed_linear_without_dense_materialization(qtype, n, k, dtype, monkeypatch):
    weight, dense = _weight(qtype, n, k, dtype)
    monkeypatch.setenv('WGP_GGUF_LLAMACPP_CUDA_MATMUL_MODE', 'mmq')
    x = torch.randn(3, k, 5, device='cuda', dtype=dtype).transpose(1, 2)
    bias = torch.randn(2 * n, device='cuda', dtype=torch.float32)[::2]
    gold = torch.nn.functional.linear(x.float(), dense.to(dtype).float(), bias.to(dtype).float())
    out = weight.linear(x, bias)
    assert out.dtype == dtype and out.shape == (3, 5, n)
    assert torch.isfinite(out).all()
    relative = (out.float() - gold).norm() / gold.norm()
    assert relative < (2e-6 if dtype == torch.float32 else .001 if dtype == torch.float16 else .004)
    torch.testing.assert_close(weight.linear(torch.zeros_like(x), bias), bias.to(dtype).expand_as(out), atol=0, rtol=0)


@pytest.mark.parametrize('qtype', QTYPES)
@torch.inference_mode()
def test_only_output_storage_is_allocated(qtype):
    weight, _ = _weight(qtype, 512, 4304, torch.bfloat16)
    x = torch.randn(256, 4304, device='cuda', dtype=torch.bfloat16)
    weight.linear(x)  # compile/tune before the allocation measurement
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = weight.linear(x)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    assert extra == result.numel() * result.element_size()


@torch.inference_mode()
def test_empty_input_and_graph_replay_without_tuning_during_capture(monkeypatch):
    weight, _ = _weight('Q4_K', 16, 4304, torch.bfloat16)
    empty = torch.empty((0, 4304), device='cuda', dtype=torch.bfloat16)
    assert weight.linear(empty).shape == (0, 16)
    x = torch.randn(9, 4304, device='cuda', dtype=torch.bfloat16)
    weight.linear(x)
    monkeypatch.setattr(_TUNER, '_tuned', {})
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = weight.linear(x)
    assert _TUNER._tuned == {}
    for _ in range(3):
        x.normal_()
        graph.replay()
        torch.testing.assert_close(out, weight.linear(x), atol=0, rtol=0)


def test_fake_output_contract():
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        x = torch.empty((3, 5, 1152), device='cuda', dtype=torch.bfloat16)
        raw = torch.empty((3168,), device='cuda', dtype=torch.uint8)
        result = flat_linear(x, raw, [16, 1152], 'Q5_K', None, torch.bfloat16)
        assert result.shape == (3, 5, 16) and result.dtype == x.dtype


@pytest.mark.parametrize('qtype', QTYPES)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_outliers_subnormals_and_zero_scales(qtype, dtype):
    weight, dense = _weight(qtype, 16, 1136, dtype)
    x = torch.zeros((9, 1136), device='cuda', dtype=dtype)
    x[0, 127] = 1000
    x[1, 1025] = -1000
    x[2:4] = torch.finfo(dtype).tiny / 2
    x[4:6] = -torch.finfo(dtype).tiny / 2
    x[6, ::2] = 1
    x[6, 1::2] = -1
    result = weight.linear(x)
    gold = torch.nn.functional.linear(x.float(), dense.to(dtype).float())
    # Q6_K's signed scales can make the 1000-valued outlier exceed FP16's
    # output range. Match that genuine output overflow as well as finite values.
    assert torch.equal(torch.isfinite(result), torch.isfinite(gold.to(dtype)))
    torch.testing.assert_close(result.float(), gold.to(dtype).float(), rtol=.008 if dtype == torch.bfloat16 else .001,
                               atol=max(torch.finfo(dtype).tiny * 1152, 2e-5))
    size = gguf.GGML_QUANT_SIZES[weight._tensor_type][1]
    offsets = [-4, -2] if qtype == 'Q2_K' else [-2] if qtype in ('Q3_K', 'Q6_K') else [0, 2] if qtype in ('Q4_K', 'Q5_K', 'Q4_1', 'Q5_1') else [0]
    for offset in offsets:
        offset %= size
        weight._data.reshape(-1, size)[:, offset:offset + 2].zero_()
    assert torch.count_nonzero(weight.linear(x)) == 0


def test_optional_kernel_eligibility_is_cached(monkeypatch):
    dispatch.implementation.cache_clear()
    checks = []
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: checks.append(device) or (8, 0))
    assert dispatch.implementation(0) is None
    assert dispatch.implementation(0) is None
    assert checks == [0]
    dispatch.implementation.cache_clear()


@pytest.mark.parametrize('capability', [(8, 6), (8, 9), (12, 0)])
def test_supported_families_use_same_packed_implementation(capability, monkeypatch):
    dispatch.implementation.cache_clear()
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: capability)
    assert dispatch.implementation(0) is flat_linear
    dispatch.implementation.cache_clear()


def test_turing_uses_native_packed_implementation(monkeypatch):
    from llamacpp_gguf_cuda._packed_turing import flat_linear as turing_flat_linear
    dispatch.implementation.cache_clear()
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (7, 5))
    assert dispatch.implementation(0) is turing_flat_linear
    dispatch.implementation.cache_clear()
