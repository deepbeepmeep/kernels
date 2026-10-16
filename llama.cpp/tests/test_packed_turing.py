"""Packed row-boundary decode, numerical precision, allocation and graph checks."""
from pathlib import Path
import pytest
import torch
import triton
import triton.language as tl
from llamacpp_gguf_cuda import _packed as dispatch
import gguf
import llamacpp_gguf_cuda as api
from llamacpp_gguf_cuda._packed_turing import flat_linear, _TUNER, kernel_source
from llamacpp_gguf_cuda._cuda_jit import compile_ptx, Kernel
import ctypes as ct
from packed_fixtures import _weight, _packed

QTYPES = ['Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0']

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('qtype', QTYPES)
@pytest.mark.parametrize('n,k', [(16, 1136), (16, 4304), (16, 80), (256, 31)])
@torch.inference_mode()
def test_packed_decode_is_exact(qtype, n, k):
    raw, _, dense = _packed(qtype, n, k)
    raw = torch.from_numpy(raw).cuda()
    actual = torch.empty((n, k), device='cuda', dtype=torch.float32)
    source = kernel_source(n, k, qtype, torch.float32, torch.float32, False, (32, 32, 32))
    source += '\nextern "C" __global__ void decode_test(const u8* W, float* Y) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < N*K) Y[i] = decode<QTYPE>(W, i); }'
    kernel = Kernel(compile_ptx(source, 75), 'decode_test')
    kernel.launch(((n*k+255)//256, 1), 256, [ct.c_void_p(raw.data_ptr()), ct.c_void_p(actual.data_ptr())])
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


@pytest.fixture(autouse=True)
def use_turing_backend(monkeypatch):
    monkeypatch.setattr(dispatch, 'implementation', lambda device: flat_linear)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_bf16_range_is_preserved_inside_tiles(dtype):
    weight, dense = _weight('Q4_K', 16, 80, dtype)
    x = torch.zeros((9, 80), device='cuda', dtype=dtype)
    x[0, 1] = 1e20
    x[1, 17] = 1e-20
    x[2, 33] = -1e20
    x[3, 67] = -1e-20
    out = weight.linear(x)
    ref = torch.nn.functional.linear(x.float(), dense.to(dtype).float()).to(dtype)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize('input_dtype,target_dtype', [
    (torch.float32, torch.float16), (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16),
    (torch.bfloat16, torch.float32), (torch.float16, torch.float32),
    (torch.float64, torch.float16),
])
@torch.inference_mode()
def test_input_cast_contract(input_dtype, target_dtype):
    weight, dense = _weight('Q4_K', 16, 80, target_dtype)
    x = torch.randn(9, 80, device='cuda', dtype=input_dtype)
    out = weight.linear(x)
    ref = torch.nn.functional.linear(x.to(target_dtype).float(), dense.to(target_dtype).float())
    assert out.dtype == target_dtype
    assert (out.float() - ref).norm() / ref.norm() < (2e-6 if target_dtype == torch.float32 else .004)
