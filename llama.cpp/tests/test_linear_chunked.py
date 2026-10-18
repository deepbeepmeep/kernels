"""linear_chunked (single-launch row loop) against row slices of the ordinary linear.

Row slices compute an FP32 product per slice with `linear`, copy it to the 16-bit output and add the bias in the output type. The
row loop must match bit for bit wherever MMQ computes whole tiles; slices where MMQ itself splits tiles (stream-K) may differ by the
FP32 summation order only. Also checks repeated calls, CUDA graph replay and the eligibility query. Run directly or with pytest.
"""
import sys

import gguf
import numpy as np
import torch

import llamacpp_gguf_cuda as kernels

SLICE_BYTES = 64 << 20


def packed(name, n, k, seed=7):
    qtype = gguf.GGMLQuantizationType[name]
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    raw = np.random.default_rng(seed).integers(0, 256, size=(n * k // block_size, type_size), dtype=np.uint8)
    offsets = [-2] if name == 'Q6_K' else [0, 2] if name in ('Q4_K', 'Q5_K') else [0]
    for offset in offsets:
        offset %= type_size
        raw[:, offset:offset + 2] = np.frombuffer(np.float16(.01).tobytes(), dtype=np.uint8)
    return torch.from_numpy(raw).flatten().cuda()


def slices(raw, qtype, shape, x, bias, dtype):
    n, k = shape
    rows = x.shape[0]
    step = max(128, SLICE_BYTES // (4 * n + k * 9 // 8) // 128 * 128)
    step = -(-rows // (-(-rows // step) * 128)) * 128
    out = torch.empty(rows, n, dtype=dtype, device=x.device)
    for start in range(0, rows, step):
        out[start:start + step].copy_(kernels.linear(raw, qtype, shape, x[start:start + step], None, torch.float32))
    if bias is not None:
        out.add_(bias.to(out.device, out.dtype))
    return out, step


def split_rows(rows, step, n):
    """Rows of slices where MMQ uses stream-K (tile efficiency below 90% of the waves)."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    mask = torch.zeros(rows, dtype=torch.bool, device='cuda')
    for start in range(0, rows, step):
        count = min(step, rows - start)
        tiles = -(-count // 128) * (n // 128)
        mask[start:start + count] = 100 * tiles // (sms * -(-tiles // sms)) < 90
    return mask


CASES = [('Q4_K', 2048, 4096, 20000), ('Q4_K', 1024, 8192, 9999), ('Q5_K', 2048, 4096, 12000), ('Q6_K', 2048, 4096, 12000), ('Q8_0', 1024, 4096, 15000)]


def check_case(qtype, n, k, rows, dtype, bias_kind):
    raw = packed(qtype, n, k)
    torch.manual_seed(3)
    x = torch.randn(rows, k, device='cuda', dtype=dtype)
    bias = None if bias_kind == 'none' else torch.randn(n, device='cuda', dtype=dtype if bias_kind == '16-bit' else torch.float32)
    assert kernels.linear_chunked_supported(qtype, (n, k), dtype, bias is not None, 0)
    reference, step = slices(raw, qtype, (n, k), x, bias, dtype)
    scratch = step * (4 * n + k * 9 // 8)
    result = kernels.linear_chunked(raw, qtype, (n, k), x, bias, dtype, scratch)
    assert result is not None and result.shape == reference.shape and result.dtype == dtype
    assert torch.isfinite(result).all()
    assert torch.equal(result, kernels.linear_chunked(raw, qtype, (n, k), x, bias, dtype, scratch))
    differs = (result != reference).any(dim=1)
    outside = int((differs & ~split_rows(rows, step, n)).sum())
    assert outside == 0, f'{qtype} {k}->{n} {rows} rows {dtype} bias={bias_kind}: {outside} rows differ outside stream-K slices'
    return int(differs.sum())


def test_matches_row_slices():
    for qtype, n, k, rows in CASES:
        for dtype in (torch.bfloat16, torch.float16):
            for bias_kind in ('none', '16-bit', 'fp32'):
                check_case(qtype, n, k, rows, dtype, bias_kind)


def test_graph_replay():
    raw = packed('Q4_K', 2048, 4096)
    x = torch.randn(20000, 4096, device='cuda', dtype=torch.bfloat16)
    bias = torch.randn(2048, device='cuda', dtype=torch.bfloat16)
    eager = kernels.linear_chunked(raw, 'Q4_K', (2048, 4096), x, bias, torch.bfloat16, SLICE_BYTES)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        graph.capture_begin()
        captured = kernels.linear_chunked(raw, 'Q4_K', (2048, 4096), x, bias, torch.bfloat16, SLICE_BYTES)
        graph.capture_end()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)
    x.copy_(torch.randn_like(x))
    graph.replay()
    assert torch.equal(captured, kernels.linear_chunked(raw, 'Q4_K', (2048, 4096), x, bias, torch.bfloat16, SLICE_BYTES))


def test_eligibility():
    assert not kernels.linear_chunked_supported('Q4_0', (4096, 4096), torch.bfloat16, False, 0)
    assert not kernels.linear_chunked_supported('Q4_K', (4096, 4096), torch.float32, False, 0)
    assert not kernels.linear_chunked_supported('Q4_K', (4000, 4096), torch.bfloat16, False, 0)
    x = torch.randn(256, 4096, device='cuda', dtype=torch.bfloat16)
    assert kernels.linear_chunked(packed('Q4_K', 2048, 4096), 'Q4_K', (2048, 4096), x.t().contiguous().t(), None, torch.bfloat16, SLICE_BYTES) is None


def main():
    torch.cuda.init()
    kernels.prepare_runtime_buffers(0)
    if torch.version.hip is not None or not kernels.linear_chunked_supported('Q4_K', (2048, 4096), torch.bfloat16, False, 0):
        print('linear_chunked not supported on this device/build: skipped')
        return 0
    test_matches_row_slices()
    test_graph_replay()
    test_eligibility()
    print('PASSED linear_chunked:', len(CASES) * 6, 'cases, graph replay, eligibility')
    return 0


if __name__ == '__main__':
    sys.exit(main())
