"""BF16 output written by the stream-k MMQ kernels (prefill-sized batches) is the FP32 product rounded once: bit-identical to the
FP32 path converted afterwards, for complete tiles, tiles split between blocks (fixup) and partial edge tiles."""
import numpy as np
import pytest
import torch
import gguf
import llamacpp_gguf_cuda as kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not kernels.has_typed_mmq(), reason="CUDA build with typed MMQ output")

TYPES = ["Q4_K", "Q6_K", "Q8_0", "Q5_K", "Q4_0", "Q5_0", "Q4_1", "Q5_1", "Q2_K", "Q3_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL", "IQ1_S"]


def packed(name, n, k, seed=7):
    qtype = gguf.GGMLQuantizationType[name]
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    raw = np.random.default_rng(seed).integers(0, 256, size=(n * k // block_size, type_size), dtype=np.uint8)
    offsets = [-4, -2] if name == "Q2_K" else [-2] if name in ("Q3_K", "Q6_K") else [0, 2] if name in ("Q4_K", "Q5_K", "Q4_1", "Q5_1") else [0]
    for offset in offsets:
        raw[:, offset % type_size:offset % type_size + 2] = np.frombuffer(np.float16(.025).tobytes(), dtype=np.uint8)
    return torch.from_numpy(raw).flatten().cuda()


def packed_ptq1(n, k, seed=11):
    """Prism ternary blocks: 128 base-3 values in 28 bytes (24 packed bytes, 2 shared digits bytes, FP16 scale)."""
    generator = torch.Generator().manual_seed(seed)
    blocks = torch.empty(n * k // 128, 28, dtype=torch.uint8)
    blocks[:, :24] = torch.randint(0, 243, (blocks.shape[0], 24), generator=generator, dtype=torch.uint8)
    blocks[:, 24:26] = torch.randint(0, 81, (blocks.shape[0], 2), generator=generator, dtype=torch.uint8)
    blocks[:, 26:28] = torch.empty(blocks.shape[0], 1, dtype=torch.float16).uniform_(0.005, 0.03, generator=generator).view(torch.uint8)
    return blocks.reshape(-1).cuda()


@pytest.mark.parametrize("shape", [(1024, 5120), (1152, 5120), (6240, 5120), (1024, 17408)])
@pytest.mark.parametrize("rows", [9, 129, 1000, 1024, 2048])
def test_ptq1_bf16_matches_fp32_rounded(shape, rows):
    if not kernels.supports_linear_qtype_name("PTQ1_0"):
        pytest.skip("PTQ1_0 kernels unavailable")
    raw = packed_ptq1(*shape)
    x = torch.randn((rows, shape[1]), device="cuda", dtype=torch.bfloat16)
    expected = kernels.linear(raw, "PTQ1_0", list(shape), x, None, torch.float32).to(torch.bfloat16)
    result = kernels.linear(raw, "PTQ1_0", list(shape), x, None, torch.bfloat16)
    assert torch.isfinite(result).all()
    assert torch.equal(result, expected), f"PTQ1_0 {shape} {rows} rows: {(result != expected).sum().item()} elements differ"


def reference(raw, name, shape, x):
    return kernels.linear(raw, name, list(shape), x, None, torch.float32).to(torch.bfloat16)


@pytest.mark.parametrize("name", TYPES)
@pytest.mark.parametrize("shape", [(1024, 5120), (1152, 5120), (6240, 5120), (1024, 17408)])
@pytest.mark.parametrize("rows", [9, 129, 1000, 1024, 2048])
def test_bf16_matches_fp32_rounded(name, shape, rows):
    if name in ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S") and shape[1] == 17408:
        pytest.skip("one wide shape per family is enough")
    raw = packed(name, *shape)
    x = torch.randn((rows, shape[1]), device="cuda", dtype=torch.bfloat16)
    expected = reference(raw, name, shape, x)
    result = kernels.linear(raw, name, list(shape), x, None, torch.bfloat16)
    assert result.dtype == torch.bfloat16 and result.shape == expected.shape
    assert torch.isfinite(result).all()
    assert torch.equal(result, expected), f"{name} {shape} {rows} rows: {(result != expected).sum().item()} elements differ"


def test_bias_added_after_rounding():
    raw = packed("Q4_K", 1152, 5120)
    x = torch.randn((1000, 5120), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(1152, device="cuda", dtype=torch.bfloat16)
    expected = (kernels.linear(raw, "Q4_K", [1152, 5120], x, None, torch.float32).to(torch.bfloat16) + bias)
    assert torch.equal(kernels.linear(raw, "Q4_K", [1152, 5120], x, bias, torch.bfloat16), expected)


def test_fp16_output_keeps_the_fp32_path():
    raw = packed("Q4_K", 1152, 5120)
    x = torch.randn((1000, 5120), device="cuda", dtype=torch.float16)
    expected = kernels.linear(raw, "Q4_K", [1152, 5120], x, None, torch.float32).to(torch.float16)
    assert torch.equal(kernels.linear(raw, "Q4_K", [1152, 5120], x, None, torch.float16), expected)


def test_graph_replay():
    raw = packed("Q6_K", 6240, 5120)
    x = torch.randn((1000, 5120), device="cuda", dtype=torch.bfloat16)
    expected = kernels.linear(raw, "Q6_K", [6240, 5120], x, None, torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        out = kernels.linear(raw, "Q6_K", [6240, 5120], x, None, torch.bfloat16)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
