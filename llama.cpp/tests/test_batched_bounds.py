"""The batched kernel takes a weight row by steps of 256 values: a weight of 32-value blocks whose width is not a multiple of 256 must
not be read past its end (the bytes after it changed the last outputs, or made them NaN), whatever the rows of the call."""
import numpy as np
import pytest
import torch

import llamacpp_gguf_cuda as kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
BLOCK_BYTES = {"Q4_0": 18, "Q4_1": 20, "Q5_0": 22, "Q5_1": 24, "Q8_0": 34, "IQ4_NL": 18}


@pytest.mark.parametrize("qtype", list(BLOCK_BYTES))
@pytest.mark.parametrize("features,columns", [(37, 640), (37, 96), (3, 32), (1, 96), (37, 992), (37, 1280)])
@torch.inference_mode()
def test_outputs_do_not_depend_on_the_bytes_after_the_weight(qtype, features, columns):
    if not kernels.may_support_linear_qtype_name(qtype):
        pytest.skip(f"{qtype} is not served")
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    generator = torch.Generator().manual_seed(features * columns)
    raw = torch.randint(0, 0x40, (features * columns // 32 * BLOCK_BYTES[qtype],), generator=generator, dtype=torch.uint8)  # every FP16 scale finite
    dense = torch.from_numpy(dequantize(raw.numpy().reshape(-1, BLOCK_BYTES[qtype]), GGMLQuantizationType[qtype]).astype(np.float64)).reshape(features, columns).cuda()
    buffers = []
    for fill in (0x00, 0xFF):  # 0xFFFF is an FP16 NaN
        buffer = torch.full((raw.numel() + (1 << 16),), fill, dtype=torch.uint8, device="cuda")
        buffer[:raw.numel()] = raw.cuda()
        buffers.append(buffer[:raw.numel()])
    bias = torch.randn(features, generator=generator).cuda()
    for rows in (1, 2, 8, 9, 16, 64, 300):
        x = torch.randn(rows, columns, generator=generator).cuda().to(torch.bfloat16)
        zero, ones = (kernels.linear(buffer, qtype, [features, columns], x, None, torch.float32) for buffer in buffers)
        assert bool(torch.isfinite(ones).all()) and torch.equal(zero, ones), f"{rows} rows"
        if rows * features >= 64:  # a relative error means something over enough outputs
            reference = x.double() @ dense.T
            assert float((zero.double() - reference).norm() / reference.norm()) < 2e-2, f"{rows} rows"
        biased = kernels.linear(buffers[1], qtype, [features, columns], x, bias, torch.float32)
        torch.testing.assert_close(biased, zero + bias, rtol=1e-5, atol=1e-5)
    assert kernels.has_bounded_batched_reads() and not kernels.linear_chunked_supported(qtype, [features, columns], torch.bfloat16, False, 0) or columns % 256 == 0
