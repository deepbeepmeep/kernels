"""Compare the new short-batch fused GGUF path with the established path."""
import os

import torch
import torch.nn.functional as F

import llamacpp_gguf_cuda as kernels
from validate_release import packed_fixture
from test_ptq1 import fixture as ptq1_fixture


@torch.inference_mode()
def main():
    os.environ["WGP_GGUF_LLAMACPP_CUDA_MATMUL_MODE"] = "mmq"
    torch.manual_seed(709)
    checks = 0
    for qtype in ("Q4_K", "Q6_K", "IQ3_S", "PTQ1_0"):
        if qtype == "PTQ1_0":
            packed, _ = ptq1_fixture(128, 512)
        else:
            packed, _ = packed_fixture(qtype)
        weight = torch.from_numpy(packed.copy()).flatten().to("cuda")
        for dtype in (torch.float16, torch.bfloat16):
            for rows in (1, 2, 4, 7):
                if not kernels.supports_linear_fusions(qtype, rows, torch.cuda.current_device()):
                    continue
                x = torch.randn(rows, 512, device="cuda", dtype=dtype) * 0.1
                expected = kernels.linear(weight, qtype, (128, 512), x, None, dtype)
                actual = kernels.linear(weight, qtype, (128, 512), x, None, dtype, fused_output=True)
                torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
                torch.testing.assert_close(actual, kernels.linear(weight, qtype, (128, 512), x, None, dtype, fused_output=True), atol=0, rtol=0)
                checks += 1
                bias = torch.randn(128, device="cuda", dtype=dtype) * 0.01
                expected_bias = kernels.linear(weight, qtype, (128, 512), x, bias, dtype)
                actual_bias = kernels.linear(weight, qtype, (128, 512), x, bias, dtype, fused_output=True)
                torch.testing.assert_close(actual_bias, expected_bias, atol=0.01, rtol=0.01)
                checks += 1

            gate = torch.randn(1, 512, device="cuda", dtype=dtype) * 0.1
            value = torch.randn(1, 512, device="cuda", dtype=dtype) * 0.1
            x = torch.cat((gate, value), dim=-1)
            expected = kernels.linear(weight, qtype, (128, 512), F.silu(gate) * value, None, dtype)
            actual = kernels.linear(weight, qtype, (128, 512), x, None, dtype, fused_output=True, silu_mul=True)
            torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = kernels.linear(weight, qtype, (128, 512), x, None, dtype, fused_output=True, silu_mul=True)
            graph.replay()
            torch.testing.assert_close(captured, actual, atol=0, rtol=0)
            checks += 1
    assert checks, "No fused kernel path was exercised"
    print(f"PASS: {checks} fused GGUF configurations and graph replay", flush=True)


if __name__ == "__main__":
    main()
