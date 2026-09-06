import pytest
import torch
import torch.nn.functional as F

from shared.llm_engines.nanovllm.layers.activation import SiluAndMul, triton


@pytest.mark.skipif(not torch.cuda.is_available() or triton is None, reason="CUDA and Triton required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_silu_preserves_rounding_for_every_finite_16bit_input(dtype):
    gate = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16).view(dtype)
    gate = gate[torch.isfinite(gate)]
    value = torch.randn(gate.shape, dtype=dtype, device="cuda", generator=torch.Generator(device="cuda").manual_seed(41))
    inputs = [torch.stack((gate, value), -1)]
    actual = SiluAndMul().forward_list(inputs)
    torch.testing.assert_close(actual[:, 0], F.silu(gate) * value, atol=0, rtol=0, equal_nan=True)
    assert inputs == []


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_silu_list_handles_strides_and_repeated_graphs(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = torch.randn((3, 2, 10240), dtype=dtype, device=device).transpose(0, 1)
    module = SiluAndMul()

    def run():
        # The CPU implementation consumes its input in place.
        return module.forward_list([source.clone()])

    run()
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
    for scale in [1.0, 3.0]:
        source.mul_(scale)
        if device == "cuda":
            graph.replay()
        else:
            actual = run()
        gate, value = source.chunk(2, -1)
        torch.testing.assert_close(actual, F.silu(gate) * value, atol=0, rtol=0)
        assert actual.is_contiguous()
