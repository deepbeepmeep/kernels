import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("fla")
from fla.modules.convolution import causal_conv1d_update
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from shared.llm_engines.nanovllm.layers.speculative_state import conv_verify, recurrent_verify


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("tokens", [2, 3, 5, 7, 9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_verify_preserves_every_committable_prefix(tokens, dtype):
    torch.manual_seed(912)
    batch, channels, width, heads, value_heads, dim_k, dim_v = 2, 96, 4, 4, 12, 64, 32
    x = torch.randn(batch, tokens, channels, device="cuda", dtype=dtype)
    conv_state = torch.randn(batch, channels, width, device="cuda", dtype=dtype)
    weight = torch.randn(channels, width, device="cuda", dtype=dtype)
    bias = torch.randn(channels, device="cuda", dtype=dtype)
    conv_snapshots = torch.empty(tokens - 1, batch, channels, width, device="cuda", dtype=dtype)
    expected_state = conv_state.clone()
    expected_outputs, expected_snapshots = [], []
    for index in range(tokens):
        output, expected_state = causal_conv1d_update(x[:, index:index + 1], expected_state, weight=weight, bias=bias, activation="silu")
        expected_outputs.append(output)
        expected_snapshots.append(expected_state.clone())
    output, last_state = conv_verify(x, conv_state, weight, bias, conv_snapshots)
    torch.testing.assert_close(output, torch.cat(expected_outputs, 1), atol=0, rtol=0)
    torch.testing.assert_close(conv_snapshots, torch.stack(expected_snapshots[:-1]), atol=0, rtol=0)
    assert last_state.data_ptr() == conv_state.data_ptr()
    torch.testing.assert_close(last_state, expected_snapshots[-1], atol=0, rtol=0)

    q, k = [torch.randn(batch, tokens, heads, dim_k, device="cuda", dtype=dtype) for _ in range(2)]
    v = torch.randn(batch, tokens, value_heads, dim_v, device="cuda", dtype=dtype)
    g = -torch.rand(batch, tokens, value_heads, device="cuda")
    beta = torch.rand(batch, tokens, value_heads, device="cuda", dtype=dtype)
    initial = torch.randn(batch, value_heads, dim_k, dim_v, device="cuda", dtype=dtype)
    initial_before = initial.clone()
    snapshots = torch.empty(tokens - 1, *initial.shape, device="cuda", dtype=torch.float32)
    expected_state = initial.clone()
    expected_outputs, expected_snapshots = [], []
    for index in range(tokens):
        output, expected_state = fused_recurrent_gated_delta_rule(q[:, index:index + 1], k[:, index:index + 1], v[:, index:index + 1], g=g[:, index:index + 1], beta=beta[:, index:index + 1], initial_state=expected_state, output_final_state=True, use_qk_l2norm_in_kernel=True)
        expected_outputs.append(output)
        expected_snapshots.append(expected_state.clone())
    output, last_state = recurrent_verify(q, k, v, g, beta, initial, snapshots)
    torch.testing.assert_close(output, torch.cat(expected_outputs, 1), atol=1e-5, rtol=torch.finfo(dtype).eps * 1.1)
    expected_snapshots = torch.stack(expected_snapshots)
    torch.testing.assert_close(snapshots, expected_snapshots[:-1], atol=1e-6, rtol=1e-5)
    assert (snapshots - expected_snapshots[:-1]).norm() / expected_snapshots[:-1].norm() < 1e-6
    assert last_state.data_ptr() == initial.data_ptr()
    torch.testing.assert_close(last_state, expected_snapshots[-1].to(dtype), atol=1e-6, rtol=torch.finfo(dtype).eps * 1.1)
    stored_snapshots = torch.empty_like(snapshots, dtype=dtype)
    initial.copy_(initial_before)
    replay_output, _ = recurrent_verify(q, k, v, g, beta, initial, stored_snapshots)
    torch.testing.assert_close(replay_output, output, atol=0, rtol=0)
    torch.testing.assert_close(stored_snapshots, expected_snapshots[:-1].to(dtype), atol=1e-6, rtol=torch.finfo(dtype).eps * 1.1)
