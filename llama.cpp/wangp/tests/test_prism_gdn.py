import copy
import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("fla")
from fla.modules.convolution import ShortConvolution
from shared.kernels.qwen_gdn import GDNShortConvolution, prepare_decode, install_gdn_decode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.inference_mode()
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_convolution_decode_matches_reference_and_graph_replay(weight_dtype):
    torch.manual_seed(140)
    reference = ShortConvolution(10240, 4, bias=False, device='cuda', dtype=weight_dtype)
    candidate = copy.deepcopy(reference)
    candidate.__class__ = GDNShortConvolution
    x = torch.randn(2, 1, 10240, device='cuda', dtype=torch.bfloat16)
    expected, reference_state = reference(x, output_final_state=True)
    actual, state = candidate(x, output_final_state=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, reference_state, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    original = state.clone()
    with torch.cuda.graph(graph):
        out, _ = candidate(x, cache=state, output_final_state=True)
    state.copy_(original)
    for _ in range(3):
        x.normal_()
        graph.replay()
        expected, reference_state = reference(x, cache=reference_state, output_final_state=True)
        # Different Triton tile layouts can round a few SiLU values by one
        # BF16 ULP. Convolution state must still agree bit for bit.
        torch.testing.assert_close(out, expected, rtol=.008, atol=1e-8)
        assert ((out.float() - expected.float()).norm() / expected.float().norm()) < 1e-4
        torch.testing.assert_close(state, reference_state, rtol=0, atol=0)


@torch.inference_mode()
def test_gdn_preparation_strides_extreme_gates_and_graph_replay():
    torch.manual_seed(141)
    qkv = torch.randn(2, 1, 10240, device='cuda', dtype=torch.bfloat16)
    q = qkv[..., :2048].view(2, 1, 16, 128)
    k = qkv[..., 2048:4096].view(2, 1, 16, 128)
    ab = torch.randn(2, 1, 96, device='cuda', dtype=torch.bfloat16)
    a, b = ab.chunk(2, -1)
    ssm_a = -torch.rand(48, device='cuda', dtype=torch.float32)
    dt = torch.linspace(-100, 100, 48, device='cuda', dtype=torch.float32)
    prepare_decode(q, k, a, b, ssm_a, dt, 48)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = prepare_decode(q, k, a, b, ssm_a, dt, 48)
    for _ in range(3):
        qkv.normal_()
        ab.normal_()
        graph.replay()
        expected = (q.repeat_interleave(3, 2), k.repeat_interleave(3, 2),
                    ssm_a * torch.nn.functional.softplus(a.float() + dt), b.sigmoid())
        for i, (actual, reference) in enumerate(zip(outputs, expected)):
            torch.testing.assert_close(actual, reference, rtol=1e-6 if i == 2 else 0, atol=1e-7 if i == 2 else 0)


@pytest.mark.parametrize("layout", ["grouped", "tiled", "interleaved"])
@torch.inference_mode()
def test_complete_gdn_block_preserves_checkpoint_layout_and_state(layout):
    from types import SimpleNamespace
    from shared.prompt_enhancer.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from shared.llm_engines.nanovllm.models.qwen3_5 import Qwen3_5Block
    from shared.llm_engines.nanovllm.utils.context import set_context, reset_context

    if torch.version.hip is not None or torch.cuda.get_device_capability(0)[0] < 8:
        pytest.skip("Installer requires NVIDIA Ampere or newer")
    config = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32, layer_types=['linear_attention'])
    torch.manual_seed(145)
    with torch.device('cpu'):
        reference = Qwen3_5Block(config, 0).to(dtype=torch.bfloat16)
    reference._gguf_v_head_reordered = layout == "tiled"
    reference._gguf_ssm_param_reordered = layout == "tiled"
    reference._gguf_interleave_ssm_ab = layout == "interleaved"
    optimized = copy.deepcopy(reference)
    install_gdn_decode(SimpleNamespace(blk=[optimized]))
    assert optimized._gdn_prepare_decode is not None
    for block in (reference, optimized):
        block.cuda()
        block.prepare_sequence_state(1, torch.device('cuda'), torch.bfloat16)
        block.prepare_speculative_state(3)
    try:
        for tokens in (1, 1, 3, 1):
            x = torch.randn(1, tokens, 128, device='cuda', dtype=torch.bfloat16) * .1
            results = []
            for block in (reference, optimized):
                set_context(False, has_previous_state=True, speculative_verify=tokens > 1)
                results.append(block._forward_linear_attention([x.clone()], 0, None, None))
            torch.testing.assert_close(results[1], results[0], atol=.002, rtol=.02)
            torch.testing.assert_close(optimized.conv_state_buffer, reference.conv_state_buffer, atol=0, rtol=0)
            torch.testing.assert_close(optimized.recurrent_state_buffer, reference.recurrent_state_buffer, atol=.002, rtol=.02)
    finally:
        reset_context()


@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_raw_gate_recurrence_preserves_outputs_and_every_prefix(tokens, dtype):
    from shared.kernels.qwen_gdn import recurrent_raw_gates
    from shared.llm_engines.nanovllm.layers.speculative_state import recurrent_verify
    torch.manual_seed(715)
    batch, heads, hv, dim = 2, 2, 6, 128
    qkv = torch.randn(batch, tokens, (heads * 2 + hv) * dim, device="cuda", dtype=dtype) * .2
    q, k, v = torch.split(qkv, [heads * dim, heads * dim, hv * dim], dim=-1)
    q, k = (x.view(batch, tokens, heads, dim) for x in (q, k))
    v = v.view(batch, tokens, hv, dim)
    ab = torch.randn(batch, tokens, hv * 2, device="cuda", dtype=dtype)
    a, b = ab.chunk(2, -1)
    ssm_a = -torch.rand(hv, device="cuda", dtype=torch.float32)
    dt = torch.linspace(-25, 25, hv, device="cuda", dtype=torch.float32)
    for state_dtype in (dtype, torch.float32):
        initial = torch.randn(batch, hv, dim, dim, device="cuda", dtype=state_dtype) * .1
        reference = initial.clone()
        snapshots = torch.empty(tokens - 1, *initial.shape, device="cuda", dtype=state_dtype)
        reference_snapshots = torch.empty_like(snapshots)
        g = ssm_a * torch.nn.functional.softplus(a.float() + dt)
        expected, _ = recurrent_verify(q, k, v, g, b.sigmoid(), reference, reference_snapshots)
        actual, state = recurrent_raw_gates(q, k, v, a, b, ssm_a, dt, initial, snapshots if tokens > 1 else None)
        assert state is initial
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=torch.finfo(dtype).eps * 1.1)
        torch.testing.assert_close(state, reference, atol=2e-5, rtol=torch.finfo(state_dtype).eps * 1.1)
        torch.testing.assert_close(snapshots, reference_snapshots, atol=2e-5, rtol=torch.finfo(state_dtype).eps * 1.1)
        if state_dtype == torch.float32:
            assert (state - reference).norm() / reference.norm() < 2e-6


@pytest.mark.parametrize("tokens", [1, 8])
@torch.inference_mode()
def test_raw_gate_graph_replay_changed_inputs_and_rollback(tokens):
    from shared.kernels.qwen_gdn import recurrent_raw_gates
    from shared.llm_engines.nanovllm.layers.speculative_state import recurrent_verify
    q = torch.randn(1, tokens, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, 6, 128, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(1, tokens, 6, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    sa = -torch.rand(6, device="cuda")
    dt = torch.randn(6, device="cuda")
    state = torch.zeros(1, 6, 128, 128, device="cuda", dtype=torch.bfloat16)
    snapshots = torch.empty(tokens - 1, *state.shape, device="cuda", dtype=state.dtype)
    recurrent_raw_gates(q, k, v, a, b, sa, dt, state, snapshots if tokens > 1 else None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, _ = recurrent_raw_gates(q, k, v, a, b, sa, dt, state, snapshots if tokens > 1 else None)
    for _ in range(3):
        if tokens > 1:
            state.copy_(snapshots[2])
        reference = state.clone()
        reference_snapshots = torch.empty_like(snapshots)
        q.normal_(); a.normal_(); b.normal_()
        graph.replay()
        expected, _ = recurrent_verify(q, k, v, sa * torch.nn.functional.softplus(a.float() + dt), b.sigmoid(), reference, reference_snapshots)
        torch.testing.assert_close(output, expected, atol=.002, rtol=.008)
        torch.testing.assert_close(state, reference, atol=.002, rtol=.008)
        torch.testing.assert_close(snapshots, reference_snapshots, atol=.002, rtol=.008)


@pytest.mark.parametrize("rows", [1, 3, 8, 9])
@pytest.mark.parametrize("residual", [False, True])
@torch.inference_mode()
def test_small_batch_norm_launch_preserves_results_and_replay(rows, residual):
    from shared.llm_engines.nanovllm.layers.layernorm import RMSNorm
    torch.manual_seed(48)
    norm = RMSNorm(5120).to(device='cuda')
    x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)
    r = torch.randn_like(x) if residual else None
    expected = norm(x, r)
    norm._small_batch_num_warps = 16
    norm(x, r)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = norm(x, r)
    for _ in range(3):
        x.normal_()
        graph.replay()
        norm._small_batch_num_warps = 4
        expected = norm(x, r)
        norm._small_batch_num_warps = 16
        if residual:
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=0, rtol=0)
        else:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
