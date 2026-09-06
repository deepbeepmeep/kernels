import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")
pytest.importorskip("fla")

from shared.llm_engines.nanovllm.layers import activation, attention, speculative_state
from test_q8_paged_attention import _case, _reference


pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or activation.triton is None, reason="CUDA and Triton required")


def _record_kernels(monkeypatch, module, names):
    hashes = {name: set() for name in names}
    for name in names:
        kernel = getattr(module, name)
        original = kernel.run

        def run(*args, _original=original, _name=name, **kwargs):
            compiled = _original(*args, **kwargs)
            hashes[_name].add(compiled.hash)
            return compiled

        monkeypatch.setattr(kernel, "run", run)
    return hashes


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_silu_shares_one_kernel_across_length_and_alignment_changes(monkeypatch, dtype):
    hashes = _record_kernels(monkeypatch, activation, ["_silu_mul_kernel"])
    for rows in [1, 2, 3, 15, 16, 17, 31, 32, 33, 511, 512, 513]:
        x = torch.randn(rows, 190, device="cuda", dtype=dtype)
        expected = F.silu(x[:, :95]) * x[:, 95:]
        actual = activation.SiluAndMul().forward_list([x])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(hashes["_silu_mul_kernel"]) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_state_kernels_share_batch_and_token_lengths(monkeypatch, dtype):
    hashes = _record_kernels(monkeypatch, speculative_state, ["conv_verify_kernel", "recurrent_verify_kernel"])
    for batch in [1, 2, 3]:
        for tokens in [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17]:
            x = torch.randn(batch, tokens, 96, device="cuda", dtype=dtype)
            state = torch.randn(batch, 96, 4, device="cuda", dtype=dtype)
            weight = torch.randn(96, 4, device="cuda", dtype=dtype)
            snapshots = torch.empty(tokens - 1, *state.shape, device="cuda", dtype=dtype)
            output, final = speculative_state.conv_verify(x, state, weight, None, snapshots)
            assert torch.isfinite(output).all() and torch.isfinite(final).all() and torch.isfinite(snapshots).all()
            q, k = [torch.randn(batch, tokens, 4, 64, device="cuda", dtype=dtype) for _ in range(2)]
            v = torch.randn(batch, tokens, 12, 32, device="cuda", dtype=dtype)
            g = -torch.rand(batch, tokens, 12, device="cuda")
            beta = torch.rand(batch, tokens, 12, device="cuda", dtype=dtype)
            state = torch.randn(batch, 12, 64, 32, device="cuda", dtype=dtype)
            snapshots = torch.empty(tokens - 1, *state.shape, device="cuda", dtype=dtype)
            output, final = speculative_state.recurrent_verify(q, k, v, g, beta, state, snapshots)
            assert torch.isfinite(output).all() and torch.isfinite(final).all() and torch.isfinite(snapshots).all()
    assert all(len(values) == 1 for values in hashes.values())


@torch.inference_mode()
def test_attention_reuses_fixed_ranges_across_queries_and_table_capacities(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    hashes = _record_kernels(monkeypatch, attention, ["q8_paged_prefill_kernel", "q8_grouped_partials_kernel", "q8_grouped_reduce_kernel"])
    for queries, widths in [([1, 3, 9], [1]), ([2, 4, 5, 6, 7, 8], [3, 16, 17, 128])]:
        for batch in [1, 2]:
            for tokens in queries:
                q, k, v, ks, vs, context = _case(torch.bfloat16, 64, [tokens] * batch, [221] * batch)
                expected = _reference(q, k, v, ks, vs, context)
                for width in widths:
                    context.block_tables = F.pad(context.block_tables[:, :1].contiguous(), (0, width - 1), value=-1)
                    actual = attention._q8_grouped_attention(q, k, v, ks, vs, context.block_tables, context.context_lens, 64 ** -.5)
                    torch.testing.assert_close(actual, expected, atol=.0003, rtol=.009)
                    actual = attention._q8_paged_prefill(q, k, v, ks, vs, context, 64 ** -.5)
                    torch.testing.assert_close(actual, expected, atol=.0003, rtol=.009)
        counts = {name: len(values) for name, values in hashes.items()}
        if widths == [1]:
            warmed = counts
        else:
            assert counts == warmed
    assert counts["q8_paged_prefill_kernel"] == 1
    assert counts["q8_grouped_partials_kernel"] <= 16
    assert counts["q8_grouped_reduce_kernel"] <= 8
