from types import SimpleNamespace

import pytest
import torch

from shared.llm_engines.nanovllm.layers import attention


pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or attention.triton is None, reason="CUDA and Triton required")


@pytest.fixture(autouse=True)
def float32_reference(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def _case(dtype, head_dim, query_lengths, context_lengths, heads=24, kv_heads=4):
    torch.manual_seed(41)
    pages = [(length + 255) // 256 for length in context_lengths]
    key = torch.randint(-127, 128, (sum(pages), 256, kv_heads, head_dim), device="cuda", dtype=torch.int8)
    value = torch.randint(-127, 128, key.shape, device="cuda", dtype=torch.int8)
    key_scale = torch.rand((*key.shape[:-1], head_dim // 32), device="cuda", dtype=torch.float16) * .015 + .005
    value_scale = torch.rand_like(key_scale) * .015 + .005
    query = torch.randn((sum(query_lengths), heads, head_dim), device="cuda", dtype=dtype)
    tables = torch.full((len(pages), max(pages)), -1, device="cuda", dtype=torch.int32)
    permutation = torch.randperm(sum(pages), device="cuda", dtype=torch.int32)
    offset = 0
    for index, count in enumerate(pages):
        tables[index, :count] = permutation[offset:offset + count]
        offset += count
    lengths = torch.tensor(context_lengths, device="cuda", dtype=torch.int32)
    cu_q = torch.tensor([0] + query_lengths, device="cuda", dtype=torch.int32).cumsum(0).int()
    cu_k = torch.tensor([0] + context_lengths, device="cuda", dtype=torch.int32).cumsum(0).int()
    return query, key, value, key_scale, value_scale, SimpleNamespace(block_tables=tables, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, context_lens=lengths)


def _reference(query, key, value, key_scale, value_scale, context):
    outputs = []
    heads, dim = query.shape[1:]
    kv_heads = key.shape[2]
    for sequence in range(context.context_lens.numel()):
        begin, end = context.cu_seqlens_q[sequence:sequence + 2].tolist()
        length = int(context.context_lens[sequence])
        blocks = context.block_tables[sequence, :(length + 255) // 256].long()
        k = (key[blocks].float().reshape(-1, kv_heads, dim // 32, 32) * key_scale[blocks].float().reshape(-1, kv_heads, dim // 32, 1)).reshape(-1, kv_heads, dim)[:length]
        v = (value[blocks].float().reshape(-1, kv_heads, dim // 32, 32) * value_scale[blocks].float().reshape(-1, kv_heads, dim // 32, 1)).reshape(-1, kv_heads, dim)[:length]
        q = query[begin:end].float().transpose(0, 1).reshape(kv_heads, heads // kv_heads, end - begin, dim)
        scores = torch.matmul(q, k.permute(1, 2, 0)[:, None]) * dim ** -.5
        causal = torch.arange(length, device=query.device)[None, :] <= length - (end - begin) + torch.arange(end - begin, device=query.device)[:, None]
        scores.masked_fill_(~causal, -float("inf"))
        out = torch.matmul(scores.softmax(-1), v.transpose(0, 1)[:, None])
        outputs.append(out.reshape(heads, end - begin, dim).transpose(0, 1))
    return torch.cat(outputs).to(query.dtype)


@pytest.mark.parametrize("dtype,dim,queries,lengths", [(torch.bfloat16, 256, [17, 33], [257, 531]), (torch.float16, 128, [31], [20000])])
@torch.inference_mode()
def test_prefill_matches_float32_with_ragged_shuffled_pages(dtype, dim, queries, lengths):
    args = _case(dtype, dim, queries, lengths)
    expected = _reference(*args)
    actual = attention._q8_paged_prefill(*args, dim ** -.5)
    torch.testing.assert_close(actual, expected, atol=.0003, rtol=.009)


@pytest.mark.parametrize("dtype,dim,queries,lengths,heads,kv_heads", [(torch.bfloat16, 256, 3, [20000], 24, 4), (torch.float16, 128, 2, [257, 531], 8, 2), (torch.float16, 64, 1, [257], 4, 4)])
@torch.inference_mode()
def test_grouped_decode_and_verification_graph_reuse(dtype, dim, queries, lengths, heads, kv_heads):
    q, k, v, ks, vs, context = _case(dtype, dim, [queries] * len(lengths), lengths, heads, kv_heads)
    def run():
        return attention._q8_grouped_attention(q, k, v, ks, vs, context.block_tables, context.context_lens, dim ** -.5)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for length in [min(lengths), queries]:
        context.context_lens.fill_(length)
        graph.replay()
        expected = _reference(q, k, v, ks, vs, context)
        torch.testing.assert_close(actual, expected, atol=.0003, rtol=.009)
    context.context_lens.zero_()
    graph.replay()
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
