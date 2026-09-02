import math

import torch

import llamacpp_gguf_cuda


def _reference(query, key_cache, value_cache, block_table, context_lens, scale):
    outputs = []
    query_count = query.shape[0]
    for query_index in range(query_count):
        sequence = query_index if block_table.shape[0] == query_count else 0
        context = int(context_lens[sequence]) - (query_count - 1 - query_index if block_table.shape[0] == 1 else 0)
        physical = block_table[sequence, :math.ceil(context / key_cache.shape[1])].long()
        keys = key_cache[physical].reshape(-1, key_cache.shape[-2], key_cache.shape[-1])[:context]
        values = value_cache[physical].reshape(-1, value_cache.shape[-2], value_cache.shape[-1])[:context]
        groups = query.shape[1] // key_cache.shape[-2]
        keys = keys.repeat_interleave(groups, dim=1)
        values = values.repeat_interleave(groups, dim=1)
        scores = torch.einsum("hd,thd->ht", query[query_index].float(), keys.float()) * scale
        outputs.append(torch.einsum("ht,thd->hd", scores.softmax(dim=-1), values.float()))
    return torch.stack(outputs).to(query.dtype).unsqueeze(1)


def run_case(dtype, query_count):
    torch.manual_seed(19)
    page, query_heads, kv_heads, head_dim = 256, 24, 4, 256
    context = 777
    blocks = math.ceil(context / page)
    query = torch.randn(query_count, query_heads, head_dim, device="cuda", dtype=dtype)
    keys = torch.randn(blocks, page, kv_heads, head_dim, device="cuda", dtype=dtype)
    values = torch.randn_like(keys)
    block_table = torch.arange(blocks, device="cuda", dtype=torch.int32).unsqueeze(0)
    context_lens = torch.tensor([context], device="cuda", dtype=torch.int32)
    scale = head_dim**-0.5
    actual = llamacpp_gguf_cuda.dense_paged_attention(query, keys, values, block_table, context_lens, scale)
    expected = _reference(query, keys, values, block_table, context_lens, scale)
    torch.testing.assert_close(actual, expected, atol=2e-3 if dtype == torch.float16 else 2e-2, rtol=2e-3 if dtype == torch.float16 else 2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = llamacpp_gguf_cuda.dense_paged_attention(query, keys, values, block_table, context_lens, scale)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, actual, atol=0, rtol=0)


if __name__ == "__main__":
    for test_dtype in (torch.float16, torch.bfloat16):
        for queries in (1, 2, 3):
            run_case(test_dtype, queries)
            print(f"passed dtype={test_dtype} queries={queries}")
