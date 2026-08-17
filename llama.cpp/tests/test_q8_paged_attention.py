import math

import torch

import llamacpp_gguf_cuda


Q8_BLOCK_SIZE = 32


def _quantize(cache: torch.Tensor):
    blocks = cache.reshape(*cache.shape[:-1], cache.shape[-1] // Q8_BLOCK_SIZE, Q8_BLOCK_SIZE)
    scales = blocks.abs().amax(dim=-1).div(127).clamp_min(1e-8).to(torch.float16)
    quantized = blocks.div(scales.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8).reshape_as(cache)
    return quantized, scales


def _reference(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, scale):
    outputs = []
    q8_blocks = query.shape[-1] // Q8_BLOCK_SIZE
    for batch, sequence_length in enumerate(context_lens.tolist()):
        block_count = math.ceil(sequence_length / key_cache.shape[1])
        physical_blocks = block_table[batch, :block_count].long()
        keys = key_cache[physical_blocks].reshape(-1, key_cache.shape[-2], key_cache.shape[-1])[:sequence_length]
        values = value_cache[physical_blocks].reshape(-1, value_cache.shape[-2], value_cache.shape[-1])[:sequence_length]
        key_scale = key_scales[physical_blocks].reshape(-1, key_cache.shape[-2], q8_blocks)[:sequence_length]
        value_scale = value_scales[physical_blocks].reshape(-1, value_cache.shape[-2], q8_blocks)[:sequence_length]
        keys = keys.reshape(sequence_length, key_cache.shape[-2], q8_blocks, Q8_BLOCK_SIZE).float().mul(key_scale.float().unsqueeze(-1)).reshape(sequence_length, key_cache.shape[-2], -1)
        values = values.reshape(sequence_length, value_cache.shape[-2], q8_blocks, Q8_BLOCK_SIZE).float().mul(value_scale.float().unsqueeze(-1)).reshape(sequence_length, value_cache.shape[-2], -1)
        groups = query.shape[1] // key_cache.shape[-2]
        keys = keys.repeat_interleave(groups, dim=1)
        values = values.repeat_interleave(groups, dim=1)
        logits = torch.einsum("hd,thd->ht", query[batch].float(), keys) * scale
        outputs.append(torch.einsum("ht,thd->hd", logits.softmax(dim=-1), values))
    return torch.stack(outputs).to(query.dtype).unsqueeze(1)


def run_case(dtype: torch.dtype, head_dim: int):
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch, query_heads, kv_heads = 2, 16, 4
    cache_block_size, num_cache_blocks = 256, 8
    query = torch.randn(batch, query_heads, head_dim, device=device, dtype=dtype)
    keys = torch.randn(num_cache_blocks, cache_block_size, kv_heads, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    key_cache, key_scales = _quantize(keys)
    value_cache, value_scales = _quantize(values)
    block_table = torch.tensor([[5, 1, 7, 0], [3, 6, 2, -1]], dtype=torch.int32, device=device)
    context_lens = torch.tensor([777, 513], dtype=torch.int32, device=device)
    scale = head_dim**-0.5
    actual = llamacpp_gguf_cuda.q8_paged_attention(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, scale)
    expected = _reference(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, scale)
    torch.testing.assert_close(actual, expected, atol=2e-3 if dtype == torch.float16 else 2e-2, rtol=2e-3 if dtype == torch.float16 else 2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = llamacpp_gguf_cuda.q8_paged_attention(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, scale)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, actual, atol=0, rtol=0)


def run_speculative_case(dtype: torch.dtype):
    torch.manual_seed(11)
    device = torch.device("cuda")
    query_count, query_heads, kv_heads, head_dim = 5, 24, 4, 256
    cache_block_size, num_cache_blocks = 256, 4
    query = torch.randn(query_count, query_heads, head_dim, device=device, dtype=dtype)
    keys = torch.randn(num_cache_blocks, cache_block_size, kv_heads, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    key_cache, key_scales = _quantize(keys)
    value_cache, value_scales = _quantize(values)
    block_table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device=device)
    context_lens = torch.tensor([777], dtype=torch.int32, device=device)
    scale = head_dim**-0.5
    actual = llamacpp_gguf_cuda.q8_paged_attention(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, scale)
    expected = torch.cat([
        _reference(query[index:index + 1], key_cache, value_cache, key_scales, value_scales, block_table, context_lens - (query_count - 1 - index), scale)
        for index in range(query_count)
    ])
    torch.testing.assert_close(actual, expected, atol=2e-3 if dtype == torch.float16 else 2e-2, rtol=2e-3 if dtype == torch.float16 else 2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = llamacpp_gguf_cuda.q8_paged_attention(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, scale)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, actual, atol=0, rtol=0)


if __name__ == "__main__":
    for dtype in (torch.float16, torch.bfloat16):
        for head_dim in (128, 256):
            run_case(dtype, head_dim)
            print(f"passed dtype={dtype} head_dim={head_dim}")
        run_speculative_case(dtype)
        print(f"passed speculative dtype={dtype}")
