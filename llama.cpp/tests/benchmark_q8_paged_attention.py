import argparse
import math

import torch

import llamacpp_gguf_cuda

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None


def _time_ms(callback, warmup=10, repeats=100):
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(repeats):
        callback()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _graph_time_ms(callback, warmup=10, repeats=100):
    callback()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = callback()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats, graph_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--capacity", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--splits", type=int, default=0)
    parser.add_argument("--queries", type=int, default=1)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    block_size, query_heads, kv_heads, head_dim = 256, 24, 4, 256
    capacity = args.capacity or args.tokens
    num_blocks = math.ceil(capacity / block_size)
    query = torch.randn(args.queries, query_heads, head_dim, device="cuda", dtype=dtype)
    key_cache = torch.randint(-127, 128, (num_blocks, block_size, kv_heads, head_dim), device="cuda", dtype=torch.int8)
    value_cache = torch.randint_like(key_cache, -127, 128)
    key_scales = torch.rand(num_blocks, block_size, kv_heads, head_dim // 32, device="cuda", dtype=torch.float16) * 0.02
    value_scales = torch.rand_like(key_scales) * 0.02
    active_blocks = math.ceil(args.tokens / block_size)
    block_table = torch.arange(active_blocks, device="cuda", dtype=torch.int32).unsqueeze(0)
    context_lens = torch.tensor([args.tokens], device="cuda", dtype=torch.int32)
    selected_splits = llamacpp_gguf_cuda.q8_paged_attention_num_splits(query, capacity)
    callback = lambda: llamacpp_gguf_cuda.q8_paged_attention(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, head_dim**-0.5, args.splits)
    native_ms = _time_ms(callback)
    native_graph_ms, native_output = _graph_time_ms(callback)
    print(f"tokens={args.tokens} capacity={capacity} queries={args.queries} dtype={dtype} splits={args.splits or 'auto'} selected_splits={selected_splits} native_q8_ms={native_ms:.3f} native_q8_graph_ms={native_graph_ms:.3f}")
    if flash_attn_with_kvcache is not None and args.queries == 1:
        def materialized_flash():
            keys = key_cache.reshape(num_blocks, block_size, kv_heads, head_dim // 32, 32).to(dtype).mul(key_scales.to(dtype).unsqueeze(-1)).reshape_as(key_cache)
            values = value_cache.reshape(num_blocks, block_size, kv_heads, head_dim // 32, 32).to(dtype).mul(value_scales.to(dtype).unsqueeze(-1)).reshape_as(value_cache)
            return flash_attn_with_kvcache(query.unsqueeze(1), keys, values, cache_seqlens=context_lens, block_table=block_table, softmax_scale=head_dim**-0.5, causal=True)

        fallback_ms = _time_ms(materialized_flash, warmup=3, repeats=20)
        fallback_graph_ms, fallback_output = _graph_time_ms(materialized_flash, warmup=3, repeats=20)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        materialized_flash()
        torch.cuda.synchronize()
        temporary_mib = (torch.cuda.max_memory_allocated() - baseline) / 1024**2
        difference = native_output.float().sub(fallback_output.float()).abs()
        print(f"materialized_flash_ms={fallback_ms:.3f} eager_speedup={fallback_ms / native_ms:.2f}x materialized_flash_graph_ms={fallback_graph_ms:.3f} graph_speedup={fallback_graph_ms / native_graph_ms:.2f}x temporary_vram={temporary_mib:.1f} MiB max_abs_error={difference.max().item():.6g} mean_abs_error={difference.mean().item():.6g} rms_error={difference.square().mean().sqrt().item():.6g}")


if __name__ == "__main__":
    main()
