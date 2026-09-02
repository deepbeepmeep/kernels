import math
import statistics

import torch
from flash_attn import flash_attn_with_kvcache
from llamacpp_gguf_cuda import dense_paged_attention


def elapsed_ms(fn, warmup=20, iterations=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def run(context, queries=1, dtype=torch.bfloat16):
    page, q_heads, kv_heads, dim = 256, 24, 4, 256
    blocks = math.ceil(context / page)
    q = torch.randn(queries, q_heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(blocks, page, kv_heads, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    table = torch.arange(blocks, device="cuda", dtype=torch.int32).unsqueeze(0)
    lengths = torch.tensor([context], device="cuda", dtype=torch.int32)
    scale = dim**-0.5
    ours = lambda: dense_paged_attention(q, k, v, table, lengths, scale)
    flash = lambda: flash_attn_with_kvcache(q.unsqueeze(0), k, v, cache_seqlens=lengths, block_table=table, softmax_scale=scale, causal=True).squeeze(0)
    actual, expected = ours().squeeze(1), flash()
    torch.cuda.synchronize()
    error = (actual.float() - expected.float()).abs()
    print(f"context={context:5d} q={queries} dense={elapsed_ms(ours):.4f} ms flash={elapsed_ms(flash):.4f} ms max_err={error.max().item():.5f} mean_err={error.mean().item():.6f}")


if __name__ == "__main__":
    print(torch.cuda.get_device_name(), torch.version.cuda)
    for query_count in (1, 2, 3):
        for context_length in (512, 1536, 4096, 8192, 16384, 32000):
            run(context_length, query_count)
    page, q_heads, kv_heads, dim, context = 256, 24, 4, 256, 32000
    blocks = math.ceil(context / page)
    q = torch.randn(1, q_heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(blocks, page, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.arange(blocks, device="cuda", dtype=torch.int32).unsqueeze(0)
    lengths = torch.tensor([context], device="cuda", dtype=torch.int32)
    for splits in (1, 2, 4, 8, 16, 32):
        fn = lambda splits=splits: dense_paged_attention(q, k, v, table, lengths, dim**-0.5, splits)
        print(f"32K forced_splits={splits:2d}: {elapsed_ms(fn):.4f} ms")
