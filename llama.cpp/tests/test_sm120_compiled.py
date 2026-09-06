"""Standalone compiled SM120 checks: no WanGP or Triton import required."""
from types import SimpleNamespace
import hashlib
import torch
from llamacpp_gguf_cuda import sm120

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

@torch.inference_mode()
def run_cases():
    torch.backends.cuda.matmul.allow_tf32 = False
    count = 0
    assert len(sm120._MANIFEST['kernels']) == 4
    for key, spec in sm120._MANIFEST['kernels'].items():
        assert hashlib.sha256((sm120._ROOT / (key + '.cubin')).read_bytes()).hexdigest() == spec['sha256']
    for dtype in (torch.float16, torch.bfloat16):
        for queries, heads, kv_heads in ((1, 24, 4), (3, 24, 4), (5, 16, 4), (9, 8, 2), (33, 24, 4)):
            q, k, v, ks, vs, context = _case(dtype, 256, [queries, queries], [531, 991], heads, kv_heads)
            splits = 16
            partial = torch.empty((*q.shape[:2], splits, 256), device='cuda')
            maximum = torch.empty((*q.shape[:2], splits), device='cuda')
            denominator = torch.empty_like(maximum)
            def run():
                prefill = sm120.q8_paged_prefill(q, k, v, ks, vs, context, .0625)
                sm120.q8_grouped_partials(q, k, v, ks, vs, context.block_tables, context.context_lens, partial, maximum, denominator, splits, .0625)
                weights = torch.exp(maximum - maximum.amax(-1, keepdim=True))
                grouped = ((partial * weights.unsqueeze(-1)).sum(-2) / (denominator * weights).sum(-1).clamp_min(1e-20).unsqueeze(-1)).to(dtype)
                return prefill, grouped
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run()
            for iteration in range(2):
                if iteration:
                    q.mul_(.75)
                    context.context_lens.copy_(torch.tensor([257, 769], device='cuda', dtype=torch.int32))
                    context.cu_seqlens_k.copy_(torch.tensor([0, 257, 1026], device='cuda', dtype=torch.int32))
                    context.block_tables[:, :2] = context.block_tables[:, :2].flip(1)
                graph.replay()
                expected = _reference(q, k, v, ks, vs, context)
                for output in actual:
                    torch.testing.assert_close(output, expected, atol=.0003, rtol=.009)
                eager = run()
                for output, reference in zip(actual, eager):
                    torch.testing.assert_close(output, reference, atol=0, rtol=0)
            count += 2
    print(f'passed {count} compiled SM120 attention configurations with graph mutations', flush=True)
    return count


if __name__ == '__main__':
    run_cases()
