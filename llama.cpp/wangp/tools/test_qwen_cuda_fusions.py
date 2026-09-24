"""GPU numerical comparisons and timings for the optional Qwen CUDA fusions."""
import argparse
import gc
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
import gguf
import llamacpp_gguf_cuda as native
from shared.llm_engines.nanovllm.layers.activation import SiluAndMul
from shared.llm_engines.nanovllm.layers.attention import store_kvcache
from shared.llm_engines.nanovllm.models.qwen3_5 import apply_rotary_pos_emb
from shared.kernels import qwen_rope_cache as rope

spec = importlib.util.spec_from_file_location('validate_release', 'E:/ML/kernels/llama.cpp/tests/validate_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def timed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):
            fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(7):
        start.record()
        for _ in range(30):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (16 * 30))
    graph.reset()
    return statistics.median(values)


def allocated_peak(fn):
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def create_baseline(path):
    torch.manual_seed(713)
    cases = []
    activation = SiluAndMul()
    for name in release.QTYPES:
        packed, _ = release.packed_fixture(name, rows=128, columns=512)
        raw = torch.from_numpy(packed).flatten().to('cuda')
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for rows in (1, 3, 5, 8):
                for silu in (False, True):
                    width = 1024 if silu else 512
                    source = torch.randn(rows, width * 2, dtype=dtype, device='cuda') * .2
                    x = source[:, ::2].contiguous()
                    bias = torch.randn(128, dtype=dtype, device='cuda') * .1
                    activated = activation.forward_list([x]) if silu else x
                    expected = native.linear(raw, name, (128, 512), activated, bias, dtype)
                    cases.append(dict(qtype=name, raw=raw.cpu(), x=x.cpu(), bias=bias.cpu(), silu=silu, expected=expected.cpu()))
        print('BASELINE', name, flush=True)
    torch.save(cases, path)
    return dict(cases=len(cases), package=native.__file__)


def compare_native(path):
    cases = torch.load(path, weights_only=True)
    fused_cases = 0
    for case in cases:
        name, silu = case['qtype'], case['silu']
        x, raw, bias = (case[key].to('cuda') for key in ('x', 'raw', 'bias'))
        expected = case['expected'].to('cuda')
        activated = SiluAndMul().forward_list([x]) if silu else x
        ordinary = native.linear(raw, name, (128, 512), activated, bias, x.dtype)
        torch.testing.assert_close(ordinary, expected, atol=0, rtol=0)
        if not native.supports_linear_fusions(name, x.shape[0], 0):
            continue
        actual = native.linear(raw, name, (128, 512), x, bias, x.dtype, fused_output=True, silu_mul=silu)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        fused_cases += 1
    # Captured inputs must stay live and accept different values on replay.
    for silu in (False, True):
        raw = torch.from_numpy(release.packed_fixture('Q4_K')[0]).flatten().to('cuda')
        x = torch.randn(3, 1024 if silu else 512, dtype=torch.bfloat16, device='cuda')
        def fused():
            return native.linear(raw, 'Q4_K', (128, 512), x, None, x.dtype, fused_output=True, silu_mul=silu)
        fused()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fused()
        for _ in range(3):
            x.normal_()
            graph.replay()
            activated = SiluAndMul().forward_list([x]) if silu else x
            expected = native.linear(raw, 'Q4_K', (128, 512), activated, None, x.dtype)
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
        graph.reset()
    return dict(installed_baseline_cases=len(cases), exact_fused_cases=fused_cases, graph_cases=6)


def rope_case(dtype, dim, tokens, cache_type, replay=False):
    batch, heads, kv_heads, rotary = 2, 4, 2, min(dim // 2, 64)
    q = torch.randn(batch, heads, tokens, dim, dtype=dtype, device='cuda').transpose(1, 2)
    k = torch.randn(batch, kv_heads, tokens, dim, dtype=dtype, device='cuda').transpose(1, 2)
    v = torch.randn(batch, tokens, kv_heads, dim * 2, dtype=dtype, device='cuda')[..., :dim]
    angles = torch.randn(batch, tokens, rotary, dtype=dtype, device='cuda')
    cos, sin = angles.cos(), angles.sin()
    capacity = ((batch * tokens * 3 + 255) // 256) * 256
    slots = torch.arange(batch * tokens, device='cuda', dtype=torch.int64) * 3
    slots[::5] = -1
    def cache():
        shape = (capacity // 256, 256, kv_heads, dim)
        scale_shape = (*shape[:-1], dim // 32)
        return SimpleNamespace(k_cache=torch.zeros(shape, dtype=cache_type, device='cuda'),
            v_cache=torch.zeros(shape, dtype=cache_type, device='cuda'),
            k_scale=torch.zeros(scale_shape, dtype=torch.float16, device='cuda'),
            v_scale=torch.zeros(scale_shape, dtype=torch.float16, device='cuda'))
    reference, candidate = cache(), cache()
    qr, kr = q.clone(), k.clone()
    apply_rotary_pos_emb([qr, kr], cos, sin)
    store_kvcache(kr.reshape(-1, kv_heads, dim).contiguous(), v.reshape(-1, kv_heads, dim).contiguous(), reference.k_cache, reference.v_cache, slots, reference.k_scale, reference.v_scale)
    assert rope.supported(q, k, v, cos, sin)
    rope.apply_rope_cache(q, k, v, cos, sin, candidate, slots)
    torch.testing.assert_close(q, qr, atol=0, rtol=0)
    torch.testing.assert_close(k, kr, atol=0, rtol=0)
    for name in ('k_cache', 'v_cache', 'k_scale', 'v_scale'):
        torch.testing.assert_close(getattr(candidate, name), getattr(reference, name), atol=0, rtol=0)
    if replay:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rope.apply_rope_cache(q, k, v, cos, sin, candidate, slots)
        for _ in range(3):
            q.normal_()
            k.normal_()
            v.normal_()
            slots.copy_(slots.flip(0))
            qr, kr = q.clone(), k.clone()
            apply_rotary_pos_emb([qr, kr], cos, sin)
            store_kvcache(kr.reshape(-1, kv_heads, dim).contiguous(), v.reshape(-1, kv_heads, dim).contiguous(), reference.k_cache, reference.v_cache, slots, reference.k_scale, reference.v_scale)
            graph.replay()
            torch.testing.assert_close(q, qr, atol=0, rtol=0)
            torch.testing.assert_close(k, kr, atol=0, rtol=0)
            for name in ('k_cache', 'v_cache', 'k_scale', 'v_scale'):
                torch.testing.assert_close(getattr(candidate, name), getattr(reference, name), atol=0, rtol=0)
        graph.reset()


def check_rope():
    count = 0
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for dim in (32, 64, 128, 256):
            for tokens in (1, 3, 65):
                for cache_type in (dtype, torch.int8):
                    rope_case(dtype, dim, tokens, cache_type, replay=dim == 256 and tokens == 3)
                    count += 1
            print('ROPE', dtype, dim, flush=True)
    kernels = rope._rope_cache_kernel.device_caches[0][0].values()
    resources = [dict(registers=k.n_regs, spills=k.n_spills, shared=k.metadata.shared) for k in kernels]
    assert all(k['spills'] == 0 for k in resources), resources
    # Model warmup rotates without allocated KV storage.
    q = torch.randn(1, 3, 24, 256, dtype=torch.bfloat16, device='cuda')
    k, v = torch.randn(2, 1, 3, 4, 256, dtype=q.dtype, device=q.device).unbind()
    angle = torch.randn(1, 3, 64, dtype=q.dtype, device=q.device)
    cos, sin = angle.cos(), angle.sin()
    qr, kr = q.clone(), k.clone()
    apply_rotary_pos_emb([qr, kr], cos, sin)
    empty = torch.empty(0, device='cpu')
    cache = SimpleNamespace(k_cache=empty, v_cache=empty, k_scale=empty, v_scale=empty)
    assert not rope.apply_rope_cache(q, k, v, cos, sin, cache, None)
    torch.testing.assert_close(q, qr, atol=0, rtol=0)
    torch.testing.assert_close(k, kr, atol=0, rtol=0)
    return dict(exact_cases=count, graph_replays=18, warmup_without_cache=True, resources=resources)


def benchmark(checkpoint):
    reader = gguf.GGUFReader(str(checkpoint), mode='r')
    records = []
    for name in ('blk.0.ffn_down.weight', 'blk.0.ffn_gate.weight', 'blk.3.attn_q.weight'):
        weight = next(t for t in reader.tensors if t.name == name)
        raw = torch.from_numpy(np.array(weight.data, copy=True)).flatten().to('cuda')
        shape = tuple(map(int, reversed(weight.shape)))
        for rows in (1, 3, 5):
            x = torch.randn(rows, shape[1], dtype=torch.bfloat16, device='cuda')
            original = lambda: native.linear(raw, weight.tensor_type.name, shape, x, None, x.dtype)
            fused = lambda: native.linear(raw, weight.tensor_type.name, shape, x, None, x.dtype, fused_output=True)
            torch.testing.assert_close(fused(), original(), atol=0, rtol=0)
            old_time, new_time = timed(original), timed(fused)
            records.append(dict(weight=name, rows=rows, operation='typed_output', baseline_us=old_time, fused_us=new_time,
                speedup=old_time/new_time, baseline_peak_bytes=allocated_peak(original), fused_peak_bytes=allocated_peak(fused)))
            if 'ffn_down' in name:
                gate_up = torch.randn(rows, shape[1] * 2, dtype=x.dtype, device=x.device)
                old_ffn = lambda: native.linear(raw, weight.tensor_type.name, shape, SiluAndMul().forward_list([gate_up]), None, x.dtype)
                fused_ffn = lambda: native.linear(raw, weight.tensor_type.name, shape, gate_up, None, x.dtype, fused_output=True, silu_mul=True)
                torch.testing.assert_close(fused_ffn(), old_ffn(), atol=0, rtol=0)
                old_time, new_time = timed(old_ffn), timed(fused_ffn)
                records.append(dict(weight=name, rows=rows, operation='silu_q8_and_typed_output', baseline_us=old_time, fused_us=new_time,
                    speedup=old_time/new_time, baseline_peak_bytes=allocated_peak(old_ffn), fused_peak_bytes=allocated_peak(fused_ffn)))
            print('BENCH', records[-1], flush=True)
    for tokens in (1, 3, 512):
        for cache_type in (torch.bfloat16, torch.int8):
            q = torch.randn(1, tokens, 24, 256, dtype=torch.bfloat16, device='cuda')
            k, v = torch.randn(2, 1, tokens, 4, 256, dtype=q.dtype, device=q.device).unbind()
            angle = torch.randn(1, tokens, 64, dtype=q.dtype, device=q.device)
            cos, sin = angle.cos(), angle.sin()
            capacity = ((tokens + 255) // 256) * 256
            cache = SimpleNamespace(k_cache=torch.zeros(capacity//256, 256, 4, 256, dtype=cache_type, device='cuda'),
                v_cache=torch.zeros(capacity//256, 256, 4, 256, dtype=cache_type, device='cuda'),
                k_scale=torch.zeros(capacity//256, 256, 4, 8, dtype=torch.float16, device='cuda'),
                v_scale=torch.zeros(capacity//256, 256, 4, 8, dtype=torch.float16, device='cuda'))
            slots = torch.arange(tokens, device='cuda', dtype=torch.int64)
            def original_rope():
                apply_rotary_pos_emb([q, k], cos, sin)
                store_kvcache(k.reshape(-1, 4, 256), v.reshape(-1, 4, 256), cache.k_cache, cache.v_cache, slots, cache.k_scale, cache.v_scale)
            fused_rope = lambda: rope.apply_rope_cache(q, k, v, cos, sin, cache, slots)
            old_time, new_time = timed(original_rope), timed(fused_rope)
            records.append(dict(operation='rope_cache', rows=tokens, cache=str(cache_type), baseline_us=old_time, fused_us=new_time,
                speedup=old_time/new_time, baseline_peak_bytes=allocated_peak(original_rope), fused_peak_bytes=allocated_peak(fused_rope)))
            print('BENCH', records[-1], flush=True)
    return records


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('baseline', 'native', 'rope', 'bench'), required=True)
    parser.add_argument('--baseline', type=Path, default=Path('D:/AMD/cuda-fusions-20260922/native-baseline.pt'))
    parser.add_argument('--checkpoint', type=Path, default=Path('E:/ML/Wan2GP/ckpts/Qwen3_8_27B_Uncensored/Qwen3.8-27B-Uncensored-Q4_K_M.gguf'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'baseline':
        result = create_baseline(args.baseline)
    elif args.phase == 'native':
        result = compare_native(args.baseline)
    elif args.phase == 'rope':
        result = check_rope()
    else:
        result = benchmark(args.checkpoint)
    args.output.write_text(json.dumps(dict(phase=args.phase, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), package=native.__file__, result=result), indent=2), encoding='utf8')
    print('PASS', args.phase, flush=True)


if __name__ == '__main__':
    main()
