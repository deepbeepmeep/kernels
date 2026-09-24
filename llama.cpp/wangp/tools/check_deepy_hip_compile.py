"""Compile shared Deepy Triton kernels for AMD without claiming GPU execution."""
import argparse
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from shared.llm_engines.nanovllm.layers import activation, attention, layernorm
from shared.kernels import convrot_int8_triton, quanto_int8_triton, qwen_gdn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', default='gfx1201')
    parser.add_argument('--warp-size', type=int, choices=(32, 64), default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = GPUTarget('hip', args.arch, args.warp_size)
    results = []

    def compile_kernel(fn, pointers, constants, floats=(), options=None):
        signature = {name: pointers[name] if name in pointers else 'constexpr' if name in constants else 'fp32' if name in floats else 'i32' for name in fn.arg_names}
        kernel = triton.compile(ASTSource(fn=fn, signature=signature, constexprs=constants), target=target, options=options)
        assert kernel.asm['hsaco']
        assert '__nv_' not in kernel.asm['llir']
        results.append(dict(kernel=fn.__name__, signature=signature, constants=constants, code_bytes=len(kernel.asm['hsaco'])))

    for dtype in ('fp16', 'bf16', 'fp32'):
        compile_kernel(activation._silu_mul_kernel, {'x': '*'+dtype, 'out': '*'+dtype}, {'COLUMNS': 17408, 'BLOCK': 256})
        for residual in (False, True):
            pointers = {name: '*'+dtype for name in layernorm._rmsnorm_kernel.arg_names if name.endswith('_ptr')}
            compile_kernel(layernorm._rmsnorm_kernel, pointers, {'HAS_RESIDUAL': residual, 'STORE_RESIDUAL': residual, 'BLOCK_SIZE': 8192}, ('eps',))
    for dtype in ('fp16', 'bf16'):
        compile_kernel(attention.store_kvcache_kernel, {'key_ptr': '*'+dtype, 'value_ptr': '*'+dtype, 'k_cache_ptr': '*'+dtype, 'v_cache_ptr': '*'+dtype, 'slot_mapping_ptr': '*i32'}, {'D': 1024})
        compile_kernel(attention.store_int8_kvcache_kernel, {'key_ptr': '*'+dtype, 'value_ptr': '*'+dtype, 'k_cache_ptr': '*i8', 'v_cache_ptr': '*i8', 'k_scale_ptr': '*fp16', 'v_scale_ptr': '*fp16', 'slot_mapping_ptr': '*i32'}, {'H': 4, 'D': 256, 'QB': 8})
        constants = dict(q_stride_t=6144, q_stride_h=256, cache_stride_block=262144, cache_stride_token=1024, cache_stride_head=256, scale_stride_block=8192, scale_stride_token=32, scale_stride_head=8, out_stride_t=6144, out_stride_h=256, softmax_scale=.0625, H_Q=24, H_KV=4, D=256, PAGE=256, BLOCK_M=16, BLOCK_N=32)
        pointers = dict(q_ptr='*'+dtype, k_ptr='*i8', v_ptr='*i8', ks_ptr='*fp16', vs_ptr='*fp16', block_tables_ptr='*i32', cu_q_ptr='*i32', cu_k_ptr='*i32', out_ptr='*'+dtype)
        compile_kernel(attention.q8_paged_prefill_kernel, pointers, constants)
        pointers = dict(q_ptr='*'+dtype, k_ptr='*i8', v_ptr='*i8', ks_ptr='*fp16', vs_ptr='*fp16', tables_ptr='*i32', lengths_ptr='*i32', partial_ptr='*fp32', maximum_ptr='*fp32', sum_ptr='*fp32')
        compile_kernel(attention.q8_grouped_partials_kernel, pointers, dict(H_Q=24, H_KV=4, D=256, PAGE=256, SPLITS=4, M=16, N=32, SCALE=.0625))
        for fn in (quanto_int8_triton._fused_dynamic_int8_gemm_kernel, quanto_int8_triton._fused_dynamic_int8_blockscale_gemm_kernel):
            compile_kernel(fn, dict(a_ptr='*'+dtype, b_ptr='*i8', s_ptr='*fp32', c_ptr='*'+dtype), dict(block_m=16, block_n=32, block_k=64))
        for bias in (False, True):
            compile_kernel(convrot_int8_triton._convrot_int8_kernel, dict(x='*'+dtype, w='*i8', scale='*fp32', out='*'+dtype, bias='*'+dtype), dict(SX0=1024, SX1=1, SW0=1024, SW1=1, BM=16, BN=32, HAS_BIAS=bias))
        pointers = {name: '*'+dtype for name in ('Q', 'K', 'A', 'B', 'SSM_DT', 'Q_OUT', 'K_OUT', 'BETA')}
        pointers.update(SSM_A='*fp32', G='*fp32')
        compile_kernel(qwen_gdn._prepare_kernel, pointers, dict(Q_BATCH=2048, Q_HEAD=128, A_BATCH=48, B_BATCH=48, HEADS=48, REPEAT=3, DIM=128, BLOCK=128))
        pointers = {name: '*'+dtype for name in ('Q', 'K', 'V', 'A', 'BETA_IN', 'DT', 'OUT')}
        pointers.update(SSM_A='*fp32', STATE='*fp32', SNAPSHOTS='*fp32')
        compile_kernel(qwen_gdn._recurrent_raw_kernel, pointers, dict(T=1, H=16, HV=48, DK=128, DV=128, QS0=2048, QS1=2048, QS2=128, KS0=2048, KS1=2048, KS2=128, VS0=6144, VS1=6144, VS2=128, AS0=48, AS1=48, BS0=48, BS1=48, BATCH=1, BK=128, BV=32, SAVE_PREFIX=False), options=dict(num_warps=1, num_stages=3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(arch=args.arch, triton=triton.__version__, executed_on_gpu=False, cases=results), indent=2), encoding='utf-8')
    print(f'Compiled {len(results)} kernels for {args.arch}; no GPU execution performed.')


if __name__ == '__main__':
    main()
