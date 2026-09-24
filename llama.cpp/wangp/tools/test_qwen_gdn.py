"""Validate shared GDN decode on real Qwen 27B GGUF checkpoints."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mmgp import offload
from shared.kernels.qwen_gdn import GDNShortConvolution
from shared.prompt_enhancer.qwen35_text import load_qwen35_text_prompt_enhancer
from shared.utils import files_locator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draft', type=int, choices=(0, 2), default=0)
    parser.add_argument('--engine', choices=('legacy', 'cg', 'vllm'), default='vllm')
    args = parser.parse_args()
    if args.engine != 'vllm':
        import triton
        from shared.qtypes import gguf
        def forbidden_kernel(*args, **kwargs):
            raise AssertionError('Legacy/cg entered a Triton or new GGUF fusion kernel')
        triton.runtime.JITFunction.run = forbidden_kernel
        gguf.linear_fused = forbidden_kernel
    files_locator._checkpoints_paths = [str(args.checkpoint.parent.parent), 'ckpts', '.']
    torch.set_default_device('cpu')
    model = load_qwen35_text_prompt_enhancer(
        model_path=str(args.checkpoint), assets_dir=str(args.checkpoint.parent),
        default_dtype=torch.bfloat16, backend='gguf', requested_lm_engine=args.engine,
        variant='27b', speculative_decoding=args.draft > 0, kv_cache_int8=True)
    model._prompt_enhancer_speculative_tokens = args.draft
    model._prompt_enhancer_speculative_sampling_tokens = args.draft
    blocks = [b for b in model.blk if b.layer_type == 'linear_attention']
    capability = torch.cuda.get_device_capability(0)
    enabled = args.engine == 'vllm' and torch.version.hip is None and capability[0] >= 8
    tuned_launches = enabled and capability == (12, 0)
    assert len(blocks) == 48
    assert all((b._gdn_prepare_decode is not None) == enabled for b in blocks)
    assert all((b._gdn_recurrent_raw is not None) == enabled for b in blocks)
    assert all(b.attn_norm._small_batch_num_warps == (16 if tuned_launches else 4) for b in blocks)
    assert all(isinstance(b.ssm_conv1d, GDNShortConvolution) == tuned_launches for b in blocks)
    if args.engine != 'vllm':
        for block in model.blk:
            assert not block.mlp_act_fn.use_triton
            if block.layer_type == 'full_attention':
                assert not block.attn.use_triton_kv_cache
                assert block.attn.flash_attn_varlen_func is None
                assert block.attn.flash_attn_with_kvcache is None
    from shared.qtypes.gguf import GGUFWeightTensor
    projections = [m for m in model.modules() if isinstance(getattr(m, 'weight', None), GGUFWeightTensor)]
    assert all(bool(getattr(m, '_use_optimized_kernels', False)) == (args.engine == 'vllm' and torch.version.hip is None)
        for m in projections if hasattr(m, '_use_optimized_kernels'))
    manager = offload.profile({'llm': model}, profile_no=1, budgets={'llm': 0},
        pinnedMemory=False, quantizeTransformer=False, convertWeightsFloatTo=torch.bfloat16, verboseLevel=1)
    records = []
    try:
        with torch.inference_mode():
            for device in ('cpu', 'cuda'):
                torch.set_default_device(device)
                calls = 0
                def stop():
                    nonlocal calls
                    calls += 1
                    return calls >= 4
                try:
                    model.generate_messages([[{'role': 'user', 'content': 'Count upwards from one to one hundred.'}]],
                        max_new_tokens=128, do_sample=False, seed=42, thinking_enabled=False, stop_requested=stop)
                    raise AssertionError('Cancellation was not propagated')
                except InterruptedError:
                    assert calls == 4
                results = []
                for prompt in ('What is the capital of France? Answer with just the city name.',
                               'Write one sentence explaining why leaves look green.'):
                    text = model.generate_messages([[{'role': 'user', 'content': prompt}]],
                        max_new_tokens=70, do_sample=False, seed=42, thinking_enabled=False)[0]
                    print('RESULT', device, text, flush=True)
                    results.append(text)
                assert results[0].strip().rstrip('.') == 'Paris', results
                assert 'chlorophyll' in results[1].lower(), results
                records.append(dict(default_device=device, cancelled=True, results=results))
    finally:
        torch.set_default_device('cpu')
        model.unload()
        manager.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), engine=args.engine,
        draft=args.draft, fused_gdn=enabled, records=records), indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
