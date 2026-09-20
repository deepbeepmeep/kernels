"""Real Bonsai checkpoint validation: dtypes, cancellation and repeated generation."""
import argparse
import json
from pathlib import Path
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mmgp import offload
from shared.prompt_enhancer.qwen35_text import load_qwen35_text_prompt_enhancer
from shared.qtypes.gguf import GGUFWeightTensor
from shared.utils import files_locator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=('legacy', 'cg', 'vllm'), default='vllm')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--default-device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--draft', type=int, choices=range(5), default=0)
    args = parser.parse_args()
    assets = args.assets or args.checkpoint.parent
    files_locator._checkpoints_paths = [str(assets.parent), 'ckpts', '.']
    torch.set_default_device(args.default_device)
    model = load_qwen35_text_prompt_enhancer(model_path=str(args.checkpoint), assets_dir=str(assets), default_dtype=torch.bfloat16, backend='gguf', requested_lm_engine=args.engine, variant='27b', speculative_decoding=args.draft > 0, kv_cache_int8=True)
    if args.draft:
        model._prompt_enhancer_speculative_tokens = args.draft
        model._prompt_enhancer_speculative_sampling_tokens = args.draft
    original = {n: str(p.dtype) for n,p in model.named_parameters() if not isinstance(p, GGUFWeightTensor)}
    assert sum(v == 'torch.float32' for v in original.values()) == 353 + (7 if args.draft else 0), {v:list(original.values()).count(v) for v in set(original.values())}
    assert sum(v == 'torch.bfloat16' for v in original.values()) == 48  # Paired alpha/beta matrices preserve BF16.
    manager = offload.profile({'llm':model}, profile_no=1, budgets={'llm':0}, pinnedMemory=False, quantizeTransformer=False, convertWeightsFloatTo=torch.float16, verboseLevel=1)
    assert original == {n: str(p.dtype) for n,p in model.named_parameters() if not isinstance(p, GGUFWeightTensor)}
    records=[]
    try:
        with torch.inference_mode():
            calls = 0
            def stop_after_steps():
                nonlocal calls
                calls += 1
                return calls >= 4
            try:
                model.generate_messages([[{'role':'user','content':'Count upwards from one to one hundred.'}]], max_new_tokens=128, do_sample=False, seed=42, thinking_enabled=False, stop_requested=stop_after_steps)
            except InterruptedError:
                cancelled = True
            else:
                raise AssertionError('Expected cancellation')
            assert calls == 4, calls
            records.append(dict(cancelled=cancelled, stop_checks=calls))
            for prompt in ('What is the capital of France? Answer with just the city name.', 'Write one sentence explaining why leaves look green.'):
                started=time.perf_counter()
                result=model.generate_messages([[{'role':'user','content':prompt}]], max_new_tokens=70, do_sample=False, seed=42, thinking_enabled=False)
                print('RESULT', result, flush=True)
                records.append(dict(prompt=prompt, result=result, seconds=time.perf_counter()-started))
            assert records[1]['result'][0].strip().rstrip('.') == 'Paris', records
            assert 'chlorophyll' in records[2]['result'][0].lower(), records
    finally:
        model.unload()
        manager.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(engine=args.engine, draft=args.draft, default_device=args.default_device, checkpoint=str(args.checkpoint), gpu=torch.cuda.get_device_name(), peak_vram_bytes=torch.cuda.max_memory_allocated(), records=records), indent=2, default=str), encoding='utf8')


if __name__ == '__main__':
    main()
