"""Real shared-loader block-draft checks for Qwen Q2/Q3/Q4 and Bonsai."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mmgp import offload
from shared.prompt_enhancer.loader import load_prompt_enhancer_runtime
from shared.prompt_enhancer.qwen35_assistant_runtime import Qwen35AssistantRuntime
from shared.utils import files_locator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dflash2", "dspark", "mtp"), default="dflash2")
    parser.add_argument('--backend', choices=('gguf', 'gguf_q3', 'gguf_q2', 'gguf_ptq1'), required=True)
    parser.add_argument('--config', type=Path, default=Path('wgp_config.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vision', action='store_true', help='Also exercise a real Deepy image inspection and restoration.')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    files_locator.set_checkpoints_paths(config['checkpoints_paths'])
    torch.set_default_device('cpu')
    if args.vision:
        import requests
        from PIL import Image
        from io import BytesIO
        response = requests.get('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png', timeout=60)
        response.raise_for_status()
        fixture = Image.open(BytesIO(response.content)).convert('RGB')

    def require_assets(repoId, sourceFolderList, fileList, targetFolderList=None):
        for folder, names in zip(targetFolderList or sourceFolderList, fileList):
            for name in names:
                files_locator.locate_file(f'{folder}/{name}' if folder else name)

    bonsai = args.backend == 'gguf_ptq1'
    tokens = 2 if args.method == "mtp" else (5 if bonsai and args.method == "dflash2" else 7)
    selection = {'method': args.method, 'tokens': tokens}
    loaded = load_prompt_enhancer_runtime(require_assets, 5, lm_decoder_engine='vllm', qwen_backend=args.backend,
                                         speculative_decoding=selection, deepy_kv_cache_quantization='int8')
    model = loaded.llm_model
    if args.vision:
        model._prompt_enhancer_min_model_len_hint = 32768
    if args.method in ('dflash2', 'dspark'):
        from shared.prompt_enhancer.block_draft import block_draft_spec
        expected_folder = block_draft_spec(args.method, bonsai=bonsai)['folder']
        assert model._block_draft_asset_folder == expected_folder
    else:
        expected_folder = 'native MTP'
        assert not getattr(model, '_block_draft', False)
    native = {key: value.dtype for key, value in model.mtp.named_parameters()}
    manager = offload.profile(loaded.pipe_models, profile_no=1, budgets=loaded.budgets, coTenantsMap=loaded.co_tenants,
                             pinnedMemory=False, quantizeTransformer=False, verboseLevel=1)
    assert native == {key: value.dtype for key, value in model.mtp.named_parameters()}
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
                    model.generate_messages([[{'role':'user', 'content':'Count from one to one hundred.'}]],
                                            max_new_tokens=128, do_sample=False, seed=42, thinking_enabled=False, stop_requested=stop)
                    raise AssertionError('Cancellation was not propagated')
                except InterruptedError:
                    assert calls == 4
                text = model.generate_messages([[{'role':'user', 'content':'What is the capital of France? Answer with just the city name.'}]],
                                               max_new_tokens=24, do_sample=False, seed=42, thinking_enabled=False)[0]
                assert text.strip().rstrip('.') == 'Paris', text
                runner = model._prompt_enhancer_vllm_engine._llm.model_runner
                assert runner._max_speculative_draft_tokens == tokens
                sampled = model.generate_messages([[{'role':'user', 'content':'Explain why leaves look green in one sentence.'}]],
                                                   max_new_tokens=70, do_sample=True, temperature=0.7, top_p=0.9, top_k=20, seed=42, thinking_enabled=False)[0]
                assert 'chlorophyll' in sampled.lower(), sampled
                runner = model._prompt_enhancer_vllm_engine._llm.model_runner
                counter_prefix = "_dflash" if args.method == "dflash2" else "_" + args.method
                rounds = getattr(runner, counter_prefix + "_gpu_acceptance_rounds", 0)
                assert rounds > 0
                records.append(dict(default_device=device, cancelled=True, greedy=text, sampled=sampled,
                                    gpu_acceptance_rounds=rounds,
                                    exact_sampler_fallbacks=getattr(runner, counter_prefix + "_acceptance_fallbacks", 0)))
            runtime = Qwen35AssistantRuntime(model)
            ids = loaded.llm_tokenizer.apply_chat_template([{'role':'user','content':'Explain photosynthesis in detail.'}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            runtime.prime_context(ids, seed=42)
            def continuation():
                return runtime.generate_segment(max_new_tokens=24, seed=42, do_sample=False, thinking_enabled=False, temperature=None, top_p=None, top_k=None).raw_text
            continuation()
            snapshot = runtime.snapshot_context()
            rewind = runtime.snapshot_rewind_state(snapshot)
            expected = continuation()
            runtime.restore_rewind_state(rewind)
            assert continuation() == expected
            runtime.restore_snapshot(snapshot)
            assert continuation() == expected
            records.append(dict(snapshot_and_rewind=True, continuation=expected))
            # Compare identical greedy continuations through both acceptance
            # implementations, including a thinking-budget boundary.
            for thinking in (False, True):
                prompt = loaded.llm_tokenizer.apply_chat_template([{'role':'user','content':'Explain why the sky looks blue.'}], tokenize=True, add_generation_prompt=True, enable_thinking=thinking)
                def compare_continuation(disabled):
                    runtime.prime_context(prompt, seed=47)
                    runner = model._prompt_enhancer_vllm_engine._llm.model_runner
                    setattr(runner, "_disable" + counter_prefix + "_gpu_acceptance", disabled)
                    if args.method == "dspark":
                        runner._disable_dspark_gpu_draft = disabled
                    result = runtime.generate_segment(max_new_tokens=48, seed=47, do_sample=False, thinking_enabled=thinking,
                        max_thinking_tokens=5 if thinking else None, temperature=None, top_p=None, top_k=None)
                    return result.raw_text
                legacy = compare_continuation(True)
                fast = compare_continuation(False)
                assert fast == legacy, (legacy, fast)
                records.append(dict(gpu_acceptance_greedy_parity=True, thinking=thinking, continuation=fast))
            runner = model._prompt_enhancer_vllm_engine._llm.model_runner
            vision = loaded.pipe_models['prompt_enhancer_image_caption_vision_tower_model']
            assert all(p.device.type != 'cuda' for p in vision.parameters())
            if args.vision:
                from shared.deepy import vision as vision_api
                assert vision_api.can_keep_text_resident(runtime, manager)
                before_inspection = runtime.snapshot_context()
                expected_after = continuation()
                runtime.restore_snapshot(before_inspection)
                with vision_api.resident_inspection(runtime, loaded.image_caption_model, manager) as unload_vision:
                    prompt, embeds, positions, offset = vision_api.build_image_question_prompt(loaded.image_caption_model, loaded.image_caption_processor, fixture, 'Describe the animals in this image.', resident=True)
                    embeds, positions = embeds.detach().to('cpu'), positions.detach().to('cpu')
                    unload_vision()
                    assert all(p.device.type != 'cuda' for p in vision.parameters())
                    answer = runtime.generate_embedded_answer(prompt, embeds, positions, offset, max_new_tokens=96, seed=42, do_sample=False, temperature=None, top_p=None, top_k=None, min_model_len=vision_api.VISION_MIN_MODEL_LEN)
                    assert 'cat' in answer.lower(), answer
                assert continuation() == expected_after
                assert all(p.device.type != 'cuda' for p in vision.parameters())
                records.append(dict(vision_answer=answer, restored_after_vision=True))
    finally:
        torch.set_default_device('cpu')
        model.unload()
        manager.release()
    result = dict(backend=args.backend, selection=selection, draft_folder=expected_folder, records=records,
                  peak_vram_bytes=torch.cuda.max_memory_allocated(), vision_on_cpu_outside_inspection=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('PASSED', json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
