"""Benchmark Deepy's actual Qwen runtime with uncached prompts and warm graphs."""

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from mmgp import offload
from shared.prompt_enhancer.qwen35_text import load_qwen35_text_prompt_enhancer
from shared.prompt_enhancer.qwen35_assistant_runtime import Qwen35AssistantRuntime
from shared.qtypes.gguf import get_gguf_compute_dtype
from shared.utils import files_locator


STORY = "Write a 10 chapters long story about a man who survived a murder attempt, but is taken for dead, and organizes from the shadows the murder of one of his murderers and manages to frame the murderer's accomplice for this crime. Make sure there is strong consistency and continuity throughout the story. Write the story directly; do not call tools."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--variant", choices=["4b", "9b", "27b"], default="27b")
    parser.add_argument("--checkpoint", default="Qwen3.8-27B-Uncensored-Q4_K_M.gguf")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--system-prompt", default="You are a fiction writer. The following public-domain prose is reference material.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 20000])
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--draft", type=int, choices=range(9), default=2)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--profile", choices=["decode", "prefill"])
    parser.add_argument("--kernel-library", type=Path)
    parser.add_argument("--reference-attention", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reference_attention:
        from shared.llm_engines.nanovllm.layers import attention
        spec = importlib.util.spec_from_file_location("shared.llm_engines.nanovllm.layers.attention_reference", args.reference_attention)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        attention.Attention.forward = reference.Attention.forward
    if args.kernel_library:
        import llamacpp_gguf_cuda
        spec = importlib.util.spec_from_file_location("candidate._C", args.kernel_library)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        llamacpp_gguf_cuda._C = candidate
    import llamacpp_gguf_cuda
    from shared.llm_engines.nanovllm.layers import attention
    properties = torch.cuda.get_device_properties(0)
    binary = args.kernel_library or Path(llamacpp_gguf_cuda._C.__file__)
    metadata = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=properties.name, compute_capability=[properties.major, properties.minor], multiprocessors=properties.multi_processor_count,
                    kernel_package_version=llamacpp_gguf_cuda.__version__, kernel_binary=str(binary), kernel_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    checkpoint=str(args.assets / args.checkpoint), variant=args.variant, context_capacity=32768, kv_cache="Q8 with FP16 scales per 32 values",
                    sampling=dict(temperature=.6, top_p=.95, top_k=20, min_p=.05, repetition_penalty=1.05, seed=123),
                    warmup_context=max(args.contexts), requested_draft_tokens=args.draft,
                    sm120_backend=getattr(attention._SM120_Q8, "__file__", None))
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    files_locator._checkpoints_paths = [str(args.assets.parent), "ckpts", "."]
    checkpoint = args.assets / args.checkpoint
    dtype = get_gguf_compute_dtype()
    started = time.perf_counter()
    model = load_qwen35_text_prompt_enhancer(model_path=str(checkpoint), assets_dir=str(args.assets), default_dtype=dtype, backend="gguf", requested_lm_engine="vllm", variant=args.variant, speculative_decoding=args.draft > 0, kv_cache_int8=True)
    metadata["actual_decoder_engine"] = model._prompt_enhancer_engine_name
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model._prompt_enhancer_min_model_len_hint = 32768
    model._prompt_enhancer_speculative_tokens = args.draft
    model._prompt_enhancer_speculative_sampling_tokens = args.draft
    model._prompt_enhancer_speculative_confidence = args.confidence
    manager = offload.profile({"llm": model}, profile_no=1, budgets={"llm": 0}, pinnedMemory=False, quantizeTransformer=False, convertWeightsFloatTo=dtype, verboseLevel=1)
    runtime = Qwen35AssistantRuntime(model)
    tokenizer = model._prompt_enhancer_tokenizer
    corpus = tokenizer.encode(args.corpus.read_text(encoding="utf-8")[:600000], add_special_tokens=False)
    prefix = tokenizer.encode("<|im_start|>system\n" + args.system_prompt + "\n", add_special_tokens=False)
    task = STORY if args.prompt_file is None else args.prompt_file.read_text(encoding="utf-8")
    suffix = tokenizer.encode("\n<|im_end|>\n<|im_start|>user\n" + task + "<|im_end|>\n<|im_start|>assistant\n<think>\n", add_special_tokens=False)

    def prompt(length, repeat):
        count = length - len(prefix) - len(suffix)
        assert count > 0
        offset = 10000 + repeat * 23000
        return prefix + corpus[offset:offset + count] + suffix

    def decode(tokens):
        start_tokens = runtime._get_active_sequence().num_tokens
        return runtime.generate_segment(max_new_tokens=tokens, seed=123, do_sample=not args.greedy, temperature=0.6, top_p=0.95, top_k=20, thinking_enabled=True, stop_requested=lambda: runtime._get_active_sequence().num_tokens - start_tokens >= tokens)

    records = []
    try:
        with torch.inference_mode():
            # Warm the prefix kernel and final partial prefill chunk as well as
            # the decode graphs, so first-use compilation is outside timings.
            runtime.prime_context(prompt(max(args.contexts), 0), seed=123)
            decode(64)
            torch.cuda.synchronize()
            print(f"WARMUP_SECONDS={time.perf_counter() - started:.3f}", flush=True)
            for context in args.contexts:
                for repeat in range(args.repeats):
                    ids = prompt(context, repeat)
                    (args.output / f"prompt_{context}_{repeat}.json").write_text(json.dumps(ids), encoding="utf-8")
                    (args.output / f"prompt_{context}_{repeat}.txt").write_text(tokenizer.decode(ids), encoding="utf-8")
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    runtime.prime_context(ids, seed=123)
                    torch.cuda.synchronize()
                    prefill = time.perf_counter() - start
                    runner = runtime._get_live_llm().model_runner
                    runner.set_mtp_stage_profile_enabled(True)
                    start = time.perf_counter()
                    result = decode(args.tokens)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    stages = runner.mtp_stage_profile_samples()
                    runner.set_mtp_stage_profile_enabled(False)
                    record = dict(context_tokens=len(ids), repeat=repeat, generated_tokens=result.token_count, stop_reason=result.stop_reason,
                                  prefill_seconds=prefill, prefill_tps=len(ids) / prefill, decode_seconds=elapsed, decode_tps=result.token_count / elapsed,
                                  draft=args.draft, confidence=args.confidence, greedy=args.greedy, speculative_stats=runner.speculative_stats,
                                  peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30, prompt_sha256=hashlib.sha256(bytes(json.dumps(ids), "utf-8")).hexdigest(), stages=stages)
                    if args.draft > 0:
                        sequence = runtime._get_active_sequence()
                        record["alignment"] = runner.speculative_telemetry(sequence.seq_id, sequence.num_tokens)
                        assert record["alignment"]["sync_delta"] == 0, record["alignment"]
                    records.append(record)
                    record.update(allocated_vram_gib=torch.cuda.memory_allocated() / 2**30, reserved_vram_gib=torch.cuda.memory_reserved() / 2**30, peak_reserved_vram_gib=torch.cuda.max_memory_reserved() / 2**30)
                    print("RESULT=" + json.dumps({k: v for k, v in record.items() if k != "stages"}), flush=True)
                    (args.output / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
                    (args.output / f"completion_{context}_{repeat}.txt").write_text(result.raw_text, encoding="utf-8")
                    if repeat == 0:
                        segments = torch.cuda.memory_snapshot()
                        pools = {}
                        for segment in segments:
                            pool = str(segment.get('segment_pool_id'))
                            entry = pools.setdefault(pool, dict(total_bytes=0, allocated_bytes=0, active_bytes=0, segments=0))
                            for key in ('total_bytes', 'allocated_bytes', 'active_bytes'):
                                entry[key] += segment[key.replace('_bytes', '_size')]
                            entry['segments'] += 1
                        (args.output / f'memory_{context}.json').write_text(json.dumps(pools, indent=2), encoding='utf-8')
            if args.profile:
                ids = prompt(args.contexts[-1], 0)
                runtime.prime_context(ids, seed=123)
                decode(32)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                    if args.profile == "prefill":
                        runtime.append_suffix(corpus[100000:101024])
                    else:
                        decode(64)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.output / f"{args.profile}_trace.json"))
                (args.output / f"{args.profile}_profile.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=70), encoding="utf-8")
    finally:
        model.unload()
        manager.release()


if __name__ == "__main__":
    main()
