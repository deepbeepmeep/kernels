"""Compare parallel prediction, native MTP and ordinary decode on fixed prompts.

Run methods in separate processes so their graph allocations do not coexist.
Keep greedy and sampled results separate; sampled methods need not emit the same
continuation. Timings exclude loading, warmup, profiling and memory polling.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import psutil
from mmgp import offload
from shared.prompt_enhancer.qwen35_assistant_runtime import Qwen35AssistantRuntime
from shared.prompt_enhancer.qwen35_text import load_qwen35_text_prompt_enhancer
from shared.qtypes.gguf import get_gguf_compute_dtype
from shared.utils import files_locator


TASKS = {
    "math": "Find all pairs of positive integers (a, b) for which a squared plus b squared plus 1 equals 3ab. Prove your characterization completely, explaining why it includes every solution.",
    "code": "Implement a Python least-recently-used cache with a fixed capacity, O(1) get and put operations, and an optional expiration time for each key. Explain the invariants and include tests for replacement, expiration and capacity zero.",
}


def other_gpu_jobs():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    gpu_pids = {int(line.strip()) for line in output.splitlines() if line.strip().isdigit()}
    jobs = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        info = process.info
        if info["pid"] == os.getpid() or info["pid"] not in gpu_pids or not str(info["name"]).lower().startswith("python"):
            continue
        command = " ".join(info["cmdline"] or [])
        if "--server-port 7861" in command or "hf_hub_download" in command:
            continue
        jobs.append(dict(pid=info["pid"], command=command))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, help="Tokenizer/config folder, defaulting to the checkpoint folder")
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--fiction-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("disabled", "mtp", "dflash2", "dspark"), required=True)
    parser.add_argument("--drafts", type=int)
    parser.add_argument("--compare-drafts", type=int, nargs="+", help="Interleave draft-count limits in one runtime; graph capacity stays at the largest count")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--sampling", choices=("greedy", "sampled"), nargs="+", default=["greedy", "sampled"])
    parser.add_argument("--prompts", choices=("fiction", "math", "code"), nargs="+", default=["fiction", "math", "code"])
    parser.add_argument("--legacy-draft-rope", action="store_true", help="Reproduce the old loader ignoring the modern RoPE schema")
    parser.add_argument("--compare-rope", action="store_true", help="Paired ABBA/BAAB old/fixed RoPE in one resident runtime")
    parser.add_argument("--compare-dspark", action="store_true", help="Paired ABBA/BAAB reference/GPU DSpark prediction and acceptance")
    args = parser.parse_args()
    if jobs := other_gpu_jobs():
        raise RuntimeError(f"Other Python jobs have GPU contexts: {jobs}")
    if args.compare_rope and (args.method != "dflash2" or args.legacy_draft_rope):
        parser.error("--compare-rope requires DFlash2 and the corrected loader")
    if args.compare_dspark and (args.method != "dspark" or args.compare_rope or args.compare_drafts):
        parser.error("--compare-dspark requires DSpark and cannot combine with another paired experiment")
    if args.compare_drafts and (args.compare_rope or args.drafts is not None or args.method == "disabled"):
        parser.error("--compare-drafts requires a predictor and cannot combine with --compare-rope or --drafts")
    if args.compare_drafts and any(count < 1 or count > (8 if args.method == "mtp" else 7) for count in args.compare_drafts):
        parser.error("Draft counts exceed the predictor's supported range")
    if args.legacy_draft_rope:
        from transformers import Qwen3Config
        from shared.prompt_enhancer import block_draft
        block_draft._load_draft_config = lambda path: Qwen3Config.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    args.output.mkdir(parents=True, exist_ok=True)
    files_locator.set_checkpoints_paths([str(args.checkpoint.parent.parent), str(args.checkpoints_root), "ckpts", "."])
    dtype = get_gguf_compute_dtype()
    drafts = args.drafts if args.drafts is not None else dict(disabled=0, mtp=2, dflash2=7, dspark=7)[args.method]
    if args.compare_drafts:
        drafts = max(args.compare_drafts)
    selection = dict(disabled=False, mtp=True, dflash2="dflash2", dspark="dspark")[args.method]
    model = load_qwen35_text_prompt_enhancer(model_path=str(args.checkpoint), assets_dir=str(args.assets or args.checkpoint.parent),
        default_dtype=dtype, backend="gguf", requested_lm_engine="vllm", variant="27b", speculative_decoding=selection, kv_cache_int8=True)
    model._prompt_enhancer_min_model_len_hint = 32768
    model._prompt_enhancer_speculative_tokens = model._prompt_enhancer_speculative_sampling_tokens = drafts
    model._prompt_enhancer_speculative_confidence = .3
    tokenizer = model._prompt_enhancer_tokenizer
    prompts = {"fiction": json.loads(args.fiction_prompt.read_text(encoding="utf-8"))}
    for name, task in TASKS.items():
        prompts[name] = tokenizer.apply_chat_template([{"role": "user", "content": task}], tokenize=True,
            add_generation_prompt=True, enable_thinking=True)
    import llamacpp_gguf_cuda
    binary = Path(llamacpp_gguf_cuda._C.__file__)
    report = dict(argv=sys.argv, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), checkpoint=str(args.checkpoint),
        binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), drafts=drafts, method=args.method,
        compared_drafts=args.compare_drafts,
        sampling=dict(temperature=.6, top_k=20, top_p=.95, min_p=model._prompt_enhancer_default_min_p,
                      repetition_penalty=1.05, seed=123), records=[], completed=False)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / "shared/qtypes/gguf.py", root / "shared/prompt_enhancer/qwen35_text.py", root / "shared/prompt_enhancer/block_draft.py",
               root / "shared/kernels/qwen_gdn.py", root / "shared/llm_engines/nanovllm/engine/model_runner.py",
               root / "shared/llm_engines/nanovllm/engine/block_draft_runner.py", root / "shared/llm_engines/nanovllm/models/block_draft.py",
               root / "shared/llm_engines/nanovllm/engine/speculative_sampling.py", root / "shared/prompt_enhancer/config.py"]
    report["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    if args.method in ("dflash2", "dspark"):
        report["draft_rope_theta"] = model.mtp.config.rope_theta
        report["draft_rope_scaling"] = model.mtp.config.rope_scaling
    manager = offload.profile({"llm": model}, profile_no=1, budgets={"llm": 0}, pinnedMemory=False,
        quantizeTransformer=False, convertWeightsFloatTo=dtype, verboseLevel=1)
    runtime = Qwen35AssistantRuntime(model)
    graph_captures = 0
    capture_begin = torch.cuda.CUDAGraph.capture_begin
    def counted_capture(graph, *pos, **kw):
        nonlocal graph_captures
        graph_captures += 1
        return capture_begin(graph, *pos, **kw)
    torch.cuda.CUDAGraph.capture_begin = counted_capture

    def decode(count, greedy):
        seq = runtime._get_active_sequence()
        start = seq.num_tokens
        def stop():
            remaining = count - (seq.num_tokens - start)
            seq.speculative_max_emission = max(0, remaining)
            return remaining <= 0
        result = runtime.generate_segment(max_new_tokens=count, max_total_tokens=count, seed=123, do_sample=not greedy,
            temperature=.6, top_p=.95, top_k=1 if greedy else 20, thinking_enabled=True, stop_requested=stop)
        return result, list(seq.token_ids[start:])

    def select_variant(variant):
        if args.compare_dspark:
            runner._disable_dspark_gpu_draft = variant == "legacy"
        if args.compare_drafts:
            count = int(variant.removeprefix("drafts_"))
            model._prompt_enhancer_speculative_tokens = model._prompt_enhancer_speculative_sampling_tokens = count
        if not args.compare_rope:
            return None
        theta = 10000. if variant == "legacy" else model.mtp.config.rope_parameters["rope_theta"]
        dim = model.mtp.config.head_dim
        expected = 1. / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim))
        model.mtp.rotary_emb.inv_freq.copy_(expected)
        return expected

    try:
        with torch.inference_mode():
            for sampling in args.sampling:
                greedy = sampling == "greedy"
                runtime.prime_context(prompts["fiction"], seed=123)
                decode(64, greedy)
                runner = runtime._get_live_llm().model_runner
                runner.set_mtp_stage_profile_enabled(False)
                # Replay the complete measured trajectories, including rare
                # confidence lengths, page boundaries and final short blocks.
                variants = ("legacy", "fixed") if args.compare_rope or args.compare_dspark else tuple(f"drafts_{count}" for count in args.compare_drafts) if args.compare_drafts else ("fixed",)
                for variant in variants:
                    select_variant(variant)
                    for name in args.prompts:
                        runtime.prime_context(prompts[name], seed=123)
                        decode(args.tokens, greedy)
                for repeat in range(args.repeats):
                    cases = []
                    for index, name in enumerate(args.prompts if repeat % 2 == 0 else reversed(args.prompts)):
                        order = ("legacy", "fixed", "fixed", "legacy") if (repeat + index) % 2 == 0 else ("fixed", "legacy", "legacy", "fixed")
                        if args.compare_drafts:
                            offset = args.prompts.index(name) % len(variants)
                            order = variants[offset:] + variants[:offset]
                            if repeat % 2:
                                order = tuple(reversed(order))
                        cases.extend((name, variant, position) for position, variant in enumerate(order if args.compare_rope or args.compare_dspark or args.compare_drafts else ("fixed",)))
                    for name, variant, position in cases:
                        jobs_before = other_gpu_jobs()
                        if jobs_before:
                            raise RuntimeError(f"Other GPU jobs started before timing: {jobs_before}")
                        expected_frequency = select_variant(variant)
                        ids = prompts[name]
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        runtime.prime_context(ids, seed=123)
                        torch.cuda.synchronize()
                        prefill = time.perf_counter() - started
                        before = copy.deepcopy(runner.speculative_stats)
                        if expected_frequency is not None:
                            torch.testing.assert_close(model.mtp.rotary_emb.inv_freq.cpu(), expected_frequency, rtol=0, atol=0)
                        torch.cuda.reset_peak_memory_stats()
                        captures_before = graph_captures
                        started = time.perf_counter()
                        result, tokens = decode(args.tokens, greedy)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - started
                        jobs_after = other_gpu_jobs()
                        stats = copy.deepcopy(runner.speculative_stats)
                        for key, value in before.items():
                            stats[key] = [a-b for a,b in zip(stats[key], value)] if isinstance(value, list) else stats[key]-value
                        record = dict(prompt=name, prompt_tokens=len(ids), sampling=sampling, repeat=repeat, variant=variant, position=position, tokens=len(tokens),
                            prefill_seconds=prefill, decode_seconds=seconds, decode_tps=len(tokens)/seconds,
                            graph_captures_during_decode=graph_captures-captures_before,
                            end_to_end_tps=len(tokens)/(seconds+prefill), token_ids=tokens, completion=result.raw_text,
                            prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), speculative_stats=stats,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
                        record["other_gpu_jobs_after"] = jobs_after
                        record["timing_usable"] = not jobs_after
                        if drafts:
                            seq = runtime._get_active_sequence()
                            record["alignment"] = runner.speculative_telemetry(seq.seq_id, seq.num_tokens)
                            assert record["alignment"]["sync_delta"] == 0
                        report["records"].append(record)
                        (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                        print(json.dumps({k:v for k,v in record.items() if k not in ("token_ids", "completion", "alignment")}), flush=True)
                        if jobs_after:
                            raise RuntimeError(f"Another job started during timing; result invalid: {jobs_after}")
            report["completed"] = True
    finally:
        torch.cuda.CUDAGraph.capture_begin = capture_begin
        (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        model.unload()
        manager.release()


if __name__ == "__main__":
    main()
