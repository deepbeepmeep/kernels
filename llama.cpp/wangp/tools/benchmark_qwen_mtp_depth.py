"""Sweep native MTP depths with one resident model and fixed maximum graph capacity.

Maximum-depth graphs and rollback snapshots coexist for every measurement. This
isolates depth scheduling for speed research, not production-memory acceptance.
"""
import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmgp import offload
from shared.prompt_enhancer.qwen35_assistant_runtime import Qwen35AssistantRuntime
from shared.prompt_enhancer.qwen35_text import load_qwen35_text_prompt_enhancer
from shared.qtypes.gguf import get_gguf_compute_dtype
from shared.utils import files_locator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt-ids", nargs="+", required=True, type=Path)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4, 6, 7, 2])
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--confidence", type=float, default=.3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if min(args.depths) < 1 or max(args.depths) > 7:
        parser.error("This experiment keeps draft depths within1..7 (verification stays within8rows).")
    args.output.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(path.read_text(encoding="utf-8")) for path in args.prompt_ids]
    if len({hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts}) != len(prompts):
        parser.error("Prompt fixtures must be distinct.")
    files_locator.set_checkpoints_paths([str(args.checkpoint.parent.parent), "ckpts", "."])
    dtype = get_gguf_compute_dtype()
    cap = max(args.depths)
    report = dict(checkpoint=str(args.checkpoint), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), graph_draft_capacity=cap,
                  precision=str(dtype), confidence=args.confidence, greedy=True,
                  requested_tokens=args.tokens, production_memory_comparison=False, records=[])
    model = manager = None
    try:
        import llamacpp_gguf_cuda
        binary = Path(llamacpp_gguf_cuda._C.__file__)
        report.update(native=str(binary), native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        model = load_qwen35_text_prompt_enhancer(model_path=str(args.checkpoint),
            assets_dir=str(args.checkpoint.parent), default_dtype=dtype, backend="gguf",
            requested_lm_engine="vllm", variant="27b", speculative_decoding=True, kv_cache_int8=True)
        model._prompt_enhancer_min_model_len_hint = max(4096, max(map(len, prompts)) + args.tokens + 256)
        model._prompt_enhancer_speculative_tokens = cap
        model._prompt_enhancer_speculative_sampling_tokens = cap
        model._prompt_enhancer_speculative_confidence = args.confidence
        manager = offload.profile({"llm": model}, profile_no=1, budgets={"llm": 0},
            pinnedMemory=False, quantizeTransformer=False, convertWeightsFloatTo=dtype, verboseLevel=1)
        runtime = Qwen35AssistantRuntime(model)

        def decode(budget):
            seq = runtime._get_active_sequence()
            start = seq.num_tokens
            def stop():
                remaining = budget - (seq.num_tokens - start)
                seq.speculative_max_emission = max(0, remaining)
                return remaining <= 0
            result = runtime.generate_segment(max_new_tokens=budget, seed=123, do_sample=False,
                temperature=.6, top_p=.95, top_k=1, thinking_enabled=True, stop_requested=stop)
            return result, list(seq.token_ids[start:])

        with torch.inference_mode():
            runtime.prime_context(prompts[0], seed=123)
            decode(64)
            torch.cuda.synchronize()
            original_runner = runtime._get_live_llm().model_runner
            assert original_runner._max_speculative_draft_tokens == cap
            report["runtime_capacity"] = original_runner.config.max_model_len
            for prompt_index, ids in enumerate(prompts):
                order = args.depths if prompt_index % 2 == 0 else list(reversed(args.depths))
                prompt_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                depth2 = None
                for run_index, depth in enumerate(order):
                    model._prompt_enhancer_speculative_tokens = depth
                    model._prompt_enhancer_speculative_sampling_tokens = depth
                    runtime.prime_context(ids, seed=123)
                    runner = runtime._get_live_llm().model_runner
                    if runner is not original_runner or runner._max_speculative_draft_tokens != cap:
                        raise RuntimeError("Runtime or graph capacity changed during the depth sweep.")
                    runner.set_mtp_stage_profile_enabled(False)
                    before_stats = copy.deepcopy(runner.speculative_stats)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    result, tokens = decode(args.tokens)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    stats = copy.deepcopy(runner.speculative_stats)
                    for key, value in before_stats.items():
                        stats[key] = [a-b for a,b in zip(stats[key], value)] if isinstance(value, list) else stats[key]-value
                    seq = runtime._get_active_sequence()
                    alignment = runner.speculative_telemetry(seq.seq_id, seq.num_tokens)
                    if alignment["sync_delta"] != 0:
                        raise AssertionError(alignment)
                    if depth == 2 and depth2 is None:
                        depth2 = tokens
                    prefix = 0
                    if depth2 is not None:
                        for left, right in zip(tokens, depth2):
                            if left != right:
                                break
                            prefix += 1
                    record = dict(prompt=prompt_index, run=run_index, depth=depth,
                        prompt_tokens=len(ids), prompt_sha256=prompt_hash, tokens=len(tokens),
                        overshoot=max(0, len(tokens)-args.tokens), seconds=elapsed, tokens_per_second=len(tokens)/elapsed,
                        reported_tokens=result.token_count, stop_reason=result.stop_reason,
                        token_ids=tokens, completion=result.raw_text, speculative_stats=stats,
                        alignment=alignment, exact_depth2_tokens=tokens == depth2,
                        common_depth2_prefix=prefix, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    report["records"].append(record)
                    name = f"prompt{prompt_index}_run{run_index}_depth{depth}"
                    (args.output / f"{name}.txt").write_text(result.raw_text, encoding="utf-8")
                    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    print(json.dumps({key: value for key, value in record.items() if key not in ("token_ids", "completion", "alignment")}), flush=True)
            report["summary"] = {str(depth): dict(
                median_tokens_per_second=statistics.median(r["tokens_per_second"] for r in report["records"] if r["depth"] == depth),
                exact_depth2=all(r["exact_depth2_tokens"] for r in report["records"] if r["depth"] == depth),
                tokens_per_target_pass=statistics.mean(r["tokens"] / max(1, r["speculative_stats"]["target_passes"]) for r in report["records"] if r["depth"] == depth))
                for depth in sorted(set(args.depths))}
    finally:
        if model is not None:
            model.unload()
        if manager is not None:
            manager.release()
        (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report.get("summary", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
