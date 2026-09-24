"""Pair real MTP generation with/without batching two confidence readbacks.

The experimental installer changes only benchmark runner methods. The target
verification width, precision, confidence threshold and sampling remain fixed.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmgp import offload
from experiments.qwen_mtp_device_confidence import install
from shared.prompt_enhancer.qwen35_assistant_runtime import Qwen35AssistantRuntime
from shared.prompt_enhancer.qwen35_text import load_qwen35_text_prompt_enhancer
from shared.qtypes.gguf import get_gguf_compute_dtype
from shared.utils import files_locator


def state_digest(model):
    digest = hashlib.sha256()
    for layer in model.blk:
        if layer.layer_type == "linear_attention":
            for name in ("conv_state_buffer", "recurrent_state_buffer"):
                value = getattr(layer, name).detach().to("cpu", copy=True).contiguous()
                digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--confidence", type=float, default=.3)
    parser.add_argument("--rounds", type=int, default=1, help="Four full generation calls per prompt/round")
    parser.add_argument("--capture", action="store_true", help="Capture both confidence checks and the MTP draft together")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(path.read_text(encoding="utf-8")) for path in args.prompt_ids]
    files_locator.set_checkpoints_paths([str(args.checkpoint.parent.parent), "ckpts", "."])
    dtype = get_gguf_compute_dtype()
    model = manager = None
    restore = None
    report = dict(checkpoint=str(args.checkpoint), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), dtype=str(dtype), argv=sys.argv,
                  confidence=args.confidence, tokens=args.tokens, rounds=args.rounds,
                  measurement="actual end-to-end greedy MTP decode; excludes prefill",
                  experimental=True, capture=args.capture, completed=False, records=[])
    report["source_sha256"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__), Path(__file__).parent / "experiments/qwen_mtp_device_confidence.py")}
    try:
        import llamacpp_gguf_cuda
        native = Path(llamacpp_gguf_cuda._C.__file__)
        report.update(native=str(native), native_sha256=hashlib.sha256(native.read_bytes()).hexdigest())
        model = load_qwen35_text_prompt_enhancer(model_path=str(args.checkpoint),
            assets_dir=str(args.checkpoint.parent), default_dtype=dtype, backend="gguf",
            requested_lm_engine="vllm", variant="27b", speculative_decoding=True, kv_cache_int8=True)
        model._prompt_enhancer_min_model_len_hint = max(4096, max(map(len, prompts)) + args.tokens + 256)
        model._prompt_enhancer_speculative_tokens = 2
        model._prompt_enhancer_speculative_sampling_tokens = 2
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
            result = runtime.generate_segment(max_new_tokens=budget, max_total_tokens=budget,
                seed=123, do_sample=False, temperature=.6, top_p=.95, top_k=1,
                thinking_enabled=True, stop_requested=stop)
            return result, list(seq.token_ids[start:])

        with torch.inference_mode():
            runtime.prime_context(prompts[0], seed=123)
            decode(32)
            runner = runtime._get_live_llm().model_runner
            runner.set_mtp_stage_profile_enabled(False)
            report["runtime_capacity"] = runner.config.max_model_len
            report["allocated_before_candidate_warmup"] = torch.cuda.memory_allocated()
            # Warm the experimental path too; exclude capture/first-use effects.
            restore, counters = install(runner, capture=args.capture)
            runtime.prime_context(prompts[0], seed=123)
            decode(32)
            report["warmup_candidate_counts"] = copy.deepcopy(counters)
            report["allocated_after_candidate_warmup"] = torch.cuda.memory_allocated()
            restore()
            restore = None
            references = {}
            for round_index in range(args.rounds):
                for prompt_index, ids in enumerate(prompts):
                    order = ("baseline", "candidate", "candidate", "baseline")
                    if (round_index + prompt_index) % 2:
                        order = ("candidate", "baseline", "baseline", "candidate")
                    for position, kind in enumerate(order):
                        counters = {}
                        if kind == "candidate":
                            restore, counters = install(runner, capture=args.capture)
                        runtime.prime_context(ids, seed=123)
                        if runtime._get_live_llm().model_runner is not runner:
                            raise RuntimeError("Runner changed during paired benchmark")
                        before_stats = copy.deepcopy(runner.speculative_stats)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                        result, tokens = decode(args.tokens)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - started
                        peak = torch.cuda.max_memory_allocated()
                        reserved = torch.cuda.max_memory_reserved()
                        final_digest = state_digest(model)
                        stats = copy.deepcopy(runner.speculative_stats)
                        for key, value in before_stats.items():
                            stats[key] = [a-b for a,b in zip(stats[key], value)] if isinstance(value, list) else stats[key]-value
                        seq = runtime._get_active_sequence()
                        alignment = runner.speculative_telemetry(seq.seq_id, seq.num_tokens)
                        if alignment["sync_delta"] != 0 or len(tokens) > args.tokens:
                            raise AssertionError((alignment, len(tokens)))
                        if restore is not None:
                            restore()
                            restore = None
                        reference = references.setdefault(prompt_index, (tokens, final_digest, stats))
                        exact_tokens, exact_state, exact_stats = tokens == reference[0], final_digest == reference[1], stats == reference[2]
                        record = dict(round=round_index, prompt=prompt_index, position=position,
                            variant=kind, order=order, prompt_tokens=len(ids),
                            prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                            seconds=elapsed, tokens=len(tokens), tokens_per_second=len(tokens)/elapsed,
                            token_ids=tokens, completion=result.raw_text, speculative_stats=stats,
                            state_sha256=final_digest, exact_tokens=exact_tokens, exact_state=exact_state,
                            exact_stats=exact_stats, alignment=alignment, counters=copy.deepcopy(counters),
                            peak_allocated_bytes=peak, peak_reserved_bytes=reserved)
                        report["records"].append(record)
                        (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                        print(json.dumps({k:v for k,v in record.items() if k not in ("token_ids", "completion", "alignment", "speculative_stats", "order")}), flush=True)
                        if not (exact_tokens and exact_state and exact_stats):
                            raise AssertionError("Candidate or repeated reference changed tokens, live state or acceptance statistics")
            summaries = []
            for round_index in range(args.rounds):
                for prompt_index in range(len(prompts)):
                    records = [r for r in report["records"] if r["round"] == round_index and r["prompt"] == prompt_index]
                    durations = {k: statistics.geometric_mean(r["seconds"] for r in records if r["variant"] == k)
                                 for k in ("baseline", "candidate")}
                    summaries.append(dict(round=round_index, prompt=prompt_index,
                        speedup=durations["baseline"] / durations["candidate"], seconds=durations))
            report["paired_summaries"] = summaries
            report["geometric_speedup"] = math.exp(statistics.mean(math.log(r["speedup"]) for r in summaries))
            report["completed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        if restore is not None:
            restore()
        if model is not None:
            model.unload()
        if manager is not None:
            manager.release()
        (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["paired_summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
