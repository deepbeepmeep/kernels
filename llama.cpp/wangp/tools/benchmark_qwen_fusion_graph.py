"""Pair full Qwen decode/target-verification graphs on resident weights and caches.

This is fixed-input GPU work, not generated tokens/s. Each measured graph contains
the full model and output projection. State restoration is outside timed regions.
Separate graph pools are retained for A/B; this is not a production VRAM audit.
Three-row verification includes all target logits and rollback prefix snapshots,
but does not time drafting, acceptance, rollback commits, or MTP cache refresh.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmgp import offload
from shared.kernels import qwen_gdn, qwen_rope_cache
from shared.llm_engines.nanovllm.utils.context import get_context, reset_context
from shared.prompt_enhancer.qwen35_assistant_runtime import Qwen35AssistantRuntime
from shared.prompt_enhancer.qwen35_text import load_qwen35_text_prompt_enhancer
from shared.qtypes.gguf import get_gguf_compute_dtype
from shared.utils import files_locator


def gpu_status():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw",
        "--format=csv"], text=True).strip()


def run_proof(args, graphs, mutable, initial, report, configure, target_logits):
    """ABBA timing with GPU-side restoration and two event pairs per variant.

    Each wrapper restores state, then times exactly N full target passes. A
    single host launch submits this whole unit; CPU submission stalls cannot
    occur between its timed start and end. The two wrapper slots have distinct
    events so both appearances of A/B in a quartet retain their own timestamps.
    """
    initial_gpu = [saved.to(tensor.device) for saved, tensor in zip(initial, mutable)]
    wrappers = {}
    stream = torch.cuda.Stream(device=mutable[0].device)
    stream.wait_stream(torch.cuda.current_stream(mutable[0].device))
    for kind in graphs:
        configure(kind)
        for slot in range(2):
            start = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            wrapper = torch.cuda.CUDAGraph()
            with torch.cuda.graph(wrapper, stream=stream):
                for tensor, saved in zip(mutable, initial_gpu):
                    tensor.copy_(saved)
                start.record()
                for _ in range(args.replays):
                    logits = target_logits()
                end.record()
            wrappers[kind, slot] = (wrapper, start, end, logits)
    torch.cuda.synchronize()
    # Check the exact wrapper being measured, including the actual replay count.
    for tensor, saved in zip(mutable, initial_gpu):
        tensor.copy_(saved)
    for _ in range(args.replays):
        graphs["installed"][0].replay()
    reference = [graphs["installed"][1].detach().to("cpu", copy=True)]
    reference.extend(tensor.detach().to("cpu", copy=True) for tensor in mutable)
    report["wrapper_reference"] = "N direct installed graph replays from the same restored state"
    report["wrapper_parity"] = []
    for (kind, slot), (wrapper, start, end, logits) in wrappers.items():
        wrapper.replay()
        end.synchronize()
        actual = [logits.detach().to("cpu", copy=True)]
        actual.extend(tensor.detach().to("cpu", copy=True) for tensor in mutable)
        for index, (value, expected) in enumerate(zip(actual, reference)):
            torch.testing.assert_close(value, expected, rtol=0, atol=0,
                                       msg=f"{kind}/{slot} timed-wrapper logits/state index {index}")
        report["wrapper_parity"].append(dict(variant=kind, slot=slot, exact=True))
    del reference, actual
    report["parity_replays"] = sorted(set(report["parity_replays"] + [args.replays]))
    rng = random.Random(args.order_seed)
    comparisons = {"main": ("installed", "candidate"),
                   "duplicate_control": ("installed", "installed_copy"),
                   "build_control": ("installed", "baseline")}
    schedules = {}
    for comparison in comparisons:
        reverse = [False, True] * (args.rounds // 2)
        rng.shuffle(reverse)
        schedules[comparison] = reverse
    report.update(protocol="GPU-state-restore then external-event timed full target passes in one graph; balanced ABBA/BAAB",
                  variants=list(graphs), order_seed=args.order_seed,
                  test_only_gpu_snapshot_bytes=sum(x.numel()*x.element_size() for x in initial_gpu),
                  timing_scope="full target model and output projection GPU time; fixed inputs, excludes state restore",
                  no_rounds_trimmed=True)
    # Equal steady-state warmup before measurements (roughly two seconds).
    for _ in range(8):
        for wrapper, _, _, _ in wrappers.values():
            wrapper.replay()
    torch.cuda.synchronize()
    for block_id in range(args.rounds):
        comparison_order = list(comparisons)
        rng.shuffle(comparison_order)
        for comparison in comparison_order:
            a, b = comparisons[comparison]
            order = [b, a, a, b] if schedules[comparison][block_id] else [a, b, b, a]
            used = {a: 0, b: 0}
            launches = []
            enqueue_start = time.perf_counter_ns()
            for kind in order:
                slot = used[kind]
                used[kind] += 1
                wrapper, start, end, _ = wrappers[kind, slot]
                wrapper.replay()
                launches.append((kind, slot, start, end))
            enqueue_ns = time.perf_counter_ns() - enqueue_start
            launches[-1][3].synchronize()
            samples = []
            for position, (kind, slot, start, end) in enumerate(launches):
                total_ms = start.elapsed_time(end)
                samples.append(dict(variant=kind, slot=slot, position=position,
                                    gpu_ms_total=total_ms, gpu_ms=total_ms / args.replays))
            report["records"].append(dict(block_id=block_id, comparison=comparison,
                                           order=order, samples=samples, enqueue_ns=enqueue_ns))
        # Save all completed blocks even if a later block fails.
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        main = next(record for record in reversed(report["records"]) if record["comparison"] == "main")
        medians = {kind: statistics.geometric_mean(s["gpu_ms"] for s in main["samples"] if s["variant"] == kind)
                   for kind in comparisons["main"]}
        print(json.dumps(dict(block=block_id, main_gpu_ms=medians,
                              main_speedup=medians["installed"] / medians["candidate"])), flush=True)
    report["gpu_after_measurement"] = gpu_status()
    wrappers.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--verify-tokens", type=int, choices=(1, 3), default=1,
                        help="Use the native MTP three-row target verification context")
    parser.add_argument("--installed-native", type=Path, help="Capture a third reference with the installed extension")
    parser.add_argument("--proof", action="store_true", help="Balanced ABBA/BAAB with exact timed-wrapper parity and A/A controls")
    parser.add_argument("--run-id", default="unnamed")
    parser.add_argument("--order-seed", type=int, default=20260922)
    parser.add_argument("--runtime-manifest", type=Path)
    args = parser.parse_args()
    if args.proof and (args.installed_native is None or args.rounds < 2 or args.rounds % 2 or args.ablations):
        parser.error("--proof requires --installed-native, an even positive number of rounds, and no --ablations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files_locator.set_checkpoints_paths([str(args.checkpoint.parent.parent), "ckpts", "."])
    dtype = get_gguf_compute_dtype()
    model = manager = None
    graphs = {}
    candidate_native = None
    rope_supported = qwen_rope_cache.supported
    recurrent_raw_gates = qwen_gdn.recurrent_raw_gates
    report = dict(checkpoint=str(args.checkpoint), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), context=args.context,
                  rounds=args.rounds, replays=args.replays, verify_tokens=args.verify_tokens,
                  measurement="target_verification" if args.verify_tokens > 1 else "target_decode",
                  full_mtp_cycle=False, records=[], run_id=args.run_id, argv=sys.argv,
                  gpu_before_load=gpu_status(), completed=False)
    try:
        frozen_recurrent = recurrent_raw_gates
        if args.proof:
            from experiments.qwen_frozen_gdn_reference import load_frozen_reference
            frozen_recurrent, report["frozen_gdn"] = load_frozen_reference(Path(__file__).resolve().parents[1], args.output.parent / "frozen")
            if args.runtime_manifest:
                report["runtime_manifest"] = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
            source_paths = [Path(__file__), Path(qwen_gdn.__file__), Path(qwen_rope_cache.__file__),
                            Path("shared/llm_engines/nanovllm/models/qwen3_5.py"), Path("shared/qtypes/gguf.py"),
                            Path("shared/llm_engines/nanovllm/layers/attention.py"),
                            Path("shared/prompt_enhancer/qwen35_text.py")]
            report["source_sha256"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
            report["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        import llamacpp_gguf_cuda
        candidate_native = llamacpp_gguf_cuda._C
        installed_native = None
        if args.installed_native:
            spec = importlib.util.spec_from_file_location("baseline._C", args.installed_native)
            installed_native = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(installed_native)
            report.update(installed_native=str(args.installed_native),
                          installed_native_sha256=hashlib.sha256(args.installed_native.read_bytes()).hexdigest())
        binary = Path(llamacpp_gguf_cuda._C.__file__)
        report.update(native=str(binary), native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        model = load_qwen35_text_prompt_enhancer(model_path=str(args.checkpoint),
            assets_dir=str(args.checkpoint.parent), default_dtype=dtype, backend="gguf",
            requested_lm_engine="vllm", variant="27b", speculative_decoding=args.verify_tokens > 1, kv_cache_int8=True)
        model._prompt_enhancer_min_model_len_hint = max(4096, args.context + 256)
        if args.verify_tokens > 1:
            model._prompt_enhancer_speculative_tokens = args.verify_tokens - 1
            model._prompt_enhancer_speculative_sampling_tokens = args.verify_tokens - 1
        manager = offload.profile({"llm": model}, profile_no=1, budgets={"llm": 0},
            pinnedMemory=False, quantizeTransformer=False, convertWeightsFloatTo=dtype, verboseLevel=1)
        runtime = Qwen35AssistantRuntime(model)
        tokenizer = model._prompt_enhancer_tokenizer
        if args.prompt_ids:
            prompt = json.loads(args.prompt_ids.read_text(encoding="utf-8"))[:args.context]
            if len(prompt) != args.context:
                raise ValueError("Prompt fixture is shorter than requested context.")
        else:
            text = "Explain how a steam engine works, using clear examples and precise language. "
            tokens = tokenizer.encode(text, add_special_tokens=False)
            prompt = (tokens * ((args.context + len(tokens) - 1) // len(tokens)))[:args.context]
        report["prompt_sha256"] = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        with torch.inference_mode():
            runtime.prime_context(prompt, seed=123)
            # Consume an ordinary short segment to leave a real pending decode
            # token on an allocated partial page, beyond the 2048-token boundary.
            start_tokens = runtime._get_active_sequence().num_tokens
            def stop_warmup():
                seq = runtime._get_active_sequence()
                remaining = 8 - (seq.num_tokens - start_tokens)
                seq.speculative_max_emission = max(0, remaining)
                return remaining <= 0
            warmup = runtime.generate_segment(max_new_tokens=8, max_total_tokens=8, seed=123, do_sample=False,
                temperature=0.6, top_p=0.95, top_k=1, thinking_enabled=True,
                stop_requested=stop_warmup)
            runner = runtime._get_live_llm().model_runner
            seq = runtime._get_active_sequence()
            report.update(warmup_tokens=seq.num_tokens - start_tokens, warmup_stop_reason=warmup.stop_reason)
            if seq is None or len(seq.block_table) < seq.num_blocks:
                raise RuntimeError("No allocated pending decode sequence is available.")
            if args.verify_tokens > 1:
                if seq.num_tokens - start_tokens != 8:
                    raise RuntimeError(f"MTP warmup produced {seq.num_tokens - start_tokens} tokens ({warmup.stop_reason}); expected eight.")
                if runner._max_speculative_draft_tokens != args.verify_tokens - 1:
                    raise RuntimeError("Runtime draft capacity does not match the verification length.")
                if seq.seq_id in runner._speculative_pending:
                    raise RuntimeError("Warmup left pending prefill logits instead of an MTP draft state.")
                if runner.block_size - seq.last_block_num_tokens - 1 < args.verify_tokens - 1:
                    raise RuntimeError("Verification would cross the allocated partial page.")
                sample_params = runner.prepare_sample([seq], is_cfg_batch=False)
                confidence = model._prompt_enhancer_speculative_confidence
                try:
                    # A fixed two-draft target pass is the unit under test. Build
                    # actual greedy MTP proposals, without confidence truncation.
                    model._prompt_enhancer_speculative_confidence = 0.0
                    draft_tokens, mtp_cache_length, _ = runner._build_mtp_drafts(
                        seq, sample_params, args.verify_tokens - 1,
                        len(seq) - 1 + int(getattr(seq, "position_offset", 0) or 0))
                finally:
                    model._prompt_enhancer_speculative_confidence = confidence
                if len(draft_tokens) != args.verify_tokens - 1:
                    raise RuntimeError("MTP did not produce the requested two verification drafts.")
                report["draft_token_ids"] = draft_tokens.tolist() if torch.is_tensor(draft_tokens) else draft_tokens
                model.mtp.truncate_cache(mtp_cache_length)
                input_ids, positions = runner._prepare_speculative_verify(seq, draft_tokens)
            else:
                input_ids, positions = runner.prepare_decode([seq])
            context = get_context()
            if bool(context.speculative_verify) != (args.verify_tokens > 1):
                raise RuntimeError("Unexpected target verification context.")
            slots = [int(slot) for slot in context.slot_mapping.tolist()]
            if len(slots) != args.verify_tokens or len(set(slots)) != len(slots) or min(slots) < 0:
                raise RuntimeError("Verification slots must be distinct allocated cache entries.")
            named_mutable = []
            for layer_index, layer in enumerate(model.blk):
                if layer.layer_type != "linear_attention":
                    continue
                names = ("conv_state_buffer", "recurrent_state_buffer")
                if args.verify_tokens > 1:
                    names += ("speculative_conv_state_buffer", "speculative_recurrent_state_buffer")
                for name in names:
                    tensor = getattr(layer, name)
                    if name.startswith("speculative_") and tensor.shape[0] != args.verify_tokens - 1:
                        raise RuntimeError("Prefix snapshot capacity does not match the verification pass.")
                    named_mutable.append((f"layer{layer_index}.{name}", tensor))
            for slot in slots:
                block, offset = divmod(slot, runner.block_size)
                named_mutable.append((f"kv_slot{slot}", runner.kv_cache[:, :, block, offset]))
                if hasattr(runner, "kv_cache_scales"):
                    named_mutable.append((f"kv_scales_slot{slot}", runner.kv_cache_scales[:, :, block, offset]))
            mutable = [tensor for _, tensor in named_mutable]
            initial = [tensor.detach().to("cpu", copy=True) for tensor in mutable]
            flags = [(module, name, getattr(module, name)) for module in model.modules()
                     for name in ("_use_optimized_kernels", "_fuse_silu_mul") if hasattr(module, name)]
            recurrent_functions = [(layer, layer._gdn_recurrent_raw) for layer in model.blk
                                   if layer.layer_type == "linear_attention"]
            report.update(active_context=len(seq), dtype=str(dtype), input_ids=input_ids.tolist(),
                          positions=positions.tolist(), restored_state_bytes=sum(x.numel()*x.element_size() for x in initial),
                          weight_pointer=int(model.output.weight._data.data_ptr()), kv_pointer=runner.kv_cache.data_ptr())
            report.update(slot_mapping=slots, attention_context_tokens=context.max_seqlen_k if args.verify_tokens > 1 else len(seq),
                          mutable_tensors=[dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype))
                                           for name, tensor in named_mutable])

            def target_logits():
                hidden = runner.model(input_ids=input_ids, positions=positions)
                # Native MTP projects every verification row directly. The
                # ordinary compute_logits helper selects the final prefill row.
                return runner.model.output(hidden)[0] if args.verify_tokens > 1 else runner.model.compute_logits(hidden)

            def restore():
                for tensor, saved in zip(mutable, initial):
                    tensor.copy_(saved)

            def configure(kind):
                llamacpp_gguf_cuda._C = installed_native if kind in ("installed", "installed_copy") else candidate_native
                linear = kind in ("candidate", "linear")
                layout = kind in ("candidate", "rope_gdn")
                qwen_gdn.recurrent_raw_gates = recurrent_raw_gates if layout else frozen_recurrent
                for layer, function in recurrent_functions:
                    layer._gdn_recurrent_raw = function if layout else frozen_recurrent
                for module, name, original in flags:
                    setattr(module, name, original if linear else False)
                for layer in model.blk:
                    layer._gdn_direct_layout = layout
                qwen_rope_cache.supported = rope_supported if layout else lambda *args: False

            variants = ("baseline", "candidate", "linear", "rope_gdn") if args.ablations else ("baseline", "candidate")
            if installed_native is not None:
                variants = ("installed", *variants)
            if args.proof:
                variants = ("installed", "installed_copy", "baseline", "candidate")
            stream = torch.cuda.Stream(device=input_ids.device)
            for kind in variants:
                configure(kind)
                restore()
                stream.wait_stream(torch.cuda.current_stream(input_ids.device))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        target_logits()
                stream.synchronize()
                restore()
                stream.wait_stream(torch.cuda.current_stream(input_ids.device))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    logits = target_logits()
                torch.cuda.synchronize()
                graphs[kind] = (graph, logits)
                if logits.shape[0] != args.verify_tokens:
                    raise RuntimeError(f"Expected logits for all {args.verify_tokens} target rows, got {logits.shape}.")
                print(f"Captured {kind}", flush=True)

            references = {}
            for kind in variants:
                restore()
                graphs[kind][0].replay()
                actual = [graphs[kind][1].detach().to("cpu", copy=True)]
                actual.extend(tensor.detach().to("cpu", copy=True) for tensor in mutable)
                if kind == variants[0]:
                    references["one"] = actual
                else:
                    for index, (value, expected) in enumerate(zip(actual, references["one"])):
                        torch.testing.assert_close(value, expected, rtol=0, atol=0,
                                                   msg=f"{kind} one-step logits/state index {index}")
                print(f"Exact one-step logits/state: {kind}", flush=True)
            # Repeated replays exercise evolving recurrent state and repeated
            # writes to the same slot. Every variant starts at the same snapshot.
            for kind in variants:
                restore()
                for _ in range(3):
                    graphs[kind][0].replay()
                actual = [graphs[kind][1].detach().to("cpu", copy=True)]
                actual.extend(tensor.detach().to("cpu", copy=True) for tensor in mutable)
                if kind == variants[0]:
                    references["three"] = actual
                else:
                    for index, (value, expected) in enumerate(zip(actual, references["three"])):
                        torch.testing.assert_close(value, expected, rtol=0, atol=0,
                                                   msg=f"{kind} repeated logits/state index {index}")
            report["exact_logits_and_state"] = True
            report["parity_replays"] = [1, 3]
            del references
            if args.proof:
                run_proof(args, graphs, mutable, initial, report, configure, target_logits)
                report["completed"] = True
                return
            # Warm all graph variants equally before alternating their order.
            for kind in variants:
                restore()
                for _ in range(3):
                    graphs[kind][0].replay()
            torch.cuda.synchronize()
            for round_index in range(args.rounds):
                order = variants if round_index % 2 == 0 else tuple(reversed(variants))
                times = {}
                for kind in order:
                    restore()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.replays):
                        graphs[kind][0].replay()
                    end.record()
                    end.synchronize()
                    times[kind] = start.elapsed_time(end) / args.replays
                report["records"].append(dict(round=round_index, order=order, gpu_ms=times,
                    speedup=times["baseline"] / times["candidate"]))
                if installed_native is not None:
                    report["records"][-1]["installed_speedup"] = times["installed"] / times["candidate"]
                print(json.dumps(report["records"][-1]), flush=True)
            report["median_gpu_ms"] = {kind: statistics.median(r["gpu_ms"][kind] for r in report["records"]) for kind in variants}
            report["median_paired_speedup"] = statistics.median(r["speedup"] for r in report["records"])
            if installed_native is not None:
                report["median_installed_paired_speedup"] = statistics.median(r["installed_speedup"] for r in report["records"])
        report["completed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        qwen_rope_cache.supported = rope_supported
        qwen_gdn.recurrent_raw_gates = recurrent_raw_gates
        if candidate_native is not None:
            llamacpp_gguf_cuda._C = candidate_native
        torch.cuda.synchronize()
        graphs.clear()
        reset_context()
        if model is not None:
            model.unload()
        if manager is not None:
            manager.release()
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k in ("median_gpu_ms", "median_paired_speedup", "exact_logits_and_state")}), flush=True)


if __name__ == "__main__":
    main()
