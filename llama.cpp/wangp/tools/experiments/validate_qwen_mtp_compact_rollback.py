"""GPU-only validation for the unintegrated compact GDN rollback experiment.

Example (single-layer 27B dimensions, bounded matrix):
  python tools/experiments/validate_qwen_mtp_compact_rollback.py --tokens 3 \
      --layouts grouped tiled interleaved --graph --benchmark --output result.json

No production module imports this tool or its candidate kernel. Exactness failures
exit nonzero and suppress timing; there is deliberately no tolerance override.
"""
import argparse
import gc
import itertools
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import triton
from shared.kernels.qwen_gdn import _recurrent_raw_kernel, recurrent_raw_gates
from tools.experiments.qwen_mtp_compact_rollback import (
    BatchedReplay, CompactWorkspace, _record_raw_kernel, _replay_prefix_kernel,
    _replay_batched_kernel, record_raw_gates, replay_batched_prefix, replay_prefix,
)

LAYOUTS = {
    "grouped": (False, False, False),
    "tiled": (True, True, False),
    "interleaved": (False, False, True),
    "tiled_interleaved": (True, True, True),
}


def bitwise_equal(actual, expected, label):
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError(f"{label}: shape/dtype mismatch")
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise AssertionError(f"{label}: nonfinite result")
    integer_dtype = {2: torch.int16, 4: torch.int32}[actual.element_size()]
    unequal = actual.contiguous().view(integer_dtype) != expected.contiguous().view(integer_dtype)
    if bool(unequal.any()):
        count = int(unequal.sum())
        maximum = float((actual.float() - expected.float()).abs().max())
        raise AssertionError(f"{label}: {count}/{actual.numel()} differing bit patterns, max_abs={maximum}")


def fixture(batch, tokens, *, dtype, device, seed, strided):
    generator = torch.Generator(device=device).manual_seed(seed)

    def rand(shape, scale=0.2, selected_dtype=dtype):
        return torch.randn(shape, device=device, dtype=selected_dtype, generator=generator) * scale

    h, hv, dk, dv = 16, 48, 128, 128
    if strided:
        packed = rand((batch, tokens, 2 * h * dk + hv * dv))
        q = packed[..., :h * dk].view(batch, tokens, h, dk)
        k = packed[..., h * dk:2 * h * dk].view(batch, tokens, h, dk)
        v = packed[..., 2 * h * dk:].view(batch, tokens, hv, dv)
        gate_packed = rand((batch, tokens, 2 * hv), 3.0)
        a, b = gate_packed[..., :hv], gate_packed[..., hv:]
    else:
        q, k, v = rand((batch, tokens, h, dk)), rand((batch, tokens, h, dk)), rand((batch, tokens, hv, dv))
        a, b = rand((batch, tokens, hv), 3.0), rand((batch, tokens, hv), 3.0)
    ssm_a = -torch.exp(rand((hv,), 0.3, torch.float32))
    ssm_dt = rand((hv,), 0.5, torch.float32)
    # Exercise zero norms, softplus's threshold, saturated beta, and nonzero state.
    q[:, 0, 0].zero_()
    k[:, 0, 1].zero_()
    ssm_dt[:6].zero_()
    a[:, 0, :6] = torch.tensor([-30, -0.0, 0, 19.875, 20, 20.125], device=device, dtype=dtype)
    b[:, 0, :6] = torch.tensor([-30, -8, -0.0, 0, 8, 30], device=device, dtype=dtype)
    initial = rand((batch, hv, dk, dv), 0.1)
    initial[..., 0, 0] = -0.0
    return (q, k, v, a, b, ssm_a, ssm_dt), initial


def make_workspace(inputs, state, capacity):
    q, k, v, _, b, _, _ = inputs
    return CompactWorkspace.allocate(state, max_tokens=capacity, key_heads=q.shape[2],
                                     key_dtype=k.dtype, value_dtype=v.dtype, beta_dtype=b.dtype)


def reference(inputs, initial, layout):
    state = initial.clone()
    snapshots = initial.new_empty((inputs[0].shape[1] - 1, *initial.shape))
    output, _ = recurrent_raw_gates(*inputs, state, snapshots, **layout)
    return output, state, snapshots


def graph_capture(fn, device):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.synchronize(device)
    allocation = {
        "allocated_growth_bytes": torch.cuda.memory_allocated(device) - before,
        "reserved_growth_bytes": torch.cuda.memory_reserved(device) - before_reserved,
    }
    return graph, allocation


def validate_case(inputs, initial, layout, *, capacity, graphs):
    tokens = inputs[0].shape[1]
    expected, expected_final, expected_prefix = reference(inputs, initial, layout)
    state = initial.clone()
    workspace = make_workspace(inputs, state, capacity)
    pointers = tuple(t.data_ptr() for t in workspace.tensors())
    output = torch.empty_like(expected)
    record_raw_gates(*inputs, state, workspace, output=output, **layout)
    bitwise_equal(output, expected, "verify output")
    bitwise_equal(state, expected_final, "verify final state")
    bitwise_equal(workspace.initial, initial, "initial backup")
    # Full acceptance must leave verification's live final state untouched.
    replay_prefix(state, workspace, tokens, tokens)
    bitwise_equal(state, expected_final, "full acceptance")
    for prefix in reversed(range(1, tokens)):
        replay_prefix(state, workspace, prefix, tokens)
        bitwise_equal(state, expected_prefix[prefix - 1], f"prefix {prefix}")
        bitwise_equal(workspace.initial, initial, f"backup after prefix {prefix}")
    # Run a shorter verification on the same physical capacity after overwriting
    # the input state; this exposes accidentally using current T as log stride.
    shorter = max(2, tokens - 1)
    shortened = tuple(t[:, :shorter] if i < 5 else t for i, t in enumerate(inputs))
    state.copy_(initial * 0.5)
    expected_short, expected_short_state, expected_short_prefix = reference(shortened, state, layout)
    short_output = output[:, :shorter].contiguous()
    record_raw_gates(*shortened, state, workspace, output=short_output, **layout)
    bitwise_equal(short_output, expected_short, "reused verify output")
    bitwise_equal(state, expected_short_state, "reused verify state")
    replay_prefix(state, workspace, 1, shorter)
    bitwise_equal(state, expected_short_prefix[0], "reused prefix")

    if graphs:
        for prefix in range(1, tokens + 1):
            def operation():
                state.copy_(initial)
                record_raw_gates(*inputs, state, workspace, output=output, **layout)
                replay_prefix(state, workspace, prefix, tokens)

            graph = None
            try:
                graph, _ = graph_capture(operation, state.device)
                # Fixed pointers, different contents after capture.
                for tensor in inputs[:5]:
                    tensor.add_(0.015625)
                initial.mul_(0.875)
                expected, expected_final, expected_prefix = reference(inputs, initial, layout)
                graph.replay()
                torch.cuda.synchronize(state.device)
                bitwise_equal(output, expected, f"graph output {prefix}")
                committed = expected_final if prefix == tokens else expected_prefix[prefix - 1]
                bitwise_equal(state, committed, f"graph prefix {prefix}")
                bitwise_equal(workspace.initial, initial, f"graph backup {prefix}")
            finally:
                if graph is not None:
                    graph.reset()
    if pointers != tuple(t.data_ptr() for t in workspace.tensors()):
        raise AssertionError("Workspace pointers changed")
    return {"tokens": tokens, "batch": initial.shape[0], "layout": layout,
            "workspace_bytes": workspace.nbytes,
            "old_snapshot_bytes": (tokens - 1) * initial.numel() * initial.element_size(),
            "bitwise_pass": True, "graphs": graphs}


def baseline_into(inputs, state, output, snapshots, layout):
    q, k, v, a, b, ssm_a, ssm_dt = inputs
    batch, tokens, heads, dim_k = q.shape
    hv, dim_v = v.shape[-2:]
    _recurrent_raw_kernel[(triton.cdiv(dim_v, 8), batch * hv)](
        *inputs, state, output, snapshots, tokens, heads, hv, dim_k, dim_v,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        *a.stride()[:2], *b.stride()[:2], batch, triton.next_power_of_2(dim_k), 8,
        True, layout["v_heads_tiled"], layout["ssm_params_tiled"], layout["interleave_ab"],
        num_warps=1, num_stages=3,
    )


def time_graph(graph, device, repeats=7, iterations=200):
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def benchmark(inputs, initial, layout):
    """One layer only; includes identical input-state reset in both paths."""
    tokens = inputs[0].shape[1]
    state = initial.clone()
    output = torch.empty(inputs[2].shape, dtype=inputs[2].dtype, device=initial.device)
    snapshots = initial.new_empty((tokens - 1, *initial.shape))
    workspace = make_workspace(inputs, state, tokens)
    rows = []
    for prefix in range(1, tokens + 1):
        graphs = []
        try:
            def baseline():
                state.copy_(initial)
                baseline_into(inputs, state, output, snapshots, layout)
                if prefix < tokens:
                    torch._foreach_copy_([state], [snapshots[prefix - 1]])

            def compact():
                state.copy_(initial)
                record_raw_gates(*inputs, state, workspace, output=output, **layout)
                replay_prefix(state, workspace, prefix, tokens)

            baseline_graph, baseline_allocation = graph_capture(baseline, initial.device)
            graphs.append(baseline_graph)
            compact_graph, compact_allocation = graph_capture(compact, initial.device)
            graphs.append(compact_graph)
            baseline_before = time_graph(baseline_graph, initial.device)
            compact_time = time_graph(compact_graph, initial.device)
            baseline_after = time_graph(baseline_graph, initial.device)
            baseline_us = (baseline_before["median_us"] + baseline_after["median_us"]) / 2
            rows.append({
                "prefix": prefix, "tokens": tokens,
                "baseline_before": baseline_before, "compact": compact_time,
                "baseline_after": baseline_after,
                "speedup_ratio": baseline_us / compact_time["median_us"],
                "baseline_capture": baseline_allocation, "compact_capture": compact_allocation,
            })
        finally:
            for graph in graphs:
                graph.reset()
    return {"scope": "One GDN layer; recurrence only, convolution unchanged; state reset included",
            "old_rollback_bytes": snapshots.numel() * snapshots.element_size(),
            "compact_rollback_bytes": workspace.nbytes, "timings": rows}


def benchmark_batched(layers, tokens, *, dtype, device, layout, strided):
    """Uniform independent layers; no model/runner integration or depth change."""
    fixtures = [fixture(1, tokens, dtype=dtype, device=device, seed=73000 + i, strided=strided)
                for i in range(layers)]
    inputs = [item[0] for item in fixtures]
    seeds = [item[1] for item in fixtures]
    states = [seed.clone() for seed in seeds]
    outputs = [torch.empty(item[2].shape, device=device, dtype=dtype) for item in inputs]
    snapshots = [seed.new_empty((tokens - 1, *seed.shape)) for seed in seeds]
    workspaces = [make_workspace(item, state, tokens) for item, state in zip(inputs, states)]
    batch = BatchedReplay.prepare(states, workspaces)
    rows = []
    for prefix in range(1, tokens + 1):
        graphs = []
        try:
            def baseline():
                torch._foreach_copy_(states, seeds)
                for item, state, output, snapshot in zip(inputs, states, outputs, snapshots):
                    baseline_into(item, state, output, snapshot, layout)
                if prefix < tokens:
                    torch._foreach_copy_(states, [snapshot[prefix - 1] for snapshot in snapshots])

            def compact_record():
                torch._foreach_copy_(states, seeds)
                for item, state, output, workspace in zip(inputs, states, outputs, workspaces):
                    record_raw_gates(*item, state, workspace, output=output, **layout)

            def compact_batched():
                compact_record()
                replay_batched_prefix(batch, prefix, tokens)

            def compact_separate():
                compact_record()
                for state, workspace in zip(states, workspaces):
                    replay_prefix(state, workspace, prefix, tokens)

            baseline_graph, baseline_allocation = graph_capture(baseline, device)
            graphs.append(baseline_graph)
            batched_graph, batched_allocation = graph_capture(compact_batched, device)
            graphs.append(batched_graph)
            separate_graph, separate_allocation = graph_capture(compact_separate, device)
            graphs.append(separate_graph)
            # Change contents after capture without rebuilding any table or graph.
            for index in (0, layers - 1):
                seeds[index].mul_(0.875)
                for tensor in inputs[index][:5]:
                    tensor.add_(0.015625)
            baseline_graph.replay()
            expected_states = [state.clone() for state in states]
            expected_outputs = [output.clone() for output in outputs]
            for name, graph in (("batched", batched_graph), ("separate", separate_graph)):
                graph.replay()
                for index, (state, output, expected_state, expected_output) in enumerate(
                        zip(states, outputs, expected_states, expected_outputs)):
                    bitwise_equal(state, expected_state, f"{name} layer {index} prefix {prefix}")
                    bitwise_equal(output, expected_output, f"{name} layer {index} output {prefix}")
                    bitwise_equal(workspaces[index].initial, seeds[index], f"{name} layer {index} backup {prefix}")
            del expected_states, expected_outputs
            baseline_before = time_graph(baseline_graph, device)
            batched_time = time_graph(batched_graph, device)
            separate_time = time_graph(separate_graph, device)
            baseline_after = time_graph(baseline_graph, device)
            baseline_us = (baseline_before["median_us"] + baseline_after["median_us"]) / 2
            rows.append({"prefix": prefix, "tokens": tokens, "bitwise_pass": True,
                         "baseline_before": baseline_before, "compact_batched": batched_time,
                         "compact_separate": separate_time, "baseline_after": baseline_after,
                         "batched_speedup_ratio": baseline_us / batched_time["median_us"],
                         "separate_speedup_ratio": baseline_us / separate_time["median_us"],
                         "baseline_capture": baseline_allocation,
                         "batched_capture": batched_allocation, "separate_capture": separate_allocation})
            print(json.dumps({"batched_prefix": rows[-1]}), flush=True)
        finally:
            for graph in graphs:
                graph.reset()
    return {"scope": "Independent uniform GDN layers, 27B dimensions; recurrence only; matched foreach state reset included",
            "layers": layers, "tokens": tokens,
            "old_rollback_bytes": sum(t.numel() * t.element_size() for t in snapshots),
            "compact_rollback_bytes": sum(w.nbytes for w in workspaces),
            "pointer_table_bytes": batch.pointers.numel() * batch.pointers.element_size(),
            "timings": rows}


def resources():
    report = {}
    for function in (_recurrent_raw_kernel, _record_raw_kernel, _replay_prefix_kernel, _replay_batched_kernel):
        variants = []
        for cache in function.device_caches.values():
            for kernel in cache[0].values():
                variants.append({"registers": getattr(kernel, "n_regs", None),
                                 "spills": getattr(kernel, "n_spills", None),
                                 "shared_bytes": kernel.metadata.shared})
        report[function.__name__] = variants
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--batch", type=int, nargs="+", default=[1])
    parser.add_argument("--capacity", type=int, default=9)
    parser.add_argument("--layouts", nargs="+", choices=list(LAYOUTS), default=list(LAYOUTS))
    parser.add_argument("--all-layouts", action="store_true")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--contiguous", action="store_true", help="Use separate contiguous projections instead of strided slices")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Time only the first selected case, after all exactness checks pass")
    parser.add_argument("--batched-layers", type=int, default=0, help="Validate and time this many uniform layers at the first selected T/layout")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.tokens) < 2 or max(args.tokens) > args.capacity or min(args.batch) < 1:
        parser.error("Require 2 <= tokens <= capacity and positive batch")
    if torch.version.hip is not None or not torch.cuda.is_available():
        parser.error("This isolated experiment requires NVIDIA CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    layouts = list(itertools.product((False, True), repeat=3)) if args.all_layouts else [LAYOUTS[name] for name in args.layouts]
    result = {"experiment_only": True, "torch": torch.__version__, "triton": triton.__version__,
              "device": torch.cuda.get_device_name(device), "dtype": args.dtype,
              "capacity": args.capacity, "strided": not args.contiguous,
              "validation": [], "status": "running"}
    try:
        for batch, tokens, flags in itertools.product(args.batch, args.tokens, layouts):
            layout = dict(zip(("v_heads_tiled", "ssm_params_tiled", "interleave_ab"), flags))
            inputs, initial = fixture(batch, tokens, dtype=dtype, device=device, seed=72019, strided=not args.contiguous)
            row = validate_case(inputs, initial, layout, capacity=args.capacity, graphs=args.graph)
            result["validation"].append(row)
            print(json.dumps(row), flush=True)
            del inputs, initial
            gc.collect()
        if args.benchmark:
            layout = dict(zip(("v_heads_tiled", "ssm_params_tiled", "interleave_ab"), layouts[0]))
            inputs, initial = fixture(args.batch[0], args.tokens[0], dtype=dtype, device=device, seed=72019, strided=not args.contiguous)
            result["benchmark"] = benchmark(inputs, initial, layout)
            print(json.dumps({"benchmark": result["benchmark"]}), flush=True)
        if args.batched_layers:
            layout = dict(zip(("v_heads_tiled", "ssm_params_tiled", "interleave_ab"), layouts[0]))
            result["batched_benchmark"] = benchmark_batched(
                args.batched_layers, args.tokens[0], dtype=dtype, device=device,
                layout=layout, strided=not args.contiguous)
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        raise
    finally:
        result["resources"] = resources()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "cases": len(result["validation"]), "resources": result["resources"]}), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
