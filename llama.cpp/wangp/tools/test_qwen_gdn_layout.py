"""Exact GDN layout, graph-replay and rollback checks; optional isolated timings.

Run in the supported CUDA environment with no competing GPU work:
  python tools/test_qwen_gdn_layout.py --benchmark --output <report.json>

The reference materializes the existing model permutations before the unchanged
raw recurrence. The candidate addresses the original tensors directly. Timings
include recurrence, gated RMSNorm and all reference permutations, in CUDA graphs.
Graph memory counters describe PyTorch allocations, not driver or whole-model VRAM.
"""
import argparse
import gc
import itertools
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from shared.kernels.qwen_gdn import _recurrent_raw_kernel, recurrent_raw_gates
from shared.llm_engines.nanovllm.models.qwen3_5 import (
    _DEFAULT_FUSED_RMSNORM_GATED,
    _forward_gated_norm_list,
    _interleave_axis_halves,
    _maybe_reorder_gguf_ssm_param,
    _reorder_v_head_axis_grouped_to_tiled,
    _reorder_v_head_axis_tiled_to_grouped,
    _reorder_v_heads_tiled_to_grouped,
)


def exact(actual, expected, name):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)


def measure(graph, repeats):
    values = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / repeats)
    return statistics.median(values)


def compiled_resources():
    """Inspect kernels already launched by this process; never compile or launch."""
    records = []
    for device, caches in _recurrent_raw_kernel.device_caches.items():
        for kernel in caches[0].values():
            constants = {_recurrent_raw_kernel.arg_names[key[0]]: value
                         for key, value in kernel.src.constants.items() if len(key) == 1}
            records.append(dict(device=device, constants=constants, signature=kernel.src.signature,
                                n_regs=kernel.n_regs, n_spills=kernel.n_spills,
                                shared_bytes=kernel.metadata.shared,
                                num_warps=kernel.metadata.num_warps, num_stages=kernel.metadata.num_stages))
    return records


@torch.inference_mode()
def check_case(shape, tokens, dtype, flags, *, graph_checks, benchmark, repeats, state_dtype=None):
    batch, heads, value_heads, dim_k, dim_v = shape
    device = torch.device("cuda", 0)
    v_tiled, params_tiled, interleave = flags
    layout = dict(v_heads_tiled=v_tiled, ssm_params_tiled=params_tiled, interleave_ab=interleave)
    state_dtype = dtype if state_dtype is None else state_dtype
    # Real projection slices have a wider token stride than their logical width.
    packed = torch.randn(batch, tokens, 2 * heads * dim_k + value_heads * dim_v, device=device, dtype=dtype)
    q, k, v = packed.split((heads * dim_k, heads * dim_k, value_heads * dim_v), -1)
    q, k = (x.reshape(batch, tokens, heads, dim_k) for x in (q, k))
    v = v.reshape(batch, tokens, value_heads, dim_v)
    gates = torch.randn(batch, tokens, value_heads * (dim_v + 2), device=device, dtype=dtype)
    z, a, b = gates.split((value_heads * dim_v, value_heads, value_heads), -1)
    z = z.reshape(batch, tokens, value_heads, dim_v)
    ssm_a = -torch.rand(value_heads, device=device, dtype=torch.float32)
    ssm_dt = torch.randn(value_heads, device=device, dtype=torch.float32)
    initial = torch.randn(batch, value_heads, dim_k, dim_v, device=device, dtype=state_dtype)
    states = [initial.clone(), initial.clone()]
    snapshots = [initial.new_empty((tokens - 1, *initial.shape)) if tokens > 1 else None for _ in range(2)]
    norm = _DEFAULT_FUSED_RMSNORM_GATED(dim_v, eps=1e-6).to(device=device, dtype=dtype)
    norm.weight.uniform_(0.5, 1.5)

    def run(direct):
        index = int(direct)
        vv, aa, bb, zz = v, a, b, z
        sa, dt = ssm_a, ssm_dt
        if not direct:
            if v_tiled:
                vv, zz = (_reorder_v_head_axis_tiled_to_grouped(x, 2, heads, value_heads) for x in (v, z))
                aa, bb = (_reorder_v_heads_tiled_to_grouped(x, -1, heads, value_heads, 1) for x in (a, b))
            elif interleave:
                aa, bb = (_interleave_axis_halves(x, -1) for x in (a, b))
            sa, dt = (_maybe_reorder_gguf_ssm_param(x, interleave_halves=interleave,
                       tiled_to_grouped=params_tiled, num_k_heads=heads, num_v_heads=value_heads)
                      for x in (ssm_a, ssm_dt))
        out, _ = recurrent_raw_gates(q, k, vv, aa, bb, sa, dt, states[index], snapshots[index],
                                     **(layout if direct else {}))
        normalized = _forward_gated_norm_list(norm, [out.reshape(-1, dim_v), zz.reshape(-1, dim_v)])
        normalized = normalized.reshape(batch, tokens, value_heads, dim_v)
        if not direct and v_tiled:
            normalized = _reorder_v_head_axis_grouped_to_tiled(normalized, 2, heads, value_heads)
        return out, normalized

    def compare(outputs, label):
        expected_raw = (_reorder_v_head_axis_grouped_to_tiled(outputs[0][0], 2, heads, value_heads)
                        if v_tiled else outputs[0][0])
        exact(outputs[1][0], expected_raw, f"{label}: raw output")
        exact(outputs[1][1], outputs[0][1], f"{label}: gated norm/output layout")
        exact(states[1], states[0], f"{label}: live recurrent state")
        if tokens > 1:
            exact(snapshots[1], snapshots[0], f"{label}: grouped prefix snapshots")

    outputs = [run(False), run(True)]
    compare(outputs, "eager")
    del outputs
    result = dict(shape=shape, tokens=tokens, dtype=str(dtype), state_dtype=str(state_dtype),
                  layout=layout, exact=True, graph_replays=0, rollback_prefixes=[])
    if not graph_checks:
        return result

    graphs, outputs, live_bytes, peak_bytes, reserved_bytes = [], [], [], [], []
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    for direct in (False, True):
        with torch.cuda.stream(stream):
            for _ in range(3):
                run(direct)
        stream.synchronize()
        before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = run(direct)
        graphs.append(graph)
        outputs.append(output)
        live_bytes.append(torch.cuda.memory_allocated(device) - before)
        peak_bytes.append(torch.cuda.max_memory_allocated(device) - before)
        reserved_bytes.append(torch.cuda.memory_reserved(device) - reserved_before)
    for replay in range(max(3, tokens)):
        # Later replays start from each possible committed prefix. All states
        # retain the original grouped layout, including when output is tiled.
        if replay and replay < tokens:
            for state, prefix in zip(states, snapshots):
                state.copy_(prefix[replay - 1])
            result["rollback_prefixes"].append(replay)
        else:
            initial.normal_()
            for state in states:
                state.copy_(initial)
        packed.normal_()
        gates.normal_()
        ssm_a.uniform_(-1.5, -0.01)
        ssm_dt.normal_()
        for graph in graphs:
            graph.replay()
        compare(outputs, f"graph replay {replay}")
        result["graph_replays"] += 1
    result["graph_live_tensor_bytes"] = dict(materialized=live_bytes[0], direct=live_bytes[1])
    result["graph_capture_peak_bytes"] = dict(materialized=peak_bytes[0], direct=peak_bytes[1])
    result["graph_reserved_growth_bytes"] = dict(materialized=reserved_bytes[0], direct=reserved_bytes[1])
    if benchmark:
        # Alternate order to reduce simple warmup/clock ordering bias.
        timings = [[], []]
        for order in ((0, 1), (1, 0), (0, 1)):
            for index in order:
                timings[index].append(measure(graphs[index], repeats))
        original, direct = (statistics.median(values) for values in timings)
        result["gpu_ms"] = dict(materialized=original, direct=direct, speedup=original / direct)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--focus", action="store_true", help="Only the real 27B tiled-layout cases")
    parser.add_argument("--tokens", nargs="+", type=int)
    parser.add_argument("--dtypes", nargs="+", choices=("fp16", "bf16"), default=("fp16", "bf16"))
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("This harness requires an NVIDIA CUDA GPU.")
    if _DEFAULT_FUSED_RMSNORM_GATED is None:
        raise SystemExit("The supported FLA gated RMSNorm implementation is required.")
    torch.manual_seed(9917)
    report = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), cases=[])
    flags = [(True, True, False)] if args.focus else list(itertools.product((False, True), repeat=3))
    counts = args.tokens or ((1, 3, 5) if args.quick else (1, 2, 3, 5, 7, 9))
    dtypes = [{"fp16": torch.float16, "bf16": torch.bfloat16}[name] for name in args.dtypes]
    cases = [(shape, count, dtype, layout, None)
             for shape in ((1, 16, 48, 128, 128),)
             for count in counts for dtype in dtypes for layout in flags]
    if not args.quick and not args.focus:
        # Include batched/unequal dimensions, odd heads (interleave is a no-op),
        # and FP32 state storage without changing the production state dtype.
        cases.extend((shape, 3, torch.bfloat16, layout, torch.float32)
                     for shape in ((2, 4, 8, 64, 96), (1, 3, 9, 32, 48)) for layout in flags)
        cases.append(((1, 4, 8, 64, 64), 3, torch.float32, (True, True, False), None))
    for index, (shape, count, dtype, layout, state_dtype) in enumerate(cases, 1):
        graphs = layout == (True, True, False) or count == 3
        result = check_case(shape, count, dtype, layout, graph_checks=graphs,
                            benchmark=args.benchmark and layout == (True, True, False),
                            repeats=args.repeats, state_dtype=state_dtype)
        report["cases"].append(result)
        print(f"[{index}/{len(cases)}] exact T={count} {dtype} layout={layout} "
              f"graphs={result['graph_replays']} {result.get('gpu_ms', '')}", flush=True)
        gc.collect()
    report["passed"] = len(report["cases"])
    report["compiled_resources"] = compiled_resources()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
