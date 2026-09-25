"""Measure the GGUF short-batch tensor-core linear on this GPU and write a shareable report.

Speculative decoding verifies 2-8 tokens per step. For Q4_K and PTQ1_0 weights the
llamacpp_gguf_cuda package can run these linears with llama.cpp MMVQ or with INT8
tensor cores (compute capability 8.0+). WanGP already picks the faster one per shape
when a model loads; this optional tool reports accuracy and speed of both on synthetic
Qwen3.8-27B-sized weights, so results from other GPUs can be compared.

    python tools/validate_gguf_short_batch.py --output gguf_short_batch_report.json
"""
import argparse
import json
import math
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.kernels import gguf_short_batch  # noqa: E402

SHAPES = {"Q4_K": [(17408, 5120), (5120, 17408), (6144, 5120), (1024, 5120)],
          "PTQ1_0": [(17408, 5120), (5120, 17408), (6144, 5120), (1024, 5120)]}
BYTES = {"Q4_K": (256, 144), "PTQ1_0": (128, 28)}


def synthetic(qtype, rows, cols, generator, device):
    values, size = BYTES[qtype]
    blocks = torch.randint(0, 256, (rows * cols // values, size), generator=generator, dtype=torch.uint8)
    if qtype == "Q4_K":
        scales = torch.empty(blocks.shape[0], 2, dtype=torch.float16)
        scales[:, 0].uniform_(0.002, 0.02, generator=generator)
        scales[:, 1].uniform_(0.0, 0.01, generator=generator)
        blocks[:, :4] = scales.view(torch.uint8).reshape(-1, 4)
    else:
        blocks[:, :24] = torch.randint(0, 243, (blocks.shape[0], 24), generator=generator, dtype=torch.uint8)
        blocks[:, 24:26] = torch.randint(0, 81, (blocks.shape[0], 2), generator=generator, dtype=torch.uint8)
        blocks[:, 26:28] = torch.empty(blocks.shape[0], 1, dtype=torch.float16).uniform_(0.005, 0.03, generator=generator).view(torch.uint8)
    return blocks.reshape(-1).to(device)


def dense_rows(qtype, raw, rows, cols):
    """FP32 reference for the first `rows` output rows."""
    values, size = BYTES[qtype]
    blocks = raw[:rows * cols // values * size].reshape(-1, size).cpu()
    if qtype == "PTQ1_0":
        from shared.qtypes.gguf import _dequantize_blocks_PTQ1_0
        return _dequantize_blocks_PTQ1_0(blocks, values, size, torch.float32).reshape(rows, cols)
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    return torch.from_numpy(dequantize(blocks.numpy(), GGMLQuantizationType.Q4_K)).float().reshape(rows, cols)


def relative_error(value, reference):
    return float((value.float() - reference).norm() / reference.norm())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("gguf_short_batch_report.json"))
    parser.add_argument("--rows", type=int, nargs="+", default=list(range(2, 9)))
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    import llamacpp_gguf_cuda as native
    if not torch.cuda.is_available() or torch.version.hip is not None:
        sys.exit("A CUDA GPU is required.")
    if not native.has_short_batch_policy():
        sys.exit(f"llamacpp_gguf_cuda {native.__version__} has no short-batch tensor-core path; update the package.")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    if props.major < 8:
        sys.exit(f"{props.name} (compute capability {props.major}.{props.minor}) has no INT8 MMA/cp.async path; nothing to measure.")
    dtype = getattr(torch, args.dtype)
    l2 = int(getattr(props, "L2_cache_size", 0))
    generator = torch.Generator().manual_seed(0)
    report = dict(gpu=props.name, compute_capability=f"{props.major}.{props.minor}", multiprocessors=props.multi_processor_count, l2_bytes=l2,
                  memory_gib=round(props.total_memory / 2**30, 1), driver=gguf_short_batch._driver_version(), package=native.__version__,
                  torch=torch.__version__, cuda=torch.version.cuda, platform=platform.platform(), dtype=args.dtype,
                  method_version=gguf_short_batch.METHOD_VERSION, margin=gguf_short_batch.MARGIN, results=[])
    default_mma = (props.major, props.minor) == (12, 0)
    print(f"{props.name}  cc {props.major}.{props.minor}  {props.multi_processor_count} SMs  L2 {l2 / 2**20:.0f} MB  package {native.__version__}")
    print(f"{'format':7s} {'out x in':>13s} {'rows':>4s} {'MMVQ us':>8s} {'MMA us':>8s} {'speedup':>7s} {'MMVQ GB/s':>9s} {'err MMVQ':>9s} {'err MMA':>9s}  auto")
    with torch.inference_mode():
        for qtype, shapes in SHAPES.items():
            if not native.supports_linear_qtype_name(qtype):
                continue
            for out_features, in_features in shapes:
                values, size = BYTES[qtype]
                weight_bytes = out_features * in_features // values * size
                # Rotate over enough distinct weights to defeat L2, as WanGP does with a model's layers.
                count = max(8, min(64, math.ceil(4 * l2 / weight_bytes), (2 << 30) // weight_bytes))
                peers = [synthetic(qtype, out_features, in_features, generator, device) for _ in range(count)]
                check_rows = min(out_features, 128)
                reference_weight = dense_rows(qtype, peers[0], check_rows, in_features).to(device)
                for rows in args.rows:
                    x = torch.randn((rows, in_features), generator=generator).to(device, dtype)
                    reference = x.float() @ reference_weight.T
                    errors = {}
                    for mode in ("native", "mma"):
                        native.set_short_batch_mode(mode)
                        errors[mode] = relative_error(native.linear(peers[0], qtype, (out_features, in_features), x, None, dtype)[:, :check_rows], reference)
                    native.set_short_batch_mode("auto")
                    # The host-side reference work above lets the GPU idle: warm up before timing.
                    timings, agreement = gguf_short_batch._measure(native, qtype, (out_features, in_features), peers, x, dtype, warm_seconds=.05)
                    entry = dict(qtype=qtype, out_features=out_features, in_features=in_features, rows=rows, weights=count,
                                 error_native=errors["native"], error_mma=errors["mma"], mma_vs_native=agreement, accurate=timings is not None)
                    if timings is not None:
                        speedup = timings["native"] / timings["mma"]
                        auto = speedup > gguf_short_batch.MARGIN if default_mma else speedup >= 1 / gguf_short_batch.MARGIN
                        entry.update(native_us=1000 * timings["native"], mma_us=1000 * timings["mma"], speedup=speedup, auto_uses_mma=auto,
                                     native_gbps=weight_bytes / timings["native"] / 1e6, mma_gbps=weight_bytes / timings["mma"] / 1e6)
                        print(f"{qtype:7s} {out_features:>6d}x{in_features:<6d} {rows:>4d} {entry['native_us']:8.1f} {entry['mma_us']:8.1f} {speedup:6.2f}x "
                              f"{entry['native_gbps']:9.0f} {errors['native']:9.2e} {errors['mma']:9.2e}  {'MMA' if auto else 'MMVQ'}")
                    else:
                        print(f"{qtype:7s} {out_features:>6d}x{in_features:<6d} {rows:>4d}  tensor cores disagree with MMVQ ({agreement:.1e}) -> MMVQ")
                    report["results"].append(entry)
                del peers, reference_weight
                torch.cuda.empty_cache()  # synthetic weights of the next shape
    for qtype in SHAPES:
        speedups = [r["speedup"] for r in report["results"] if r["qtype"] == qtype and r["accurate"]]
        if speedups:
            report[f"{qtype}_geomean_speedup"] = math.exp(sum(map(math.log, speedups)) / len(speedups))
            print(f"{qtype}: geometric-mean tensor-core speedup {report[f'{qtype}_geomean_speedup']:.2f}x over {len(speedups)} cases")
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Report written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
