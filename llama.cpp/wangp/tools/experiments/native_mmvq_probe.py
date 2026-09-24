"""Build/run an isolated Q4_K/Q6_K N3 scheduling experiment; never installs it.

Build (CPU/compiler only): python native_mmvq_probe.py --build
GPU run, after obtaining the shared GPU slot: python native_mmvq_probe.py --run

Each timed graph flushes 256 MiB before its start event. The flush is OUTSIDE
the measured interval and exceeds twice the RTX 5090's 96 MiB L2. Alternate
variant order per round. Report both kernel-only and complete Q8+MMVQ timings;
compare the row2 control against the installed production linear operation.
The flush and simultaneous experiment graphs are test infrastructure, not a
proposed production allocation or a production peak-memory measurement.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
DEFAULT_BUILD = Path("D:/AMD/cuda-fusions-20260922/native-mmvq-probe")
VARIANTS = {0: "row2_serial", 1: "row2_distributed", 2: "row1_serial", 3: "row4_serial", 4: "row4_distributed"}
DEFAULT_WEIGHTS = (
    "blk.0.ffn_gate.weight+blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight", "blk.0.attn_qkv.weight",
    "blk.0.ssm_out.weight", "blk.3.attn_q.weight", "output.weight",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_environment():
    """Child-process environment only; never invokes vcvars or edits the shell."""
    env = os.environ.copy()
    if os.name != "nt":
        return env
    vc_root = Path("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC")
    vc = sorted(p for p in vc_root.iterdir() if p.is_dir())[-1]
    sdk = Path("C:/Program Files (x86)/Windows Kits/10")
    version = sorted(p.name for p in (sdk / "Lib").iterdir() if p.is_dir())[-1]
    env["PATH"] = os.pathsep.join((str(vc / "bin/Hostx64/x64"), str(sdk / "bin" / version / "x64"), env.get("PATH", "")))
    env["INCLUDE"] = os.pathsep.join(str(p) for p in (
        vc / "include", *(sdk / "Include" / version / x for x in ("ucrt", "shared", "um", "winrt", "cppwinrt"))))
    env["LIB"] = os.pathsep.join(str(p) for p in (vc / "lib/x64", sdk / "Lib" / version / "ucrt/x64", sdk / "Lib" / version / "um/x64"))
    return env


def build(args):
    args.build_dir.mkdir(parents=True, exist_ok=True)
    env = build_environment()
    nvcc = args.cuda_root / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    ggml = args.kernel_repo / "_vendor/llama.cpp/ggml"
    command = [str(nvcc), str(ROOT / "native_mmvq_probe.cu"), "--shared", "-std=c++17", "-O3",
               f"-arch=sm_{args.arch}", "--ptxas-options=-v", "-o", str(args.build_dir / library_name()),
               *[f"-I{ggml / p}" for p in ("include", "src", "src/ggml-cuda")]]
    if os.name == "nt":
        command += ["-Xcompiler", "/Zc:preprocessor"]
    result = subprocess.run(command, env=env, cwd=args.build_dir, text=True, capture_output=True)
    (args.build_dir / "build.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    manifest = dict(command=command, source_sha256=digest(ROOT / "native_mmvq_probe.cu"),
                    header_sha256={p: digest(ggml / "src/ggml-cuda" / p) for p in ("common.cuh", "vecdotq.cuh", "convert.cuh")},
                    compiler=subprocess.check_output([str(nvcc), "--version"], env=env, text=True),
                    returncode=result.returncode)
    (args.build_dir / "build.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, source in (("LICENSE.llama.cpp", args.kernel_repo / "_vendor/llama.cpp/LICENSE"),
                         ("LICENSE.ggml", ggml / "LICENSE")):
        shutil.copyfile(source, args.build_dir / name)
    print(result.stdout + result.stderr, flush=True)
    result.check_returncode()
    cuobjdump = args.cuda_root / "bin" / ("cuobjdump.exe" if os.name == "nt" else "cuobjdump")
    resources = subprocess.check_output([str(cuobjdump), "--dump-resource-usage", str(args.build_dir / library_name())], text=True)
    (args.build_dir / "resources.txt").write_text(resources, encoding="utf-8")
    print(f"Built isolated probe: {args.build_dir / library_name()}")


def library_name():
    return "native_mmvq_probe.dll" if os.name == "nt" else "native_mmvq_probe.so"


def load_library(args):
    manifest = json.loads((args.build_dir / "build.json").read_text(encoding="utf-8"))
    if manifest["returncode"] != 0 or manifest["source_sha256"] != digest(ROOT / "native_mmvq_probe.cu"):
        raise RuntimeError("Probe source differs from its successful build; rebuild explicitly before running.")
    # Retain DLL-directory handles until every captured graph has been released.
    handles = []
    if os.name == "nt":
        for directory in (args.cuda_root / "bin", args.build_dir):
            if directory.is_dir():
                handles.append(os.add_dll_directory(str(directory)))
    lib = ctypes.CDLL(str(args.build_dir / library_name()))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.mmvq_probe_quantize.argtypes = [ptr, ptr, integer, integer, integer, ptr]
    lib.mmvq_probe_quantize.restype = integer
    lib.mmvq_probe_launch.argtypes = [ptr, ptr, ptr, integer, integer, integer, integer, integer, ptr]
    lib.mmvq_probe_launch.restype = integer
    return lib, handles


def check(code):
    if code:
        raise RuntimeError(f"Probe CUDA runtime error: {code}")


def run(args):
    import gc
    import numpy as np
    import torch
    import gguf
    import llamacpp_gguf_cuda as native

    if not torch.__version__.startswith("2.10.") or torch.version.hip is not None:
        raise RuntimeError(f"This experiment expects supported CUDA PyTorch 2.10, found {torch.__version__}")
    if args.flush_mib < 192:
        raise ValueError("Use at least 192 MiB cache flushing on the RTX 5090; default is 256 MiB.")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    lib, dll_handles = load_library(args)
    generator = torch.Generator(device=device).manual_seed(20260922)
    flush = torch.zeros(args.flush_mib * 2**20 // 4, device=device, dtype=torch.float32)
    reader = gguf.GGUFReader(str(args.checkpoint), mode="r")
    tensors = {t.name: t for t in reader.tensors}
    records = []
    metadata = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(device), native=str(native.__file__),
                    checkpoint=str(args.checkpoint), dll_sha256=digest(args.build_dir / library_name()),
                    source_sha256=digest(ROOT / "native_mmvq_probe.cu"), rounds=args.rounds, flush_mib=args.flush_mib,
                    timer="CUDA graph external timing events; flush occurs before start event; alternating variant order")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(dict(metadata=metadata, cases=records), indent=2), encoding="utf-8")

    for weight_name in args.weights:
        parts = [tensors[n] for n in weight_name.split("+")]
        qtype = parts[0].tensor_type.name
        shapes = [tuple(map(int, reversed(t.shape))) for t in parts]
        if qtype not in ("Q4_K", "Q6_K") or any(t.tensor_type.name != qtype for t in parts) or len({s[1] for s in shapes}) != 1:
            raise ValueError(f"Incompatible weight group: {weight_name}")
        m, k = sum(s[0] for s in shapes), shapes[0][1]
        if m % 4 or k % 256:
            raise ValueError(f"Probe requires M divisible by 4 and K divisible by 256: {(m, k)}")
        raw_cpu = np.concatenate([np.asarray(t.data).reshape(-1) for t in parts])
        raw = torch.from_numpy(raw_cpu).to(device)
        del raw_cpu
        qtype_id = int(parts[0].tensor_type)
        padded = ((k + 511) // 512) * 512
        q8 = torch.empty(3 * padded // 32 * 36, dtype=torch.uint8, device=device)
        output = torch.empty((3, m), dtype=torch.float32, device=device)
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            dtype_code = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[dtype]
            x = torch.randn((3, k), dtype=dtype, device=device, generator=generator)

            def quantize():
                stream = torch.cuda.current_stream(device).cuda_stream
                check(lib.mmvq_probe_quantize(x.data_ptr(), q8.data_ptr(), dtype_code, k, padded, stream))

            def launch(variant):
                stream = torch.cuda.current_stream(device).cuda_stream
                check(lib.mmvq_probe_launch(raw.data_ptr(), q8.data_ptr(), output.data_ptr(), qtype_id, variant, m, k, padded // 32, stream))

            def production():
                return native.linear(raw, qtype, (m, k), x, None, torch.float32)

            # Exact reference checks include changed input before any benchmark.
            for input_case in ("random", "zero", "scaled"):
                if input_case == "zero":
                    x.zero_()
                elif input_case == "scaled":
                    x.normal_(generator=generator).mul_(8)
                reference = production()
                quantize()
                for variant in VARIANTS:
                    launch(variant)
                    torch.testing.assert_close(output, reference, atol=0, rtol=0, msg=f"{weight_name}/{dtype_name}/{input_case}/{VARIANTS[variant]}")
            x.normal_(generator=generator)
            reference = production()
            quantize()
            graphs = {}
            # Every graph reuses the same output and Q8 buffers. Reference graph
            # owns its output, which is kept alive until graph teardown.
            for scope in ("kernel", "full"):
                for variant, name in VARIANTS.items():
                    quantize()
                    launch(variant)
                    start = torch.cuda.Event(enable_timing=True, external=True)
                    end = torch.cuda.Event(enable_timing=True, external=True)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        flush.add_(1)
                        if scope == "kernel":
                            quantize()
                        start.record()
                        if scope == "full":
                            quantize()
                        launch(variant)
                        end.record()
                    graphs[f"{scope}/{name}"] = (graph, start, end)
            production()
            start = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                flush.add_(1)
                start.record()
                production_output = production()
                end.record()
            graphs["full/production"] = (graph, start, end)
            for name, (graph, _, _) in graphs.items():
                graph.replay()
                torch.testing.assert_close(production_output if name == "full/production" else output, reference, atol=0, rtol=0)
            samples = {name: [] for name in graphs}
            names = list(graphs)
            for round_index in range(args.rounds + 3):
                for name in names if round_index % 2 == 0 else list(reversed(names)):
                    graph, start, end = graphs[name]
                    graph.replay()
                    end.synchronize()
                    if round_index >= 3:
                        samples[name].append(start.elapsed_time(end) * 1000)
            medians = {name: statistics.median(values) for name, values in samples.items()}
            ratios = {name: statistics.median([a / b for a, b in zip(samples[f"{name.split('/')[0]}/row2_serial"], values)]) for name, values in samples.items() if name != "full/production"}
            record = dict(weight=weight_name, qtype=qtype, shape=[m, k], tokens=3, dtype=dtype_name,
                          exact_inputs=["random", "zero", "scaled", "captured_random"], us=medians,
                          paired_speedup_vs_row2_serial=ratios,
                          control_vs_production=medians["full/production"] / medians["full/row2_serial"], samples_us=samples)
            records.append(record)
            save()
            print(json.dumps({k: v for k, v in record.items() if k != "samples_us"}), flush=True)
            # All graph references, including the loop's last graph, must die
            # before tensors change storage or another weight is allocated.
            del graph, graphs, production_output, reference, x
            gc.collect()
            torch.cuda.synchronize(device)
        del raw, q8, output
        gc.collect()
        torch.cuda.synchronize(device)
    del flush
    torch.cuda.synchronize(device)
    del lib
    for handle in dll_handles:
        handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--kernel-repo", type=Path, default=Path("E:/ML/kernels/llama.cpp"))
    parser.add_argument("--cuda-root", type=Path, default=Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")))
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--arch", default="120")
    parser.add_argument("--checkpoint", type=Path, default=Path("E:/ML/Wan2GP/ckpts/Qwen3_8_27B_Uncensored/Qwen3.8-27B-Uncensored-Q4_K_M.gguf"))
    parser.add_argument("--weights", nargs="+", default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--dtypes", nargs="+", choices=("bfloat16", "float16", "float32"), default=["bfloat16"])
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUILD / "results.json")
    args = parser.parse_args()
    if args.rounds < 5:
        parser.error("Use at least five paired timing rounds.")
    build(args) if args.build else run(args)


if __name__ == "__main__":
    main()
