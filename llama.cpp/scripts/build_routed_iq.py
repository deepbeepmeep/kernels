"""Compile the routed IQ device library using the CUDA or HIP toolkit of the wheel."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def build(compiler, output, archs):
    root = Path(__file__).resolve().parents[1]
    ggml = root / "_vendor/llama.cpp/ggml"
    includes = [ggml / "include", ggml / "src", ggml / "src/ggml-cuda", root / "csrc"]
    env = os.environ.copy()
    args = [str(compiler), "-std=c++17", "-O3", "-shared"]
    if "nvcc" in compiler.name.lower():
        codes = archs or [code.removeprefix("sm_") for code in subprocess.check_output([str(compiler), "--list-gpu-code"], text=True).split() if code.removeprefix("sm_").isdigit() and int(code[3:]) >= 61]
        codes = sorted(set(codes), key=int)
        if not codes:
            raise RuntimeError("The toolkit has no supported native integer-dot target")
        for sm in codes:
            args.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
        args.append(f"-gencode=arch=compute_{codes[-1]},code=compute_{codes[-1]}")
        args.extend(["--threads", "3", "--expt-relaxed-constexpr", "-Xcompiler", "/Zc:preprocessor" if os.name == "nt" else "-fPIC"])
    else:
        sdk = compiler.resolve().parents[1]
        env.update(HIP_PATH=str(sdk), ROCM_PATH=str(sdk), HIP_PLATFORM="amd")
        env["PATH"] = os.pathsep.join([str(sdk/"lib/llvm/bin"), str(sdk/"bin"), env["PATH"]])
        targets = archs or ["gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1200", "gfx1201"]
        args += ["-x", "hip", "-DGGML_USE_HIP", "-DHIPBLAS_V2", f"--rocm-path={sdk}",
                 f"--rocm-device-lib-path={sdk / 'lib/llvm/amdgcn/bitcode'}",
                 *[f"--offload-arch={a}" for a in targets]]
        if os.name != "nt":
            args.extend(["-fPIC", "-fvisibility=hidden"])
        args += shlex.split(os.environ.get("LLAMACPP_GGUF_HIP_EXTRA_FLAGS", ""))
    output = output.resolve()
    output.parent.mkdir(exist_ok=True, parents=True)
    args.extend([*["-I" + str(p) for p in includes], str(root / "csrc/routed_iq.cu"), "-o", str(output)])
    with tempfile.TemporaryDirectory() as cwd:
        subprocess.run(args, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arch", action="append")
    args = parser.parse_args()
    build(args.compiler, args.output, args.arch)
