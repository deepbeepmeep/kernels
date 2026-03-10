from pathlib import Path
import os
import subprocess
import sys

from setuptools import find_packages, setup


def _vc_tools_dir() -> str:
    base = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC")
    entries = [entry for entry in base.iterdir() if entry.is_dir()]
    if not entries:
        raise FileNotFoundError(f"Missing MSVC tools in {base}")
    return str(sorted(entries)[-1])


def _windows_sdk_dir() -> tuple[str, str]:
    base = Path(r"C:\Program Files (x86)\Windows Kits\10")
    lib_root = base / "Lib"
    versions = [entry.name for entry in lib_root.iterdir() if entry.is_dir()]
    if not versions:
        raise FileNotFoundError(f"Missing Windows SDK in {lib_root}")
    version = sorted(versions)[-1]
    return str(base), version


def _configure_windows_build_env() -> None:
    if os.name != "nt":
        return
    vc_tools = _vc_tools_dir()
    sdk_root, sdk_version = _windows_sdk_dir()
    python_root = os.path.dirname(sys.executable)
    python_scripts = os.path.join(python_root, "Scripts")
    host_bin = os.path.join(vc_tools, "bin", "Hostx64", "x64")
    sdk_bin = os.path.join(sdk_root, "bin", sdk_version, "x64")
    include_dirs = [
        os.path.join(vc_tools, "include"),
        os.path.join(sdk_root, "Include", sdk_version, "ucrt"),
        os.path.join(sdk_root, "Include", sdk_version, "shared"),
        os.path.join(sdk_root, "Include", sdk_version, "um"),
        os.path.join(sdk_root, "Include", sdk_version, "winrt"),
        os.path.join(sdk_root, "Include", sdk_version, "cppwinrt"),
    ]
    lib_dirs = [
        os.path.join(vc_tools, "lib", "x64"),
        os.path.join(sdk_root, "Lib", sdk_version, "ucrt", "x64"),
        os.path.join(sdk_root, "Lib", sdk_version, "um", "x64"),
    ]

    env = os.environ
    env["DISTUTILS_USE_SDK"] = "1"
    env["MSSdk"] = "1"
    env["VCToolsInstallDir"] = vc_tools + os.sep
    env["WindowsSdkDir"] = sdk_root + os.sep
    env["WindowsSDKVersion"] = sdk_version + os.sep
    cuda_root = env.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")
    env["CUDA_PATH"] = cuda_root
    env["CUDA_HOME"] = env.get("CUDA_HOME", cuda_root)
    env["PATH"] = os.pathsep.join([python_scripts, host_bin, sdk_bin, env.get("PATH", "")])
    env["INCLUDE"] = os.pathsep.join([path for path in include_dirs if os.path.isdir(path)])
    env["LIB"] = os.pathsep.join([path for path in lib_dirs if os.path.isdir(path)])


_configure_windows_build_env()

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CSRC = Path("csrc")
GGML = Path("_vendor") / "llama.cpp" / "ggml"
BASE_VERSION = "1.0.2"
VERSION_SUFFIX = os.environ.get("LLAMACPP_GGUF_CUDA_VERSION_SUFFIX", "").strip()
PACKAGE_VERSION = BASE_VERSION + VERSION_SUFFIX
PACKAGE_DESCRIPTION = os.environ.get("LLAMACPP_GGUF_CUDA_DESCRIPTION", "Reusable GGUF CUDA kernels built from llama.cpp CUDA code paths.")
BASE_NVCC_FLAGS = [
    "-std=c++17",
    "-O3",
    "-Xcudafe",
    "--diag_suppress=177",
    "-Xcudafe",
    "--diag_suppress=550",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]


def _fatbin_arch_flags() -> list[str]:
    env_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if env_list:
        return []
    cuda_home = Path(os.environ["CUDA_HOME"])
    nvcc = cuda_home / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    output = subprocess.check_output([str(nvcc), "--list-gpu-code"], text=True)
    codes = sorted({line.strip() for line in output.splitlines() if line.strip().startswith("sm_")}, key=lambda x: int(x.split("_", 1)[1]))
    if not codes:
        raise RuntimeError("nvcc did not report any GPU codes")
    flags = []
    for idx, code in enumerate(codes):
        num = code.split("_", 1)[1]
        flags.append(f"-gencode=arch=compute_{num},code=sm_{num}")
        if idx == len(codes) - 1:
            flags.append(f"-gencode=arch=compute_{num},code=compute_{num}")
    return flags


NVCC_FLAGS = BASE_NVCC_FLAGS + _fatbin_arch_flags()


def _write_version_file() -> None:
    version_file = ROOT / "src" / "llamacpp_gguf_cuda" / "version.py"
    version_file.write_text(f'__version__ = "{PACKAGE_VERSION}"\n', encoding="utf-8")


_write_version_file()

extra_compile_args = {
    "cxx": ["/std:c++17", "/EHsc", "/O2"] if os.name == "nt" else ["-std=c++17", "-O3"],
    "nvcc": NVCC_FLAGS,
}

ext_modules = [
    CUDAExtension(
        name="llamacpp_gguf_cuda._C",
        sources=[
            str(CSRC / "gguf_llamacpp_bindings.cpp"),
            str(CSRC / "gguf_llamacpp_kernels.cu"),
            str(GGML / "src" / "ggml-cuda" / "quantize.cu"),
            str(GGML / "src" / "ggml-cuda" / "mmvq.cu"),
            str(GGML / "src" / "ggml-cuda" / "convert.cu"),
        ],
        include_dirs=[
            str(ROOT),
            str(ROOT / GGML / "include"),
            str(ROOT / GGML / "src"),
            str(ROOT / GGML / "src" / "ggml-cuda"),
            str(ROOT / CSRC),
        ],
        extra_compile_args=extra_compile_args,
        libraries=["cublas"],
    )
]

setup(
    version=PACKAGE_VERSION,
    description=PACKAGE_DESCRIPTION,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
