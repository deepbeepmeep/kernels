"""HIP build of the same modified GGML and custom kernels as the CUDA wheel."""
import os
from pathlib import Path
import shutil
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py


def build_hip():
    import torch
    if torch.__version__.split('+', 1)[0] != '2.10.0' or torch.version.hip is None:
        raise RuntimeError('The supported HIP wheel requires ROCm PyTorch 2.10.0; got ' + torch.__version__)
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('The supported HIP wheel requires Python 3.11.')
    from torch.utils.cpp_extension import BuildExtension, include_paths, library_paths
    from torch.utils.hipify import hipify_python

    root = Path(__file__).resolve().parent
    rocm_tag = ''.join(torch.version.hip.split('.')[:2])
    build_tag = 'torch210rocm' + rocm_tag + 'py311'
    build_base = 'build/hip-' + build_tag
    stage = root / 'build' / ('hip-src-' + build_tag)
    previous = {p: (p.read_bytes(), p.stat().st_atime_ns, p.stat().st_mtime_ns)
                for p in (stage / 'csrc').glob('*') if p.is_file()}
    # Keep generated HIP sources separate from the maintained CUDA/HIP sources.
    def copy_changed(src, dst):
        if not Path(dst).exists() or Path(src).read_bytes() != Path(dst).read_bytes():
            shutil.copy2(src, dst)
        return dst

    for folder in ('csrc', '_vendor'):
        shutil.copytree(root / folder, stage / folder, dirs_exist_ok=True, copy_function=copy_changed)
    sources = [str(stage / 'csrc' / p.name) for p in (root / 'csrc').glob('*') if p.suffix in ('.cu', '.cpp', '.h') and p.name != 'sm120_bindings.cpp']
    converted = hipify_python.hipify(project_directory=str(stage), output_directory=str(stage), includes=[str(stage / 'csrc' / '*')], ignores=[str(stage / '_vendor' / '*')], extra_files=sources, is_pytorch_extension=True, hipify_extra_files_only=True, show_detailed=False)
    # hipify rewrites some headers in place; unchanged generated headers should
    # not force every GGML template to be compiled again on the next build.
    for path, (content, atime, mtime) in previous.items():
        if path.exists() and path.read_bytes() == content:
            os.utime(path, ns=(atime, mtime))

    def source(name):
        path = str(stage / name)
        result = converted.get(path)
        return Path(result.hipified_path if result and result.hipified_path else path).relative_to(root).as_posix()

    ggml = stage / '_vendor' / 'llama.cpp' / 'ggml'
    includes = [str(stage), str(stage / 'csrc'), str(ggml / 'include'), str(ggml / 'src'), str(ggml / 'src' / 'ggml-cuda')] + include_paths(device_type='cuda')
    flags = ['-O3', '-std=c++17', '-DGGML_USE_HIP', '-DHIPBLAS_V2', '-U__HIP_NO_HALF_OPERATORS__', '-U__HIP_NO_HALF_CONVERSIONS__']
    modules = {
        '_prism': ['csrc/prism_decode_bindings.cpp', 'csrc/prism_decode.cu'],
        '_C': ['csrc/gguf_llamacpp_bindings.cpp', 'csrc/prism_hadamard.cu', 'csrc/gguf_llamacpp_kernels.cu'] + ['_vendor/llama.cpp/ggml/src/ggml-cuda/' + name + '.cu' for name in ('quantize', 'mmvq', 'convert')],
        '_attention': ['csrc/q8_paged_attention_bindings.cpp', 'csrc/q8_paged_attention.cu'],
    }
    extensions = [Extension('llamacpp_gguf_cuda.' + name, sources=[source(s) for s in files], include_dirs=includes, library_dirs=library_paths(device_type='cuda'), libraries=['c10', 'torch', 'torch_cpu', 'torch_python', 'amdhip64', 'c10_hip', 'torch_hip', 'hipblas'], extra_compile_args={'cxx': ['-O2', '-std=c++17', '-DGGML_USE_HIP'], 'nvcc': list(flags)}, language='c++') for name, files in modules.items()]
    suffix = os.environ.get('LLAMACPP_GGUF_CUDA_VERSION_SUFFIX', '+' + build_tag)
    version = '1.0.24' + suffix

    class HIPBuildPy(build_py):
        def find_package_modules(self, package, package_dir):
            return [entry for entry in super().find_package_modules(package, package_dir)
                    if not (package == 'llamacpp_gguf_cuda' and entry[1] == 'sm120')]

        def run(self):
            super().run()
            package = (Path(self.build_lib) / 'llamacpp_gguf_cuda').resolve()
            # Remove only obsolete generated NVIDIA artifacts from an incremental build.
            if package == (root / 'src' / 'llamacpp_gguf_cuda').resolve():
                raise RuntimeError('HIP build output must be separate from maintained sources.')
            for name in ('kernels', 'sm120.py'):
                artifact = package / name
                if artifact.exists():
                    if artifact.resolve().parent != package:
                        raise RuntimeError('Unexpected build artifact path: ' + str(artifact))
                    if artifact.is_dir():
                        shutil.rmtree(artifact)
                    else:
                        artifact.unlink()
            (package / 'version.py').write_text(
                f'__version__ = {version!r}\n__torch_version__ = {torch.__version__!r}\n'
                f'__hip_version__ = {torch.version.hip!r}\n', encoding='utf-8')

    (root / build_base).mkdir(parents=True, exist_ok=True)
    setup(version=version, description='GGUF HIP kernels including WanGP PTQ1, Prism and paged attention.', packages=find_packages(where='src'), package_dir={'': 'src'}, include_package_data=False, options={'build': {'build_base': build_base}, 'egg_info': {'egg_base': build_base}}, ext_modules=extensions, cmdclass={'build_ext': BuildExtension, 'build_py': HIPBuildPy}, license_files=['_vendor/llama.cpp/LICENSE', '_vendor/llama.cpp/ggml/LICENSE'], zip_safe=False)
