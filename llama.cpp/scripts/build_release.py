"""Build a complete, toolkit-wide wheel using the current Python environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--target', choices=('py310', 'py311'), required=True)
parser.add_argument('--cuda-home', type=Path, required=True)
parser.add_argument('--work-dir', type=Path, required=True)
parser.add_argument('--dist-dir', type=Path, required=True)
parser.add_argument('--max-jobs', type=int, default=1)
parser.add_argument('--nvcc-threads', type=int, default=2)
args = parser.parse_args()

import torch

expected = {'py310': ((3, 10), '2.7.1', '12.8', '+torch271cu128py310'), 'py311': ((3, 11), '2.10.0', '13.0', '+torch210cu130py311')}[args.target]
if sys.version_info[:2] != expected[0] or torch.__version__.split('+')[0] != expected[1] or torch.version.cuda != expected[2]:
    raise SystemExit(f'Wrong environment: Python {sys.version_info[:2]}, torch {torch.__version__}, CUDA {torch.version.cuda}; expected {expected[:3]}')
cuda_home = args.cuda_home.resolve()
nvcc = cuda_home / 'bin' / ('nvcc.exe' if os.name == 'nt' else 'nvcc')
codes = subprocess.check_output([str(nvcc), '--list-gpu-code'], text=True).split()
compiler = subprocess.check_output([str(nvcc), '--version'], text=True)
if f'release {expected[2].split(".")[0]}.' not in compiler:
    raise SystemExit('CUDA toolkit major version must match the PyTorch runtime.')
work = args.work_dir.resolve()
dist = args.dist_dir.resolve()
if work == ROOT or ROOT in work.parents or work in ROOT.parents:
    raise SystemExit('Use a separate build directory outside the source tree.')
work.mkdir(parents=True, exist_ok=True)
dist.mkdir(parents=True, exist_ok=True)
source = work / 'source'
source.mkdir(exist_ok=True)
changed_sources = [p.relative_to(ROOT) for directory in ('csrc', '_vendor') for p in (ROOT / directory).rglob('*') if p.is_file() and (source / p.relative_to(ROOT)).is_file() and p.read_bytes() != (source / p.relative_to(ROOT)).read_bytes()]
for name in ('setup.py', 'pyproject.toml', 'MANIFEST.in', 'README.md', 'THIRD_PARTY_NOTICES.md'):
    shutil.copy2(ROOT / name, source / name)
for name in ('csrc', '_vendor', 'src', 'scripts', 'tests', 'wangp'):
    shutil.copytree(ROOT / name, source / name, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyd', '*.so', '*.egg-info', '.git'))

# A changed source can predate the last build's object (e.g. checkout/copy2).
# Force Ninja to rebuild changed inputs, while retaining unchanged fatbin objects.
for relative in changed_sources:
    os.utime(source / relative, None)

environment = os.environ.copy()
environment.pop('TORCH_CUDA_ARCH_LIST', None)
environment.pop('LLAMACPP_GGUF_CUDA_BUILD_COMPONENTS', None)
environment.update(CUDA_HOME=str(cuda_home), CUDA_PATH=str(cuda_home), MAX_JOBS=str(args.max_jobs), LLAMACPP_GGUF_CUDA_NVCC_THREADS=str(args.nvcc_threads), LLAMACPP_GGUF_CUDA_VERSION_SUFFIX=expected[3], PYTHONUNBUFFERED='1')
environment['PATH'] = os.pathsep.join([str(cuda_home / 'bin'), str(Path(sys.executable).parent), environment.get('PATH', '')])
if os.name != 'nt':
    include_dirs = [p for p in (cuda_home / 'include', cuda_home / 'include/cccl') if p.is_dir()]
    library_dirs = [p for p in (cuda_home / 'lib64', cuda_home / 'lib', cuda_home / 'lib64/stubs', cuda_home / 'lib/stubs', Path('/usr/lib/wsl/lib')) if p.is_dir()]
    for packages in site.getsitepackages():
        nvidia = Path(packages) / 'nvidia'
        include_dirs.extend(sorted(nvidia.glob('*/include')))
        library_dirs.extend(sorted(nvidia.glob('*/lib')))
    shims = work / 'linker'
    shims.mkdir(exist_ok=True)
    for directory in library_dirs:
        for library in sorted(directory.glob('lib*.so.*')):
            alias = shims / (library.name.split('.so.')[0] + '.so')
            if not alias.exists():
                alias.symlink_to(library)
    library_dirs.insert(0, shims)
    for name in ('C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'LLAMACPP_GGUF_CUDA_INCLUDE_DIRS'):
        environment[name] = os.pathsep.join(map(str, include_dirs))
    for name in ('LIBRARY_PATH', 'LD_LIBRARY_PATH', 'LLAMACPP_GGUF_CUDA_LIB_DIRS'):
        environment[name] = os.pathsep.join(map(str, library_dirs))
    for prefix in (Path(sys.executable).resolve().parents[1], cuda_home):
        if (prefix / 'nvvm/bin/cicc').is_file():
            environment['CICC_PATH'] = str(prefix / 'nvvm/bin')
            break
report = {'target': args.target, 'python': sys.version, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'cuda_home': str(cuda_home), 'compiler': compiler, 'architectures': codes, 'max_jobs': args.max_jobs, 'nvcc_threads': args.nvcc_threads, 'source_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file() and p.suffix in ('.py', '.cu', '.cpp', '.cuh', '.h')}}
(work / 'build_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
command = [sys.executable, '-m', 'pip', 'wheel', '.', '--no-build-isolation', '--no-deps', '--no-cache-dir', '-v', '-w', str(dist)]
print(f'Building {args.target} for {codes}; log: {work / "build.log"}', flush=True)
started = time.time()
with (work / 'build.log').open('w', encoding='utf-8') as log:
    result = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
report.update(exit_code=result.returncode, elapsed_seconds=time.time() - started)
if result.returncode == 0:
    wheels = sorted(dist.glob(f'*{expected[3]}*.whl'))
    report['wheels'] = []
    for wheel in wheels:
        digest = hashlib.sha256()
        with wheel.open('rb') as reader:
            for chunk in iter(lambda: reader.read(1024 * 1024), b''):
                digest.update(chunk)
        report['wheels'].append({'name': wheel.name, 'size': wheel.stat().st_size, 'sha256': digest.hexdigest()})
(work / 'build_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: report[k] for k in ('target', 'exit_code', 'elapsed_seconds')}), flush=True)
raise SystemExit(result.returncode)
