"""Extract a wheel and verify its native SASS/PTX inventory before release."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--cuda-home', type=Path, required=True)
    parser.add_argument('--cuobjdump', type=Path, help='Inspector path when a minimal toolkit omits cuobjdump.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suffix = '.exe' if (args.cuda_home / 'bin/nvcc.exe').exists() else ''
    nvcc = args.cuda_home / ('bin/nvcc' + suffix)
    cuobjdump = args.cuobjdump or args.cuda_home / ('bin/cuobjdump' + suffix)
    expected = set(re.findall(r'sm_(\d+)', subprocess.check_output([str(nvcc), '--list-gpu-code'], text=True)))
    args.target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.wheel) as archive:
        archive.extractall(args.target)
    package = args.target / 'llamacpp_gguf_cuda'
    records = []
    for path in sorted(package.iterdir()):
        if path.suffix not in ('.so', '.pyd'):
            continue
        elf = subprocess.check_output([str(cuobjdump), '--list-elf', str(path)], text=True)
        ptx = subprocess.check_output([str(cuobjdump), '--list-ptx', str(path)], text=True)
        sass = set(re.findall(r'sm_(\d+)', elf))
        virtual = set(re.findall(r'sm_(\d+)', ptx))
        if sass != expected or str(max(map(int, expected))) not in virtual:
            raise ValueError(f'Incomplete architecture inventory: {path.name}: SASS={sass}, PTX={virtual}, expected={expected}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        record = dict(file=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest(),
                      sass=sorted(sass, key=int), ptx=sorted(virtual, key=int))
        if path.suffix == '.so':
            dynamic = subprocess.check_output(['readelf', '-d', str(path)], text=True)
            needed = re.findall(r'\(NEEDED\).*?\[(.*?)\]', dynamic)
            search_paths = re.findall(r'\((?:RUNPATH|RPATH)\).*?\[(.*?)\]', dynamic)
            major = re.search(r'release (\d+)\.', subprocess.check_output([str(nvcc), '--version'], text=True)).group(1)
            if f'libcudart.so.{major}' not in needed:
                raise ValueError(f'Wrong CUDA runtime dependency: {path.name}: {needed}')
            if any(library.startswith('libcublas.so.') and library != f'libcublas.so.{major}' for library in needed):
                raise ValueError(f'Wrong cuBLAS dependency: {path.name}: {needed}')
            if not search_paths or any(not entry.startswith('$ORIGIN/') for value in search_paths for entry in value.split(':')):
                raise ValueError(f'Non-relocatable runtime search path: {path.name}: {search_paths}')
            record.update(needed=needed, runtime_search_paths=search_paths)
        records.append(record)
    if len(records) != 3:
        raise ValueError('Expected GGUF, attention and Prism native extensions')
    for manifest in (package / 'kernels').glob('*/manifest.json'):
        for cubin in manifest.parent.glob('*.cubin'):
            if cubin.stat().st_size == 0:
                raise ValueError(f'Empty GPU program: {cubin}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
