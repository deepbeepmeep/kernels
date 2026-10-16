"""Build a Python-only release from verified 1.0.25 CUDA wheels, without NVCC.

Native extensions and bundled GPU binaries are copied byte-for-byte. This tool
deliberately refuses native source changes; use build_release.py for those.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

from wheel.wheelfile import WheelFile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'd19749f'
VERSION = '1.0.26'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-wheels', type=Path, required=True)
    parser.add_argument('--dist', type=Path, required=True)
    args = parser.parse_args()
    # These are all native implementation/build inputs. setup.py only changes
    # package version, data and licenses in this Python-only release.
    native_paths = ['csrc', '_vendor', 'src/llamacpp_gguf_cuda/kernels']
    subprocess.run(['git', 'diff', '--exit-code', BASELINE, '--', *native_paths], cwd=ROOT, check=True)
    old_setup = subprocess.check_output(['git', 'show', f'{BASELINE}:llama.cpp/setup.py'], cwd=ROOT, text=True)
    expected_setup = old_setup.replace('BASE_VERSION = "1.0.25"', 'BASE_VERSION = "1.0.26"')
    # Packaging-only edits are explicitly allowed; compilation settings are not.
    def normalize_setup(text):
        return text.replace(', "LICENSE-WanGP.txt"', '').replace(', "_packed_turing.cu"', '')
    if normalize_setup((ROOT / 'setup.py').read_text()) != normalize_setup(expected_setup):
        raise RuntimeError('setup.py contains changes beyond this release packaging allowlist')
    old_hip = subprocess.check_output(['git', 'show', f'{BASELINE}:llama.cpp/hip_build.py'], cwd=ROOT, text=True)
    if (ROOT / 'hip_build.py').read_text() != old_hip.replace("version = '1.0.25'", "version = '1.0.26'"):
        raise RuntimeError('HIP build changes beyond the version number require a native build')
    baseline = json.loads((ROOT / 'release/1.0.25/summary.json').read_text())
    args.dist.mkdir(parents=True, exist_ok=True)
    report = {'release': f'gguf-v{VERSION}', 'build_method': 'repackage; no native compilation', 'baseline_commit': BASELINE, 'stacks': {}}
    for stack, prior in baseline['stacks'].items():
        if 'rocm' in prior['wheel']:
            continue
        source = args.old_wheels / prior['wheel']
        if digest(source.read_bytes()) != prior['sha256']:
            raise RuntimeError(f'Published baseline checksum mismatch: {source}')
        name = source.name.replace('1.0.25', VERSION, 1)
        target = args.dist / name
        binary_hashes = {}
        with zipfile.ZipFile(source) as old:
            old_info = next(p.split('/')[0] for p in old.namelist() if p.endswith('.dist-info/METADATA'))
            new_info = old_info.replace('1.0.25', VERSION, 1)
            files = {}
            for entry in old.namelist():
                if entry.endswith('/') or entry.endswith('.dist-info/RECORD'):
                    continue
                data = old.read(entry)
                path = entry.replace(old_info, new_info, 1)
                if path.endswith('/METADATA'):
                    header = data.decode().split('\n\n', 1)[0]
                    header = re.sub(r'(?m)^Version: 1\.0\.25', f'Version: {VERSION}', header)
                    if 'License-File: LICENSE-WanGP.txt' not in header:
                        header += '\nLicense-File: LICENSE-WanGP.txt'
                    data = (header + '\n\n' + (ROOT / 'README.md').read_text(encoding='utf-8')).encode()
                files[path] = data
                if entry.endswith(('.pyd', '.so', '.cubin', '.ptx')):
                    binary_hashes[entry] = digest(data)
            for path in (ROOT / 'src/llamacpp_gguf_cuda').rglob('*.py'):
                files[path.relative_to(ROOT / 'src').as_posix()] = path.read_bytes()
            for filename in ('_packed_turing.cu',):
                files[f'llamacpp_gguf_cuda/{filename}'] = (ROOT / 'src/llamacpp_gguf_cuda' / filename).read_bytes()
            suffix = name.split('-')[1]
            files['llamacpp_gguf_cuda/version.py'] = f'__version__ = "{suffix}"\n'.encode()
            files[f'{new_info}/licenses/LICENSE-WanGP.txt'] = (ROOT / 'LICENSE-WanGP.txt').read_bytes()
            with WheelFile(target, 'w') as output:
                for path, data in sorted(files.items()):
                    output.writestr(path, data)
        with WheelFile(target) as result:
            for path, sha in binary_hashes.items():
                if digest(result.read(path)) != sha:
                    raise RuntimeError(f'Native binary changed: {path}')
            # Reading every entry verifies the newly generated RECORD hashes.
            for path in result.namelist():
                result.read(path)
        report['stacks'][stack] = dict(wheel=name, bytes=target.stat().st_size, sha256=digest(target.read_bytes()),
            baseline_wheel=source.name, baseline_sha256=prior['sha256'], torch=prior['torch'], cuda=prior['cuda'],
            architectures=prior['architectures'], unchanged_binary_sha256=binary_hashes)
        print('Repackaged', name, flush=True)
        (args.dist / 'repackage_manifest.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
