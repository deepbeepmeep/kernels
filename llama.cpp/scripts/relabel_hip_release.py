"""Relabel the published HIP wheel without changing executable code or binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from wheel.wheelfile import WheelFile

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0.26'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-wheel', type=Path, required=True)
    parser.add_argument('--dist', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((ROOT / 'release/1.0.25/win_hip_py311.json').read_text())
    if args.old_wheel.name != baseline['wheel'] or sha(args.old_wheel.read_bytes()) != baseline['sha256']:
        raise RuntimeError('Expected the checksum-verified published 1.0.25 HIP wheel')
    args.dist.mkdir(parents=True, exist_ok=True)
    target = args.dist / args.old_wheel.name.replace('1.0.25', VERSION, 1)
    unchanged = {}
    with zipfile.ZipFile(args.old_wheel) as old, WheelFile(target, 'w') as output:
        for name in old.namelist():
            if name.endswith('/') or name.endswith('.dist-info/RECORD'):
                continue
            data = old.read(name)
            renamed = name.replace('-1.0.25+', f'-{VERSION}+', 1) if '.dist-info/' in name else name
            if name.endswith('.dist-info/METADATA'):
                data = data.replace(b'\nVersion: 1.0.25+', f'\nVersion: {VERSION}+'.encode(), 1)
            elif name == 'llamacpp_gguf_cuda/version.py':
                data = data.replace(b'1.0.25+', f'{VERSION}+'.encode(), 1)
            else:
                unchanged[renamed] = sha(data)
            output.writestr(renamed, data)
    with WheelFile(target) as result:
        for name in result.namelist():
            data = result.read(name)  # Verify RECORD, including version changes.
            if name in unchanged and sha(data) != unchanged[name]:
                raise RuntimeError(f'Unintended HIP payload change: {name}')
    report = dict(wheel=target.name, bytes=target.stat().st_size, sha256=sha(target.read_bytes()),
        baseline_wheel=args.old_wheel.name, baseline_sha256=baseline['sha256'],
        torch=baseline['torch'], hip=baseline['hip'], architectures=baseline['architectures'],
        changed_payloads=['llamacpp_gguf_cuda/version.py: __version__ only', 'METADATA: Version only', 'dist-info directory and RECORD'],
        unchanged_payload_sha256=unchanged,
        hardware_validation='No AMD GPU available; version-only relabel, not a new HIP implementation.')
    (args.dist / 'hip_relabel_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
