"""Validate a release wheel in isolation with the current Python/PyTorch stack."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--reference-wheel', type=Path, required=True)
    parser.add_argument('--reference-snapshot', type=Path, help='Reuse a snapshot already generated with this stack and reference wheel.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for wheel, folder in ((args.wheel, 'package'), (args.reference_wheel, 'reference')):
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(out / folder)
    records = []

    def run(name, script, arguments=(), reference=False):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(out / ('reference' if reference else 'package'))
        env['PYTHONUNBUFFERED'] = '1'
        command = [sys.executable, str(ROOT / 'tests' / script), *map(str, arguments)]
        started = time.monotonic()
        print('Validating', name, flush=True)
        with (out / f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        records.append(dict(name=name, exit_code=result.returncode, seconds=time.monotonic() - started))
        (out / 'validation.json').write_text(json.dumps(dict(wheel=args.wheel.name, python=sys.version, checks=records, passed=False), indent=2), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {out / (name + ".log")}')

    snapshot = args.reference_snapshot.resolve() if args.reference_snapshot else out / 'old_quant_reference.pt'
    if args.reference_snapshot is None:
        run('old_quant_reference', 'test_quant_compatibility.py', ['--save', snapshot], reference=True)
    run('old_quant_exact', 'test_quant_compatibility.py', ['--compare', snapshot])
    arguments = ['--output', out / 'standard.json']
    if args.checkpoint:
        arguments += ['--checkpoint', args.checkpoint.resolve()]
    run('standard', 'validate_release.py', arguments)
    run('ptq1', 'test_ptq1.py', ['--output', out / 'ptq1.json'])
    run('hadamard', 'test_prism_hadamard.py')
    run('prism_decode', 'test_prism_decode.py', ['--output', out / 'prism_decode.json'])
    run('linear_fusions', 'test_linear_fusions.py')
    report = dict(wheel=args.wheel.name, python=sys.version, checks=records, passed=True)
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('PASSED', args.wheel.name, flush=True)


if __name__ == '__main__':
    main()
