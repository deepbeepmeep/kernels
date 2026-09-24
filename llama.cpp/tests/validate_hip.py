"""Run on an AMD GPU after installing the HIP wheel; compilation is not validation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import llamacpp_gguf_cuda as kernels


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError('This validation requires ROCm PyTorch and an accessible AMD GPU.')
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING='1', HIP_LAUNCH_BLOCKING='1')
    tests = Path(__file__).resolve().parent
    commands = [
        ['validate_release.py', '--output', str(args.output / 'standard-formats.json')],
        ['test_ptq1.py', '--output', str(args.output / 'ptq1.json')],
        ['test_prism_hadamard.py'],
    ]
    for command in commands:
        with (args.output / (Path(command[0]).stem + '.log')).open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, str(tests / command[0]), *command[1:]], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Check the new fused module against the independent CPU ternary decoder.
    from test_ptq1 import fixture
    from test_prism_hadamard import reference
    import numpy as np
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for width in (1024, 5120, 6144):
            rows = 130
            raw, dense = fixture(rows, width)
            raw = torch.from_numpy(raw).to(device='cuda')
            x_cpu = torch.randn((1, width), dtype=dtype, device='cpu') * .1
            signs_cpu = torch.randint(0, 2, (width,), dtype=torch.int8, device='cpu') * 2 - 1
            x, signs = x_cpu.to(device='cuda'), signs_cpu.to(device='cuda')
            group = (16, 3, 128) if width == 6144 else (0, 0, 0)
            for tile in (1, 2, 4):
                def run():
                    return kernels.prism_decode(x, raw, signs, None, rows, group, tile)
                run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = run()
                for _ in range(2):
                    x_cpu.normal_(std=.1)
                    x.copy_(x_cpu)
                    graph.replay()
                    expected = reference(x_cpu, signs_cpu, False, group).float() @ torch.from_numpy(np.asarray(dense)).T
                    actual = output.float().cpu()
                    relative = (actual - expected).norm() / expected.norm().clamp_min(1e-20)
                    assert torch.isfinite(actual).all() and relative < .025, (dtype, width, tile, relative)
    report = dict(passed=True, torch=torch.__version__, hip=torch.version.hip, gpu=torch.cuda.get_device_name(), kernels=kernels.__version__, scope='Native kernel numerical and graph tests; full Deepy checkpoints must also be tested.')
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
