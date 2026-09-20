"""Snapshot existing qtypes before/after a kernel change, in separate processes."""
import argparse
import os
from pathlib import Path

import torch
import llamacpp_gguf_cuda as kernels
from validate_release import QTYPES, packed_fixture


@torch.inference_mode()
def snapshot():
    torch.manual_seed(457)
    results = {}
    for name in QTYPES:
        raw, dense = packed_fixture(name)
        raw = torch.from_numpy(raw).flatten().to(device='cuda')
        for mode in ('mmq', 'cublas'):
            os.environ['WGP_GGUF_LLAMACPP_CUDA_MATMUL_MODE'] = mode
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for rows in (1, 2, 3, 7, 8, 17, 64, 129):
                    x = torch.randn(rows, 512, device='cuda', dtype=dtype) * .1
                    bias = torch.randn(128, device='cuda', dtype=dtype) * .01
                    y = kernels.linear(raw, name, (128, 512), x, bias, dtype)
                    results[f'{name}/{mode}/{dtype}/{rows}'] = y.cpu()
        if kernels.supports_embedding_qtype_name(name):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                ids = torch.tensor([[127, 0, 17], [1, 17, 126]], device='cuda')
                results[f'{name}/embedding/{dtype}'] = kernels.embedding(raw, name, (128, 512), ids, dtype).cpu()
        print(f'Snapshotted {name}', flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    outputs = snapshot()
    if args.save:
        torch.save(outputs, args.save)
    if args.compare:
        baseline = torch.load(args.compare, map_location='cpu', weights_only=True)
        assert outputs.keys() == baseline.keys()
        for key, value in outputs.items():
            torch.testing.assert_close(value, baseline[key], rtol=0, atol=0, msg=key)
        print(f'PASS: {len(outputs)} existing-qtype outputs are bit-identical.')
