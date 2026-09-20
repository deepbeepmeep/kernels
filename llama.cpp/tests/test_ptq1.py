"""PTQ1_0 tests using the independent CPU base-3 codec and CUDA graph replay."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import llamacpp_gguf_cuda as kernels


def decode(raw):
    """Prism CPU codec: uint8 wrap then multiply/high-byte, not the GPU SIMD routine."""
    blocks = np.asarray(raw, dtype=np.uint8).reshape(-1, 28)
    scale = blocks[:, 26:28].copy().view('<f2').astype(np.float32)
    pieces = []
    for offset, width, trits in ((0, 16, 5), (16, 8, 5), (24, 2, 4)):
        for n in range(trits):
            q = (blocks[:, offset:offset + width].astype(np.uint16) * 3 ** n) & 255
            pieces.append(((q * 3 >> 8).astype(np.int16) - 1).astype(np.float32) * scale)
    return np.concatenate(pieces, axis=1)


def fixture(rows, columns):
    rng = np.random.default_rng(1234)
    trits = rng.integers(0, 3, size=(rows * columns // 128, 128), dtype=np.uint16)
    raw = np.empty((len(trits), 28), dtype=np.uint8)
    for start, offset, width, count in ((0, 0, 16, 5), (80, 16, 8, 5), (120, 24, 2, 4)):
        q = np.zeros((len(trits), width), dtype=np.uint16)
        for n in range(count):
            q = q * 3 + trits[:, start + n * width:start + (n + 1) * width]
        if count == 4:
            q *= 3
        raw[:, offset:offset + width] = (q * 256 + 242) // 243
    scales = rng.uniform(.001, .09, size=len(trits)).astype('<f2')
    scales[::31] = 0
    raw[:, 26:28] = scales.view(np.uint8).reshape(-1, 2)
    dense = (trits.astype(np.float32) - 1) * scales.astype(np.float32)[:, None]
    np.testing.assert_array_equal(decode(raw), dense)
    return raw.flatten(), dense.reshape(rows, columns)


@torch.inference_mode()
def run_case(rows, columns, cases):
    packed, dense = fixture(rows, columns)
    raw = torch.from_numpy(packed).to(device='cuda')
    weight = torch.from_numpy(dense).to(device='cuda')
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        ids = torch.tensor([[rows - 1, 0, 17], [3, 17, 1]], device='cuda')
        torch.testing.assert_close(kernels.embedding(raw, 'PTQ1_0', (rows, columns), ids, dtype), weight[ids].to(dtype), rtol=0, atol=0)
        for mode in ('mmq', 'cublas'):
            os.environ['WGP_GGUF_LLAMACPP_CUDA_MATMUL_MODE'] = mode
            for batch in (1, 2, 3, 7, 8, 17, 64, 129):
                x = torch.randn(batch, columns, device='cuda', dtype=dtype) * .1
                bias = torch.randn(rows, device='cuda', dtype=dtype) * .01
                def run():
                    return kernels.linear(raw, 'PTQ1_0', (rows, columns), x, bias, dtype)
                for _ in range(3):
                    actual = run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed = run()
                for _ in range(2):
                    x.normal_(std=.1)
                    bias.normal_(std=.01)
                    graph.replay()
                    expected = x.float() @ weight.T + bias.float()
                    error = ((replayed.float() - expected).norm() / expected.norm().clamp_min(1e-20)).item()
                    assert torch.isfinite(replayed).all() and error < .025, (rows, columns, batch, dtype, mode, error)
                    torch.testing.assert_close(replayed, run(), rtol=0, atol=0)
                cases.append(dict(rows=rows, columns=columns, batch=batch, dtype=str(dtype), mode=mode, relative_rms=error))
        print(f'PASS PTQ1_0 {rows}x{columns} {dtype}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(912)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert kernels.supports_linear_qtype_name('PTQ1_0')
    assert kernels.supports_embedding_qtype_name('PTQ1_0')
    cases = []
    for shape in ((128, 512), (130, 512), (256, 5120), (130, 128), (128, 384)):
        run_case(*shape, cases)
    args.output.write_text(json.dumps(dict(passed=True, gpu=torch.cuda.get_device_name(), version=kernels.__version__, cases=cases), indent=2))
    print(f'PASS: {len(cases)} PTQ1_0 linear configurations and embeddings.')
