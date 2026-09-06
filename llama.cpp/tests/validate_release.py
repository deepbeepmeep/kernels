"""Numerical and graph-reuse checks for an installed release wheel (CUDA required)."""
import argparse
import json
from pathlib import Path
import sys

import gguf
import numpy as np
import torch
import llamacpp_gguf_cuda as kernels


QTYPES = ('Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'IQ2_S', 'IQ3_S', 'IQ1_S', 'IQ2_XS', 'IQ2_XXS', 'IQ3_XXS', 'IQ4_NL', 'IQ4_XS')


def packed_fixture(name, rows=128, columns=512):
    qtype = gguf.GGMLQuantizationType[name]
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    raw = np.random.default_rng(321).integers(0, 256, size=(rows * columns // block_size, type_size), dtype=np.uint8)
    offsets = [-4, -2] if name == 'Q2_K' else [-2] if name in ('Q3_K', 'Q6_K') else [0, 2] if name in ('Q4_K', 'Q5_K', 'Q4_1', 'Q5_1') else [0]
    for offset in offsets:
        offset %= type_size
        raw[:, offset:offset + 2] = np.frombuffer(np.float16(.025).tobytes(), dtype=np.uint8)
    dense = gguf.quants.dequantize(raw, qtype).reshape(rows, columns)
    return raw, torch.from_numpy(dense)


@torch.inference_mode()
def check_linear(name, packed, dense, cases):
    raw = torch.from_numpy(np.array(packed, copy=True)).flatten().cuda()
    reference_weight = dense.cuda().float()
    shape = tuple(dense.shape)
    for dtype in (torch.float16, torch.bfloat16):
        for rows in (1, 2, 4, 8, 17, 64, 129):
            inputs = torch.randn(rows, shape[1], device='cuda', dtype=dtype) * .1
            def run():
                return kernels.linear(raw, name, shape, inputs, None, dtype)
            for _ in range(3):
                actual = run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = run()
            for _ in range(2):
                inputs.normal_(std=.1)
                graph.replay()
                expected = inputs.float() @ reference_weight.T
                relative_rms = ((replayed.float() - expected).norm() / expected.norm().clamp_min(1e-20)).item()
                if not torch.isfinite(replayed).all() or relative_rms >= .025:
                    raise AssertionError(f'{name} {dtype} rows={rows}: relative RMS={relative_rms}')
                torch.testing.assert_close(replayed, run(), atol=0, rtol=0)
            cases.append({'qtype': name, 'dtype': str(dtype), 'rows': rows, 'relative_rms': relative_rms})
        if kernels.supports_embedding_qtype_name(name):
            ids = torch.tensor([127, 0, 17, 17], device='cuda')
            actual = kernels.embedding(raw, name, shape, ids, dtype)
            torch.testing.assert_close(actual, reference_weight[ids].to(dtype), atol=0, rtol=0)
    print(f'passed nonzero linear/graph checks: {name} {shape}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = []
    for name in QTYPES:
        raw, dense = packed_fixture(name)
        check_linear(name, raw, dense, cases)
    if args.checkpoint:
        reader = gguf.GGUFReader(str(args.checkpoint), mode='r')
        for name in ('Q4_K', 'Q6_K'):
            tensor = next(t for t in reader.tensors if t.tensor_type.name == name and len(t.shape) == 2 and min(t.shape) >= 128)
            columns = int(tensor.shape[0])
            block, size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            raw = np.array(tensor.data.reshape(-1)[:128 * columns // block * size], copy=True)
            dense = torch.from_numpy(gguf.quants.dequantize(raw, tensor.tensor_type).reshape(128, columns))
            check_linear(name, raw, dense, cases)
    import test_q8_paged_attention as q8
    import test_dense_paged_attention as dense_attention
    import test_mmq_activation_padding as padding
    for dtype in (torch.float16, torch.bfloat16):
        for dim in (128, 256):
            q8.run_case(dtype, dim)
        q8.run_speculative_case(dtype)
        for queries in (1, 2, 3):
            dense_attention.run_case(dtype, queries)
    for rows in (1, 2, 3):
        padding.run_case(rows)
    import test_sm120_compiled
    sm120_cases = test_sm120_compiled.run_cases() if torch.cuda.get_device_capability() == (12, 0) else 0
    torch.cuda.synchronize()
    report = {'version': kernels.__version__, 'package': kernels.__file__, 'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(), 'linear_cases': cases, 'attention_cases': 12, 'sm120_cases': sm120_cases, 'padding_cases': 3, 'checkpoint': str(args.checkpoint) if args.checkpoint else None, 'passed': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'PASS {kernels.__version__}: {len(cases)} linear configurations, 12 attention cases, 3 padding cases.', flush=True)


if __name__ == '__main__':
    main()
