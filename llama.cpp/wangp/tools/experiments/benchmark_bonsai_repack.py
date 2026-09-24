"""Measure a lossless PTQ1-to-Q8 weight representation, without app changes.

The signed ternary integers and FP16 scales are copied exactly. CUDA reduction
order can differ, so report numerical differences as well as speed/memory.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import llamacpp_gguf_cuda as native
from shared.qtypes.gguf import _gguf_get_index, _gguf_open_tensor_numpy, PrismQuantizationType, _gguf_dequantize_tensor, gguf


def repack_q8(raw):
    source = np.asarray(raw).reshape(-1, 28)
    out = np.empty((len(source), 4, 34), dtype=np.uint8)
    for start in range(0, len(source), 16384):
        blocks = source[start:start + 16384]
        pieces = []
        for offset, width, count in ((0, 16, 5), (16, 8, 5), (24, 2, 4)):
            packed = blocks[:, offset:offset + width].astype(np.uint16)
            for n in range(count):
                pieces.append(((((packed * 3**n) & 255) * 3 >> 8).astype(np.int8) - 1))
        trits = np.concatenate(pieces, axis=1)
        result = out[start:start + len(blocks)]
        result[:, :, :2] = blocks[:, None, 26:28]
        result[:, :, 2:] = trits.reshape(-1, 4, 32).view(np.uint8)
    return out.reshape(raw.shape[0], -1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(817)
    index = _gguf_get_index(args.checkpoint)
    mapped = np.memmap(args.checkpoint, mode="r")
    unique = {}
    for info in index.tensor_infos:
        if info.tensor_type == PrismQuantizationType.PTQ1_0 and info.name.startswith("blk."):
            unique.setdefault(tuple(reversed(info.raw_shape)), info)
    records = []
    for shape, info in unique.items():
        source = _gguf_open_tensor_numpy(mapped, index, info)
        expanded = repack_q8(source)
        # Exhaustively verify each integer/scale representation through the
        # independent existing dequantizers, in bounded CPU slices.
        for start in range(0, shape[0], 64):
            old = _gguf_dequantize_tensor(torch.from_numpy(source[start:start+64].copy()), PrismQuantizationType.PTQ1_0, (min(64,shape[0]-start),shape[1]), torch.float32)
            new = _gguf_dequantize_tensor(torch.from_numpy(expanded[start:start+64]), gguf.GGMLQuantizationType.Q8_0, tuple(old.shape), torch.float32)
            torch.testing.assert_close(new, old, rtol=0, atol=0)
        weights = {"PTQ1_0":torch.from_numpy(source.copy()).to("cuda"), "Q8_0":torch.from_numpy(expanded).to("cuda")}
        for rows in (1, 3, 6, 8):
            x = torch.randn((rows, shape[-1]), dtype=torch.bfloat16, device="cuda")
            def call(kind):
                return native.linear(weights[kind], kind, shape, x, None, x.dtype)
            expected = call("PTQ1_0")
            actual = call("Q8_0")
            diff = actual.float() - expected.float()
            record = dict(weight=info.name,shape=shape,rows=rows,weights_exact=True,output_exact=bool(torch.equal(actual,expected)),
                          relative_rms=float(diff.square().mean().sqrt()/expected.float().square().mean().sqrt()),
                          different=int(torch.count_nonzero(diff)),max_abs=float(diff.abs().max()),
                          weight_bytes={k:v.numel() for k,v in weights.items()})
            graphs, outputs = {}, {}
            try:
                for kind in weights:
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(8):
                            outputs[kind]=call(kind)
                    graphs[kind]=graph
                    graph.replay()
                timings={k:[] for k in weights}
                for repeat in range(8):
                    for kind in (list(weights) if repeat%2==0 else list(reversed(weights))):
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(4):graphs[kind].replay()
                        end.record();end.synchronize()
                        timings[kind].append(start.elapsed_time(end)*1000/32)
                record.update(median_us={k:statistics.median(v) for k,v in timings.items()},samples_us=timings)
                records.append(record)
                args.output.write_text(json.dumps(records,indent=2),encoding="utf-8")
                print(json.dumps({k:v for k,v in record.items() if k!='samples_us'}),flush=True)
            finally:
                torch.cuda.synchronize()
                outputs.clear()
                for graph in graphs.values():graph.reset()
                graphs.clear()


if __name__ == "__main__":
    main()
