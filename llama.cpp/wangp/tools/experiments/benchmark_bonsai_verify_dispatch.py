"""Probe existing PTQ1 short-batch launches using real checkpoint matrices.

No model defaults or weights change. Compare complete calls, including chunk
assembly, under CUDA graphs; report exact output equality before timings.
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
from shared.qtypes.gguf import _gguf_get_index, _gguf_open_tensor_numpy, PrismQuantizationType


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    args = parser.parse_args()
    torch.manual_seed(817)
    index = _gguf_get_index(args.checkpoint)
    mapped = np.memmap(args.checkpoint, mode="r")
    unique = {}
    for info in index.tensor_infos:
        if info.tensor_type == PrismQuantizationType.PTQ1_0 and not info.name.startswith("token_embd"):
            unique.setdefault(tuple(reversed(info.raw_shape)), info)
    records = []
    for shape, info in unique.items():
        raw = torch.from_numpy(_gguf_open_tensor_numpy(mapped, index, info).copy()).to("cuda")
        for rows in args.rows:
            x = torch.randn((rows, shape[-1]), dtype=torch.bfloat16, device="cuda")
            def call(chunk):
                if not chunk:
                    return native.linear(raw, "PTQ1_0", shape, x, None, x.dtype)
                return torch.cat([native.linear(raw, "PTQ1_0", shape, part, None, x.dtype) for part in x.split(chunk)])
            expected = call(0)
            graphs, outputs, timings, equal = {}, {}, {}, {}
            try:
                for chunk in (0, 1, 2, 3):
                    result = call(chunk)
                    difference = (result.float() - expected.float()).abs()
                    equal[chunk] = dict(exact=bool(torch.equal(result, expected)), different=int(torch.count_nonzero(difference)), max_abs=float(difference.max()))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(8):
                            outputs[chunk] = call(chunk)
                    graphs[chunk] = graph
                    graph.replay()
                    timings[chunk] = []
                for repeat in range(8):
                    order = (0, 1, 2, 3) if repeat % 2 == 0 else (3, 2, 1, 0)
                    for chunk in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(4):
                            graphs[chunk].replay()
                        end.record()
                        end.synchronize()
                        timings[chunk].append(start.elapsed_time(end) * 1000 / 32)
                record = dict(weight=info.name, shape=shape, rows=rows, equality=equal,
                              median_us={k:statistics.median(v) for k,v in timings.items()}, samples_us=timings)
                records.append(record)
                print(json.dumps({k:v for k,v in record.items() if k != "samples_us"}), flush=True)
                args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
            finally:
                torch.cuda.synchronize()
                outputs.clear()
                for graph in graphs.values():
                    graph.reset()
                graphs.clear()
        del raw


if __name__ == "__main__":
    main()
