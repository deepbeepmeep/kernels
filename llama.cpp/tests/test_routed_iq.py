"""Independent gguf-py reference for native CUDA/HIP routed inference.

Run with pytest on the target GPU. Finite random blocks encode distinct expert weights.
Set GGUF_ROUTED_TIMINGS=1 for CUDA-event comparisons at prompt sizes.
"""
import os
import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")
from llamacpp_gguf_cuda import routed_iq as iq


def quantized(rows, columns, kind, rng):
    qtype = gguf.GGMLQuantizationType(kind) if kind != 42 else None
    width, size = gguf.GGML_QUANT_SIZES[qtype] if qtype is not None else (64, 18)
    blocks = rng.integers(0, 256, size=(rows * columns // width, size), dtype=np.uint8)
    # Scales are small finite halves. IQ1_M distributes the half across four u16 fields.
    scale = (0.006 * rng.uniform(.4, 1.2, len(blocks))).astype(np.float16)
    if kind == 29:
        half = scale.view(np.uint16)
        for index in range(4):
            byte = size - 7 + 2 * index
            blocks[:, byte] = (blocks[:, byte] & 15) | (((half >> (4 * index)) & 15) << 4)
    else:
        blocks[:, :2] = scale.view(np.uint8).reshape(-1, 2)
    if kind == 42:
        values = ((blocks[:, 2:, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3).astype(np.float32) - 1
        dense = values.reshape(-1, 64) * scale[:, None]
    else:
        dense = gguf.quants.dequantize(blocks, qtype)
    return torch.from_numpy(blocks.reshape(rows, -1)).cuda(), torch.from_numpy(dense.reshape(rows, columns)).cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an NVIDIA or AMD GPU")
@pytest.mark.parametrize("gate,down", [(19,42), (29,42), (16,42), (16,20), (17,20), (22,20), (18,20), (22,42)])
def test_routed(gate, down):
    rng = np.random.default_rng(97)
    torch.manual_seed(97)
    experts, hidden, intermediate = 3, 512, 256
    weights, dense = [], []
    for _ in range(experts):
        gu, a = quantized(2 * intermediate, hidden, gate, rng)
        d, b = quantized(hidden, intermediate, down, rng)
        weights.append((*gu.chunk(2, dim=0), d))
        dense.append((*a.chunk(2, dim=0), b))
    pointers = torch.tensor([[w.data_ptr() for w in bank] for bank in weights] + [[0,0,0]], dtype=torch.int64, device="cuda")
    types, caps = (gate, gate, down), (4., 7.)
    def direct():
        return iq.forward(x, ids, prob, pointers, intermediate, types, caps)
    for tokens in (1, 5, 65):
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        ids = torch.tensor(rng.integers(0, experts+1, (tokens,2)), device="cuda", dtype=torch.int32)
        prob = torch.softmax(torch.randn(tokens,2,device="cuda"),dim=-1)
        ref = torch.zeros_like(x, dtype=torch.float32)
        for expert, (w, u, d) in enumerate(dense):
            a, b = x.float() @ w.T, x.float() @ u.T
            act = caps[0] * (a / caps[0]).tanh() * a.sigmoid() * (caps[1] * (b / caps[1]).tanh())
            ref += ((act @ d.T) * ((ids == expert)*prob).sum(1)[:,None])
        single = direct()
        def check(out):
            error = (out-ref).norm() / ref.norm().clamp_min(1.e-10)
            assert torch.isfinite(out).all() and error < .045, (gate,down,tokens,error.item())
        check(single)
        job = iq.GroupedPass(x,ids.cpu(),prob,intermediate,types,caps,[2,0,3,1])
        job.run(pointers,0,2)
        job.run(pointers,2,4)
        check(job.finish())
        if tokens == 5:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                recorded = direct()
            x.mul_(.5)
            graph.replay()
            torch.testing.assert_close(recorded, direct(), rtol=0, atol=0)
        if tokens == 65 and os.environ.get("GGUF_ROUTED_TIMINGS"):
            def run():
                job.run(pointers,0,4)
                return job.finish()
            def timing():
                for _ in range(5): run()
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(30): run()
                b.record();b.synchronize()
                return a.elapsed_time(b)/30
            tiled=timing()
            job.tiles=None
            job.x=iq._pack(x,job.stream)
            print(gate,down,"grouped matrix ms",tiled,"grouped dot ms",timing(),flush=True)
