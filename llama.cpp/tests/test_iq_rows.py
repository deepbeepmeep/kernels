"""IQ3_S / IQ2_S multi-row kernels (_iq_rows) against an FP64 reference and llama.cpp's MMVQ: their error must not be larger, a row must
not depend on the other rows of its call, and a weight at any alignment or at the end of its buffer must be read correctly. Every
variant of the GPU is checked (DP4A from compute capability 6.1, the tensor-core ones from 8.0).
Usage: test_iq_rows.py [IQ3_S IQ2_S] [variants]"""
import sys

import numpy as np
import torch

import llamacpp_gguf_cuda as native
from llamacpp_gguf_cuda import _iq_rows as kernel

BLOCK = {"IQ3_S": 110, "IQ2_S": 82}
if not kernel.eligible(0):
    print("SKIPPED: no multi-row IQ kernel for this GPU or build")
    sys.exit(0)


def weights(qtype, n, k, generator, scale=(0.002, 0.02)):
    """Random valid blocks of an [n, k] weight (uint8 [bytes]) and its dense FP64 values."""
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    blocks = torch.randint(0, 256, (n * k // 256, BLOCK[qtype]), generator=generator, dtype=torch.uint8)
    blocks[:, :2] = torch.empty(blocks.shape[0], 1, dtype=torch.float16).uniform_(*scale, generator=generator).view(torch.uint8)
    dense = torch.from_numpy(dequantize(blocks.numpy(), GGMLQuantizationType[qtype]).astype(np.float64)).reshape(n, k)
    return blocks.reshape(-1), dense


def baseline(raw, qtype, shape, x, dtype, silu=False):
    """The installed package's kernel for these rows (MMVQ up to 8 rows)."""
    if silu or native.supports_linear_fusions(qtype, x.shape[0], x.device.index):
        return native._C.linear(raw, qtype, list(shape), x, None, str(dtype).replace("torch.", ""), native._linear_mode(), True, silu)
    return native._C.linear(raw, qtype, list(shape), x, None, str(dtype).replace("torch.", ""), native._linear_mode())


def error(result, reference):
    return float((result.double() - reference).norm() / reference.norm())


qtypes = [name for name in sys.argv[1:] if name.startswith("IQ")] or ["IQ3_S", "IQ2_S"]
variants = [name for name in sys.argv[1:] if not name.startswith("IQ")] or list(kernel.variants(0))
failures = 0


def check(name, condition, detail=""):
    global failures
    failures += not condition
    print(("ok   " if condition else "FAIL ") + name, detail, flush=True)


def close(result, expected, offset):
    """Bit-identical at even offsets; at odd ones the tensor-core variant hands the call to the other kernel (its 16-bit loads need even addresses): one rounding apart."""
    return torch.equal(result, expected) or offset % 2 == 1 and float((result.double() - expected.double()).norm() / expected.double().norm()) < 4e-3


def run(variant):
  global failures

  def linear(x, raw, qtype, shape, bias, dtype, silu):
      return kernel.iq_linear(x, raw, qtype, shape, bias, dtype, silu, variant)

  for qtype in qtypes:
    if True:
        for n, k in ((1056, 5120), (512, 17408), (1031, 5376), (37, 256)):  # 5376 / 256 = 21 blocks: odd rows start 14 bytes off a 16-byte boundary
            generator = torch.Generator().manual_seed(n + k)
            raw, dense = weights(qtype, n, k, generator)
            raw, dense = raw.cuda(), dense.cuda()
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                x = torch.randn(8, k, generator=generator).to("cuda", torch.bfloat16 if dtype == torch.float32 else dtype)
                x[3, : k // 2] = 0  # blocks of zeros
                x[5] *= 40
                reference = x.double() @ dense.T
                single = torch.cat([linear(x[i:i + 1], raw, qtype, [n, k], None, dtype, False) for i in range(8)])
                worst = (0.0, 0.0, 0)
                for rows in range(1, 9):
                    ours = linear(x[:rows], raw, qtype, [n, k], None, dtype, False)
                    theirs = baseline(raw, qtype, [n, k], x[:rows], dtype).to(dtype)
                    ours_error, theirs_error = error(ours, reference[:rows]), error(theirs, reference[:rows])
                    if not torch.isfinite(ours).all() or ours_error > theirs_error * 1.02 + 1e-7:
                        check(f"{qtype} [{n}, {k}] {dtype} rows {rows}: error against FP64", False, f"ours {ours_error:.3e} MMVQ {theirs_error:.3e}")
                    if not torch.equal(ours, single[:rows]):
                        check(f"{qtype} [{n}, {k}] {dtype} rows {rows}: rows independent of the call", False)
                    if ours_error >= worst[0]:
                        worst = (ours_error, theirs_error, rows)
                check(f"{qtype} [{n}, {k}] {str(dtype)[6:]}: rows 1-8", True, f"largest error {worst[0]:.3e} (MMVQ {worst[1]:.3e}) at {worst[2]} rows")
            # the same weight inside a larger buffer at every byte offset of a 16-byte boundary, and ending exactly at the buffer's end
            x = torch.randn(3, k, generator=generator).to("cuda", torch.bfloat16)
            expected = linear(x, raw, qtype, [n, k], None, torch.bfloat16, False)
            same = True
            for offset in (1, 2, 7, 8, 13, 15, 16):
                holder = torch.full((raw.numel() + offset,), 255, dtype=torch.uint8, device="cuda")
                holder[offset:] = raw
                same &= close(linear(x, holder[offset:], qtype, [n, k], None, torch.bfloat16, False), expected, offset)
                holder = torch.full((raw.numel() + offset + 64,), 255, dtype=torch.uint8, device="cuda")
                holder[offset:offset + raw.numel()] = raw
                same &= close(linear(x, holder[offset:offset + raw.numel()], qtype, [n, k], None, torch.bfloat16, False), expected, offset)
            check(f"{qtype} [{n}, {k}]: weight at any offset of a buffer", bool(same))
        # fused SwiGLU input (gate then value), against the separate operations stored in the activation type
        n, k = 512, 5120
        generator = torch.Generator().manual_seed(11)
        raw, dense = weights(qtype, n, k, generator)
        raw, dense = raw.cuda(), dense.cuda()
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn(5, 2 * k, generator=generator).to("cuda", dtype)
            gate, value = x[:, :k].float(), x[:, k:].float()
            activated = ((gate / (1 + torch.exp(-gate))).to(dtype).float() * value).to(dtype)
            reference = activated.double() @ dense.T
            for rows in (1, 2, 5):
                fused = linear(x[:rows], raw, qtype, [n, k], None, dtype, True)
                separate = linear(activated[:rows], raw, qtype, [n, k], None, dtype, False)
                theirs = baseline(raw, qtype, [n, k], x[:rows], dtype, silu=True)
                check(f"{qtype} SwiGLU {str(dtype)[6:]} rows {rows}", error(fused, reference[:rows]) <= error(theirs, reference[:rows]) * 1.02 + 1e-7,
                      f"ours {error(fused, reference[:rows]):.3e} MMVQ {error(theirs, reference[:rows]):.3e}; equal to the separate operations: {bool(torch.equal(fused, separate))}")
        # edge cases: zeros, one outlier per block, subnormal and huge activations
        x = torch.zeros(4, k, dtype=torch.bfloat16, device="cuda")
        x[1, ::32] = 1000.0
        x[2] = 1e-30
        x[3] = torch.randn(k, generator=generator).to("cuda") * 1e30
        reference = x.double() @ dense.T
        ours = linear(x, raw, qtype, [n, k], None, torch.float32, False)
        theirs = baseline(raw, qtype, [n, k], x, torch.float32)
        check(f"{qtype} zero row", bool((ours[0] == 0).all()))
        for row, label in ((1, "outliers"), (2, "tiny"), (3, "huge")):
            mmvq_error = error(theirs[row], reference[row])  # MMVQ keeps the activation scale in FP16: zero for tiny rows, infinite for huge ones
            check(f"{qtype} {label} row", bool(torch.isfinite(ours[row]).all()) and (error(ours[row], reference[row]) <= mmvq_error * 1.02 + 1e-7 or mmvq_error != mmvq_error),
                  f"ours {error(ours[row], reference[row]):.3e} MMVQ {error(theirs[row], reference[row]):.3e}")
with torch.inference_mode():
    for name in variants:
        print("==== variant", name, flush=True)
        run(name)
print("FAILURES:", failures)
sys.exit(1 if failures else 0)
