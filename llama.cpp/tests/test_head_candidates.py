"""head_candidates: the kept rows are each group's best by the one-bit score, computed as the kernel's arithmetic says (activations of a
sign and three magnitude bits with a scale per 64, exact integer sums), for every input type, several rows of input and ties."""
import pytest
import torch

import llamacpp_gguf_cuda as kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not kernels.has_head_candidates(0), reason="CUDA required")


def reference(x, bits, row_scale, ids, groups, keep):
    """The kernel's arithmetic in float64: scores of every row, then each group's best (the first of equal scores)."""
    rows, columns = x.shape
    blocks = x.double().view(rows, -1, 64)
    scale = (blocks.abs().amax(dim=2, keepdim=True).float() / torch.tensor(7.0, device=x.device)).double()  # a true division: PyTorch multiplies by the reciprocal of a Python scalar
    ratio = blocks.float() / scale.float().clamp_min(1e-45)  # the kernel's FP32 quotient, rounded half away from zero (roundf)
    q = torch.where(scale > 0, (torch.sign(ratio) * torch.floor(ratio.abs() + 0.5)).double(), torch.zeros_like(blocks))
    signs = ((bits.long()[:, :, None] >> torch.arange(32, device=x.device)) & 1).view(bits.shape[0], -1).double() * 2 - 1
    scores = ((q * scale).view(rows, -1) @ signs.T) * row_scale.double()
    best = scores.view(rows, groups, -1).topk(keep, dim=2, sorted=True).indices
    return scores, ids.view(groups, -1)[torch.arange(groups, device=x.device)[None, :, None], best].view(rows, -1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("count,columns,groups,keep,rows", [(4096, 5120, 16, 4, 1), (3072, 1024, 64, 2, 3), (2048, 8192, 2, 4, 2), (1280, 128, 320, 1, 5)])
@torch.inference_mode()
def test_kept_rows_are_each_group_best(dtype, count, columns, groups, keep, rows):
    generator = torch.Generator().manual_seed(count + columns + rows)
    weight = torch.randn(count, columns, generator=generator).cuda()
    bits, row_scale = kernels.head_sign_bits(weight)
    assert torch.equal(((bits.long()[:, :, None] >> torch.arange(32, device="cuda")) & 1).view(count, -1).bool(), weight >= 0)
    ids = torch.randperm(count, generator=generator).int().cuda()
    x = (torch.randn(rows, columns, generator=generator) * torch.rand(rows, columns, generator=generator).pow(4) * 30).cuda().to(dtype)
    grouped = kernels.head_group_bits(bits, groups)
    assert grouped.shape == bits.shape and torch.equal(grouped.view(groups, -1, count // groups, 4).permute(0, 2, 1, 3), bits.view(groups, count // groups, -1, 4))
    out = kernels.head_candidates(x, grouped, row_scale, ids, groups, keep)
    assert out.shape == (rows, groups * keep) and out.dtype == torch.int32
    scores, expected = reference(x.float(), bits, row_scale, ids, groups, keep)
    position = torch.empty(count, dtype=torch.long, device="cuda")
    position[ids.long()] = torch.arange(count, device="cuda")
    chosen, wanted = scores.gather(1, position[out.long()]), scores.gather(1, position[expected.long()])
    # float32 sums (a difference of two large ones) against float64: the kept rows may differ where two scores are within the rounding; their scores do not
    torch.testing.assert_close(chosen, wanted, rtol=0, atol=3e-4 * float(scores.abs().max()))
    assert (out.view(rows, groups, keep).sort(dim=2).values.diff(dim=2) != 0).all()  # no row twice
    group_of = position[out.long()].div(count // groups, rounding_mode="floor")
    assert torch.equal(group_of, torch.arange(groups, device="cuda").repeat_interleave(keep)[None].expand(rows, -1))
    assert torch.equal(out, kernels.head_candidates(x, grouped, row_scale, ids, groups, keep))  # the same launch twice


@torch.inference_mode()
def test_equal_scores_keep_the_first_rows_and_zero_inputs_are_served():
    weight = torch.ones(512, 256).cuda()
    bits, row_scale = kernels.head_sign_bits(weight)
    ids = torch.arange(512, dtype=torch.int32).cuda()
    bits = kernels.head_group_bits(bits, 4)
    out = kernels.head_candidates(torch.ones(1, 256).cuda().half(), bits, row_scale, ids, 4, 3)
    assert out.view(4, 3).tolist() == [[0, 1, 2], [128, 129, 130], [256, 257, 258], [384, 385, 386]]
    assert kernels.head_candidates(torch.zeros(2, 256).cuda(), bits, row_scale, ids, 4, 3).view(2, 4, 3).tolist() == [out.view(4, 3).tolist()] * 2


def test_unsupported_shapes_are_refused():
    bits, row_scale = kernels.head_sign_bits(torch.randn(96, 256).cuda())
    ids = torch.arange(96, dtype=torch.int32).cuda()
    for groups, keep in ((5, 1), (96, 2)):
        with pytest.raises(ValueError):
            kernels.head_candidates(torch.randn(1, 256).cuda(), bits, row_scale, ids, groups, keep)
    with pytest.raises(ValueError):
        kernels.head_sign_bits(torch.randn(4, 100).cuda())
    with pytest.raises(ValueError):
        kernels.head_group_bits(bits, 5)
