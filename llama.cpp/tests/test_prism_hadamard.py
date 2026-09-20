"""Compare the fused CUDA FWHT with independent CPU Sylvester butterflies."""
import torch
import llamacpp_gguf_cuda as kernels


def reference(x, signs, inverse, grouped_shape):
    shape = x.shape
    nk, rep, hd = grouped_shape
    if nk:
        x = x.reshape(*shape[:-1], rep, nk, hd).transpose(-3, -2).reshape(shape)
    y = x.float() if inverse else x.float() * signs
    for stride in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        a, b = y.reshape(-1, 2, stride).unbind(1)
        y = torch.stack((a+b, a-b), 1)
    y = y.reshape(shape) / 32
    return (y * signs if inverse else y).to(x.dtype)


@torch.inference_mode()
def test():
    torch.manual_seed(42)
    count = 0
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for width in (1024, 5120, 6144, 17408):
            signs = torch.randint(0, 2, (width,), device='cpu', dtype=torch.int8) * 2 - 1
            for batch in (1, 3, 129):
                for inverse, group in ((False, (0,0,0)), (True, (0,0,0)), *(([(False, (16,3,128))]) if width == 6144 else [])):
                    x = torch.randn(batch, width, device='cpu', dtype=dtype)
                    gx, gs = x.cuda(), signs.cuda()
                    actual = kernels.prism_hadamard(gx, gs, inverse, group)
                    torch.testing.assert_close(actual.cpu(), reference(x, signs, inverse, group), rtol=0, atol=0)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = kernels.prism_hadamard(gx, gs, inverse, group)
                    x.normal_()
                    gx.copy_(x)
                    graph.replay()
                    torch.testing.assert_close(output.cpu(), reference(x, signs, inverse, group), rtol=0, atol=0)
                    count += 1
    print(f'PASS: {count} signed FWHT configurations including GDN permutation and graph replay.')


if __name__ == '__main__':
    test()
