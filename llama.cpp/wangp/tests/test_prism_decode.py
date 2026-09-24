"""Fused PTQ1 precision, replay, tuning and backend isolation contracts."""
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from shared.kernels import prism_decode as tuning
from shared.qtypes.prism import _decode_cuda, install_prism_decode


def test_fake_decode():
    with FakeTensorMode():
        x = torch.empty((1, 1, 1024), device='cuda', dtype=torch.bfloat16)
        raw = torch.empty(130*8*28, device='cuda', dtype=torch.uint8)
        signs = torch.empty(1024, device='cuda', dtype=torch.int8)
        result = _decode_cuda(x, raw, signs, None, 130, [0, 0, 0], 4)
        assert result.shape == (1, 1, 130) and result.dtype == x.dtype
        assert tuning.select_decode(x, raw, signs, None, 130, [0, 0, 0], None) == 0


def test_pre_ampere_gpu_unchanged(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (7, 5))
    # A model need not be inspected or mutated on unsupported architectures.
    install_prism_decode(None)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('default_device', ['cpu', 'cuda'])
def test_fused_rounding_and_replay(dtype, default_device):
    native = pytest.importorskip('llamacpp_gguf_cuda')
    if not native.has_prism_decode():
        pytest.skip('Fused PTQ1 extension not installed')
    previous = torch.get_default_device()
    torch.set_default_device(default_device)
    try:
        rows, width = 130, 1024
        raw = torch.randint(0, 256, (rows*width//128, 28), device='cpu', dtype=torch.uint8)
        raw[:, 26:28] = torch.tensor([.01], dtype=torch.float16, device='cpu').view(torch.uint8)
        raw = raw.cuda()
        x = torch.randn((1, 1, width*2), device='cuda', dtype=dtype)[..., ::2]
        signs = torch.randint(0, 2, (width,), device='cuda', dtype=torch.int8)*2-1
        bias = torch.randn(rows, device='cuda', dtype=dtype)
        def reference():
            return native.linear(raw, 'PTQ1_0', (rows, width), native.prism_hadamard(x, signs), bias, dtype)
        for tile in (1, 2, 4):
            def run():
                return _decode_cuda(x, raw, signs, bias, rows, [0, 0, 0], tile)
            torch.testing.assert_close(run(), reference(), rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = run()
            for zero in (True, False):
                x.zero_() if zero else x.normal_()
                bias.normal_()
                graph.replay()
                torch.testing.assert_close(result, reference(), rtol=0, atol=0)
    finally:
        torch.set_default_device(previous)


def test_capture_does_not_cache_untuned_choice(monkeypatch):
    tuning._choices.clear()
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    x = torch.empty((1, 1024), device='cuda')
    assert tuning.select_decode(x, None, None, None, 130, [0, 0, 0], None) == 0
    assert not tuning._choices
