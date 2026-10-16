"""Public capability and malformed-layout checks must precede any GPU launch."""
from unittest.mock import patch

import pytest
import torch
import llamacpp_gguf_cuda as api
from llamacpp_gguf_cuda import _packed
from packed_fixtures import _weight


def test_supported_devices_do_not_require_wangp():
    import sys
    assert not any(name == 'shared' or name.startswith('shared.') for name in sys.modules)
    for capability in ((7, 5), (8, 6), (8, 9), (12, 0)):
        _packed.implementation.cache_clear()
        with patch.object(torch.cuda, 'get_device_capability', return_value=capability):
            assert api.supports_linear_layout('Q4_K', (16, 1136), 'cuda:0', torch.float16)
            assert api.packed_row_kernel('Q4_K', 0, torch.float16) is _packed.implementation(0)
    _packed.implementation.cache_clear()


def test_unsupported_layouts():
    assert not api.supports_linear_layout('IQ4_XS', (16, 1136), 'cuda:0', torch.float16)
    assert not api.supports_linear_layout('Q4_K', (16, 1136), 'cpu', torch.float16)
    assert not api.supports_linear_layout('Q4_K', (16, 1136), 'cuda:0', torch.float64)
    assert not api.supports_linear_layout('Q4_K', (1136,), 'cuda:0', torch.float16)
    assert api.packed_row_kernel('IQ4_XS', 0, torch.float16) is None
    assert api.supports_linear_layout('IQ4_XS', (16, 256), 'cuda:0', torch.float16)


@pytest.mark.parametrize('invalid', ['bytes', 'raw_dtype', 'input_width', 'bias', 'blocks'])
def test_bad_metadata_is_rejected(invalid):
    w, _ = _weight('Q4_K', 16, 1136, torch.float16)
    raw, shape = w._data, list(w.shape)
    x, bias = torch.ones(2, shape[1], device='cuda', dtype=torch.float16), None
    if invalid == 'bytes': raw = raw[:-1]
    if invalid == 'raw_dtype': raw = raw.to(torch.int8)
    if invalid == 'input_width': x = x[:, :-1]
    if invalid == 'bias': bias = torch.zeros(15, device='cuda')
    if invalid == 'blocks': shape[0] -= 1
    with pytest.raises(ValueError):
        api.linear(raw, 'Q4_K', shape, x, bias, torch.float16)
