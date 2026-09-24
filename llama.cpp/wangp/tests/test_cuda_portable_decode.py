"""CPU-only dispatch/probe tests. No CUDA launch or kernel compilation."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from shared.kernels import prism_decode, qwen_gdn, quanto_int8_triton
from shared.qtypes import prism


@pytest.fixture(autouse=True)
def clear_choices(monkeypatch):
    qwen_gdn._portable_choices.clear()
    prism_decode._choices.clear()
    monkeypatch.setattr(torch.version, 'hip', None)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    yield
    qwen_gdn._portable_choices.clear()
    prism_decode._choices.clear()


@pytest.mark.parametrize('capability', [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)])
def test_gdn_install_separates_portable_fusions_from_tuned_launches(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: capability)
    convolution = SimpleNamespace(backend='triton')
    norms = [SimpleNamespace(weight=torch.empty(5120, device='cpu'), _small_batch_num_warps=4) for _ in range(2)]
    block = SimpleNamespace(attn_norm=norms[0], post_attention_norm=norms[1],
                            layer_type='linear_attention', _fast_recurrent_gated_delta_rule=object(),
                            _use_short_convolution=True, ssm_conv1d=convolution, conv_kernel_size=4)
    # The SM120 class replacement needs a real compatible module; other targets
    # must preserve this sentinel object without trying to change its class.
    if capability == (12, 0):
        class Convolution:
            backend = 'triton'
        block.ssm_conv1d = Convolution()
    qwen_gdn.install_gdn_decode(SimpleNamespace(blk=[block]))
    validated = capability == (12, 0)
    assert block._gdn_prepare_decode is (qwen_gdn.prepare_decode if validated else qwen_gdn.prepare_decode_checked)
    assert block._gdn_recurrent_raw is (qwen_gdn.recurrent_raw_gates if validated else qwen_gdn.recurrent_raw_gates_checked)
    assert all(n._small_batch_num_warps == (16 if validated else 4) for n in norms)
    if not validated:
        assert block.ssm_conv1d is convolution


@pytest.mark.parametrize('hip,major,expected', [(None, 7, False), (None, 8, True), (None, 9, True), (None, 12, True), ('7.14', 12, False)])
def test_graph_timing_is_nvidia_only(monkeypatch, hip, major, expected):
    monkeypatch.setattr(torch.version, 'hip', hip)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(major=major))
    assert quanto_int8_triton._use_cg_autotune(0) is expected


def test_hip_and_pre_ampere_installers_do_not_mutate_models(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (7, 5))
    qwen_gdn.install_gdn_decode(None)
    prism.install_prism_decode(None)
    monkeypatch.setattr(torch.version, 'hip', '7.14')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (12, 0))
    qwen_gdn.install_gdn_decode(None)
    prism.install_prism_decode(None)


def gate_args():
    q = torch.ones(1, 1, 2, 4, device='cpu')
    a = torch.zeros(1, 1, 4, device='cpu')
    return q, q.clone(), a, a.clone(), -torch.ones(4, device='cpu'), torch.zeros(4, device='cpu'), 4


def test_gate_mismatch_falls_back_and_capture_does_not_freeze_choice(monkeypatch):
    args = gate_args()
    monkeypatch.setattr(qwen_gdn, '_probe_blocked', lambda _: True)
    qwen_gdn.prepare_decode_checked(*args)
    assert not qwen_gdn._portable_choices
    monkeypatch.setattr(qwen_gdn, '_probe_blocked', lambda _: False)
    def wrong(*args):
        outputs = list(qwen_gdn._prepare_reference(*args))
        outputs[2] += 1
        return tuple(outputs)
    monkeypatch.setattr(qwen_gdn, 'prepare_decode', wrong)
    outputs = qwen_gdn.prepare_decode_checked(*args)
    for actual, expected in zip(outputs, qwen_gdn._prepare_reference(*args)):
        torch.testing.assert_close(actual, expected)
    assert list(qwen_gdn._portable_choices.values()) == [False]


@pytest.mark.parametrize('mismatch', [False, True])
def test_recurrence_probe_preserves_live_state_and_prefixes(monkeypatch, mismatch):
    q = torch.ones(1, 3, 1, 2, device='cpu')
    a = torch.zeros(1, 3, 1, device='cpu')
    sa = torch.zeros(1, device='cpu')
    state = torch.zeros(1, 1, 2, 2, device='cpu')
    prefixes = torch.full((4, *state.shape), -99., device='cpu')
    def reference(q, k, v, a, b, sa, dt, initial, snapshots, **layout):
        for token in range(q.shape[1]):
            initial.add_(1)
            if snapshots is not None and token < q.shape[1] - 1:
                snapshots[token].copy_(initial)
        return torch.ones_like(v), initial
    def candidate(*args, **layout):
        output, result = reference(*args, **layout)
        if mismatch:
            result.add_(5)
        return output, result
    monkeypatch.setattr(qwen_gdn, '_probe_blocked', lambda _: False)
    monkeypatch.setattr(qwen_gdn, '_recurrent_reference', reference)
    monkeypatch.setattr(qwen_gdn, 'recurrent_raw_gates', candidate)
    args = (q, q, q, a, a, sa, sa, state, prefixes)
    _, result = qwen_gdn.recurrent_raw_gates_checked(*args)
    assert result is state
    assert torch.all(state == 3)
    assert torch.all(prefixes[0] == 1) and torch.all(prefixes[1] == 2)
    assert torch.all(prefixes[2:] == -99)
    assert list(qwen_gdn._portable_choices.values()) == [not mismatch]
    qwen_gdn.recurrent_raw_gates_checked(*args)
    assert torch.all(state == 6)


def test_prism_numerical_mismatch_selects_existing_kernel(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'device', lambda _: nullcontext())
    x = torch.ones(1, 1024, device='cpu')
    native = SimpleNamespace(prism_hadamard=lambda *args: x,
                             linear=lambda *args: torch.zeros(1, 16, device='cpu'),
                             prism_decode=lambda *args: torch.ones(1, 16, device='cpu'))
    assert prism_decode.select_decode(x, None, None, None, 16, (0, 0, 0), native) == 0


def test_prism_installer_enables_ampere_candidates(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (8, 6))
    monkeypatch.setattr(prism, '_gguf_cuda_module', lambda: SimpleNamespace(has_prism_decode=lambda: True))
    layer = SimpleNamespace(_router_forward_impl=prism.PrismLinear.forward,
                            weight=SimpleNamespace(_tensor_type=prism.PrismQuantizationType.PTQ1_0))
    prism.install_prism_decode(SimpleNamespace(modules=lambda: [layer]))
    assert layer._prism_decode_enabled


def test_recurrence_capture_first_preserves_later_probe(monkeypatch):
    q = torch.ones(1, 1, 1, 2, device='cpu')
    a = torch.zeros(1, 1, 1, device='cpu')
    sa = torch.zeros(1, device='cpu')
    state = torch.zeros(1, 1, 2, 2, device='cpu')
    monkeypatch.setattr(qwen_gdn, '_probe_blocked', lambda _: True)
    monkeypatch.setattr(qwen_gdn, '_recurrent_reference', lambda *args, **layout: (args[2], args[7]))
    def forbidden(*args):
        raise AssertionError('Probe must not launch inside capture')
    monkeypatch.setattr(qwen_gdn, 'recurrent_raw_gates', forbidden)
    qwen_gdn.recurrent_raw_gates_checked(q, q, q, a, a, sa, sa, state)
    assert not qwen_gdn._portable_choices


def test_recurrence_probe_memory_is_bounded(monkeypatch):
    q = torch.ones(1, 1, 1, 2, device='cpu')
    a = torch.zeros(1, 1, 1, device='cpu')
    sa = torch.zeros(1, device='cpu')
    state = torch.zeros(1, 1, 2, 2, device='cpu')
    monkeypatch.setattr(qwen_gdn, '_MAX_PROBE_STATE_BYTES', 1)
    monkeypatch.setattr(qwen_gdn, '_recurrent_reference', lambda *args, **layout: (args[2], args[7]))
    def forbidden(*args):
        raise AssertionError('Oversized probe must not launch')
    monkeypatch.setattr(qwen_gdn, 'recurrent_raw_gates', forbidden)
    qwen_gdn.recurrent_raw_gates_checked(q, q, q, a, a, sa, sa, state)
    assert list(qwen_gdn._portable_choices.values()) == [False]
