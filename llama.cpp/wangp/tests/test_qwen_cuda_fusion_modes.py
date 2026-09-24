"""CPU/FakeTensor checks for decoder-mode routing and MMGP module calls."""
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from shared.qtypes import gguf
from shared.llm_engines.nanovllm.layers import activation, attention
from shared.llm_engines.nanovllm.models.qwen3_5 import Qwen3_5Block
from shared.llm_engines.nanovllm.utils.context import Context
from shared.prompt_enhancer.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from shared.prompt_enhancer.qwen35_text import _apply_qwen35_projection_fusions


def quantized_linear(columns=256, rows=128, router=False):
    qtype = gguf.gguf.GGMLQuantizationType.Q4_0
    dense = np.random.default_rng(781).normal(0, .03, (rows, columns)).astype(np.float32)
    raw = torch.from_numpy(gguf.gguf.quants.quantize(dense, qtype))
    packed = gguf.GGUFSourceTensor.wrap(raw, tensor_type=qtype, tensor_shape=(rows, columns))
    if router:
        from mmgp.quant_router import QLinearQuantoRouter, _load_with_qmodule
        module = QLinearQuantoRouter(columns, rows, bias=False, dtype=torch.bfloat16, device='cpu', weights=gguf._GGUF_QTYPE)
        errors = []
        _load_with_qmodule(module, gguf.QLinearGGUF, {'weight': packed}, '', {}, True, [], [], errors)
        assert not errors
    else:
        module = gguf.QLinearGGUF(columns, rows, bias=False, dtype=torch.bfloat16, device='cpu', weights=gguf._GGUF_QTYPE)
        module.load_state_dict({'weight': packed})
    return module


def make_model(mode, router=False):
    config = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64, layer_types=['full_attention'])
    config._prompt_enhancer_safe_legacy = mode != 'vllm'
    model = torch.nn.Module()
    model.config = config
    with torch.device('cpu'):
        block = Qwen3_5Block(config, 0).to(torch.bfloat16)
    block.ffn_gate = quantized_linear(128, 256, router)
    block.ffn_up = quantized_linear(128, 256, router)
    block.ffn_down = quantized_linear(router=router)
    model.blk = torch.nn.ModuleList([block])
    model.output = quantized_linear(128, 32, router)
    return model


@pytest.mark.parametrize('mode', ['legacy', 'cg', 'vllm'])
@pytest.mark.parametrize('hip', [None, '7.14'])
def test_existing_mode_controls_fusions_without_replacing_forwards(monkeypatch, mode, hip):
    monkeypatch.setattr(torch.version, 'hip', hip)
    model = make_model(mode)
    untouched = quantized_linear()
    _apply_qwen35_projection_fusions(model)
    block = model.blk[0]
    enabled = mode == 'vllm' and hip is None
    for projection in (block.ffn_gate_up, block.ffn_down, model.output):
        assert projection._use_optimized_kernels == enabled
        assert 'forward' not in projection.__dict__
    assert block.ffn_down._fuse_silu_mul == enabled
    assert block.mlp_act_fn.use_triton == (mode == 'vllm')
    assert block.attn.use_triton_kv_cache == (mode == 'vllm' and attention.triton is not None)
    assert not getattr(untouched, '_use_optimized_kernels', False)


def test_prism_retains_separate_activation_before_hadamard():
    model = make_model('vllm')
    _apply_qwen35_projection_fusions(model, prism=True)
    assert not model.blk[0].ffn_down._fuse_silu_mul


@pytest.mark.parametrize('silu_mul', [False, True])
def test_native_fusion_fake_contract(silu_mul):
    with FakeTensorMode():
        x = torch.empty(2, 3, 512 if silu_mul else 256, dtype=torch.bfloat16, device='cuda')
        raw = torch.empty(8, 144, dtype=torch.uint8, device='cuda')
        out = gguf.linear_fused(x, raw, 'Q4_K', [8, 256], None, x.dtype, silu_mul)
        assert out.shape == (2, 3, 8) and out.dtype == x.dtype and out.device == x.device


@pytest.mark.parametrize('silu_mul', [False, True])
def test_older_wheel_keeps_existing_linear_signature(monkeypatch, silu_mul):
    projection = quantized_linear()
    projection._use_optimized_kernels = True
    monkeypatch.setattr(gguf, '_gguf_cuda_module', lambda: SimpleNamespace())
    monkeypatch.setattr(activation, 'triton', None)
    calls = []
    def existing_linear(weight, x, bias=None):
        calls.append(x.shape)
        return x.new_empty((*x.shape[:-1], weight.shape[0]))
    def forbidden(*args, **kwargs):
        pytest.fail('An older wheel must not receive the new fused signature')
    monkeypatch.setattr(gguf.GGUFWeightTensor, 'linear', existing_linear)
    monkeypatch.setattr(gguf, 'linear_fused', forbidden)
    with FakeTensorMode():
        x = torch.empty(1, 512 if silu_mul else 256, dtype=torch.bfloat16, device='cuda')
        out = projection([x] if silu_mul else x)
    assert calls == [torch.Size([1, 256])] and out.shape == (1, 128)


@pytest.mark.parametrize('rows', [2, 3, 5, 512])
@pytest.mark.parametrize('fuse_activation', [False, True])
def test_multitoken_linear_keeps_existing_kernel_without_capability_probe(monkeypatch, rows, fuse_activation):
    projection = quantized_linear()
    projection._use_optimized_kernels = True
    monkeypatch.setattr(activation, 'triton', None)
    def forbidden(*args, **kwargs):
        pytest.fail('Multi-token linear must retain its existing kernel without a native capability probe')
    monkeypatch.setattr(gguf, '_gguf_cuda_module', forbidden)
    monkeypatch.setattr(gguf, 'linear_fused', forbidden)
    monkeypatch.setattr(gguf.GGUFWeightTensor, 'linear', lambda weight, x, bias=None: x.new_empty((*x.shape[:-1], weight.shape[0])))
    with FakeTensorMode():
        x = torch.empty(rows, 512 if fuse_activation else 256, dtype=torch.bfloat16, device='cuda')
        out = projection([x] if fuse_activation else x)
    assert out.shape == (rows, 128)


def test_activation_handoff_releases_original_before_unfused_linear(monkeypatch):
    projection = quantized_linear()
    handoff = [torch.randn(3, 512, dtype=torch.bfloat16, device='cpu')]
    original = weakref.ref(handoff[0])
    def existing_linear(weight, x, bias=None):
        assert original() is None
        return x.new_empty((*x.shape[:-1], weight.shape[0]))
    monkeypatch.setattr(gguf.GGUFWeightTensor, 'linear', existing_linear)
    out = projection(handoff)
    assert handoff == [] and out.shape == (3, 128)


@pytest.mark.parametrize('mode', ['legacy', 'cg', 'vllm'])
@pytest.mark.parametrize('router', [False, True])
@pytest.mark.parametrize('tokens', [1, 3])
@torch.inference_mode()
def test_block_ffn_preserves_module_hooks_and_values(monkeypatch, mode, router, tokens):
    torch.manual_seed(171)
    model = make_model(mode, router)
    _apply_qwen35_projection_fusions(model)
    block = model.blk[0]
    # Exercise the real block on CPU; GPU arithmetic still needs separate validation.
    block.attn.flash_attn_varlen_func = block.attn.flash_attn_with_kvcache = None
    cu = torch.tensor([0, tokens], dtype=torch.int32, device='cpu')
    monkeypatch.setattr(attention, 'get_context', lambda: Context(is_prefill=True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=tokens, max_seqlen_k=tokens))
    def forbidden(*args, **kwargs):
        pytest.fail('CPU / legacy / cg must not invoke a new CUDA or Triton kernel')
    monkeypatch.setattr(gguf, 'linear_fused', forbidden)
    monkeypatch.setattr(activation, '_silu_mul_kernel', SimpleNamespace(__getitem__=forbidden))
    x = torch.randn(1, tokens, 128, dtype=torch.bfloat16, device='cpu')
    angle = torch.randn(1, tokens, 16, dtype=torch.bfloat16, device='cpu')
    positions = (angle.cos(), angle.sin())
    hooks, handed_off = [], []
    def on_down(module, inputs):
        hooks.append(isinstance(inputs[0], list))
        if isinstance(inputs[0], list):
            handed_off.append(inputs[0])
    block.ffn_down.register_forward_pre_hook(on_down)
    actual, residual = block([x.clone(), None], positions, 0)
    block.ffn_down._fuse_silu_mul = False
    expected, expected_residual = block([x.clone(), None], positions, 0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(residual, expected_residual, atol=0, rtol=0)
    assert hooks == [mode == 'vllm' and torch.version.hip is None and tokens == 1, False]
    assert all(handoff == [] for handoff in handed_off)
    if router:
        assert block.ffn_down._router_forward_impl is gguf.QLinearGGUF.forward


@pytest.mark.parametrize('cache_written', [False, True])
def test_attention_writes_cache_exactly_once(monkeypatch, cache_written):
    layer = attention.Attention(2, 32, 32 ** -.5, 1)
    layer.use_triton_kv_cache = False
    layer.k_cache = torch.zeros(1, 4, 1, 32, device='cpu')
    layer.v_cache = torch.zeros_like(layer.k_cache)
    q = torch.randn(1, 2, 32, device='cpu')
    k, v = torch.randn(2, 1, 1, 32, device='cpu').unbind()
    slots = torch.zeros(1, dtype=torch.int64, device='cpu')
    context = Context(is_prefill=True, slot_mapping=slots)
    monkeypatch.setattr(attention, 'get_context', lambda: context)
    writes = []
    original_store = attention.store_kvcache
    def store(*args, **kwargs):
        writes.append(True)
        return original_store(*args, **kwargs)
    monkeypatch.setattr(attention, 'store_kvcache', store)
    layer.flash_attn_varlen_func = lambda query, *args, **kwargs: query
    if cache_written:
        original_store(k, v, layer.k_cache, layer.v_cache, slots, use_triton_kv_cache=False)
    handoff = [q, k, v]
    layer.forward_list(handoff, cache_written=cache_written)
    torch.testing.assert_close(layer.k_cache[0, 0], k[0], atol=0, rtol=0)
    torch.testing.assert_close(layer.v_cache[0, 0], v[0], atol=0, rtol=0)
    assert len(writes) == (0 if cache_written else 1)
    assert handoff == []
