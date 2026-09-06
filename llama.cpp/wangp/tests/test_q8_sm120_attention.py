"""Numerics, replay and bounded compilation for optional SM120 Q8 attention."""
import re
import sys

import pytest
import torch
import torch.nn.functional as F

from shared.llm_engines.nanovllm.layers import attention
from test_q8_paged_attention import _case, _reference

sm120 = pytest.importorskip("shared.llm_engines.nanovllm.layers.attention_sm120")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="SM120 required")


@pytest.fixture(autouse=True)
def use_gluon_implementation(monkeypatch):
    monkeypatch.setattr(attention, "_SM120_Q8", sm120)


def _shared(monkeypatch, function, *args):
    with monkeypatch.context() as context:
        context.setattr(attention, "_SM120_Q8", None)
        return function(*args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("queries,lengths", [([1, 17, 33], [257, 531, 991]), ([1024], [20000])])
@torch.inference_mode()
def test_prefill_exact_with_shuffled_pages_and_tails(monkeypatch, dtype, queries, lengths):
    args = _case(dtype, 256, queries, lengths)
    actual = attention._q8_paged_prefill(*args, 256 ** -.5)
    expected = _shared(monkeypatch, attention._q8_paged_prefill, *args, 256 ** -.5)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("queries", [1, 2, 3, 5, 9])
@torch.inference_mode()
def test_grouped_exact_with_partial_and_empty_splits(monkeypatch, dtype, queries):
    q, k, v, ks, vs, context = _case(dtype, 256, [queries, queries], [531, 20000])
    args = (q, k, v, ks, vs, context.block_tables, context.context_lens, 256 ** -.5)
    run = lambda: attention._q8_grouped_attention(*args)
    expected = _shared(monkeypatch, run)
    actual = run()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for length in (queries, 257, 0):
        context.context_lens.fill_(length)
        q.mul_(1.125)
        context.block_tables[:, :2] = context.block_tables[:, :2].flip(1)
        graph.replay()
        expected = _shared(monkeypatch, run)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_prefill_graph_replays_changed_queries_lengths_and_pages(monkeypatch, dtype):
    q, k, v, ks, vs, context = _case(dtype, 256, [17, 33], [531, 991])
    run = lambda: attention._q8_paged_prefill(q, k, v, ks, vs, context, 256 ** -.5)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for queries, lengths in (([1, 49], [257, 513]), ([33, 17], [511, 769])):
        context.cu_seqlens_q.copy_(torch.tensor([0, queries[0], sum(queries)], device="cuda", dtype=torch.int32))
        context.cu_seqlens_k.copy_(torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32))
        context.context_lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        context.block_tables[:, :2] = context.block_tables[:, :2].flip(1)
        q.mul_(1.25)
        graph.replay()
        expected = _shared(monkeypatch, run)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.testing.assert_close(output, _reference(q, k, v, ks, vs, context), atol=.0003, rtol=.009)


@pytest.mark.parametrize("capability,dim,installed", [((8, 9), 256, True), ((9, 0), 256, True), ((10, 0), 256, True), ((12, 1), 256, True), ((12, 0), 128, True), ((12, 0), 256, False)])
@torch.inference_mode()
def test_other_architectures_dimensions_and_old_triton_keep_shared_kernels(monkeypatch, capability, dim, installed):
    def forbidden(*args, **kwargs):
        raise AssertionError("SM120 staging was selected outside its supported architecture/shape")
    monkeypatch.setattr(sm120, "q8_paged_prefill", forbidden)
    monkeypatch.setattr(sm120, "q8_grouped_partials", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    if not installed:
        monkeypatch.setattr(attention, "_SM120_Q8", None)
    q, k, v, ks, vs, context = _case(torch.bfloat16, dim, [3], [257])
    expected = _shared(monkeypatch, attention._q8_paged_prefill, q, k, v, ks, vs, context, dim ** -.5)
    actual = attention._q8_paged_prefill(q, k, v, ks, vs, context, dim ** -.5)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    args = (q, k, v, ks, vs, context.block_tables, context.context_lens, dim ** -.5)
    expected = _shared(monkeypatch, attention._q8_grouped_attention, *args)
    torch.testing.assert_close(attention._q8_grouped_attention(*args), expected, atol=0, rtol=0)


@torch.inference_mode()
def test_one_prefill_and_eight_grouped_variants_cover_length_ranges(monkeypatch):
    hashes = {"q8_prefill_async_kernel": set(), "q8_grouped_async_kernel": set()}
    for name in hashes:
        kernel = getattr(sm120, name)
        original = kernel.run
        def run(*args, _original=original, _name=name, **kwargs):
            compiled = _original(*args, **kwargs)
            hashes[_name].add(compiled.hash)
            assert re.search(r"cp\.async\.(ca|cg)\.shared\.global", compiled.asm["ptx"])
            assert "cp.async.bulk.tensor" not in compiled.asm["ptx"]
            return compiled
        monkeypatch.setattr(kernel, "run", run)
    for batch, queries, length, width in [(1, 1, 33, 1), (1, 3, 257, 3), (1, 5, 991, 16), (1, 9, 20000, 79), (2, 2, 221, 17), (2, 7, 20000, 128), (1, 33, 512, 2), (1, 1024, 20000, 128)]:
        q, k, v, ks, vs, context = _case(torch.bfloat16, 256, [queries] * batch, [length] * batch)
        context.block_tables = F.pad(context.block_tables, (0, width - context.block_tables.shape[1]), value=-1)
        actual = attention._q8_paged_prefill(q, k, v, ks, vs, context, 256 ** -.5)
        expected = _shared(monkeypatch, attention._q8_paged_prefill, q, k, v, ks, vs, context, 256 ** -.5)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if queries <= 9:
            args = (q, k, v, ks, vs, context.block_tables, context.context_lens, 256 ** -.5)
            actual = attention._q8_grouped_attention(*args)
            expected = _shared(monkeypatch, attention._q8_grouped_attention, *args)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(hashes["q8_prefill_async_kernel"]) == 1
    assert 1 <= len(hashes["q8_grouped_async_kernel"]) <= 8


@pytest.mark.skipif(sys.platform != "win32", reason="Windows driver stack regression")
@torch.inference_mode()
def test_async_staging_does_not_expand_driver_stack():
    import ctypes
    driver = ctypes.WinDLL("nvcuda.dll")
    torch.cuda.synchronize()
    before, after = ctypes.c_size_t(), ctypes.c_size_t()
    assert driver.cuCtxGetLimit(ctypes.byref(before), 0) == 0
    q, k, v, ks, vs, context = _case(torch.bfloat16, 256, [1024], [20000])
    attention._q8_paged_prefill(q, k, v, ks, vs, context, 256 ** -.5)
    attention._q8_grouped_attention(q[:5], k, v, ks, vs, context.block_tables, context.context_lens, 256 ** -.5)
    torch.cuda.synchronize()
    assert driver.cuCtxGetLimit(ctypes.byref(after), 0) == 0
    assert after.value == before.value
