import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

pytest.importorskip("triton")
from shared.kernels import quanto_int8_triton as kernels
from shared.kernels import quanto_int8_inject as inject


def test_decode_recovery_preserves_quantization_on_small_shared_memory_gpu(monkeypatch):
    baseline = (1, 32, 512, 4, 5)
    attempted = []
    monkeypatch.setattr(kernels, '_autotune_is_blocked', lambda: False)
    monkeypatch.setattr(kernels, '_create_bench_tensors', lambda *args: ())

    def probe(kind, cfg, tensors, m, k, n):
        attempted.append(cfg)
        assert kind == 'fused' and cfg[2] == 512
        if cfg[0] < 16:
            return None, RuntimeError('Input shapes should have M >= 16')
        if cfg[4] >= 4:
            return None, RuntimeError('out of resource: shared memory')
        return object(), None

    monkeypatch.setattr(kernels, '_run_candidate_once_with_error', probe)
    selected, error = kernels._ensure_compile_compatible_config('fused', 0, 'test', baseline, baseline, 1, 6144, 2048, ())
    assert error is None
    assert selected == (16, 32, 512, 4, 1)
    assert attempted[0] == baseline


@pytest.mark.parametrize("major,name,enabled", [(12, "NVIDIA GeForce RTX 5090", True), (12, "NVIDIA GeForce RTX 5070 Laptop GPU", True), (8, "NVIDIA GeForce RTX 4090", True), (9, "NVIDIA H100", True), (12, "NVIDIA RTX PRO 6000 Blackwell", True), (7, "NVIDIA GeForce RTX 2080", False)])
def test_timing_policy_extends_supported_nvidia(monkeypatch, major, name, enabled):
    monkeypatch.setattr(torch.version, "hip", None)
    props = SimpleNamespace(major=major, minor=0, name=name, multi_processor_count=80)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: props)
    assert kernels._use_cg_autotune(0) is enabled
    assert ("|bench=cg-v1" in kernels._device_fingerprint(0)) is enabled


@pytest.mark.parametrize("m,k,n", [(1, 2048, 2048), (512, 4096, 4096)])
def test_compile_skips_tuning_and_probes(monkeypatch, m, k, n):
    def forbidden(*args, **kwargs):
        raise AssertionError("Compilation must not allocate, probe or benchmark")
    monkeypatch.setattr(kernels, "_create_bench_tensors", forbidden)
    monkeypatch.setattr(kernels, "_launch_candidate", forbidden)
    baseline = kernels._select_static_triton_int8_config(m, k, n)
    expected = kernels._LOW_SHARED_MEMORY_CONFIG if baseline == kernels._HIGH_SHARED_MEMORY_CONFIG else baseline

    def function(x):
        cfg = kernels._select_triton_int8_config(m, k, n, device=x.device)
        assert cfg == expected
        assert kernels._benchmark_config_ms("fused", baseline, (), x.device, 1, 2048, 2048) is None
        assert kernels._autotune_config("fused", 0, 1, 2048, 2048, baseline, "test", ()) == baseline
        assert kernels._ensure_compile_compatible_config("fused", 0, "test", baseline, baseline, 1, 2048, 2048, ()) == (baseline, None)
        return x + cfg[0]

    x = torch.ones(4)
    actual = torch.compile(function, backend="eager", fullgraph=True)(x)
    torch.testing.assert_close(actual, x + expected[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_capture_defers_tuning_without_poisoning_launch_cache(monkeypatch):
    monkeypatch.setattr(inject, "_TRITON_MODULE", kernels)
    monkeypatch.setattr(inject, "_FUSED_LAUNCH_CACHE", {})
    monkeypatch.setattr(inject, "_FUSED_LAUNCH_CACHE_FIFO", [])
    monkeypatch.setattr(kernels, "_AUTOTUNE_SESSION_CACHE", {})
    monkeypatch.setattr(kernels, "_AUTOTUNE_CONFIG_CACHE", {})
    monkeypatch.setattr(kernels, "_AUTOTUNE_CACHE_LOADED", True)
    baseline = kernels._select_static_triton_int8_config(1, 2048, 2048)
    device = torch.device("cuda", torch.cuda.current_device())
    x, y = torch.ones(4, device=device), torch.empty(4, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert kernels._benchmark_config_ms("fused", baseline, (), device, 1, 2048, 2048) is None
        assert kernels._autotune_config("fused", device.index, 1, 2048, 2048, baseline, "test", ()) == baseline
        assert kernels._ensure_compile_compatible_config("fused", device.index, "test", baseline, baseline, 1, 2048, 2048, ()) == (baseline, None)
        assert inject._fused_launch_params(1, 2048, 2048, device)[:5] == baseline
        torch.add(x, 1, out=y)
    graph.replay()
    torch.cuda.synchronize()
    graph.reset()
    assert not inject._FUSED_LAUNCH_CACHE
    assert not kernels._AUTOTUNE_SESSION_CACHE
    tuned = (4, 64, 64, 4, 4)
    monkeypatch.setattr(kernels, "_select_triton_int8_config", lambda *args, **kwargs: tuned)
    assert inject._fused_launch_params(1, 2048, 2048, device)[:5] == tuned
    assert inject._FUSED_LAUNCH_CACHE
    torch.testing.assert_close(y, x + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fail_capture", [False, True])
@pytest.mark.parametrize("kind", ["fused", "convrot", "convrot_bias"])
@torch.inference_mode()
def test_temporary_graphs_are_reset_and_discarded(monkeypatch, fail_capture, kind):
    monkeypatch.setattr(kernels, "_use_cg_autotune", lambda _: True)
    graph_type = torch.cuda.CUDAGraph
    references, resets = [], []
    class TrackedGraph(graph_type):
        def __init__(self):
            super().__init__()
            references.append(weakref.ref(self))
        def reset(self):
            resets.append(True)
            return super().reset()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", TrackedGraph)
    device = torch.device("cuda", torch.cuda.current_device())
    cfg = (4, 64, 64, 4, 4)
    tensors = kernels._create_bench_tensors(kind, device, 1, 2048, 1024)
    original_launch = kernels._launch_candidate
    original_launch(kind, cfg, tensors, 1, 1024, 2048)
    torch.cuda.synchronize()
    if fail_capture:
        def launch(*args, **kwargs):
            if torch.cuda.is_current_stream_capturing():
                original_launch(*args, **kwargs)
                raise RuntimeError("Deliberate benchmark capture failure")
            return original_launch(*args, **kwargs)
        monkeypatch.setattr(kernels, "_launch_candidate", launch)
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        value = kernels._benchmark_config_ms(kind, cfg, tensors, device, 1, 2048, 1024)
        assert (value is None) is fail_capture
    gc.collect()
    torch.cuda.synchronize()
    assert len(resets) == 3
    assert all(reference() is None for reference in references)
    assert torch.cuda.memory_allocated() <= allocated + 4096


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["fused", "convrot", "convrot_bias"])
@torch.inference_mode()
def test_compatibility_benchmark_does_not_create_graphs(monkeypatch, kind):
    monkeypatch.setattr(kernels, "_use_cg_autotune", lambda _: False)
    def forbidden(*args, **kwargs):
        raise AssertionError("The compatibility benchmark must not create CUDA graphs")
    monkeypatch.setattr(torch.cuda, "CUDAGraph", forbidden)
    device = torch.device("cuda", torch.cuda.current_device())
    tensors = kernels._create_bench_tensors(kind, device, 1, 2048, 1024)
    assert kernels._benchmark_config_ms(kind, (4, 64, 64, 4, 4), tensors, device, 1, 2048, 1024) > 0
