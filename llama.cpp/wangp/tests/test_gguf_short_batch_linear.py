"""Short-batch GGUF linear (speculative verification) must be as accurate as single-row MMVQ.

The native package can route 2-8 row Q4_K and PTQ1_0 linears to INT8 tensor-core
kernels (compute capability 8.0+; default on 12.0, elsewhere chosen by measurement).
Synthetic weights keep this independent of checkpoints.
"""
from types import SimpleNamespace

import pytest
import torch

from shared.kernels import gguf_short_batch

native = pytest.importorskip("llamacpp_gguf_cuda")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not native.supports_linear_qtype_name("Q4_K"), reason="GGUF CUDA kernels required")


def _q4k(rows, cols, generator):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    blocks = torch.randint(0, 256, (rows * cols // 256, 144), generator=generator, dtype=torch.uint8)
    scales = torch.empty(rows * cols // 256, 2, dtype=torch.float16)
    scales[:, 0].uniform_(0.002, 0.02, generator=generator)
    scales[:, 1].uniform_(0.0, 0.01, generator=generator)
    blocks[:, :4] = scales.view(torch.uint8).reshape(-1, 4)
    dense = torch.from_numpy(dequantize(blocks.numpy(), GGMLQuantizationType.Q4_K)).float().reshape(rows, cols)
    return blocks.reshape(-1), dense


def _ptq1(rows, cols, generator):
    from shared.qtypes.gguf import _dequantize_blocks_PTQ1_0
    blocks = torch.empty(rows * cols // 128, 28, dtype=torch.uint8)
    blocks[:, :24] = torch.randint(0, 243, (blocks.shape[0], 24), generator=generator, dtype=torch.uint8)
    blocks[:, 24:26] = torch.randint(0, 81, (blocks.shape[0], 2), generator=generator, dtype=torch.uint8)
    blocks[:, 26:28] = torch.empty(blocks.shape[0], 1, dtype=torch.float16).uniform_(0.005, 0.03, generator=generator).view(torch.uint8)
    return blocks.reshape(-1), _dequantize_blocks_PTQ1_0(blocks, 128, 28, torch.float32).reshape(rows, cols)


def _linear(raw, qtype, shape, x, dtype, silu=False):
    return native.linear(raw, qtype, shape, x, None, dtype, fused_output=silu, silu_mul=silu).float()


@pytest.mark.parametrize("qtype,rows,cols", [("Q4_K", 1056, 5120), ("Q4_K", 512, 17408), ("PTQ1_0", 1056, 5120), ("PTQ1_0", 7040, 5120), ("PTQ1_0", 512, 17408)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_batched_rows_match_single_row_accuracy(qtype, rows, cols, dtype):
    if qtype == "PTQ1_0" and not native.supports_linear_qtype_name(qtype):
        pytest.skip("PTQ1_0 kernels unavailable")
    generator = torch.Generator().manual_seed(rows + cols)
    raw, dense = (_q4k if qtype == "Q4_K" else _ptq1)(rows, cols, generator)
    raw, dense = raw.cuda(), dense.cuda()
    x = torch.randn(9, cols, generator=generator).to("cuda", torch.bfloat16 if dtype == torch.float32 else dtype)
    reference = x.float() @ dense.T
    single = torch.cat([_linear(raw, qtype, [rows, cols], x[i:i + 1], dtype) for i in range(9)])
    for count in range(2, 9):  # 9+ rows use native MMQ, as before
        batched = _linear(raw, qtype, [rows, cols], x[:count], dtype)
        assert torch.isfinite(batched).all()
        error = (batched - reference[:count]).norm() / reference[:count].norm()
        single_error = (single[:count] - reference[:count]).norm() / reference[:count].norm()
        assert error <= single_error * 1.02 + 1e-5, (count, float(error), float(single_error))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_fused_silu_batches_match_single_row_accuracy(dtype):
    generator = torch.Generator().manual_seed(11)
    raw, dense = _q4k(512, 5120, generator)
    raw, dense = raw.cuda(), dense.cuda()
    x = torch.randn(5, 2 * 5120, generator=generator).to("cuda", dtype)
    gate, up = x[:, :5120].float(), x[:, 5120:].float()
    activated = (gate / (1 + torch.exp(-gate))).to(dtype).float() * up
    reference = activated.to(dtype).float() @ dense.T
    for count in range(2, 6):
        if not native.supports_linear_fusions("Q4_K", count, 0):
            continue
        batched = _linear(raw, "Q4_K", [512, 5120], x[:count], dtype, silu=True)
        single = torch.cat([_linear(raw, "Q4_K", [512, 5120], x[i:i + 1], dtype, silu=True) for i in range(count)])
        error = (batched - reference[:count]).norm() / reference[:count].norm()
        single_error = (single - reference[:count]).norm() / reference[:count].norm()
        assert error <= single_error * 1.02 + 1e-5, (count, float(error), float(single_error))


@pytest.mark.parametrize("qtype", ["Q4_K", "PTQ1_0"])
@torch.inference_mode()
def test_short_batch_graph_replay(qtype):
    generator = torch.Generator().manual_seed(5)
    raw, _ = (_q4k if qtype == "Q4_K" else _ptq1)(1056, 5120, generator)
    raw = raw.cuda()
    x = torch.randn(5, 5120, device="cuda", dtype=torch.bfloat16)
    _linear(raw, qtype, [1056, 5120], x, torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = native.linear(raw, qtype, [1056, 5120], x, None, torch.bfloat16)
    for _ in range(3):
        x.normal_()
        graph.replay()
        torch.testing.assert_close(out, native.linear(raw, qtype, [1056, 5120], x, None, torch.bfloat16), atol=0, rtol=0)


policy = pytest.mark.skipif(not getattr(native, "has_short_batch_policy", lambda: False)(), reason="package without short-batch policy")


@pytest.fixture
def clean_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(gguf_short_batch, "_cache_path", lambda: tmp_path / "choices.json")
    monkeypatch.setattr(gguf_short_batch, "_disk", None)
    monkeypatch.setattr(gguf_short_batch, "_choices", {})
    monkeypatch.setattr(gguf_short_batch, "_cold_bytes", lambda device: 1)
    native.clear_short_batch_decisions()
    native.set_short_batch_mode("auto")
    yield tmp_path / "choices.json"
    native.clear_short_batch_decisions()
    native.set_short_batch_mode("auto")


def _model(counts):
    """A module holding GGUF weights as the engine does, e.g. {("Q4_K", 1056, 5120): 3}."""
    from gguf import GGMLQuantizationType
    from shared.qtypes.gguf import GGUFWeightTensor, PrismQuantizationType
    model = torch.nn.Module()
    for (qtype, rows, cols), count in counts.items():
        for index in range(count):
            raw, _ = (_q4k if qtype == "Q4_K" else _ptq1)(rows, cols, torch.Generator().manual_seed(index))
            tensor_type = GGMLQuantizationType.Q4_K if qtype == "Q4_K" else PrismQuantizationType.PTQ1_0
            layer = torch.nn.Module()
            layer._parameters["weight"] = GGUFWeightTensor.create(raw_tensor=raw.cuda(), size=(rows, cols), stride=(cols, 1), dtype=torch.bfloat16,
                                                                  tensor_type=tensor_type, tensor_shape=(rows, cols))
            model.add_module(f"{qtype}_{rows}_{index}", layer)
    return model


@policy
@pytest.mark.parametrize("qtype", ["Q4_K", "PTQ1_0"])
@torch.inference_mode()
def test_auto_policy_applies_recorded_decisions(qtype, clean_policy):
    raw, _ = (_q4k if qtype == "Q4_K" else _ptq1)(1056, 5120, torch.Generator().manual_seed(3))
    raw = raw.cuda()
    x = torch.randn(5, 5120, device="cuda", dtype=torch.bfloat16)
    forced = {}
    for mode in ("native", "mma"):
        native.set_short_batch_mode(mode)
        forced[mode] = native.linear(raw, qtype, [1056, 5120], x, None, torch.bfloat16)
    native.set_short_batch_mode("auto")
    assert not torch.equal(forced["native"], forced["mma"])  # distinguishable kernels
    for enabled, mode in ((False, "native"), (True, "mma")):
        native.set_short_batch_decision(qtype, 5, 1056, 5120, enabled)
        torch.testing.assert_close(native.linear(raw, qtype, [1056, 5120], x, None, torch.bfloat16), forced[mode], atol=0, rtol=0)
    with pytest.raises(ValueError):
        native.set_short_batch_mode("fast")


@policy
@torch.inference_mode()
def test_prepare_times_model_weights_once_and_caches_on_disk(clean_policy, monkeypatch):
    model = _model({("Q4_K", 1056, 5120): 3, ("PTQ1_0", 512, 5120): 2, ("Q4_K", 512, 1000): 1})
    # Cache-cold rotation needs all three Q4_K weights; the PTQ1_0 group is too small to time.
    monkeypatch.setattr(gguf_short_batch, "_cold_bytes", lambda device: 2 * 1056 * 5120 // 256 * 144 + 1)
    gguf_short_batch.prepare(model, native)
    entries = next(iter(gguf_short_batch._load_disk().values()))
    assert sorted(entries) == sorted(f"Q4_K|{r}|1056|5120" for r in gguf_short_batch.ROWS)  # K % 256 != 0 is ignored
    assert all(record["error"] < gguf_short_batch.AGREEMENT and record["weights"] == 3 for record in entries.values())
    default = torch.cuda.get_device_capability() == (12, 0)
    assert all(value == default for key, value in gguf_short_batch._choices.items() if key[1] == "PTQ1_0")
    assert len(gguf_short_batch._choices) == 14 and clean_policy.exists()
    # A new process reuses the disk choices without timing and applies the same decisions.
    before = dict(gguf_short_batch._choices)
    monkeypatch.setattr(gguf_short_batch, "_choices", {})
    monkeypatch.setattr(gguf_short_batch, "_disk", None)
    native.clear_short_batch_decisions()
    monkeypatch.setattr(gguf_short_batch, "_measure", lambda *args, **kwargs: pytest.fail("cached choice was re-measured"))
    gguf_short_batch.prepare(model, native)
    assert gguf_short_batch._choices == before


@policy
@pytest.mark.parametrize("speedup,default_mma,expected", [(1.06, False, True), (1.04, False, False), (0.97, True, True), (0.94, True, False)])
@torch.inference_mode()
def test_prepare_keeps_default_within_margin(speedup, default_mma, expected, clean_policy, monkeypatch):
    props = torch.cuda.get_device_properties(0)
    fake = SimpleNamespace(name=props.name, uuid="fake", major=12 if default_mma else 8, minor=0 if default_mma else 9,
                           multi_processor_count=props.multi_processor_count, L2_cache_size=props.L2_cache_size)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: fake)
    monkeypatch.setattr(gguf_short_batch, "_measure", lambda *args, **kwargs: ({"native": speedup, "mma": 1.0}, 1e-3))
    gguf_short_batch.prepare(_model({("PTQ1_0", 512, 5120): 2}), native)
    assert set(gguf_short_batch._choices.values()) == {expected} and len(gguf_short_batch._choices) == 7


@policy
@torch.inference_mode()
def test_small_groups_disagreement_and_forced_modes(clean_policy, monkeypatch):
    model = _model({("Q4_K", 1056, 5120): 2})
    monkeypatch.setattr(gguf_short_batch, "_cold_bytes", lambda device: 1 << 40)
    monkeypatch.setattr(gguf_short_batch, "_measure", lambda *args, **kwargs: pytest.fail("small groups must not be timed"))
    gguf_short_batch.prepare(model, native)
    default = torch.cuda.get_device_capability() == (12, 0)
    assert set(gguf_short_batch._choices.values()) == {default} and not clean_policy.exists()  # default kept, nothing cached
    monkeypatch.setattr(gguf_short_batch, "_choices", {})
    monkeypatch.setattr(gguf_short_batch, "_cold_bytes", lambda device: 1)
    monkeypatch.setattr(gguf_short_batch, "_measure", lambda *args, **kwargs: (None, 0.5))
    gguf_short_batch.prepare(model, native)
    assert set(gguf_short_batch._choices.values()) == {False}
    monkeypatch.setattr(gguf_short_batch, "_choices", {})
    native.set_short_batch_mode("native")  # LLAMACPP_GGUF_SHORT_BATCH=native
    gguf_short_batch.prepare(model, native)
    assert not gguf_short_batch._choices
