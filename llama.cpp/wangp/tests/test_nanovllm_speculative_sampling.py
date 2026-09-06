import math
from types import SimpleNamespace
from types import MethodType

import torch
import pytest

from shared.llm_engines.nanovllm.engine.model_runner import ModelRunner
from shared.llm_engines.nanovllm.layers.sampler import apply_sparse_repetition_penalty_
from shared.prompt_enhancer.qwen35_text import _build_prompt_logits_processor


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("virtual_count", [0, 6, 7, 8, 15])
@torch.inference_mode()
def test_repetition_penalty_handles_long_speculative_prefix(device, virtual_count):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = torch.linspace(-4, 4, 64, device=device)
    stored = [1, 33]
    new = [5, 37]
    virtual = list(range(40, 40 + virtual_count))
    token_buffer = torch.full((64,), -1, device=device, dtype=torch.int64)
    token_buffer[:len(stored)] = torch.tensor(stored, device=device)
    expected = source.clone()
    ids = stored + new + virtual
    expected[ids] = torch.where(source[ids] < 0, source[ids] * 1.2, source[ids] / 1.2)
    apply_sparse_repetition_penalty_(source, token_buffer, len(stored), new, virtual, 1.2)
    torch.testing.assert_close(source, expected)
    assert token_buffer[:4].tolist() == stored + new
    if device == "cuda":
        assert token_buffer[4:].eq(-1).all()


class _FakeLinearAttention:
    layer_type = "linear_attention"

    def __init__(self, device):
        self.conv_state_buffer = torch.empty((2, 3), dtype=torch.float32, device=device)
        self.recurrent_state_buffer = torch.empty((2, 2, 4), dtype=torch.float32, device=device)
        self.speculative_conv_state_buffer = torch.empty(0, device=device)
        self.speculative_recurrent_state_buffer = torch.empty(0, device=device)

    def prepare_speculative_state(self, max_verify_tokens):
        self.speculative_conv_state_buffer = torch.empty((max_verify_tokens - 1, *self.conv_state_buffer.shape), dtype=self.conv_state_buffer.dtype, device=self.conv_state_buffer.device)
        self.speculative_recurrent_state_buffer = torch.empty((max_verify_tokens - 1, *self.recurrent_state_buffer.shape), dtype=self.recurrent_state_buffer.dtype, device=self.recurrent_state_buffer.device)

    def commit_speculative_state(self, processed_tokens, verified_tokens):
        if processed_tokens == verified_tokens:
            return
        self.conv_state_buffer.copy_(self.speculative_conv_state_buffer[processed_tokens - 1])
        self.recurrent_state_buffer.copy_(self.speculative_recurrent_state_buffer[processed_tokens - 1])


def _reference_distribution(logits, top_k, top_p, min_p):
    if top_k is None:
        universe_logits = logits
        universe_ids = None
    else:
        threshold = torch.topk(logits, top_k).values[-1]
        universe_ids = torch.nonzero(logits >= threshold, as_tuple=False).flatten()
        universe_logits = logits[universe_ids]
    if min_p is None:
        candidate_logits = universe_logits
        candidate_ids = universe_ids
    else:
        min_p_mask = universe_logits >= universe_logits.max() + math.log(min_p)
        candidate_logits = universe_logits[min_p_mask]
        candidate_ids = torch.nonzero(min_p_mask, as_tuple=False).flatten() if universe_ids is None else universe_ids[min_p_mask]
    if top_p is not None:
        log_normalizer = torch.logsumexp(universe_logits, dim=0)
        excluded_mass = 1.0 - torch.exp(candidate_logits - log_normalizer).sum()
        order = torch.argsort(candidate_logits)
        candidate_logits = candidate_logits[order]
        candidate_ids = order if candidate_ids is None else candidate_ids[order]
        keep = excluded_mass + torch.exp(candidate_logits - log_normalizer).cumsum(dim=0) > 1.0 - top_p
        keep[-1] = True
        candidate_logits = candidate_logits[keep]
        candidate_ids = candidate_ids[keep]
    elif candidate_ids is None:
        return torch.softmax(candidate_logits, dim=-1)
    probabilities = torch.zeros_like(logits)
    probabilities[candidate_ids] = torch.softmax(candidate_logits, dim=-1)
    return probabilities


def _runner_and_sequence(top_k, top_p, min_p):
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=257), max_model_len=512)
    runner.enforce_eager = False
    runner._speculative_sampling_graphs = {}
    runner.use_triton_sampling = False
    runner._logits_bias_cache = {}
    runner._repetition_token_cache = {}
    sequence = SimpleNamespace(
        seq_id=1,
        token_ids=[1, 2, 3],
        num_tokens=3,
        repetition_penalty_start=3,
        predictive_penalty=True,
        logits_processor=None,
        logits_bias=None,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
    )
    sample_params = (torch.tensor([0.6]), None, None, None, None, None)
    return runner, sequence, sample_params


@torch.inference_mode()
def test_speculative_distribution_matches_dynamic_reference():
    source = torch.randn(257, generator=torch.Generator().manual_seed(1234))
    for top_k, top_p, min_p in ((None, 0.9, 0.05), (20, 0.9, 0.05), (None, None, 0.05), (None, 0.9, None)):
        runner, sequence, sample_params = _runner_and_sequence(top_k, top_p, min_p)
        expected = _reference_distribution(source / 0.6, top_k, top_p, min_p)
        actual = runner._speculative_distribution(sequence, source, sample_params, [])
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


@torch.inference_mode()
def test_speculative_distribution_records_sampled_survivor_telemetry_without_changing_output():
    source = torch.randn(257, generator=torch.Generator().manual_seed(1234))
    runner, sequence, sample_params = _runner_and_sequence(None, 0.9, 0.05)
    expected = runner._speculative_distribution(sequence, source, sample_params, [])
    slot = {
        "distribution_counts": torch.empty((5, 2), dtype=torch.int64),
        "distribution_masses": torch.empty(5, dtype=torch.float32),
    }
    profile = {"slot": slot, "distributions": []}

    actual = runner._speculative_distribution(sequence, source, sample_params, [], profile=profile, profile_role="draft0")

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert profile["distributions"] == [{"role": "draft0", "predictive": False, "vocab": 257, "top_p": 0.9, "min_p": 0.05}]
    scaled = source.float().div(0.6)
    masked = scaled.masked_fill(scaled < scaled.max() + math.log(0.05), float("-inf"))
    log_normalizer = torch.logsumexp(scaled, dim=0)
    excluded_mass = 1.0 - torch.exp(masked - log_normalizer).sum()
    sorted_logits = torch.sort(masked).values
    keep = excluded_mass + torch.exp(sorted_logits - log_normalizer).cumsum(dim=0) > 0.1
    sorted_logits[:-1].masked_fill_(~keep[:-1], float("-inf"))
    assert int(slot["distribution_counts"][0, 0]) == int(torch.isfinite(masked).sum())
    assert int(slot["distribution_counts"][0, 1]) == int(torch.isfinite(sorted_logits).sum())
    torch.testing.assert_close(slot["distribution_masses"][0], excluded_mass.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@torch.inference_mode()
def test_speculative_sampling_graph_matches_eager_ties_and_releases_cache():
    runner, sequence, params = _runner_and_sequence(20, .95, .05)
    params = (params[0].cuda(), *params[1:])
    generator = torch.Generator(device="cuda").manual_seed(891)
    for vocab in [257, 98304]:
        for _ in range(12):
            logits = torch.randn(vocab, device="cuda", dtype=torch.bfloat16, generator=generator)
            runner.enforce_eager = True
            expected = runner._speculative_distribution(sequence, logits, params, [])
            runner.enforce_eager = False
            actual = runner._speculative_distribution(sequence, logits, params, [])
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert 0 < len(runner._speculative_sampling_graphs) <= 8
    runner._graph_cache = {}
    runner.clear_graph_cache()
    assert not runner._speculative_sampling_graphs


def test_speculative_statistics_follow_requested_capacity():
    runner = ModelRunner.__new__(ModelRunner)
    for count in [2, 3, 4, 6, 8]:
        runner._max_speculative_draft_tokens = count
        stats = runner._new_speculative_stats()
        assert len(stats["drafted_by_position"]) == count
        assert len(stats["accepted_by_position"]) == count


def test_logits_processor_can_declare_input_ids_unused():
    sequence = SimpleNamespace(token_ids=list(range(32_000)))

    def processor(input_ids, logits):
        assert input_ids is None
        return logits.add_(1)

    processor._requires_input_ids = False
    logits = torch.zeros((1, 8))
    result = ModelRunner._call_logits_processor(sequence, processor, logits)
    torch.testing.assert_close(result, torch.ones_like(logits))


def test_qwen_processors_declare_input_ids_unused():
    model = SimpleNamespace(_prompt_enhancer_penalty_mode="none")
    processor, _update_state = _build_prompt_logits_processor(model, thinking_enabled=False, suppress_token_ids=(3,))
    assert processor._requires_input_ids is False
    assert processor._supports_partial_vocab() is True
    logits = torch.zeros((1, 8))
    processed = ModelRunner._call_logits_processor(SimpleNamespace(token_ids=list(range(32_000))), processor, logits)
    assert torch.isneginf(processed[0, 3])


def test_partial_vocab_qwen_processor_does_not_expand_draft_logits():
    model = SimpleNamespace(_prompt_enhancer_penalty_mode="none")
    processor, _update_state = _build_prompt_logits_processor(model, thinking_enabled=False, suppress_token_ids=(3, 200))
    runner, sequence, _sample_params = _runner_and_sequence(None, 0.9, 0.05)
    sequence.logits_processor = processor
    logits = torch.zeros(64)
    processed = runner._apply_speculative_logit_rules(sequence, logits, 1.0, [], vocab_size=64, predictive=True)
    assert processed.shape == logits.shape
    assert torch.isneginf(processed[3])


def test_rejection_sampling_keeps_mtp_drafts_on_device_until_verification():
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = SimpleNamespace(mtp=SimpleNamespace(draft_vocab_size=4, get_cache_length=lambda: 9))
    runner._speculative_drafts = {1: {"logits": torch.zeros(4), "hidden_states": torch.zeros((1, 1, 2))}}
    runner._sampling_generator = torch.Generator().manual_seed(1)
    seen_virtual_tokens = []

    def distribution(_self, _seq, _logits, _sample_params, virtual_tokens, _vocab_size=None, predictive=False, profile=None, profile_role="target"):
        assert predictive is True
        assert profile is None
        seen_virtual_tokens.append(virtual_tokens)
        return torch.tensor([0.0, 1.0, 0.0, 0.0])

    def mtp_forward(_self, input_ids, _positions, hidden_states, last_logits_only=False):
        assert input_ids.shape == (1, 1)
        assert input_ids.device == hidden_states.device
        assert last_logits_only is True
        return hidden_states, torch.zeros((1, 1, 4))

    runner._speculative_distribution = MethodType(distribution, runner)
    runner._run_mtp_forward = MethodType(mtp_forward, runner)
    sequence = SimpleNamespace(seq_id=1, top_k=50, top_p=0.9, min_p=0.05, predictive_penalty=False, logits_processor=None)
    draft_tokens, cache_length, draft_distributions = runner._build_mtp_drafts(sequence, (None, None, None, None, None, None), 2, 10)

    assert torch.equal(draft_tokens, torch.tensor([1, 1]))
    assert cache_length == 9
    assert len(draft_distributions) == 2
    assert seen_virtual_tokens == [(), ()]


@torch.inference_mode()
def test_batched_speculative_state_commit_matches_individual_copies():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modules = [_FakeLinearAttention(device), _FakeLinearAttention(device)]
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = SimpleNamespace(blk=modules)
    runner._max_speculative_draft_tokens = 2
    runner._speculative_commit_destinations = []
    runner._speculative_commit_sources = {}
    runner._prepare_target_speculative_state()

    for verified_tokens in (2, 3):
        for processed_tokens in range(1, verified_tokens + 1):
            for module_index, module in enumerate(modules):
                module.speculative_conv_state_buffer.fill_(float("nan"))
                module.speculative_recurrent_state_buffer.fill_(float("nan"))
                for prefix in range(1, verified_tokens):
                    module.speculative_conv_state_buffer[prefix - 1].fill_(100 * module_index + 10 * prefix + 1)
                    module.speculative_recurrent_state_buffer[prefix - 1].fill_(100 * module_index + 10 * prefix + 2)
                module.conv_state_buffer.fill_(100 * module_index + 10 * verified_tokens + 1)
                module.recurrent_state_buffer.fill_(100 * module_index + 10 * verified_tokens + 2)
            runner._commit_speculative_target_state(processed_tokens, verified_tokens)
            for module_index, module in enumerate(modules):
                assert torch.all(module.conv_state_buffer == 100 * module_index + 10 * processed_tokens + 1)
                assert torch.all(module.recurrent_state_buffer == 100 * module_index + 10 * processed_tokens + 2)


def test_speculative_commit_cache_contains_only_original_buffers_and_views():
    modules = [_FakeLinearAttention("cpu"), _FakeLinearAttention("cpu")]
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = SimpleNamespace(blk=modules)
    runner._max_speculative_draft_tokens = 2
    runner._speculative_commit_destinations = []
    runner._speculative_commit_sources = {}
    runner._prepare_target_speculative_state()

    expected_destinations = [tensor for module in modules for tensor in (module.conv_state_buffer, module.recurrent_state_buffer)]
    assert [tensor.data_ptr() for tensor in runner._speculative_commit_destinations] == [tensor.data_ptr() for tensor in expected_destinations]
    for processed_tokens, sources in runner._speculative_commit_sources.items():
        expected_sources = [tensor for module in modules for tensor in (module.speculative_conv_state_buffer[processed_tokens - 1], module.speculative_recurrent_state_buffer[processed_tokens - 1])]
        assert [tensor.data_ptr() for tensor in sources] == [tensor.data_ptr() for tensor in expected_sources]
        assert all(source.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr() for source, expected in zip(sources, expected_sources))


def test_mtp_advance_batches_verified_states_and_uses_matching_graph_token():
    runner = ModelRunner.__new__(ModelRunner)
    runner.enforce_eager = False
    runner.mtp_graph = object()
    runner.mtp_graph_vars = {
        "input_ids": torch.zeros((1, 1), dtype=torch.int64),
        "positions": torch.zeros(1, dtype=torch.int64),
        "next_token": torch.tensor([7], dtype=torch.int64),
    }
    runner.mtp_refresh_graphs = {2: {"input_ids": torch.zeros((1, 2), dtype=torch.int64), "next_token": torch.tensor([17], dtype=torch.int64)}}
    runner._speculative_drafts = {}
    calls = []

    def mtp_forward(_self, input_ids, positions, hidden_states, last_logits_only=False):
        calls.append((input_ids.data_ptr(), positions.data_ptr(), input_ids.clone(), positions.clone()))
        assert last_logits_only is True
        return hidden_states, torch.zeros((1, 1, 8))

    runner._run_mtp_forward = MethodType(mtp_forward, runner)
    hidden_states = torch.zeros((1, 2, 4))
    runner._advance_mtp(SimpleNamespace(seq_id=3), [4, 5], torch.tensor([10, 11]), hidden_states)

    assert len(calls) == 1
    assert calls[0][0] == runner.mtp_refresh_graphs[2]["input_ids"].data_ptr()
    assert calls[0][2].tolist() == [[4, 5]]
    assert calls[0][3].tolist() == [10, 11]
    assert int(runner._speculative_drafts[3]["next_token"]) == 17


def test_mtp_pending_uses_scalar_text_position_after_multimodal_prefill():
    runner = ModelRunner.__new__(ModelRunner)
    runner._speculative_drafts = {3: {"unused": torch.tensor(1)}}
    runner._speculative_pending = {}
    positions = torch.tensor([[[0, 40]], [[0, 40]], [[0, 40]]])

    runner._store_speculative_pending(SimpleNamespace(seq_id=3), torch.zeros((2, 8)), torch.zeros((1, 2, 4)), positions)

    pending = runner._speculative_pending[3]
    assert 3 not in runner._speculative_drafts
    assert pending["positions"].shape == (1,)
    assert int(pending["positions"][0]) == 40


def test_speculative_verify_reuses_graph_metadata_buffers(monkeypatch):
    class FakeSequence:
        token_ids = list(range(300))
        block_table = [4, 7]
        last_block_num_tokens = 12
        last_token = 9
        position_offset = 5

        def __len__(self):
            return len(self.token_ids)

    graph_vars = {
        "input_ids": torch.zeros(3, dtype=torch.int64),
        "positions": torch.zeros(3, dtype=torch.int64),
        "slot_mapping": torch.zeros(3, dtype=torch.int32),
        "cu_seqlens_q": torch.tensor([0, 3], dtype=torch.int32),
        "cu_seqlens_k": torch.zeros(2, dtype=torch.int32),
        "block_tables": torch.full((1, 8), -1, dtype=torch.int32),
        "outputs": torch.ones((1, 3, 4)),
    }
    runner = ModelRunner.__new__(ModelRunner)
    runner.enforce_eager = False
    runner.block_size = 256
    runner.speculative_graph_vars = {3: graph_vars}
    runner._cpu_speculative_input_ids = torch.zeros(3, dtype=torch.int64)
    runner._cpu_block_tables = torch.zeros((1, 8), dtype=torch.int32)
    runner._get_runtime_device = MethodType(lambda _self: torch.device("cpu"), runner)
    recorded_context = {}
    monkeypatch.setattr("shared.llm_engines.nanovllm.engine.model_runner.set_context", lambda *args, **kwargs: recorded_context.update(args=args, kwargs=kwargs))

    input_ids, positions = runner._prepare_speculative_verify(FakeSequence(), torch.tensor([11, 12]))

    assert input_ids.data_ptr() == graph_vars["input_ids"].data_ptr()
    assert positions.data_ptr() == graph_vars["positions"].data_ptr()
    assert torch.equal(input_ids, torch.tensor([9, 11, 12]))
    assert torch.equal(positions, torch.tensor([304, 305, 306]))
    assert torch.equal(graph_vars["slot_mapping"], torch.tensor([1803, 1804, 1805], dtype=torch.int32))
    assert torch.equal(graph_vars["cu_seqlens_k"], torch.tensor([0, 302], dtype=torch.int32))
    assert torch.equal(graph_vars["block_tables"], torch.tensor([[4, 7, -1, -1, -1, -1, -1, -1]], dtype=torch.int32))
    assert graph_vars["block_table_signature"] == (4, 7)
    assert recorded_context["args"][1].data_ptr() == graph_vars["cu_seqlens_q"].data_ptr()


def test_speculative_verify_list_drafts_do_not_reuse_single_sequence_decode_buffer(monkeypatch):
    class FakeSequence:
        token_ids = list(range(300))
        block_table = [4, 7]
        last_block_num_tokens = 12
        last_token = 9
        position_offset = 5

        def __len__(self):
            return len(self.token_ids)

    graph_vars = {
        "input_ids": torch.zeros(3, dtype=torch.int64),
        "positions": torch.zeros(3, dtype=torch.int64),
        "slot_mapping": torch.zeros(3, dtype=torch.int32),
        "cu_seqlens_q": torch.tensor([0, 3], dtype=torch.int32),
        "cu_seqlens_k": torch.zeros(2, dtype=torch.int32),
        "block_tables": torch.full((1, 8), -1, dtype=torch.int32),
    }
    runner = ModelRunner.__new__(ModelRunner)
    runner.enforce_eager = False
    runner.block_size = 256
    runner.speculative_graph_vars = {3: graph_vars}
    runner._cpu_input_ids = torch.zeros(1, dtype=torch.int64)
    runner._cpu_speculative_input_ids = torch.zeros(3, dtype=torch.int64)
    runner._cpu_block_tables = torch.zeros((1, 8), dtype=torch.int32)
    monkeypatch.setattr("shared.llm_engines.nanovllm.engine.model_runner.set_context", lambda *args, **kwargs: None)

    input_ids, _positions = runner._prepare_speculative_verify(FakeSequence(), [11, 12])

    assert runner._cpu_input_ids.numel() == 1
    assert torch.equal(input_ids, torch.tensor([9, 11, 12]))
