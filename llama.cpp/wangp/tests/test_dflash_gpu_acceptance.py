from types import SimpleNamespace

import pytest
import torch

from shared.llm_engines.nanovllm.engine.dflash_sampling import apply_rules, target_probabilities, accept_block
from shared.prompt_enhancer.qwen35_text import _build_prompt_logits_processor
from test_nanovllm_speculative_sampling import _reference_distribution


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_p,min_p", [(None, None), (.95, .05), (.5, None), (None, .1)])
def test_batched_distribution_matches_reference_or_flags_ambiguous_ties(device, dtype, top_p, min_p):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator(device=device).manual_seed(135)
    logits = torch.randn((32, 257), generator=generator, device=device, dtype=dtype)
    probabilities, unsafe = target_probabilities(logits, 20, top_p, min_p, .6)
    for i in range(len(logits)):
        if not unsafe[i]:
            expected = _reference_distribution(logits[i].float() / .6, 20, top_p, min_p)
            torch.testing.assert_close(probabilities[i], expected, atol=2e-7, rtol=2e-6)


def test_top_k_support_overflow_is_not_silently_truncated():
    _, unsafe = target_probabilities(torch.ones((2, 257)), 20, .95, .05, .6)
    assert unsafe.all()


@pytest.mark.parametrize("valid_length", [0, 1, 3])
@pytest.mark.parametrize("stop", [-1, 1, 2])
@pytest.mark.parametrize("greedy", [False, True])
def test_acceptance_matches_sequential_reference(valid_length, stop, greedy):
    drafts = torch.tensor([1, 2, 3])
    p = torch.tensor([[.1, .7, .1, .1], [.1, .1, .7, .1], [.4, .1, .1, .4], [.1, .2, .6, .1]])
    q = torch.tensor([[.1, .7, .1, .1], [.1, .1, .7, .1], [.1, .1, .1, .7]])
    uniforms = torch.tensor([.2, .8, .9])
    noise = torch.tensor([.2, .7, .4, .8])
    stops = torch.arange(4) == stop
    emitted, accepted, visited = [], 0, 0
    for i in range(valid_length):
        if greedy:
            chosen = int(p[i].argmax())
            ok = chosen == int(drafts[i])
        else:
            ok = uniforms[i] < min(1., float(p[i, drafts[i]] / q[i, drafts[i]]))
            chosen = int(drafts[i]) if ok else int(((p[i] - q[i]).clamp_min(0) / noise).argmax())
        visited += 1
        emitted.append(chosen)
        if not ok:
            break
        accepted += 1
        if chosen == stop:
            break
    else:
        emitted.append(int(p[valid_length].argmax()) if greedy else int((p[valid_length] / noise).argmax()))
    packed = accept_block(p, torch.zeros(4, dtype=torch.bool), drafts, q, torch.tensor(valid_length), uniforms, noise, stops, greedy).tolist()
    assert packed[:3] == [len(emitted), accepted, visited]
    assert packed[5:5 + len(emitted)] == emitted


@pytest.mark.parametrize("remaining", [0, 1, 4])
@pytest.mark.parametrize("close_position", [-1, 0, 2])
def test_batched_thinking_and_repetition_match_sequential_rules(remaining, close_position):
    model = SimpleNamespace(_prompt_enhancer_penalty_mode="repetition", _prompt_enhancer_close_think_token_id=5,
                            _prompt_enhancer_stop_token_ids=(7,), _prompt_enhancer_thinking_max_tokens=4)
    processor, update = _build_prompt_logits_processor(model, thinking_enabled=True, suppress_token_ids=(6,))
    for _ in range(4 - remaining):
        update(1)
    rules = processor._speculative_batch_rules()
    drafts = torch.tensor([1, 1, 2])
    if close_position >= 0:
        drafts[close_position] = 5
    logits = torch.arange(-4., 4.).expand(4, -1).clone()
    history = torch.arange(8) == 1
    bias = torch.zeros(8)
    actual = apply_rules(logits, drafts, history, bias, torch.arange(8) == 6, torch.arange(8) == 7,
                         torch.tensor(rules["thinking"]), 1.05)
    seen = {1}
    for row in range(4):
        expected = logits[row].clone()
        ids = list(seen)
        expected[ids] = torch.where(expected[ids] < 0, expected[ids] * 1.05, expected[ids] / 1.05)
        expected = processor(None, expected.unsqueeze(0))[0]
        torch.testing.assert_close(actual[row], expected)
        if row < 3:
            update(int(drafts[row]))
            seen.add(int(drafts[row]))


def test_presence_processor_is_not_advertised_as_batch_safe():
    model = SimpleNamespace(_prompt_enhancer_penalty_mode="presence", _prompt_enhancer_presence_penalty=1.)
    processor, _ = _build_prompt_logits_processor(model, thinking_enabled=False)
    assert not hasattr(processor, "_speculative_batch_rules")


def test_fallback_choice_is_independent_of_acceptance_random_numbers():
    p = torch.tensor([[.1, .9], [.5, .5], [.1, .9]])
    q = torch.tensor([[.8, .2], [.5, .5]])
    drafts = torch.tensor([0, 1])
    unsafe = torch.tensor([False, False, True])
    for uniforms in (torch.zeros(2), torch.ones(2)):
        packed = accept_block(p, unsafe, drafts, q, torch.tensor(2), uniforms,
                              torch.ones(2), torch.zeros(2, dtype=torch.bool))
        assert packed[3] == 1


def test_rejection_preserves_target_distribution():
    generator = torch.Generator(device="cpu").manual_seed(7231)
    p = torch.tensor([.1, .25, .4, .25])
    q = torch.tensor([.6, .1, .1, .2])
    draws = 6000
    proposals = torch.multinomial(q, draws, replacement=True, generator=generator)
    uniforms = torch.rand(draws, 1, generator=generator)
    noise = torch.empty(draws, 4).exponential_(generator=generator)
    counts = torch.zeros(4)
    for i in range(draws):
        packed = accept_block(p.expand(2, -1), torch.zeros(2, dtype=torch.bool), proposals[i:i+1],
                              q[None], torch.tensor(1), uniforms[i], noise[i], torch.zeros(4, dtype=torch.bool))
        counts[packed[5]] += 1
    torch.testing.assert_close(counts / draws, p, atol=.022, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("method", ["dflash", "mtp"])
@torch.inference_mode()
def test_warm_block_has_no_scalar_reads_or_dynamic_candidate_selection(method):
    from shared.llm_engines.nanovllm.engine.block_draft_runner import BlockDraftRunner
    from shared.llm_engines.nanovllm.engine.model_runner import ModelRunner
    from shared.llm_engines.nanovllm.engine.sequence import Sequence
    from shared.llm_engines.nanovllm.sampling_params import SamplingParams
    runner = object.__new__(BlockDraftRunner if method == "dflash" else ModelRunner)
    runner.model = SimpleNamespace(mtp=SimpleNamespace(method="dflash2"))
    runner.config = SimpleNamespace(eos=0, max_num_seqs=1, hf_config=SimpleNamespace(vocab_size=129))
    runner.use_triton_sampling = True
    runner._logits_bias_cache = {}
    runner._repetition_token_cache = {}
    runner._sampling_generator = torch.Generator(device="cuda").manual_seed(5)
    runner._dflash_valid_length = torch.tensor(3, device="cuda")
    runner.speculative_stats = dict(drafted=0, accepted=0, drafted_by_position=[0]*3, accepted_by_position=[0]*3)
    seq = Sequence([1, 2], SamplingParams(temperature=.6, top_k=20, top_p=.95, min_p=.05, repetition_penalty=1.05))
    seq.append_token(4)
    logits = torch.linspace(-5, 5, 129, device="cuda").repeat(4, 1)
    drafts = torch.tensor([128, 127, 126], device="cuda")
    q = list(torch.softmax(logits[:3], dim=-1).unbind())
    runner._sample_verified_block(seq, logits, drafts, q, None)
    state = next(iter(runner._speculative_acceptance_graphs.values()))
    graph = state["graph"]
    if method == "mtp":
        # Reuse the graph with both full and restricted draft vocabularies.
        drafts = torch.tensor([63, 62, 61], device="cuda")
        q = [torch.softmax(row[:64], dim=-1) for row in logits[:3]]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        emitted, _ = runner._sample_verified_block(seq, logits, drafts, q, None)
    keys = {event.key for event in profile.key_averages()}
    assert "aten::_local_scalar_dense" not in keys
    assert "aten::nonzero" not in keys
    assert emitted and getattr(runner, f"_{method}_gpu_acceptance_rounds") == 2
    assert next(iter(runner._speculative_acceptance_graphs.values()))["graph"] is graph

    if method == "mtp":
        assert state["q"][:, 64:].count_nonzero() == 0
        torch.testing.assert_close(state["q"][:, :64], torch.stack(q))
        mixed_q = [q[0], torch.softmax(logits[1], dim=-1), q[2]]
        runner._sample_verified_block(seq, logits, drafts, mixed_q, None)
        torch.testing.assert_close(state["q"][1], mixed_q[1])
        runner._sample_verified_block(seq, logits, drafts, q, None)
        assert state["q"][:, 64:].count_nonzero() == 0
        runner._speculative_sampling_graphs = {}
        runner._graph_cache = {}
        runner.clear_graph_cache()
        assert not runner._speculative_acceptance_graphs
        return

    # No valid proposals still emits one exact target sample, using row zero.
    runner._dflash_valid_length.zero_()
    emitted, accepted = runner._sample_verified_block(seq, logits, drafts, q, None)
    assert len(emitted) == 1 and accepted == 0


@pytest.mark.parametrize("triton,eager,top_k,allowed", [(False, False, 20, False), (True, True, 20, False),
                                                        (True, False, 0, False), (True, False, 129, False),
                                                        (True, False, 20, True)])
def test_shared_acceptance_backend_and_support_guard(triton, eager, top_k, allowed):
    from shared.llm_engines.nanovllm.engine.speculative_sampling import can_batch_acceptance
    runner = SimpleNamespace(use_triton_sampling=triton, enforce_eager=eager)
    seq = SimpleNamespace(top_k=top_k, logits_processor=None, logits_processor_update_state=None)
    assert can_batch_acceptance(runner, seq) == allowed
    seq.logits_processor = lambda ids, scores: scores
    assert not can_batch_acceptance(runner, seq)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_native_draft_filter_reuses_graph_without_sync_and_retains_each_q():
    from shared.llm_engines.nanovllm.engine.speculative_sampling import draft_probabilities
    runner = SimpleNamespace(_speculative_sampling_graphs={})
    seq = SimpleNamespace(top_k=20, top_p=.95, min_p=.05, temperature=.6)
    logits = torch.linspace(-5, 5, 257, device="cuda")
    first = draft_probabilities(runner, seq, logits)
    expected = first.clone()
    # Overflowing ties are permitted for proposals, with normalized returned q.
    logits.fill_(1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        second = draft_probabilities(runner, seq, logits)
    keys = {event.key for event in profile.key_averages()}
    assert "aten::_local_scalar_dense" not in keys and "aten::nonzero" not in keys
    assert len(runner._speculative_sampling_graphs) == 1
    torch.testing.assert_close(first, expected)
    assert second.isfinite().all()
    torch.testing.assert_close(second.sum(), torch.ones((), device="cuda"))
