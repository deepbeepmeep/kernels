"""DSpark drafts run as one CUDA graph for greedy and sampled decoding.

Confidence exits shorten the valid prefix instead of synchronizing after each
draft; sampled drafts come exactly from the filtered q used for acceptance.
"""
from types import SimpleNamespace

import pytest
import torch

from shared.llm_engines.nanovllm.engine.block_draft_runner import BlockDraftRunner
from shared.llm_engines.nanovllm.engine.dflash_sampling import target_probabilities
from test_block_draft import draft_model


def _runner(threshold, seed=38):
    runner = object.__new__(BlockDraftRunner)
    model = draft_model("dspark", "cuda")
    runner.model = SimpleNamespace(mtp=model, _prompt_enhancer_speculative_confidence=threshold, _block_draft=True)
    runner._logits_bias_cache = {}
    runner._speculative_sampling_graphs = {}
    runner.use_triton_sampling, runner.enforce_eager = True, False
    runner._sampling_generator = torch.Generator(device="cuda").manual_seed(seed)
    return runner, model


def _seq(top_k):
    return SimpleNamespace(top_k=top_k, top_p=.95, min_p=.05, temperature=.7,
                           last_token=3, logits_bias=None, logits_processor=None, predictive_penalty=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("threshold", [0., .3, .6, .5000000000000001])
@torch.inference_mode()
def test_greedy_chain_matches_sequential_scores_and_confidence(threshold):
    torch.manual_seed(712)
    runner, model = _runner(threshold)
    model.confidence_head.proj.weight.zero_()
    model.confidence_head.proj.bias.zero_()
    seq = _seq(1)
    hidden = torch.randn(5, 128, dtype=torch.bfloat16, device="cuda")
    unary = torch.randn(5, 64, dtype=torch.float32, device="cuda")
    before_rng = runner._sampling_generator.get_state()
    tokens, _, probabilities = runner._build_dspark_gpu_chain(seq, 5, hidden, unary)
    assert probabilities is None and tokens.numel() == 5
    assert torch.equal(runner._sampling_generator.get_state(), before_rng)
    previous, expected = seq.last_token, []
    for index in range(5):
        scores, confidence = model.proposal_logits(hidden[index], unary[index], previous)
        if index > 0 and threshold > 0 and confidence.item() < threshold:
            break
        previous = int(scores.argmax().item())
        expected.append(previous)
    assert len(expected) == (1 if threshold > .5 else 5)
    assert int(runner._draft_valid_length) == len(expected)
    assert tokens[:len(expected)].tolist() == expected
    graph = next(iter(runner._dspark_sampling_graphs.values()))[0]
    seq.logits_bias = torch.full((64,), -torch.inf, device="cuda")
    runner._build_dspark_gpu_chain(seq, 5, hidden * 2, unary + 1)
    assert int(runner._draft_valid_length) == 0
    seq.logits_bias = None
    seq.last_token = 7
    runner._build_dspark_gpu_chain(seq, 5, hidden * 2, unary + 1)
    assert int(runner._draft_valid_length) == len(expected)
    assert next(iter(runner._dspark_sampling_graphs.values()))[0] is graph

    def force_close(_ids, logits):
        logits.fill_(-torch.inf)
        logits[:, 9] = 0
        return logits
    force_close._is_token_mask = True
    seq.logits_processor = force_close
    seq.logits_bias = torch.full((64,), -torch.inf, device="cuda")
    forced, _, _ = runner._build_dspark_gpu_chain(seq, 5, hidden, unary)
    assert forced[:int(runner._draft_valid_length)].tolist() == [9] * len(expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_sampled_chain_draws_from_the_distribution_it_reports():
    torch.manual_seed(9)
    runner, model = _runner(0.)
    seq = _seq(20)
    hidden = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda")
    unary = torch.randn(3, 64, dtype=torch.float32, device="cuda") * 2
    scores, _ = model.proposal_logits(hidden[0], unary[0], seq.last_token)
    expected = target_probabilities(scores[None], 20, seq.top_p, seq.min_p, seq.temperature)[0][0]
    draws, counts = 4000, torch.zeros(64, device="cuda")
    for _ in range(draws):
        tokens, _, probabilities = runner._build_dspark_gpu_chain(seq, 3, hidden, unary)
        counts[tokens[0]] += 1
        # Each later draft is drawn from its own reported q, conditioned on the previous draft.
        assert all(float(q[token]) > 0 for q, token in zip(probabilities, tokens.tolist()))
    torch.testing.assert_close(probabilities[0], expected)
    assert len(runner._dspark_sampling_graphs) == 1 and int(runner._draft_valid_length) == 3
    torch.testing.assert_close(counts / draws, expected, atol=.03, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("top_k,use_triton,eager,gpu", [(20, True, False, True), (1, True, False, True), (1, False, False, False), (20, True, True, False)])
@torch.inference_mode()
def test_dispatch_uses_graph_chain_only_on_the_vllm_path(monkeypatch, top_k, use_triton, eager, gpu):
    runner = object.__new__(BlockDraftRunner)
    model = draft_model("dspark", "cuda")
    hidden = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda")
    unary = torch.randn(3, 64, dtype=torch.float32, device="cuda")
    monkeypatch.setattr(model, "propose", lambda *args: (hidden, unary))
    runner.model = SimpleNamespace(mtp=model, token_embd=None, output=None, _block_draft=True,
                                   _prompt_enhancer_speculative_confidence=0.)
    runner.use_triton_sampling, runner.enforce_eager = use_triton, eager
    runner._logits_bias_cache, runner._speculative_sampling_graphs = {}, {}
    runner._sampling_generator = torch.Generator(device="cuda").manual_seed(12)
    if not gpu:
        monkeypatch.setattr(runner, "_build_dspark_gpu_chain", lambda *args: pytest.fail("graph chain used outside the vLLM path"))
    seq = SimpleNamespace(top_k=top_k, top_p=.95, min_p=.05, temperature=.7, repetition_penalty=1.,
                          last_token=3, logits_bias=None, logits_processor=None,
                          logits_processor_update_state=None, predictive_penalty=False)
    params = (torch.tensor([seq.temperature], device="cuda"), None, None, None, None, None)
    tokens, _, probabilities = runner._build_mtp_drafts(seq, params, 3, 0)
    assert torch.is_tensor(tokens) == gpu and len(tokens) == 3
    assert (probabilities is None) == (top_k == 1)
