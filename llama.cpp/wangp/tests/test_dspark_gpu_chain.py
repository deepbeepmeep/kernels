"""DSpark greedy graphs retain confidence exits; sampled decoding is unchanged."""
from types import SimpleNamespace

import pytest
import torch

from shared.llm_engines.nanovllm.engine.block_draft_runner import BlockDraftRunner
from test_block_draft import draft_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("threshold", [0., .3, .6, .5000000000000001])
@torch.inference_mode()
def test_dspark_chain_matches_sequential_scores_and_confidence(threshold):
    torch.manual_seed(712)
    runner = object.__new__(BlockDraftRunner)
    model = draft_model("dspark", "cuda")
    model.confidence_head.proj.weight.zero_()
    model.confidence_head.proj.bias.zero_()
    runner.model = SimpleNamespace(mtp=model, _prompt_enhancer_speculative_confidence=threshold, _block_draft=True)
    runner._logits_bias_cache = {}
    runner._speculative_sampling_graphs = {}
    runner.use_triton_sampling, runner.enforce_eager = True, False
    runner._sampling_generator = torch.Generator(device="cuda").manual_seed(38)
    seq = SimpleNamespace(top_k=1, top_p=.95, min_p=.05, temperature=.7,
                          last_token=3, logits_bias=None, logits_processor=None, predictive_penalty=False)
    hidden = torch.randn(5, 128, dtype=torch.bfloat16, device="cuda")
    unary = torch.randn(5, 64, dtype=torch.float32, device="cuda")
    before_rng = runner._sampling_generator.get_state()
    tokens, _, probabilities = runner._build_dspark_gpu_chain(seq, 5, hidden, unary)
    assert probabilities is None
    after_rng = runner._sampling_generator.get_state()
    runner._sampling_generator.set_state(before_rng)
    state = next(iter(runner._dspark_sampling_graphs.values()))
    previous, expected = seq.last_token, []
    for index in range(5):
        scores, confidence = model.proposal_logits(hidden[index], unary[index], previous)
        if index > 0 and threshold > 0 and confidence.item() < threshold:
            break
        previous = int(scores.argmax().item())
        expected.append(previous)
    assert tokens.tolist() == expected
    assert torch.equal(runner._sampling_generator.get_state(), after_rng)
    assert len(expected) == (1 if threshold > .5 else 5)
    graph = state[0]
    seq.logits_bias = torch.full((64,), -torch.inf, device="cuda")
    blocked, _, _ = runner._build_dspark_gpu_chain(seq, 5, hidden * 2, unary + 1)
    assert blocked.numel() == 0
    assert torch.equal(runner._sampling_generator.get_state(), after_rng)
    seq.logits_bias = None
    seq.last_token = 7
    restored, _, _ = runner._build_dspark_gpu_chain(seq, 5, hidden * 2, unary + 1)
    assert restored.numel() == len(expected)
    assert next(iter(runner._dspark_sampling_graphs.values()))[0] is graph

    def force_close(_ids, logits):
        logits.fill_(-torch.inf)
        logits[:, 9] = 0
        return logits
    force_close._is_token_mask = True
    seq.logits_processor = force_close
    seq.logits_bias = torch.full((64,), -torch.inf, device="cuda")
    forced, _, _ = runner._build_dspark_gpu_chain(seq, 5, hidden, unary)
    assert forced.tolist() == [9] * len(expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("top_k,use_triton,eager", [(20, True, False), (1, False, False), (1, False, True)])
@torch.inference_mode()
def test_sampled_and_non_vllm_paths_keep_original_dispatch(monkeypatch, top_k, use_triton, eager):
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
    monkeypatch.setattr(runner, "_build_dspark_gpu_chain", lambda *args: pytest.fail("Greedy vLLM graph used outside its supported path"))
    seq = SimpleNamespace(top_k=top_k, top_p=.95, min_p=.05, temperature=.7, repetition_penalty=1.,
                          last_token=3, logits_bias=None, logits_processor=None,
                          logits_processor_update_state=None, predictive_penalty=False)
    params = (torch.tensor([seq.temperature], device="cuda"), None, None, None, None, None)
    tokens, _, probabilities = runner._build_mtp_drafts(seq, params, 3, 0)
    assert isinstance(tokens, list) and len(tokens) == 3
    assert (probabilities is None) == (top_k == 1)
