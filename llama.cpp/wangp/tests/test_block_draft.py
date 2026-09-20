"""Block-draft state and cached-attention contracts, independent of checkpoints."""
import pytest
import torch
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config

from shared.llm_engines.nanovllm.models.block_draft import BlockDraft


def draft_model(method, device):
    config = Qwen3Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                        vocab_size=64, use_sliding_window=method == "dflash2",
                        sliding_window=16, max_window_layers=2)
    config.markov_rank = 8
    config.dflash_config = dict(block_size=8, mask_token_id=63, target_layer_ids=[0, 2],
                               conv_kernel_size=2, conv_group_size=16, selector_rank=8, selector_top_k=4)
    with torch.device(device):
        model = BlockDraft(config, method)
        for name, parameter in model.named_parameters():
            nn.init.normal_(parameter, std=0.02)
            if name.endswith("norm.weight"):
                nn.init.ones_(parameter)
        for layer in model.layers:
            if layer.attention_conv is not None:
                for conv in (layer.attention_conv, layer.mlp_conv):
                    conv.base_kernel.zero_()
                    conv.base_kernel[:, 0].fill_(1)
    return model.to(dtype=torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("method", ["dspark", "dflash2"])
@torch.inference_mode()
def test_cached_attention_graph_growth_rewind_and_restore(method, monkeypatch):
    from shared.llm_engines.nanovllm.layers import attention
    torch.manual_seed(612)
    model = draft_model(method, "cuda")
    embedding = nn.Embedding(64, 128, device="cuda", dtype=torch.bfloat16)
    output = nn.Linear(128, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    model.prepare_cache(80, "cuda", torch.bfloat16)
    features = torch.randn(1, 40, 256, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(40, device="cuda")
    model.append_context(features[:, :24], positions[:24])
    snapshot = model.snapshot_sequence_state()
    assert snapshot["keys"].device.type == "cpu"

    def dense_attention(q, kc, vc, *, k, v, cache_seqlens, causal, window_size):
        length = int(cache_seqlens.item())
        all_k = torch.cat((kc[:, :length], k), dim=1)
        all_v = torch.cat((vc[:, :length], v), dim=1)
        mask = None
        if window_size[0] >= 0:
            query_positions = torch.arange(length, length + q.shape[1], device=q.device)
            key_positions = torch.arange(all_k.shape[1], device=q.device)
            mask = key_positions[None, :] >= query_positions[:, None] - window_size[0]
        return F.scaled_dot_product_attention(q.transpose(1, 2).float(),
                    all_k.transpose(1, 2).float(), all_v.transpose(1, 2).float(),
                    attn_mask=mask, enable_gqa=True).transpose(1, 2).to(q.dtype)

    eager = model(3, 24, embedding, output)[1].clone()
    with monkeypatch.context() as context:
        context.setattr(attention, "flash_attn_with_kvcache", dense_attention)
        reference = model(3, 24, embedding, output)[1]
    torch.testing.assert_close(eager, reference, atol=0.015, rtol=0.02)
    captured = model.propose(3, 24, embedding, output)[1].clone()
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)
    graph = model._proposal_graph
    model.append_context(features[:, 24:], positions[24:])
    changed = model.propose(7, 40, embedding, output)[1].clone()
    torch.testing.assert_close(changed, model(7, 40, embedding, output)[1], atol=0, rtol=0)
    assert model._proposal_graph is graph
    assert not torch.equal(changed, captured)
    model.truncate_cache(24)
    torch.testing.assert_close(model.propose(3, 24, embedding, output)[1], captured, atol=0, rtol=0)
    model.reset_sequence_state()
    model.append_context(features[:, :12], positions[:12])
    model.restore_sequence_state(snapshot)
    torch.testing.assert_close(model.propose(3, 24, embedding, output)[1], captured, atol=0, rtol=0)
    model.release_sequence_state()
    assert model._proposal_graph is None and model._keys is None


@pytest.mark.parametrize("top_k", [1, 4])
@torch.inference_mode()
def test_dflash2_ends_proposal_when_grammar_excludes_candidate_set(top_k):
    from shared.llm_engines.nanovllm.engine.block_draft_runner import BlockDraftRunner
    runner = object.__new__(BlockDraftRunner)
    drafter = SimpleNamespace(method="dflash2", get_cache_length=lambda: 12,
        propose=lambda *args: (torch.zeros(2, 8), torch.zeros(2, 16)),
        proposal_candidates=lambda *args: (torch.full((4,), -torch.inf), torch.arange(4)))
    runner.model = SimpleNamespace(mtp=drafter, token_embd=None, output=None,
                                  _prompt_enhancer_speculative_confidence=0.3)
    runner._apply_speculative_logit_rules = lambda seq, logits, *args, **kwargs: logits
    seq = SimpleNamespace(last_token=2, top_k=top_k)
    tokens, length, distributions = runner._build_mtp_drafts(seq, (None,) * 6, 2, 12)
    assert tokens == [] and length == 12
    assert distributions == ([] if top_k != 1 else None)


@pytest.mark.parametrize("top_k", [0, 4, 16, 20])
@pytest.mark.parametrize("top_p,min_p", [(None, None), (0.9, None), (0.95, 0.05), (None, 0.2)])
@torch.inference_mode()
def test_sparse_draft_distribution_matches_dense_reference(top_k, top_p, min_p):
    from shared.llm_engines.nanovllm.engine.block_draft_runner import BlockDraftRunner
    runner = object.__new__(BlockDraftRunner)
    runner.use_triton_sampling = False
    scores = torch.linspace(-4.1, 2.7, 16, device="cpu")
    ids = torch.arange(16, device="cpu") * 13 + 2
    seq = SimpleNamespace(top_k=top_k, top_p=top_p, min_p=min_p)
    compact = runner._compact_draft_distribution(seq, scores, 0.7)
    dense = torch.full((248320,), -torch.inf, device="cpu")
    dense[ids] = scores / 0.7
    if top_k:
        threshold = dense.topk(top_k).values[-1]
        dense[dense < threshold] = -torch.inf
    reference = runner._filter_speculative_distribution(dense, dense, None, top_p, min_p)
    torch.testing.assert_close(compact, reference[ids], atol=1e-7, rtol=1e-6)
    assert torch.isfinite(compact).all() and compact.sum().item() == pytest.approx(1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("method", ["dspark", "dflash2"])
@torch.inference_mode()
def test_append_graph_matches_eager_and_does_not_overwrite_rewound_prefix(method):
    model = draft_model(method, "cuda")
    model.prepare_cache(80, "cuda", torch.bfloat16)
    features = torch.randn(1, 16, 256, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(50, 66, device="cuda")
    for count in (1, 3, 7):
        model.reset_sequence_state()
        model.append_context(features[:, :9], positions[:9])
        model._append_context_tensors(features[:, 9:9+count], positions[9:9+count])
        expected_keys = model._keys[:, :, :9+count].clone()
        expected_values = model._values[:, :, :9+count].clone()
        model.append_context(features[:, 9:9+count], positions[9:9+count])
        torch.testing.assert_close(model._keys[:, :, :9+count], expected_keys, atol=0, rtol=0)
        torch.testing.assert_close(model._values[:, :, :9+count], expected_values, atol=0, rtol=0)
        graph = model._append_graphs[count][0]
        model.truncate_cache(9)
        model.append_context(features[:, 9:9+count] * 2, positions[9:9+count] + 100)
        assert model._append_graphs[count][0] is graph
        torch.testing.assert_close(model._keys[:, :, :9], expected_keys[:, :, :9], atol=0, rtol=0)
        assert not torch.equal(model._keys[:, :, 9:9+count], expected_keys[:, :, 9:9+count])
    model.release_sequence_state()
    assert not model._append_graphs and model._append_graph_pool is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("top_k", [1, 20])
@torch.inference_mode()
def test_dflash_gpu_chain_matches_eager_and_replays_changed_bias(top_k):
    from shared.llm_engines.nanovllm.engine.block_draft_runner import BlockDraftRunner
    runner = object.__new__(BlockDraftRunner)
    runner.model = SimpleNamespace(mtp=draft_model("dflash2", "cuda"))
    runner.use_triton_sampling = False
    runner._logits_bias_cache = {}
    runner._sampling_generator = torch.Generator(device="cuda").manual_seed(38)
    seq = SimpleNamespace(top_k=top_k, top_p=0.95, min_p=0.05, temperature=0.7,
                          last_token=3, logits_bias=None, logits_processor=None, predictive_penalty=False)
    hidden = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda")
    unary = torch.randn(3, 64, dtype=torch.float32, device="cuda")
    tokens, _, probabilities = runner._build_dflash_gpu_chain(seq, 3, hidden, unary)
    state = next(iter(runner._dflash_sampling_graphs.values()))
    expected_tokens, expected_probs, length = runner._dflash_gpu_chain(seq, hidden, unary,
        state[3], state[4], state[5])
    torch.testing.assert_close(tokens, expected_tokens)
    if top_k != 1:
        for actual, expected in zip(probabilities, expected_probs):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert length.item() == 3
    graph = state[0]
    seq.logits_bias = torch.full((64,), -torch.inf, device="cuda")
    blocked, _, _ = runner._build_dflash_gpu_chain(seq, 3, hidden * 2, unary + 1)
    assert blocked.numel() == 0
    seq.logits_bias = None
    seq.last_token = 7
    restored, _, _ = runner._build_dflash_gpu_chain(seq, 3, hidden * 2, unary + 1)
    assert restored.numel() == 3
    assert next(iter(runner._dflash_sampling_graphs.values()))[0] is graph
    forced = int(unary[0].argmax().item())
    def force_close(_ids, logits):
        logits.fill_(-torch.inf)
        logits[:, forced] = 0
        return logits
    force_close._is_token_mask = True
    seq.logits_processor = force_close
    seq.logits_bias = torch.full((64,), -torch.inf, device="cuda")
    forced_tokens, _, _ = runner._build_dflash_gpu_chain(seq, 3, hidden, unary)
    assert forced_tokens.numel() >= 1 and int(forced_tokens[0].item()) == forced
