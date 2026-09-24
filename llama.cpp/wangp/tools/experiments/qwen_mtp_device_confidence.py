"""Benchmark-only batching of two native MTP confidence readbacks.

Install on an existing vLLM runner with ``restore, counters = install(runner)``.
The target still receives the original zero, one, or two proposals. Only the
draft confidence readbacks are combined; target sampling and rollback are not
patched. No production module imports this experiment.
"""

from types import MethodType, SimpleNamespace

import torch


def _fallback_reason(runner, seq, sample_params, draft_count, threshold):
    if not runner.use_triton_sampling or runner.enforce_eager:
        return "mode"
    if getattr(torch.version, "hip", None) is not None:
        return "backend"
    if getattr(runner.model, "_block_draft", False):
        return "block_draft"
    if seq.top_k != 1 or draft_count != 2 or not threshold > 0.0:
        return "shape_or_sampling"
    if sample_params[-1] is not None and seq.predictive_penalty:
        return "predictive_repetition"
    processor = runner._speculative_logits_processor(seq, predictive=True)
    if processor is not None:
        # These built-in masks read state; only accepted target tokens update
        # it. Calling a custom/stateful processor for an unused second draft
        # would not be equivalent when the first confidence fails.
        if (getattr(processor, "_requires_input_ids", True)
                or not getattr(processor, "_is_token_mask", False)
                or processor.__module__ != "shared.prompt_enhancer.qwen35_text"):
            return "processor"
    if not runner._speculative_drafts[seq.seq_id]["logits"].is_cuda:
        return "device"
    if getattr(runner, "mtp_graph", None) is None:
        return "missing_graph"
    return None


def _confidence_pair(runner, seq, logits, draft_vocab_size):
    # Keep the exact reduction, casts and rounding of _build_mtp_drafts. The
    # gate excludes any rule that uses the hypothetical first token here.
    processed_logits = runner._apply_speculative_logit_rules(
        seq, logits, 1.0, [], draft_vocab_size, predictive=True)
    top_logit, top_token = torch.max(processed_logits, dim=0)
    confidence = torch.exp(top_logit.float() - torch.logsumexp(processed_logits.float(), dim=0))
    return top_token.reshape(1), torch.stack((top_token.float(), confidence))


def _capture_rule_key(runner, seq):
    processor = runner._speculative_logits_processor(seq, predictive=True)
    if processor is None:
        return ()
    rules_fn = getattr(seq.logits_processor, "_speculative_batch_rules", None)
    if not callable(rules_fn):
        return None
    rules = rules_fn()
    close, remaining, active = rules["thinking"]
    phase = 0 if not active else (1 if remaining > 0 else 2)
    # Only the phase changes the Python branches recorded by these built-in
    # masks. Keying the remaining count would capture a graph every token.
    return (tuple(rules["suppressed"]), tuple(rules["thinking_stops"]), close, phase)


def _captured_pairs(runner, seq, start_position, counters):
    source = runner._speculative_drafts[seq.seq_id]
    mtp = runner.model.mtp
    bias = runner._get_logits_bias(seq, source["logits"].unsqueeze(0))
    rule_key = _capture_rule_key(runner, seq)
    assert rule_key is not None
    key = (id(runner.mtp_graph), id(mtp._cache), source["logits"].device,
           source["logits"].dtype, tuple(source["logits"].shape), rule_key,
           None if bias is None else (bias.dtype, tuple(bias.shape)))
    cache = getattr(runner, "_benchmark_device_confidence_graphs", None)
    if cache is None:
        cache = runner._benchmark_device_confidence_graphs = {}
    entry = cache.get(key)
    if entry is None:
        if len(cache) >= 4:
            cache.pop(next(iter(cache)))
        entry = dict(logits=source["logits"].clone(),
                     hidden=source["hidden_states"].clone(),
                     positions=torch.full_like(runner.mtp_graph_vars["positions"], start_position),
                     bias=None if bias is None else bias.clone())
        # Sequence.__getstate__ deliberately omits sampling attributes, so do
        # not use copy.copy(seq) here. Keep captured mask-index owners alive.
        capture_seq = SimpleNamespace(**seq.__dict__)
        capture_seq.logits_bias = entry["bias"]
        entry["seq"] = capture_seq

        def compute():
            first_token, first_pair = _confidence_pair(
                runner, capture_seq, entry["logits"], mtp.draft_vocab_size)
            _hidden, logits = mtp(first_token.view(1, 1), entry["positions"],
                                 entry["hidden"], last_logits_only=True, cache_prepared=True)
            _second_token, second_pair = _confidence_pair(
                runner, capture_seq, logits[0, -1], mtp.draft_vocab_size)
            return torch.stack((first_pair, second_pair))

        # Warmup/capture writes only the first unused MTP cache slot. With
        # cache_prepared=True neither call advances its logical length.
        length = mtp.get_cache_length()
        mtp._cache.prepare_append()
        stream = torch.cuda.Stream(device=source["logits"].device)
        stream.wait_stream(torch.cuda.current_stream(source["logits"].device))
        with torch.cuda.stream(stream):
            compute()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            packed = compute()
        torch.cuda.current_stream(source["logits"].device).wait_stream(stream)
        assert mtp.get_cache_length() == length
        entry.update(graph=graph, packed=packed)
        cache[key] = entry
        counters["graph_captures"] += 1
    else:
        counters["graph_cache_hits"] += 1
    entry["logits"].copy_(source["logits"])
    entry["hidden"].copy_(source["hidden_states"])
    entry["positions"].fill_(start_position)
    if bias is not None:
        entry["bias"].copy_(bias)
    mtp._cache.prepare_append()
    entry["graph"].replay()
    mtp._cache.advance(1)
    return entry["packed"]


def _batch_two(runner, seq, start_position, threshold, counters, capture=False):
    draft_state = runner._speculative_drafts[seq.seq_id]
    draft_vocab_size = getattr(runner.model.mtp, "draft_vocab_size", None)
    cache_length = runner.model.mtp.get_cache_length()
    advanced = False
    try:
        if capture:
            packed = _captured_pairs(runner, seq, start_position, counters)
            advanced = True
        else:
            first_token, first_pair = _confidence_pair(
                runner, seq, draft_state["logits"], draft_vocab_size)
            # first_token is a separate torch.max output, not the mutable
            # graph's next_token buffer overwritten by the following replay.
            draft_position = runner.mtp_graph_vars["positions"]
            draft_position.fill_(start_position)
            _hidden, logits = runner._run_mtp_forward(
                first_token.view(1, 1), draft_position, draft_state["hidden_states"],
                last_logits_only=True)
            advanced = True
            _second_token, second_pair = _confidence_pair(
                runner, seq, logits[0, -1], draft_vocab_size)
            packed = torch.stack((first_pair, second_pair))
        first, second = packed.tolist()
        counters["confidence_readbacks"] += 1
        # Compare Python floats exactly as the reference does, including NaN
        # semantics and thresholds between adjacent FP32 values.
        if first[1] < threshold:
            tokens = []
            # The existing zero-draft path skips the normal MTP truncate.
            # Undo this experiment's otherwise-unused speculative append.
            runner.model.mtp.truncate_cache(cache_length)
            advanced = False
            counters["extra_mtp_forwards"] += 1
            counters["reference_confidence_readbacks"] += 1
        else:
            tokens = [int(first[0])]
            if not second[1] < threshold:
                tokens.append(int(second[0]))
            counters["reference_confidence_readbacks"] += 2
        counters[f"proposed_length_{len(tokens)}"] += 1
        return tokens, cache_length, None
    except Exception:
        if advanced:
            runner.model.mtp.truncate_cache(cache_length)
        raise


def install(runner, capture=False):
    """Return an idempotent restore callable and mutable scalar counters.

    Unsupported calls use the saved original method. Clear counters only by
    setting their existing values to zero, or subtract before/after snapshots.
    capture=True keeps at most four extra graph entries on this test runner
    across install/restore. Release the runner after the benchmark.
    """
    marker = "_benchmark_device_confidence_installed"
    if getattr(runner, marker, False):
        raise RuntimeError("Device-confidence experiment is already installed")
    original = runner._build_mtp_drafts
    owned_original = runner.__dict__.get("_build_mtp_drafts")
    had_override = "_build_mtp_drafts" in runner.__dict__
    counters = dict(calls=0, eligible_rounds=0, fallback_rounds=0,
                    confidence_readbacks=0, reference_confidence_readbacks=0,
                    extra_mtp_forwards=0, proposed_length_0=0,
                    proposed_length_1=0, proposed_length_2=0,
                    graph_captures=0, graph_cache_hits=0)

    def build(self, seq, sample_params, draft_count, start_position, profile=None):
        counters["calls"] += 1
        threshold = float(getattr(self.model, "_prompt_enhancer_speculative_confidence", 0.0))
        reason = _fallback_reason(self, seq, sample_params, draft_count, threshold)
        if reason is None and capture and _capture_rule_key(self, seq) is None:
            reason = "capture_rules"
        if reason is not None:
            counters["fallback_rounds"] += 1
            key = f"fallback_{reason}"
            counters[key] = counters.get(key, 0) + 1
            return original(seq, sample_params, draft_count, start_position, profile=profile)
        counters["eligible_rounds"] += 1
        return _batch_two(self, seq, start_position, threshold, counters, capture=capture)

    runner._build_mtp_drafts = MethodType(build, runner)
    setattr(runner, marker, True)
    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        if had_override:
            runner._build_mtp_drafts = owned_original
        else:
            del runner._build_mtp_drafts
        delattr(runner, marker)
        restored = True

    return restore, counters
