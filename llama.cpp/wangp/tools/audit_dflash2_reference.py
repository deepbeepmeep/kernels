"""Compare real cached DFlash2 proposals with a pinned publisher reference.

The external module and its license are supplied explicitly. Its parameters alias
the loaded drafter; target features, embedding and output head are held constant.
This is a correctness diagnostic, not a throughput benchmark.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmgp import offload
from shared.prompt_enhancer.qwen35_assistant_runtime import Qwen35AssistantRuntime
from shared.prompt_enhancer.qwen35_text import load_qwen35_text_prompt_enhancer
from shared.qtypes.gguf import get_gguf_compute_dtype
from shared.utils import files_locator


def difference(actual, reference):
    a, b = actual.float(), reference.float()
    return dict(exact=torch.equal(actual, reference), max_abs=(a-b).abs().max().item(),
                rms=(a-b).square().mean().sqrt().item(),
                relative_rms=((a-b).square().mean() / b.square().mean()).sqrt().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("dflash_publisher_reference", args.reference)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    files_locator.set_checkpoints_paths([str(args.checkpoint.parent.parent), str(args.checkpoints_root), "ckpts", "."])
    ids = json.loads(args.prompt_ids.read_text(encoding="utf-8"))
    dtype = get_gguf_compute_dtype()
    model = load_qwen35_text_prompt_enhancer(model_path=str(args.checkpoint), assets_dir=str(args.checkpoint.parent),
        default_dtype=dtype, backend="gguf", requested_lm_engine="vllm", variant="27b",
        speculative_decoding="dflash2", kv_cache_int8=True)
    model._prompt_enhancer_min_model_len_hint = 4096
    manager = offload.profile({"llm": model}, profile_no=1, budgets={"llm": 0}, pinnedMemory=False,
        quantizeTransformer=False, convertWeightsFloatTo=dtype, verboseLevel=1)
    runtime = Qwen35AssistantRuntime(model)
    draft = model.mtp
    collected = torch.empty((1, 4096, draft.fc.in_features), dtype=draft.fc.weight.dtype, device="cpu")
    original_append, original_propose = draft.append_context, draft.propose
    original_convolve = reference_module._grouped_dynamic_convolve
    records, calls, reference = [], 0, None

    def append(self, features, positions):
        start, end = self._length, self._length + features.shape[1]
        collected[:, start:end].copy_(features.detach().to("cpu"))
        return original_append(features, positions)

    def local_convolve(hidden, dynamic, base, group_size):
        batch, length, width = hidden.shape
        groups, taps = width // group_size, base.shape[0]
        coefficients = base.reshape(1, 1, taps, groups, group_size) + dynamic.unsqueeze(-1)
        grouped = hidden.reshape(batch, length, groups, group_size)
        out = coefficients[:, :, 0] * grouped
        for tap in range(1, taps):
            out[:, tap:] += coefficients[:, tap:, tap] * grouped[:, :-tap]
        return out.reshape_as(hidden)

    def propose(self, anchor, start_position, embedding, output):
        nonlocal calls, reference
        hidden, logits = original_propose(anchor, start_position, embedding, output)
        if calls in (0, 8, 16):
            if reference is None:
                # Independently honor the publisher's modern RoPE schema. Do
                # not share the local rotary module: that would hide a loader
                # error in the very model being checked.
                reference_config = copy.deepcopy(self.config)
                rope = reference_config.rope_parameters
                reference_config.rope_theta = rope["rope_theta"]
                reference_config.rope_scaling = {k: v for k, v in rope.items() if k != "rope_theta"}
                with torch.device("meta"):
                    reference = reference_module.DFlash2DraftModel(reference_config)
                weights = {name + ".weight" if name in ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook") else name: value
                           for name, value in self.state_dict().items() if not name.startswith("rotary_emb.")}
                reference.load_state_dict(weights, strict=True, assign=True)
                reference.rotary_emb = reference_module.Qwen3RotaryEmbedding(reference_config, device="cpu").to(hidden.device)
                reference.eval()
                torch.testing.assert_close(self.rotary_emb.inv_freq, reference.rotary_emb.inv_freq, rtol=0, atol=0)
            assert self._length == start_position
            block_ids = torch.full((1, self.block_size), self.mask_token_id, device=hidden.device, dtype=torch.long)
            block_ids[0, 0] = anchor
            noise = embedding(block_ids).to(self.fc.weight.dtype)
            features = collected[:, :self._length].to(hidden.device)
            positions = torch.arange(start_position + self.block_size, device=hidden.device).unsqueeze(0)
            record = dict(proposal=calls, context=start_position)
            for name, convolution in (("publisher", original_convolve), ("publisher_with_local_convolution", local_convolve)):
                reference_module._grouped_dynamic_convolve = convolution
                expected = reference(position_ids=positions, noise_embedding=noise, target_hidden=features)[:, 1:]
                expected_logits = output(expected)[0].float()
                record[name] = dict(hidden=difference(hidden, expected[0]), logits=difference(logits, expected_logits),
                    same_top1=(logits.argmax(-1) == expected_logits.argmax(-1)).tolist(),
                    top16_overlap=[len(set(a) & set(b)) for a,b in zip(logits.topk(16).indices.tolist(), expected_logits.topk(16).indices.tolist())])
            reference_module._grouped_dynamic_convolve = original_convolve
            records.append(record)
            report = dict(reference=str(args.reference), reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                          checkpoint=str(args.checkpoint), prompt_sha256=hashlib.sha256(args.prompt_ids.read_bytes()).hexdigest(), records=records)
            (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(record), flush=True)
        calls += 1
        return hidden, logits

    draft.append_context = types.MethodType(append, draft)
    draft.propose = types.MethodType(propose, draft)
    try:
        with torch.inference_mode():
            runtime.prime_context(ids, seed=123)
            runtime.generate_segment(max_new_tokens=96, max_total_tokens=96, seed=123, do_sample=True,
                temperature=.6, top_p=.95, top_k=20, thinking_enabled=True)
            assert len(records) == 3
    finally:
        draft.append_context, draft.propose = original_append, original_propose
        model.unload()
        manager.release()


if __name__ == "__main__":
    main()
