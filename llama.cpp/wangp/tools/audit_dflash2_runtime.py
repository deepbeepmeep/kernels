"""Process-local DFlash2 profiling/ablation wrapper; does not alter app defaults.

Pass the ordinary benchmark_qwen38_engine.py arguments after these audit flags.
Always compare throughput without --stage-every; events perturb timing.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage-every", type=int, default=0)
    parser.add_argument("--upstream-proposal", action="store_true")
    parser.add_argument("--legacy-acceptance", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.stage_every < 0:
        parser.error("--stage-every must be nonnegative")
    sys.argv = [sys.argv[0], *remaining]
    import torch
    import benchmark_qwen38_engine as benchmark
    from shared.llm_engines.nanovllm.engine.block_draft_runner import BlockDraftRunner
    from shared.llm_engines.nanovllm.engine.model_runner import ModelRunner

    counts = {"proposals": 0, "gpu_chain": 0, "gpu_acceptance": 0, "acceptance_fallbacks": 0}
    original_build = BlockDraftRunner._build_dflash_drafts
    original_gpu = BlockDraftRunner._build_dflash_gpu_chain
    original_accept = BlockDraftRunner._sample_verified_block
    if args.legacy_acceptance:
        BlockDraftRunner._disable_dflash_gpu_acceptance = True

    def accept(self, *pos, **kw):
        before = getattr(self, "_dflash_gpu_acceptance_rounds", 0)
        before_fallback = getattr(self, "_dflash_acceptance_fallbacks", 0)
        result = original_accept(self, *pos, **kw)
        counts["gpu_acceptance"] += getattr(self, "_dflash_gpu_acceptance_rounds", 0) - before
        counts["acceptance_fallbacks"] += getattr(self, "_dflash_acceptance_fallbacks", 0) - before_fallback
        return result

    def build(self, *pos, **kw):
        counts["proposals"] += 1
        return original_build(self, *pos, **kw)

    def gpu(self, *pos, **kw):
        counts["gpu_chain"] += 1
        return original_gpu(self, *pos, **kw)

    BlockDraftRunner._build_dflash_drafts = build
    BlockDraftRunner._build_dflash_gpu_chain = gpu
    BlockDraftRunner._sample_verified_block = accept
    if args.upstream_proposal:
        # SGLang's selector samples softmax(scores / T), without target
        # top-p/min-p filtering. The target sampler remains unchanged.
        BlockDraftRunner._compact_draft_distribution = (
            lambda self, seq, scores, temperature: torch.softmax(scores.float() / temperature, dim=-1)
        )
    if args.stage_every:
        original_start = ModelRunner._start_mtp_stage_profile

        def start(self):
            if not self._mtp_profile_enabled:
                return None
            n = getattr(self, "_audit_profile_calls", 0) + 1
            self._audit_profile_calls = n
            self._mtp_profile_passes = 63 if n % args.stage_every == 0 else 0
            return original_start(self)

        ModelRunner._start_mtp_stage_profile = start
    output = Path(remaining[remaining.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    try:
        benchmark.main()
    finally:
        (output / "audit.json").write_text(json.dumps({**vars(args), **counts}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
