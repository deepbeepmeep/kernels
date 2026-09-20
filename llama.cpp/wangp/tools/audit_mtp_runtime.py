"""A/B benchmark of production MTP GPU acceptance and draft filtering.

Without --gpu-acceptance/--gpu-draft, use the retained reference paths.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-acceptance", action="store_true")
    parser.add_argument("--gpu-draft", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    import torch
    import benchmark_qwen38_engine as benchmark
    from shared.llm_engines.nanovllm.engine.model_runner import ModelRunner
    counters = dict(gpu_acceptance_rounds=0, exact_fallbacks=0)
    original_init = ModelRunner.__init__
    original_accept = ModelRunner._sample_verified_block
    original_build = ModelRunner._build_mtp_drafts

    def init(self, *pos, **kw):
        original_init(self, *pos, **kw)
        if getattr(self.model, "_block_draft", False):
            raise ValueError("This audit is for native MTP")
        self._disable_mtp_gpu_acceptance = not args.gpu_acceptance
        self._disable_mtp_gpu_draft = not args.gpu_draft

        def accept(self, seq, logits, drafts, distributions, params, profile=None):
            started = time.perf_counter()
            if args.timing_only and profile is not None:
                profile["slot"]["events"][5].record(torch.cuda.current_stream())
            before = getattr(self, "_mtp_gpu_acceptance_rounds", 0)
            fallback_before = getattr(self, "_mtp_acceptance_fallbacks", 0)
            result = original_accept(self, seq, logits, drafts, distributions, params, None if args.timing_only else profile)
            counters["gpu_acceptance_rounds"] += getattr(self, "_mtp_gpu_acceptance_rounds", 0) - before
            counters["exact_fallbacks"] += getattr(self, "_mtp_acceptance_fallbacks", 0) - fallback_before
            if args.timing_only:
                self._mark_mtp_stage_profile(profile, 6, "sampling", started)
            return result

        def build(self, seq, params, count, start, profile=None):
            return original_build(self, seq, params, count, start, profile=None if args.timing_only else profile)

        self._sample_verified_block = MethodType(accept, self)
        self._build_mtp_drafts = MethodType(build, self)

    ModelRunner.__init__ = init
    if args.timing_only:
        original_start = ModelRunner._start_mtp_stage_profile
        def start(self):
            if self._mtp_profile_enabled:
                n = getattr(self, "_audit_passes", 0) + 1
                self._audit_passes = n
                self._mtp_profile_passes = 63 if n % 8 == 0 else 0
            return original_start(self)
        ModelRunner._start_mtp_stage_profile = start
    output = Path(remaining[remaining.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    try:
        benchmark.main()
    finally:
        (output / "mtp_audit.json").write_text(json.dumps({**vars(args), **counters}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
