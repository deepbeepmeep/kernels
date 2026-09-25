# Qwen3.8 Q4: GPU sampling for Deepy actions and exact-scale Q8 prefill (25 September 2026)

Target: Qwen3.8-27B-Uncensored Q4_K_M, the configured Deepy setup (`llm_engines.deepy=qwen38_27b`, GGUF, `vllm` resolved engine, Q8 KV cache, four MTP drafts). RTX 5090 / SM120, Windows, py311 (PyTorch 2.10.0+cu130, Triton 3.6.0). Constraints for this round: no precision loss, no additional VRAM, no GPU-specific tuning that cannot be validated beyond this card.

## Outcome

| Change | Effect | Evidence |
| --- | --- | --- |
| Install GGUF kernels 1.0.23 in py311 (was 1.0.22) | 20K decode 102–111 → ~131 tok/s; 2K 107–112 → 113–119 (top-k 20 benchmark protocol). Identical emitted tokens and MTP statistics. | Separate processes, same prompts |
| Exact-scale Q8 prefix prefill (all CUDA GPUs) | Prefill +1.5% at 2K, +12% at 8K, +11% at 20K, +19% at 30K; lower error vs FP32; same peak VRAM | Same-process alternating A/B |
| Deepy actions use GPU block acceptance (Qwen's recommended top-k 20, as the prompt enhancer already used; `top_k=None` also supported exactly) | Per-cycle decode cost −5.5% at 2K, −1.5% at 20K on the real thought-action path | Separate-process before/after |

Absolute rates vary about ±10% between processes (clocks, sampling luck: 176–213 target passes per 512 tokens), so decode gains are reported as per-cycle cost where possible.

## 1. Deepy actions bypassed GPU acceptance

Live Deepy decoding (`Qwen35AssistantRuntime.generate_action`) never used the shared GPU block acceptance (`engine/speculative_sampling.py`) or the fixed-shape MTP draft filter, because:

1. Deepy actions ran with no top-k (the prompt enhancer already defaulted to 20), and both paths required `1 <= top_k <= 128`.
2. Action logits processors (thinking budget) lacked `_speculative_batch_rules`, which `can_batch_acceptance` requires. The prompt-enhancer processor already exposed them.

The benchmark (`generate_segment`, top-k 20) therefore measured a faster path than live Deepy used. `tools/benchmark_qwen38_engine.py --deepy-action` now measures the real action path; `--top-k/--top-p`, `--reference-sampling` and `--ab-sampling` were added, and results record `gpu_acceptance_rounds` / `acceptance_fallbacks`.

Changes:

- `qwen35_assistant_runtime._install_action_processors`: when presence penalty is disabled (the default repetition mode), the thinking-budget processor publishes the same batch rules as the prompt path (`suppressed=()`, thinking stops, `(close, remaining, in_thinking)`).
- `dflash_sampling.target_probabilities(top_k=None)`: exact nucleus/min-p support using the top `NUCLEUS_CAPACITY=1024` logits and the full-vocabulary normalizer. A token's nucleus decision depends only on the mass ranked above it, so support inside the capacity is exact; the row is flagged for the reference sampler if the first excluded token survives min-p and the retained mass is below top-p (+1e-4 margin), or ties the lowest kept value. `bounded_support()` gates both acceptance and draft filtering; unbounded full-softmax sampling keeps the reference path.
- `start_generation_action` resolves top-k through `qwen35_text._resolve_prompt_top_k`, so Deepy actions use Qwen's recommended 20 (`QWEN35_PROMPT_DEFAULT_TOP_K`) like the prompt enhancer. This is not exposed as a setting: Qwen recommends 20 for every mode, and temperature 0.6 vs 1.0 did not change MTP tokens per target pass (2.4–2.8 in all tested configurations). An explicit 0 still disables it internally.

Measured: `topk` over 5×248,320 costs ~200 µs regardless of k, so `top_k=None` with capacity 1,024 costs about the same as top-k 20. About 5–6% of rounds fall back (BF16 tie straddling the nucleus boundary, same guard as before), costing one extra reference evaluation. Same-process A/B (top-k off): per-cycle time ~23.9 → ~23.4 ms at 2K; the sampling tail is a small share of the cycle.

## 2. Exact-scale Q8 prefix prefill

The Q8 prefix-prefill kernel was 47% of GPU time for a 1,024-token suffix over 20K (12.9 ms/layer on the SM120 cubin). It dequantized K/V to FP32 and split them into high/low pairs: 2 QK + 4 PV tensor-core products per tile.

INT8 values are exact in FP16/BF16, and the scales are per 32 head dimensions, so:

```text
QK[i, j] = sum_b ks[j, b] * (Q[i, b] . Kq[j, b])          exact products, FP32 accumulation
O[i, b]  = sum_j (64 * P[i, j] * vs[j, b]) * Vq[j, b] / 64 P*vs as a high/low pair, Vq exact
```

This is 1 QK + 2 PV products and removes the K/V rounding residual. The exact 2^6 factor keeps FP16 residuals out of subnormals: without it FP16 error was up to 3.4× worse than the old kernel; with it FP16 and BF16 are at or below the old error in every tested case. It cannot overflow FP16: a scale derived from FP16 values is at most 65504/127.

| Case, relative L2 vs FP32 reference | Old high/low | New |
| --- | ---: | ---: |
| FP16 D256, 1,024 queries over 20K | 2.10e-4 | 1.70e-4 |
| FP16 D256, ragged short | 3.97e-5 | 2.63e-5 |
| FP16 D128 / D64 | 9.50e-5 / 2.00e-5 | 7.49e-5 / 9.57e-6 |
| BF16 D256 20K / short | 6.52e-4 / 1.52e-4 | 4.40e-4 / 8.92e-5 |
| BF16 D128 / D64 | 3.39e-4 / 2.88e-4 | 1.82e-4 / 1.88e-4 |

Isolated kernel, BF16 D256, 24/4 heads: 1,024 queries over 20K took 14.04 ms (SM120 cubin), 16.56 ms (shared Triton), **9.00 ms** (new); over 2K 1.22 / 1.35 / **0.77** ms. Config 32×32 tile, 4 warps, 2 stages: 50 KB shared memory, 255 registers, 8-byte spill. It fits Turing's 64 KB. Larger tiles exceeded 99 KB or regressed.

Integration: `layers/attention.py` appends `q8_prefill_exact_kernel` after all existing kernels (preserving line-number cache keys) and `_q8_paged_prefill` uses it on every CUDA GPU in vllm mode; the SM120 prefill cubin is no longer called. ROCm keeps the established high/low kernel (not validated here). legacy/cg never reach this path. One compiled variant per geometry/dtype; lengths and table widths stay runtime values. Decode/verification grouped attention is unchanged.

Real model, same process, alternating order, three prompts each (median prefill tok/s): 2,048: 4,378 → 4,444; 8,000: 3,638 → 4,066; 20,000: 3,013 → 3,346; 30,000: 2,508 → 2,987. Peak allocation identical (17.21 GiB in the harness). Final-position target logits after prefill: same argmax in all 7 comparisons, KL 1.6e-7 to 9.6e-4 nats (2K–30K).

Tests: `test_exact_prefill_is_at_least_as_accurate_as_high_low` (new, FP16/BF16, D64/128/256, long and ragged) asserts the new error does not exceed the old kernel's. Shape-range and SM120 tests now track `q8_prefill_exact_kernel`; the removed test compared the no-longer-used prefill cubin with the shared kernel.

## 3. Investigated, not retained

- **MMVQ→MMQ threshold at 5 rows.** L2-resident microbenchmarks suggested MMQ was 1.8× faster for 5-row Q4_K. Cache-cold (24 layers cycled in a CUDA graph) the two were within ±10%, with MMVQ better on small matrices. No change.
- **Multi-column Q4_K GEMV** (`tools/experiments/q4k_multicolumn_gemv`): decodes each weight block once for all verification columns, 16-byte weight loads, exact activation block sums. Matches the dense reference (≈5.6e-3 rel. L2, the inherent Q8 activation error; native MMQ at 8 rows measured 1.2–1.5e-2). Cache-cold at 5 columns it was faster on some matrices (up to ~30%) and slower on others, with the best rows-per-warp varying by shape: roughly 10–15% of Q4_K verify time, ~5–7% end-to-end. Rejected under the portability constraint: a new native kernel whose tuning could only be validated on this card.
- **Context.** Decode is GPU-bound: per cycle ~22 ms of GPU time vs ~23.5 ms wall. 5-row Q4_K/Q6_K MMVQ take ~11.5/4.0 ms at ~1.0 TB/s cache-cold; a plain FP16 read on this card reaches ~1.3 TB/s, so the practical headroom there is ~30%, not the spec-sheet 1.79 TB/s. The four MTP draft-head matvecs (98K×5120 Q6_K) cost ~1.3 ms per cycle.
- Applying the exact-scale form to decode grouped attention: attention is ~1–7% of decode GPU time; not pursued.

## Reproduction

```powershell
$py = 'C:\Users\Marc\anaconda3\envs\py311\python.exe'
$assets = 'E:\ML\wan2gp\ckpts\Qwen3_8_27B_Uncensored'; $corpus = 'D:\AMD\cuda-fusions-20260922\corpus.txt'
& $py tools\benchmark_qwen38_engine.py --assets $assets --corpus $corpus --output out\after --contexts 2048 20000 --repeats 3 --tokens 512 --draft 4 --no-stage-profile --no-memory-poll --top-k 20 --top-p 0.9 --deepy-action
& $py tools\benchmark_qwen38_engine.py ... --top-k 0 --top-p 0.9 --deepy-action --reference-sampling   # former live Deepy path
& $py tools\experiments\qwen38_prefill_exact_ab.py --assets $assets --corpus $corpus --output out\prefill_ab.json
& $py -m pytest -q tests/test_q8_paged_attention.py tests/test_dflash_gpu_acceptance.py tests/test_triton_shape_ranges.py tests/test_q8_sm120_attention.py
```

Regression run: 343 passed; `test_detailed_telemetry_requires_environment_opt_in` fails as previously documented (environment read cached at import). Kernel 1.0.23 was installed over 1.0.22 while a WanGP server held the old binaries; they were renamed `*.old122` in `site-packages/llamacpp_gguf_cuda` and can be deleted after that server restarts.
