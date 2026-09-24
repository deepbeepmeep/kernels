# Qwen CUDA fusions - RTX 5090 validation

## Baseline attribution correction

This September 22 optimization round starts from the existing 1.0.22 kernels and already-optimized Qwen engine. Its initial Q4/two-draft-MTP baseline is **149.22 tok/s at 2K and 140.77 tok/s at 16K**, calculated as total emitted tokens divided by total decode time. Q4 was already above 100 tok/s before this round. The older 64.3-to-101.3 tok/s table in `QWEN38_ENGINE_PERFORMANCE.md` is not this work's starting baseline. Attributing that table's +58% decode or 2.30x prefill result to the current work was incorrect; those cumulative claims are withdrawn for this round.

No matched original-to-current cumulative benchmark has been established. The controlled rerun below supports an 8.3-8.8% improvement in fixed-input single-token GPU decode at 2K context. End-to-end MTP measurements show small, variable changes and do not establish that same gain. Experimental kernel percentages must not be added to either result.

## Controlled rerun after the performance claim was challenged

The earlier five-round result was preliminary. Its baseline swapped the native linear extension but still shared newer attention binaries and the modified GDN kernel with layout flags disabled. The rerun holds installed attention, Prism and SM120 assets constant in both arms, loads the installed 1.0.22 linear extension for the reference, and freezes the literal committed GDN source at W10 commit `0d8ea0f61aed23f1d551825529839398c6392049`. Both arms keep the same resident real Qwen3.8-27B Q4_K_M weights, BF16 computation and Q8 cache. This isolates this round's optional fusions; it is not a benchmark of the entire historical original application.

Four fresh processes each measured 40 randomized balanced ABBA/BAAB quartets for the main comparison, 40 duplicate-reference control quartets and 40 old/new-binary control quartets. Each sample contains two full target passes in one CUDA graph, with external timing events. GPU state restoration precedes the timed interval; CPU enqueue stalls cannot occur between the timed start/end. Exact logits and all mutated live state/cache tensors were checked after one, two and three passes. The later three processes additionally compare the timed wrapper with two direct reference graph replays. The first harness version is archived; its timing protocol is identical. All 1,920 samples and 480 quartets are retained, with no outlier removal. A failed nested-graph setup attempt produced no timings and is retained separately.

| Process / prompt context | Mean reference GPU time | Mean optimized GPU time | Geometric paired speed gain |
| --- | ---: | ---: | ---: |
| 2K prompt 0, first process | 12.010 ms | 11.036 ms | +8.83% |
| 2K prompt 0, fresh-process repeat | 11.940 ms | 11.014 ms | +8.41% |
| 2K prompt 1 | 11.939 ms | 11.020 ms | +8.34% |
| 16K prompt 0, noisy process | 15.845 ms | 14.688 ms | +7.90% |

All 120 main 2K quartets favor the candidate. The two identical-prompt processes' duplicate controls differ by -0.054% and +0.035%, far below the main change. Their observed process range is a more honest reproducibility summary than a narrow within-process confidence interval. At 16K, six of forty main quartets are slower and duplicate-control variation is substantial: the conditional 95% whole-quartet bootstrap interval for the main gain is +5.58% to +10.26%, while the duplicate control spans -1.99% to +3.11%. This does not establish a precise 7-8% gain or absence of regressions at 16K. Neither context establishes an end-to-end MTP throughput improvement or peak-VRAM result.

Evidence is under `D:/AMD/cuda-fusions-20260922/proof/`: raw process JSON/logs, `analysis/report.md`, `analysis/samples.csv`, `analysis/blocks.csv`, binary manifests, literal GDN reference and source snapshots. `tools/experiments/analyze_qwen_speed_proof.py` performs the independent CPU-only analysis, including timing-order effects, unchanged controls, and an adjacent-two-quartet sensitivity interval. Bootstrap intervals are conditional on tested processes/workloads, not predictions for other prompts, GPUs or LLM modes.

These changes use the existing Qwen layer forwards and decoder-mode configuration. There is no new vLLM installer or replacement of an instance's `forward`. The existing projection-fusion setup propagates the mode to GGUF projections after merging their weights. FFN calls still enter the normal module call and MMGP router; their list handoff releases the consumed activation input.

| Optimization | Precision and allocation intent | Dispatch |
| --- | --- | --- |
| Direct FP16/BF16 MMVQ output stores | Keep the FP32 reduction and its order; perform the existing final cast in the store, removing the FP32 output allocation | Single-token decode, using existing architecture-specific MMVQ eligibility and launch geometry |
| SiLU/multiply plus Q8 activation preparation | Preserve both activation rounding boundaries and the existing Q8 quantization; remove the separate activated input tensor | Single-token FFN; Prism keeps its activation before the Hadamard transform |
| Partial RoPE plus paged KV write | Preserve multiplication and addition rounding, cache dtype, Q8 blocks and scale calculation; write existing cache storage | CUDA vllm attention with supported head dimensions |
| GDN direct head addressing | Read original projection layouts and write the required output layout; remove seven permutations per affected layer, without changing recurrent arithmetic or grouped rollback snapshots | Existing raw CUDA vllm recurrence for decode and MTP verification |

All four are selected only in vllm mode. Legacy and cg keep their previous paths. AMD retains its existing kernels. The native changes extend the existing `linear` API with optional flags; older installed wheels keep their original call signature. The CUDA source uses existing architecture dispatch, without new RTX 5090-specific tuning parameters. Hardware performance validation is limited to the RTX 5090; portable source support does not establish a speedup on other GPUs.

## Environment and evidence

Validation on 2026-09-22 used `py311`, Python 3.11.9, PyTorch 2.10.0+cu130, Triton 3.6.0 and RTX 5090. The user's later request to run on this GPU superseded the earlier no-compilation instruction. The native candidate was built separately at `E:/ML/kernels/llama.cpp/build/cuda-fusions-rtx5090/lib.win-amd64-cpython-311`; the installed wheel was left intact and used as the numerical/performance baseline. No wheel was published for this CUDA change.

Reports and logs are in `D:/AMD/cuda-fusions-20260922/`:

- `native.json`: 432 candidate ordinary-path results match the installed wheel bit for bit; 402 eligible fused results also match exactly, plus six changed-input graph comparisons. Fixtures cover 18 quantization formats and FP16/BF16/FP32.
- `rope-final.json`: 72 exact comparisons across dense/Q8 caches, strided tensors, partial rotation, masked slots and head dimensions; 18 changed-input/slot graph replays and warmup with no cache. No spills in inspected kernels.
- `gdn-layout.json`: 113 exact comparisons across layout flags, FP16/BF16/FP32, token lengths 1/2/3/5/7/9, batched and unequal dimensions. Includes normalized output, live state, every tested rollback prefix and graph replay.
- `gdn-layout-repeat.json`: focused GDN sequence timings improve 1.41-1.65x for token lengths 1/3/5. All 12 inspected raw-kernel variants have zero spills; capture peak allocations decrease. Isolated sequence timings are not whole-model speedups.
- Native resource inspection found no local memory or stack spills in 924 relevant MMVQ/quantization specializations.
- 41 CPU/FakeTensor routing/HIP tests and 18 portable dispatch/probe tests pass. Tests include real Qwen blocks through MMGP's quantization router, single/multi-token FFN hook behavior, and a weak-reference check that the activation handoff releases its original tensor before unfused linear execution.

## Regression rejected

Typed output/FFN fusion was initially enabled for short multi-token batches. Measurements showed 1-5% kernel slowdowns at three/five tokens and a larger end-to-end MTP slowdown. WanGP now restricts those fusions to single-token decode and skips native capability checks for prefill and verification rows. The established multi-token FFN route is retained, including activation release before the down-projection's MMGP load hook. The single-token handoff also avoids keeping an extra reference to its original input if it uses the existing activation path. Repeated full-model measurements recover the slowdown; generated text and MTP acceptance counts stay identical.

## End-to-end Q4 MTP

Actual Deepy Qwen3.8-27B Q4_K_M, Q8 KV cache, two MTP drafts, confidence 0.3, greedy sampling, warm graphs, uncached 2K/16K prompts, 256 generated tokens, three distinct prompts per context. The following initial instrumented measurements report the median over those prompts:

| Context | Installed baseline | Final candidate | Change |
| --- | ---: | ---: | ---: |
| 2,048 | 149.11 tok/s | 156.41 tok/s | +4.89% |
| 16,384 | 140.38 tok/s | 147.31 tok/s | +4.94% |

All six completions are byte-identical, as are MTP acceptance and state-alignment statistics. Peak allocated memory remains 17.480 GiB and peak reserved memory 17.830 GiB. Sampled whole-device memory is 19.545 GiB; this device-wide 50 ms sample can miss transient peaks and includes unrelated applications. Reports: `q4-mtp-baseline/` and `q4-mtp-final/`.

## Measurement limits and additional regressions

The same-process graph control reduces exposure to cross-process clock/workload drift. `tools/benchmark_qwen_fusion_graph.py` loads the real Q4 model once, captures baseline/candidate graphs against the same resident weights and KV addresses, and restores recurrent/convolution state and the overwritten KV slot outside each timed region. These graphs include the output projection but exclude host scheduling and sampling. Their separate test graph pools coexist, so this experiment is not a production memory measurement. The following older five-round figures are preliminary and superseded by the controlled rerun above.

| Full GPU decode, 2,056 active tokens | Median time |
| --- | ---: |
| Installed extension, original paths | 14.819 ms |
| Candidate extension, original paths | 14.860 ms |
| Candidate extension, optimized paths | 13.783 ms |

`q4-paired-native.json` reports five alternating rounds of ten replays: median paired speedup is 8.08% against the installed extension and 7.81% against the same-binary baseline. All variants match exactly in logits, recurrent/convolution state, KV bytes and scales after both one and three replays. One round was unusually fast; every round favors the candidate. An earlier four-way ablation at 4,096 active tokens (`q4-paired-graph.json`) finds 7.06% median paired improvement, with individual rounds between 6.39% and 7.46%. These are fixed-input full GPU decode measurements, not generated tokens/s or MTP gains.

The same harness with `--verify-tokens 3` tests real native-MTP proposals and the target's three-token verification path (`q4-paired-verify3.json`). It compares all three logits rows, 96 live target convolution/recurrent tensors, 96 prefix snapshot tensors, and all three overwritten KV slots/scales after one and three replays. All match exactly across installed, rebuilt baseline and candidate. This comparison covers target verification, not the MTP drafting/acceptance/refresh cycle. Timings do **not** establish a gain against the installed baseline: median paired ratio is 0.998x, with individual rounds from 0.951x to 1.055x; the same-binary ratio is 1.032x. Intra-run drift is substantial, so neither a universal MTP speedup nor a systematic verification slowdown is established by this run.

GPU timing varied during this Windows desktop session. In particular, some instrumented runs shifted both prefill and decode by about 15%. Removing background driver-memory polling reduces measurement interference but has not established the cause of every outlier. Throughput comparisons therefore also use separate `--no-memory-poll --no-stage-profile` runs, and sum generated tokens / elapsed decode time across matching prompt hashes. The `--repeats` option changes corpus offsets, so its rows are distinct workloads rather than repetitions of one prompt.

Sampled Q4 MTP baseline/candidate/baseline runs (`q4-sampling-timing-*`) produce 140.01 / 144.13 / 142.00 aggregate tok/s. Outputs, acceptance statistics and allocator high-water marks match exactly. This supports a small aggregate benefit; paired sub-percent slowdowns and prefill outliers do not establish a universal per-prompt speedup.

IQ3_S, IQ2_M and Bonsai PTQ1 MTP runs also preserve generated text, MTP acceptance and peak allocated/reserved memory. Actual Q4 generation passed legacy, cg, vllm and vllm+MTP checks, with both CPU and CUDA default devices, cancellation followed by successful generation, and repeated calls. Legacy/cg tests replace Triton and the new GGUF fusion entry point with raising sentinels and assert FlashAttention is disabled. IQ3/IQ2/Bonsai pass the same repeated/cancellation/default-device checks in vllm+MTP.

The 2x end-to-end target has not been achieved. Profiling places approximately 72% of GPU time in the three-token Q4/Q6 matrix kernels. SASS inspection shows that NVCC already shares their main weight unpacking across token rows; adding another source-level multi-row helper would duplicate that optimization. Raising MTP depth retains additional state snapshots (about 75.75 MiB per draft position for this model), so the draft default was not increased under the unchanged-VRAM requirement.

## Further experiments, not enabled in production

### Batched MTP confidence decisions

After the controlled decode proof, the next experiment focused on one idea: batch the two greedy MTP confidence decisions into one GPU readback. The user subsequently allowed increased VRAM if a substantial acceleration could be offered as an option; the precision and mode requirements remain. The profiler showed 52 confidence readbacks across 26 target cycles. The experiment preserves the original confidence math, Python threshold comparison, proposal length, target verification width, reference sampler and state commit. It computes one unnecessary draft forward when the first confidence fails, then restores the original MTP cache length. This extra work is a performance risk for low-confidence workloads.

`tools/experiments/qwen_mtp_device_confidence.py` installs the prototype on benchmark runners only; production does not import it. The first version batches the two existing confidence pairs. A refinement captures both unchanged PyTorch reductions and the intervening MTP forward in one extra graph, with bounded cached mask phases and copied bias inputs. No model forwards or production engine methods were replaced.

`tools/benchmark_qwen_mtp_confidence.py` tests actual 256-token greedy generation, confidence 0.3, two MTP proposals, two distinct 2K prompts, one resident model and a fixed 4K runtime capacity. Each prompt has ABBA or BAAB full-generation ordering. The prefill, warmup and validation are outside timed decode. The compared baseline already includes this round's validated fusions, so the result is incremental and must not be added directly to the fixed-input GPU percentage above.

| Experiment | Prompt 0 paired gain | Prompt 1 paired gain | Conclusion |
| --- | ---: | ---: | --- |
| Two confidence checks, one readback | +0.93% | -0.33% | No useful established benefit |
| Both confidence checks and MTP forward captured together | +1.99% | +1.18% | Modest preliminary benefit; not the requested large speedup |

All eight generation calls in each experiment produced identical token lists, live target convolution/recurrent state hashes and acceptance statistics for their prompt; cache alignment remained exact. Each candidate generation exercised 111 eligible rounds. It reduced 218/217 reference confidence readbacks to 111, with four/five extra forwards caused by rejected first proposals. The captured version performed zero graph captures during measured calls. Its candidate rates were 161.4-161.7 tok/s; matched reference rates were 157.8-159.7 tok/s. These are a small number of whole-generation comparisons, not a broad regression guarantee.

The captured runner retained 207,872 additional allocated bytes after warmup. Peak reserved memory was 22 MiB higher than the uncaptured experiment. Because candidate graph pools coexist with the reference during the paired benchmark, this is indicative allocator accounting rather than an isolated production-memory comparison. CPU confidence/rollback checks covered 108 FP16/BF16/FP32 cases, including exact threshold boundaries and NaNs. Actual full-runtime GPU testing here covered BF16 Q4 only, not all thinking-mask transitions, sampling settings or high-confidence-rejection workloads.

The experiment remains disabled in production: its modest speed result does not justify adding a user-facing VRAM tradeoff option or asserting regression-free acceleration. The broader 2x target has not been achieved. Raw results, logs and source snapshots are in `D:/AMD/cuda-fusions-20260922/mtp-confidence/{paired,captured}/` and the containing directory.

The larger opportunity is to amortize target weight reads across more accepted tokens. Compact exact recurrent rollback is being investigated separately in `QWEN_MTP_COMPACT_ROLLBACK.md`. Its purpose is to make longer verification fit the existing state-memory budget; replay cost and graph allocations still need measurement. Depth must be evaluated independently: depth 4-7 changes the generic MMVQ launch geometry, and depth 8 verifies nine rows, crossing the current eight-row MMVQ limit into MMQ. A different depth is not itself a proof of unchanged output or speed.

The actual-model depth sweep (`mtp-depth-sweep/results.json`, `tools/benchmark_qwen_mtp_depth.py`) tests two 2,048-token prompts, exactly 256 greedy output tokens each, confidence 0.3 and bracketing depth-2 controls. One model and maximum-depth-7 graph allocation remain resident throughout. Median rates are 126.82 / 127.69 / 122.75 / 127.88 / 125.01 tok/s at depths 2 / 3 / 4 / 6 / 7. Depth 6 ranges from -2.72% to +4.81% against its prompt's bracketing control. Although accepted output per target pass rises from approximately 2.29 to 2.93, the added work cancels that advantage. Depth-2 repetitions are token-identical; other depths sometimes diverge after 63-169 common tokens. Caches remain aligned and no run exceeds its token budget. This rejects a blanket depth increase for these workloads. It is not a comparison against the earlier 32K-capacity throughput runs or a production memory validation.

The compact rollback prototype subsequently passed bounded BF16 T=3 single-layer and 48-layer graph parity. Its initial batched implementation failed strict parity in eight BF16 state elements; checked pointer-alignment annotations restored the direct-argument reduction layout and exact results. Both attempts are retained. Rollback payload, including unchanged convolution snapshots and the pointer table, drops from 151.5 to 81.792 MiB. Batched old/compact timing ratios are 1.021x / 0.985x / 1.031x for committing prefixes one / two / full acceptance, with substantial timing variance; separate per-layer replay launches lose performance. This remains an isolated experiment because no reliable speed benefit without regression or full-runtime peak-memory result has been established. See `compact-rollback-batched-aligned.json` and the separate proposal for the remaining validation scope.

The saved 64-token instrumented MTP trace has 368.85 ms of GPU work within a 433.70 ms span. Removing every idle gap would yield at most 1.176x for that trace; profiling inflates some launch gaps, so this is an optimistic ceiling. CPU synchronization timings mostly include waiting for useful GPU work. Greedy block acceptance with bounded scratch could remove some readbacks, but the existing vocabulary-sized acceptance cache cannot simply be enabled under the memory constraint.

The checkpoint header gives 15,041,593,344 bytes for target blocks and 1,042,944,000 bytes for the output head (`target-weight-traffic.json`). Streaming those approximately 16.08 GB once at the RTX 5090's published 1,792 GB/s peak takes about 8.98 ms, before drafting, attention, activations, launch costs or imperfect bandwidth utilization. At roughly 2.2-2.3 accepted output tokens per target pass, that is an optimistic weight-traffic ceiling of approximately 245-256 tok/s for the current proposal behavior, not a promised achievable rate. Small cache reuse makes this an estimate rather than a strict mathematical bound. Doubling the initial approximately 150 tok/s MTP result therefore needs fewer target passes per emitted token as well as faster kernels. Hardware source: [NVIDIA RTX Blackwell architecture whitepaper, pages 12-14](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

An offline greedy prompt/history lookup experiment selected proposals using only previously visible tokens, then compared them with the six saved 256-token Q4 completions. Two-token minimum matches, up to seven proposed tokens, yield only 1.17 tokens per idealized verification pass (three/four-token minimum matches yield 1.10/1.06). This excludes proposal and verification cost and does not justify replacing MTP on these story workloads. Lookup may remain useful for copying or repetitive tool/code workloads, which were not measured here. Evidence: `lookup-opportunity.json` and its reproducible script in the report directory.

An isolated native scheduling probe (`tools/experiments/native_mmvq_probe.*`) was compiled with CUDA 13.1 and run on six actual Q4_K/Q6_K projection shapes, three token rows, BF16 inputs. Five variants preserve the original four-warp K partition, integer dots, FP32 cross-warp addition order, lane reduction and output-lane selection. All match installed-native FP32 results exactly for random, zero, scaled and captured inputs; inspected kernels have no spills. Each paired timing flushes 256 MiB before the timed interval to avoid claiming hot-L2 gains. Row-one tiling and distributed final reductions are usually neutral or slower. Row-four serial tiling improves the Q6_K FFN down-projection `[5120,17408]` by about 4-6%, including a separate BF16/FP16/FP32 repeat. Other shapes do not show a general benefit. Reports: `native-mmvq-probe/results.json`, `q6-repeat.json`, `resources.txt`. This narrowly tuned result is not integrated into production and is not a whole-model gain. The existing MMVQ integer-MMA alternative cannot simply be substituted while retaining its current scaled-partial FP32 grouping.
