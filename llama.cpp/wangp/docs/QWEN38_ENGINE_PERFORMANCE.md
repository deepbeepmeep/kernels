# Qwen3.8 engine performance and implementation report

Q4 remains the primary quality/performance target. Earlier shared optimizations and frozen snapshots are retained below; the latest [SM120 follow-up](#sm120-follow-up-tma-investigation-and-retained-asynchronous-copies) records the TMA experiments, VRAM diagnosis, and final asynchronous-copy implementation.

The subsequent [compaction checkpoint fix](#compaction-checkpoint-retention-fix) prevents an avoidable long-prefix replay while keeping the existing three-checkpoint RAM budget.

**Current outcome:** the SM120 follow-up removes the approximately **3.27 GiB driver-memory increase** caused by the tested TMA path. The retained implementation uses ordinary `cp.async`, with **17–24% faster isolated prefix attention**, **4.4–4.6% higher full-20K prefill throughput**, and **7.3% higher throughput for a 1,024-token suffix on a cached 20K prefix**. It does not establish a reliable model-decode gain. Q4/Q8 precision is unchanged. Other GPU architectures keep the shared kernels; legacy/cg keep their PyTorch/native paths. The earlier shared decode/prefill gains remain documented separately. This follow-up postdates source snapshot **1.0.20** and does not rebuild the native package.

## Latest round at a glance

This summary covers only the SM120/TMA round. Earlier rounds remain below as historical evidence.

| Q4 workload | Before | Revised | Matched throughput gain |
| --- | ---: | ---: | ---: |
| Full 20K prefill, two drafts | 7.831 s | 7.484 s | +4.36% |
| Full 20K prefill, four drafts | 7.838 s | 7.481 s | +4.57% |
| 1,024-token suffix on a cached 20K prefix | 0.539 s | 0.502 s | +7.33% |

Times are medians over three prompts; gains are medians of matched ratios. The supported expectation is **4–5% faster long prefill and about 7% faster cached-prefix additions**. This round establishes **no reliable decode gain**: median matched changes were -1.01% with two drafts and +0.14% with four at 20K. Short-prefill results were mixed. Isolated attention gains of 17–24% for prefill and about 10% for long-context verification are not whole-model gains.

The retained code uses fixed-layout Gluon kernels with ordinary `cp.async` staging and the same high/low Q8 attention arithmetic with FP32 accumulation. TMA was rejected because its first launch consumed an extra **3.27 GiB** of driver memory on this setup. The replacement restores the shared path's footprint: **17.634 GiB PyTorch peak, 18.016 GiB reserved**, and about **21.9–22.0 GiB whole-GPU usage** in the four-draft audit. This removes a prototype regression, rather than saving 3.27 GiB below the pre-round engine.

Average prefill power increased from **507–510 to 521–525 W**, depending on draft depth. Decode power remained similar: **502 -> 509 W** with two drafts and **483 -> 480 W** with four. Both paths regularly reach the unchanged 600 W budget. Prefill energy/request changed by -2.4%/+0.5% at two/four drafts; decode energy/token changed by +0.6%/+1.4%. No substantial efficiency change is established. No competing model process was found, and all test workers have finished.

Q4/Q8 precision, sampling, and MTP verification are unchanged. All **38 matched performance/power completions** are identical to their shared references, and **24 SM120 GPU tests** pass. Real-checkpoint legacy/cg/vllm checks also pass; legacy/cg retain zero Triton/FA2 calls. Other architectures keep the shared kernels; the new staging is selected only on SM120/D256. Compilation ranges are bounded and cached variants stay silent. Q3/Q2 were not rebenchmarked in this round. The native binary and snapshot 1.0.20 are unchanged; live Python changes postdate the snapshot. Restart WanGP to load them and release any old TMA reservation.


## First pass: Q4 attention, native linear dispatch, and state storage

Measured on 5 September 2026, on the local RTX 5090 with Windows, Python 3.11, PyTorch 2.10.0+cu130, and `Qwen3.8-27B-Uncensored-Q4_K_M.gguf` (16,810,714,528 bytes). The checkpoint, Q8 KV cache, and two-token MTP configuration remain unchanged.

The original WanGP result is reproducible: 67.1 and 72.7 output tokens/s with **20,000 tokens actually in the prompt**. The approximately 150 tokens/s llama.cpp reports have supporting firsthand measurements, but they do not establish 150 tokens/s for this same checkpoint, prompt, platform, and sampling configuration.

- A [firsthand Qwen discussion](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/112) reports 151.2 tokens/s with thinking disabled, while reasoning and different MTP depths produce materially different rates.
- A [native Linux RTX 5090 report](https://njannasch.dev/blog/aorus-5090-ai-box-gmktec-proxmox-lxc/) reaches 150.5 tokens/s on a short coding request with four draft tokens and a Q6 checkpoint. Its filled 96K and 128K tests fall to 92.7 and 84.1 tokens/s. Allocated context capacity and occupied context are different measurements.

The local llama.cpp comparison uses official Windows CUDA binaries, **b10809**, with the exact same local Q4 GGUF and prompt token IDs. Settings: full GPU offload, FlashAttention, Q8_0 K/V, 32,768 context capacity, native MTP with two draft tokens, batch 2,048, microbatch 1,024, one slot, no prompt reuse. Sampling uses temperature 0.6, top-p 0.95, top-k 20, min-p 0.05, seed 123, and repetition penalty 1.05. llama.cpp's 512-token repetition window differs from Deepy's completion-local repetition scope; this comparison is close, not bit-identical.

| Runtime | Active prompt | Prefill tokens/s | Decode tokens/s |
| --- | ---: | ---: | ---: |
| Original WanGP | 2,048 | 3,117–3,304 | 78.2–81.0 |
| Original WanGP | 20,000 | 1,189–1,204 | 67.1–72.7 |
| llama.cpp b10809 | 2,048 | 2,515–2,534 | 83.1–85.9 |
| llama.cpp b10809 | 20,000 | 2,744–2,752 | 76.0–77.7 |

Each initial row combines two different uncached prompts with a 512-token generation budget after warmup. The prompts contain public-domain *Count of Monte Cristo* prose followed by the requested story task. Downloaded source: [Project Gutenberg](https://www.gutenberg.org/cache/epub/1184/pg1184.txt). No checkpoint was downloaded for these tests.

The final comparison repeated three prompts per context, warming every relevant prefill shape and the decode graphs before timing. These are **medians** from the final runs:

| Metric | Original WanGP | Improved WanGP | llama.cpp b10809 |
| --- | ---: | ---: | ---: |
| Decode, 2,048 active tokens | 90.5 tok/s | **101.5 tok/s** | 93.3 tok/s |
| Decode, 20,000 active tokens | 76.9 tok/s | **93.4 tok/s** | 85.2 tok/s |
| Prefill, 2,048 tokens | 3,628 tok/s | **4,118 tok/s** | 2,810 tok/s |
| Prefill, 20,000 tokens | 1,387 tok/s | **2,834 tok/s** | 3,095 tok/s |
| 20,000-token prefill time | 14.42 s | **7.06 s** | 6.46 s |
| Peak PyTorch allocation, warm 32K capacity | 17.63 GiB | **17.55 GiB** | Not measured with PyTorch |

At 20K this is **21.5% higher median decode throughput and 2.04× prefill throughput**. Target verification GPU time fell from a median 21.70 to 17.64 ms. Improved decode ranged from 93.0 to 101.1 tok/s across the three prompts. llama.cpp retains an approximately 9% prefill advantage in this comparison.

The initial and final session rates differ; both sets are retained. The improvement percentages above use the final paired runs. MTP can emit one or two tokens beyond the requested benchmark budget; WanGP throughput uses the actual emitted count. llama.cpp rates use its reported timing fields. Every final llama.cpp response reported 20,000 evaluated prompt tokens for the long case, zero reused prompt tokens, and no truncation.

VRAM savings are modest: approximately **75 MiB**, or 0.4% of peak PyTorch allocation. This metric excludes the desktop, driver, and other processes. Weight and cache quantization were not reduced to obtain the saving.

The implementation changes address measured bottlenecks:

1. The old Q8 prefix-prefill kernel consumed about 82% of prefill GPU time. It used scalar FP32 matrix products. The new tiled kernel uses tensor cores, representing dequantized cache values and attention probabilities as high/low pairs, with FP32 accumulation. It also skips entirely masked future tiles. This retains substantially more precision than simply rounding the dequantized cache to BF16. A 1,024-token suffix over a 20K prefix fell from 62.4 to 15.2 ms per attention layer in the isolated check.
2. Decode and speculative verification reuse K/V reads across grouped query heads and speculative tokens, with split-context parallelism scaled to the GPU's SM count. The existing Q8 cache is read directly; no dense cache is allocated. Representative 20K attention checks improved by 1.6× for one query and 2.1× for three queries.
3. The native GGUF wrapper now dispatches short batches through llama.cpp's existing MMVQ implementation, using its architecture-aware selector. Larger batches retain MMQ. Previously every batch used MMQ. Sources are in `E:\ML\kernels\llama.cpp`; the local `1.0.14+mmvq` wheel is installed in `py311`. New binaries were compiled for `sm_120` only. The installed attention extension was preserved.
4. Speculative recurrent and convolution states no longer retain an unused pre-verification snapshot. Commit indexing now points directly to the state after each processed input. This removes approximately 75 MiB of persistent state and the corresponding per-verification copies, without changing state precision.

Four-token MTP was also tested in this first pass. It added approximately 150 MiB versus the optimized two-token path and did not deliver a reliable throughput benefit at that stage. The second pass below revisits larger depths with additional optimizations; Auto still uses two drafts.

Validation includes the actual Qwen3.8 Q4 runtime, repeated prompt calls, Qwen3.5 9B Q4 without MTP, real quantized-weight matrix checks, and CUDA graph replay after changing context lengths. The 9B MTP sidecar is absent locally, so its MTP path was not tested. Attention tests cover shuffled pages, ragged prefill, FP16/BF16, multiple head layouts, long contexts, and empty graph-padding sequences.

On 256 identical teacher-forced continuation tokens at each context size, highest-probability token agreement with the original engine was 97.66% at 2K and 99.22% at 20K. Mean distribution KL was 0.00355 and 0.00292 nats. Cross-entropy changed by +0.00943 and +0.00005 nats/token respectively. All logits were finite. These are numerical regression checks on original-engine continuations, not a broad language-quality benchmark or a claim of bit-identical output.

All **17 affected GPU attention and speculative-state tests pass**, including a run with `CUDA_LAUNCH_BLOCKING=1`. Six additional incremental-boundary checks passed. One existing test in that file, `test_detailed_telemetry_requires_environment_opt_in`, fails because it changes an environment variable after the runtime has cached that variable at import. That runtime file is unchanged by this work.

An actual `wgp.py --ask-deepy` stress run wrote and revised a ten-chapter story, used file tools, reused context, and exercised compaction. Recorded contexts ranged from 8,222 to 30,426 tokens. Its aggregate decode telemetry was 103.2 tok/s for thinking and 99.4 tok/s for tool arguments, with all reported MTP synchronization deltas equal to zero. This is workload coverage, separate from the controlled benchmark. The worker was stopped after collecting that coverage; its queued follow-up review was not run to completion. The story and logs are retained in the isolated test directory.

The algorithms are not tied to the RTX 5090. Performance on other GPUs has not been measured. `wgp.py`, MMGP, user configuration, sampling rules, and other model implementations were not modified. Restart WanGP to load the installed native binary and the updated engine code.

Reproduction and raw evidence are in [`tools/benchmark_qwen38_engine.py`](../tools/benchmark_qwen38_engine.py) and [`_temp_codex/qwen38_perf_20260905`](../_temp_codex/qwen38_perf_20260905). The final records are `original_warm/results.json`, `optimized_warm/results.json`, and `llama_final/results.json`; metadata identifies the GPU, software versions, checkpoint, and native binary hash. Earlier results, prompt IDs, profiler traces, numerical checks, the native wheel and original-package backup, and the isolated Deepy story session are also retained. The llama.cpp test server used port 7865 and was stopped after the comparison.

Example optimized benchmark:

```powershell
C:\Users\Marc\anaconda3\envs\py311\python.exe tools\benchmark_qwen38_engine.py --assets E:\ML\wan2gp\ckpts\Qwen3_8_27B_Uncensored --corpus _temp_codex\qwen38_perf_20260905\corpus.txt --output _temp_codex\qwen38_recheck --contexts 2048 20000 --repeats 3 --tokens 512
```

`run_original.py` in the evidence directory loads the saved original attention, original state implementation, and backed-up native binary without replacing the installed engine. `run_llama_final.py` starts and stops the isolated llama.cpp server for the comparison.

## Implementation and measurement details

The sections below document every retained optimization, its integration, the experiments used to choose it, and the limits of the evidence. The versioned source copy is [`E:\ML\kernels\llama.cpp-1.0.16`](../../kernels/llama.cpp-1.0.16/SOURCE_COPY.md), with a [source ZIP](../../kernels/llama.cpp-1.0.16-source.zip). Its package version is **1.0.16**; the measured and installed binary remains **1.0.14+mmvq**. The source copy is not a new binary build or performance measurement.

### Model and runtime configuration

| Item | Value |
| --- | --- |
| Checkpoint | `E:\ML\wan2gp\ckpts\Qwen3_8_27B_Uncensored\Qwen3.8-27B-Uncensored-Q4_K_M.gguf` |
| Weight format | Original mixed Q4_K/Q6_K tensors in the Q4_K_M checkpoint |
| GPU | RTX 5090, compute capability 12.0, 170 SMs |
| Software | Windows; Python 3.11.9; PyTorch 2.10.0+cu130; CUDA runtime 13.0 |
| Build toolkit | CUDA 13.1; native binaries generated for `sm_120` only |
| Architecture | 64 layers: 48 linear attention, 16 full attention; hidden size 5,120 |
| Full attention | 24 query heads, 4 KV heads, head dimension 256; 6 query heads per KV head |
| Linear attention | 16 key heads, 48 value heads, key/value head dimensions 128; convolution width 4 |
| KV layout | INT8 K/V, FP16 scales per 32 values, 256-token pages |
| Capacity / occupied prompts | 32,768 / 2,048 and 20,000 tokens |
| MTP | Existing native head, maximum 2 draft tokens, up to 3 target inputs per verification |
| Memory policy | Existing MMGP full-GPU profile; no additional weight quantization or CPU offload |

The benchmark runs the actual Deepy path: `load_qwen35_text_prompt_enhancer` → `Qwen35AssistantRuntime` → shared nano-vLLM `ModelRunner` → `Qwen3_5ForCausalLM`, with GGUF qtype linears. It requests the existing `vllm` decoder and Q8 cache. The application selector remains `wgp_config.json -> lm_decoder_engine`; no new switch was added.

### Benchmark protocol and complete final decode records

[`benchmark_qwen38_engine.py`](../tools/benchmark_qwen38_engine.py) loads through the existing read-only MMGP path and sets a 32,768-token capacity hint. It tokenizes the first 600,000 characters of the downloaded corpus. A fixed system prefix and the story request surround a corpus slice beginning at token `10000 + 23000 * prompt_index`. The slice produces the exact requested total prompt length. Token IDs, decoded prompts, and SHA-256 hashes are saved.

The final warmup primes a 20,000-token prompt and generates 64 tokens. This warms prefix attention, the final partial prefill chunk, and decode/verification graphs. Prefix chunks are 1,024 tokens; the long case also exercises a final 544-token chunk. Each measured prompt is primed afresh. CUDA synchronization brackets `time.perf_counter()` measurements; load time, warmup, and first-use compilation are excluded from the warmed results.

The 512-token budget is enforced by a sequence-length stop callback because Deepy can otherwise grant additional thinking tokens. A speculative pass may overshoot by one or two tokens. `stop_reason="interrupted"` in the saved results is this intentional benchmark stop, not a runtime failure. Throughput uses actual emitted tokens, including thinking tokens.

| Prompt index, 20K | Original tokens / decode seconds | Original tok/s | Improved tokens / decode seconds | Improved tok/s | Improved prefill seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 512 / 6.6598 | 76.879 | 513 / 5.0757 | 101.069 | 7.0447 |
| 1 | 512 / 6.1679 | 83.010 | 513 / 5.4914 | 93.418 | 7.0562 |
| 2 | 513 / 7.0329 | 72.943 | 514 / 5.5241 | 93.047 | 7.0701 |

| Context | Original decode range | Improved decode range | Original aggregate decode | Improved aggregate decode |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 88.0–91.3 tok/s | 93.9–102.8 tok/s | 89.9 tok/s | 99.2 tok/s |
| 20,000 | 72.9–83.0 tok/s | 93.0–101.1 tok/s | 77.4 tok/s | 95.7 tok/s |

Aggregate decode is total emitted tokens divided by total decode seconds; it differs from the median. At 2K, median decode improves by 12.2%. At 20K, improved WanGP decode is approximately 9.7% above the matched llama.cpp median, while llama.cpp prefill throughput is approximately 9.2% higher. The reason for initial-versus-final session-rate variation was not established. Three prompts demonstrate this local improvement, not a tight confidence interval or universal throughput.

Small numerical changes alter sampled continuations and MTP acceptance even with a fixed seed. These whole-model runs are not an ablation assigning a fixed number of tokens/s to each individual optimization. Isolated kernel gains must not be added together.

Peak allocated/reserved memory, current allocation, allocator pool summaries, MTP statistics, and sampled stage timings are recorded. Stage timings use the runner's existing CUDA-event telemetry; wall-clock decode also includes CPU work. The `target_passes` counter includes priming work and should not be interpreted as a decode-only pass count without accounting for it. Optional profiler traces cover a separate 1,024-token appended suffix or 64-token decode window.

The final llama.cpp harness posts the saved WanGP token IDs to `/completion`, sets `cache_prompt=false`, and excludes a 20K/64-token warmup. Its rates come from server timing fields; WanGP is timed around runtime calls. Repetition-window scope and arithmetic differ, as noted above. The exact server command is in [command.json](../_temp_codex/qwen38_perf_20260905/llama_final/command.json).

Final records: [original](../_temp_codex/qwen38_perf_20260905/original_warm/results.json), [improved](../_temp_codex/qwen38_perf_20260905/optimized_warm/results.json), [llama.cpp](../_temp_codex/qwen38_perf_20260905/llama_final/results.json), and [summary](../_temp_codex/qwen38_perf_20260905/summary.json).

### Changed files and ownership

| File | Change |
| --- | --- |
| [attention.py](../shared/llm_engines/nanovllm/layers/attention.py) | High/low tensor-core prefill, causal tile skipping, grouped decode/verification kernels, and dispatch |
| [qwen3_5.py](../shared/llm_engines/nanovllm/models/qwen3_5.py) | Removes unused speculative snapshot; updates convolution/recurrent writes and individual commits |
| [model_runner.py](../shared/llm_engines/nanovllm/engine/model_runner.py) | Adjusts cached commit source views to the new snapshot indexing |
| [gguf_llamacpp_kernels.cu](../../kernels/llama.cpp/csrc/gguf_llamacpp_kernels.cu) | Connects short packed linear batches to upstream MMVQ |
| [native wrapper](../../kernels/llama.cpp/src/llamacpp_gguf_cuda/__init__.py) | Updates the one-time message to identify MMVQ/MMQ dispatch |
| [native version](../../kernels/llama.cpp/src/llamacpp_gguf_cuda/version.py) and [README](../../kernels/llama.cpp/README.md) | Identifies the tested local package and documents dispatch |
| [test_q8_paged_attention.py](../tests/test_q8_paged_attention.py) | Five new GPU cases for arithmetic, layouts, and graph replay |
| [test_nanovllm_speculative_sampling.py](../tests/test_nanovllm_speculative_sampling.py) | Updates existing fake-state fixtures and commit expectations |
| [benchmark_qwen38_engine.py](../tools/benchmark_qwen38_engine.py) | Real-checkpoint timing, profiling, and memory records |

Only three tracked production files changed in WanGP: `attention.py`, `qwen3_5.py`, and `model_runner.py`. Existing untracked tests and other user work were preserved. The native project is a separate repository; [native_mmvq.patch](../_temp_codex/qwen38_perf_20260905/native_mmvq.patch) isolates this task's native changes from pre-existing edits.

### Optimization 1: tensor-core Q8 prefix prefill

**Root cause.** The old `q8_paged_prefill_kernel` converted Q to FP32, dequantized K/V to FP32, and used `tl.dot(..., input_precision="ieee")` for QK and probability-times-V. These strict-FP32 products were the main prefill bottleneck. In the [original suffix profile](../_temp_codex/qwen38_perf_20260905/baseline/prefill_profile.txt), this kernel consumed 81.93% of GPU time: 16 calls, 1.227 seconds total, 76.7 ms per full-attention layer. Q4 and Q6 MMQ consumed another 10.41% and 3.37%; FLA chunked recurrence was about 0.53%.

**Arithmetic.** Q stays in its incoming BF16/FP16 dtype. Q8 integers and FP16 scales are still multiplied in FP32. Each dequantized value is split into a high part and a residual in the query dtype:

```text
hi(x) = cast_to_query_dtype(x)
lo(x) = cast_to_query_dtype(x - float32(hi(x)))

QK ≈ Q · hi(K) + Q · lo(K)
PV ≈ hi(P) · hi(V) + lo(P) · hi(V)
   + hi(P) · lo(V) + lo(P) · lo(V)
```

Both QK terms and all four PV terms use tensor-core products with FP32 accumulation. Passing the existing accumulator into `tl.dot` avoids separate full-size intermediate accumulators. Online-softmax maxima, denominators, and output accumulators remain FP32. Exponentiation uses `exp2(x * log2(e))`, and output retains the query dtype.

The residual preserves more information than rounding dequantized K/V to one BF16 value. This is still approximate arithmetic: residual rounding and accumulation order differ from strict FP32. Numerical checks below quantify that difference; output is not bit-identical.

**Traversal.** Existing pages, block-table lookup, per-32-value scales, ragged cumulative sequence lengths, and row-wise causality are retained. No full dense cache temporary is allocated. The loop previously visited the complete context, including tiles entirely in the future of a query block. Its new upper bound is:

```text
min(context_length, prefix_length + (query_block + 1) * BLOCK_M)
```

The row-wise mask still handles partially visible tiles. Masked query loads explicitly use zero. The final launch is **BLOCK_M=32, BLOCK_N=32, 4 warps, 1 stage**, replacing 32×16, 8 warps, 1 stage. The grid remains sequence × query head × query block.

**Isolated check.** The reference is the old strict-FP32 Triton prefill. These BF16 cases use shuffled pages, seed 44, warmup, and five CUDA-event timing repetitions. Relative L2 is `norm(candidate-reference) / norm(reference)`.

| Head dimension | Query lengths / prefix lengths | Original ms | Improved ms | Speedup | Relative L2 | Max absolute error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 17 / 0 | 0.0691 | 0.0569 | 1.21× | 0.0000767 | 0.00390625 |
| 256 | 47, 73 / 251, 529 | 0.5155 | 0.2123 | 2.43× | 0.0001449 | 0.00390625 |
| 256 | 1,024 / 20,000 | 62.3675 | 15.2357 | **4.09×** | 0.0006648 | 0.00024414 |

Errors are measured after output dtype conversion; a short-case absolute error can be a BF16 output step. These per-layer suffix timings differ from complete 20K prefill. Evidence: [check script](../_temp_codex/qwen38_perf_20260905/check_prefill.py) and [final log](../_temp_codex/qwen38_perf_20260905/check_prefill_final.log).

### Optimization 2: grouped Q8 decode and verification

**Sharing K/V reads.** The old native adapter treated speculative queries as separate decodes by expanding block tables and lengths. The new `q8_grouped_partials_kernel` combines query tokens and heads sharing one KV head, so one K/V tile serves multiple tensor-core rows.

For `G = H_Q / H_KV` and `Q_PER_SEQ` inputs, a row within a KV head maps as follows:

```text
query_index = sequence * Q_PER_SEQ + row // G
query_head  = kv_head * G + row % G
valid_row   = row < Q_PER_SEQ * G
visible_key = key_position <= context_length - Q_PER_SEQ + row // G
```

Here `G=6`: decode has six useful rows per KV head, while three-input verification has eighteen, spanning two 16-row tiles. The same high/low arithmetic reads the existing Q8 cache. Q retains BF16/FP16 precision; the former native Q8 attention path additionally quantized queries to Q8_1.

**Parallelism.** The partial kernel uses **M=16, N=32, 4 warps, 1 stage**. Its split count is derived from shape and SM count:

```text
query_blocks = ceil(Q_PER_SEQ * G / 16)
base_blocks  = sequence_count * H_KV * query_blocks
splits       = min(128, next_power_of_two(ceil(4 * SM_count / base_blocks)))
```

The factor of four aims for several blocks per SM at batch one. On the 170-SM test GPU, both one-token decode and three-token verification select 128 splits. This rule has no RTX 5090 device-name branch. Each split handles a contiguous range of 32-key tiles and writes FP32 partial output, maximum, and denominator.

`q8_grouped_reduce_kernel` combines splits with stable softmax rescaling:

```text
m = max(split_maxima)
w[s] = exp(split_maxima[s] - m)
output = sum(partial_output[s] * w[s]) / sum(partial_denominator[s] * w[s])
```

Empty splits use zero probabilities and denominator. A finite sentinel maximum and denominator floor of `1e-20` give zero output for a zero-length graph-padding sequence. Empty contexts do not read block-table entries.

FP32 scratch shapes are `[total_queries, H_Q, splits, D]` for partial output and `[total_queries, H_Q, splits]` for each statistic. Scratch depends on query/split counts, not occupied context. Three queries × 24 heads × 128 splits × dimension 256 require **9.0703 MiB per call**, excluding output. Actual graph-pool residency is included in peak-memory measurements.

**Integration.** `Attention.forward` selects `_q8_grouped_attention` within the existing eligible Q8-cache branch when Triton is available, Q is contiguous CUDA BF16/FP16, head dimension is 32/64/128/256, and the call is decode or verification. Existing outer native-backend availability gates remain. Existing paths for other layouts and non-Q8 attention remain available; no exception-catching fallback was introduced.

Verification uses its existing single-sequence block table and `context.cu_seqlens_k[1:]`, avoiding per-query expanded metadata. Decode reads `context.context_lens`. Verification returns the result directly; decode adds the existing singleton dimension, preserving the caller's shape contract. A one-time log identifies grouped Triton Q8 attention.

Lengths are read on the GPU: no context-length `.item()` or host synchronization was added. Shapes and launch sizes are fixed at graph capture, while GPU lengths can change on replay. Existing one-token decode graphs, two/three-input target graphs, and the separate MTP graph pool are reused.

**Isolated results.** Timing compares with the installed native Q8 attention; numerical errors compare with the old strict-FP32 prefill reference. Each timing captures 20 calls and replays the graph five times. Native speculative metadata expansion is prepared outside this microbenchmark, so its avoided runtime cost is not included.

| Dtype / head dimension | Queries per sequence / context | Native ms | Grouped ms | Speedup | Relative L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 / 256 | 1 / 20,000 | 0.10133 | 0.06264 | **1.62×** | 0.0000913 |
| BF16 / 256 | 3 / 20,000 | 0.23178 | 0.11266 | **2.06×** | 0.0001021 |
| BF16 / 256 | 2 / 2,048 | 0.03538 | 0.01288 | 2.75× | 0.0000496 |
| FP16 / 128 | 3 / 257, 531 | 0.01539 | 0.00772 | 1.99× | 0.0000239 |
| BF16 / 256 | 5 / 17 | 0.00858 | 0.01298 | **0.66×** | 0.0000965 |
| BF16 / 256 | 1 / 1 | 0.00606 | 0.00802 | **0.76×** | 0 |

Split/reduction overhead loses on extremely short cases. Five queries is kernel coverage beyond the retained production draft limit. This implementation benefits the measured 2K/20K workloads, not every context length. Evidence: [script](../_temp_codex/qwen38_perf_20260905/check_grouped.py), [final log](../_temp_codex/qwen38_perf_20260905/check_grouped_final.log), and [sweep](../_temp_codex/qwen38_perf_20260905/sweep_grouped.log).

### Optimization 3: packed MMVQ dispatch

**Root cause.** The wrapper sent all packed linear row counts through MMQ, including one-token decode and short verification. Upstream `mmvq.cu` was already in the compiled sources, but the wrapper did not use its dispatch.

`run_linear_cuda` now calls `ggml_cuda_should_use_mmvq(type, cc, batch_rows)` before MMQ. If selected, it:

1. Converts input to FP32 through the existing CUDA casting helper.
2. Allocates Q8_1 activation scratch: `batch_rows * padded_row * sizeof(block_q8_1) / QK8_1` bytes.
3. Calls `quantize_row_q8_1_cuda` on the current stream.
4. Creates lightweight ggml input/output descriptors and calls `ggml_cuda_op_mul_mat_vec_q` for the full output and input batch.
5. Checks the CUDA launch and returns FP32 output through the existing dtype-conversion wrapper.

Weights stay in their original packed GGUF representation; no dense weight matrix is materialized. Larger batches retain MMQ. The existing packed config mode is still called `mmq`; its internal dispatch now includes MMVQ.

Thresholds come from the [vendored upstream selector](../../kernels/llama.cpp/_vendor/llama.cpp/ggml/src/ggml-cuda/mmvq.cu). Blackwell uses MMVQ for up to 5 Q4_K rows and 7 Q6_K rows, with other thresholds for other architectures. No additional GPU-specific threshold was introduced. The Q4_K_M checkpoint's Q6_K tensors are also covered. The package log now describes “Packed MMVQ/MMQ” and architecture-aware dispatch.

**Validation.** [check_mmvq.py](../_temp_codex/qwen38_perf_20260905/check_mmvq.py) tested 30 combinations: five real matrices at 1/2/3/5/8/16 input rows. Matrices were `blk.0.attn_qkv.weight`, `blk.0.ffn_down.weight`, `blk.0.ffn_gate.weight`, `blk.3.attn_q.weight`, and `output.weight`. Outputs were finite and checked against a dense dequantized reference over up to 512 output rows. Candidate MSE had to be at most `1.3 * original_MSE + 1e-7`.

Individual cache-resident matrix timings were mixed, including some three-row regressions. Actual model profiling and generation established the usefulness of this route; no uniform matrix-level speedup is claimed. [Raw matrix results](../_temp_codex/qwen38_perf_20260905/mmvq_results.json) retain all cases.

Pre-existing native edits are not attributed to this task: MMQ output/scratch `zeros`→`empty`, the native attention Blackwell 64-split setting, and a newline-only `mmq.cuh` change. The installed `_attention` binary was retained byte-for-byte. Its hash matches the pre-existing `q8-attention-blackwell-64-split` snapshot, whose binding source also matches the current source. The 1.0.16 copy preserves that tested baseline along with the new MMVQ dispatch.

### Optimization 4: speculative-state allocation and copies

With two drafts, verification processes up to three inputs. Previously each linear-attention layer allocated four snapshots: index 0 before verification, then indices 1/2/3 after each input. Verification always commits at least one processed input, even if the first draft is rejected. The pre-verification snapshot was never selected.

| Operation | Original | New |
| --- | --- | --- |
| Allocate for maximum `V` verification inputs | `V + 1` snapshots | `V` snapshots |
| Before verification | Copy convolution and recurrent states to slot 0 | No snapshot copy |
| Save after zero-based `token_idx` | Slot `token_idx + 1` | Slot `token_idx` |
| Commit `processed_tokens = p` | Read slot `p` | Read slot `p - 1` |
| Insufficient buffer condition | `snapshot_count <= seq_len` | `snapshot_count < seq_len` |

Both convolution implementations and per-token FLA recurrent writes were updated. Individual commits and `ModelRunner._prepare_target_speculative_state` now use the same index. Commit dictionary keys stay 1/2/3; values remain storage views and the existing batched `foreach_copy_` is retained. Reset/release ownership, state precision, and acceptance/rejection behavior are unchanged.

For batch one and the existing BF16 state buffers, the exact removed snapshot is:

| One snapshot | Elements | Bytes |
| --- | ---: | ---: |
| Recurrent, one layer: `48 * 128 * 128` | 786,432 | 1,572,864 |
| Convolution, one layer: `(2 * 16 * 128 + 48 * 128) * 4` | 40,960 | 81,920 |
| Total per layer | 827,392 | 1,654,784 |
| Across 48 layers | 39,714,816 | **79,429,632 = 75.75 MiB** |

Speculative snapshots alone fall from 303.0 to 227.25 MiB. Each verification also avoids 96 tensor copies carrying 75.75 MiB of payload. This is structural arithmetic, not a separate measurement of bandwidth or whole-model decode savings.

Measured warm peak allocation falls from 17.627315998 to 17.553776264 GiB: **75.3047 MiB**. The net peak includes other workspace changes, so it differs slightly from snapshot-only arithmetic. No state dtype, checkpoint quantization, KV precision, or context capacity was reduced.

For scale, the target's 16 full-attention layers alone require about 1 GiB of Q8 K/V plus 64 MiB of scales at 32,768 positions. Those caches and the much larger weights are not compressed by this change.

### Experiments not retained

| Experiment | Observation and decision |
| --- | --- |
| `tf32x3` prefill | A 64-key/two-stage trial needed 180,224 shared-memory bytes against a 101,376-byte limit. A smaller working trial improved the long isolated case from 74.56 to 47.17 ms, 1.58×. Replaced by faster high/low arithmetic. |
| Separate high/low PV temporaries | Long-case timing was 75.42→21.49 ms, relative L2 0.000134. Chained `tl.dot` accumulation avoided separate large intermediates; its different accumulation order gave relative L2 around 0.000665 and better timing. |
| Prefill tile/warp sweep | 32×32/4 warps took 14.80 ms versus 28.70 ms for 32×32/8 warps and 17.54–17.74 ms for 32×64. The retained tile used 32 KiB shared memory; spills remained. Final separate check: 15.24 ms. |
| Grouped tile/split sweep | 16×32/4 warps at 16/32/64/128 splits took 0.237/0.170/0.150/0.107 ms. 32×32 reached 0.105 ms; 16×64 was slower. Retained 16×32 and the generic four-SM-wave rule, capped at 128. |
| Four draft tokens | Long runs produced 89.7 and 84.8 tok/s, peak 17.727 GiB versus 17.575 GiB in the preceding two-draft candidate: about 155 MiB extra. No reliable benefit; production remains two drafts. |

Intermediate model runs remain in `prefill_opt`, `mmvq_model`, `grouped_model`, and `draft4`. These occurred at different times and can produce different continuations; they are not a controlled whole-model ablation. The benchmark may temporarily raise the draft limit in its own process, without changing the production constant.

No Q3 checkpoint, lower-precision KV cache, shortened occupied context, disabled-thinking default, or new application setting was used for the final improvement. Evidence: [TF32 failure](../_temp_codex/qwen38_perf_20260905/check_prefill.log), [working TF32 checks](../_temp_codex/qwen38_perf_20260905/check_prefill_v2.log), [prefill sweep](../_temp_codex/qwen38_perf_20260905/sweep_prefill_hilo.log), [grouped sweep](../_temp_codex/qwen38_perf_20260905/sweep_grouped.log), and [four-draft results](../_temp_codex/qwen38_perf_20260905/draft4/results.json).

### Validation methods and limits

The **17 passing affected tests** comprise five new attention cases and twelve existing speculative sampling/state/graph checks. Only the state fixtures and expectations in the latter file were adjusted. The final run with `CUDA_LAUNCH_BLOCKING=1` recorded 17 passed, 18 warnings in 9.89 seconds.

New attention tests use seed 41, shuffled physical pages, and explicit FP32 dequantization/matmul/causal-softmax reference with TF32 disabled. Results are converted to output dtype and compared with `atol=0.0003, rtol=0.009`.

| Path | Dtype / head dimension | Query lengths | Context lengths | Query / KV heads |
| --- | --- | --- | --- | --- |
| Prefill | BF16 / 256 | 17, 33 | 257, 531 | 24 / 4 |
| Prefill | FP16 / 128 | 31 | 20,000 | 24 / 4 |
| Grouped verification | BF16 / 256 | 3 | 20,000 | 24 / 4 |
| Grouped verification | FP16 / 128 | 2, 2 | 257, 531 | 8 / 2 |
| Grouped decode | FP16 / 64 | 1 | 257 | 4 / 4 |

Grouped tests capture CUDA graphs, replay with changed GPU context lengths down to the query count, and require exactly zero output at zero context. The standalone grouped check also compares initially ragged lengths before capture. State tests check committed contents and verify cached sources remain views of the expected buffers. These complement repeated actual-checkpoint graph calls.

The additional incremental-boundary test outcome remains six passes and the unrelated telemetry failure described above. Logs: [affected tests](../_temp_codex/qwen38_perf_20260905/gpu_tests_final.log), [additional tests](../_temp_codex/qwen38_perf_20260905/unit_tests.log).

For model-level numerical regression, the actual MTP runtime teacher-forced the original continuation at 256 positions per context. Raw target logits were saved before forcing the next token; draft sampling and verification still ran normally. Target IDs were identical, distributions cover the full vocabulary, and GPU-to-CPU transfers were blocking.

| Metric | 2,048-token prompt | 20,000-token prompt |
| --- | ---: | ---: |
| Positions compared | 256 | 256 |
| Finite logits | All | All |
| Top-1 agreement | 97.65625% | 99.21875% |
| Mean KL(original || improved), nats | 0.00355110 | 0.00292281 |
| Original cross-entropy, nats/token | 0.38121948 | 0.36408836 |
| Improved cross-entropy, nats/token | 0.39065337 | 0.36413893 |
| Cross-entropy change | +0.00943390 | +0.00005057 |
| Perplexity ratio | 1.00947857 | 1.00005054 |
| Logits RMSE | 0.12414635 | 0.17525901 |

The checks required finite logits, mean KL below 0.01 nats, and perplexity ratio below 1.01. All passed. Only 512 positions were checked, and original-engine continuations favor the baseline. This supports limited numerical compatibility, not unchanged quality on every task. MTP rejection sampling and logits/sampling rules were not changed, but distribution-preserving sampling does not imply bit-identical outputs across different arithmetic kernels.

Evidence: [teacher-forcing harness](../_temp_codex/qwen38_perf_20260905/run_quality.py), [comparison](../_temp_codex/qwen38_perf_20260905/compare_quality.py), [quality results](../_temp_codex/qwen38_perf_20260905/quality_results.json).

The 9B compatibility run used real Qwen3.5 Q4 with `--draft 0`, repeated 2K prompts, and 256 output tokens. Its missing MTP sidecar prevented 9B MTP testing; first-use compilation affected its first prefill, so those numbers are not comparable 27B performance evidence.

The actual Deepy story session used isolated configuration/sessions, file writes confined to its artifacts, and external MCP discovery disabled. It exercised context reuse and compaction, wrote ten chapters, and revised continuity details. Weighted decode telemetry was:

| Stage | Tokens | Decode seconds | Aggregate tok/s |
| --- | ---: | ---: | ---: |
| Thinking | 20,731 | 200.948 | 103.166 |
| Statements | 902 | 8.671 | 104.025 |
| Tool arguments | 4,482 | 45.095 | 99.390 |

These rates exclude tool execution time. The queued follow-up review was not completed before stopping the test worker. Evidence: [harness](../_temp_codex/qwen38_perf_20260905/run_deepy_story.py), [summary](../_temp_codex/qwen38_perf_20260905/deepy_story_summary.json), and [session](../_temp_codex/qwen38_perf_20260905/deepy_story).

### Native build and installation

The existing native `setup.py` configures MSVC and Windows SDK environment variables directly; `vcvars64.bat` was not called. The log identifies MSVC 14.35.32215 and Windows SDK 10.0.22000.0. CUDA 13.1 reported a minor-version mismatch with PyTorch's CUDA 13.0; that warning remains in the log, and the resulting binary passed the checks above.

Build command from `E:\ML\kernels\llama.cpp`:

```powershell
$env:TORCH_CUDA_ARCH_LIST = '12.0'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
$env:CUDA_HOME = $env:CUDA_PATH
$env:LLAMACPP_GGUF_CUDA_VERSION_SUFFIX = '+mmvq'
$env:MAX_JOBS = '4'
& 'C:\Users\Marc\anaconda3\envs\py311\python.exe' setup.py build_ext --build-lib E:\ML\W10\_temp_codex\qwen38_perf_20260905\kernel_lib --build-temp E:\ML\W10\_temp_codex\qwen38_perf_20260905\kernel_build
```

NVCC logs show `-gencode=arch=compute_120,code=sm_120`, with no other GPU or PTX target generated. Both extensions were built by setup; the installed wheel uses the new `_C` and the original installed `_attention`.

The one-time [packaging script](../_temp_codex/qwen38_perf_20260905/package_kernel.py) backed up the package and metadata, staged these binaries and the updated wrapper/version, then rebuilt wheel metadata and SHA-256 `RECORD` entries. It expects the pre-install `1.0.14+outq8empty` metadata; it should not be rerun unchanged against the updated environment. The saved wheel is reusable:

```powershell
& 'C:\Users\Marc\anaconda3\envs\py311\python.exe' -m pip install --no-deps --force-reinstall E:\ML\W10\_temp_codex\qwen38_perf_20260905\dist\llamacpp_gguf_cuda-1.0.14+mmvq-cp311-cp311-win_amd64.whl
```

This installation was completed without updating PyTorch or MMGP. Binary identities:

| Component | SHA-256 |
| --- | --- |
| Tested/installed `_C.cp311-win_amd64.pyd` | `7d749ac18211841f2d33a9cb3f0dad731359235c90afff702e3d302cb4d90901` |
| Preserved `_attention.cp311-win_amd64.pyd` | `994d28d415d9585d3fbe2f392a22e7d55bb5f694cbcbd90b7b0371d819704156` |

The source algorithms are generic; this binary is SM120-only. Other GPUs have not been measured. Restart WanGP to load the changed engine and installed binary.

### Source copy 1.0.16 and baseline preservation

Version availability was checked against local source/build versions, installed metadata, git tags/history, and current assets of the [published GGUF Kernels release](https://github.com/deepbeepmeep/kernels/releases/tag/GGUF_Kernels). Assets reached 1.0.14 at the check; 1.0.16 was unused in those locations.

[`llama.cpp-1.0.16`](../../kernels/llama.cpp-1.0.16/SOURCE_COPY.md) contains the native source, vendored GGML, build scripts, and native tests. Only the copy's `setup.py` base version and Python version file are changed to 1.0.16, with a source-copy note added to its README. Its `wangp/` subtree preserves the new Triton kernels, Qwen/runner integration, affected tests, benchmark, and this report. Those WanGP changes are required for the combined improvement; installing a native wheel alone does not apply them.

The copy also records isolated native and WanGP patches, version-audit evidence, source revisions, and a per-file SHA-256 manifest. It preserves pre-existing native changes used during measurement and excludes binaries, build products, weights, and caches. It is a source snapshot, not a published release or replacement installation.

To build that copy for SM120 later, use separate output directories and clear the old local version suffix:

```powershell
Set-Location E:\ML\kernels\llama.cpp-1.0.16
$env:TORCH_CUDA_ARCH_LIST = '12.0'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
$env:CUDA_HOME = $env:CUDA_PATH
$env:LLAMACPP_GGUF_CUDA_VERSION_SUFFIX = ''
$env:MAX_JOBS = '4'
& 'C:\Users\Marc\anaconda3\envs\py311\python.exe' setup.py build_ext --build-lib build_lib --build-temp build_temp
```

The [original package backup](../_temp_codex/qwen38_perf_20260905/installed_kernel_backup) includes native/Python files and `1.0.14+outq8empty` metadata. [Reference sources](../_temp_codex/qwen38_perf_20260905/reference) retain original attention, Qwen blocks, and runner. If restoring, allocation and commit implementations must use matching indexing; mixing old/new state layouts is invalid.

`run_original.py` loads saved sources and the backed-up `_C` inside its process without changing the installation. An initial reference-harness attempt failed because duplicate module cache classes broke an `isinstance` branch during graph capture. Sharing the actual cache class and shared cache aliases fixed the harness; repeated real-checkpoint checks then passed with `CUDA_LAUNCH_BLOCKING=1`. This was not a new production crash.

Early metadata's `kernel_version` identifies the installed wrapper even if the harness overrides its binary. The binary path and SHA-256 identify the code actually tested. The benchmark now labels that field `kernel_package_version` to clarify the distinction.

### Reproduction details and artifact index

Run from `E:\ML\W10`, using fresh output directories and with the GPU otherwise available:

```powershell
$python = 'C:\Users\Marc\anaconda3\envs\py311\python.exe'
$evidence = 'E:\ML\W10\_temp_codex\qwen38_perf_20260905'
$assets = 'E:\ML\wan2gp\ckpts\Qwen3_8_27B_Uncensored'
& $python tools\benchmark_qwen38_engine.py --assets $assets --corpus "$evidence\corpus.txt" --output _temp_codex\qwen38_recheck --contexts 2048 20000 --repeats 3 --tokens 512
& $python "$evidence\run_original.py" --assets $assets --corpus "$evidence\corpus.txt" --output _temp_codex\qwen38_original_recheck --contexts 2048 20000 --repeats 3 --tokens 512
```

Add `--profile prefill` or `--profile decode` to a separate invocation; profiling runs after the timed cases. For isolated attention and affected tests:

```powershell
& $python "$evidence\check_prefill.py"
& $python "$evidence\check_grouped.py"
$env:CUDA_LAUNCH_BLOCKING = '1'
& $python -m pytest tests/test_q8_paged_attention.py tests/test_nanovllm_speculative_sampling.py -q
Remove-Item Env:CUDA_LAUNCH_BLOCKING
```

The original `check_mmvq.py` imports its baseline from the installed package. After installation, an unchanged rerun compares against the new package; explicitly load the backed-up `_C` to repeat the original comparison. The quality reference similarly needs original attention and the backed-up binary. Saved results reflect the intended old/new implementations.

`run_llama_final.py` starts/stops localhost port **7865** and writes to its fixed `llama_final` directory; preserve those records before rerunning. The executable is `D:\qwen38_benchmark\llamacpp_b10809\llama-server.exe`, from the [official b10809 release](https://github.com/ggml-org/llama.cpp/releases/tag/b10809). The comparison server and Deepy test workers were stopped; the user's active WanGP port was not used.

| Evidence | Location under `_temp_codex/qwen38_perf_20260905` |
| --- | --- |
| Initial measurements | `baseline/`, `baseline_v2.log`, `llama_results.json` |
| Final paired measurements | `original_warm/`, `optimized_warm/`, `llama_final/`, `summary.json` |
| Original prefill profile | `baseline/prefill_profile.txt`, `baseline/prefill_trace.json` |
| Improved decode profile | `grouped_model/decode_profile.txt`, `grouped_model/decode_trace.json` |
| Isolated checks/tuning | `check_prefill*.log`, `check_grouped_final.log`, `mmvq_results.json`, `sweep_*.log` |
| Logits regression | `quality_baseline/`, `quality_optimized/`, `quality_results.json` |
| Tests | `gpu_tests_final.log`, `unit_tests.log`, `qwen9b_no_mtp/` |
| Deepy workload | `deepy_story/`, `deepy_story.log`, `deepy_story_summary.json` |
| Native build and backup | `build_mmvq.log`, `native_mmvq.patch`, `dist/`, `installed_kernel_backup/` |
| Source-copy procedure | `create_source_copy.py`; output at `E:\ML\kernels\llama.cpp-1.0.16` |

PowerShell-redirected `.log` files can be UTF-16; Python-generated JSON, scripts, and this report are UTF-8.

### Remaining bottlenecks and scope

In the recorded 64-token [improved decode profile](../_temp_codex/qwen38_perf_20260905/grouped_model/decode_profile.txt), three-row Q4 MMVQ consumed 40.66% of GPU time and three-row Q6 MMVQ 15.53%; grouped attention partials consumed 9.12%. Convolution updates were 3.78%, Q8_1 activation quantization 2.15%, recurrent FLA 2.14%, and BF16-to-FP32 casts 1.92%. Additional one-row MMVQ work is present. These are sampled-profile GPU shares, not final whole-run wall-clock percentages or an ablation.

Larger VRAM reductions were not demonstrated at the same checkpoint/cache quality and speed. Extremely short grouped-attention cases regress by a few microseconds, other GPUs remain untested, and numerical quality checks cover a limited set of continuations.

Existing weight sharing, truncated MTP draft vocabulary, CUDA graphs, and offload management are not new optimizations claimed here. No forced `torch.compile`, new application config key, or new fallback behavior was added.

## Second pass: MTP depth, sampling, and Q3/Q2

The second pass keeps Q4, thinking enabled, the original sampling parameters, and the Q8 cache as the primary comparison. It tests 2, 3, 4, 6, and 8 draft tokens, then measures Q3 and Q2 separately. All evidence for this pass is under [`_temp_codex/qwen38_perf_round2`](../_temp_codex/qwen38_perf_round2).

### Final second-pass measurements

These are medians over three uncached prompts per context, with a 512-token generation budget, warm graphs, and the complete production integration. The reference restores the first-pass implementation's sampling, serial target verification, and MTP behavior in a separate process. It keeps the first-pass attention and MMVQ improvements. Q3/Q2 use the final production engine and two drafts.

| Engine/checkpoint | Drafts | Decode 2K, tok/s | Decode 20K, tok/s | Prefill 2K, tok/s | Prefill 20K, tok/s | Peak allocation, GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First-pass Q4 reference | 2 | 90.3 | 82.6 | 3,689 | 2,513 | 17.554 |
| Final Q4 | 2 | **102.3** | **93.9** | 3,648 | 2,504 | **17.559** |
| Final Q4 | 4 | 94.5 | 94.7 | 3,648 | 2,493 | 17.712 |
| Final Q3 | 2 | 109.8 | 97.2 | 3,739 | 2,545 | 13.872 |
| Final Q2 | 2 | 122.8 | 105.5 | 2,948 | 2,148 | 11.781 |

The retained second-pass changes add **13.7% median Q4 decode throughput at 20K** and 13.3% at 2K relative to the first-pass reference measured in this session. Prefill is essentially unchanged by this pass; the large prefill improvement came from the first-pass attention kernel. New sampler/refresh graphs cost about 5.8 MiB at two drafts, leaving most of the earlier 75 MiB state saving intact.

Four drafts are approximately 0.9% faster at 20K in this final story sample and slower at 2K. Earlier two-prompt story ablations showed a larger four-draft benefit; coding favored two. **Auto therefore stays at two drafts**, while explicit choices allow workloads that benefit from deeper speculation. Six and eight did not deliver a useful gain in the tested workloads. None of these controlled 20K runs reached 150 tok/s.

| Final configuration | 2K decode range | 20K decode range |
| --- | ---: | ---: |
| Q4, two drafts | 98.4–108.0 | 92.2–95.1 |
| Q4, four drafts | 91.4–94.6 | 93.8–99.4 |
| Q3, two drafts | 107.5–113.2 | 96.3–100.9 |
| Q2, two drafts | 120.4–129.0 | 101.9–108.1 |

Compared with Q4/two drafts, Q3 saves **3.69 GiB** and Q2 **5.78 GiB** of peak PyTorch allocation in this setup. These are checkpoint alternatives with different quantization quality, not additional same-quality Q4 memory savings. The metric excludes desktop/driver/other-process allocations; each engine reserves capacity for 32,768 tokens, with 20,000 actually occupied initially in the long cases.

Final production decode profiles put three-row Q4_K/Q6_K MMVQ at 44.63%/17.36% of GPU time, grouped attention at 9.64%, recurrent verification at 1.63%, and convolution verification at 0.32%. Q3's main IQ3_S MMVQ is 49.26%, with grouped attention at 10.80%. Q2's main IQ2_S MMVQ is 39.50%, with grouped attention at 11.59%. More drafts alone cannot remove these costs, and deeper target batches also change the selected matrix kernel.

Evidence: [`final_suite.json`](../_temp_codex/qwen38_perf_round2/final_suite.json); [`final_reference_d2/results.json`](../_temp_codex/qwen38_perf_round2/final_reference_d2/results.json); production results/profiles in [`final_q4_d2`](../_temp_codex/qwen38_perf_round2/final_q4_d2), [`final_q4_d4`](../_temp_codex/qwen38_perf_round2/final_q4_d4), [`final_q3_d2`](../_temp_codex/qwen38_perf_round2/final_q3_d2), and [`final_q2_d2`](../_temp_codex/qwen38_perf_round2/final_q2_d2). Each folder records checkpoint/native-binary metadata, exact prompt IDs/hashes, completions, VRAM, and cache alignment. All reported synchronization deltas are zero. GPU stage telemetry samples only some passes; diagnostic sampling passes run the filter without its graph to collect counters, so their sampling-stage time is not the optimized graph's timing. The separate torch-profiler traces execute outside that diagnostic mode.

### Additional retained optimizations

#### Multi-token convolution verification

New source: [`layers/speculative_state.py`](../shared/llm_engines/nanovllm/layers/speculative_state.py), `conv_verify_kernel` and `conv_verify`. Integration: `Qwen3_5Block._forward_linear_attention` in [`models/qwen3_5.py`](../shared/llm_engines/nanovllm/models/qwen3_5.py).

Previously, a verification pass with T inputs launched the one-token convolution T times, copied T cache snapshots separately, and concatenated the outputs. The new kernel processes all T inputs in one launch. Its grid covers channel blocks, batch items, and token positions. Each token gathers its causal width-4 window from the original convolution state and the new input sequence, computes convolution plus SiLU, and writes both its output and its post-token state snapshot.

The input cache is read-only during that launch. This avoids an inter-block race between a token still reading history and another token updating it. The existing caller copies the final snapshot into the live cache, and acceptance later selects the appropriate prefix snapshot. Parameters are 32 channels per block, the next power of two of the convolution width, and four warps. They do not branch on GPU model. Accumulation is FP32; input/output/state storage keeps the existing dtype. Only CUDA speculative verification using the existing FLA short-convolution path takes this branch. Existing single-token, prefill, and CPU paths retain their behavior.

#### Multi-token recurrent verification

The same new source supplies `recurrent_verify_kernel` and `recurrent_verify`. The old implementation launched the FLA recurrent kernel once per token, copied every state, and concatenated outputs. The new kernel keeps each state tile in FP32 registers while iterating over all verification inputs. It performs the same Q/K L2 normalization with epsilon 1e-6, key-dimension scaling, exponential decay, gated delta update, and output reduction. Every iteration writes a committable state snapshot directly into the existing buffer.

The launch uses eight value columns per tile, the next power of two of the key dimension, one warp, and three stages, following the existing FLA layout. Q/K heads can be shared by several value heads. Inputs are made contiguous once for the batch. The returned final state is a view of the last snapshot; no separate final-state allocation is retained. Snapshot dtype remains the caller's dtype. This preserves FP32 intermediate state across the token loop, as the serial reference does; it does not introduce per-token BF16 rounding of the carried state.

The kernels derive from FLA's MIT-licensed convolution and gated-delta implementations and include the license notice. Synthetic checks at the real 48-head, 128×128 state dimensions found exact convolution outputs/snapshots, maximum FP32 recurrent-state error 2.384e-7, and relative L2 error 7.4e-9 to 1.27e-8 across T=2,3,5,7,9. Low-precision snapshots agree within one ULP. Additional tests cover batch two, grouped heads, unequal key/value dimensions, FP16, and BF16. See [`fused_state_checks.json`](../_temp_codex/qwen38_perf_round2/fused_state_checks.json) and [`test_speculative_state.py`](../tests/test_speculative_state.py). Target verification GPU time in the two-draft ablation fell from approximately 17.67 to 16.51 ms; whole-run gains also depend on acceptance and host sampling.

#### Capture the exact compact sampling tail

Source: [`engine/model_runner.py`](../shared/llm_engines/nanovllm/engine/model_runner.py), `_speculative_distribution`, `_filter_speculative_distribution`, and `clear_graph_cache`.

Sampling contains many small GPU operations separated by Python launches. The retained change keeps the original top-k threshold and `nonzero` candidate compaction, including every token tied at the threshold, then captures the remaining min-p/nucleus/softmax/scatter operations in a CUDA graph. The graph cache holds at most eight entries, keyed by device, vocabulary size, compact candidate count, whether candidate IDs are present, top-p, and min-p. Each call copies current candidate values/IDs into the matching graph's buffers, replays it, and clones the output distribution so that drafting another token cannot overwrite an earlier proposal distribution still needed by rejection sampling.

The original filter operation order, compact sort size, nucleus cutoff, and tie behavior are retained. Logits processors, repetition penalties, temperature, acceptance/rejection, and RNG stay outside this graph and retain their rules. Eager execution, CPU execution, and sampled diagnostic-profile passes use the same extracted filter directly. Graph state is released with the engine's graph cache.

This design matters for quality. An earlier candidate padded the compact candidates to the full vocabulary to avoid dynamic shapes. It was faster, but 12 of 144 synthetic cases changed support at tied nucleus boundaries, with a maximum probability difference of about 0.0374. That candidate was rejected. PyTorch CUDA uses different small/large sorting implementations, so sorting padded arrays is not necessarily equivalent to sorting the compact array when ties occur; see [PyTorch 2.10 CUDA Sort.cu](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/Sort.cu). The retained compact graph matched **all 144 cases exactly**, with zero support changes, across FP32/BF16, vocabularies 257/98,304/248,320, and top-k disabled/20/100. Production tests separately compare graph and eager distributions and exercise cache eviction and clearing. Raw checks: [`compact_sampling_checks.json`](../_temp_codex/qwen38_perf_round2/compact_sampling_checks.json).

#### Refresh MTP from verified target states

Source: `ModelRunner._run_native_mtp`, `_advance_mtp`, `_run_mtp_forward`, and `capture_cudagraph`; removal of the obsolete reuse flag in [`qwen35_text.py`](../shared/prompt_enhancer/qwen35_text.py).

Recursive MTP drafting predicts both tokens and intermediate hidden states. Previously, accepted proposal-cache entries could be retained even though those entries had been calculated from predicted hidden states. A correct token does not imply that its predicted hidden state equals the target model's verified hidden state. This becomes more consequential with a longer draft chain.

The engine now truncates the MTP cache to its length before drafting and refreshes **all committed positions using verified target hidden states**. It batches this refresh into one MTP forward and computes logits only for the final position. In addition to the original single-token graph, it captures refresh graphs for lengths 2 through `draft_depth + 1`, sharing the existing MTP graph pool. It selects the next-token result from the graph that actually ran, then clones the final hidden state, logits, and next token for the next proposal. Cache length advances by the batch length. Graph keys, reuse, reset, and release include the selected depth and refresh graphs.

This follows the verified-state catch-up principle in [llama.cpp b10809's MTP implementation](https://github.com/ggml-org/llama.cpp/blob/b10809/common/speculative.cpp). Inspection of [its Qwen3.5 model implementation](https://github.com/ggml-org/llama.cpp/blob/b10809/src/models/qwen35.cpp) confirms the use of post-normalization hidden states; this work keeps that convention. Serial and batched refresh probes differed by less than 0.000222 nats in draft-logit KL. This is a proposal-state consistency fix, not a claim of a large independent speedup.

Target verification still controls every emitted token: sampled decoding retains rejection sampling and the residual distribution on rejection; greedy decoding retains exact token comparison. More drafts do not enable unchecked acceptance. The pre-existing 98,304-token draft-head prefix remains unchanged, while target verification still evaluates the full 248,320-token vocabulary.

#### Depth controls, memory allocation, and the repetition limit

The existing Configuration dropdown is now **Speculative Decoding (MTP)**, with Auto, Disabled, and Enabled with **2, 3, 4, 6, or 8 draft tokens**. The label no longer promises a universal 2× speedup. Auto retains the existing VRAM policy and selects two drafts when enabled. An explicit setting applies to both sampled and greedy drafting.

No config key or per-model setting was added. The shared key remains `wgp_config.json -> prompt_enhancer_speculative_decoding`. Its stored values deliberately preserve old configurations:

| Stored value | Meaning |
| ---: | --- |
| 0 | Disabled |
| 1 | Enabled with 2 draft tokens; legacy Yes |
| 2 | Auto; **not** an explicit two-draft value |
| 3, 4, 6, 8 | Explicit maximum draft count |

The choice list and normalization/validation are in [`shared/prompt_enhancer/config.py`](../shared/prompt_enhancer/config.py). [`loader.py`](../shared/prompt_enhancer/loader.py) applies the selected depth to the model before the runner is created. [`plugins/configuration/plugin.py`](../plugins/configuration/plugin.py) reuses the existing dropdown and save path. Its existing change handler discards the old runtime snapshot and reloads the engine when the MTP value changes, so graphs cannot retain the previous depth. The selector remains available for the existing supported 9B/27B model IDs; 4B remains unsupported. The local real-MTP tests cover 27B; the missing 9B MTP sidecar remains a validation limitation.

`ModelRunner` supports up to eight drafts but allocates per-instance buffers and graphs only for the selected maximum. Target snapshot capacity, CPU input staging, position telemetry arrays, graph cache keys, capture signatures, verification graphs, and MTP refresh graphs all follow that value. Selecting two therefore does not allocate eight-token state. One extra verification state on this model is approximately 75 MiB; graph allocations add a smaller amount. The measured two-to-four increase is approximately 156 MiB, and two-to-eight approximately 481 MiB.

Eight drafts first exposed an existing six-token limit in the sparse repetition update. [`layers/sampler.py`](../shared/llm_engines/nanovllm/layers/sampler.py), `apply_sparse_repetition_penalty_`, now processes additional **virtual** repetition IDs in chunks of six after the normal persistent/new/first-virtual kernel call. The caller already supplies unique virtual IDs absent from persistent history. Additional calls use zero persistent/new counts, so previously penalized IDs are not penalized again. The existing limit for newly committed persistent IDs is retained; its caller already rebuilds that cache for larger increments. The default two-draft fast path still uses one launch. CPU/CUDA tests cover virtual counts 0, 6, 7, 8, and 15 against the reference penalty.

The actual shared loader was exercised at depths four and eight, with repeated prompts, `CUDA_LAUNCH_BLOCKING=1`, and assertions on selected depth, refresh graph lengths, telemetry array length, and zero cache synchronization delta. The four-draft case covered both 2K and 20K prompts; eight covered 20K. These synchronized-debug runs are correctness checks and are excluded from throughput comparisons. Logs: [`loader_d4.log`](../_temp_codex/qwen38_perf_round2/loader_d4.log), [`loader_d8.log`](../_temp_codex/qwen38_perf_round2/loader_d8.log).

### Other approaches tested and discarded

These exploratory ranges use two 20K prompts, ordinarily 512 generated tokens, and are not a single controlled ablation across the entire session. Unchanged prefill varied from approximately 2,530 to 2,850 tok/s between groups; final production/reference measurements are therefore reported separately. Full commands, repeats, acceptance counts, and memory are preserved in the linked suite JSON files.

| Experiment | Depth | Decode range, tok/s | Decision |
| --- | ---: | ---: | --- |
| More drafts before the second-pass changes | 3 / 4 / 6 | 81.3–81.9 / 78.8–86.0 / 78.0–89.0 | Increasing depth alone did not help reliably |
| Serial refresh of verified MTP states | 4 | 81.2–90.1 | Corrected state provenance but adds serial work |
| Batched verified-state refresh alone | 2 / 3 / 4 / 6 | 88.0–88.3 / 87.4–92.1 / 83.3–91.2 / 85.7–88.1 | Retained for state consistency and cheaper refresh |
| Full-vocabulary padded sampling | 2 / 4 | 96.2–102.1 / 83.8–90.2 | Rejected: changes tie behavior |
| CUDA graphs around that padded sampler | 2 / 4 / 6 / 8 | 103.7–111.3 / 90.2–98.0 / 90.5–103.6 / 74.5–80.4 | Rejected despite speed: same tie problem |
| Greedy proposals with exact sampled target rejection | 2 / 4 / 6 | 96.1–97.6 / 94.1–100.9 / 85.7–87.3 | No compelling gain over the retained combination |
| Greedy proposals plus batched refresh | 4 / 6 | 91.1–93.3 / 85.7–93.5 | Not retained |
| Exact compact sampling graphs | 2 | 100.0–107.8 | Retained; distribution checks exact |
| Fused states plus exact compact sampler | 2 / 4 / 6 / 8 | 99.1–105.2 / 105.0–106.7 / 88.2–94.0 / 80.3–84.8 | Retained kernels; high depth remains expensive |
| Same fused/compact combination, coding task | 2 / 4 / 6 | 116.6–118.4 / 109.5–117.5 / 111.0–112.8 | Two drafts wins this workload |
| Complete combination with verified refresh, 1,024-token runs | 2 / 4 | 101.7–102.6 / 100.5–106.2 | Synchronized throughout; retained integration |

The coding task requests a Python inventory CLI and twenty unit tests, with thinking still enabled and 20,000 occupied prompt tokens. It is specified in [`coding_prompt.txt`](../_temp_codex/qwen38_perf_round2/coding_prompt.txt). It is not substituted for the story task to inflate the principal result. Speculative acceptance and serial proposal cost both matter: later drafts may be correct, but they must save more target passes than their additional MTP and wider verification work costs.

Sources: [`suite.json`](../_temp_codex/qwen38_perf_round2/suite.json), [`followup_suite.json`](../_temp_codex/qwen38_perf_round2/followup_suite.json), [`third_suite.json`](../_temp_codex/qwen38_perf_round2/third_suite.json), [`fourth_suite.json`](../_temp_codex/qwen38_perf_round2/fourth_suite.json), and [`fifth_suite.json`](../_temp_codex/qwen38_perf_round2/fifth_suite.json). Failed exploratory checks were retained too: the initial eight-draft repetition error was fixed directly; an initially bitwise recurrent-state assertion was investigated against the FP32 reference, rather than treating a 2.384e-7 near-zero difference as cache corruption.

### Q3/Q2 checkpoint and prefill investigation

All three runs use the same model architecture and tokenizer. Their weight formats differ:

| Label | Local checkpoint | File bytes | MTP weights |
| --- | --- | ---: | --- |
| Q4 | `Qwen3.8-27B-Uncensored-Q4_K_M.gguf` | 16,810,714,528 | Embedded |
| Q3 | `Qwen3.8-27B-Uncensored-noMTP-IQ3_S.gguf` | 12,588,187,168 | Separate `Qwen3.8-27B-Uncensored-MTP-Q4_K_M.gguf`, 274,258,176 bytes |
| Q2 | `Qwen3.8-27B-Uncensored-IQ2_M.gguf` | 10,624,771,968 | Embedded |

The GGUF inventory was read from the actual downloaded files, rather than inferred from their names. [Full inventory](../_temp_codex/qwen38_perf_round2/quantization_inventory.json):

- Q4 contains approximately 21.800 billion Q4_K parameters and 5.518 billion Q6_K parameters, plus F32 tensors. Its Q6_K payload alone is about 4.527 GB.
- Q3's main file contains approximately 21.750 billion IQ3_S parameters, 3.872 billion Q4_K parameters, and a 1.271-billion-parameter Q6_K output head, plus F32 tensors. The MTP sidecar is additional.
- Q2 contains approximately 20.534 billion IQ2_S parameters, 1.216 billion IQ3_S parameters, 3.872 billion Q4_K parameters, a Q5_K output head, Q8_0 tensors, and F32 tensors.

Thus “Q2”, “Q3”, and “Q4” are mixed checkpoint formats, not a uniform two-, three-, or four-bit multiply throughout the model. Lower packed byte count helps memory bandwidth, but lookup/dequantization cost, matrix kernel efficiency, attention, the output head, sampling, and MTP acceptance remain measurable costs. The measured Q2 prefill result specifically contradicts a monotonic “fewer bits must always be faster” assumption; this is a kernel/profile finding, not a reason to stop optimizing Q4.

The 1,024-token suffix prefill profiles over a 20K prefix show:

| GPU work | Q3 share/time | Q2 share/time |
| --- | ---: | ---: |
| Q8 paged prefix attention, 16 layers | 51.48%; 247.7 ms | 46.85%; 248.2 ms |
| Main unfused IQ matrix kernels | IQ3_S: 32.92%; 158.5 ms / 224 calls | IQ2_S: 36.75%; 194.7 ms / 200 calls |
| Q4_K matrix kernels | 4.52%; 21.7 ms | 3.74%; 19.8 ms |
| Fused IQ matrix kernels | IQ3_S: 2.39%; 11.5 ms | IQ2_S: 3.04%; 16.1 ms |

These shares are profiler GPU time, not total prefill wall time. Q2's smaller packed weights still take longer in the principal prefill matrix kernels, while attention time is essentially unchanged. Sources: [`q3_prefill/prefill_profile.txt`](../_temp_codex/qwen38_perf_round2/q3_prefill/prefill_profile.txt), [`q2_prefill/prefill_profile.txt`](../_temp_codex/qwen38_perf_round2/q2_prefill/prefill_profile.txt).

The existing dequantize-plus-cuBLAS prefill mode was also tested. At 20K, Q2 prefill increased from approximately 2,480 to 2,668 tok/s, while peak allocation rose from 11.775 to 12.019 GiB, about 250 MiB. Q4 slowed from approximately 2,840 to 2,646 tok/s. That existing mode uses FP16 accumulation; it was not selected, and no new precision-reducing prefill path or fallback was added. Commands/results are in [`third_suite.json`](../_temp_codex/qwen38_perf_round2/third_suite.json).

The earlier decode profiles show the main three-row IQ3_S MMVQ at 45.31% of GPU time and IQ2_S MMVQ at 37.01%, with grouped attention at 10.21% and 11.70%, respectively. Both still execute higher-bit matrices. Final production profiles are collected separately after timed runs. Q3/Q2 are profiled as alternative checkpoints; this report does **not** claim that they preserve Q4's language quality. The primary Q4 optimization retains the Q4 checkpoint and Q8 cache.

### Second-pass numerical and runtime validation

[`quality_results.json`](../_temp_codex/qwen38_perf_round2/quality_results.json) compares the first-pass implementation against the retained second-pass implementation at four drafts. Both score the same 256 teacher-forced continuation tokens per context, using the real Q4 checkpoint and speculative target path. Raw logits are saved before forcing the target token.

| Metric | 2,048-token prompt | 20,000-token prompt |
| --- | ---: | ---: |
| Finite logits | All | All |
| Highest-probability token agreement | 99.22% | 99.61% |
| Mean KL, nats | 0.000929 | 0.000843 |
| Reference cross-entropy, nats/token | 0.388864 | 0.364866 |
| Production cross-entropy, nats/token | 0.388003 | 0.361838 |
| Production/reference perplexity ratio | 0.999140 | 0.996976 |

Continuation loss did not increase in this check. Floating-point reduction order can still alter sampled continuations; bit-identical generated text is not promised. These 512 positions are a numerical regression check, not a broad evaluation proving equal quality for every task. The exact sampler checks and target rejection algorithm are separate safeguards against changing the intended sampling distribution.

The production test command passed **56 tests**, including the Q8 attention suite, prefix snapshot checks, sparse repetition boundaries, graph sampler equality, graph cache clearing, MTP refresh token selection, and config compatibility:

```powershell
$env:CUDA_LAUNCH_BLOCKING = '1'
& C:\Users\Marc\anaconda3\envs\py311\python.exe -m pytest -q tests/test_speculative_state.py tests/test_nanovllm_speculative_sampling.py tests/test_prompt_enhancer_speculative_config.py tests/test_q8_paged_attention.py
Remove-Item Env:CUDA_LAUNCH_BLOCKING
```

Log: [`production_unit_tests.log`](../_temp_codex/qwen38_perf_round2/production_unit_tests.log). The existing deprecation warnings are retained. No MMGP, `wgp.py`, or other model-family source was changed. The new verification kernels are generic Triton code; only the installed RTX 5090 compiled them during this work. No additional native binary or other-GPU kernel build was required in the second pass.

Two additional four-draft runs generated **4,096 and 4,097 tokens** from separate 20K prompts, ending at occupied lengths 24,096 and 24,097. Both retained zero MTP synchronization delta, with peak allocation stable at approximately 17.712 GiB. Their averages were 82.4 and 87.2 tok/s over the longer continuation, showing why the initial 512-token rate should not be presented as guaranteed sustained throughput. The text transitions from planning into coherent story prose. This is a cache/page-boundary stress test, not a completed ten-chapter story or a broad language-quality evaluation. Evidence: [`long_q4_d4/results.json`](../_temp_codex/qwen38_perf_round2/long_q4_d4/results.json) and the adjacent saved completions.

### Total improvement confirmation

An additional three-prompt 20K block restored all pre-task components together: original Q8 attention, original speculative-state layout, original sampler/MTP behavior, and the backed-up native MMQ-only binary. It then ran the final two- and four-draft implementations. All changes to reference behavior were process-local.

| Configuration | Median decode, tok/s | Decode range | Median prefill, tok/s | Peak allocation, GiB |
| --- | ---: | ---: | ---: | ---: |
| Pre-task Q4, two drafts | 64.3 | 54.9–64.7 | 1,195 | 17.627 |
| Final Q4, two drafts | 101.3 | 100.7–102.1 | 2,743 | 17.559 |
| Final Q4, four drafts | 90.8 | 88.5–91.2 | 2,452 | 17.712 |

For this block alone, two-draft median decode improved by approximately 58% and prefill by 2.30×; retained allocation fell by approximately 69.5 MiB. The absolute rates shifted between blocks, including the unchanged prefill path, so these percentages should not be combined with the first-/second-pass percentages or treated as universal gains. The earlier complete Q4/Q3/Q2 table remains the direct cross-checkpoint comparison. All three repetitions, including the low original decode result, are retained in [`total_suite.json`](../_temp_codex/qwen38_perf_round2/total_suite.json).

### Actual Deepy tool output above 20K

The real `wgp.py --ask-deepy` path was tested with four drafts, first at approximately 8K–10K context, then above 20K. The task writes a small JSON file, reads it back, changes one Boolean field in a second user turn, and verifies the file again. This exercises grammar-constrained tool arguments, tool results appended to the live context, repeated prompts, runtime snapshots, and target/MTP state reuse. The final on-disk objects were checked programmatically. All runs reported zero MTP synchronization delta.

For the long-context case, 12,064 tokens of the same public-domain prose were added as reference-only material to the actual approximately 8K Deepy system/tool prompt. Logged occupied contexts were **20,254–23,667** with four drafts and **20,299–23,265** with two. These interactive-runtime tests use a 32,000-token capacity hint, versus 32,768 in the controlled benchmark. Thinking stayed enabled; Q4 weights and the Q8 cache were retained. The two-draft run uses the same final kernels and the same task, with the requested draft count in the JSON changed accordingly.

| Actual Deepy phase, above 20K | Two drafts | Four drafts |
| --- | ---: | ---: |
| Thinking, aggregate decode | 112.1 tok/s over 1,475 tokens | 108.8 tok/s over 1,469 tokens |
| Tool arguments, aggregate decode | **128.6 tok/s over 288 tokens** | **160.2 tok/s over 454 tokens** |
| Tool-argument span range | 102.1–141.8 tok/s | 136.5–191.5 tok/s |
| Statements, aggregate decode | 114.5 tok/s over 298 tokens | 126.0 tok/s over 303 tokens |
| All decode phases combined | 114.5 tok/s over 2,061 tokens | 118.8 tok/s over 2,226 tokens |

Aggregate rates divide total emitted tokens by summed action decode time; they exclude prefill, loading, and file-tool execution time. The action durations in the log are rounded to milliseconds. The model generated its own tool sequence, so the two runs have different token counts and are not an identical-token ablation. Four drafts improve the predictable tool-output phase by about 25% on this task and pass 150 tok/s there, while thinking does not improve. This is the concrete reason to expose four drafts without changing Auto for all requests. The 191 tok/s value is a short tool-argument span, not a sustained story or whole-request rate.

At 8K–10K, four-draft tool arguments averaged 186.3 tok/s across 351 tokens, with a 204.1 tok/s peak span. Those shorter-context results are kept separately. Sources: [`deepy_tools_summary.json`](../_temp_codex/qwen38_perf_round2/deepy_tools_summary.json), [`deepy_tools_20k.log`](../_temp_codex/qwen38_perf_round2/deepy_tools_20k.log), [`deepy_tools_20k_d2.log`](../_temp_codex/qwen38_perf_round2/deepy_tools_20k_d2.log), and [`run_deepy_tools.py`](../_temp_codex/qwen38_perf_round2/run_deepy_tools.py).

The actual Configuration tab was also served on isolated port **7865**. Its serialized dropdown contained Auto/Disabled/2/3/4/6/8, with four selected, interactive, and visible. Saved component evidence: [`ui/dropdown.json`](../_temp_codex/qwen38_perf_round2/ui/dropdown.json). The test server was stopped. The UI fixture initially omitted the application's required `attention_mode` key; copying the existing config into the isolated fixture fixed that setup failure without changing application code. An earlier reference-harness relative-import failure was likewise confined to the test harness and fixed before the successful measurements.

### Source snapshots and reproduction

The requested **1.0.16** source copy remains frozen at the completed first pass. The additional kernels and full final integration are preserved separately as **1.0.17**, after checking that version was unused:

- [1.0.16 source-copy notes](../../kernels/llama.cpp-1.0.16/SOURCE_COPY.md) and [original ZIP](../../kernels/llama.cpp-1.0.16-source.zip).
- [1.0.17 source-copy notes](../../kernels/llama.cpp-1.0.17/SOURCE_COPY.md), [complete source ZIP](../../kernels/llama.cpp-1.0.17-source.zip), and [ZIP SHA-256](../../kernels/llama.cpp-1.0.17-source.zip.sha256).

The new snapshot contains the complete native project/vendor sources plus a `wangp/` overlay with all nine production files, all four affected test files, the benchmark, and this report. Its manifest records file hashes and source origins; version metadata is changed only inside the copy. **The installed native package is still 1.0.14+mmvq.** Neither source snapshot is a newly built, installed, or published wheel. The second pass needs the updated WanGP source and already installed native dispatch, not another native compilation. Restart WanGP to load the updated Python engine and dropdown.

Reproduce the final Q4, Q3, and Q2 cases with the same actual-context protocol:

```powershell
$python = 'C:\Users\Marc\anaconda3\envs\py311\python.exe'
$assets = 'E:\ML\wan2gp\ckpts\Qwen3_8_27B_Uncensored'
$corpus = '_temp_codex\qwen38_perf_20260905\corpus.txt'
& $python tools\benchmark_qwen38_engine.py --assets $assets --corpus $corpus --output _temp_codex\q4_recheck --contexts 2048 20000 --repeats 3 --tokens 512 --draft 2
& $python tools\benchmark_qwen38_engine.py --assets $assets --corpus $corpus --output _temp_codex\q4_d4_recheck --contexts 2048 20000 --repeats 3 --tokens 512 --draft 4
& $python tools\benchmark_qwen38_engine.py --assets $assets --corpus $corpus --checkpoint Qwen3.8-27B-Uncensored-noMTP-IQ3_S.gguf --output _temp_codex\q3_recheck --contexts 2048 20000 --repeats 3 --tokens 512 --draft 2
& $python tools\benchmark_qwen38_engine.py --assets $assets --corpus $corpus --checkpoint Qwen3.8-27B-Uncensored-IQ2_M.gguf --output _temp_codex\q2_recheck --contexts 2048 20000 --repeats 3 --tokens 512 --draft 2
```

Add `--profile decode` or `--profile prefill` for a separate post-timing profile. `--tokens 4096` reproduces the longer cache stress case. The benchmark's `--draft` takes the actual count, unlike the legacy shared config value 2, which means Auto. `run_reference.py` restores the first-pass methods; `run_original_all.py` additionally restores the pre-task attention, state layout, and backed-up native binary for the full comparison. These overrides are process-local and do not replace the installed files.

`run_final_suite.py` records the production/config/quality/profile sequence in `final_suite.json`; successful cases are skipped when that script is resumed. Use fresh output locations for independent repetitions. The quality harness and comparison are `run_quality_round2.py` and `compare_quality.py`; `summarize_round2.py` condenses the experiment records without discarding raw per-prompt results. The source-copy procedure is [`create_source_copy.py`](../_temp_codex/qwen38_perf_round2/create_source_copy.py).

### Follow-up: chat streaming recovery and source snapshot 1.0.18

A reported approximately 20-second pause in visible thinking text, while decoding continued, led to a reproducible failure in the browser event consumer. After a missing text-event sequence, `WAC.consumePayload` silently discarded subsequent `append_block_text` and `replace_block_text` events until a structural event, such as thought finalization, arrived. A reproduction delivered 80 further text updates with none applied and no recovery requested. This establishes a matching failure mechanism; the specific reported incident was not captured in a browser trace.

The fix removes that single early-return line in [`shared/gradio/assistant_chat.py`](../shared/gradio/assistant_chat.py). Text-event gaps now reach the existing `markSyncRequired` path. Its existing pending-request flag allows one canonical recovery request at a time, and normal incremental streaming resumes when the sync arrives. The regular 250 ms emission interval, decoder, sampling rules, kernels, and GPU allocations are unchanged. Recovery uses CPU/network work only after a detected gap; it is not a claim that a full recovery sync has zero cost.

All **16** tests in [`tests/deepy_streaming_protocol.test.js`](../tests/deepy_streaming_protocol.test.js) passed. The two affected regression cases failed before the fix and passed afterward. They cover recovery during an unfinished thought, 80 subsequent updates without duplicate recovery requests, replacement-event gaps, and the existing duplicate/coalesced-event behavior. These are tests of the production JavaScript functions in Node; no new GPU benchmark or end-to-end browser timing is claimed. Restart WanGP and reload the browser to load the changed embedded JavaScript.

The complete source is preserved as **1.0.18**: [source-copy notes](../../kernels/llama.cpp-1.0.18/SOURCE_COPY.md), [source ZIP](../../kernels/llama.cpp-1.0.18-source.zip), and [SHA-256](../../kernels/llama.cpp-1.0.18-source.zip.sha256). It contains the complete native project and vendor sources, ten WanGP production files, five affected test files, the benchmark, and this report. The addition since 1.0.17 is the chat recovery fix and its tests/documentation; the performance kernels are unchanged. Earlier snapshots remain frozen, and no wheel was built, installed, or published for this source-copy request.

## Third pass: fewer conversions, better reuse, and one fewer state snapshot

This pass, measured on 5 September 2026, starts from the completed second-pass engine and the 1.0.18 source snapshot. Q4 weights, Q8 KV storage, sampling parameters, repetition penalties, MTP confidence, and draft-depth configuration are unchanged. The algorithms use tensor dimensions and existing architecture-aware dispatch; they do not contain an RTX 5090 model-name special case. Native compilation and Triton runtime validation were limited to the local SM120 GPU.

### Complete retained implementation

1. **Quantize MMVQ inputs directly from their original dtype.** In native [`csrc/gguf_llamacpp_kernels.cu`](../../kernels/llama.cpp/csrc/gguf_llamacpp_kernels.cu), `quantize_mmvq_q8_1_typed` and its launcher accept FP32, FP16, or BF16. The former MMVQ path first allocated an FP32 activation tensor and launched a conversion, then quantized it to Q8_1. The new path loads each original element into FP32 inside the quantizer. It retains the same warp maximum and sum, scale `amax / 127`, `roundf(x / scale)`, zero-block behavior, half2 scale/sum storage, row padding, and CUDA stream/launch helpers. `run_linear_cuda` now supplies the packed Q8 input directly to the existing MMVQ entry point; the logical FP32 GGML descriptor supplies shape metadata, and the unused FP32 data argument is null. The MMQ prefill path was already typed and is unchanged. This removes one launch and one temporary FP32 activation allocation per qualifying linear call, without retaining unpacked weights or changing activation quantization.

2. **Reuse packed weight work across verification columns.** In native [`mmvq.cu`](../../kernels/llama.cpp/_vendor/llama.cpp/ggml/src/ggml-cuda/mmvq.cu), the two unrolled inner loops now visit a weight row before iterating over destination columns. This gives the compiler an opportunity to reuse weight loads/dequantization across draft columns. Each output still accumulates the same dot products in the same order. Dispatch thresholds, weight formats, accumulation precision, and supported batch sizes are unchanged. The separate five-row Q4 experiment increased median 20K decode from 109.75 to 112.79 tok/s relative to the typed-input candidate without this loop change; it is supporting evidence, not an additional percentage to multiply into the final comparison.

3. **Fuse SiLU, multiplication, and contiguous output.** [`layers/activation.py`](../shared/llm_engines/nanovllm/layers/activation.py), `_silu_mul_kernel` and `SiluAndMul.forward_list`, replace separate CUDA SiLU, multiply, and output-copy work with one Triton launch. The calculation explicitly rounds the SiLU result to the output dtype **before** multiplication, preserving the previous intermediate FP16/BF16 rounding. It uses the same float exponential/division expression, rather than a lower-precision approximate sigmoid. Input-list ownership/clearing is preserved. Noncontiguous input is made contiguous once; normal contiguous model input needs no such copy. The ordinary `forward` method and existing CPU/no-Triton implementation are retained. This adds no persistent buffer, new user option, or exception-based fallback. Only Qwen3.5-family code currently calls `forward_list`.

   The final cleanup makes `TOTAL` a runtime kernel argument instead of a compile-time constant. New prompt lengths therefore reuse the same compiled activation kernel; feature width and block width remain specialized. The actual Deepy test had exposed repeated approximately 100 ms compilations for fresh prefill lengths. Across seven lengths, two model feature widths, and three dtypes, the cleanup reduced seven compiled variants to one per width/dtype, with all 42 outputs matching exactly. Warm kernel timings were essentially unchanged at the measured microsecond scale. This avoids a newly introduced first-use prefill cost; it is not another steady-state prefill speedup. Evidence: [`dynamic_activation_checks.json`](../_temp_codex/qwen38_perf_round3/dynamic_activation_checks.json).

4. **Fit grouped verification attention into one query tile.** In [`layers/attention.py`](../shared/llm_engines/nanovllm/layers/attention.py), `_q8_grouped_attention` uses a 32-row query tile when the grouped query count exceeds 16; smaller groups retain 16. With this model's six query heads per KV head, three verification inputs give 18 grouped rows and five give 30, so both now fit one tile. The previous two 16-row tiles read the same packed KV pages twice. Key tiles remain 32, with four warps and one stage. The split-count calculation deliberately retains the previous 16-row work estimate, so the change does not increase split scratch allocation. Q8 scales, masks, high/low BF16 decomposition, softmax, FP32 accumulation, and final reduction are unchanged. Isolated 20K tests reduced three-query attention from about 134 to 104 microseconds and five-query attention from 131 to 104 microseconds. Those are layer-kernel timings, not whole-engine gains.

5. **Write final verification state in place and retain only rollback prefixes.** [`layers/speculative_state.py`](../shared/llm_engines/nanovllm/layers/speculative_state.py), [`models/qwen3_5.py`](../shared/llm_engines/nanovllm/models/qwen3_5.py), and [`engine/model_runner.py`](../shared/llm_engines/nanovllm/engine/model_runner.py) now allocate `D` snapshots for a configured maximum of `D` drafts, instead of `D + 1`. A verification of `T` inputs writes its first `T - 1` states to prefix snapshots and its final state directly to the live cache. Recurrent state still stays in FP32 registers throughout the token loop; only stored states use the existing cache dtype. Value/head tiles own disjoint state elements, so the final write needs no extra buffer.

   The convolution kernel now processes all tokens for a channel within one program, instead of launching independent token programs. That ordering is necessary: independent token programs could read initial convolution history while another overwrote it with the final state. The final store occurs only after all outputs for that channel have read their history. Bias, window arithmetic, activation, and stored dtype are unchanged. Both verification helpers return the live final-state tensor. Existing caller copies therefore become self-copies on this path.

   If all actually verified inputs are committed, the runner skips state restoration entirely. If a suffix is rejected, it copies the accepted prefix from the existing batched snapshot views. The commit API receives **actual verification length**, not just configured capacity: this handles shortened draft batches without reading stale snapshots. Capacity checks and the existing nonfused verification paths were updated for the same `T - 1` contract. No MTP target-state refresh or rejection-sampling rule changed.

   For 48 linear-attention layers, one recurrent snapshot is `48 × 48 × 128 × 128 × 2 = 75,497,472` bytes; one convolution snapshot is `48 × 10,240 × 4 × 2 = 3,932,160` bytes. This removes **79,429,632 bytes = 75.75 MiB** of persistent state at every enabled draft depth. At two drafts, prefix storage falls from 227.25 to 151.5 MiB; at four, from 378.75 to 303 MiB. The live cache remains separate. This is additional to the different unused snapshot removed in the first pass. Whole-model peak differences also include temporary allocations and allocator rounding.

### Controlled third-pass results

| Checkpoint / drafts | Before decode, tok/s | After decode, tok/s | Median matched-prompt gain | Before peak, GiB | After peak, GiB | Saved, MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4 / 2 | 92.59 | 100.17 | +8.28% | 17.559 | 17.485 | 76.1 |
| Q4 / 4 | 96.03 | 105.81 | +6.59% | 17.712 | 17.634 | 80.1 |
| Q3 / 2 | 97.26 | 105.51 | +8.49% | 13.872 | 13.797 | 76.9 |
| Q2 / 2 | 105.32 | 113.66 | +8.19% | 11.781 | 11.706 | 76.8 |

| Checkpoint / drafts | Before prefill, tok/s | After prefill, tok/s |
| --- | ---: | ---: |
| Q4 / 2 | 2514 | 2562 |
| Q4 / 4 | 2542 | 2678 |
| Q3 / 2 | 2556 | 2562 |
| Q2 / 2 | 2138 | 2149 |

Source: [`comparison.json`](../_temp_codex/qwen38_perf_round3/comparison.json) and [`state_suite.json`](../_temp_codex/qwen38_perf_round3/state_suite.json).

Every final pair uses three recorded 20,000-token prompts, 512 requested output tokens, seed 123, temperature 0.6, top-p 0.95, top-k 20, min-p 0.05, repetition penalty 1.05, thinking enabled, Q8 KV, and 32,768 capacity. Baselines restore the previous native binary and Python activation/attention/state implementations inside the benchmark process. Q4/four-draft and Q2 pairs reverse the before/after execution order. The reference components are copied files, not reconstructions of the old arithmetic. `state_suite.json` records commands and state-source hashes; complete text and per-prompt results are preserved beside each log.

The speedup column is the **median of three matched-prompt ratios**. Before/after throughput columns are independently calculated medians, so their quotient can differ from that column. Peak VRAM means maximum PyTorch allocated memory across those runs, not total driver reservation or all memory occupied by the desktop/application. Do not multiply these percentages by earlier session percentages.

A preliminary block, before the final-state memory change and native loop reordering, found Q4/two-draft 20K gains of 5.39%, then 5.76% in a separate repeat. Q3 improved 5.26%. Q2 was mixed: -2.01% at 20K and -9.23% at 2K in that preliminary block; its prefill also shifted substantially despite no corresponding prefill-kernel change. Those results remain in [`comparison.json`](../_temp_codex/qwen38_perf_round3/comparison.json), rather than being discarded. Absolute rates varied with session conditions: GPU monitoring recorded substantial SM-clock movement during sustained load. The final paired block measures the complete arithmetic/state changes before the subsequent activation compile-key cleanup; final installed-code checks also cover that cleanup below; these small samples do not establish a universal gain on every prompt or GPU.

### Prefill investigation and rejected experiments

No additional large prefill acceleration is claimed. The previous approximately 2× prefill gain remains a first-pass result. In this pass's Q4 prefix profile, grouped Q8 prefill accounted for approximately 50.1% of recorded GPU time, Q4 MMQ for 29.8%, and Q6 MMQ for 9.5%; SiLU was about 0.6%. The retained typed quantizer and MMVQ loop change primarily affect decode. State verification changes do not rewrite the ordinary prefill algorithm.

The 1,024-query/20K-prefix attention sweep tried 24 tile/warp/precision combinations. The existing 32×32, four-warp BF16 high/low path was fastest in that sweep at about 14.9 ms per attention layer; alternative BF16 shapes took roughly 17–34 ms and TF32x3 variants roughly 18–35 ms. Wider decode key tiles also increased register spilling and regressed. A direct FP32 vector attention experiment took approximately 1.2–1.5 ms versus about 0.13 ms for grouped tensor-core attention in the checked case. These exploratory implementations were not promoted. TF32x3 was tested, sometimes improved numerical closeness to the FP32 reference, but did not deliver a useful speed gain and was not retained.

The installed four-draft decode profile still spends about 46.35% of GPU time in the principal five-row Q4 MMVQ kernel and 16.65% in five-row Q6 MMVQ, versus 7.17% in grouped Q8 attention. Weight-matrix work remains the main bottleneck after these changes; attention or streaming tweaks alone cannot turn the measured sustained story rate into 150 tok/s. See [`production decode profile`](../_temp_codex/qwen38_perf_round3/production_long_q4_d4/decode_profile.txt). This profile precedes only the activation compile-key cleanup, which changes compilation reuse rather than model arithmetic.

The repetition-penalty path was also inspected. The suspect scalar read was already on a CPU tensor; it did not establish a device-synchronization bottleneck. Penalty scope, values, and application frequency remain unchanged. This pass does not attribute tool speed to disabling repetition penalties, change chat refresh frequency, increase draft depth, or add new config keys.

Evidence: [`prefill_profile.txt`](../_temp_codex/qwen38_perf_round3/typed_silu_q4_d2/prefill_profile.txt), [`prefill_tuning.json`](../_temp_codex/qwen38_perf_round3/prefill_tuning.json), [`attention_tuning_narrow.json`](../_temp_codex/qwen38_perf_round3/attention_tuning_narrow.json), [`attention_tuning_tf32.json`](../_temp_codex/qwen38_perf_round3/attention_tuning_tf32.json), and [`attention_vector_results.json`](../_temp_codex/qwen38_perf_round3/attention_vector_results.json). Partial/aborted exploration logs are kept separately; they are not counted as complete sweeps.

### Numerical checks and actual runtime validation

The final native binary matched the previous binary exactly in **270 real-checkpoint linear comparisons**: five matrices from each of Q4/Q3/Q2, six batch sizes (1, 2, 3, 5, 8, 16), and FP32/FP16/BF16 inputs. This includes Q4_K, Q6_K, IQ3_S, IQ2_S, and Q5_K output-head work and all-zero quantization blocks. Results: [`interchange_mmvq_results.json`](../_temp_codex/qwen38_perf_round3/interchange_mmvq_results.json). The earlier typed-only candidate passed the same 270 checks independently. These are output comparisons, not a claim that every possible input was enumerated.

The fused activation matched the previous PyTorch calculation for all **65,280 finite BF16 values** and **63,488 finite FP16 values**, with recorded random multipliers, plus random FP32/FP16/BF16 tensors at decode and prefill shapes. This explicitly checks intermediate rounding. [`activation_results.json`](../_temp_codex/qwen38_perf_round3/activation_results.json) records the comparisons; [`test_fused_silu_mul.py`](../tests/test_fused_silu_mul.py) also covers strided input, CPU execution, list clearing, and CUDA graph replay.

The in-place state change passed **38 additional exact checks** against the copied previous kernels: recurrent output/prefix/final state across T=1,2,3,5,7,9 and three dtypes; convolution across T=2,3,5,9, FP16/BF16, bias/no bias, and masked tail channels; three CUDA graph replays with initial-state reset; and every commit length in shortened verification batches at draft capacities 1,2,4,8. Unused test snapshots contain NaNs to catch stale-prefix selection. The existing FLA-reference tests remain in place with their original numerical tolerances.

The first real-model in-place trial exposed a capacity guard that still required the old extra snapshot. It was corrected to require `seq_len - 1`, then Q4/four-draft passed repeated 2K/20K prompt calls with `CUDA_LAUNCH_BLOCKING=1`. This was an explicit runtime error, not a silent recovery path. A separate initial test-fixture dtype mismatch was corrected without changing production math or relaxing the original FLA-reference tolerances.

The installed native binary and promoted Python source passed **64 production tests** with `CUDA_LAUNCH_BLOCKING=1`: state/commit/sampling, shared MTP config compatibility, Q8 attention, and fused activation. All **48 matched completion comparisons** in the preliminary/repeat/final blocks, including six final Q4 calls after the compile-key cleanup, had byte-identical text, identical speculative statistics, and zero MTP synchronization delta. This includes all 12 full-combination comparisons across Q4/two drafts, Q4/four drafts, Q3/two drafts, and Q2/two drafts. No before/after result failed that comparison. After the activation compile-key cleanup, all eight affected activation tests passed again, and both Deepy tool workflows were rerun with that final source.

Logs: [`production_tests.log`](../_temp_codex/qwen38_perf_round3/production_tests.log), [`state_exact_tests.log`](../_temp_codex/qwen38_perf_round3/state_exact_tests.log), [`state_runtime_check.log`](../_temp_codex/qwen38_perf_round3/state_runtime_check.log), and [`promoted_sources.json`](../_temp_codex/qwen38_perf_round3/promoted_sources.json).

Two installed-code Q4/four-draft continuations stressed the live cache beyond 24K occupied tokens, reusing the engine for the second prompt:

| Repeat | Generated tokens | Final occupied context | Decode, tok/s | Peak allocated, GiB | MTP sync delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 24096 | 88.22 | 17.634 | 0 |
| 2 | 4097 | 24097 | 93.20 | 17.634 | 0 |

These are sustained test-length averages, not the initial 512-token rate. They are a cache/page-boundary and graph-reuse check; no matched previous-version 4K block was run in this pass, so their difference from historical long runs is not presented as an isolated speedup. Results and saved continuations: [`production_long_q4_d4/results.json`](../_temp_codex/qwen38_perf_round3/production_long_q4_d4/results.json). Its decode profile was collected separately after timing. These long runs preceded the final activation compile-key cleanup; the later six Q4 calls and real Deepy reruns validate that cleanup.

The actual `wgp.py --ask-deepy` path also completed both JSON write/read/update tasks at two and four drafts using the installed binary, live source, isolated config, and Q4/Q8. Both final on-disk files contain the requested model/draft count and `validated: true`. Tool results and repeated user turns exercise runtime state save/restore and the grammar-constrained tool sampler:

| Actual Deepy phase above 20K | Two drafts, tok/s | Four drafts, tok/s |
| --- | ---: | ---: |
| Thinking | 113.5 | 117.3 |
| Tool arguments | 131.3 | 178.0 |
| Statements | 119.8 | 136.2 |
| All decode phases | 116.6 | 128.7 |

Occupied contexts were 20,299–23,265 at two drafts and 20,287–23,667 at four; all reported zero MTP synchronization delta. Rates divide summed emitted tokens by summed action decode time, excluding prefill/loading/tool execution. The two runs generated different tool sequences, and their Deepy defaults are copied from the user's config (including top-p 0.9), so these are functional phase measurements, not matched-token speed comparisons against the prior pass or the controlled top-p 0.95 benchmark. Short tool spans must not be presented as sustained thinking speed.

Evidence: [`deepy_tools_summary.json`](../_temp_codex/qwen38_perf_round3/deepy_tools_summary.json), [`deepy_tools_20k_final.log`](../_temp_codex/qwen38_perf_round3/deepy_tools_20k_final.log), [`deepy_tools_20k_d2_final.log`](../_temp_codex/qwen38_perf_round3/deepy_tools_20k_d2_final.log), and [`run_deepy_tools.py`](../_temp_codex/qwen38_perf_round3/run_deepy_tools.py).

These checks preserve the Q4 checkpoint and intended sampler; they do not establish equal language quality between Q4 and the separately quantized Q3/Q2 checkpoints, or prove equivalence for every possible workload. No precision reduction was introduced as a speed shortcut.

### Build, source preservation, and reproduction

The new native `_C` binary was built using py311, CUDA 13.1, and `TORCH_CUDA_ARCH_LIST=12.0`; build output confirms only `compute_120/sm_120`. MSVC/SDK environment setup did not invoke `vcvars64.bat`. The existing `_attention` binary was preserved. The package's installed version metadata remains **1.0.14+mmvq**; the new binary is identified by its hash. No new wheel was published.

| Binary in py311 | SHA-256 |
| --- | --- |
| `_C.cp311-win_amd64.pyd` | `b53b4538298bde0a00dfe161b77f989fb4c5ea9ddb86ffad8237a091835fb37f` |
| `_attention.cp311-win_amd64.pyd` | `994d28d415d9585d3fbe2f392a22e7d55bb5f694cbcbd90b7b0371d819704156` |

The old `_C` binary and both modified native sources remain under [`native_baseline/`](../_temp_codex/qwen38_perf_round3/native_baseline). Build logs: [`build.log`](../_temp_codex/qwen38_perf_round3/build.log), [`build_interchange.log`](../_temp_codex/qwen38_perf_round3/build_interchange.log). Installation verification: [`installed_binary.json`](../_temp_codex/qwen38_perf_round3/installed_binary.json).

The complete source is preserved as **1.0.19**, after checking local and published version use: [notes](../../kernels/llama.cpp-1.0.19/SOURCE_COPY.md), [source ZIP](../../kernels/llama.cpp-1.0.19-source.zip), and [ZIP SHA-256](../../kernels/llama.cpp-1.0.19-source.zip.sha256). The full native project/vendor sources accompany a WanGP overlay containing all eleven production files, six test files, the benchmark, and this report. A manifest records file hashes and original source hashes. Only snapshot version metadata and its README annotation differ from the live native source. Versions 1.0.16, 1.0.17, and 1.0.18 remain frozen.

Run the existing `tools/benchmark_qwen38_engine.py` commands from the second-pass reproduction section to test the installed code, without `--kernel-library`, `--reference-attention`, or source overrides. Use fresh output directories. This pass's baseline/candidate commands, order, source hashes, full results, logs, and GPU monitoring are recorded in [`final_suite.json`](../_temp_codex/qwen38_perf_round3/final_suite.json), [`state_suite.json`](../_temp_codex/qwen38_perf_round3/state_suite.json), [`production_checks.json`](../_temp_codex/qwen38_perf_round3/production_checks.json), and [`activation_final_checks.json`](../_temp_codex/qwen38_perf_round3/activation_final_checks.json). [`run_state_suite.py`](../_temp_codex/qwen38_perf_round3/run_state_suite.py) uses the frozen reference/candidate files; [`run_production_checks.py`](../_temp_codex/qwen38_perf_round3/run_production_checks.py) exercises the live installation. Timing runs leave `CUDA_LAUNCH_BLOCKING` unset; correctness runs that enable it are not used for throughput claims.

Restart WanGP to load both the new native binary and Python implementation. The user's running server was not stopped or replaced. No MMGP, `wgp.py`, other model-family implementation, or shared config value was changed in this round. Test Deepy uses an isolated config and filesystem output folder; its command selects port 7865 and opens no UI server. Test helpers are stopped at completion.

## Follow-up: bounded Triton variants and strict decoder modes

This follow-up addresses the requested limits on shape-triggered compilation and the separation between `legacy`, `cg`, and `vllm`. It changes Python dispatch and compatibility code; the third-pass native binaries and checkpoint precision remain unchanged.

### Fixed compilation ranges

Previously, making a length a runtime scalar was not sufficient: Triton could still specialize for scalar value 1 or alignment divisibility. Exact MTP lengths and KV table widths were also still compile-time arguments. The changed kernels now suppress both value and alignment specialization on those runtime values.

| Kernel | Runtime dimensions shared by compiled code | Fixed variant bound per model geometry, dtype, and device |
| --- | --- | --- |
| Fused SiLU/multiply | Total output elements, including different row counts and alignment | One |
| MTP convolution verification | Batch size `B` and verification length `T` | One per bias configuration |
| MTP recurrent verification | Batch size `B` and verification length `T` | One |
| Q8 prefix prefill | Query/context lengths and block-table row stride | One |
| Grouped Q8 decode/verification partials | Queries per sequence and block-table width | At most 16: tile height 16/32 multiplied by eight split counts, 1/2/4/8/16/32/64/128 |
| Grouped Q8 reduction | Number of output rows | At most eight, one per split count |

Geometry here means feature/head dimensions, head grouping, convolution width, and the cache layout. A different model geometry or dtype can require its own variants. First use of an unwarmed variant still compiles; changing a covered length then reuses it. There is no growing list of exact token-length variants, no padding of live MTP state, and no extra buffer allocation. The pre-existing FLA normalization autotuning uses four NB ranges (1, 2–15, 16–24, 25+); this statement does not claim to bound every kernel inside the upstream FLA library.

`tests/test_triton_shape_ranges.py` records actual returned compiled-kernel hashes. It checks SiLU across three dtypes and 12 row counts (including 1, 16, 17, 512, and 513); state updates across batches 1/2/3 and verification lengths 1–9/15/16/17; and attention across query counts 1–9 and table widths 1/3/16/17/128. After warming representative attention configurations, the remaining covered shapes produce no new compiled hashes. On this Qwen/5090 workload, single-batch MTP lengths 1–9 use just three partial-attention configurations.

The affected production tests passed (23), followed by the compiled-range and exact state/reference tests (44), all with `CUDA_LAUNCH_BLOCKING=1`. The exact state checks include graph replay, every committable prefix, and final in-place state. Evidence: [`affected_tests.log`](../_temp_codex/qwen38_triton_ranges/affected_tests.log), [`range_exact_tests.log`](../_temp_codex/qwen38_triton_ranges/range_exact_tests.log).

Matched real Q4 benchmarks used the same native binary and checkpoint, Q8 cache, capacity 32,768, two repeats of each context, 512 requested output tokens, and the existing temperature 0.6/top-p 0.95/top-k 20/min-p 0.05/repetition 1.05/seed 123 settings. Only the three range-related source modules were switched between the frozen reference and candidate. CUDA launch blocking was unset for timing.

| Drafts | Prompt tokens | Before decode, tok/s | After decode, tok/s | Before prefill, tok/s | After prefill, tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2,048 | 122.10 | 124.34 | 4,054 | 4,058 |
| 2 | 20,000 | 112.29 | 112.75 | 2,882 | 2,867 |
| 4 | 2,048 | 110.81 | 110.27 | 4,107 | 4,050 |
| 4 | 20,000 | 116.55 | 116.72 | 2,864 | 2,868 |

These are medians over two repeats, with unlocked clocks. The small changes are consistent with timing variation; this is a compilation-latency fix, not another claimed throughput increase. All eight matched completions are byte-identical, including identical emitted counts and MTP acceptance statistics. Every synchronization delta is zero. Peak allocated VRAM is unchanged within allocator rounding: about 17.485 GiB at two drafts and 17.634 GiB at four. Evidence and exact commands: [`comparison.json`](../_temp_codex/qwen38_triton_ranges/comparison.json), [`suite.json`](../_temp_codex/qwen38_triton_ranges/suite.json), [`run_suite.py`](../_temp_codex/qwen38_triton_ranges/run_suite.py), and the frozen [`baseline_manifest.json`](../_temp_codex/qwen38_triton_ranges/baseline_manifest.json).

### Decoder mode contract and dispatch audit

| Path | `legacy` | `cg` | `vllm` |
| --- | --- | --- | --- |
| CUDA graph capture/replay | Disabled | Enabled, subject to the existing graph config | Enabled, subject to the existing graph config |
| Triton SiLU, RMSNorm, KV storage/attention, sampling | Disabled | Disabled | Allowed |
| FLA convolution, recurrent/chunked attention, gated normalization | PyTorch implementations | PyTorch implementations | FLA/Triton allowed |
| External FlashAttention 2 | Disabled | Disabled | Allowed |
| Available GGUF/llama.cpp CUDA kernels | Allowed | Allowed | Allowed |
| Missing GGUF operation or package | PyTorch | PyTorch, including graph capture | PyTorch weights; Triton Q8 attention can still run independently |

PyTorch SDPA may use the CUDA implementations shipped with PyTorch. It does not call the separately installed `flash_attn` or Triton packages in the two PyTorch modes. The mode continues to come from the existing shared `lm_decoder_engine` config; no duplicate switch or new model setting was added.

The audit found and corrected four dispatch problems:

1. The newly fused SiLU list path ignored the mode. `SiluAndMul` now takes a per-instance `use_triton` flag, set from the existing Qwen safe-kernel config for both target and MTP blocks.
2. Sparse repetition penalty did not receive the runner's existing `use_triton_sampling` flag. It now does, retaining its PyTorch implementation for legacy/cg. Min-p already honored the flag.
3. Grouped Q8 decode/verification and Q8 prefix prefill bypassed the attention instance's Triton permission. They now obey it. Native Q8 attention remains available in all modes; if it is absent, PyTorch materializes the Q8 cache and calls SDPA. In vllm, grouped Triton Q8 attention no longer incorrectly depends on the native attention package being present.
4. Explicit legacy/cg selection still ran the startup Triton smoke kernel through the shared resolver. The resolver now returns those choices before probing Triton/FA2. The FLA autotuning setup is also skipped for those modes.

The existing per-instance controls for RMSNorm, FlashAttention, ShortConvolution, gated RMSNorm, and recurrent/chunked FLA attention were checked and retained. Legacy uses the existing eager runner; cg retains target, MTP, verification, and sampler graph reuse. This intentionally changes the shared resolver in `vllm_support.py`; unrelated engine/model implementations, MMGP, and `wgp.py` are untouched.

### Complete PyTorch fallbacks for the advertised GGUF operations

Native Q8 attention is selected by a callable entry point and the matching cache-format identifier, rather than assuming every installed package contains that operation. Missing/older package APIs select the compatibility path. GGUF linear/embedding dispatch also checks that the corresponding callable exists; the existing runtime probe and per-operation fallback remain in place. The active native or PyTorch path is logged once.

The fallback audit fixed existing correctness and capture issues:

- **Attention masks:** the shared SDPA helper converted boolean masks into floating 0/1 values. These became additive scores, allowing padded/future slots to contribute. Boolean masks now retain their boolean meaning for decode and speculative verification, including graph replay.
- **Q8 cache storage:** PyTorch previously rounded the scale to FP16 before dividing key/value elements and could divide at reduced precision. It now quantizes with FP32 values/scales, then rounds the stored scale. Tests match the FP32 reference exactly and the Triton scale format exactly. Rare FP32 half-way rounding cases can differ by one integer quantization step from Triton's approximate division; tests explicitly constrain those differences to half-way cases, rather than claiming all cross-backend payloads are bit-identical.
- **Weight dequantization:** unpacking/scaling now uses FP32 intermediates followed by one final cast. Existing K/classic quantization types are checked exactly against the `gguf` NumPy FP32 decoder followed by that same cast. This avoids premature BF16 scale rounding.
- **IQ weights:** added PyTorch decoders for IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, and IQ4_XS. Together with Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1, and Q8_0, this covers all 18 linear qtypes advertised by the installed package, including both local Q3/Q2 checkpoints. Small immutable lookup tables use the existing `gguf` tables; host-to-device copies are capture-compatible and their CPU storage remains alive. No persistent GPU lookup cache was added.
- **Embedding capture:** the PyTorch path no longer calls dynamically sized `torch.unique()`. It gathers requested packed rows directly, then dequantizes those rows, preserving duplicates and input order. It avoids materializing the whole embedding matrix.
- **Reduced MTP head:** `GGUFFirstRowsLinear` now flattens the packed storage before taking its byte prefix. Its old slice could retain the whole two-dimensional packed matrix, which native kernels tolerated but full-matrix fallback could not reshape to the reduced vocabulary. The new slice is still a view.
- **Temporary memory:** large fallback matrices are dequantized in chunks of at most 65,536 quantization blocks (16,777,216 values for K/IQ blocks) into the final dense matrix. This bounds the FP32 unpacking intermediates while preserving the exact dense result. It does not cache a dense copy of the whole model. The final matrix and graph workspaces still require more VRAM than packed native execution.

These are explicitly requested compatibility fallbacks. They preserve the stored checkpoint's quantization and do not requantize it, but dense PyTorch fallback is slower and needs more temporary memory than the installed packed kernels. Full-model sampled output is not promised to be byte-identical across different arithmetic backends.

The combined regression suite passed **142 tests**, including actual compiled-range hashes. After bounding large weight conversion, the affected fallback/isolation suite passed **89 tests** again. Tests cover all 18 qtypes on CPU and CUDA, FP32/FP16/BF16 reference casts, duplicate embedding indices, repeated graph replay with changed inputs, chunk boundaries, old packages with missing entry points, the partial MTP head, and Q8 decode/verification/prefix attention with native and Triton calls forbidden. All correctness runs use `CUDA_LAUNCH_BLOCKING=1`. Evidence: [`production_checks.log`](../_temp_codex/qwen38_backend_modes/production_checks.log), [`bounded_fallback_checks.log`](../_temp_codex/qwen38_backend_modes/bounded_fallback_checks.log), [`test_gguf_torch_fallback.py`](../tests/test_gguf_torch_fallback.py), and [`test_llm_backend_isolation.py`](../tests/test_llm_backend_isolation.py).

### Real-checkpoint compatibility validation

**Thirteen process-isolated runs / 26 prompt calls passed** on the downloaded checkpoints. Each process runs two different short story prompts, appends a cached prefix, and performs sampled MTP decoding. The main matrix requests 48 tokens per call; two additional vllm/Q3/Q2 missing-package checks request 16. Speculative batches can emit one or two additional tokens at the requested stopping boundary. These are functional checks with `CUDA_LAUNCH_BLOCKING=1`, not throughput benchmarks.

| Checkpoint and native package | Legacy | cg | vllm |
| --- | --- | --- | --- |
| Q4, installed package | Passed | Passed | Passed |
| Q4, package import blocked | Passed | Passed | Passed |
| Q3, package import blocked | Passed | Passed | Passed |
| Q2, package import blocked | Passed | Passed | Passed |
| Q4, simulated older package | — | Passed | — |

The older-package simulation retains the valid package/linear probe but removes Q8 attention, embedding, embedding capability, and runtime-buffer preparation APIs; it also reports Q4_K linear unsupported. This exercises partial availability, including the embedding and dense linear fallback, rather than only a missing import.

The harness intercepts Triton launches and external FA2 calls and raises if either is attempted in legacy/cg. It also forbids CUDA graph capture in legacy. Across every legacy/cg run the launch counts remain **Triton=0, FA2=0**; legacy capture/replay counts are also zero. cg performs 298–340 graph replays across its two prompts. Target decode graph identities are unchanged between repeated prompts. vllm uses Triton/FA2 and graph replay, including when GGUF kernels are unavailable. Every MTP synchronization delta is zero. The generated text was inspected and remains coherent for the requested task; the short samples are not a broad language-quality evaluation.

The chunked materialization fix also has a matched real-Q4 memory check: cg without native kernels peaks at **18.550 GiB**, down from **24.946 GiB** with unbounded unpacking intermediates, saving **6.396 GiB**. Both completions and their MTP statistics/alignment are identical before and after chunking. With installed kernels, the same short cg test peaks at 16.109 GiB. These short compatibility tests use a 1,024-token minimum-capacity hint and should not be compared directly with the 32,768-capacity throughput benchmarks above. Q3/Q2 missing-kernel cg tests peak at 14.862/12.831 GiB. Native packed kernels remain the practical fast path; the kernel-free IQ tests are substantially slower.

Evidence: [`summary.json`](../_temp_codex/qwen38_backend_modes/summary.json), [`bounded_suite.json`](../_temp_codex/qwen38_backend_modes/bounded_suite.json), [`extra_checks.json`](../_temp_codex/qwen38_backend_modes/extra_checks.json), and [`run_runtime.py`](../_temp_codex/qwen38_backend_modes/run_runtime.py). The native package was disabled or altered only inside these test processes. The installed package and the user's server were not changed by the simulations.

### Normal vllm performance after mode separation

The final live code was compared with the frozen pre-separation Python modules under the same installed native binary, using the earlier 2K/20K Q4 benchmark parameters and two repeats per context. These runs unset `CUDA_LAUNCH_BLOCKING`. Eight live completions match both the earlier range-test completions and the frozen-reference completions byte for byte; all MTP statistics and synchronization deltas match as well.

| Drafts / comparison | Prompt tokens | Reference decode, tok/s | Final decode, tok/s | Reference prefill, tok/s | Final prefill, tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 2,048 | 110.12 | 109.86 | 3,607 | 3,655 |
| 2 | 20,000 | 99.57 | 99.27 | 2,559 | 2,582 |
| 4, initial comparison | 2,048 | 106.11 | 98.11 | 3,978 | 3,645 |
| 4, initial comparison | 20,000 | 105.83 | 102.82 | 2,623 | 2,577 |
| 4, reverse-order repeat | 2,048 | 99.09 | 98.12 | 3,668 | 3,668 |
| 4, reverse-order repeat | 20,000 | 103.85 | 103.57 | 2,595 | 2,568 |

The initial four-draft comparison varied enough to justify one reverse-order repeat, which is retained alongside the first result rather than replacing it. All four repeated completion/statistic comparisons are also exact. The final paired differences are about −0.3% at 20K for both draft depths, and −0.2%/−1.0% at 2K. Prefill is similarly close in the repeat. This supports preserving normal vllm performance within the observed timing variation, not a new speedup claim. The much longer fallback validation session preceded these runs; GPU clocks were not locked, and a spot reading during final benchmarking was 80°C. Absolute rates from this block should not be compared with the earlier, cooler range-check block as though only code had changed.

Peak allocated VRAM remains 17.485 GiB at two drafts and 17.634 GiB at four, with the 32,768-token capacity. The per-mode permissions and fallback implementations do not add persistent buffers to the installed-kernel vllm path.

Evidence and full commands: [`performance_suite.json`](../_temp_codex/qwen38_backend_modes/performance_suite.json), [`reference_suite.json`](../_temp_codex/qwen38_backend_modes/reference_suite.json), [`counterbalanced_suite.json`](../_temp_codex/qwen38_backend_modes/counterbalanced_suite.json), [`summary.json`](../_temp_codex/qwen38_backend_modes/summary.json), and [`run_reference.py`](../_temp_codex/qwen38_backend_modes/run_reference.py). The final IQ4 lookup constant is explicitly CPU-resident so an ambient default device cannot create a persistent GPU table; its eight affected reference/capture tests were rerun in [`iq4_final_checks.log`](../_temp_codex/qwen38_backend_modes/iq4_final_checks.log).

### Source preservation and deployment

The complete updated source is preserved as **1.0.20**: [source notes](../../kernels/llama.cpp-1.0.20/SOURCE_COPY.md), [source ZIP](../../kernels/llama.cpp-1.0.20-source.zip), and [SHA-256 sidecar](../../kernels/llama.cpp-1.0.20-source.zip.sha256). The version is checked against local snapshot/build names, installed metadata, git tags/history, and published GGUF release assets before creation. The snapshot contains the full native project plus thirteen WanGP production files, nine test files, the benchmark, and this report. `patches/bounded_kernels_and_modes.patch` isolates this follow-up. File hashes, live-overlay equality, archive contents/CRC, Python syntax, and the earlier snapshot ZIP hashes are verified in [`delivery_verified.json`](../_temp_codex/qwen38_backend_modes/delivery_verified.json).

Snapshots 1.0.16–1.0.19 remain unchanged. The installed native package is still `1.0.14+mmvq`, with the same third-pass `_C` and `_attention` hashes listed above. This follow-up built no new native binary or wheel; Triton compilation/testing ran only on the RTX 5090. No package was uninstalled to simulate missing kernels.

Restart WanGP to pick up these Python changes. The user's running server was left in place, and all test processes have finished. No shared config value, MMGP code, `wgp.py`, or other model-family implementation was changed. The pre-existing unrelated film-grain and Triton-logging edits were preserved.

### Triton compilation notifications: follow-up after snapshot 1.0.20

All LLM Triton kernel modules now register the existing shared compilation logger when imported: `activation.py`, `attention.py`, `sampler.py`, `speculative_state.py`, and the existing `layernorm.py`. The normal vllm capability probe already registered it, but direct module use could bypass that probe. Registration at module import closes that gap, including direct headless calls. Setup is placed after the existing definitions: installed Triton includes source line numbers in its cache keys, so inserting imports above kernels would unnecessarily invalidate existing compiled variants. Every original function and its line numbers are preserved, verified against snapshot 1.0.20 (and the unchanged git version of RMSNorm). `AGENTS.md` now records these requirements for future Triton additions.

The shared logger and its console format are unchanged. A real compilation prints a start message immediately, then a completion message with Triton's measured compilation duration. For example, the fresh RTX 5090 test produced:

```text
[WanGP][Triton] Preparing _silu_mul_kernel (compiling, please wait)...
[WanGP][Triton] Compiled _silu_mul_kernel in 272 ms.
```

Both messages are suppressed when the compiled variant is already available in memory or on disk, including after a process restart. A newly required variant or an invalidated cache correctly produces a new pair. Repeated registration is idempotent and preserves any pre-existing compilation listener. Setup happens at import time; this change adds no logging, timers, cache checks, GPU work, or buffers to the per-token path. Kernel source bodies, fixed shape ranges, precision, and backend permissions are unchanged.

Validation used py311 on the RTX 5090 (SM 120), with a separate empty cache directory for each module. **All eleven LLM JIT entry points passed in ten isolated processes:** SiLU; floating-point and Q8 cache stores; Q8 prefill, grouped partials and reduction; sparse repetition and min-p; convolution and recurrent verification; and RMSNorm. Each compiled variant produced exactly one start/completion pair on its cold launch. Fresh-process disk hits, repeated in-memory launches, graph capture, and two graph replays were silent. The existing cache was left in place. The logger, backend-isolation, and fixed-range test suite also passed **24 tests**.

Evidence: [`verify_runtime.py`](../_temp_codex/qwen38_triton_logging/verify_runtime.py), [isolated-process results and individual logs](../_temp_codex/qwen38_triton_logging/run_1788613948629442100/results.json), [`final_tests.log`](../_temp_codex/qwen38_triton_logging/final_tests.log), and [`source_results.json`](../_temp_codex/qwen38_triton_logging/source_results.json). These registration changes postdate the frozen 1.0.20 source snapshot.

The real Q4 checkpoint also passed two repeated prompt calls in each of legacy, cg, and vllm with MTP enabled and `CUDA_LAUNCH_BLOCKING=1`. All six generated texts match the previous mode-validation outputs exactly. Legacy and cg each recorded zero Triton launches, zero external FA2 calls, and zero Triton compilation messages; legacy captured no graphs, while cg completed 329 graph replays. vllm exercised 5,798 Triton launches, 42 external FA2 calls, and 304 graph replays, with zero compilation messages when reusing the existing disk cache in its fresh process. MTP synchronization remained exact and target graphs were reused across prompts. Evidence: [`verify_modes.py`](../_temp_codex/qwen38_triton_logging/verify_modes.py) and [`mode_results_final.json`](../_temp_codex/qwen38_triton_logging/mode_results_final.json). These are functional checks, not throughput measurements.

## SM120 follow-up: TMA investigation and retained asynchronous copies

This follow-up targets the local RTX 5090 after the shared optimizations above. The final production kernels use **ordinary `cp.async` transfers, not TMA**. TMA was implemented and measured first, but its launch caused a reproducible driver-memory increase of approximately **3.27 GiB** on this machine. Replacing the transfer mechanism preserves the useful attention-kernel improvement without that allocation. The Q4 checkpoint, Q8 cache format, sampling, MTP acceptance rules, and model precision remain unchanged.

Environment: Windows/WDDM, NVIDIA driver **616.56**, RTX 5090/SM120, py311, PyTorch 2.10.0+cu130, Triton 3.6.0. Testing and JIT compilation used this GPU only. The installed native GGUF binaries were not rebuilt. SM120 has different execution facilities from datacenter Blackwell; the implementation uses warp-level MMA and does not assume SM100 tensor memory or multicast. NVIDIA documents the separate SM120 path in its [CUTLASS Blackwell reference](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms).

### VRAM regression: what was measured and corrected

The initial suspicion of another process was insufficient. A fresh-process comparison reproduced the increase with the same Q4 model, four MTP drafts, 32,768-token capacity, and a 20,000-token prompt. The table includes warmup and inference, but excludes the separately executed profiler:

| Implementation / session | Peak PyTorch allocation, GiB | Peak PyTorch reservation, GiB | Peak whole-GPU NVML usage, GiB |
| --- | ---: | ---: | ---: |
| Shared kernels, original memory audit | 17.634 | 18.016 | 21.316 |
| TMA candidate, original memory audit | 17.634 | 18.016 | 24.690 |
| Shared kernels, final memory audit | 17.634 | 18.016 | 21.864 |
| Retained `cp.async`, final memory audit | 17.634 | 18.016 | 21.996 |

Thus the extra memory was invisible to PyTorch's allocator statistics. The final shared/`cp.async` runs also have the same peak used-memory reading from `cudaMemGetInfo`: **19.754 GiB**, with the context stack remaining **1,024 bytes per thread** throughout. Whole-GPU usage includes the desktop and other applications and changed by about 0.13 GiB between these final runs. CUDA and NVML readings are recorded separately because their accounting differs under this WDDM setup. Values in this section are GiB, not decimal GB. Earlier 21.5/25.5 GB observations should not be compared to PyTorch allocation alone.

The diagnosis was narrowed to the first actual TMA launch:

1. Loading the compiled CUDA module and encoding its host tensor descriptors did not cause the large increase. Launching the TMA kernel changed `CU_LIMIT_STACK_SIZE` from **1,024 to 14,448 bytes**, with approximately **3.268 GiB** extra used CUDA memory in the isolated process.
2. This was not explained by larger declared kernel-local storage: the inspected TMA prefill cubin used a 24-byte stack frame, versus 48 bytes of reported local storage in the shared prefill kernel. Grouped TMA reported 32 bytes. The much larger context-wide stack setting arose at launch.
3. A minimal 32-by-256 TMA copy reproduced the problem without attention or a model: an 8-byte declared frame still raised the context stack to 14,432 bytes. Plain copy and barrier-only controls retained 1,024 bytes. INT8/FP16/FP32 and different shared-memory swizzles reproduced it.
4. Building descriptors on the GPU instead of the host did not remove the increase. Resetting the context stack to 1,024 bytes released the memory temporarily, but the very next TMA launch enlarged it again. Those experiments are not in production; the application does not repeatedly change global CUDA limits or install a custom allocator.
5. Ordinary asynchronous copies retained the same numerical results and left the stack at 1,024 bytes, both in isolated tests and in the full-model memory audit. The profiler itself added only about 0.08 GiB to the final CUDA reading and was not the multi-GiB cause.

NVIDIA documents that the driver can grow the per-thread stack at launch and does not automatically restore its previous size in [the CUDA context-limit API](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__CTX.html). The unusually large TMA-triggered growth above is a **local experimental finding**. Its driver-internal cause is not established, and it has not been tested on Linux or another driver release.

Evidence: [final source/memory audit](../_temp_codex/qwen38_sm120_tma/final_source_memory.json), [original memory summary](../_temp_codex/qwen38_sm120_tma/memory_summary.json), [final memory summary](../_temp_codex/qwen38_sm120_tma/memory_final_summary.json), [audit harness](../_temp_codex/qwen38_sm120_tma/audit_memory.py), [launch/stack isolation](../_temp_codex/qwen38_sm120_tma/isolate_tma_driver.log), [minimal TMA reproduction](../_temp_codex/qwen38_sm120_tma/minimal_tma_memory.py), and [stack-reset experiment](../_temp_codex/qwen38_sm120_tma/isolate_stack_restore.log). Samples are taken every 50 ms; exact PyTorch peak counters supplement sampled readings. The sampler/profiler runs are memory checks, not valid decode-speed measurements.

### Retained implementation, buffers, and dispatch

The only production additions for this follow-up are [`attention_sm120.py`](../shared/llm_engines/nanovllm/layers/attention_sm120.py) and its two dispatch points in [`attention.py`](../shared/llm_engines/nanovllm/layers/attention.py). They replace Q8 attention's data staging on eligible SM120/D256 calls:

- **Prefill:** a fixed 16-query by 32-cache-token tile, four warps, and two shared-memory K/V stages. INT8 cache tiles are copied directly from their existing paged storage into swizzled shared memory. The next tile is transferred while the current tile's tensor-core products run. Waits and block barriers make each stage ready before consumption and prevent reuse before readers finish. Entirely future causal tiles are still skipped.
- **Decode/MTP verification:** fixed 16-row by 32-cache-token tiles with one K/V stage. Query heads and speculative queries are grouped as before; short split-context loops favor the smaller staging footprint. The existing split-count selection, output indexing, FP32 partial buffers, maxima/sums, and reduction kernel are reused. No additional per-draft buffers or padded MTP state are introduced.
- **Exact existing arithmetic:** Q8 values are scaled in FP32, then represented by the same high/low FP16 or BF16 pairs. Two QK and four probability/value products retain FP32 accumulation and the existing softmax and causal masks. Explicit MMA-v2 layouts use four warps arranged as `[1, 4]` and operand packing width 4, matching the shared kernel's sum order. An earlier width-2 prototype produced one-ULP BF16 differences and was rejected.
- **Storage:** prefill's two INT8 K/V stages total 32 KiB of on-chip tile storage; grouped attention's single stages total 16 KiB. These are shared-memory tiles per thread block, not persistent VRAM allocations. The compiler can also use shared memory for layout conversion. There is no dense KV materialization, model-weight copy, descriptor tensor, global GPU allocator, or new precision reduction in this path.
- **Eligibility and compatibility:** automatic selection requires compute capability `(12, 0)`, head dimension 256, and the existing vllm Triton permission. Other architectures and dimensions retain the original shared kernels. An explicitly supported fallback was added for Triton installations without the optional Gluon APIs: import failure leaves the shared implementation selected. This preserves older installations without hiding errors from an already selected running kernel. Existing PyTorch/native fallbacks remain available under the earlier mode contract.
- **Integration:** the existing once-only backend message identifies SM120 asynchronous copies. `legacy` and `cg` retain their PyTorch/native implementations and do not launch these kernels. No UI/config key, MMGP code, `wgp.py`, checkpoint, sampling rule, or other model implementation was changed in this follow-up.

`cp.async` itself also exists on older NVIDIA architectures; it is not an RTX50-only feature. The tile/layout tuning and dispatch here are confined to SM120 because that is the architecture tested. The other GPUs retain all previous shared optimizations.

### Fixed compilation ranges and console behavior

For a fixed model geometry, dtype, and scale, prefill uses **one compiled variant** across query/context lengths. Grouped partials use **at most eight**, corresponding to split counts 1/2/4/8/16/32/64/128. Query count, occupied length, page-table width/stride, and their alignment are runtime values rather than exact-length specializations. The shared reduction retains its existing eight split-count variants. No autotuning search is added.

The new module registers the existing compilation logger once at import. Both entry points were tested with an empty private cache, then in a fresh process using that disk cache, then through repeated launches and CUDA graph replay. Each cold compilation printed exactly one start/completion pair, including its duration (681 ms prefill and 595 ms grouped in this run); all memory/disk hits and graph replays were silent. The shared logger itself is unchanged. All five original attention JIT functions retain their exact syntax trees and source line positions, so their existing disk-cache identities are preserved.

Evidence: [`test_q8_sm120_attention.py`](../tests/test_q8_sm120_attention.py), [range test](../_temp_codex/qwen38_sm120_tma/async_range_test.log), [runtime/logging validation](../_temp_codex/qwen38_sm120_tma/final_runtime.log), and [source audit](../_temp_codex/qwen38_sm120_tma/final_source_memory.json). The range test checks returned compiled hashes across query counts 1 through 1,024, multiple batches, short/20K contexts, and table capacities through 128 pages. It also verifies ordinary asynchronous-copy PTX is emitted and TMA tensor-copy PTX is absent.

### Isolated attention performance of the retained code

These are CUDA-event timings of warmed graph replays, with eight calls per graph, four timing samples, and both execution orders. Inputs use Qwen's 24 query heads, four KV heads, D256, BF16 I/O, Q8 cache, shuffled pages, and exact comparisons with the shared implementation. Throughput gain is `shared_time / revised_time - 1`:

| Query rows | Occupied context | Shared attention, ms | `cp.async` attention, ms | Throughput change |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 2,048 | 0.01794 | 0.01801 | -0.4% |
| 3 | 20,000 | 0.08453 | 0.07626 | +10.8% |
| 5 | 2,048 | 0.01988 | 0.02173 | -8.5% |
| 5 | 20,000 | 0.08757 | 0.07988 | +9.6% |
| 1,024 | 2,048 | 1.44104 | 1.16237 | +24.0% |
| 1,024 | 20,000 | 14.9727 | 12.8254 | +16.7% |

The short five-query regression is about 1.9 microseconds per attention call; it is retained in the results rather than omitted. These attention-only percentages are not model decode-speed improvements. All checked outputs are bit-identical to the shared kernels. Evidence: [timing script](../_temp_codex/qwen38_sm120_tma/check_async_copy.py) and [complete isolated results](../_temp_codex/qwen38_sm120_tma/check_async_copy.json). The measured prototype and retained production kernel bodies match.

### TMA results retained for traceability

The discarded TMA implementation is preserved in [`attention_tma_before_memory_fix.py`](../_temp_codex/qwen38_sm120_tma/attention_tma_before_memory_fix.py). An initial high-level Triton descriptor prototype regressed a long-prefill attention case (about 22.8 versus 17.6 ms). Explicit Gluon layouts, a 16-by-32 tile, two prefill stages, and matching operand packing produced roughly 15–24% attention-only prefill gains instead. The selected grouped TMA candidate used one stage. Larger tile/stage variants and D128 configurations were not promoted when they regressed or failed exact matching.

TMA's first paired full 20K prefills were essentially unchanged (7.102 versus 7.089 s); its 1,024-token suffix over an existing 20K prefix improved from about 0.542 to 0.499 s (+8.5% throughput). A later same-process four-draft decode comparison found a median matched gain of only **1.32%** at 20K, and a **3.95% regression** at 2K. An earlier separate-process result suggested about 13% faster decoding, but that was not reproduced by the stronger alternating-order comparison and is not used as a reliable gain. Profiling still showed quantized matrix-vector operations dominating decode; attention occupied roughly 7% of GPU time. TMA's isolated attention improvement could not explain a general double-digit model-decode increase.

Those historical figures describe the rejected TMA candidate, not the retained `cp.async` implementation. Evidence: [TMA full-prefill comparison](../_temp_codex/qwen38_sm120_tma/model_d2/results.json), [TMA suffix comparison](../_temp_codex/qwen38_sm120_tma/model_suffix_d4/results.json), [same-process TMA decode comparison](../_temp_codex/qwen38_sm120_tma/production_pairs_d4/results.json), and [TMA decode profile](../_temp_codex/qwen38_sm120_tma/production_pairs_d4/tma_decode_profile.txt).

### Final Q4 model performance and validation

The retained production code was compared with the shared kernels in the same model process, using separately warmed graph sets and alternating the execution order for each successive prompt. Target graph identities were checked after every prompt. Settings remain Q4_K_M, Q8 KV, 32,768-token capacity, temperature 0.6, top-p 0.95, top-k 20, min-p 0.05, repetition penalty 1.05, and seed 123. Each full-prefill row contains three different prompt pairs with 512 requested output tokens. The suffix row appends 1,024 tokens to each 20K prompt and requests 256 output tokens. Actual emitted counts are used for throughput. CUDA launch blocking and profiling are disabled for these timings.

Times and rates below are medians. The gain columns are the median of the three matched throughput ratios, which need not equal the ratio of the two displayed medians. All individual ratios are retained in the linked JSON.

| Drafts / workload | Shared prefill, s | Revised prefill, s | Matched prefill gain | Shared decode, tok/s | Revised decode, tok/s | Matched decode gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 / 2,048 prompt | 0.5544 | 0.5761 | -2.08% | 107.74 | 106.71 | -1.05% |
| 2 / 20,000 prompt | 7.8314 | 7.4844 | +4.36% | 101.08 | 98.05 | -1.01% |
| 4 / 2,048 prompt | 0.6892 | 0.5599 | +22.59% | 96.11 | 99.78 | +3.82% |
| 4 / 20,000 prompt | 7.8376 | 7.4809 | +4.57% | 98.77 | 98.54 | +0.14% |
| 4 / 1,024 suffix on 20K | 0.5389 | 0.5021 | +7.33% | 85.39 | 85.49 | +1.59% |

The repeatable benefit is long-context prefill: about **4.4–4.6% higher full-20K throughput**. The short-prompt results vary substantially between sessions: four-draft prefill has two unusually slow shared samples, while two-draft prefill is about 2% slower with the candidate. They do not establish a general short-prefill gain. Decode results are small and mixed, including roughly 1% lower median paired throughput with two drafts; **no reliable model-decode speedup is claimed for this follow-up**. Clocks were not locked, and a spot GPU reading during this suite was 75 degrees C / 2,812 MHz. Absolute rates from separate optimization rounds are not directly comparable.

The final profiled four-draft decode still spends about 7.2% of GPU time in grouped attention; its two largest native quantized matrix-vector kernels account for about 62%. Faster cache staging therefore addresses only a small part of decode. The [final decode profile](../_temp_codex/qwen38_sm120_tma/memory_final_async_d4/decode_profile.txt) is separate from the timing suite and was collected during the memory audit.

All **15 matched pairs / 30 completions** have identical output text, token counts, and MTP acceptance statistics; every synchronization delta is zero. These are regression checks on the tested continuations, not a broad language-quality benchmark. Peak allocated memory remains about 17.485 GiB for two drafts and 17.634 GiB for four. The alternating benchmark retains two graph sets for comparison, so the separate fresh-process memory audit above is the production VRAM reference.

Real-checkpoint mode validation also passed **two repeated prompt calls in each of legacy, cg, and vllm**, with CUDA_LAUNCH_BLOCKING=1. All six texts match the previous mode-validation reference. Legacy recorded zero Triton calls, zero external FA2 calls, and zero graph captures/replays. cg recorded zero Triton/FA2 calls and 329 graph replays. vllm exercised both new SM120 attention entry points, 5,798 Triton calls, 42 external FA2 calls, and 304 graph replays. Its warmed disk cache produced no compilation messages. Target graphs were reused and MTP synchronization remained exact.

All **24 SM120 GPU regression checks pass** with CUDA_LAUNCH_BLOCKING=1 in [the final test run](../_temp_codex/qwen38_sm120_tma/final_sm120_tests.log). They cover FP16/BF16 exact matching, shuffled/ragged pages, 20K contexts, one through nine verification inputs, empty splits, graph replay after changing data/lengths/page tables, bounded compiled hashes, dispatch exclusion for other architectures/dimensions/missing Gluon, and unchanged driver-stack reservation. During test development, a PTX assertion initially expected the `ca` cache modifier; the compiler correctly emitted `cg`. The assertion now accepts either ordinary-copy modifier while still rejecting TMA tensor copies. This was a test correction; no production arithmetic was changed.

Evidence: [paired summary](../_temp_codex/qwen38_sm120_tma/final_performance_summary.json), [exact commands and kernel hashes](../_temp_codex/qwen38_sm120_tma/final_performance_suite.json), [comparison harness](../_temp_codex/qwen38_sm120_tma/run_model_comparison.py), [four-draft records](../_temp_codex/qwen38_sm120_tma/async_pairs_d4/results.json), [two-draft records](../_temp_codex/qwen38_sm120_tma/async_pairs_d2/results.json), [suffix records](../_temp_codex/qwen38_sm120_tma/async_suffix_d4/results.json), and [mode/logging results](../_temp_codex/qwen38_sm120_tma/final_validation_1788623850816900200/results.json). This SM120 follow-up adds no new Q3/Q2 full-model speed claim; their earlier profiles remain above.

These Python changes postdate frozen snapshot 1.0.20. The installed native package remains 1.0.14+mmvq with the same binary hash. No checkpoint was downloaded, no new native binary or wheel was built, and no test UI server was started. Restart the WanGP process to load the revised kernels and discard any driver stack reservation left by an earlier TMA launch; unloading the model alone did not release that reservation in the measured process.

### Power consumption and concurrent-process check

The high power draw was reproduced by the benchmark itself. With model work stopped, no Python/model workload remained. Windows GPU-engine counters identified desktop composition (`dwm.exe`), VS Code, and the Codex/ChatGPT app as the visible background activity, typically about 6–11% GPU utilization in the spot checks. A sample during inference attributed 81% to the benchmark process and 6% to desktop composition. GPU-context listings alone were not treated as evidence that every listed app was actively computing.

Two 12-second idle samples, before the power suite and immediately after model unload, averaged **73.7 W** and **111.7 W**, with sampled peaks of 112.1/130.9 W. Background rendering, clocks, and cooling state vary, so these are observations of this desktop rather than a minimum idle-power specification. Neither sample reported a power-cap or thermal-slowdown event. The default, requested, and enforced limits were all **600 W**, unchanged before and after testing.

The comparison uses the same Q4/Q8 runtime, 20,000-token prompts, two/four MTP drafts, and two alternating shared/revised prompt pairs per depth. Each continuation requests 1,024 output tokens to give a longer decode measurement. Both graph sets are warmed first. Only one model process runs at a time; sampling is in its separate NVML thread. No GPU power limit, voltage, clock, fan setting, driver, or production sampling setting was changed.

Mean watts below are total NVML energy-counter increase divided by synchronized phase duration, aggregated over both repeats. Joules/token divides the same energy by actual processed/emitted tokens. Peak watts use the instantaneous-power field sampled about every 100 ms; shorter transients may be missed. NVIDIA distinguishes instantaneous power from its one-second averaged power query and provides the accumulated energy counter in its [NVML device API](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html). These are GPU readings, not whole-PC wall-socket measurements. Background GPU activity remains included; no uncertain idle-power subtraction is applied.

| Drafts / phase | Shared mean W | Revised mean W | Shared / revised sampled peak W | Shared J/token | Revised J/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 / prefill | 507.2 | 521.4 | 620.8 / 614.5 | 0.2075 | 0.2026 |
| 2 / decode | 502.4 | 509.1 | 595.0 / 604.5 | 5.3799 | 5.4143 |
| 4 / prefill | 509.6 | 525.1 | 611.6 / 622.5 | 0.2002 | 0.2012 |
| 4 / decode | 482.7 | 480.2 | 610.6 / 607.3 | 5.0316 | 5.1002 |

Prefill J/token refers to input tokens; decode J/token refers to emitted output tokens. They represent different operations and should not be compared as interchangeable efficiency measures.

The revised prefill path draws about **3% more average power** in this comparison, while finishing sooner. Total prefill energy changes by about **-2.4% with two drafts and +0.5% with four**, so this is not a substantial energy-saving optimization. Decode average power changes by about +1.4%/-0.5% at two/four drafts; energy per output token changes by approximately **+0.6%/+1.4%**. With two repeats and an active Windows desktop, these small differences do not establish a meaningful energy-efficiency improvement or regression.

Both shared and revised kernels regularly trigger the existing software power-cap condition. It appeared in about **91–100% of prefill samples** and **83–100% of decode samples**, depending on draft depth and implementation. Instantaneous sampled peaks were around 590–622 W while the configured limit remained 600 W; average draw was lower because the phases contain different kinds of GPU work. Maximum observed GPU temperature was **85 degrees C**. No NVML software/hardware thermal-slowdown flag was observed in these samples. This identifies power limiting during inference, without evidence of a second model job causing the load.

All **four matched power-test pairs / eight completions** retain identical text, token counts, MTP acceptance statistics, and zero synchronization delta. NVML sampling is confined to the benchmark harness, and its measurements are kept separate from the earlier uninstrumented performance results. All model, sampler, and benchmark processes exited after the suite. The post-unload counter snapshot showed desktop composition and the read-only power sampler, which subsequently exited too.

Evidence: [power summary](../_temp_codex/qwen38_sm120_tma/power_summary.json), [commands and source hash](../_temp_codex/qwen38_sm120_tma/power_suite.json), [power meter](../_temp_codex/qwen38_sm120_tma/power_meter.py), [two-draft records](../_temp_codex/qwen38_sm120_tma/power_pairs_d2/results.json), [four-draft records](../_temp_codex/qwen38_sm120_tma/power_pairs_d4/results.json), [two-draft raw samples](../_temp_codex/qwen38_sm120_tma/power_pairs_d2/power_samples.json), [four-draft raw samples](../_temp_codex/qwen38_sm120_tma/power_pairs_d4/power_samples.json), [idle before](../_temp_codex/qwen38_sm120_tma/power_idle_before.summary.json), [idle after](../_temp_codex/qwen38_sm120_tma/power_idle_after.summary.json), [background counters before](../_temp_codex/qwen38_sm120_tma/power_background_before.txt), and [post-unload counters/processes](../_temp_codex/qwen38_sm120_tma/power_background_during.txt).

## Compaction checkpoint retention fix

The live Deepy session on September 5 exposed an avoidable replay while summarizing a long active turn. At 20:26, the runtime had 73,565 cached tokens, but the compaction preparation restored the 7,926-token system/tools checkpoint and reprocessed 64,685 tokens in 64 chunks, taking approximately 48 seconds. The summary then completed and reduced the rendered context from 74,799 to 15,304 tokens. The earlier 20:14 compaction followed the separate completed-history path: it restored its full turn-start snapshot, reused 72,139 tokens, and appended only 480 instruction tokens. This explains why the replay was not present on every compaction. Evidence: [live log](../debug_deepy/debug_deepy_20260905_201036.log).

### Cause and implementation

`AssistantEngine._capture_semantic_boundary` retained the last three capture events. A tool action generates a capture after the assistant's request and another after its results have been prefilled. Compaction, however, preserves two **action groups**, each comprising an assistant tool request and its tool results. Counting capture events therefore evicted the checkpoint before those two groups. A minimal example needed boundary 3 but retained boundaries `[4, 5, 6]`.

The fix uses the existing `_turn_step_ranges` grouping, including the initial user boundary, and retains checkpoints matching the last three group endpoints. A later tool-result checkpoint replaces the earlier request checkpoint for that same group. Repeated captures at one message boundary still replace the previous capture. The example now retains `[3, 5, 6]` before the last result is prefilled, preserving the exact boundary needed by compaction. Multiple tool results, partial-result captures, ordinary assistant statements, and mixed action sequences keep the same bounded behavior.

The only production method changed by this fix is [`_capture_semantic_boundary`](../shared/deepy/engine.py). Checkpoint formats, the existing snapshot/rewind implementations, and the three-checkpoint limit are preserved. Selection runs at action checkpoints, outside the per-token loop. No new fallback or config key was introduced. [Source audit](../_temp_codex/deepy_compaction_boundaries/source_audit.json) verifies that the other engine methods and six protected source files remain unchanged from the start of this fix, including `wgp.py`, the Qwen runtime/model, the model runner, and both attention implementations.

### Validation and memory

The added regression tests fail on the copied previous implementation for single-tool actions, three-tool actions, and mixed statements/tools. With the fix, **20 tests and two subtests pass**, covering compaction retention plus the existing context-resize and tool-recovery tests. Logs: [before](../_temp_codex/deepy_compaction_boundaries/before_tests.log), [after](../_temp_codex/deepy_compaction_boundaries/after_tests.log). Tests: [`test_deepy_compaction.py`](../tests/test_deepy_compaction.py).

The actual Qwen3.8 Q4/Q8 runtime was then exercised in py311 on the RTX5090 with four MTP drafts, 82,944-token cache capacity, and `CUDA_LAUNCH_BLOCKING=1`. The harness builds three tool-action groups, retrieves the retained compaction boundary, invokes the production compaction preparation, and compares a 128-token greedy continuation against an independent full-snapshot restore of that boundary plus the identical instruction. Source replay paths are replaced with assertions inside the test harness so an unexpected replay fails the check.

| Initial context | Tools per action | Cached prefix reused | Instruction tokens prefilled | Preparation time | Continuation versus full snapshot |
| --- | ---: | ---: | ---: | ---: | --- |
| 2,048 | 1 | 2,111 | 544, one append | 1.087 s | 128/128 tokens identical |
| 72,000 | 3 | 72,147 | 544, one append | 1.573 s | 128/128 tokens identical |

The same CUDA graph objects were reused across both prompts and both reference comparisons. The KV allocation pointer stayed unchanged, and every continuation ended with MTP synchronization delta zero. These are synchronized correctness checks of the compaction preparation and continuation window, not a matched speed benchmark against the earlier live session or a complete summary-quality evaluation.

Each lightweight checkpoint held **79,936,520 bytes (76.233 MiB)** of CPU tensor data: 75.75 MiB of recurrent/convolution state plus small MTP pending/draft tensors. Three checkpoints hold **228.700 MiB**, with Python token lists and metadata additional. They continue to reference the live GPU KV prefix rather than copying the KV cache; the retention fix adds no GPU tensor allocation. The full reference snapshot was approximately 2.772 GiB at the short boundary and 3.039 GiB at the long boundary because that separate snapshot type also copies the allocated main KV cache and occupied MTP cache into RAM. These figures describe retained tensor payload, not transient copy peaks or whole-process RAM.

The isolated GPU worker ran after the user's Deepy model unloaded and exited after validation. Restart WanGP to load the corrected checkpoint selection. Existing checkpoints from a running process are not retroactively repaired; this fix also does not eliminate the necessary prefill of rewritten summary context after compaction commits.

Evidence: [runtime harness](../_temp_codex/deepy_compaction_boundaries/verify_runtime.py), [runtime results](../_temp_codex/deepy_compaction_boundaries/runtime_results.json), [runtime log](../_temp_codex/deepy_compaction_boundaries/runtime.log).

## Deepy inspection: keep Qwen weights resident

This round targets **local Deepy inspection, Qwen3.5/3.8, fully resident main weights**. On the tested Qwen3.8 Q4/Q8 setup, inspection avoids every main-weight unload/reload and stays within the assistant's reserved GPU footprint. Prompt enhancement keeps its existing lifecycle. No MMGP source, `wgp.py`, checkpoint, native kernel, or Triton kernel changed in this round.

### Capacity and eligibility

The vision encoder itself has **no autoregressive KV cache**. The temporary cache belongs to Qwen's subsequent visual question answering. The assistant's runtime allocations are released before the tower loads; the tower is unloaded before QA begins. QA uses a **16,384-token minimum capacity**, meaning total capacity rather than an 8K input/8K output split. Local inspection's answer limit increases from 1,024 to **2,048 tokens**; remote inspection remains unchanged. Tests include approximately 2,048 question tokens plus image tokens/labels and answers reaching 2,048 tokens. Longer requests still follow the existing dynamic capacity calculation, without silent truncation to 16K.

Eligibility checks the actual Qwen runtime, MMGP ownership, active-model membership, fully preloaded main blocks, and CUDA residency of all main parameters. Main KV storage, including Q8 scales, must also cover at least the tower's weight storage. This is a necessary capacity check, not an exact bound on every possible workspace.

That restriction comes from a measured limit: **Qwen3.5-9B at 32K with Q8 KV cannot cover its tower's weights**. The unrestricted prototype increased reserved VRAM by 0.43 GiB. Such cases retain the existing unload/reload path; the co-residency guarantee does not cover them. Qwen3.8-27B qualifies at 32K. Qwen3.5-4B and 9B qualify and passed at 64K. Layer-offloaded main models remain on their existing path. The measurements below do not establish a universal peak bound for arbitrary context sizes, precisions or external GPU workloads.

### Changes made

1. **Keep GPU ownership.** The eligible `AssistantEngine._pause_runtime("vision")` path retains the live runtime and GPU ownership. The controller supplies its existing enhancer offload manager through an optional runtime hook; there is no new offload object or application config.
2. **Snapshot and orderly teardown.** The existing full CPU snapshot saves KV values/scales, page allocation/token state, recurrent/convolution state, MTP state, and sampling/presence state. Existing synchronized engine teardown releases graphs and cache/state references before the tower loads. Main weight storage addresses remain unchanged.
3. **Temporary MMGP cotenants and shared embeddings.** The scoped cotenant map adds both directions between `prompt_enhancer_llm_model` and `prompt_enhancer_image_caption_vision_tower_model`, retaining existing rules. The caption adapter temporarily uses the main model's resident input embedding module, avoiding a second MMGP embedding-alias load. Both changes are restored afterward.
4. **Bound encoder work.** Images retain their existing resolution and 1,024-visual-token limit. Independent images/frames are encoded in batches bounded by that single-image budget: 4,096 raw patches for the existing 2-by-2 merger. Small video frames can share a batch. Only projected image features survive between batches.
5. **Remove a large indexing temporary.** Profiling five images found a **619,315,200-byte (590.625 MiB)** `nonzero` allocation from placeholder validation indexing an expanded token-by-hidden-channel mask. The resident, precomputed-feature path assigns complete image-token rows directly into fresh embeddings. It also avoids the expanded-mask scatter workspace and embedding clone. The original pixel-input helper path remains unchanged for prompt enhancement. [Allocation trace](../_temp_codex/deepy_resident_vision/nomtp_profile_vision_allocations.json).
6. **Explicit tower-only unloading.** Final prompt embeddings and positions move to CPU using blocking transfers. Deepy then synchronizes, unloads the tower's MMGP root/split blocks, removes its active-manager entries, and releases unused allocator blocks. The main model remains active. Cleanup also runs on encoder/decoder errors.
7. **Bound multimodal prefill.** A private, scoped hint reuses the assistant's 1,024-token chunk size. Non-MTP prefix chunks update caches without logits or sampling; the final chunk follows normal sampling. MTP chunks carry the preceding target hidden state and its full three-dimensional position across seams, pairing each state with the next token's embedding. Final target logits use only the final hidden row. The selected MTP draft count and sampling rules are unchanged. Ordinary text/prompt-enhancer calls do not set this hint.
8. **Exact assistant restoration.** The temporary QA engine closes before the assistant is rebuilt at its original capacity. The snapshot is restored, and valid semantic boundaries receive the replacement cache signature/pointer because their exact underlying prefix was restored. Old graph pools/caches do not accumulate. Existing graph teardown and snapshot code themselves are unchanged.

Real `legacy` testing also exposed an existing Q8 MTP metadata bug before inspection: an eager block table widened from nine to ten pages while its cached per-query table retained the old width. The attention wrapper now replaces that entry when the width changes. There remains one entry per query count, without accumulating entries for every context width. The corrected real-checkpoint test passed. `legacy` and `cg` retain their PyTorch/native paths; no new Triton/FA2 path is introduced.

### Qwen3.8 Q4 results

RTX5090, py311, Qwen3.8-27B-Uncensored-Q4_K_M, Q8 KV, 32,768 assistant capacity, four MTP drafts unless specified. The harness calls the actual Deepy pause/visual-query methods with the real caption model and MMGP pipeline. Only one GPU test process runs at a time. All memory figures below are **GiB**. PyTorch reserved memory includes allocator pools; whole-GPU NVML includes the desktop, driver and other processes.

| Inspection workload | Previous cycle | Resident cycle | Previous peak reserved | Resident peak reserved |
| --- | ---: | ---: | ---: | ---: |
| One image, short question/answer, first call | 5.29 s | 3.60 s | 18.05 | 18.00 |
| Five images, short question/answer | 6.52 s | 4.91 s | 19.68 | 17.55 |
| One image, short question/answer, repeat | 4.82 s | 3.40 s | 19.68 | 17.55 |
| One image, approximately 2K question, long answer | 19.51 s | 18.14 s | 18.87 | 18.00 |
| Five images, approximately 2K question, 2,048-token answer | 20.49 s | 20.80 s | 20.48 | 17.57 |

Cycle time includes context preservation/restoration, preprocessing/encoding, QA prefill and generation. The long baseline uses the same 2K answer budget. The previous path transfers main weights twice per inspection; reloading alone takes about **1.9 seconds**. Every eligible resident case records **zero main-weight transfers** and unchanged weight pointers.

Short inspections improve by about **25–32% in observed elapsed time**. These are workload timings, not identical-token decode benchmarks: short one-image answers contain 9 baseline versus 12 resident tokens; five-image answers contain 12 in both. Long one-image runs end naturally at 1,840 resident tokens versus the baseline's 2,048-token cap, so that time difference is not a matched throughput gain. Both five-image long answers reach 2,048 tokens and show **no net speed gain**. Graph rebuilding and bounded prefill consume part of the saved reload time. No general decode-speed improvement is claimed.

The final short and long runs include the row-assignment memory fix. In the final long run, reserved memory never exceeds its **17.996 GiB** starting footprint. Whole-GPU NVML peaks at **22.526 GiB**, equal to the initial reading, versus **25.017 GiB** for the previous five-image long inspection. Peak allocated memory is 17.420 GiB against 17.380 GiB initially: transient working tensors still exist inside the reserved footprint. Allocated memory after each of three calls is exactly **18,663,339,008 bytes**, without accumulation.

With **MTP disabled**, two five-image inspections with approximately 2K question tokens and 2,048-token answers pass. Peak reserved memory is **17.270 / 17.127 GiB**, against 17.270 GiB initially; NVML stays within the initial 21.763 GiB. These diagnostics use `CUDA_LAUNCH_BLOCKING=1`; their roughly 43-second times are not normal performance measurements.

### Validation, RAM and tradeoffs

Real repeated calls cover `vllm`, `cg`, `legacy`, MTP on/off, one/five images, 80-frame/eight-frame video inspection, and an injected failure after real vision encoding. Eligible cases verify that the tower returns to CPU, only the main model stays active, and main weight addresses survive. The restored assistant reproduces **64/64 continuation tokens**, and its retained semantic boundary can be restored afterward. Corrected `legacy`, failure and MTP-off diagnostics use launch blocking to expose asynchronous CUDA errors.

Qwen3.5-9B and 4B, each at 64K Q8 capacity without MTP, pass one- and five-image calls with roughly 2K question tokens. Their peak reserved memory stays within the starting **6.781 GiB / 3.744 GiB** footprints respectively. The 9B/32K guard was separately exercised and correctly retained the existing main-weight unload/reload behavior. That older path itself can peak above its steady footprint; this change does not solve the insufficient-cache case.

Visual preprocessing's token IDs, three-dimensional positions and RoPE offset match the previous path. One-image embeddings match bit-for-bit. Different five-image batching produces ordinary BF16 rounding differences: cosine similarity **0.9999961**, relative RMS difference **0.279%** in the recorded comparison. Direct row assignment matches the previous scatter exactly in regression testing. Resolution, feature precision, model quantization, sampling rules and MTP verification remain unchanged. These are targeted correctness checks, not a broad visual-quality benchmark or a claim of identical visual answers.

The final regression suite passes **41 tests and two subtests**, covering eligibility, cleanup after errors, shared rules/embeddings, chunk alignment, direct row assignment, and existing compaction/context-resize/tool-recovery/pause behavior. [Test log](../_temp_codex/deepy_resident_vision/unit_tests_final.log), [real embedding comparison](../_temp_codex/deepy_resident_vision/embedding_comparison.json). All isolated model workers exited; no UI test server was started and the user's WanGP process was left running.

At a 20K occupied assistant prefix, the temporary full snapshot contains **1,302,703,128 bytes (1.213 GiB)** of CPU tensor payload and takes about **0.17–0.18 s** to capture. It is released after inspection. This excludes Python metadata, existing checkpoints and transient CPU copy peaks. It avoids conversation re-prefill but does not preserve graph objects.

The proposed **embedding/head split is not implemented**. This Q4 checkpoint's input embedding is Q4_K, **0.666 GiB**; the output head is a separate, untied Q6_K tensor, **0.971 GiB**. Vision weights occupy about **0.858 GiB** before workspace, so embeddings alone are insufficient. Evicting both would free about **1.64 GiB**. Existing decode/verification/MTP graphs capture these weight addresses; an MMGP split alone cannot make them safe after reloading. Preserving graphs requires moving affected embedding/projection operations outside capture with stable boundary buffers, then measuring any added decode overhead.

Evidence: [summary](../_temp_codex/deepy_resident_vision/summary.json), [real inspection harness](../_temp_codex/deepy_resident_vision/verify_inspection.py), [short baseline](../_temp_codex/deepy_resident_vision/baseline_final_results.json), [final short run](../_temp_codex/deepy_resident_vision/resident_short_final_results.json), [long baseline](../_temp_codex/deepy_resident_vision/baseline_long_results.json), [final long run](../_temp_codex/deepy_resident_vision/resident_long_final_results.json), [MTP-off checks](../_temp_codex/deepy_resident_vision/resident_nomtp_fixed_results.json), [regression tests](../tests/test_deepy_resident_vision.py), and [source audit](../_temp_codex/deepy_resident_vision/source_audit.json). Restart WanGP to load the changes.

## Local Deepy inspection: eight images and 4K answers

This follow-up raises three constants in [`shared/deepy/vision.py`](../shared/deepy/vision.py): `VISION_MAX_IMAGES` from 5 to **8**, `VISION_VIDEO_MAX_IMAGES` from 80 to **128**, and `VISION_LOCAL_ANSWER_MAX_NEW_TOKENS` from 2,048 to **4,096**. The existing fourfold sampling divisor makes the higher-resolution video limit **32** instead of 20. The preceding section's five-image/2K measurements describe the earlier limits.

| Local inspection mode | Maximum inputs per call | Pixel budget per input | Maximum visual tokens per input | Total visual-token budget |
| --- | ---: | ---: | ---: | ---: |
| Images and/or explicitly selected video frames | 8 | 1,024 × 1,024 | 1,024 | 8,192 |
| Automatic video, default resolution | 128 frames | 256 × 256 | 64 | 8,192 |
| Automatic video, higher resolution (`mid_res_sampling=True`) | 32 frames | 512 × 512 | 256 | 8,192 |

These are pixel-area budgets for the supported local Qwen3.5/3.8 processors, which preserve aspect ratio and align image dimensions to their patch grid. They are not forced square dimensions or longest-edge limits. Smaller images can use fewer tokens. The image tool also supports explicitly selecting up to eight full-resolution video frames; automatic video sampling retains its existing two resolution modes.

Every local inspection can now produce up to **4,096 answer tokens in total**, not per image/frame. The existing **two-samples-per-second** cap remains, so short video intervals return fewer frames. Local tool schemas and input validation automatically use the revised constants. Remote limits remain **10 images, 160 default-resolution video frames, 40 higher-resolution frames, and 1,024 answer tokens**. Prompt enhancement's limits and lifecycle are unchanged. There is no new config key, setting, kernel or MMGP change.

The encoder still processes at most 1,024 visual tokens per batch on the resident path. Eight maximum-size images increase the retained Qwen3.8 image features from 50 to **80 MiB** of BF16 data, without increasing that encoder batch workspace. QA capacity continues to grow from its 16,384-token minimum when required. In the 128-frame test, approximately 2K question tokens plus frame labels and visual tokens total **13,927 input tokens**; adding the 4,096-token answer budget requests **18,023 total tokens**. Nothing is truncated to fit 16K. Actual overhead depends on labels and question length.

### Memory check and validation of the larger limits

The first eight-image Qwen3.8 run with MTP disabled exposed a small allocator peak: **17.299 GiB reserved versus 17.270 GiB initially**, a 30 MiB increase during vision prompt assembly. An allocation trace showed the concatenated/cast image embedding temporary, **80 MiB** at this limit, remained referenced after its rows had already been copied into the prompt. The precomputed-feature branch of `_prepare_multimodal_vllm_prompt` now deletes that temporary immediately after row assignment. This single added `del image_embeds` lets the allocator reuse its storage before assembling the final prompt. The pixel-input prompt-enhancer branch is unchanged; arithmetic, resolution and precision are unchanged.

Two repeated, profiled eight-image calls after this adjustment peak at **17.270 / 17.199 GiB reserved**, both within the starting footprint. Allocated memory after each call is exactly **17,944,131,584 bytes**, with no accumulation. Peak allocated memory itself remains 16.821 GiB: the improvement is reuse within the reserved pool, not a reduction in every live-tensor peak. Both answers match all 18 output tokens from the corresponding pre-adjustment short-answer run. The earlier long-answer MTP-off check also passed, emitting 3,017 tokens before EOS under the 4,096-token budget. Profiling timings are not used to claim a speed improvement. Evidence: [allocation trace](../_temp_codex/deepy_vision_limits/images8_nomtp_allocation_vision_allocations.json), [before/after comparison](../_temp_codex/deepy_vision_limits/temporary_release_comparison.json), [repeated final checks](../_temp_codex/deepy_vision_limits/images8_nomtp_release_results.json).

The larger-limit suite uses real Q4 checkpoints, Q8 KV, `vllm`, py311 and the RTX5090. The assistant has 32K capacity for Qwen3.8 and 64K for Qwen3.5-4B/9B. Each inspection has approximately 2K question tokens. The video cases preserve and restore a 20K occupied assistant prefix. The image fixture fills the 1,024² budget; the 70-second video fixture fills the relevant 256²/512² budgets. They are derived from the public [PyTorch sample image](https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg) and [W3C Sintel trailer](https://media.w3.org/2010/05/sintel/trailer.mp4).

| Real workload | Calls | Actual answer tokens per call | Starting reserved / largest inspection peak, GiB |
| --- | ---: | ---: | ---: |
| Qwen3.8, MTP4, eight maximum-size images | 2 | 3,653 / 3,653 | 17.996 / 17.996 |
| Qwen3.8, MTP4, 128 default-resolution video frames | 2 | 4,096 / 4,096 | 17.996 / 17.996 |
| Qwen3.8, MTP4, 32 higher-resolution video frames | 2 | 4,096 / 4,096 | 17.996 / 17.996 |
| Qwen3.5-9B, MTP off, eight maximum-size images | 1 | 1,266 | 6.781 / 6.781 |
| Qwen3.5-4B, MTP off, eight maximum-size images, short answer | 1 | 10 | 3.744 / 3.744 |
| Qwen3.8, MTP off, eight maximum-size images, final short-answer checks | 2 | 18 / 18 | 17.270 / 17.270 |

The first five rows were measured before the final one-line temporary release; the last row validates that adjustment. All requests have a 4,096-token answer allowance; EOS can end an answer earlier. These are capacity and regression checks, not matched speed or broad visual-quality benchmarks. In every case the tower returns to CPU, main weights retain their storage addresses, MMGP records zero main-weight transfers, and the restored conversation reproduces all **64 continuation tokens** with working semantic rewind. Whole-GPU NVML readings include desktop activity: maximum-video peaks were within about 1 MiB of their starting readings; the smaller-model/final MTP-off runs showed differences up to about 19 MiB, so no exact whole-GPU bound is claimed.

The final CPU regression run passes **17 tests**, including actual tool-schema/validation routing, local/remote sampling caps, the unchanged remote answer allowance, and resident-vision lifecycle/prompt regression tests. The source audit verifies exactly **three constant changes and one early temporary release**; the six protected engine/controller/remote/model-runner/`wgp.py`/MMGP sources remain unchanged from this follow-up's starting state. No test UI server was started. Restart WanGP to load the new limits.

Evidence: [regression tests](../tests/test_deepy_vision_limits.py), [final test log](../_temp_codex/deepy_vision_limits/tests_final.log), [runtime summary](../_temp_codex/deepy_vision_limits/summary.json), [suite commands](../_temp_codex/deepy_vision_limits/commands.json), [real-runtime harness](../_temp_codex/deepy_vision_limits/verify_inspection.py), and [source audit](../_temp_codex/deepy_vision_limits/source_audit.json).

## Live-server slowdown profile, 6 September 2026

The reported drop to approximately **20 tok/s is supported by the running server's log**. At 00:57:07.198–00:57:11.420 local time, a tool-argument action emitted **75 tokens in 4.222 seconds: 17.76 tok/s**, with **30,206 occupied context tokens**. This is OUT-to-IN log elapsed time, including outbound transcript formatting/writing and some bookkeeping, not an isolated CUDA decode measurement. That interval predates the external stack profiler. The adjacent runtime records show an existing live cache/graph signature; there is no logged model reload or compilation in that action.

The session did not run continuously at that rate. Parsing all completed generation calls with at least 64 output tokens gives:

| Action ending | Calls | Output tokens | Aggregate wall throughput |
| --- | ---: | ---: | ---: |
| Thinking completed | 27 | 16,716 | 98.06 tok/s |
| Tool arguments completed | 8 | 1,651 | 63.19 tok/s |
| Statement entering a tool call | 3 | 257 | 37.73 tok/s |
| Final response stopped | 1 | 423 | 73.99 tok/s |

These are total tokens divided by summed elapsed time, not averages of displayed rates. The table excludes very short actions and mixes contexts and content; it is not a matched benchmark. The [parser](../_temp_codex/live_slow_20260906/parse_timings.py) and [individual timings](../_temp_codex/live_slow_20260906/timings.json) preserve the measurements, including short actions.

### Runtime and sampling checks

The actual user server was PID 115648 on port 7861, running Qwen3.8-27B-Uncensored **Q4_K_M**, **Q8 KV**, **four MTP drafts**, `vllm`, CUDA graphs and the optimized SM120 asynchronous-copy attention path. Capacity was **48,128 tokens**, with occupied context reaching approximately **39.6K**. Loaded native GGUF, Triton and FA2 modules were verified from process mappings. `CUDA_LAUNCH_BLOCKING` was unset. No competing model/benchmark process was found during observation; normal desktop GPU clients were present.

The real actions used `do_sample=True`, temperature **0.6**, top-p **0.9**, and **top-k disabled**. Target nucleus filtering therefore processes the 248,320-token vocabulary; the draft head uses its existing 98,304-token prefix. The sampling-tail CUDA graph cache is already present. Repetition penalty remains enabled for tool arguments as well as natural text; **presence penalty is the one disabled in the tool phase**. Tools are not automatically greedy. No sampling or penalty settings were changed for this investigation.

The controlled 20K optimization benchmarks elsewhere in this report used top-k 20/top-p 0.95. Their timings are not directly interchangeable with this session. The earlier **160 tok/s actual Deepy tool result** is a separate case: predictable file/JSON operations at approximately 20–24K context. It must not be dismissed as merely a top-k benchmark, nor treated as a guaranteed rate for arbitrary tool arguments at 30–40K. This live capture does not contain the per-pass MTP acceptance counters needed to quantify that workload difference.

### Where the sampled time went

A nonblocking `py-spy` capture at 75 Hz sampled the same live server for 90 seconds, including its response after video generation. It contains **86.85 seconds of readable samples**, with 235 reported stack-read errors, and **20.95 sampled seconds inside `generate_action`**:

| Host stack location during decoding | Sampled time | Share of decode samples |
| --- | ---: | ---: |
| Waiting at draft-token synchronization after draft/target GPU work | 9.84 s | 47.0% |
| Speculative distribution construction, penalties and sampling-graph work | 6.21 s | 29.7% |
| Acceptance-result synchronization | 1.59 s | 7.6% |
| Sampling a token and synchronizing its result | 0.43 s | 2.0% |
| Other decode work | 2.88 s | 13.7% |

These are **CPU stack residence times, not GPU kernel durations**. In particular, time attributed to `draft_tokens.tolist()` includes previously queued GPU work; replacing that tiny transfer alone cannot remove 47% of decode time. The capture covers later actions around 38–40K context, not the earlier 17.76 tok/s action. It identifies MTP execution/sampling as the main decode path but does not establish whether low acceptance, a particular GPU kernel, or host contention caused that individual outlier.

Outside `generate_action`, the assistant has **2.04 sampled seconds in transcript logging** and **0.59 seconds preparing action replay**. The session-save thread is simultaneously in Python JSON encoding during **6.39 sampled seconds of decoding**. That overlap shows concurrent serialization, not a measured 6.39-second loss from GIL contention. The streaming rate timer currently starts before replay preparation and outbound logging, making short phases especially sensitive to these costs. Context snapshots are a separate between-action expense: 3.79 sampled seconds in this capture.

Evidence: [raw sampled profile](../_temp_codex/live_slow_20260906/resume_90s.json), [profiler completion/error log](../_temp_codex/live_slow_20260906/resume_90s.log), [reproducible analysis](../_temp_codex/live_slow_20260906/analyze_profile.py), and [profile summary](../_temp_codex/live_slow_20260906/profile_analysis.json).

### A measured, avoidable host cost

The user server was launched under VS Code/debugpy with `justMyCode=False`, Cython tracing loaded and frame evaluation disabled. CPU-only tests replayed serialization of a copied, completed session snapshot, using the same debugger configuration in a separate process:

| Operation on this session | No debugger | Debugger attached |
| --- | ---: | ---: |
| Existing complete atomic session write | 13.47 ms | **429.79 ms** |
| Indented JSON encoding alone | 12.79 ms | **425.35 ms** |
| Compact JSON encoding alone | 3.12 ms | **2.97 ms** |

The compact and indented forms were parsed and checked equal. Avoiding indented JSON on the frequent persistence path is a concrete candidate optimization with no model arithmetic or VRAM change. It preserves the saved data while removing most of this measured serialization overhead. **It has not been applied to production in this profiling pass**, and these results do not establish an end-to-end tok/s gain. The completed snapshot also differs from a larger mid-action replay payload. Evidence: [benchmark](../_temp_codex/live_slow_20260906/bench_session.py), [plain results](../_temp_codex/live_slow_20260906/session_bench_timings_plain.json), [debugger results](../_temp_codex/live_slow_20260906/session_bench_timings_debugger.json).

Full LLM I/O logging is another smaller cost. Rendering/writing the current messages and 39K token IDs, with tools omitted, took **43.8 ms without the debugger versus 145.6 ms with it**. This partial-payload measurement does not explain the whole 4.222-second action. See [transcript benchmark](../_temp_codex/live_slow_20260906/bench_transcript.py) and [debugger results](../_temp_codex/live_slow_20260906/transcript_bench_debugger.json).

### GPU readings and conclusion

The initial decode observation showed **92% GPU utilization, 517 W, 72 C and approximately 2,865 MHz**, with about **20.77 GiB whole-device VRAM**. The subsequent minute near the 600 W power limit and up to 87 C belonged to **video generation after the LLM unloaded**, not to the earlier slow tool action. The final minute of telemetry was mostly idle after Deepy completed. Neither interval proves thermal or power throttling caused the 20 tok/s drop. [Video-phase telemetry](../_temp_codex/live_slow_20260906/live_telemetry.json), [post-response telemetry](../_temp_codex/live_slow_20260906/resume_telemetry.json).

The confirmed findings are a real short-action throughput dip, substantial debugger-amplified session serialization, overhead included in the displayed rate, and significant MTP sampling/execution time. The exact contribution of MTP acceptance and GPU kernels to the historical 20 tok/s interval remains unmeasured. The next controlled comparison should retain the current sampling distribution, measure compact persistence with and without debugger tracing, and collect the existing MTP acceptance/stage telemetry during an actual slow action. Changing top-k or disabling penalties would change the requested distribution and is not a quality-neutral fix.

This pass changed only diagnostic scripts/artifacts and this report. All profiling/benchmark workers exited. The user's server, debugger, conversation and model settings were left running as they were; no test server or second model workload was launched.

### Applied follow-up: compact session persistence

At the user's request, [`session_store._atomic_json`](../shared/deepy/session_store.py) now passes `separators=(",", ":")` instead of `indent=2` to `json.dumps`. This is exactly **one production-line change**, verified against a copy taken immediately before editing. It covers the frequent `context.json` and `session.json` writes through the existing helper. JSON values, Unicode handling, `_json_safe` conversion, temporary-file replacement and cleanup are preserved. Existing indented sessions still load normally; no schema migration is needed. Export formatting remains as previously implemented.

The updated production helper was benchmarked on the same copied session payload in py311. Five writes were recorded per mode; the table uses the median of the final four, matching the earlier benchmark:

| Complete atomic session write | Before | After |
| --- | ---: | ---: |
| Without debugger | 13.47 ms | **5.40 ms** |
| With matching debugpy configuration | 429.79 ms | **11.06 ms** |

This removes about **419 ms per write** in the debugger test, a **38.9-fold reduction in that operation's elapsed time**. The approximately 3 ms number in the profiling section was encoding alone; 11.06 ms includes value conversion and the atomic file write. Both updated benchmark runs reload the written JSON and verify it equals the original normalized payload exactly. These CPU-only measurements establish the serialization improvement, not an end-to-end decode throughput multiplier. Model arithmetic, sampling and GPU allocations are unaffected by the change.

The existing session-store suite passed **36 tests**, covering persistence, unfinished-action replay, loading, media restoration, duplication and export/import. Evidence: [test log](../_temp_codex/live_slow_20260906/compact_session_tests.log), [updated benchmark](../_temp_codex/live_slow_20260906/bench_session.py), [plain results](../_temp_codex/live_slow_20260906/session_bench_timings_compact_plain.json), [debugger results](../_temp_codex/live_slow_20260906/session_bench_timings_compact_debugger.json). The previous benchmark result files were retained. **Restart WanGP to load this Python change**; the running user server was not restarted or modified in memory.


## Release 1.0.21: compiled RTX50xx async kernels (6 September 2026)

The RTX50xx Q8 async prefill and grouped decode/verification kernels now ship as precompiled SM120 cubins in the GGUF wheel, with a native CUDA-driver launcher. Four fixed binaries per CUDA major cover FP16/BF16 and prefill/grouped operation. Query counts, head counts, context lengths, page counts and splits are runtime arguments; changing them does not compile another variant. These use `cp.async`, not TMA. The original high/low tensor-core arithmetic and partial buffers are retained; no additional persistent tensor cache or lower-precision mode was introduced.

The latest shared MMVQ/MMQ, typed activation quantization and standard attention implementation was built for every architecture supported by CUDA 12.8 and 13.1, on both Windows and Linux/WSL. The Windows/Linux PyTorch 2.7.1 and 2.10.0 wheels each passed 280 linear configurations, embeddings, 12 standard attention cases, 3 padding cases and 20 compiled SM120 attention configurations. Another 24 compiled-path/backend-isolation tests passed. Hardware validation was limited to RTX5090; the other architecture fatbins were inspected.

Repeated real Qwen Q4 calls passed on all four stacks with four-token MTP and aligned caches. Windows/PyTorch 2.10 at 20K measured 121-128 decode tokens/s and 3,025-3,073 prefill tokens/s; Linux/PyTorch 2.10 measured 129-138 and 3,006-3,021 respectively. These are validation measurements, not controlled speedup comparisons. Peak PyTorch allocation remained below 17.64 GiB (driver/code and other processes excluded).

The PyTorch 2.7 checks also identified and fixed explicit-device handling for MTP arange outputs and the startup probe's Triton symbol visibility. Unsupported pre-3.3 Triton compilers are rejected on SM120 before they can abort on reductions. Windows 2.7 vllm testing used an isolated compatible Triton; the installed environment was left unchanged. Source packaging fixes prevent stale incremental binding objects and remove Conda's absolute Linux library path.

Full source, build instructions, wheel hashes and per-stack validation evidence: [GGUF 1.0.21 release](https://github.com/deepbeepmeep/kernels/releases/tag/gguf-v1.0.21) and [release validation report](https://github.com/deepbeepmeep/kernels/blob/gguf-v1.0.21/llama.cpp/release/1.0.21/README.md). Local installation URLs are updated in INSTALLATION.md; WanGP changes remain unpushed.
