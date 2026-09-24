# DFlash2 upstream audit — 2026-09-20

Follow-up 2026-09-22: [a draft RoPE configuration error was found and corrected](DFLASH2_ROPE_AND_DRAFT_COUNTS.md).
The performance and acceptance measurements below predate that correction.

Follow-up: [GPU block acceptance implemented and validated](DFLASH2_GPU_ACCEPTANCE.md).

The local implementation runs DFlash2's selector, but does not reproduce upstream's optimized execution pipeline. Two problems matter: low useful output per verification and expensive verification/drafting. Moving acceptance to the GPU is worthwhile, but cannot alone turn this workload into a 3x speedup.

No production decoder, kernel, application default or quantization behavior was changed for this audit. Experiments use process-local overrides in `tools/audit_dflash2_runtime.py`.

## Evidence and scope

- RTX 5090 only, PyTorch 2.10/CUDA 13, GGUF kernels 1.0.22. Target: uncensored Qwen3.8 27B Q4_K_M; original BF16 Qwen DFlash2 drafter. Full checkpoint/binary identities and results are in `benchmarks/dflash2_upstream_audit_2026-09-20.json`.
- Two 2048-token fiction prompts, 256 sampled output tokens each, warm graphs, 32K capacity, Q8 KV. Temperature 0.6, top-p 0.95, top-k 20, min-p 0.05, repetition penalty 1.05; predictive repetition penalty disabled. Benchmark `interrupted` means the intentional 256-token callback limit.
- Upstream comparison: [publisher model card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) and SGLang revision `3a64faa1f22a86abd37a759c84267d929e820d5b`. Code was inspected, not run locally. That revision is current inspected source, not a verified revision of the publisher's benchmark.
- The publisher used official Qwen on H200/FA3, temperature 1.0, xhigh reasoning, up to 4096 output tokens, and benchmark suites rather than these fiction prompts. Its 2.67–3.43x is end-to-end; our short test reports decode-only. These are not controlled cross-engine speed comparisons.
- The publisher reports 4.10–5.46 output tokens per verification. Our original seven-draft run emits 512/221 = **2.317**. This includes the correction/bonus token, like upstream's acceptance-length definition. Local `drafted` counts positions visited before rejection, not all proposed tokens; it must not be used as the proposal-count denominator.

## Measured cost

27 stage samples across the two prompts, sampling every eight verification rounds:

| Stage | Mean elapsed GPU-event interval |
| --- | ---: |
| Target verification | 16.27 ms |
| Draft backbone, output head and selector | 5.50 ms |
| Target sampling and rejection loop | 3.23 ms |
| Target output head | 0.75 ms |
| Append accepted context to drafter | 0.50 ms |
| Setup, commit, truncate and pre-sampling gap combined | 0.41 ms |

These intervals include GPU idle gaps between submitted work. They are not exclusively kernel execution time. In particular, the CPU wait at `draft_tokens.tolist()` overlaps target verification and must not be counted as another 16–19 ms.

The separate 64-token PyTorch trace contains 478.75 ms of GPU kernel time: GGUF matrix kernels/fixups account for 275.40 ms (57.5%), BF16 GEMMs 67.96 ms (14.2%), and top-k kernels 8.52 ms (1.8%). The seven-row Q6_K draft output head alone is 16.90 ms over 21 rounds, about 0.80 ms per round. Trace/profiler overhead means these numbers are for attribution, not throughput scoring. Raw trace: `tmp/dflash2_audit/profile/decode_trace.json`.

The GPU selector chain was used for **231/231** proposals, including warmup. There was no unexpected Python selector fallback on this workload.

## Concrete differences

1. **Acceptance still synchronizes with Python per tested token.** `ModelRunner.run_mtp` converts draft IDs to a host list, builds target distributions one position at a time and reads each acceptance decision with `.item()`. Upstream builds the verification distributions in a batch and calls a GPU chain rejection sampler. Our 3.23 ms sampling interval is a real optimization target. Even removing that entire interval from the measured 26.66 ms stage total would only give approximately **1.14x additional speed**, not 2–3x.

2. **Our selector graph is separate and fragmented.** `BlockDraft.propose` captures backbone plus output head; `BlockDraftRunner._build_dflash_gpu_chain` copies hidden states/logits into another graph and reads `length.item()` before verification. Within that graph, seven positions each do top-k, projection, gathers and scoring. Upstream computes the candidate transition lattice in parallel, performs the path walk in a Triton kernel, and captures the selector as the tail of the draft graph. Having a CUDA graph around many small operations is not the same as those fusions.

3. **Verification is the dominant cost.** Seven drafts produce eight target rows. On SM120 the installed dispatch uses MMVQ through five rows for Q4_K and seven rows for Q6_K, then MMQ. The trace confirms eight-row MMQ dominates. This is a place to benchmark actual Q4/Q6 layer shapes and fused variants, not justification to force MMVQ: the current cutoffs are architecture-specific, and the three-draft experiment below did not improve throughput. Changes here would also benefit other speculative methods; vanilla's single-row path would not automatically benefit.

4. **Selecting fewer tokens does not shrink our draft backbone.** The loaded block size remains eight; all seven hidden states and vocabulary projections are calculated even when only three or five drafts are requested. Upstream propagates runtime block size into the model and convolutions. Shrinking our block would change the noncausal draft computation and must be validated for acceptance as well as speed. This audit did not enable that change.

5. **Draft sampling differs.** We apply target-style top-p/min-p filtering to the 16 draft candidates. Upstream samples those candidates with temperature-only softmax, then applies the target filters during verification. Both proposal distributions can be valid under exact rejection sampling; this difference is not itself proof of incorrect target output. An ablation matching upstream's proposal distribution did not close the acceptance gap.

6. **Several draft primitives remain unfused.** Local Q/K/V projections, MLP projections, RMSNorm and grouped convolution operations use the straightforward PyTorch composition inside the graph. Upstream uses merged projections and compiled/fused operations. Its FlashInfer radix top-k is also absent here. Upstream's source warns about large-vocabulary `torch.topk`, but our trace does **not** support claiming that replacing top-k would double local throughput. The BF16 backbone and quantized output projection cost substantially more here.

Source references:

- [SGLang draft model, candidate lattice and runtime block sizing](https://github.com/sgl-project/sglang/blob/3a64faa1f22a86abd37a759c84267d929e820d5b/python/sglang/srt/models/dflash.py)
- [Draft graph tail and sparse proposal-buffer handling](https://github.com/sgl-project/sglang/blob/3a64faa1f22a86abd37a759c84267d929e820d5b/python/sglang/srt/speculative/dflash_worker_v2.py)
- [Triton selector path walk](https://github.com/sgl-project/sglang/blob/3a64faa1f22a86abd37a759c84267d929e820d5b/python/sglang/kernels/ops/speculative/dflash.py)
- [GPU rejection-sampling integration](https://github.com/sgl-project/sglang/blob/3a64faa1f22a86abd37a759c84267d929e820d5b/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py)

Upstream also scatters sparse proposal probabilities into a reusable dense buffer for verification, clearing only written entries afterward. It does not keep verification entirely sparse. Our graph creates/zeros full-vocabulary proposal rows; avoiding that work is useful but not the principal measured cost.

## Ablations

| Experiment | Decode tok/s | Output tokens / target pass |
| --- | ---: | ---: |
| Original seven drafts, prior uninstrumented reference | 85.52 | 2.317 |
| Seven drafts, stage sampling every eight rounds | 90.74 | 2.317 |
| Three drafts, uninstrumented | 84.43 | 2.169 |
| Five drafts, uninstrumented | 96.28 | 2.599 |
| Seven drafts, upstream temperature-only proposal, uninstrumented | 90.40 | 2.438 |

Throughput varies between runs (the single-prompt profile run measured 95.83 tok/s with unchanged seven-draft acceptance). Instrumented and uninstrumented numbers are not a valid direct speed comparison. Five drafts is promising for these prompts, but this small sweep does not establish a universal best count or justify changing Auto. Changing the proposal law/count changes the sampled continuation, so these are workload-level ablations, not identical-token kernel comparisons.

The upstream-style proposal improves this small sample's emission count by about 5%, nowhere near the roughly doubled acceptance needed to explain the publisher's results. The checked target feature convention is correct: local capture after target layer `i` corresponds to upstream/HF hidden state `i+1`. This rules out that simple indexing error, **not** every possible numerical or context-alignment difference.

The effects of uncensoring, target quantization, prompt domain, target sampling settings and remaining implementation differences have not been isolated. It would be unsupported to blame the acceptance gap entirely on Q4 or the uncensored checkpoint.

## Work in priority order

1. Establish controlled acceptance parity using real captured target features, embeddings and output weights against the reference drafter/selector, and a small identical prompt suite. Preserve target filters, repetition penalties, tool masks, stop handling and exact rejection sampling. This separates implementation errors from target/workload mismatch.
2. Add a GPU common path for batched target distribution processing and rejection, retaining the existing path for arbitrary stateful Python processors. Fold the candidate selector into the draft graph and avoid unnecessary host handoffs. Test exact greedy continuation, sampled distributions, repeated graph reuse, cancellation and vision/context restoration.
3. Benchmark the actual SM120 small-matrix verification shapes and output head; optimize without dequantizing full weights or affecting the single-row path. Keep other GPU dispatch unchanged.
4. Propagate shorter runtime draft blocks, measure acceptance-adjusted throughput, then consider drafter projection/convolution fusions. Quantizing the drafter is a separate precision change requiring validation, not an assumption of a free gain.

At the measured 2.32 tokens per pass, merely reaching 2.67x the prior 65.45 tok/s vanilla baseline would require approximately 13.3 ms per full cycle. Current target verification alone averages 16.3 ms. Recovering 3x requires substantially better acceptance and/or lower verification cost, not just removing Python overhead.
