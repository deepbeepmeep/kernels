# Qwen3.8 block drafters (DFlash2, DSpark) versus MTP (25 September 2026)

RTX 5090, Deepy thought path, top-k 20, top-p 0.9, 512 tokens, contexts 2K and 20K, GGUF 1.0.24 candidate short-batch kernels. Cycle time is decode seconds per target pass; tokens per pass depends on the sampled text.

## Benchmark flaw found

`tools/benchmark_qwen38_engine.py` never called `int8_backend.configure(...)`, which `wgp.py` runs at startup from the `int8_kernels` setting. The INT8 ConvRot drafters (2.2 GB) therefore ran on the PyTorch fallback, which dequantizes every weight to BF16 per call (95 us elementwise kernel plus a BF16 GEMM, 21% of decode time). Earlier DFlash2/DSpark conclusions that relied on this benchmark overstated the drafter cost; MTP (GGUF) was unaffected. The benchmark now configures INT8 kernels like WanGP (`--int8-kernels`, default `auto`) and records the backend.

With `auto` = Comfy Kitchen CUDA, the DFlash2 draft stage fell from 9.8 to 4.3 ms. The Triton backend (users without Comfy Kitchen) uses the fused ConvRot INT8 kernels: draft 4.9 ms, no dequantization.

## Results (WanGP INT8 kernels)

| Qwen3.8 Q4_K_M | Cycle | Tokens/pass | tok/s |
| --- | ---: | ---: | ---: |
| MTP, 4 drafts | 17.0-17.8 ms | 2.35-2.93 | 137-168 |
| DFlash2, 7 drafts | 19.4-21.2 ms | 2.28-3.24 | 116-153 |
| DSpark, 7 drafts, before | 24.4-26.3 ms | 2.17-2.59 | 87-100 |
| DSpark, 7 drafts, single-graph chain | 20.6-21.5 ms | 2.15-2.57 | 103-125 |

| Bonsai PTQ1_0 | Cycle | tok/s |
| --- | ---: | ---: |
| MTP, 4 drafts | 13.0-13.5 ms | 168-205 |
| DFlash2, 5 drafts | 14.4-15.2 ms | 146-195 |
| No drafts | - | 118-132 |

## DSpark: single-graph chain for sampled decoding

Sampled DSpark (any top-k other than 1) used the reference loop: a full-vocabulary distribution and host sampling per draft, several scalar reads per draft, and reference acceptance. Greedy DSpark replayed one graph per draft and read its confidence flag after each.

`BlockDraftRunner._build_dspark_gpu_chain` now captures the dependent Markov chain once per `(drafts, threshold, top_k, top_p, min_p, temperature)`:

- scores = unary logits + Markov head of the previous draft + bias/token masks, as before;
- sampled: q = `target_probabilities` (the filter used by GPU acceptance), draft = argmax(q / Exp(1) noise), an exact draw from q; greedy: argmax, lowest ID among ties;
- the confidence threshold and fully masked rows end the valid prefix (`_draft_valid_length`) instead of a host read; suffix drafts are discarded by acceptance.

Acceptance uses the shared GPU block acceptance with that valid length for greedy and sampled decoding, one completed-block readback per cycle. Rejection sampling stays exact because each draft is drawn from the q passed to acceptance. Non-vLLM paths (no Triton sampling, eager) keep the reference loop. `tests/test_dspark_gpu_chain.py` checks greedy tokens and confidence prefixes against the sequential reference, sampled draws against the reported q (4000 draws), graph reuse, token masks and dispatch.

DSpark remains below MTP because its drafts are accepted less often.

## DFlash2: where the cycle goes

Per cycle at 2K (Kitchen): verify 13.6 ms (MTP 13.0: 8 rows instead of 5), draft 4.3 ms (MTP 2.6), sampling 1.7 ms. Draft kernels: Kitchen CUTLASS INT8 linears 1.1 ms, target output head on 7 rows (Q6_K MMVQ, 1.45 TB/s) 0.73 ms, Triton wide ConvRot down projections (K = 17408, beyond Kitchen's CUTLASS width) 0.58 ms at 0.76 TB/s, drafter attention 5 x 60 us, candidate top-k and small kernels.

Examined, not changed:

- Drafter attention (8 queries, 2K sliding window) is latency-bound at 50-60 us in flash-attn (all split counts), cuDNN and efficient SDPA alike: only 8-32 thread blocks on 170 SMs. A custom split-KV kernel would save about 0.25 ms per cycle (~1%).
- Target Q6_K at 8 rows runs llama.cpp MMQ at ~1.5 TB/s already.
- GPU acceptance falls back to the exact reference sampler in 10-15% of DFlash2 rounds (MTP 2-5%): a BF16 logit tie exactly at the top-p boundary in any reachable row, and DFlash2 checks 8 rows. Removing it requires either a defined tie rule shared with the reference sampler or restarting the reference only from the tied row; about 2% at stake.
- Remaining gains (wide ConvRot tuning for 8 rows, attention) total roughly 1 ms per cycle, 5%.
