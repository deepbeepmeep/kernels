# Ternary Bonsai 2 PTQ1_0

Choose **Qwen3.8-27B Uncensored** as the local prompt-enhancer/Deepy model, then choose **Bonsai 2 Abliterated PTQ1_0** in its quantization dropdown. This uses the existing `prompt_enhancer_quantization` config key with value `gguf_ptq1`. The existing `lm_decoder_engine` selects `legacy`, `cg`, or `vllm`.

The selected filename is `Qwen3_8_27B_Uncensored/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf` under a configured checkpoint root. On the development machine it resolves to `E:\ML\Wan2GP\ckpts\Qwen3_8_27B_Uncensored\Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf`. Existing tokenizer/config/vision assets in that folder are reused. The built-in download declaration follows the shared DeepBeepMeep repository convention; the local checkpoint was used for validation and has not been uploaded by this change.

Install WanGP `llamacpp-gguf-cuda` 1.0.22 or newer in the environment running WanGP, and restart an already running application after installing. The locally tested wheel is `E:\ML\kernels\dist\1.0.22-sm120\llamacpp_gguf_cuda-1.0.22-cp311-cp311-win_amd64.whl`: Windows, Python 3.11, Torch 2.10/CUDA 13, RTX 5090 (SM120). Other GPU/ABI builds must be produced from the source with the appropriate release target. The kernel source is `E:\ML\kernels\llama.cpp`.

PTQ1_0 is GGUF extension type 143: 128 ternary values in 28 bytes. Runtime support includes packed linear/embedding kernels and the checkpoint's signed Hadamard transforms. These transforms are required for correct outputs. Native GPU kernel failures propagate instead of silently expanding the full model into dense weights. The default packed path keeps weights compressed. The existing explicit `cublas` diagnostic mode still materializes weights.

The loader preserves the checkpoint's FP32 and BF16 precisions through MMGP profiling. Sign vectors are registered buffers managed by MMGP. In addition to gate/up fusion, Bonsai's linear-attention QKV and gate projections share one packed projection and Hadamard transform; full-attention Q/K/V projections likewise share one. The 96 unrotated BF16 alpha/beta matrices form 48 BF16 pairs, separately from the rotated gate. Bonsai's GDN projection rows, convolution channels and small state-space parameters are rearranged once on CPU during loading. Recurrent attention then stays in grouped head order, eliminating seven permutations per linear-attention layer per token, including the mutually cancelling output/Hadamard permutations. Packed weights remain packed. Other Qwen variants retain their existing projection schedules. Multi-token convolution verification receives contiguous QKV slices from the fused output.

Speculative decoding now uses the existing shared speculative-token control and a separate `Ternary-Bonsai-2-27B-MTP-Q8_0.gguf` sidecar in the same checkpoint folder. Two draft tokens are the recommended starting point from the local measurements below; zero disables MTP and avoids loading the sidecar. The head is adapted by [ProCreations](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP), pinned to revision `6f66852e436806f0db2b6ff5351bf30a0eaa95f9`. WanGP applies the inverse Hadamard transform to its shared token embedding and the forward transform to its shared output head, including the restricted draft-vocabulary view. The target model verifies drafts using the existing rejection sampler; the target GGUF is unchanged. **Speculative Decoding** offers Auto, Disabled and MTP. For Bonsai PTQ1, Auto disables MTP when total GPU VRAM is 10 GiB or less and selects exactly two MTP draft tokens above 10 GiB. Explicit MTP selections retain their chosen token count. DSpark and DFlash2 are available as explicit choices for the Auto/vLLM decoder engine. Both need roughly 4–5 GiB of additional VRAM and can be slower than MTP; Auto continues to select MTP. Bonsai DFlash2 allows up to five draft tokens. Increasing the count does not guarantee faster generation. Other Qwen variants retain their existing Auto VRAM policy. See [the integration and measurements](../specs/BLOCK_DRAFT_DECODING.md).

The 15-tensor sidecar contains Q8_0 matrices and FP32 effective norm multipliers, matching the publisher's export convention. `tools/prepare_bonsai_mtp.py` verifies the pinned BF16 source SHA256, exports the sidecar, rereads every tensor, and records provenance beside it. Source hash: `c7d477c1dff218744069dbf0a8e287fbc29c2a0bab38183c03dc1798bb2b0643`; prepared sidecar hash: `fe412712ca637b87df4746b0837d469e98c55b0c2b739f9e2625387ef748bc37`. The sidecar, provenance and license/notice are published in [DeepBeepMeep/Wan2.1](https://huggingface.co/DeepBeepMeep/Wan2.1/tree/main/Qwen3_8_27B_Uncensored) and available through the registered download declaration.

Prism's published RTX 5090 PTQ1_0 speed is 120.5 tok/s for `llama-bench` TG128, batch 1, starting context depth 0, without vision. The 129.9 tok/s entry is PQ2_0, a different packing. These are not speculative serving benchmarks. The official Windows CUDA 13.3 executable, release `prism-b10685-7dffb15`, was also tested on the same RTX 5090 and local abliterated checkpoint: three-repeat means were 115.1, 124.6 and 116.2 tok/s at starting depths 0, 256 and 2,048. That test uses F16 KV and random tokens without WanGP's application sampling; it establishes a local upstream reference, not an identical workload. WanGP has not demonstrated 130 tok/s target-only parity. See [the model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf#cross-platform-throughput).

Initial projection-fusion performance check, September 18, 2026: RTX 5090, installed kernel 1.0.22, abliterated PTQ1_0, vLLM/CUDA graphs, Q8 KV, 512 generated tokens, two repeats of each prompt. Model load, graph warmup and prompt processing are excluded. Rates are total generated tokens divided by total decode time. These use a prose-generation task with synthetic reference text, not Prism's zero-context benchmark or the user's live conversation. This first table predates the additional GDN layout optimization.

| Prompt tokens | Original schedule | Optimized, no MTP | Optimized, MTP 2 | Optimized, MTP 4 |
|---|---|---|---|---|
| 256 | 89.4 | 105.3 | 112.8 | 104.0 |
| 2,048 | 91.0 | 106.2 | 102.6 | 91.0 |

The target-only improvement is 16.6-17.8%. MTP acceptance was 61-70% with two drafts, and it is workload-dependent: it improved the short-context run but was slightly slower than the optimized target-only path at 2,048 tokens. Four drafts were slower than two in both tests. Target-only peak VRAM increased by about 7 MiB; MTP 2 peaked at 7.60-7.79 GiB. GPU conditions and stochastic output differences can affect short measurements. Metadata and per-run results are preserved in [the benchmark record](benchmarks/bonsai_mtp_2026-09-18.json); raw prompts/completions are under `tmp/bonsai_perf_check/final_*`. The reproducible harness is `tools/benchmark_qwen38_engine.py`, with `--unfused-prism` selecting the original projection schedule for comparison.

A later back-to-back run isolates the additional GDN layout change. Absolute speeds shifted between sessions, so the gains from these tables must not be added together:

| Prompt tokens | Fused, token-time reordering | Final grouped layout, no MTP | Final grouped layout, MTP 2 |
|---|---|---|---|
| 256 | 91.3 | 102.4 | 100.6 |
| 2,048 | 92.3 | 99.2 | 96.7 |

The final layout adds 7.4-12.1% in this paired test, with unchanged peak VRAM. An independent final-layout run measured 99.4-99.5 tok/s. Two-draft acceptance was 61-65% in the final run; MTP is supported but is not a guaranteed speedup, and zero drafts was faster for these prompts. The existing local speculative-token config was changed from four to two; selecting zero tests the optimized target-only path. The benchmark's `--prism-tiled-gdn` switch restores token-time reordering for comparison. Both optimization stages are specific to Bonsai; the shared sampler and engine scheduler are unchanged.

The next paired test isolates single-token GDN preparation and convolution launch geometry on SM120 in `vllm` mode. A single kernel produces repeated Q/K heads, decay gates and beta, replacing the separate pointwise operations and copies. The four-tap FLA convolution uses 64 channels per program instead of eight. Prefill and multi-token verification retain their established paths. These changes keep checkpoint dtypes and the native 1.0.22 ABI. They were subsequently extended to the existing Qwen 27B Q4/Q3/Q2 GGUF variants through [the shared GDN implementation](QWEN_GDN_DECODE.md); `cg` and `legacy` retain their kernels. Bonsai's load-time layout conversion remains separate and specific to Bonsai.

| Prompt tokens | Grouped layout before this change | Fused GDN decode, no MTP |
|---|---|---|
| 256 | 100.2 | 110.8 |
| 2,048 | 101.8 | 109.5 |

This adds 7.6-10.6% in the paired test, with unchanged peak VRAM (6.894 and 7.066 GiB). A separate final run measured 112.5 and 108.8 tok/s. The diagnostic `--prism-reference-gdn` switch restores the original GDN preparation and convolution launch while retaining previous optimizations. A warp-per-row PTQ matrix kernel and output-head graph capture were tested and rejected because they were slower; neither is enabled.

A final longer-context check generated 512 tokens twice from 20,000-token prompts with MTP disabled: 100.2 and 105.1 tok/s (102.6 combined), with 7.066 GiB peak allocation. Prompt processing took 12.1-12.6 seconds and is excluded from those decode rates. No before/after claim is made for this context length because this pass did not measure its previous implementation.

The upstream executable was also rerun with Q8_0 K/V and 512 generated tokens at the same two starting depths: 114.1 and 123.6 tok/s (three-repeat means). Its random-token benchmark still omits WanGP application sampling and scheduling, so this narrows the comparison without making the workloads identical. The remaining WanGP GPU profile is dominated by packed PTQ matrix multiplication (61.8%), followed by RMS normalization (6.3%), Hadamard transforms (4.5%) and activation quantization (4.0%). These results do not establish 130 tok/s target-only throughput. Raw runs and profiles are under `tmp/bonsai_warp/`; the benchmark record retains their measurements.

Validation on RTX 5090:

- 870 outputs for all 18 existing qtypes are bit-identical to the original kernel-source baseline.
- 240 PTQ1_0 linear configurations plus embedding checks pass independent CPU references and changed-input CUDA-graph replay, including odd block counts and row tails.
- 81 signed Hadamard configurations match the independent reference exactly, including GDN permutation and graph replay.
- The real checkpoint generates correct repeated answers in legacy, CUDA-graph and vLLM modes, with CPU and CUDA defaults. Each mode recovers after cancellation. Image captioning through the shared vision/embedding path correctly describes the downloaded COCO two-cats image. The `inspect_media` sequence also passes after a text decode, including MMGP unload/reload and resident reuse of the Prism embedding with CPU processor inputs.
- FakeTensor contracts cover the new native operators. Validation does not establish universal GPU compatibility, long-context quality, or full benchmark parity with the separate Prism llama.cpp runtime.
- The final 70-test regression suite covers Prism transforms, projection/layout equivalence, speculative state and sampling, configuration, Q3 sidecars and resident vision contracts. Real-checkpoint checks cover both default devices in every decoder mode. The other Qwen3.8 Q4 vision/MTP control also passes repeated calls and cancellation recovery.
- The subsequent GDN change passes numerical and changed-input graph tests, including exact convolution state, strided Q/K inputs and extreme gate values. Convolution output allows rare one-BF16-ULP differences from Triton tile geometry, with relative RMS error below 0.0001. The expanded suite passed 79 tests; one unrelated telemetry test fails because it changes an environment variable after import while the unchanged runtime reads it at import time.
- Final GDN real-checkpoint tests passed with MTP off/on and both CPU/CUDA defaults, preserving checkpoint dtypes and recovery after cancellation. Repeated vision-to-text workflows passed for Bonsai target-only, Bonsai MTP and the ordinary Qwen3.8 Q4 MTP control, including cancellation inside the vision tower and decoder, successful recovery and MMGP resident reuse. Reports are under `tmp/bonsai_warp/check_*` and `tmp/bonsai_warp/vision_*`.

Reproduce the real-checkpoint checks:

```powershell
python tools/test_bonsai_ptq1.py --checkpoint E:\ML\Wan2GP\ckpts\Qwen3_8_27B_Uncensored\Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf --engine vllm --default-device cuda --output validation/bonsai-vllm.json
python tools/test_bonsai_ptq1.py --checkpoint E:\ML\Wan2GP\ckpts\Qwen3_8_27B_Uncensored\Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf --engine vllm --default-device cuda --draft 2 --output validation/bonsai-mtp.json
python -m pytest tests/test_prism_gguf.py -q
```

Repeat the first command with `legacy` and `cg`, and with `--default-device cpu`. Kernel-side reference, compatibility and timing scripts are under the kernel repository's `tests/` folder. The source/provenance mapping is in its `PRISM_PTQ1_PORT.md`.

---

> Applies to: Bonsai 2 PTQ1_0 prompt enhancement and Deepy, including decoder selection and speculative decoding.
