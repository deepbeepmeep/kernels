# GGUF 1.0.22 release validation

Validated on 20 September 2026 with an RTX 5090, Windows and Ubuntu 22.04 under WSL. Other NVIDIA architectures were compiled and their native binary inventories inspected; they were not tested on hardware.

## Implementation

This release merges Prism PTQ1_0 into the existing GGUF packed MMVQ/MMQ implementation. It adds signed Hadamard transforms and a fused Prism decode extension while retaining the existing quantization dispatch, typed activation loaders, bounded scratch buffers, bias/output conversion and attention APIs. The default packed path does not retain full dequantized PTQ1 weights. The explicitly selected materialized/cuBLAS mode keeps its existing behavior.

All three extensions (`_C`, `_attention`, `_prism`) include the full toolkit target set. CUDA 12.8 contains SM50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101 and 120. CUDA 13.1 contains SM75, 80, 86, 87, 88, 89, 90, 100, 103, 110, 120 and 121. Each includes PTX for the highest target. CUDA 13 does not support offline compilation of pre-SM75 targets; PyTorch's own hardware support also applies.

The WanGP source overlay includes the current PTQ1 projection/layout preparation, GDN optimizations, MTP GPU acceptance and DFlash2/DSpark integration. Existing engine separation is preserved. Automatic fused Prism/GDN selection remains restricted to the validated SM120 vLLM path; other GPUs retain the packed/shared paths. Architecture-specific Q8 async cubins remain SM120-only. Other shared Triton kernels compile at runtime when selected; the source archive includes their source, not a universal precompiled Triton cache.

## Numerical and binary checks

Every final wheel passed:

- 870 bit-identical comparisons against its matching 1.0.21 wheel, covering all 18 existing formats, both linear modes, FP16/BF16/FP32 and supported embeddings.
- 280 packed-linear configurations including real Qwen Q4_K/Q6_K weight slices, 12 standard attention cases, 3 activation-padding cases and 20 compiled SM120 attention cases.
- 240 PTQ1 linear configurations and embedding checks, including tails and CUDA-graph replay with changed inputs.
- 81 signed Hadamard configurations and 360 fused Prism launch configurations, with dtype, bias, grouping and graph-replay coverage.
- The SASS/PTX inventory for all three native extensions. Linux libraries link the correct CUDA major and use only package-relative runtime search paths.
- 12 focused shared GDN regression tests per stack, covering single-token calls without snapshots, multi-token prefix states and graph replay/rollback.

The adjacent JSON reports contain actual binary hashes, architecture inventories, compiler details, test outputs and build durations. Initial full-build records are retained separately from later metadata-only packaging runs.

## Real checkpoint checks

Both Qwen3.8-27B Q4_K_M and Bonsai PTQ1 ran through WanGP's actual vLLM path on all four stacks. Each used native MTP with two draft tokens, Q8 KV, 32K cache capacity and two separate 2,048-token prompts after warmup. Each call requested 64 output tokens; the stop callback can finish the current speculative block. GPU block acceptance was active, graph reuse succeeded and every run reported `sync_delta=0`.

Generated text was inspected and contained coherent short planning prefixes. These short checks do not establish long-form answer quality. Output samples and per-call metadata are retained alongside `summary.json`.

| Stack | PyTorch | Q4 and PTQ1 repeated MTP | Q4 peak allocation | PTQ1 peak allocation |
|---|---|---|---|---|
| win_py310 | 2.7.1+cu128 | Passed | 17.49 GiB | 7.79 GiB |
| win_py311 | 2.10.0+cu130 | Passed | 17.49 GiB | 7.79 GiB |
| linux_py310 | 2.7.1+cu128 | Passed | 17.49 GiB | 7.79 GiB |
| linux_py311 | 2.10.0+cu130 | Passed | 17.49 GiB | 7.79 GiB |

The VRAM figures are peak PyTorch allocations, not total board usage; driver/context memory and other processes are excluded. These were release checks, with CPU compilation overlapping some runs, not controlled speed comparisons. No cross-platform throughput improvement is claimed from them.

Windows/PyTorch 2.7 vLLM checks used an isolated Triton 3.3.1.post21 package because the environment's installed Triton 3.2 does not support SM120. Existing installed environments were not changed.

The Linux 2.7 wheel was compiled and passed the native suite with Conda's Python 3.10.9. That interpreter encounters a decorator-source inspection error in the installed FLA package, so ordinary inference uses the existing GDN fallback. To validate the optimized path as well, the final Linux 2.7 GDN and real-model checks used the installed system Python 3.10.12 with the same Conda PyTorch 2.7.1/Triton 3.3.1 packages through an isolated process search path. The existing Conda Python 3.10 headers were supplied through `C_INCLUDE_PATH` for Triton's launcher compilation. Both Q4 and PTQ1 then used fused GDN. No interpreter or dependency was replaced in the user's environments; the wheel retains the standard CPython 3.10 ABI.

Release testing found an older-Triton compilation failure when a disabled GDN snapshot pointer was guarded by a combined static/dynamic condition. The overlay now puts the optional pointer behind a separate constexpr branch, without changing recurrence arithmetic. Focused GDN tests and real Q4/PTQ1 inference were rerun on all four stacks after this fix. The native wheels did not require rebuilding for this Python/Triton source change.

## Checkpoints and reproduction

The tested PTQ1, adapted Q8 MTP sidecar, DSpark and Qwen/Bonsai DFlash2 assets were published to [DeepBeepMeep/Wan2.1](https://huggingface.co/DeepBeepMeep/Wan2.1/commit/45aaad562bd1944267db209b704c0be6d64c5313). `checkpoints.json` records the verified remote sizes and SHA-256 hashes; provenance and available upstream license notices accompany the uploaded assets.

Use the matching environment with `scripts/build_release.py`, `scripts/inspect_wheel.py` and `scripts/validate_wheel.py`; see the native README for commands. Source archives include vendored GGML sources, license notices, native/Triton kernel sources, build scripts, SM120 GPU programs, tests and the WanGP source overlay. `wangp/SOURCE_MANIFEST.json` hashes the LF-normalized source snapshot. No checkpoints, credentials or user configuration files are in the kernel source archive.
