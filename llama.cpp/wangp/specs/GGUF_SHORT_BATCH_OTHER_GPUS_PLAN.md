# Short-batch tensor-core GGUF linear on all GPUs (compute capability 8.0+)

Status: implemented in GGUF 1.0.24 (release candidate wheels built for PyTorch 2.10/CUDA 13 and PyTorch 2.7.1/CUDA 12.8, not published) and WanGP. Background and RTX 5090 kernel results: [RTX50_SHORT_BATCH_GEMM_20260925.md](RTX50_SHORT_BATCH_GEMM_20260925.md).

## Why other GPUs can benefit, and why we measure instead of assuming

The kernel fixes two architecture-independent costs of 2-8 row speculative verification: scattered MMVQ weight reads, and PTQ1_0 trits re-decoded once per activation row. It uses INT8 `mma.sync.m16n8k32` and `cp.async`, available from compute capability 8.0 (Ampere, Ada, Hopper, datacenter and consumer Blackwell). Older targets compile trapping stubs that dispatch never selects. Shared memory (<= 43 KB per block) fits every candidate.

Correctness transfers: integer dot products are exact; outputs differ from MMVQ only by activation-scale precision and FP32 accumulation order. Speed does not transfer reliably: it depends on bandwidth per SM, L2 size and scheduling, and the tiles were tuned on one RTX 5090. Only the SM120 binary can run here, so every GPU decides for itself on its own weights.

## Kernel package (`E:/ML/kernels/llama.cpp`, 1.0.24)

- `csrc/short_batch_mma.cu` (renamed from `blackwell_short_batch.cu`), dispatched from `run_linear_cuda` for Q4_K/PTQ1_0, 2-8 rows, `K % 256 == 0`, compute capability >= 8.0, CUDA builds only.
- Policy from Python (`llamacpp_gguf_cuda`): `set_short_batch_mode("auto" | "native" | "mma")`, `short_batch_mode()`, `set_short_batch_decision(qtype, rows, out, in, enabled)`, `clear_short_batch_decisions()`, `has_short_batch_policy()`. Environment override `LLAMACPP_GGUF_SHORT_BATCH`, read at import (an invalid value prints one line and uses `auto`).
- `auto`: a recorded decision wins; otherwise tensor cores only on compute capability 12.0. `native`: MMVQ/MMQ as in 1.0.23. `mma`: tensor cores on any 8.0+ GPU.
- CUDA graph capture bakes the kernel chosen at capture time, so decisions must exist before capture.
- The version comes from `BASE_VERSION` in `setup.py` (it rewrites `version.py`). The candidate wheels were built with 1.0.23 there and relabelled to 1.0.24 with `wheel unpack/pack` (metadata and `version.py` only; the binaries contain no version string). `setup.py` now says 1.0.24.

## WanGP selection (`shared/kernels/gguf_short_batch.prepare`)

The vLLM engine calls `prepare(model, native)` inside `ModelRunner.capture_cudagraph`, right after the first decode warm-up and before any graph capture. The warm-up is what makes MMGP load the weights to the GPU; before it, all GGUF weights are still on the CPU. `prepare` does nothing for packages without the policy, forced modes and GPUs below 8.0.

1. Groups: the model's Q4_K/PTQ1_0 `GGUFWeightTensor` parameters (including fused projections, drafters' GGUF heads and Prism weights), resident on CUDA, `K % 256 == 0`, grouped by format and shape and de-duplicated by storage.
2. Rotation: up to 64 weights of a group, until their bytes reach four times L2. Groups that cannot reach that keep the architecture default for this run and are not cached (another model may hold enough weights of the shape). Single weights larger than L2 (output heads) are timed alone.
3. Correctness gate before timing: tensor cores on a random activation must match single-row MMVQ (the decode kernel) within 1e-2 relative L2 at FP32 output. Observed: 3-8e-4, because MMVQ stores Q8_1 activation block scales in FP16 and tensor cores keep FP32; BF16 output rounding alone raises this to 1.3-1.7e-3, so the gate never compares at the model's dtype. It does not compare against batched native either: at 6-8 rows native is llama.cpp MMQ, which is itself ~1.2e-2 away on real Qwen weights and made the first version reject correct kernels. A broken kernel differs by O(1).
4. Timing: each kernel captured once in a CUDA graph over the rotation; untimed replays (about 50 ms before the first measurement: an idle GPU runs GDDR7 at half clock, and fresh allocations are made resident on first touch under Windows); then 8 alternating-order rounds, taking the fastest round per kernel. Other GPU clients (desktop compositor, browsers) take time slices on Windows desktops and inflated single measurements up to tenfold; interference only adds time.
5. Decision per row count 2-8: the architecture default is kept unless the other kernel is at least 5% faster.
6. Log: one summary line and one line per timed shape with the speedups for rows 2-8.

Choices are cached in `~/.triton/autotune/wan2gp_gguf_short_batch.json`, keyed by GPU name and UUID, compute capability, SM count, driver (with `pynvml`), package version, CUDA version and method version (4). Cost on first load: 0.7-0.9 s for Qwen3.8 Q4_K_M (49 shape/row combinations) and 0.8 s for Bonsai (35) on the RTX 5090. Temporary memory: one activation and the captured outputs of one call per rotated weight. No weight copies.

Superseded first version, kept here as a pitfall record: a lazy per-call hook in `_try_llamacpp_cuda_linear` with a weak registry. Weak references on weights break MMGP's `swap_tensors` (and `gguf.py` swaps raw `_data` too); a `QLinearGGUF` module registry did not see the engine's weights, so every shape was timed on one L2-resident weight and decisions were noise (0.54x-2.22x). Prism PTQ1_0 linears bypass that function entirely.

## RTX 5090 results with the selection

Selected kernels (Qwen3.8 Q4_K_M): MMVQ for 2-3 rows on most shapes (tensor cores 0.85-0.93x), tensor cores from 4 rows (1.01-1.23x). Bonsai PTQ1_0: tensor cores for all shapes and rows (0.98-3.4x). End to end with 4 MTP drafts, selection versus always tensor cores: Qwen 17.2-18.0 versus 17.0-17.7 ms per cycle (within noise; different kernels change the sampled text), Bonsai 13.2-13.5 versus 13.1-13.6 ms.

## Validation of the 1.0.24 wheels (RTX 5090)

| Check | py311 (torch 2.10, CUDA 13) | Wan2R5 (torch 2.7.1, CUDA 12.8) |
| --- | --- | --- |
| SASS/PTX inventory (`scripts/inspect_wheel.py`) | 12 SASS + compute_121 PTX in `_C`, `_attention`, `_prism` | 17 SASS + compute_120 PTX |
| Release suite vs 1.0.23 with `LLAMACPP_GGUF_SHORT_BATCH=native` | passed, all formats bit-exact | passed, bit-exact |
| Same suite, default policy | 858/870 outputs bit-exact; the 12 others are Q4_K at 2/3/7/8 rows (max 1.7e-3) | not run |
| `tests/test_gguf_short_batch_linear.py` (27) | passed | passed |
| WanGP engine suites (drafters, acceptance, GGUF, Prism, Q8 attention) | 331 passed; 3 known `test_prism_gguf.py` fixture failures (also on 1.0.23) | not run |

Wheel sizes: 153.6 MB (py311, was 149.6 MB) and the Wan2R5 wheel beside it in `C:/temp/gguf_1024/dist`. Build times with 4 jobs x 2 nvcc threads in parallel: 57 and 66 minutes. Linux wheels were not built.

## Optional report (`tools/validate_gguf_short_batch.py`)

Runs both kernels on synthetic Qwen3.8-27B-sized Q4_K/PTQ1_0 weights (enough copies to exceed four times L2), selectable rows: error of each against a dense FP32 reference, the same measurement as WanGP, bandwidth and the choice `auto` would make; writes a JSON report. RTX 5090 (rows 2/5/8): Q4_K geometric-mean 0.99x (MMVQ wins at 2 rows and on 1024x5120, tensor cores at 5-8 rows), PTQ1_0 1.49x. Not required for the fast path.

## Open points

- Evidence from RTX 30/40, A100/H100 and DGX Spark reports before stating gains for those GPUs.
- Non-vLLM engines (legacy, no graphs) keep the architecture default; they do not call `prepare`.
- The WanGP overlay and `SOURCE_MANIFEST.json` in the kernel repository must be refreshed before a 1.0.24 source release.
