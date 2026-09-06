# GGUF 1.0.21 release validation

Validated on 6 September 2026 using one RTX5090, Windows and Ubuntu 22.04 under WSL. Other NVIDIA architectures were compiled and their fatbins inspected; no other GPU was available for hardware tests.

## Compiled implementation

The wheel contains the latest packed MMVQ/MMQ implementation, typed activation quantization and standard Q8/dense attention extensions. SM120 async prefill and grouped decode/verification are now ahead-of-time GPU binaries loaded by a native C++ CUDA-driver launcher. The async kernels use `cp.async`; they are not TMA kernels.

`csrc/sm120_async.py` is the build-only Gluon source. `scripts/compile_sm120.py` produces four cubins for each CUDA major (FP16/BF16, prefill/grouped). Head dimension is fixed at 256; query count, head count, cache pages, context lengths and split count are runtime arguments. No runtime shape compilation, Triton import or scratch allocation is required by the native async module. WanGP still uses its existing shared Triton reduction and other shared kernels in vllm mode.

The same high/low tensor-core arithmetic is retained. The compiled implementation matches the shared async implementation bit-for-bit in the integration comparisons. There is no new weight/KV quantization or sampling change. The launcher writes the existing partial buffers, retains at most four cached kernel handles per device, uses the current PyTorch CUDA stream and supports CUDA-graph replay with changing lengths and page tables. No extra persistent tensor cache was added.

The source overlay retains architecture-independent kernels and the existing legacy/cg/vllm separation. New wheels are selected automatically on SM120; the pre-existing older-wheel dispatch remains available. No new inference fallback was introduced.

## Validation

Every final wheel passed:

- 280 nonzero packed-linear configurations: 18 qtypes, FP16/BF16, 1/2/4/8/17/64/129 input rows, plus real Qwen Q4_K/Q6_K weight slices; changed inputs across repeated CUDA-graph replays.
- Supported Q4_K/Q6_K embedding checks.
- 12 standard Q8/dense attention cases and 3 activation-padding cases.
- 20 compiled SM120 attention configurations, including prefill, grouped queries, shuffled pages, changed lengths and graph replay. These standalone tests do not import Triton.

A further 24 WanGP tests passed for compiled-path equality, no runtime Triton prefill launch, the fixed binary inventory and backend isolation.

`*_binaries.json` records the actual native SASS/PTX inventory and binary hashes. CUDA 13.1 builds contain 12 SASS architectures (SM75 through the supported SM121 variants); CUDA 12.8 builds contain 17 (including SM50/52/53/60/61/62/70/72). Each includes PTX for the highest toolkit target. Both Linux extensions use only package-relative runtime search paths and link the correct CUDA 12 or 13 libraries.

## Real Qwen Q4 checks

The real Qwen3.8-27B Q4_K_M checkpoint ran with Q8 KV, 32K context capacity and four-token MTP. Prompts were repeated to exercise graph reuse; all runs had `sync_delta=0`. Generated text was inspected and was coherent. These short runs validate execution and cache alignment, not long-form story quality.

| Stack | Contexts | Requested output per call | Result |
|---|---|---|---|
| Windows / PyTorch 2.10 | 2,048 and 20,000 | 128 | Passed, two calls per context |
| Windows / PyTorch 2.7 | 2,048 | 32 | Passed twice with CUDA_LAUNCH_BLOCKING=1 |
| Linux / PyTorch 2.10 | 2,048 and 20,000 | 64 | Passed, two calls per context |
| Linux / PyTorch 2.7 | 2,048 and 20,000 | 64 | Passed, two calls per context |

All four tests above used vllm and the compiled SM120 module. Linux real-model checks preceded the relocation-only relink; the complete kernel suite was repeated on the final relinked wheels. Windows 2.7 vllm validation used an isolated Triton 3.3.1.post21 installation because the existing environment's conflicting packages imported Triton 3.2. The installed environment was not changed. A separate real-model CG check with the existing environment also passed twice under CUDA_LAUNCH_BLOCKING=1; its peak PyTorch allocation was 17.68 GiB.

Two direct compatibility fixes are included in the WanGP overlay: make the startup probe's `tl` import visible to older Triton name resolution, and explicitly pass the CUDA output device to the two in-place MTP `arange` calls on PyTorch 2.7. An SM120 compiler check prevents Triton older than 3.3 from passing the small startup probe and then aborting on reductions. The existing engine resolver handles unavailable Triton. Blackwell's Triton minimum and PyTorch/Triton version pairing are documented by the [Triton Windows maintainers](https://github.com/triton-lang/triton-windows#1-gpu).

## Timing and memory observations

These were release-validation workloads, not controlled before/after performance comparisons. CPU compilation overlapped some runs, output lengths differ, and framework/Triton versions differ between stacks.

At 20K context, Windows/PyTorch 2.10 measured 121-128 decoded tokens/s and 3,025-3,073 prefill tokens/s. Linux/PyTorch 2.10 measured 129-138 decoded tokens/s and 3,006-3,021 prefill tokens/s. Linux/PyTorch 2.7 measured 85-88 decoded tokens/s and 1,307-1,392 prefill tokens/s. The Windows 2.7 blocking check is not a normal-speed benchmark.

Peak PyTorch allocation stayed below 17.64 GiB across these real-model runs. This excludes the CUDA driver, loaded code and other processes; it is not a whole-board VRAM measurement or a controlled claim of VRAM reduction. Reserved PyTorch memory stayed below 18.05 GiB. The detailed per-call measurements, acceptance statistics, hashes and checks are in `summary.json` and the adjacent validation/build inventories.

## Reproduction and packaging

Use the matching Python/PyTorch environment and `scripts/build_release.py`; see the native README for the commands. Rebuild the optional SM120 artifacts with Triton 3.6.0 and CUDA 12.8/13.1 `ptxas`. The cubins are shared across Windows/Linux; the native launcher is compiled for each Python/PyTorch/platform ABI.

The build helper detects changed source contents before preserving timestamps, preventing stale incremental binding objects. Linux setup removes Conda's injected absolute runtime library path. Source archives include the complete vendored implementation, licenses, kernel sources, compiler scripts, tests, precompiled SM120 artifacts and WanGP overlay. The wheels do not contain model checkpoints or user sessions.
