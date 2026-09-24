# DFlash2 rotary configuration and draft-count validation

## Configuration fix

The original Qwen3.8 DFlash2 and adapted Bonsai DFlash2 configs store the trained
rotary parameters in `rope_parameters`, including `rope_theta=10000000`. The
installed, supported Transformers 4.54.0 Qwen3 implementation reads the legacy
`rope_theta` and `rope_scaling` fields. Loading these exports directly through
`Qwen3Config.from_dict` silently selected its default theta of 10000.

`shared/prompt_enhancer/block_draft.py` now translates the checkpoint's modern
schema into those legacy fields before constructing the config. Modern values
take precedence over stale legacy fields. Legacy-only configs, including the
existing DSpark config, retain their values. The change restores the drafter's
trained positional encoding; target weights, target arithmetic, rejection
sampling, precision, cache sizing and native binaries are unchanged.

This is a model-configuration correction, not a same-output kernel optimization.
Different proposals change acceptance and sampled continuations. A corrected
drafter does not guarantee faster generation for every prompt or random seed.

## Controlled old/fixed comparison

RTX 5090, PyTorch 2.10.0+cu130, Transformers 4.54.0, uncensored Qwen3.8-27B
Q4_K_M, original BF16 DFlash2, seven draft tokens, vLLM engine, 32K cache
capacity and Q8 KV. One fixed 2048-token fiction prompt, a 90-token math prompt
and a 99-token code prompt; 256 generated tokens per call. Sampled settings:
temperature 0.6, top-p 0.95, top-k 20, min-p 0.05, repetition penalty 1.05,
seed 123. Greedy runs use top-k 1. Thinking is enabled.

Each case uses ABBA or BAAB ordering in one resident runtime, two measurements
per arm. Only the drafter's rotary frequency buffer changes, before a clean
uncached prefill. The initial harness warmed 32 output tokens per prompt/variant;
rare lazy graph shapes could still be captured during the first measurement.
Later runs record graph-capture counts and warm the entire measured trajectory.
No profiler or VRAM polling runs in the timed interval. Results below are total
tokens divided by total decode wall time,
excluding prefill. These are short prompt-level measurements, not independent
random-seed trials or a broad task benchmark.

| Case | Old theta 10K, tok/s | Correct theta 10M, tok/s | Change | Emitted / target pass, old → fixed |
| --- | ---: | ---: | ---: | ---: |
| Greedy fiction | 99.86 | 121.70 | +21.9% | 2.438 → 2.977 |
| Greedy math | 160.73 | 163.48 | +1.7% | 3.821 → 3.938 |
| Greedy code | 127.75 | 150.44 | +17.8% | 3.048 → 3.556 |
| Sampled fiction | 94.41 | 133.30 | +41.2% | 2.370 → 3.325 |
| Sampled math | 162.36 | 205.64 | +26.7% | 3.710 → 4.923 |
| Sampled code | 122.60 | 114.94 | **−6.2%** | 2.813 → 2.639 |

Peak allocated memory is approximately 21.6 GiB with either rotary setting.
The fix adds no weight/cache allocation. Allocator reservation is recorded
separately in the raw results and is not the same as live tensor memory.

The publisher's roughly 3× claim compares DFlash2 with ordinary autoregressive
decoding, not with MTP, on different hardware and workloads. It is not evidence
for a 3× improvement over WanGP's existing MTP mode. See the
[publisher's model card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).

## Draft-count comparison and cap decision

The completed interleaved DFlash2 count comparison gives these decode rates:

| Case | 4 drafts, tok/s | 5 drafts, tok/s | 7 drafts, tok/s |
| --- | ---: | ---: | ---: |
| Greedy fiction | 114.27 | 122.63 | 118.78 |
| Greedy math | 155.72 | 155.78 | 158.53 |
| Greedy code | 125.61 | 111.39 | 143.22 |
| Sampled fiction | 102.90 | 109.36 | 128.90 |
| Sampled math | 165.79 | 162.98 | 200.34 |
| Sampled code | 107.37 | 117.38 | 104.67 |

Seven drafts have a useful advantage on some workloads; five is faster on
others. A global cap at four or five would remove measured DFlash2 benefits.
This comparison limits the verified prefix while retaining the trained
eight-position noncausal draft backbone. It does not shorten that backbone.

Native MTP was measured separately, with the same inputs and sampling settings,
interleaving counts 2, 4, 5, 7 and 8 in one runtime:

| Case | 2 drafts, tok/s | 4 drafts, tok/s | 5 drafts, tok/s | 7 drafts, tok/s | 8 drafts, tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy fiction | 133.84 | 127.87 | 134.72 | 127.40 | 128.55 |
| Greedy math | 138.68 | 142.61 | 139.35 | 161.74 | 131.71 |
| Greedy code | 127.94 | 130.08 | 129.03 | 120.94 | 136.70 |
| Sampled fiction | 138.53 | 128.26 | 147.21 | 139.22 | 143.19 |
| Sampled math | 154.06 | 140.72 | 192.81 | 139.28 | 140.73 |
| Sampled code | 141.43 | 138.93 | 124.20 | 149.62 | 122.00 |

These results also do not support a universal MTP cap at four or five. Seven
helps greedy math and sampled code, whereas five helps sampled math and fiction.
Counts are maxima; confidence, page boundaries and the remaining output budget
can shorten individual proposals. Changing the count changes RNG consumption
and target verification shapes, so this table is a workload comparison, not a
bitwise-identical continuation benchmark. The two repeats use the same seed.

The user's conditional request was to cap the menu and config loading if larger
counts had no useful benefit. That condition is not met. Existing limits remain:
MTP up to eight, Qwen DFlash2 up to seven, Bonsai DFlash2 up to five. Application
Auto/defaults and the user's saved four-token MTP selection are unchanged.
At the user's subsequent request, DSpark and DFlash2 are restored as explicit
menu choices for 27B GGUF targets in Auto/vLLM engine mode. They remain unavailable
in legacy/cg and for unsupported models. Bonsai DFlash2 still clamps the menu and
loaded model to its five-token application limit. The Bonsai drafter itself has
eight positions and can produce seven drafts; the limit is a performance policy.
Both predictors can be slower than
MTP on particular workloads; they are not selected by Auto.

The three configured draft assets are published in `DeepBeepMeep/Wan2.1` as of
this validation. Public repository metadata confirms the expected weight hashes
and sizes; downloaded configuration bytes match the local test assets exactly.
Earlier engineering notes describing them as unpublished are historical.

## Bonsai DFlash2

The same paired old/fixed experiment also ran on the abliterated Bonsai 2 PTQ1
target, using its adapted BF16 drafter and five draft tokens. Other inputs and
sampling settings match the Q4 experiment.

| Case | Old theta 10K, tok/s | Correct theta 10M, tok/s |
| --- | ---: | ---: |
| Greedy fiction | 85.70 | 104.74 |
| Greedy math | 116.03 | 137.57 |
| Greedy code | 87.91 | 92.22 |
| Sampled fiction | 88.32 | 109.43 |
| Sampled math | 115.94 | 117.20 |
| Sampled code | 88.17 | 92.37 |

Sampled fiction improves by 23.9%; sampled math by 1.1% and code by 4.8% in this
short fixed-seed experiment. These are improvements over the earlier DFlash2
configuration, not over MTP or ordinary target-only generation.

### Comparison with Bonsai MTP

On the same three prompts, native two-draft MTP produced 134.66, 142.70 and
125.76 tok/s in sampled fiction, math and code respectively. Ordinary decode
produced 120.22, 120.29 and 120.53 tok/s. These are separate-process controls,
not interleaved method comparisons. Correcting DFlash2 does **not** establish
a benefit over either control. Auto remains native MTP.

The interleaved count experiment (`bonsai-dflash-counts`) confirms that simply
raising the Bonsai limit to seven is harmful on these inputs:

| Case | 4 drafts, tok/s | 5 drafts, tok/s | 7 drafts, tok/s |
| --- | ---: | ---: | ---: |
| Greedy fiction | 111.83 | 102.97 | 74.71 |
| Greedy math | 134.39 | 134.00 | 101.76 |
| Greedy code | 92.54 | 88.32 | 62.09 |
| Sampled fiction | 113.23 | 107.02 | 72.34 |
| Sampled math | 150.90 | 130.50 | 110.58 |
| Sampled code | 130.33 | 110.67 | 87.84 |

Two measurements per count/case use the same seed. Graph capacity is held at
seven drafts. Different counts can change target arithmetic shape and sampled
continuations. The five-draft limit is retained; it is not a claim that five is
the best count for every prompt.

A second count experiment with the acceptance graph cache retaining eight
shapes tested shorter blocks (`bonsai-dflash-short-counts`). All 48 timed calls
recorded zero graph captures:

| Case | 2 drafts, tok/s | 3 drafts, tok/s | 4 drafts, tok/s | 5 drafts, tok/s |
| --- | ---: | ---: | ---: | ---: |
| Greedy fiction | 106.82 | 106.15 | 113.47 | 106.27 |
| Greedy math | 115.29 | 127.43 | 152.05 | 145.02 |
| Greedy code | 100.03 | 94.76 | 102.63 | 99.00 |
| Sampled fiction | 91.77 | 90.97 | 106.46 | 102.37 |
| Sampled math | 113.71 | 115.97 | 135.83 | 117.55 |
| Sampled code | 95.05 | 100.46 | 111.41 | 92.60 |

Four drafts won these cases. Two/three did not rescue DFlash2 performance despite
cheaper verification; they require more rounds of its fixed eight-position
draft backbone. This does not establish a universal optimal count or a general
DFlash2 speedup over MTP.

### Verification bottleneck and rejected experiments

`bonsai-profile` is an instrumented, short-context diagnostic (103 input tokens;
its text corpus argument does not reproduce the 2048-token fixture). It is not
used for speed ratios. Twelve sparse stage samples gave median GPU intervals
of 5.20 ms for drafting, 20.40 ms for target verification, 0.80 ms for its output
head and 0.61 ms for acceptance. The separate profiler trace attributes 66.9%
of self CUDA kernel time to six-row PTQ1 matrix-vector operations. Optimizing
only the draft sampler cannot remove that dominant target cost.

Two isolated matrix probes used real checkpoint weights and CUDA graph timing:

- Splitting four-to-eight-row PTQ1 calls into one-to-three-row calls did not
  provide a useful gain. Some shapes also changed reduction results. No dispatch
  change was enabled.
- Losslessly expanding PTQ1 to Q8_0 preserved every decoded FP32 weight exactly
  for all tested matrices. Six-row kernel speedups ranged from 1.23x to 1.73x,
  but packed weight memory grew by 4.86x and CUDA outputs were not generally
  bitwise equal (different FP32 reduction order). No conversion or runtime
  option was enabled. These are matrix-call results, not model speedups.

The probes are `tools/experiments/benchmark_bonsai_verify_dispatch.py` and
`tools/experiments/benchmark_bonsai_repack.py`. The original PTQ1 checkpoint,
target precision and application weight storage remain unchanged.

## DSpark optimization and rejected sampling changes

DSpark's Markov head was launched through separate eager operations, with a
host read of confidence followed by a host read of every proposed token.
The retained optimization captures each greedy Markov head with its confidence
test, keeps proposed IDs on the GPU between positions, and uses the existing
GPU block acceptance. Confidence still stops the chain at the same position;
it does not calculate an entire seven-token suffix after a failed confidence
test. This is restricted to the existing vLLM eligibility rules and greedy
decoding. Sampled DSpark keeps its original proposal and rejection sampler.

The shared acceptance graph cache now holds eight lengths with LRU eviction,
covering the supported one-to-eight-draft shapes. A changed-input test cycles
all eight lengths after warmup and forbids another graph capture. This avoids
the four-entry cache repeatedly evicting shapes reached by confidence exits,
page boundaries and the final shortened output block. It does not alter
acceptance arithmetic.

This cache change is not memory-neutral when more than four shapes are used.
An isolated 248,320-vocabulary BF16-logit probe, exercising lengths one through
eight, measured 25.5 MiB more live tensors and 122 MiB (greedy) / 130 MiB
(sampled) more allocator reservation with eight entries. These are cache-only
figures, not an isolated full-model peak measurement. Configurations using four
or fewer cache keys do not populate the additional entries. The benchmark
bundle contains both allocated and reserved figures; these must not be
confused with the memory-neutral RoPE configuration correction.

The final paired experiment (`dspark-final-q4`, `dspark-final-bonsai`) warms
the full 256-token trajectory for each arm and records zero graph captures in
every timed call. Both arms have the eight-entry acceptance cache; the rates
isolate the proposal/acceptance dispatch change, not the cache-size change.
All four continuations within each greedy case have identical token IDs.

| Target and greedy case | Original DSpark, tok/s | Graph DSpark, tok/s | Change |
| --- | ---: | ---: | ---: |
| Q4 fiction | 135.62 | 138.70 | +2.3% |
| Q4 math | 235.07 | 248.50 | +5.7% |
| Q4 code | 150.13 | 155.45 | +3.5% |
| Bonsai fiction | 99.96 | 102.34 | +2.4% |
| Bonsai math | 118.86 | 121.51 | +2.2% |
| Bonsai code | 93.63 | 95.69 | +2.2% |

An earlier independent paired run (`dspark-single-*`) measured the same
greedy computation at +1.7/+7.2/+4.0% on Q4 and +3.8/+1.9/+2.3% on Bonsai,
also with exact token agreement and no timed captures. Absolute throughput
changed between processes; only within-process ratios are meaningful. These
small three-prompt results do not establish a universal gain or a DSpark
advantage over MTP.

The final code compares confidence in FP64 to preserve the original Python
float comparison, including thresholds immediately above a representable
confidence. Only this scalar comparison changes dtype; target and draft model
arithmetic retain their original precision. A boundary regression test covers
this case, and the final paired results above include the change.

Sampled optimizations were rejected. The first version moved both proposals
and rejection sampling onto the GPU but changed RNG consumption, producing
different continuations and a 2.6% Q4-fiction slowdown despite larger gains in
other cases. A second version preserved the original RNG sequence and exact
continuations, but Q4 fiction/code still slowed by 0.8%/1.2%. Neither sampled
variant is enabled. Earlier whole-block and three-position-chunk prototypes
also overcomputed positions past the confidence exit and are not enabled.

## Validation

The final focused checks passed 270 distinct cases across the suite and its
targeted reruns. They cover draft configuration, graph reuse and confidence
boundaries, acceptance distributions, grammar/thinking limits, speculative
state, menu/save/load limits and decoder-mode isolation. The required product
documentation-scope test has one unrelated existing failure: unmodified
`docs/AUTHENTICATION.md` has no scope footer in HEAD. The Bonsai guide retains
its scope footer.

Real shared-loader runs passed on Q4, Q3, Q2 and Bonsai for both DFlash2 and
DSpark, with CPU and CUDA default devices, cancellation/recovery, repeated
greedy and sampled calls, and exact greedy snapshot/rewind restoration.
DSpark's graph/reference comparisons also match at a forced thinking-budget
boundary. The four-backend DSpark runtime checks preceded the scalar FP64
confidence refinement; the boundary unit test and final real Q4/Bonsai paired
benchmarks ran afterwards. DFlash2's Q4 vision test also identified the fixture
image correctly and restored the exact conversation continuation.

## Evidence and reproduction

### Published CUDA wheel compatibility

The latest published CUDA release, `gguf-v1.0.22`, was checked independently
of the local experimental binary. The downloaded Windows Python 3.11 / Torch
2.10 / CUDA 13.0 wheel matches GitHub's SHA256
`a8bfc4e1655be2b7f95215d2f76c5679a1407f4f506ce6964c0811ab83af4434`.
It passed 59 compatibility/mode-isolation tests and real Q4/DSpark and
Bonsai/DFlash2 repeated generation, cancellation recovery, CPU/CUDA defaults,
and snapshot/rewind checks on the RTX 5090. The installed environment was not
modified. [Raw compatibility evidence](benchmarks/published_cuda_compatibility_2026-09-22.json).

Current WanGP's Python and Triton changes work with this published wheel,
including the Bonsai layout, GDN/direct-layout and RoPE/cache kernels, drafter
RoPE correction and speculative scheduling. The published wrapper has no
`supports_linear_fusions` capability or new `linear` keyword arguments, so
native fused SiLU/input quantization and typed MMVQ output stores are skipped
through the existing capability check. Enabling those compiled changes needs
a new wheel. Compatibility does not transfer the combined experimental-binary
speed measurements to the published binary.

### Raw investigation records

The durable [evidence bundle](benchmarks/dflash2_dspark_2026-09-22.json) includes
raw records, complete generated IDs, the fixed fiction input, source/binary
identities, runtime checks and rejected prototypes. Sampled prototype results
are explicitly separated from the retained greedy optimization. No native
binary was rebuilt or installed during this investigation.

Raw session files are under `D:/AMD/dflash-20260922`. The valid paired experiment
is `paired-rope/results.json`; native controls are `controls/mtp/results.json`
and `controls/disabled/results.json`. Records include exact generated token IDs,
prompt hashes, source/binary hashes, prefill/decode durations, speculative
statistics and allocated/reserved memory. Do not combine separate fixed-only
runs with the paired experiment to select favourable timings.

The earlier `comparison` experiment overlapped another GPU workload and is
explicitly invalid for timing. `clean-comparison`, `controlled`, `controlled-v2`
and the old-loader arm under `rope-fix` stopped at a process guard. The first
reference diagnostic shared the tested model's rotary module and therefore
could not detect this configuration error; it is superseded by the independent
reference audit, which constructs its own rotary module from the checkpoint
schema. Instrumented profiles establish cost attribution, not speedup ratios.

`tools/benchmark_qwen_block_prediction.py --compare-rope` reproduces the paired
rotary experiment. `--compare-drafts 4 5 7` interleaves count limits in one
resident runtime. Draft-count comparisons retain graph/cache capacity for the
largest count, so they measure execution speed rather than per-count residency.
The GPU-job guard excludes the user's existing server, which must remain idle;
it checks before and after each trial, not continuously inside timing.

`tools/audit_dflash2_reference.py` compares real draft proposals against the
[pinned publisher reference](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py).
Supply that source file and its MIT license locally; the audit does not download
or vendor it. `tools/test_qwen_dflash2.py` exercises the real application loader,
generation, cancellation, snapshot/rewind and optional image inspection.
