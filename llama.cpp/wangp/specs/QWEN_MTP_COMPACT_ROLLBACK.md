# Compact GDN Rollback For Qwen MTP

**Status: production integration is unimplemented.** The isolated experiment in
`tools/experiments/qwen_mtp_compact_rollback.py` is a correctness prototype, not
an enabled optimization. A bounded single-layer GPU check has passed (see the
experiment record below); this does not establish full model correctness or a
performance benefit. It must not be imported by production code.

## Purpose And Limits

Replace the large recurrent-state snapshot for every committable speculative
prefix with one initial-state backup and small per-token replay records. Keep
the existing convolution snapshots initially. On partial acceptance, replay the
accepted prefix's GDN recurrence from the original state. Full acceptance needs
no replay: verification already wrote the correct final state to the live buffer.

The immediate objective is lower speculative-state storage at the **same draft
depth**, with bitwise-identical outputs and committed states. A higher draft depth
is a separate experiment. More drafts can cost more than their additional
acceptance saves; neither higher throughput nor a twofold speedup is promised.

The first integration, if approved by the evidence, would cover only the current
raw GDN CUDA path in `vllm` on a GPU with demonstrated exact parity. Legacy, `cg`,
AMD, non-GDN models, and unsupported layouts/dtypes retain their established paths.
Do not change sampling rules,
draft defaults, precision, MMGP ownership, or maximum KV capacity as part of this
change.

## Existing Boundaries

The relevant source is:

- `shared/kernels/qwen_gdn.py::_recurrent_raw_kernel`: grouped recurrent state,
  raw post-convolution Q/K/V, direct/tiled head layouts, FP32 recurrent carry, and
  optional rounded prefix snapshots.
- `shared/llm_engines/nanovllm/models/qwen3_5.py::prepare_speculative_state`,
  `_forward_linear_attention`, `commit_speculative_state`, and
  `release_sequence_state`: allocations, forward ownership, and lifetime.
- `shared/llm_engines/nanovllm/engine/model_runner.py::_prepare_target_speculative_state`,
  `_commit_speculative_target_state`, `_run_native_mtp`, `capture_cudagraph`,
  `_get_graph_capture_signature`, and `reset_runtime_state`: prefix selection,
  cached source lists, graph pools, and invalidation.

Today the runner prepares a list of live convolution/recurrent destinations and
a source list for every partial prefix. Commit uses `torch._foreach_copy_`.
The observed implementation is about two copy kernels and 71 microseconds per
rejection in the measured workload. A Python loop launching 48 replay kernels is
not an acceptable assumed improvement over that baseline.

`processed_tokens` is `len(emitted)`, not the count of accepted draft tokens. Even
when the first draft is rejected, prefix one represents the target input token
preceding the draft and must be committed. A fully accepted verification of
`T = draft_count + 1` tokens is a no-op at the commit boundary.

## Exact Payload For The 27B Configuration

The inspected 27B configuration has 64 blocks, with full attention every fourth
block: 48 GDN blocks. Each GDN block has 16 key heads, 48 value heads, key/value
dimensions 128, and a four-tap convolution over 10,240 channels. Runtime states
and post-convolution K/V are BF16 in the measured configuration.

For batch `B`, maximum verification length `Tmax = D + 1`, and one GDN layer:

| Payload | Shape | Dtype | Purpose |
| --- | --- | --- | --- |
| Initial recurrent backup | `[B, 48, 128, 128]` | BF16 | Exact state before verification |
| Raw post-convolution keys | `[B, Tmax, 16, 128]` | BF16 | Repeat the same key normalization |
| Raw post-convolution values | `[B, Tmax, 48, 128]` | BF16 | Values in canonical grouped head order |
| Computed decay argument `g` | `[B, Tmax, 48]` | FP32 | Exact argument passed to the existing `exp` helper |
| Rounded beta | `[B, Tmax, 48]` | BF16 | Existing sigmoid result after compute-dtype rounding |
| Existing convolution snapshots | `[D, B, 10240, 4]` | BF16 | Restore convolution history without another algorithm |

Queries are not needed to advance the recurrent state. Recording K/V plus `g`
and rounded beta is sufficient; recording Q or raw `a` is unnecessary. The logs
are independent, bounded allocations, not retained views into large projection
or graph tensors. Using computed `g` and beta also prevents rollback from reading
SSM parameter weights after MMGP may have moved them.

At `B=1`, across all 48 layers, the initial backup is exactly 72 MiB. Each
convolution prefix is 3.75 MiB. Each logged verification token uses 800,256 bytes
(0.76318359375 MiB): 196,608 key bytes, 589,824 value bytes, 9,216 decay bytes,
and 4,608 beta bytes. This includes a log slot for the final token for a simple,
uniform prototype; omitting unused final-token logs is a possible later refinement.

| Draft depth D | Verify tokens | Existing snapshots | Proposed backup/logs + unchanged conv snapshots | Saving at same depth |
| --- | --- | --- | --- | --- |
| 2 | 3 | 151.500 MiB | 81.790 MiB | 69.710 MiB |
| 3 | 4 | 227.250 MiB | 86.303 MiB | 140.947 MiB |
| 4 | 5 | 303.000 MiB | 90.816 MiB | 212.184 MiB |
| 6 | 7 | 454.500 MiB | 99.842 MiB | 354.658 MiB |
| 7 | 8 | 530.250 MiB | 104.355 MiB | 425.895 MiB |
| 8 | 9 | 606.000 MiB | 108.869 MiB | 497.131 MiB |

These figures exclude unchanged live states, model/KV storage, graph pools,
sampling tensors, and allocator fragmentation. Compact depth eight is 42.631 MiB
below the **old depth-two snapshot budget**, but this does not prove that total
peak VRAM is within budget. Retained draft probability vectors alone add about
5.68 MiB when moving from two to eight drafts at vocabulary size 248,320 and
FP32 probabilities. Graph workspaces, logits/filter temporaries, and speculative
attention scratch also change.

Configured maximum draft depth one must retain the established snapshot path:
there is only one recurrent snapshot already, so an initial backup plus logs
would add 1.526 MiB instead of saving storage. A workspace configured for depth
two or higher may still process a shortened verification near a page boundary.

The inspected depth-two capture had two private pools of 22 MiB each. Verify
lengths share one target pool; MTP refresh lengths share the separate MTP pool.
It is incorrect to multiply all temporary workspace sizes by every graph length,
or to assume that all graph outputs and persistent buffers share storage. Measure
both active allocations and pool reservations. Paged-attention splitting changes
with query rows, so its partial-output scratch need not scale smoothly with D.

## FP32-Carry Replay Is Required

The existing verification kernel loads the initial BF16 state into FP32, carries
that FP32 state through the **entire token loop**, and rounds only its snapshot
stores and final live-state store. A snapshot store does not round the internal
carry used for the next token.

For an accepted prefix of length `p`, replay must therefore:

1. Load the original initial-state backup into FP32.
2. Execute all `p` updates in one kernel invocation, carrying FP32 between them.
3. Normalize raw K with the same reduction shape, order, epsilon `1e-6`, and
   square root/division as the existing raw kernel.
4. Use the same imported `fla.ops.utils.op.exp` helper and the logged FP32 `g`.
5. Use beta after the same BF16 sigmoid rounding as the existing kernel.
6. Apply the same operations in the same order:
   `state *= exp(g)`, then
   `v = beta * (v - sum(state * k, key_axis))`, then
   `state += k * v`.
7. Store once into the live BF16 state.

Do not replay one token per kernel, restart from a rounded prefix snapshot,
normalize/log keys in reduced precision, replace the exponential implementation,
change reduction grouping, or invert the recurrence. Those are different
arithmetic paths. The isolated kernel preserves the existing `BK`, `BV=8`,
`num_warps=1`, and `num_stages=3` launch choices.

Identical source expressions do not prove identical generated arithmetic.
Removing Q/output work and adding log stores can change code generation. Every
partial prefix must match the old snapshot **bit for bit**, including signed-zero
bits, before enabling this path. A nonzero-tolerance comparison is insufficient
for this proposal. Inspect generated resources for spills as well as correctness.

Head layout matters: the live recurrent state is grouped, while value and gate
projections can be tiled and SSM parameters can have their own permutation. The
verification log stores K once per key head and V/g/beta by canonical grouped
value head; output retains the current value-head layout. Only one program writes
each K/g/beta log element. Identical concurrent stores are still data races.

## Graph, Hook, And Cache Lifetime

Allocate backups/logs before any warmup or capture. Allocate for `Tmax` and retain
those physical strides when a shorter T graph runs. Never resize or lazily create
payload storage during capture. Replaying verification overwrites the backup and
all active log slots, so it must be followed by its commit before another verify
call on the same workspace. Existing single-sequence ordering supplies this
condition; concurrent sequences need distinct slots.

Do not derive valid verification length from Python attributes assigned inside a
captured forward. Capture runs different lengths and the last captured length is
not necessarily the length later replayed. The runner already knows actual T and
p; pass those explicitly to the commit choice. Unused log slots must never be read.

Keep existing forward/module/MMGP hooks. Do not replace forwards or keep SSM
weights alive solely for replay. Logging `g` and beta makes replay state-only.
The first integration would change just three boundaries:

1. Layer speculative-state preparation and raw verification: allocate the compact
   workspace and record into it instead of recurrent prefix snapshots.
2. Runner commit: retain convolution foreach-copy and select recurrent replay;
   eliminate old recurrent snapshot references from every cached source list.
3. Reset/capture lifecycle: release compact storage, commit graphs, and any pointer
   table with their owning runtime state; invalidate graphs before storage changes.

Current graph signatures sample weight/KV storage and draft depth, but runtime
state attributes are not automatically tracked like parameters. Compact mode,
dtype, shape/capacity, and workspace/pointer-table generations must either enter
the graph signature or force the existing clear-and-reprepare path. A stable
pointer table containing stale pointees is still invalid. Weight movement, KV
growth, model switches, cancellation/reset, and changed maximum draft depth must
invalidate all dependent captures. Retain normal synchronization before freeing
storage used by in-flight GPU work. Capture warmup mutates states/logs; restore
live request state after all warmup/capture work as the established path does.

The correctness prototype is one layer. If that passes, the practical next design
is one replay launch over all 48 uniform GDN layers using device pointer tables
and a layer program index. The table must be built outside capture, own stable
references, and be tied to the same invalidation generation. Graphing a Python
loop can remove CPU launch overhead but still executes 48 GPU kernels; it is not
equivalent to a batched replay kernel. Batched replay may itself be slower than
the current snapshot copies and must be measured.

## Independent Depth And Branching Experiments

First compare old and compact storage at identical draft depth, inputs, seeds,
acceptance/rejection decisions, and model/cache settings. Only after parity and
resource checks pass should deeper drafts be tested separately, without changing
the default depth automatically.

The native GGUF path has relevant boundaries: generic MMVQ changes from four to
two warps above four verification rows, and `MMVQ_MAX_BATCH_SIZE=8` makes a
nine-row verification switch to MMQ. Thus draft depth eight changes more than
state storage. Measure depths three/four/six/seven before claiming that eight is
better. Compare exact target results at each depth against that depth's existing
kernel path; changing matrix kernels can introduce their own rounding differences.

Branching proposals are a second, substantially larger project. Existing target
attention assumes a linear causal chain; siblings need ancestor-aware attention
masking and KV indexing. GDN recurrence needs each branch's parent state rather
than a flat token loop. Accepted-path hidden states, KV slots, MTP refresh, and
sampling probability correction must all follow the selected branch. Recomputing
each root-to-node recurrence avoids a state per tree node but increases work;
keeping a register stack can spill. Existing block-draft support does not supply
these tree semantics. Do not mix a tree rewrite into compact linear rollback.

## Acceptance Tests Before Production Integration

- Isolated bitwise comparisons against current `recurrent_raw_gates`, using actual
  27B head dimensions, nonzero initial state, all T from two through nine, every
  partial prefix, and full acceptance. Cover grouped/tiled/interleaved layouts,
  strided projection/gate views, multiple batches, and finite edge gate values.
- Compare verification output, live final state, initial backup, and every replayed
  prefix. Reuse a workspace with changing T/input/state to expose stale logs.
  Replay from the original backup twice with different prefixes to prove that the
  first commit did not destroy it. Include graph replay with modified inputs and
  stable storage; assert exact bit patterns and finite results.
- Inspect compiler register/shared/local-memory use for both kernels. No hidden
  state-sized temporary, full dequantized weight, or graph-retained activation may
  replace the removed snapshots. Run an available race checker for log writers.
- After the isolated test passes, test batched replay independently against the
  single-layer reference before runner integration. Verify pointer invalidation.
- End-to-end same-depth old/compact A/B/A runs: greedy and sampled MTP, all rejection
  prefixes, acceptance counts, output token IDs, maximum allocator and device
  memory, prefill/decode throughput, repeated calls, graph reuse/rebuild, KV growth,
  cancellation then successful reuse, and MMGP offload/reload. Cover ordinary
  no-MTP vllm plus unchanged cg/legacy behavior.
- Time without a concurrent `mem_get_info` polling thread or external GPU monitor;
  memory instrumentation is a separate run. Keep prompt/seed/repetition order
  paired. Record generation work and aggregate tokens/time, not just unpaired
  medians. Do not attribute unexplained run variance to one cause without evidence.
- Measure high acceptance and frequent rejection workloads: saving verification
  writes can be outweighed by prefix replay. Enable nothing that shows a reproducible
  speed regression or increases the agreed total peak VRAM. If it only saves memory,
  describe it as a memory experiment and retain the established production path.

The prototype tool is `tools/experiments/validate_qwen_mtp_compact_rollback.py`.
It is intentionally not part of normal startup or production routing. CPU syntax
checks do not constitute the GPU or integration validation above.

## Experiment Record, 2026-09-22

The initial bounded test used the PyTorch 2.10 `py311` environment and RTX 5090:
BF16, one layer with the actual 27B head dimensions, T=3, grouped/tiled/interleaved
layouts, strided projection and gate inputs, nonzero initial state, and finite
gate edge cases. Verification output, final state, both partial prefixes, and
initial backup matched bit for bit. Full acceptance was a no-op. Reusing capacity
nine for shorter verification and CUDA graph replay after modifying inputs and
initial state also passed. All compiled variants reported zero spills.

The result is recorded in
`D:/AMD/cuda-fusions-20260922/compact-rollback-single.json`. Individual graph
iterations lasted roughly 13-18 microseconds, with enough spread to make the
observed few-percent timing differences inconclusive; they are not a speed claim.
The test recorded 1,622,880 compact rollback bytes per layer at capacity three,
versus 3,145,728 recurrent snapshot bytes. Convolution snapshots were not changed.

A separate existing-depth sweep did not establish a consistent throughput gain
from deeper drafts and sometimes changed greedy token sequences. That is an
additional reason to leave draft defaults unchanged and keep this prototype
unintegrated. A batched 48-layer pointer-table experiment is included in the tool
to distinguish replay arithmetic cost from 48 separate replay launches; its
result must be recorded separately before drawing conclusions.

The first 48-layer pointer-table attempt **failed** the exactness gate before any
timing: layer zero, prefix one differed in 8 of 786,432 BF16 state elements, with
maximum absolute difference `6.103515625e-05`. The failing compiled replay used
123 registers and 128 bytes of shared memory, versus 76 registers and 256 bytes
for the corresponding direct-pointer replay. It had no spills. This is evidence
that matching mathematical expressions and launch dimensions alone is insufficient.
The failed result is preserved in
`D:/AMD/cuda-fusions-20260922/compact-rollback-batched.json`; that variant is rejected.

A single bounded correction asserts 16-byte alignment of every actual pointee
at pointer-table preparation, then supplies that same alignment to the loaded
Triton pointers. Direct tensor arguments already provide alignment metadata;
opaque addresses loaded from a table do not. Changed reduction layout is a
plausible explanation for the failure, but the annotation is not proof of a fix.
The correction was tested with the same strict gate, without any tolerance change.

The aligned retry passed every state/output comparison across 48 independent
27B-shaped layers at T=3, for prefix one, prefix two, and full acceptance. This
included CUDA graph replay after changing inputs and initial states. The tool also
rechecked single-layer batches one and two. The aligned batched replay compiled
to 79-80 registers, 256 bytes of shared memory, and zero spills. These are bounded
synthetic tests, not model-level validation. The result is preserved in
`D:/AMD/cuda-fusions-20260922/compact-rollback-batched-aligned.json`.

The graph timing below includes 48 verification kernels, commit, and the same
foreach reset of initial states in every variant. Convolution and the rest of
the model are excluded. Baseline is current snapshots plus foreach-copy commit;
compact alternatives differ only in using one batched replay or 48 replay launches.

| Committed prefix / T=3 | Old before (us) | Old after (us) | Compact batched (us) | Compact 48 launches (us) | Old / compact batched |
| --- | --- | --- | --- | --- | --- |
| 1 | 722.72 | 723.36 | 708.11 | 766.97 | 1.021x |
| 2 | 720.26 | 668.76 | 705.00 | 843.94 | 0.985x |
| 3, full acceptance | 539.09 | 546.30 | 526.24 | 522.10 | 1.031x |

These small, mixed changes and baseline drift do **not** establish a no-regression
throughput gain. The 48-launch variant clearly loses on partial acceptance in
this test. Full-acceptance variants launch no replay, so differences between those
two compact columns are run variation, not an algorithmic difference.

Allocated recurrent rollback payload across the 48 layers is 150,994,944 bytes
(144 MiB) for old snapshots and 77,898,240 bytes (74.28955078125 MiB) for compact
backup/logs. The pointer table adds 2,304 bytes. Including the unchanged two
convolution snapshots, the corresponding payloads are 151.5 MiB and
81.791748046875 MiB. Capture added zero active tensor bytes for the compact graphs
in this bounded test; pool-reservation deltas occasionally decreased during the
baseline capture and must not be interpreted as per-variant peak memory. This
was not an end-to-end peak-VRAM measurement.

**Decision: leave the experiment unintegrated.** It demonstrates a possible
state-storage reduction and a viable bounded exact replay, but it neither
justifies increasing default draft depth nor establishes a throughput benefit
without regressions. There is no production behavior change from these files.
