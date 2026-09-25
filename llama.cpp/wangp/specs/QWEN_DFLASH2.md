# Qwen3.8 DFlash2 — 2026-09-20

Update 2026-09-23: this BF16 preparation and publication status is historical. The active INT8 ConvRot drafter is now in `DeepBeepMeep/Wan2.1/Qwen3_8_27B_Uncensored/`; the old remote BF16 folder was retired. See [the ConvRot migration and validation](BLOCK_DRAFT_CONVROT_20260923.md).

Follow-up 2026-09-22: the original measurements below used an incorrect draft
RoPE configuration on Transformers 4.54.0. See
[the configuration correction and controlled remeasurements](DFLASH2_ROPE_AND_DRAFT_COUNTS.md).
This page retains the historical asset and validation record.

Qwen3.8 Q2, IQ3_S and Q4 use the original Qwen DFlash2 drafter. Bonsai PTQ1 continues to use the adapted ProCreations drafter. Asset download selection uses the selected target; actual model loading uses Prism metadata to distinguish Bonsai. Both paths reuse the existing BlockDraft model, CUDA graphs, candidate selector and target verification. No kernel or MMGP changes are needed.

The Qwen limit is seven draft tokens; Bonsai retains five. UI choices and loader limits follow the selected asset. The loader clamps saved counts when changing to a lower-limit target. Auto, native MTP and DSpark behavior remain unchanged. Shared config schema/version remains 1.23: there is no new key or migration, and existing DFlash2 selections retain their saved count.

## Asset

- Source: https://huggingface.co/incoai/Qwen3.8-27B-DFlash2
- Revision: `015e795645c74b1a0eeef3b570031fb62e769bc5`
- Original `model.safetensors`, renamed without conversion to `Qwen3_8_27B_DFlash2_bf16.safetensors`.
- SHA256: `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
- Size: 3,848,817,896 bytes; 81 BF16 tensors. SHA256, exact keys, shapes and dtypes verified by `tools/prepare_qwen38_dflash2.py`.
- License: Apache-2.0; upstream README and provenance retained beside the checkpoint.
- Prepared locally under `D:/ML/WanGP/ckpts/Qwen3_8_27B_DFlash2`, an existing configured checkpoint root. E: lacked space; the failed partial weight download there was removed.
- Runtime download declaration follows the existing `DeepBeepMeep/Wan2.1/Qwen3_8_27B_DFlash2` convention. Assets have **not been uploaded** there. Fresh installations require the preparation tool or publication of these files.

## Validation

`tools/test_qwen_dflash2.py` exercises the actual shared loader and MMGP on each quantization with CPU and CUDA defaults, cancellation/recovery, repeated greedy and sampled generation, Deepy snapshot/rewind restoration, and CPU-only vision residency during text generation. Focused regression checks cover target-dependent assets/counts and the existing block-draft and speculative state/sampling contracts.

All four real checkpoints pass these checks: `Qwen3.8-27B-Uncensored-IQ2_M.gguf`, `Qwen3.8-27B-Uncensored-noMTP-IQ3_S.gguf`, `Qwen3.8-27B-Uncensored-Q4_K_M.gguf`, and `Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf`. Qwen runs confirm the original drafter and seven-token runner limit; Bonsai confirms the adapted drafter and five-token limit. The focused regression suite passes 125 tests.

An additional Q4/32K real Deepy image-inspection run correctly identified the two tabby cats in the downloaded Hugging Face COCO fixture. Vision was unloaded before answer decoding; the conversation was restored with an identical greedy continuation. Structured runtime results are retained in `benchmarks/qwen_dflash2_runtime_2026-09-20.json`.

## Q4 performance check

RTX 5090, same uncensored Q4 checkpoint, two 2048-token story prompts, 256 sampled completion tokens per prompt, 32K cache capacity and INT8 KV. Graphs are warm; per-stage profiling is disabled to avoid timing its events. Throughput is aggregate completion tokens / decode seconds, excluding prefill. These short measurements are workload-specific, not exhaustive tuning or a reproduction of the publisher's H200 benchmarks.

| Method | Maximum draft tokens | Decode tok/s | Live VRAM GiB | Peak VRAM GiB |
| --- | --- | --- | --- | --- |
| Disabled | 0 | 65.45 | 16.71 | 16.93 |
| MTP | 2 | 114.91 | 17.24 | 17.48 |
| DSpark | 7 | 96.46 | 21.34 | 21.65 |
| DFlash2 | 7 | 85.52 | 21.51 | 21.81 |

DFlash2 improves over vanilla by about 31%, but MTP remains faster on these prompts. It adds about 4.80 GiB of live VRAM, consistent with the existing label. No automatic selection policy is changed. Raw records and metadata: `benchmarks/qwen_dflash2_2026-09-20.json`.
