# Qwen GDN decode optimizations

Qwen3.8-27B GGUF models automatically use fused Gated DeltaNet (GDN) gates and recurrence, a 64-channel four-tap decode convolution launch, and tuned single-token RMS normalization on RTX 5090 / SM120 when the resolved decoder engine is `vllm`. There is no new setting. `cg` and `legacy` retain their existing paths. Other GPU architectures, model sizes and non-GGUF backends retain their existing paths pending validation.

The existing Q4_K_M, IQ3_S and IQ2_M selections use the same activation-level optimizations as Bonsai PTQ1_0. Q4/Q3/Q2 checkpoint weights and their head layouts are unchanged. Their existing state-space parameter permutations execute before the fused kernel. It reads Q/K heads directly, computes sigmoid beta and softplus decay inside the recurrence, and updates the live state in place. Single-token decode and multi-token speculative verification use this path; verification retains every state needed to roll back rejected tokens. Multi-token prefill keeps its established implementation. Checkpoint dtypes and MMGP residency remain under the existing loader's control. MTP remains supported.

This is separate from Bonsai's earlier load-time layout conversion, which rearranges packed projection rows, convolution channels and small state-space parameters. That conversion still applies only to Bonsai. Its measured 7.4-12.1% improvement must not be attributed to the Q4/Q3/Q2 changes here.

The preliminary paired Q4 benchmark measured 61.7 to 65.0 tok/s (+5.3%) with both activation optimizations. Protocol: RTX 5090, Qwen3.8-27B-Uncensored-Q4_K_M, vLLM/CUDA graphs, Q8 KV, 2,048 prompt tokens, 512 generated tokens, three runs per version, MTP disabled. Peak allocation remained 16.93 GiB. These measurements used a process-local prototype before permanent integration. They do not establish the same speedup for Q2, Q3, other context lengths or MTP. Raw results: `tmp/q4_gdn_quick/summary.json`.

Implementation: `shared/kernels/qwen_gdn.py`, with activation selected by `shared/prompt_enhancer/qwen35_text.py`. The loader checks `engine_name == "vllm"` explicitly. The shared model prepares SSM parameters in their proper head order before either implementation. `shared/kernels/prism_gdn.py` retains compatibility imports for the original benchmark helpers. No native wheel update is required beyond the existing supported installation; restart a running WanGP process to load the Python changes.

Validation tools:

```powershell
python -m pytest tests/test_prism_gdn.py tests/test_prism_gguf.py tests/test_speculative_state.py -q
python tools/test_qwen_gdn.py --checkpoint <checkpoint.gguf> --draft 0 --output <report.json>
python tools/test_qwen_gdn.py --checkpoint <checkpoint.gguf> --draft 2 --output <report.json>
```

The real-checkpoint harness exercises CPU and CUDA default devices, repeated prompts, cancellation/recovery and automatic kernel selection. With `--engine cg` or `--engine legacy`, it asserts the new kernels are disabled. The full-block numerical tests cover grouped, tiled and interleaved parameter layouts, single-token decode and speculative verification state. Convolution state is exact; output tolerances allow rare BF16 rounding differences from launch geometry.

Validation on September 18, 2026: 76 regression tests passed. Real IQ2_M, IQ3_S and Q4_K_M checkpoint runs passed with MTP off/on and both CPU/CUDA default devices, including cancellation followed by repeated correct answers. Q4 `cg` and `legacy` runs passed and asserted `fused_gdn: false`. All three quantizations also passed repeated vision-to-text workflows, resident model reuse, and recovery after cancelling the vision tower or decoder. Bonsai PTQ1_0 retained its native dtypes and passed MTP generation/cancellation after the shared refactor. Reports are recorded in `tmp/qwen_gdn_enable/summary.json` and the individual JSON/log files beside it.

`tools/benchmark_qwen38_engine.py --reference-gdn` selects the prior preparation/convolution path for diagnostic comparisons without adding a runtime configuration switch.

The September 19 update extends fused gates to speculative verification and updates recurrent state in place. On the tested Bonsai PTQ1_0 checkpoint, ordinary decoding measured 128.0 to 131.7 tok/s (about 3%) with unchanged peak VRAM, using three 2,048-token prompts and 512 output tokens each. Fixed target verification blocks took approximately 2-4% less GPU time. This does not make speculative decoding faster than ordinary decoding on that workload. Q4, Q3, Q2 and Bonsai real-checkpoint checks passed, including repeated calls and cancellation recovery; no equivalent speedup is claimed for every quantization.

---

> Applies to: Qwen3.8 and Bonsai accelerated vLLM decoding and quantization compatibility.
