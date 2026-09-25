# WanGP engine source overlay

These files preserve the WanGP side of the kernel optimizations at release 1.0.24. Paths are relative to the WanGP project root. SOURCE_MANIFEST.json records the base checkout and exact SHA-256 hashes of the committed (LF) content. This is a source/reference overlay, not an independently runnable WanGP distribution. Obtain the corresponding WanGP checkout and dependencies before using its integration tests or benchmark. The native wheel itself builds independently from the parent directory.

The shared kernels remain architecture-independent. Legacy and cg avoid Triton; the vllm backend can use shared Triton kernels and selects the precompiled SM120 attention module when the new wheel and matching GPU are present. Existing older-wheel dispatch is preserved.

1.0.24 adds the short-batch tensor-core linear for Q4_K and PTQ1_0 speculative verification. `shared/kernels/gguf_short_batch.py` chooses between it and MMVQ per shape and row count on each GPU, on the model's own weights, right before CUDA graph capture (`ModelRunner.capture_cudagraph`), and caches the choices. `tools/validate_gguf_short_batch.py` writes an optional per-GPU report. Engineering notes are in `specs/`.

No model weights, user sessions, credentials, build caches or profiling conversations are included.
