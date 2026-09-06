# WanGP engine source overlay

These files preserve the WanGP side of the kernel optimizations and the native SM120 dispatch at release 1.0.21. Paths are relative to the WanGP project root. SOURCE_MANIFEST.json records the base checkout and exact SHA-256 hashes. This is a source/reference overlay, not an independently runnable WanGP distribution. Obtain the corresponding WanGP checkout and dependencies before using its integration tests or benchmark. The native wheel itself builds independently from the parent directory.

The shared kernels remain architecture-independent. Legacy and cg avoid Triton; the vllm backend can use shared Triton kernels and selects the precompiled SM120 attention module when the new wheel and matching GPU are present. Existing older-wheel dispatch is preserved.

No model weights, user sessions, credentials, build caches or profiling conversations are included.
