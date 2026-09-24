#pragma once

// HIP sync intrinsics require a 64-bit lane mask, including on wave32 GPUs.
#if defined(__HIP_PLATFORM_AMD__)
#define WGP_WARP_MASK (~0ull)
#else
#define WGP_WARP_MASK 0xffffffffu
#endif
