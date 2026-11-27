# GGUF 1.0.31

Windows and Linux wheels ship a native routed expert library for the eight gate/down format combinations in Inkling 2 Mini GSQ-RCO Q2_K_XL. The GGUF weights stay packed. Device-resident addresses allow MMGP to place each expert independently and replay decoding through CUDA graphs.

| Pass | Native implementation |
| --- | --- |
| Decode and sparsely routed prompt batches | Q8_1 activations with packed integer dots, per-expert gate/up soft caps, weighted down rows and FP32 accumulation |
| Prompt batches with at least 24 routed pairs per expert on average | The gate/up projection decodes 256-column IQ tiles once for up to 16 tokens and uses NVIDIA tensor cores or AMD wave matrix instructions. The down projection uses integer dots. |
| Streamed expert batches | Each group packs only its own intermediate rows. Zero addresses contribute zero to the result. |

The four CUDA wheels target PyTorch 2.7.1 with CUDA 12.8 and PyTorch 2.10 with CUDA 13. The two ROCm 10 wheels target PyTorch 2.13 on RDNA3 (gfx1100–gfx1103) and RDNA4 (gfx1200 and gfx1201).

The CUDA numerical tests compare against independently dequantized weights and exercise graph replay, missing experts and streamed groups on an RTX 5090. Both HIP wheels load in their corresponding PyTorch environments. Their compiled gfx1100 and gfx1201 code objects were also tested numerically in an ISA emulator. AMD board performance and full-model execution remain unmeasured. The validation record lists each wheel, its code images and the measured numerical errors.

Run `python -m pytest tests/test_routed_iq.py` to repeat the hardware tests with the wheel installed. The source build compiles the new library with the selected NVIDIA or ROCm toolkit.
