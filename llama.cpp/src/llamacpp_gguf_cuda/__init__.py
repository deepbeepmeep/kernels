import math
import os
from pathlib import Path

import torch

from .version import __version__


_LINEAR_MODE_ENV = "WGP_GGUF_LLAMACPP_CUDA_LINEAR_MODE"
_MATMUL_MODE_ENV = "WGP_GGUF_LLAMACPP_CUDA_MATMUL_MODE"
_BF16_FP16_ENV = "WGP_GGUF_LLAMACPP_CUDA_BF16_FP16"
_STREAM_K_ENV = "WGP_GGUF_LLAMACPP_CUDA_STREAM_K"
_STREAM_K_BUFFER_MB_ENV = "WGP_GGUF_LLAMACPP_CUDA_STREAM_K_BUFFER_MB"
_DEFAULT_STREAM_K_BUFFER_SIZE = 16 * 1024 * 1024
_FAST_LINEAR_QTYPES = {"Q2_K", "Q3_K", "Q4_0", "Q4_1", "Q4_K", "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0", "IQ1_S", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS"}
_FAST_EMBEDDING_QTYPES = {"Q4_K", "Q6_K"}
_LOGGED = set()
_ENV_CONFIG = ("auto", False, False, True, _DEFAULT_STREAM_K_BUFFER_SIZE)


def _add_dll_dirs() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    dll_dirs = [Path(torch.__file__).resolve().parent / "lib", Path(os.environ.get("CUDA_PATH", "")) / "bin"]
    for dll_dir in dll_dirs:
        if dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))


_add_dll_dirs()

from . import _C

try:
    from . import _attention
except ImportError:  # Older wheels remain usable for GGUF linear operations.
    _attention = None


def _log_once(key: str, message: str) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    print(message.encode("ascii", errors="ignore").decode("ascii"))


def _read_bool_env(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw in ("", "auto", "default"):
        return default
    if raw in ("1", "true", "yes", "on", "enable", "enabled"):
        return True
    if raw in ("0", "false", "no", "off", "disable", "disabled"):
        return False
    raise ValueError(f"{name} must be auto, on, or off; got {raw!r}")


def _read_stream_k_buffer_size() -> int:
    raw = str(os.environ.get(_STREAM_K_BUFFER_MB_ENV, "16")).strip()
    try:
        size_mib = float(raw)
    except ValueError as exc:
        raise ValueError(f"{_STREAM_K_BUFFER_MB_ENV} must be a non-negative MiB value; got {raw!r}") from exc
    if not math.isfinite(size_mib) or size_mib < 0:
        raise ValueError(f"{_STREAM_K_BUFFER_MB_ENV} must be a finite, non-negative MiB value; got {raw!r}")
    size = int(size_mib * 1024 * 1024)
    return (size + 255) // 256 * 256


def _read_env_config() -> tuple[str, bool, bool, bool, int]:
    raw = str(os.environ.get(_MATMUL_MODE_ENV, "")).strip().lower()
    if raw in ("fast", "materialized", "dense", "cublas"):
        mode = "fast"
    elif raw in ("mmq", "packed", "low_vram"):
        mode = "mmq"
    else:
        linear_raw = str(os.environ.get(_LINEAR_MODE_ENV, "auto")).strip().lower()
        if linear_raw in ("mmq", "legacy", "v3", "mmq_v3", "v4_mmq"):
            mode = "mmq"
        elif linear_raw in ("cublas", "dequant", "v4_cublas"):
            mode = "cublas"
        else:
            mode = "auto"
    bf16_fp16 = str(os.environ.get(_BF16_FP16_ENV, "")).strip().lower() in ("1", "true", "yes", "on")
    stream_k = _read_bool_env(_STREAM_K_ENV, True)
    stream_k_buffer_size = _read_stream_k_buffer_size()
    return mode, bool(raw), bf16_fp16, stream_k and stream_k_buffer_size > 0, stream_k_buffer_size


def refresh_env() -> str:
    global _ENV_CONFIG
    _ENV_CONFIG = _read_env_config()
    _C.configure_stream_k(_ENV_CONFIG[3], _ENV_CONFIG[4])
    return _ENV_CONFIG[0]


def _linear_mode() -> str:
    return _ENV_CONFIG[0]


def _linear_mode_for_dtype(output_dtype: torch.dtype) -> str:
    mode, matmul_explicit, bf16_fp16 = _ENV_CONFIG[:3]
    if mode == "fast":
        _log_once("llamacpp_gguf_cuda_fast", "[GGUF][llama.cpp CUDA v1] matmul mode=fast (MMQ for small workloads, native-dtype materialized cuBLAS otherwise).")
    elif mode == "mmq" and matmul_explicit:
        _log_once("llamacpp_gguf_cuda_low_vram", "[GGUF][llama.cpp CUDA v1] matmul mode=mmq (packed weights, no dense weight materialization).")
    if output_dtype == torch.bfloat16 and mode == "auto" and bf16_fp16:
        _log_once("llamacpp_gguf_cuda_bf16_fp16", f"[GGUF][llama.cpp CUDA v1] {_BF16_FP16_ENV}=1: BF16 uses the legacy FP16 cuBLAS path.")
        return "cublas_fp16"
    if output_dtype == torch.bfloat16 and mode == "auto":
        _log_once("llamacpp_gguf_cuda_bf16_mmq", "[GGUF][llama.cpp CUDA v1] BF16 auto policy=MMQ on supported GPUs.")
    return mode


refresh_env()


def load_error():
    return None


def has_q8_paged_attention() -> bool:
    return _attention is not None


def q8_paged_attention_format() -> str:
    return "q8_0_fp16_scales_v1" if _attention is not None else ""


def q8_paged_attention(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, softmax_scale, forced_num_splits=0):
    if _attention is None:
        raise RuntimeError("This llamacpp-gguf-cuda wheel does not include Q8 paged attention.")
    return _attention.q8_paged_attention(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, softmax_scale, forced_num_splits)


def q8_paged_attention_num_splits(query, cache_capacity):
    if _attention is None:
        raise RuntimeError("This llamacpp-gguf-cuda wheel does not include Q8 paged attention.")
    return _attention.q8_paged_attention_num_splits(query, cache_capacity)


def may_support_linear_qtype_name(qtype_name: str) -> bool:
    return qtype_name in _FAST_LINEAR_QTYPES


def may_support_embedding_qtype_name(qtype_name: str) -> bool:
    return qtype_name in _FAST_EMBEDDING_QTYPES


def supports_linear_qtype_name(qtype_name: str) -> bool:
    if qtype_name not in _FAST_LINEAR_QTYPES:
        return False
    _log_once(f"llamacpp_gguf_cuda_mode_{_linear_mode()}", f"[GGUF][llama.cpp CUDA v1] linear mode={_linear_mode()}.")
    return bool(_C.supports_linear_qtype_name(qtype_name))


def supports_embedding_qtype_name(qtype_name: str) -> bool:
    return qtype_name in _FAST_EMBEDDING_QTYPES and bool(_C.supports_embedding_qtype_name(qtype_name))


def supports_qtype_name(qtype_name: str) -> bool:
    return supports_linear_qtype_name(qtype_name)


def linear(raw_weight: torch.Tensor, qtype_name: str, tensor_shape, input_tensor: torch.Tensor, bias: torch.Tensor | None, output_dtype: torch.dtype):
    dtype_name = str(output_dtype).replace("torch.", "")
    return _C.linear(raw_weight, qtype_name, list(tensor_shape), input_tensor, bias, dtype_name, _linear_mode_for_dtype(output_dtype))


def embedding(raw_weight: torch.Tensor, qtype_name: str, tensor_shape, indices: torch.Tensor, output_dtype: torch.dtype):
    dtype_name = str(output_dtype).replace("torch.", "")
    return _C.embedding(raw_weight, qtype_name, list(tensor_shape), indices, dtype_name)


__all__ = [
    "__version__",
    "embedding",
    "linear",
    "load_error",
    "has_q8_paged_attention",
    "q8_paged_attention_format",
    "may_support_embedding_qtype_name",
    "may_support_linear_qtype_name",
    "refresh_env",
    "q8_paged_attention",
    "q8_paged_attention_num_splits",
    "supports_embedding_qtype_name",
    "supports_linear_qtype_name",
    "supports_qtype_name",
]
