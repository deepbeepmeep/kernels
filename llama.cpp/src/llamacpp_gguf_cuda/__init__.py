import os
from pathlib import Path

import torch

from .version import __version__
from . import version as _build_version

_required_torch = getattr(_build_version, '__torch_version__', None)
if _required_torch is not None and torch.__version__ != _required_torch:
    raise ImportError(
        f'This HIP wheel requires PyTorch {_required_torch}; installed {torch.__version__}. '
        'Install the matching ROCm PyTorch build or a matching GGUF wheel.'
    )


_LINEAR_MODE_ENV = "WGP_GGUF_LLAMACPP_CUDA_LINEAR_MODE"
_MATMUL_MODE_ENV = "WGP_GGUF_LLAMACPP_CUDA_MATMUL_MODE"
_FAST_LINEAR_QTYPES = {"PTQ1_0", "Q2_K", "Q3_K", "Q4_0", "Q4_1", "Q4_K", "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0", "IQ1_S", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS"}
_FAST_EMBEDDING_QTYPES = {"PTQ1_0", "Q4_K", "Q6_K"}
_SHORT_BATCH_ENV = "LLAMACPP_GGUF_SHORT_BATCH"
_SHORT_BATCH_MODES = ("auto", "native", "mma")
_SHORT_BATCH_QTYPES = ("Q4_K", "PTQ1_0")
_LOGGED = set()
_BACKEND = "HIP" if torch.version.hip is not None else "CUDA"


def _add_dll_dirs() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for dll_dir in (Path(torch.__file__).resolve().parent / "lib", Path(os.environ.get("CUDA_PATH", "")) / "bin"):
        if dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))


_add_dll_dirs()

from . import _C

try:
    from . import _prism
except ImportError:
    _prism = None

try:
    from . import _attention
except ImportError:
    _attention = None

if hasattr(_C, "set_short_batch_mode"):
    _short_batch_mode = str(os.environ.get(_SHORT_BATCH_ENV, "auto")).strip().lower() or "auto"
    if _short_batch_mode not in _SHORT_BATCH_MODES:
        print(f"[GGUF] Ignoring {_SHORT_BATCH_ENV}={_short_batch_mode!r}; expected auto, native or mma.")
        _short_batch_mode = "auto"
    _C.set_short_batch_mode(_short_batch_mode)

if torch.version.hip is None and _attention is not None and hasattr(_attention, "load_sm120_kernel"):
    from . import sm120


def _log_once(key: str, message: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        print(message.encode("ascii", errors="ignore").decode("ascii"))


def _linear_mode() -> str:
    matmul_mode = str(os.environ.get(_MATMUL_MODE_ENV, "")).strip().lower()
    if matmul_mode in ("fast", "mmq", "packed", "low_vram"):
        return "mmq"
    if matmul_mode in ("materialized", "dense", "cublas"):
        return "cublas"
    linear_mode = str(os.environ.get(_LINEAR_MODE_ENV, "auto")).strip().lower()
    if linear_mode in ("mmq", "legacy", "v3", "mmq_v3", "v4_mmq"):
        return "mmq"
    if linear_mode in ("cublas", "dequant", "v4_cublas"):
        return "cublas"
    return "mmq"


def load_error():
    return None


def release_runtime_buffers() -> None:
    _C.release_runtime_buffers()


def prepare_runtime_buffers(device: torch.device | int | None = None) -> int:
    device_index = device if isinstance(device, int) else torch.device(device).index if device is not None else None
    if device_index is None:
        device_index = torch.cuda.current_device()
    max_size = torch.cuda.get_device_properties(device_index).multi_processor_count * 256 * 128 * 4
    reserve_size = ((max_size + 16 * 1024 * 1024 - 1) // (16 * 1024 * 1024)) * 16 * 1024 * 1024
    _C.prepare_runtime_buffers(device_index, reserve_size)
    return reserve_size


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


def dense_paged_attention(query, key_cache, value_cache, block_table, context_lens, softmax_scale, forced_num_splits=0):
    if _attention is None:
        raise RuntimeError("This llamacpp-gguf-cuda wheel does not include dense paged attention.")
    return _attention.dense_paged_attention(query, key_cache, value_cache, block_table, context_lens, softmax_scale, forced_num_splits)


def may_support_linear_qtype_name(qtype_name: str) -> bool:
    return qtype_name in _FAST_LINEAR_QTYPES


def may_support_embedding_qtype_name(qtype_name: str) -> bool:
    return qtype_name in _FAST_EMBEDDING_QTYPES


def supports_linear_qtype_name(qtype_name: str) -> bool:
    if qtype_name not in _FAST_LINEAR_QTYPES:
        return False
    mode = _linear_mode()
    _log_once(f"llamacpp_gguf_cuda_mode_{mode}", f"[GGUF][llama.cpp {_BACKEND} v1] linear mode={mode}.")
    if mode == "mmq":
        _log_once("llamacpp_gguf_cuda_packed", f"[GGUF][llama.cpp {_BACKEND} v1] Packed MMVQ/MMQ active (architecture-aware dispatch, no dense weight materialization).")
    return bool(_C.supports_linear_qtype_name(qtype_name))


def supports_embedding_qtype_name(qtype_name: str) -> bool:
    return qtype_name in _FAST_EMBEDDING_QTYPES and bool(_C.supports_embedding_qtype_name(qtype_name))


def supports_qtype_name(qtype_name: str) -> bool:
    return supports_linear_qtype_name(qtype_name)


def supports_linear_fusions(qtype_name, tokens, device):
    return (torch.version.hip is None and _linear_mode() == "mmq"
            and hasattr(_C, "supports_linear_fusions")
            and bool(_C.supports_linear_fusions(qtype_name, tokens, device)))


def has_short_batch_policy() -> bool:
    """Whether this build has the 2-8 row Q4_K/PTQ1_0 tensor-core path and its policy controls."""
    return hasattr(_C, "set_short_batch_mode")


def set_short_batch_mode(mode: str) -> None:
    """auto: per-shape decisions, else on for compute capability 12.0 only; native: MMVQ; mma: tensor cores (8.0+)."""
    # Validate here: a native check failure terminates the process on Linux builds instead of raising.
    if mode not in _SHORT_BATCH_MODES:
        raise ValueError(f"short-batch mode must be one of {_SHORT_BATCH_MODES}, got {mode!r}")
    _C.set_short_batch_mode(mode)


def short_batch_mode() -> str:
    return _C.short_batch_mode()


def set_short_batch_decision(qtype_name: str, rows: int, out_features: int, in_features: int, enabled: bool) -> None:
    """Record a measured choice for one shape; applies under the auto policy. Set before CUDA graph capture."""
    if qtype_name not in _SHORT_BATCH_QTYPES:
        raise ValueError(f"short-batch tensor-core decisions apply to {_SHORT_BATCH_QTYPES}, got {qtype_name!r}")
    _C.set_short_batch_decision(qtype_name, int(rows), int(out_features), int(in_features), bool(enabled))


def clear_short_batch_decisions() -> None:
    _C.clear_short_batch_decisions()


def linear(raw_weight: torch.Tensor, qtype_name: str, tensor_shape, input_tensor: torch.Tensor, bias: torch.Tensor | None, output_dtype: torch.dtype, *, fused_output=False, silu_mul=False):
    if fused_output or silu_mul:
        return _C.linear(raw_weight, qtype_name, list(tensor_shape), input_tensor, bias, str(output_dtype).replace("torch.", ""), _linear_mode(), fused_output, silu_mul)
    return _C.linear(raw_weight, qtype_name, list(tensor_shape), input_tensor, bias, str(output_dtype).replace("torch.", ""), _linear_mode())


def embedding(raw_weight: torch.Tensor, qtype_name: str, tensor_shape, indices: torch.Tensor, output_dtype: torch.dtype):
    return _C.embedding(raw_weight, qtype_name, list(tensor_shape), indices, str(output_dtype).replace("torch.", ""))


def prism_hadamard(input_tensor, signs, inverse=False, grouped_shape=(0, 0, 0)):
    return _C.prism_hadamard(input_tensor, signs, inverse, *grouped_shape)


def has_prism_decode():
    return _prism is not None and _linear_mode() == "mmq"


def prism_decode(input_tensor, raw_weight, signs, bias, rows, grouped_shape=(0, 0, 0), row_tile=4):
    return _prism.decode(input_tensor, raw_weight, signs, bias, rows, *grouped_shape, 4, row_tile)


__all__ = [
    "has_prism_decode", "prism_decode",
    "prism_hadamard",
    "__version__", "embedding", "linear", "load_error", "prepare_runtime_buffers", "release_runtime_buffers", "has_q8_paged_attention", "q8_paged_attention_format",
    "may_support_embedding_qtype_name", "may_support_linear_qtype_name", "q8_paged_attention", "q8_paged_attention_num_splits", "dense_paged_attention",
    "supports_embedding_qtype_name", "supports_linear_qtype_name", "supports_qtype_name", "supports_linear_fusions",
    "has_short_batch_policy", "set_short_batch_mode", "short_batch_mode", "set_short_batch_decision", "clear_short_batch_decisions",
]
