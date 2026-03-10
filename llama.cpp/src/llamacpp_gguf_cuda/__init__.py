import os
from pathlib import Path

import torch

from .version import __version__


_LINEAR_MODE_ENV = "WGP_GGUF_LLAMACPP_CUDA_LINEAR_MODE"
_FAST_LINEAR_QTYPES = {"Q2_K", "Q3_K", "Q4_0", "Q4_1", "Q4_K", "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0", "IQ1_S", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS"}
_FAST_EMBEDDING_QTYPES = {"Q4_K", "Q6_K"}
_LOGGED = set()


def _add_dll_dirs() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    dll_dirs = [Path(torch.__file__).resolve().parent / "lib", Path(os.environ.get("CUDA_PATH", "")) / "bin"]
    for dll_dir in dll_dirs:
        if dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))


_add_dll_dirs()

from . import _C


def _log_once(key: str, message: str) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    print(message.encode("ascii", errors="ignore").decode("ascii"))


def _linear_mode() -> str:
    raw = str(os.environ.get(_LINEAR_MODE_ENV, "auto")).strip().lower()
    if raw in ("mmq", "legacy", "v3", "mmq_v3", "v4_mmq"):
        return "mmq"
    if raw in ("cublas", "dequant", "v4_cublas"):
        return "cublas"
    return "auto"


def load_error():
    return None


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
    return _C.linear(raw_weight, qtype_name, list(tensor_shape), input_tensor, bias, dtype_name, _linear_mode())


def embedding(raw_weight: torch.Tensor, qtype_name: str, tensor_shape, indices: torch.Tensor, output_dtype: torch.dtype):
    dtype_name = str(output_dtype).replace("torch.", "")
    return _C.embedding(raw_weight, qtype_name, list(tensor_shape), indices, dtype_name)


__all__ = [
    "__version__",
    "embedding",
    "linear",
    "load_error",
    "may_support_embedding_qtype_name",
    "may_support_linear_qtype_name",
    "supports_embedding_qtype_name",
    "supports_linear_qtype_name",
    "supports_qtype_name",
]
