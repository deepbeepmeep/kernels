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
_LINEAR_BLOCK_SIZES = {name: 256 for name in _FAST_LINEAR_QTYPES}
_LINEAR_BLOCK_SIZES.update({name: 32 for name in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL")})
_LINEAR_BLOCK_SIZES["PTQ1_0"] = 128
_NARROW_BLOCK_BYTES = {"Q4_0": 18, "Q4_1": 20, "Q5_0": 22, "Q5_1": 24, "Q8_0": 34, "IQ4_NL": 18}  # formats of 32-value blocks
_BATCHED_STEP = 256  # values of a weight row the batched kernel (MMQ) takes per step
_SHORT_BATCH_ENV = "LLAMACPP_GGUF_SHORT_BATCH"
_SHORT_BATCH_MODES = ("auto", "native", "mma")
_SHORT_BATCH_QTYPES = ("Q4_K", "PTQ1_0")
_IQ_ROWS_QTYPES = ("IQ3_S", "IQ2_S")
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
    _C.prepare_runtime_buffers(device_index, max_size)
    return max_size


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


def has_bounded_batched_reads() -> bool:
    """linear and linear_chunked no longer read past a weight of 32-value blocks whose width is not a multiple of 256 (see _last_rows_apart)."""
    return True


def _narrow_rows(qtype_name: str, tensor_shape) -> bool:
    return qtype_name in _NARROW_BLOCK_BYTES and tensor_shape[1] % _BATCHED_STEP != 0 and tensor_shape[1] % 32 == 0


def _last_rows_apart(call, raw_weight, qtype_name, tensor_shape, bias):
    """The batched kernel takes a weight row by steps of 256 values. A row of 32-value blocks that ends inside a step is read on: into the next
    rows, whose products meet the zero padding of the activations, and for the last rows past the end of the weight, where a scale that is
    not finite gives NaN and unmapped memory a fault. The last rows (those a step can leave the weight from: one for a width of 224 or
    more) are served from a copy followed by zero blocks, the other rows as a weight of their own, followed by those. Same kernel, same results."""
    rows, columns = int(tensor_shape[0]), int(tensor_shape[1])
    blocks, block_bytes, step = columns // 32, _NARROW_BLOCK_BYTES[qtype_name], _BATCHED_STEP // 32
    apart = min(rows, -(-(step - blocks % step) // blocks))
    row_bytes, flat = blocks * block_bytes, raw_weight.reshape(-1)
    last = torch.zeros(apart * row_bytes + step * block_bytes, dtype=flat.dtype, device=flat.device)
    last[:apart * row_bytes] = flat[(rows - apart) * row_bytes:rows * row_bytes]
    tail = call(last, [apart, columns], None if bias is None else bias[rows - apart:])
    if rows == apart:
        return tail
    return torch.cat((call(flat[:(rows - apart) * row_bytes], [rows - apart, columns], None if bias is None else bias[:rows - apart]), tail), -1)


def has_short_batch_policy() -> bool:
    """Whether this build has the 2-8 row Q4_K/PTQ1_0 tensor-core path and its policy controls."""
    return hasattr(_C, "set_short_batch_mode")


def has_typed_mmq() -> bool:
    """Whether MMQ linears (beyond the MMVQ batch sizes) write BF16 outputs in the kernel: no FP32 product, no conversion pass."""
    return hasattr(_C, "has_typed_mmq") and bool(_C.has_typed_mmq())


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


def iq_rows_revision() -> int:
    """Revision of the multi-row IQ kernels: a measured decision holds for the revision it was measured with."""
    from . import _iq_rows
    return _iq_rows.REVISION


def iq_rows_variants(device_index: int) -> tuple:
    """The multi-row IQ3_S / IQ2_S kernel variants of a GPU ('dp4a', and 'mma' / 'mma2' from compute capability 8.0), empty when it has none."""
    from . import _iq_rows
    return _iq_rows.variants(device_index) if _linear_mode() == 'mmq' else ()


def set_iq_rows_decision(qtype_name: str, rows: int, out_features: int, in_features: int, variant: str | None) -> None:
    """Which kernel serves an IQ3_S / IQ2_S linear of 1-8 rows: one of iq_rows_variants, '' for llama.cpp's MMVQ, None for the default
    (what was measured on compute capability 12.0, MMVQ elsewhere)."""
    from . import _iq_rows
    _iq_rows.set_decision(qtype_name, rows, out_features, in_features, variant)


def has_head_candidates(device_index: int) -> bool:
    """head_candidates serves this GPU (CUDA)."""
    from . import _head_candidates
    return _linear_mode() == 'mmq' and _head_candidates.eligible(device_index)


def head_sign_bits(weight: torch.Tensor):
    """The one-bit copy of an output head's dense rows [N, K]: (bits int32 [N, K / 32], scale float32 [N])."""
    from . import _head_candidates
    return _head_candidates.sign_bits(weight)


def head_group_bits(bits: torch.Tensor, groups: int) -> torch.Tensor:
    """The bits of head_sign_bits, rows in the order head_candidates will take them, in the layout it reads (same shape)."""
    from . import _head_candidates
    return _head_candidates.group_bits(bits, int(groups))


def head_candidates(input_tensor: torch.Tensor, bits: torch.Tensor, row_scale: torch.Tensor, ids: torch.Tensor, groups: int, keep: int) -> torch.Tensor:
    """The rows of an output head worth an exact score, from its one-bit copy: int32 [rows of the input, groups * keep] ids, the `keep`
    best rows of each of `groups` runs of consecutive rows of the copy (bits: head_group_bits of the rows in that order; ids: the output
    index of each of them)."""
    from . import _head_candidates
    return _head_candidates.candidates(input_tensor, bits, row_scale, ids, int(groups), int(keep))


def packed_row_kernel(qtype_name, device_index, output_dtype):
    """Return the packed custom op, or None, for callers handling restored rows.

    Call as op(input, raw_uint8, [out_features, in_features], qtype_name, bias,
    output_dtype). Validation runs inside the custom-op boundary, preserving
    tensor-subclass dispatch without repeated Python metadata interception.
    """
    from . import _packed
    if qtype_name not in _packed.FORMATS or output_dtype not in _packed.DTYPES:
        return None
    return _packed.implementation(device_index)


def supports_linear_layout(qtype_name, tensor_shape, device, output_dtype):
    """Whether linear can execute this logical GGUF matrix in the selected mode.

    Under MMQ, rows crossing quantization blocks use packed tile decoding on
    supported CUDA devices. Unsupported layouts return False before native launch.
    """
    if qtype_name not in _FAST_LINEAR_QTYPES or len(tensor_shape) != 2 or min(tensor_shape) <= 0:
        return False
    if torch.device(device).type != 'cuda' or output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if _linear_mode() != 'mmq' or tensor_shape[1] % _LINEAR_BLOCK_SIZES[qtype_name] == 0:
        return True
    from . import _packed
    return _packed.supports(qtype_name, device, output_dtype)


def linear(raw_weight: torch.Tensor, qtype_name: str, tensor_shape, input_tensor: torch.Tensor, bias: torch.Tensor | None, output_dtype: torch.dtype, *, fused_output=False, silu_mul=False):
    if len(tensor_shape) != 2:
        raise ValueError('GGUF linear requires a two-dimensional weight shape.')
    mode = _linear_mode()
    if qtype_name in _IQ_ROWS_QTYPES and mode == 'mmq' and input_tensor.is_cuda:
        # 1-8 rows (a decoding step, a speculative verification): each weight block decoded once for all rows; its output is always written in
        # the output dtype by the kernel, as fused_output asks
        from . import _iq_rows
        variant = _iq_rows.variant(qtype_name, tensor_shape, input_tensor.numel() // (tensor_shape[1] * (2 if silu_mul else 1)), input_tensor.device.index)
        if variant:
            return _iq_rows.iq_linear(input_tensor, raw_weight, qtype_name, list(tensor_shape), bias, output_dtype, silu_mul, variant)
    block = _LINEAR_BLOCK_SIZES.get(qtype_name)
    if mode == 'mmq' and block is not None and tensor_shape[1] % block:
        if silu_mul:
            raise NotImplementedError('Fused SiLU is unavailable for GGUF rows crossing quantization blocks.')
        from . import _packed
        out = _packed.linear(raw_weight, qtype_name, tensor_shape, input_tensor, bias, output_dtype)
        _log_once(f'packed_flat_linear_{qtype_name}', f'[GGUF] Packed linear active for {qtype_name} rows crossing blocks.')
        return out
    if qtype_name == 'PTQ1_0' and mode == 'mmq' and not silu_mul and input_tensor.is_cuda:
        # SM75: MMVQ's rows (1-7) with the trits of each weight block decoded once for all rows (same results per row); its output is
        # always written in the output dtype by the kernel, as fused_output asks
        from . import _ptq1_turing
        if _ptq1_turing.eligible(input_tensor.device.index) and input_tensor.numel() <= _ptq1_turing.MAX_ROWS * tensor_shape[1]:
            return _ptq1_turing.ptq1_linear(input_tensor, raw_weight, list(tensor_shape), bias, output_dtype)
    dtype_name = str(output_dtype).replace("torch.", "")
    if mode == 'mmq' and not silu_mul and _narrow_rows(qtype_name, tensor_shape) and input_tensor.is_cuda:
        rows = input_tensor.numel() // tensor_shape[1]
        if rows > 8 or not supports_linear_fusions(qtype_name, rows, input_tensor.device.index):  # the batched kernel serves the call
            if fused_output:
                return _last_rows_apart(lambda raw, shape, part: _C.linear(raw, qtype_name, shape, input_tensor, part, dtype_name, mode, True, False), raw_weight, qtype_name, tensor_shape, bias)
            return _last_rows_apart(lambda raw, shape, part: _C.linear(raw, qtype_name, shape, input_tensor, part, dtype_name, mode), raw_weight, qtype_name, tensor_shape, bias)
    if fused_output or silu_mul:
        return _C.linear(raw_weight, qtype_name, list(tensor_shape), input_tensor, bias, dtype_name, mode, fused_output, silu_mul)
    return _C.linear(raw_weight, qtype_name, list(tensor_shape), input_tensor, bias, dtype_name, mode)


def linear_chunked_supported(qtype_name: str, tensor_shape, output_dtype: torch.dtype, biased: bool, device_index: int) -> bool:
    """Whether linear_chunked takes this qtype, matrix shape, 16-bit output type and bias on the device (structural limits only)."""
    # not the weights whose last rows the batched kernel would read past (_last_rows_apart): linear serves them
    return _linear_mode() == "mmq" and not _narrow_rows(qtype_name, tensor_shape) and _C.linear_chunked_supported(qtype_name, int(tensor_shape[0]), int(tensor_shape[1]), str(output_dtype).replace("torch.", ""), bool(biased), int(device_index))


def linear_chunked(raw_weight: torch.Tensor, qtype_name: str, tensor_shape, input_tensor: torch.Tensor, bias: torch.Tensor | None, output_dtype: torch.dtype, scratch_bytes: int):
    """Linear over many rows in one cooperative launch, or None when unsupported.

    Row chunks are quantized into a ring of Q8 slots (at most scratch_bytes, at least 3 x 128 rows) while other SMs multiply earlier chunks; each output
    tile accumulates its whole K range (no stream-K split), so results do not depend on the chunk size or the GPU, and the bias is
    added in the output type as a separate add would. Input must be contiguous and of the output type.
    """
    if _linear_mode() != "mmq" or not input_tensor.is_contiguous() or _narrow_rows(qtype_name, tensor_shape):
        return None
    return _C.linear_chunked(raw_weight, qtype_name, list(tensor_shape), input_tensor, bias, str(output_dtype).replace("torch.", ""), int(scratch_bytes))


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
    "__version__", "embedding", "linear", "linear_chunked", "linear_chunked_supported", "load_error", "prepare_runtime_buffers", "release_runtime_buffers", "has_q8_paged_attention", "q8_paged_attention_format",
    "may_support_embedding_qtype_name", "may_support_linear_qtype_name", "q8_paged_attention", "q8_paged_attention_num_splits", "dense_paged_attention",
    "supports_embedding_qtype_name", "supports_linear_qtype_name", "supports_qtype_name", "supports_linear_fusions",
    "supports_linear_layout", "packed_row_kernel",
    "has_bounded_batched_reads", "has_head_candidates", "head_candidates", "head_group_bits", "head_sign_bits", "has_short_batch_policy", "set_short_batch_mode", "short_batch_mode", "set_short_batch_decision", "clear_short_batch_decisions",
]
