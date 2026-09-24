"""Isolated, unintegrated experiment for exact compact Qwen GDN rollback.

Adapted from shared/kernels/qwen_gdn.py and the FLA-derived verification kernel.
https://github.com/fla-org/flash-linear-attention
Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from fla.ops.utils.op import exp


@dataclass(frozen=True)
class CompactWorkspace:
    initial: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    gates: torch.Tensor
    beta: torch.Tensor

    @classmethod
    def allocate(cls, state, *, max_tokens, key_heads, key_dtype, value_dtype, beta_dtype):
        batch, value_heads, dim_k, dim_v = state.shape
        if max_tokens < 2 or key_heads <= 0 or value_heads % key_heads:
            raise ValueError("Invalid verification capacity or grouped head count")
        if not state.is_contiguous():
            raise ValueError("The experiment requires contiguous grouped state")
        kwargs = {"device": state.device}
        return cls(
            torch.empty_like(state),
            torch.empty((batch, max_tokens, key_heads, dim_k), dtype=key_dtype, **kwargs),
            torch.empty((batch, max_tokens, value_heads, dim_v), dtype=value_dtype, **kwargs),
            torch.empty((batch, max_tokens, value_heads), dtype=torch.float32, **kwargs),
            torch.empty((batch, max_tokens, value_heads), dtype=beta_dtype, **kwargs),
        )

    @property
    def max_tokens(self):
        return self.keys.shape[1]

    @property
    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in self.tensors())

    def tensors(self):
        return (self.initial, self.keys, self.values, self.gates, self.beta)


@triton.jit
def _record_raw_kernel(Q, K, V, A, BETA_IN, SSM_A, DT, STATE, OUT,
                       INITIAL, LOG_K, LOG_V, LOG_G, LOG_BETA,
                       T: tl.constexpr, CAPACITY: tl.constexpr,
                       H: tl.constexpr, HV: tl.constexpr,
                       DK: tl.constexpr, DV: tl.constexpr,
                       QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr,
                       KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
                       VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
                       AS0: tl.constexpr, AS1: tl.constexpr,
                       BS0: tl.constexpr, BS1: tl.constexpr,
                       BK: tl.constexpr, BV: tl.constexpr,
                       V_HEADS_TILED: tl.constexpr, SSM_PARAMS_TILED: tl.constexpr,
                       INTERLEAVE_AB: tl.constexpr):
    vb, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HV, bh % HV
    kh = head // (HV // H)
    tiled_head = (head % (HV // H)) * H + kh
    interleaved_head = head // 2 + (head % 2) * (HV // 2)
    value_head = tiled_head if V_HEADS_TILED else head
    gate_head = tiled_head if V_HEADS_TILED else (interleaved_head if INTERLEAVE_AB and HV % 2 == 0 else head)
    param_head = tiled_head if SSM_PARAMS_TILED else (interleaved_head if INTERLEAVE_AB and HV % 2 == 0 else head)
    keys = tl.arange(0, BK)
    values = vb * BV + tl.arange(0, BV)
    state_index = bh * DK * DV + keys[:, None] * DV + values[None, :]
    mask = (keys[:, None] < DK) & (values[None, :] < DV)
    state = tl.load(STATE + state_index, mask, 0).to(tl.float32)
    tl.store(INITIAL + state_index, state, mask)
    sa = tl.load(SSM_A + param_head).to(tl.float32)
    dt = tl.load(DT + param_head).to(tl.float32)
    for token in range(T):
        q = tl.load(Q + batch * QS0 + token * QS1 + kh * QS2 + keys, keys < DK, 0).to(tl.float32)
        k = tl.load(K + batch * KS0 + token * KS1 + kh * KS2 + keys, keys < DK, 0).to(tl.float32)
        v = tl.load(V + batch * VS0 + token * VS1 + value_head * VS2 + values, values < DV, 0).to(tl.float32)
        # Logs use their allocated capacity stride, independent of current T.
        if vb == 0:
            if head % (HV // H) == 0:
                tl.store(LOG_K + ((batch * CAPACITY + token) * H + kh) * DK + keys, k, keys < DK)
        tl.store(LOG_V + ((batch * CAPACITY + token) * HV + head) * DV + values, v, values < DV)
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        q *= DK ** -0.5
        a = tl.load(A + batch * AS0 + token * AS1 + gate_head).to(tl.float32) + dt
        softplus = tl.where(a > 20., a, libdevice.log1p(tl.exp(a)))
        g = sa * softplus
        raw_beta = tl.load(BETA_IN + batch * BS0 + token * BS1 + gate_head)
        beta = tl.sigmoid(raw_beta.to(tl.float32)).to(BETA_IN.dtype.element_ty).to(tl.float32)
        if vb == 0:
            tl.store(LOG_G + (batch * CAPACITY + token) * HV + head, g)
            tl.store(LOG_BETA + (batch * CAPACITY + token) * HV + head, beta)
        # Match the existing raw kernel's arithmetic and FP32 carry exactly.
        state *= exp(g)
        v = beta * (v - tl.sum(state * k[:, None], 0))
        state += k[:, None] * v
        out = tl.sum(state * q[:, None], 0)
        tl.store(OUT + ((batch * T + token) * HV + value_head) * DV + values, out, values < DV)
    tl.store(STATE + state_index, state, mask)


@triton.jit
def _replay_prefix_kernel(INITIAL, LOG_K, LOG_V, LOG_G, LOG_BETA, STATE,
                          PREFIX: tl.constexpr, CAPACITY: tl.constexpr,
                          H: tl.constexpr, HV: tl.constexpr,
                          DK: tl.constexpr, DV: tl.constexpr,
                          BK: tl.constexpr, BV: tl.constexpr):
    vb, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HV, bh % HV
    kh = head // (HV // H)
    keys = tl.arange(0, BK)
    values = vb * BV + tl.arange(0, BV)
    state_index = bh * DK * DV + keys[:, None] * DV + values[None, :]
    mask = (keys[:, None] < DK) & (values[None, :] < DV)
    state = tl.load(INITIAL + state_index, mask, 0).to(tl.float32)
    for token in range(PREFIX):
        k = tl.load(LOG_K + ((batch * CAPACITY + token) * H + kh) * DK + keys, keys < DK, 0).to(tl.float32)
        v = tl.load(LOG_V + ((batch * CAPACITY + token) * HV + head) * DV + values, values < DV, 0).to(tl.float32)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        g = tl.load(LOG_G + (batch * CAPACITY + token) * HV + head).to(tl.float32)
        beta = tl.load(LOG_BETA + (batch * CAPACITY + token) * HV + head).to(tl.float32)
        state *= exp(g)
        v = beta * (v - tl.sum(state * k[:, None], 0))
        state += k[:, None] * v
    tl.store(STATE + state_index, state, mask)


def _validate_inputs(q, k, v, a, b, ssm_a, ssm_dt, state, workspace, output):
    batch, tokens, heads, dim_k = q.shape
    value_heads, dim_v = v.shape[-2:]
    if tokens < 2 or tokens > workspace.max_tokens or value_heads % heads:
        raise ValueError("Invalid verification length or grouped head count")
    expected = (batch, value_heads, dim_k, dim_v)
    if tuple(state.shape) != expected or tuple(workspace.initial.shape) != expected:
        raise ValueError("State/workspace shape does not match inputs")
    if q.shape != k.shape or v.shape[:2] != q.shape[:2]:
        raise ValueError("Incompatible Q/K/V shapes")
    if tuple(a.shape) != (batch, tokens, value_heads) or b.shape != a.shape:
        raise ValueError("Incompatible gate shapes")
    if ssm_a.numel() != value_heads or ssm_dt.numel() != value_heads:
        raise ValueError("Incompatible SSM parameter shapes")
    if workspace.keys.shape[2:] != (heads, dim_k) or workspace.values.shape[2:] != (value_heads, dim_v):
        raise ValueError("Incompatible workspace head dimensions")
    if (workspace.keys.dtype != k.dtype or workspace.values.dtype != v.dtype
            or workspace.beta.dtype != b.dtype or workspace.initial.dtype != state.dtype):
        raise ValueError("Workspace must preserve each source dtype")
    tensors = (q, k, v, a, b, ssm_a, ssm_dt, state, output, *workspace.tensors())
    if any(t.device != state.device for t in tensors) or state.device.type != "cuda":
        raise ValueError("All experimental inputs must be on one CUDA device")
    if any(t.stride(-1) != 1 for t in (q, k, v, a, b)):
        raise ValueError("The raw kernel requires unit innermost strides")
    if any(not t.is_contiguous() for t in (ssm_a, ssm_dt, state, output, *workspace.tensors())):
        raise ValueError("State, parameters, output, and logs must be contiguous")
    if output.shape != v.shape or output.dtype != v.dtype:
        raise ValueError("Output must match V shape and dtype")


def record_raw_gates(q, k, v, a, b, ssm_a, ssm_dt, state, workspace, *, output=None,
                     v_heads_tiled=False, ssm_params_tiled=False, interleave_ab=False):
    """Verify and record one layer; all workspace pointers must remain alive."""
    if output is None:
        output = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    _validate_inputs(q, k, v, a, b, ssm_a, ssm_dt, state, workspace, output)
    batch, tokens, heads, dim_k = q.shape
    value_heads, dim_v = v.shape[-2:]
    _record_raw_kernel[(triton.cdiv(dim_v, 8), batch * value_heads)](
        q, k, v, a, b, ssm_a, ssm_dt, state, output,
        *workspace.tensors(), tokens, workspace.max_tokens, heads, value_heads,
        dim_k, dim_v, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        *a.stride()[:2], *b.stride()[:2], triton.next_power_of_2(dim_k), 8,
        v_heads_tiled, ssm_params_tiled, interleave_ab,
        num_warps=1, num_stages=3,
    )
    return output, state


def replay_prefix(state, workspace, processed_tokens, verified_tokens):
    """Replay the prefix in one FP32-carry kernel; full acceptance is a no-op."""
    if not 1 <= processed_tokens <= verified_tokens <= workspace.max_tokens:
        raise ValueError("Commit prefix is outside this verification")
    if processed_tokens == verified_tokens:
        return state
    if state.shape != workspace.initial.shape or state.dtype != workspace.initial.dtype:
        raise ValueError("Commit state does not match the workspace")
    if state.device != workspace.initial.device or not state.is_contiguous():
        raise ValueError("Commit must use contiguous state on the workspace device")
    batch, value_heads, dim_k, dim_v = state.shape
    _replay_prefix_kernel[(triton.cdiv(dim_v, 8), batch * value_heads)](
        *workspace.tensors(), state, processed_tokens, workspace.max_tokens,
        workspace.keys.shape[2], value_heads, dim_k, dim_v,
        triton.next_power_of_2(dim_k), 8, num_warps=1, num_stages=3,
    )
    return state


@dataclass(frozen=True)
class BatchedReplay:
    """Pointer-table owner for uniform layers; intentionally tool-only."""
    pointers: torch.Tensor
    states: tuple
    workspaces: tuple

    @classmethod
    def prepare(cls, states, workspaces):
        states, workspaces = tuple(states), tuple(workspaces)
        if not states or len(states) != len(workspaces):
            raise ValueError("Each layer must have a state and compact workspace")
        first_state, first_workspace = states[0], workspaces[0]
        signature = tuple((tuple(t.shape), t.dtype, t.device) for t in (*first_workspace.tensors(), first_state))
        for state, workspace in zip(states, workspaces):
            if tuple((tuple(t.shape), t.dtype, t.device) for t in (*workspace.tensors(), state)) != signature:
                raise ValueError("The pointer-table experiment requires uniform layers")
            if any(not t.is_contiguous() for t in (*workspace.tensors(), state)):
                raise ValueError("The pointer-table experiment requires contiguous storage")
            if any(t.data_ptr() % 16 for t in (*workspace.tensors(), state)):
                raise ValueError("The pointer-table experiment requires 16-byte-aligned pointers")
        addresses = [[t.data_ptr() for t in (*workspace.tensors(), state)]
                     for state, workspace in zip(states, workspaces)]
        # Six rows: initial backup, raw K, raw V, FP32 g, rounded beta, live state.
        pointers = torch.tensor(list(zip(*addresses)), dtype=torch.uint64, device=first_state.device)
        return cls(pointers, states, workspaces)


@triton.jit
def _replay_batched_kernel(TABLE, PREFIX: tl.constexpr, CAPACITY: tl.constexpr,
                           H: tl.constexpr, HV: tl.constexpr,
                           DK: tl.constexpr, DV: tl.constexpr,
                           BATCH: tl.constexpr, LAYERS: tl.constexpr,
                           BK: tl.constexpr, BV: tl.constexpr,
                           STATE_DTYPE: tl.constexpr, KEY_DTYPE: tl.constexpr,
                           VALUE_DTYPE: tl.constexpr, BETA_DTYPE: tl.constexpr):
    vb, layer_bh = tl.program_id(0), tl.program_id(1)
    layer, bh = layer_bh // (BATCH * HV), layer_bh % (BATCH * HV)
    batch, head = bh // HV, bh % HV
    # Match the pointer alignment metadata on direct tensor arguments. Losing it
    # changed the reduction layout and failed exact parity in the first prototype.
    INITIAL = tl.multiple_of(tl.load(TABLE + layer).to(tl.pointer_type(STATE_DTYPE)), 16)
    LOG_K = tl.multiple_of(tl.load(TABLE + LAYERS + layer).to(tl.pointer_type(KEY_DTYPE)), 16)
    LOG_V = tl.multiple_of(tl.load(TABLE + 2 * LAYERS + layer).to(tl.pointer_type(VALUE_DTYPE)), 16)
    LOG_G = tl.multiple_of(tl.load(TABLE + 3 * LAYERS + layer).to(tl.pointer_type(tl.float32)), 16)
    LOG_BETA = tl.multiple_of(tl.load(TABLE + 4 * LAYERS + layer).to(tl.pointer_type(BETA_DTYPE)), 16)
    STATE = tl.multiple_of(tl.load(TABLE + 5 * LAYERS + layer).to(tl.pointer_type(STATE_DTYPE)), 16)
    kh = head // (HV // H)
    keys = tl.arange(0, BK)
    values = vb * BV + tl.arange(0, BV)
    state_index = bh * DK * DV + keys[:, None] * DV + values[None, :]
    mask = (keys[:, None] < DK) & (values[None, :] < DV)
    state = tl.load(INITIAL + state_index, mask, 0).to(tl.float32)
    for token in range(PREFIX):
        k = tl.load(LOG_K + ((batch * CAPACITY + token) * H + kh) * DK + keys, keys < DK, 0).to(tl.float32)
        v = tl.load(LOG_V + ((batch * CAPACITY + token) * HV + head) * DV + values, values < DV, 0).to(tl.float32)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        g = tl.load(LOG_G + (batch * CAPACITY + token) * HV + head).to(tl.float32)
        beta = tl.load(LOG_BETA + (batch * CAPACITY + token) * HV + head).to(tl.float32)
        state *= exp(g)
        v = beta * (v - tl.sum(state * k[:, None], 0))
        state += k[:, None] * v
    tl.store(STATE + state_index, state, mask)


def replay_batched_prefix(batch, processed_tokens, verified_tokens):
    """Replay all uniform layers in one launch without changing carry precision."""
    workspace = batch.workspaces[0]
    if not 1 <= processed_tokens <= verified_tokens <= workspace.max_tokens:
        raise ValueError("Commit prefix is outside this verification")
    if processed_tokens == verified_tokens:
        return
    state = batch.states[0]
    batches, value_heads, dim_k, dim_v = state.shape
    dtypes = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
    _replay_batched_kernel[(triton.cdiv(dim_v, 8), len(batch.states) * batches * value_heads)](
        batch.pointers, processed_tokens, workspace.max_tokens,
        workspace.keys.shape[2], value_heads, dim_k, dim_v, batches, len(batch.states),
        triton.next_power_of_2(dim_k), 8, dtypes[state.dtype], dtypes[workspace.keys.dtype],
        dtypes[workspace.values.dtype], dtypes[workspace.beta.dtype], num_warps=1, num_stages=3,
    )


# Tool-only module: registration is silent for in-memory and disk cache hits.
from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
