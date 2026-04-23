"""Reference PyTorch implementation of Gated DeltaNet forward pass.

Matches fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule.
Recurrent (one-token-at-a-time) form -- easy to verify, maps to NKI tile model.

Shapes (FLA convention):
    q:    [B, T, H,  K]   queries
    k:    [B, T, H,  K]   keys
    v:    [B, T, HV, V]   values (HV >= H; GVA when HV > H)
    g:    [B, T, HV]      log-space forget gate
    beta: [B, T, HV]      update gate scalar in (0, 1)

Algorithm per timestep t:
    decay    = exp(g_t)                      # scalar per head
    S        = decay * S                     # decay state
    v_hat    = k_t @ S                       # read state
    delta    = beta_t * (v_t - v_hat)        # error-weighted update
    S       += outer(k_t, delta)             # rank-1 write
    out_t    = (q_t * scale) @ S             # produce output
"""
from __future__ import annotations

import math

import torch


def ref_gated_delta_rule_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    assert HV % H == 0, "HV must be divisible by H (GVA)"
    repeat = HV // H

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    # Do math in fp32 for the accumulator to avoid bf16 drift across T steps.
    q32 = q.float()
    k32 = k.float()
    v32 = v.float()
    g32 = g.float()
    beta32 = beta.float()

    # Broadcast q/k heads H -> HV for GVA.
    if repeat > 1:
        q32 = q32.repeat_interleave(repeat, dim=2)
        k32 = k32.repeat_interleave(repeat, dim=2)

    # State S[b, h] is a (K, V) matrix per (batch, head).
    if initial_state is not None:
        S = initial_state.float().clone()
    else:
        S = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)

    out = torch.empty(B, T, HV, V, dtype=torch.float32, device=q.device)

    for t in range(T):
        q_t = q32[:, t]
        k_t = k32[:, t]
        v_t = v32[:, t]
        decay = torch.exp(g32[:, t]).unsqueeze(-1).unsqueeze(-1)
        beta_t = beta32[:, t].unsqueeze(-1)

        S = S * decay
        v_hat = torch.einsum("bhk,bhkv->bhv", k_t, S)
        delta = beta_t * (v_t - v_hat)
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q_t * scale, S)

    out = out.to(q.dtype)
    final_state = S if output_final_state else None
    return out, final_state
