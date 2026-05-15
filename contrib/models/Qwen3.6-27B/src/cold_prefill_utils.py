# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light helpers for Qwen3.6 cold-prefill guardrails."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def gdn_cte_kernel_from_env(env: dict[str, str] | None = None) -> str:
    env = env or os.environ
    if env.get("USE_PYTORCH_CHUNK") == "1":
        return "pytorch_chunk"
    if env.get("USE_NKI_FUSED", "1") != "0":
        return "fused_initial_state"
    if env.get("USE_NKI_CHUNKED") == "1":
        return "nki_chunked"
    if env.get("USE_NKI") == "1":
        return "nki_recurrent"
    return "fused_initial_state"


def hybrid_restore_mask_is_empty(hybrid_restore_mask) -> bool:
    if hybrid_restore_mask is None:
        return True
    if not hasattr(hybrid_restore_mask, "numel") or hybrid_restore_mask.numel() == 0:
        return True
    return not bool(hybrid_restore_mask.to(torch.bool).any().item())


def safe_cold_zero_conv_fast_path(
    config,
    position_ids,
    recurrent_state_cache,
    conv_state_cache,
    hybrid_restore_mask=None,
    hybrid_restore_prefix_lens=None,
) -> bool:
    if not getattr(config, "use_cold_zero_conv_fast_path", False):
        return False
    if position_ids is None:
        return False
    if recurrent_state_cache is None or conv_state_cache is None:
        return False
    if getattr(config, "use_hybrid_apc_manager", False) and torch.jit.is_tracing():
        return False
    if not bool((position_ids[:, :1].long() == 0).all().item()):
        return False
    if (
        hybrid_restore_prefix_lens is not None
        and hasattr(hybrid_restore_prefix_lens, "numel")
        and hybrid_restore_prefix_lens.numel() > 0
        and bool((hybrid_restore_prefix_lens.long() > 0).any().item())
    ):
        return False
    if getattr(config, "use_hybrid_apc_manager", False):
        if hybrid_restore_mask is None:
            return False
        if not hybrid_restore_mask_is_empty(hybrid_restore_mask):
            return False
    return True


def depthwise_causal_conv1d_from_zero(mixed, conv_weight):
    seq_len = mixed.shape[-1]
    kernel_size = conv_weight.shape[-1]
    return F.conv1d(
        mixed,
        conv_weight,
        bias=None,
        padding=kernel_size - 1,
        groups=mixed.shape[1],
    )[:, :, :seq_len]


def depthwise_causal_conv1d_with_state(mixed, conv_weight, conv_state):
    seq_len = mixed.shape[-1]
    conv_out = torch.zeros_like(mixed)
    weight = conv_weight.squeeze(1)
    conv_input = torch.cat([conv_state, mixed], dim=-1)
    for k in range(conv_weight.shape[-1]):
        conv_out = (
            conv_out
            + weight[:, k].unsqueeze(0).unsqueeze(-1)
            * conv_input[:, :, k : k + seq_len]
        )
    return conv_out


def validate_text_only_cte_vision_inputs(
    config,
    is_for_context_encoding: bool,
    vision_embeddings,
    vision_mask,
):
    if not (
        is_for_context_encoding
        and getattr(config, "use_text_only_cte_inputs", True)
    ):
        return
    assert vision_embeddings is None or vision_embeddings.numel() == 0, (
        "text-only CTE requires empty vision_embeddings; use a multimodal "
        "artifact for dense dummy vision tensors"
    )
    assert vision_mask is None or vision_mask.numel() == 0, (
        "text-only CTE requires empty vision_mask; use a multimodal artifact "
        "for dense dummy vision tensors"
    )


def prepare_cte_attention_mask(
    attention_mask,
    *,
    is_for_context_encoding: bool,
    seq_length: int,
    use_compact_cte_attention_mask: bool,
    use_neuron_cte_attention: bool,
):
    if (
        is_for_context_encoding
        and seq_length > 2048
        and not (use_compact_cte_attention_mask or use_neuron_cte_attention)
    ):
        raise ValueError(
            "Dense CTE attention masks are disabled for seq_length > 2048; "
            "enable compact CTE attention masks or Neuron CTE attention"
        )
    if not (
        attention_mask is not None
        and attention_mask.ndim == 2
        and is_for_context_encoding
        and not use_compact_cte_attention_mask
        and not use_neuron_cte_attention
    ):
        return attention_mask

    causal = torch.ones(
        (seq_length, seq_length),
        dtype=torch.bool,
        device=attention_mask.device,
    ).tril()
    padding_4d = attention_mask[:, None, None, :].to(torch.bool)
    return (causal[None, None, :, :] & padding_4d).to(attention_mask.dtype)
