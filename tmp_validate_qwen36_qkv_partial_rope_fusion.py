#!/usr/bin/env python3
"""Validate NxDI QKV fused-RoPE semantics for Qwen3.6 partial RoPE.

This is a small hardware probe, not a full model compile.  It checks whether
the shared QKV NKI kernel can safely fuse RoPE when Qwen3.6 rotates only the
first 64 of 256 head dimensions.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch

import nki
from nkilib.core.qkv.qkv import qkv
from nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType
from torch_xla.core import xla_model as xm


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _partial_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rope = q[..., :rope_dim]
    k_rope = k[..., :rope_dim]
    q_pass = q[..., rope_dim:]
    k_pass = k[..., rope_dim:]
    q_rope = (q_rope * cos) + (_rotate_half(q_rope) * sin)
    k_rope = (k_rope * cos) + (_rotate_half(k_rope) * sin)
    return torch.cat([q_rope, q_pass], dim=-1), torch.cat([k_rope, k_pass], dim=-1)


def _split_bhsd(qkv_out: torch.Tensor, num_q_heads: int, num_kv_heads: int, head_dim: int):
    q_end = num_q_heads * head_dim
    k_end = q_end + num_kv_heads * head_dim
    q, k, v = torch.tensor_split(qkv_out, (q_end, k_end), dim=2)
    bsz, seq_len, _ = q.shape
    q = q.view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2).contiguous()
    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    return q, k, v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-q-heads", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--pad-rope-to-head-dim", action="store_true")
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 2")
    torch.manual_seed(0)
    device = xm.xla_device()
    dtype = torch.bfloat16

    bsz = 1
    fused_qkv_size = (args.num_q_heads + 2 * args.num_kv_heads) * args.head_dim
    hidden_cpu = torch.randn((bsz, args.seq_len, args.hidden_size), dtype=dtype)
    weight_cpu = torch.randn((args.hidden_size, fused_qkv_size), dtype=dtype) / math.sqrt(args.hidden_size)
    pos = torch.arange(args.seq_len, dtype=torch.float32)
    inv_freq = 1.0 / (10000000.0 ** (torch.arange(0, args.rope_dim, 2, dtype=torch.float32) / args.rope_dim))
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(0).to(dtype)
    cos_cpu = emb.cos()
    sin_cpu = emb.sin()

    hidden = hidden_cpu.to(device)
    weight = weight_cpu.to(device)
    if args.pad_rope_to_head_dim:
        cos_for_kernel_cpu = torch.ones((bsz, args.seq_len, args.head_dim), dtype=dtype)
        sin_for_kernel_cpu = torch.zeros((bsz, args.seq_len, args.head_dim), dtype=dtype)
        cos_for_kernel_cpu[..., : args.rope_dim] = cos_cpu
        sin_for_kernel_cpu[..., : args.rope_dim] = sin_cpu
    else:
        cos_for_kernel_cpu = cos_cpu
        sin_for_kernel_cpu = sin_cpu
    cos = cos_for_kernel_cpu.to(device)
    sin = sin_for_kernel_cpu.to(device)

    qkv_kernel = nki.jit(qkv)
    common = dict(
        input=hidden,
        fused_qkv_weights=weight,
        output_layout=QKVOutputLayout.BSD,
        bias=None,
        fused_residual_add=False,
        mlp_prev=None,
        attention_prev=None,
        fused_norm_type=NormType.NO_NORM,
        gamma_norm_weights=None,
        norm_eps=1e-6,
        quantization_type=QuantizationType.NONE,
        qkv_w_scale=None,
        qkv_in_scale=None,
        d_head=args.head_dim,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
    )
    unfused = qkv_kernel[args.logical_nc_config](fused_rope=False, cos_cache=None, sin_cache=None, **common)
    fused = qkv_kernel[args.logical_nc_config](fused_rope=True, cos_cache=cos, sin_cache=sin, **common)
    xm.mark_step()

    q_ref, k_ref, v_ref = _split_bhsd(unfused.cpu(), args.num_q_heads, args.num_kv_heads, args.head_dim)
    q_ref, k_ref = _partial_rope(q_ref.float(), k_ref.float(), cos_cpu.float(), sin_cpu.float(), args.rope_dim)
    v_ref = v_ref.float()
    q_fused, k_fused, v_fused = _split_bhsd(fused.cpu(), args.num_q_heads, args.num_kv_heads, args.head_dim)
    q_fused = q_fused.float()
    k_fused = k_fused.float()
    v_fused = v_fused.float()

    result = {
        "seq_len": args.seq_len,
        "hidden_size": args.hidden_size,
        "num_q_heads": args.num_q_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "rope_dim": args.rope_dim,
        "pad_rope_to_head_dim": args.pad_rope_to_head_dim,
        "max_abs_q": float((q_ref - q_fused).abs().max()),
        "max_abs_k": float((k_ref - k_fused).abs().max()),
        "max_abs_v": float((v_ref - v_fused).abs().max()),
        "q_pass": False,
        "k_pass": False,
        "v_pass": False,
    }
    result["q_pass"] = bool(torch.allclose(q_ref, q_fused, atol=args.atol, rtol=args.rtol))
    result["k_pass"] = bool(torch.allclose(k_ref, k_fused, atol=args.atol, rtol=args.rtol))
    result["v_pass"] = bool(torch.allclose(v_ref, v_fused, atol=args.atol, rtol=args.rtol))
    result["passed"] = result["q_pass"] and result["k_pass"] and result["v_pass"]

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
