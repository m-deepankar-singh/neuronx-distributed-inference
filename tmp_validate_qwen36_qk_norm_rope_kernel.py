#!/usr/bin/env python3
"""Validate Qwen3.6 Q/K RMSNorm + partial-RoPE NKI kernel.

This is a small hardware probe.  It compares the custom kernel against the
current PyTorch reference path:
  B,S,H*D -> view(B,S,H,D) -> RMSNorm(D) -> transpose(B,H,S,D)
  -> partial RoPE over first 64 of 256 head dimensions.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch_xla.core import xla_model as xm

from src.nki_kernels.qwen_qk_norm_rope import qwen_qk_norm_partial_rope_kernel


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rmsnorm_heads(x: torch.Tensor, gamma: torch.Tensor, heads: int, head_dim: int, eps: float):
    batch, seq, _width = x.shape
    x = x.view(batch, seq, heads, head_dim).float()
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x = x * inv_rms * gamma.float().view(1, 1, 1, head_dim)
    return x.transpose(1, 2).contiguous()


def _partial_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int):
    cos = cos.unsqueeze(1).float()
    sin = sin.unsqueeze(1).float()
    q_rope = q[..., :rope_dim]
    q_pass = q[..., rope_dim:]
    k_rope = k[..., :rope_dim]
    k_pass = k[..., rope_dim:]
    q_rope = (q_rope * cos) + (_rotate_half(q_rope) * sin)
    k_rope = (k_rope * cos) + (_rotate_half(k_rope) * sin)
    return torch.cat([q_rope, q_pass], dim=-1), torch.cat([k_rope, k_pass], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-q-heads", type=int, default=6)
    parser.add_argument("--num-k-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--rtol", type=float, default=8e-2)
    args = parser.parse_args()

    if args.head_dim != 256 or args.rope_dim != 64:
        raise ValueError("probe expects Qwen3.6 head_dim=256 and rope_dim=64")

    os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 2")
    torch.manual_seed(0)
    device = xm.xla_device()
    dtype = torch.bfloat16
    batch = 1

    q_cpu = torch.randn(
        (batch, args.seq_len, args.num_q_heads * args.head_dim),
        dtype=dtype,
    ) / math.sqrt(args.head_dim)
    k_cpu = torch.randn(
        (batch, args.seq_len, args.num_k_heads * args.head_dim),
        dtype=dtype,
    ) / math.sqrt(args.head_dim)
    q_gamma_cpu = torch.randn((args.head_dim,), dtype=dtype)
    k_gamma_cpu = torch.randn((args.head_dim,), dtype=dtype)

    pos = torch.arange(args.seq_len, dtype=torch.float32)
    inv_freq = 1.0 / (
        10000000.0
        ** (torch.arange(0, args.rope_dim, 2, dtype=torch.float32) / args.rope_dim)
    )
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(0)
    cos_cpu = emb.cos().to(dtype)
    sin_cpu = emb.sin().to(dtype)

    q_ref = _rmsnorm_heads(q_cpu, q_gamma_cpu, args.num_q_heads, args.head_dim, args.eps)
    k_ref = _rmsnorm_heads(k_cpu, k_gamma_cpu, args.num_k_heads, args.head_dim, args.eps)
    q_ref, k_ref = _partial_rope(
        q_ref,
        k_ref,
        cos_cpu,
        sin_cpu,
        args.rope_dim,
    )

    q_out, k_out = qwen_qk_norm_partial_rope_kernel[args.logical_nc_config](
        q_cpu.to(device),
        k_cpu.to(device),
        q_gamma_cpu.to(device),
        k_gamma_cpu.to(device),
        cos_cpu.to(device),
        sin_cpu.to(device),
        args.eps,
    )
    xm.mark_step()
    q_actual = q_out.cpu().float()
    k_actual = k_out.cpu().float()

    result = {
        "seq_len": args.seq_len,
        "num_q_heads": args.num_q_heads,
        "num_k_heads": args.num_k_heads,
        "head_dim": args.head_dim,
        "rope_dim": args.rope_dim,
        "q_max_abs": float((q_ref.float() - q_actual).abs().max()),
        "k_max_abs": float((k_ref.float() - k_actual).abs().max()),
    }
    result["q_pass"] = bool(torch.allclose(q_ref.float(), q_actual, atol=args.atol, rtol=args.rtol))
    result["k_pass"] = bool(torch.allclose(k_ref.float(), k_actual, atol=args.atol, rtol=args.rtol))
    result["passed"] = result["q_pass"] and result["k_pass"]

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
