#!/usr/bin/env python3
"""Probe whether the QKV NKI kernel can carry Qwen output-gate channels.

Qwen3.6 full-attention layers split HF q_proj into query and output-gate
channels.  The model currently runs Wqkv and output_gate_proj as separate
projections over the same normalized hidden states.  This probe packs
[Q | gate | K | V] and asks the existing QKV kernel to treat Q+gate as a
wider Q section, then compares the split outputs against a CPU reference.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-q-heads", type=int, default=6)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--rtol", type=float, default=8e-2)
    args = parser.parse_args()

    os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 2")
    torch.manual_seed(0)
    device = xm.xla_device()
    dtype = torch.bfloat16

    bsz = 1
    q_width = args.num_q_heads * args.head_dim
    kv_width = args.num_kv_heads * args.head_dim
    packed_q_heads = args.num_q_heads * 2
    packed_width = (packed_q_heads + 2 * args.num_kv_heads) * args.head_dim

    hidden_cpu = torch.randn(
        (bsz, args.seq_len, args.hidden_size),
        dtype=dtype,
    ) / math.sqrt(args.hidden_size)
    q_w_cpu = torch.randn((args.hidden_size, q_width), dtype=dtype) / math.sqrt(
        args.hidden_size
    )
    gate_w_cpu = torch.randn((args.hidden_size, q_width), dtype=dtype) / math.sqrt(
        args.hidden_size
    )
    k_w_cpu = torch.randn((args.hidden_size, kv_width), dtype=dtype) / math.sqrt(
        args.hidden_size
    )
    v_w_cpu = torch.randn((args.hidden_size, kv_width), dtype=dtype) / math.sqrt(
        args.hidden_size
    )
    packed_w_cpu = torch.cat([q_w_cpu, gate_w_cpu, k_w_cpu, v_w_cpu], dim=1)
    assert packed_w_cpu.shape == (args.hidden_size, packed_width)

    qkv_kernel = nki.jit(qkv)
    common = dict(
        input=hidden_cpu.to(device),
        output_layout=QKVOutputLayout.BSD,
        bias=None,
        fused_residual_add=False,
        mlp_prev=None,
        attention_prev=None,
        fused_norm_type=NormType.NO_NORM,
        gamma_norm_weights=None,
        norm_eps=1e-6,
        fused_rope=False,
        cos_cache=None,
        sin_cache=None,
        quantization_type=QuantizationType.NONE,
        qkv_w_scale=None,
        qkv_in_scale=None,
        d_head=args.head_dim,
        num_kv_heads=args.num_kv_heads,
    )
    packed = qkv_kernel[args.logical_nc_config](
        fused_qkv_weights=packed_w_cpu.to(device),
        num_q_heads=packed_q_heads,
        **common,
    )

    xm.mark_step()
    packed_cpu = packed.cpu().float()
    expected_packed_cpu = torch.matmul(
        hidden_cpu.float(),
        packed_w_cpu.float(),
    )

    q_end = q_width
    gate_end = q_end + q_width
    k_end = gate_end + kv_width
    actual_q = packed_cpu[:, :, :q_end]
    actual_gate = packed_cpu[:, :, q_end:gate_end]
    actual_k = packed_cpu[:, :, gate_end:k_end]
    actual_v = packed_cpu[:, :, k_end:]

    expected_q = expected_packed_cpu[:, :, :q_end]
    expected_gate = expected_packed_cpu[:, :, q_end:gate_end]
    expected_k = expected_packed_cpu[:, :, gate_end:k_end]
    expected_v = expected_packed_cpu[:, :, k_end:]

    result = {
        "seq_len": args.seq_len,
        "hidden_size": args.hidden_size,
        "num_q_heads": args.num_q_heads,
        "packed_q_heads": packed_q_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "packed_width": packed_width,
        "q_max_abs": float((expected_q - actual_q).abs().max()),
        "gate_max_abs": float((expected_gate - actual_gate).abs().max()),
        "k_max_abs": float((expected_k - actual_k).abs().max()),
        "v_max_abs": float((expected_v - actual_v).abs().max()),
        "q_mean_abs": float((expected_q - actual_q).abs().mean()),
        "gate_mean_abs": float((expected_gate - actual_gate).abs().mean()),
        "k_mean_abs": float((expected_k - actual_k).abs().mean()),
        "v_mean_abs": float((expected_v - actual_v).abs().mean()),
    }
    result["q_pass"] = bool(
        torch.allclose(expected_q, actual_q, atol=args.atol, rtol=args.rtol)
    )
    result["gate_pass"] = bool(
        torch.allclose(expected_gate, actual_gate, atol=args.atol, rtol=args.rtol)
    )
    result["k_pass"] = bool(
        torch.allclose(expected_k, actual_k, atol=args.atol, rtol=args.rtol)
    )
    result["v_pass"] = bool(
        torch.allclose(expected_v, actual_v, atol=args.atol, rtol=args.rtol)
    )
    result["passed"] = all(
        result[key] for key in ("q_pass", "gate_pass", "k_pass", "v_pass")
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
