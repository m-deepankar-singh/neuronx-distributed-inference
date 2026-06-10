#!/usr/bin/env python3
"""Validate Qwen gated output projection NKI kernel on Neuron hardware."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch_xla.core import xla_model as xm

from neuronx_distributed_inference.modules.attention.nki_kernels.qwen_gated_output_projection import (
    qwen_gated_output_projection_cte,
)


def _row_quant_reference(
    attention: torch.Tensor,
    gate: torch.Tensor,
    weight_fp8: torch.Tensor,
) -> torch.Tensor:
    gated = attention.float() * torch.sigmoid(gate.float())
    batch, seq_len, num_heads, head_dim = gated.shape
    flat = gated.reshape(batch, seq_len, num_heads * head_dim)
    absmax = flat.abs().amax(dim=-1, keepdim=True)
    dequant_scale = torch.maximum(
        absmax * (1.0 / 240.0),
        torch.tensor(1.0e-5, dtype=torch.float32),
    )
    quantized = torch.clamp(flat / dequant_scale, -240.0, 240.0).to(torch.float8_e4m3fn)
    return torch.matmul(quantized.float(), weight_fp8.float()) * dequant_scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--atol", type=float, default=0.7)
    parser.add_argument("--rtol", type=float, default=0.12)
    args = parser.parse_args()

    os.environ.setdefault(
        "NEURON_CC_FLAGS",
        (
            "--target trn2 --lnc 2 "
            "--internal-hlo2tensorizer-options='"
            "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3 --verify-hlo=true'"
        ),
    )
    os.environ.setdefault("XLA_HANDLE_SPECIAL_SCALAR", "1")
    os.environ.setdefault("UNSAFE_FP8FNCAST", "1")
    torch.manual_seed(20260603)
    device = xm.xla_device()
    dtype = torch.bfloat16
    batch = 1
    nd = args.num_heads * args.head_dim

    attention_cpu = torch.randn(
        (batch, args.seq_len, args.num_heads, args.head_dim),
        dtype=dtype,
    ) / math.sqrt(args.head_dim)
    gate_cpu = torch.randn(
        (batch, args.seq_len, args.num_heads, args.head_dim),
        dtype=dtype,
    )
    weight_cpu = (
        torch.randn((nd, args.hidden_size), dtype=dtype) / math.sqrt(nd)
    ).to(torch.float8_e4m3fn)
    weight_scales_cpu = torch.ones((128, args.hidden_size), dtype=torch.float32)

    actual = qwen_gated_output_projection_cte[args.logical_nc_config](
        attention=attention_cpu.to(device),
        gate=gate_cpu.to(device),
        weight=weight_cpu.to(device),
        bias=None,
        weight_scales=weight_scales_cpu.to(device),
    )
    xm.mark_step()

    actual_cpu = actual.cpu().float()
    expected_cpu = _row_quant_reference(attention_cpu, gate_cpu, weight_cpu)
    diff = (expected_cpu - actual_cpu).abs()
    result = {
        "seq_len": args.seq_len,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "hidden_size": args.hidden_size,
        "shape": list(actual_cpu.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "passed": bool(
            torch.allclose(expected_cpu, actual_cpu, atol=args.atol, rtol=args.rtol)
        ),
    }

    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
