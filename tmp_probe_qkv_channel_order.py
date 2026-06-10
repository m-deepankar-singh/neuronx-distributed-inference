#!/usr/bin/env python3
"""Print QKV kernel output-channel order for deterministic projection weights."""

import argparse
import json
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
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-q-heads", type=int, default=2)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    args = parser.parse_args()

    os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 2")
    device = xm.xla_device()
    packed_q_heads = args.num_q_heads * 2
    width = (packed_q_heads + 2 * args.num_kv_heads) * args.head_dim

    hidden = torch.zeros((1, args.seq_len, args.hidden_size), dtype=torch.bfloat16)
    hidden[:, :, 0] = 1.0
    weight = torch.zeros((args.hidden_size, width), dtype=torch.bfloat16)
    channel_values = torch.arange(width, dtype=torch.float32).remainder(1024)
    weight[0, :] = channel_values.to(torch.bfloat16)

    kernel = nki.jit(qkv)
    out = kernel[args.logical_nc_config](
        input=hidden.to(device),
        fused_qkv_weights=weight.to(device),
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
        num_q_heads=packed_q_heads,
        num_kv_heads=args.num_kv_heads,
    )
    xm.mark_step()
    values = out.cpu().float()[0, 0, :]
    expected = channel_values
    mismatches = (values != expected).nonzero().flatten()
    result = {
        "width": width,
        "packed_q_heads": packed_q_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "first_64_actual": values[:64].tolist(),
        "first_64_expected": expected[:64].tolist(),
        "last_64_actual": values[-64:].tolist(),
        "mismatch_count": int(mismatches.numel()),
        "first_mismatches": [
            {
                "index": int(i),
                "actual": float(values[i]),
                "expected": float(expected[i]),
            }
            for i in mismatches[:32]
        ],
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
