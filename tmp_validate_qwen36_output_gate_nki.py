#!/usr/bin/env python3
"""Validate Qwen3.6 output-gate projection through the NKILib QKV CTE kernel."""

import argparse
import json
import math
import os
from pathlib import Path

import nki
import torch
from nkilib.core.qkv.qkv import qkv
from nkilib.core.utils.common_types import QKVOutputLayout, QuantizationType
from torch_xla.core import xla_model as xm


qkv_kernel = nki.jit(qkv)


def _run_case(*, quantized: bool, seq_len: int, hidden: int, output: int, logical_nc: int):
    torch.manual_seed(1234 + int(quantized))
    device = xm.xla_device()
    dtype = torch.bfloat16

    x_cpu = torch.randn((1, seq_len, hidden), dtype=dtype) / math.sqrt(hidden)
    w_cpu = torch.randn((hidden, output), dtype=dtype) / math.sqrt(hidden)
    kwargs = {
        "input": x_cpu.to(device),
        "fused_qkv_weights": w_cpu.to(device),
        "output_layout": QKVOutputLayout.BSD,
        "bias": None,
        "quantization_type": QuantizationType.NONE,
    }
    ref_w = w_cpu.float()
    if quantized:
        # Match the preprocessed ColumnParallelLinear layout used by the
        # production path: weights are [hidden, output] and scales are broadcast
        # to [128, output].  Unit scale makes the CPU reference direct.
        w_fp8_cpu = w_cpu.to(torch.float8_e4m3fn)
        scales_cpu = torch.ones((128, output), dtype=torch.float32)
        kwargs.update(
            {
                "fused_qkv_weights": w_fp8_cpu.to(device),
                "quantization_type": QuantizationType.ROW,
                "qkv_w_scale": scales_cpu.to(device),
                "qkv_in_scale": None,
            }
        )
        ref_w = w_fp8_cpu.float()

    out = qkv_kernel[logical_nc](**kwargs)
    xm.mark_step()
    actual = out.cpu().float()
    expected = torch.matmul(x_cpu.float(), ref_w)
    diff = (expected - actual).abs()
    return {
        "quantized": quantized,
        "seq_len": seq_len,
        "hidden": hidden,
        "output": output,
        "shape": list(actual.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "passed": bool(torch.allclose(expected, actual, atol=0.35, rtol=0.08)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--output", type=int, default=384)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument(
        "--bf16-only",
        action="store_true",
        help=(
            "Run only the BF16 projection case. Standalone FP8 validation needs "
            "the same internal compiler flag injection used by full NxDI model "
            "compiles, so this is useful for shape/path checks."
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 2")
    results = [
        _run_case(
            quantized=False,
            seq_len=args.seq_len,
            hidden=args.hidden,
            output=args.output,
            logical_nc=args.logical_nc_config,
        )
    ]
    if not args.bf16_only:
        results.append(
            _run_case(
                quantized=True,
                seq_len=args.seq_len,
                hidden=args.hidden,
                output=args.output,
                logical_nc=args.logical_nc_config,
            )
        )
    payload = {"results": results, "passed": all(row["passed"] for row in results)}
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
