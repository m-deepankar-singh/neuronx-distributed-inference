#!/usr/bin/env python3
"""Validate and optionally inspect/profile the fused Qwen DeltaNet NKI kernel.

The CPU reference stays off the XLA device so the generated NEFFs are from the
NKI kernel under test, not from reference PyTorch ops.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


P_MAX = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate/profile deltanet_fused_chunked_fwd against CPU math."
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--target", default="trn2")
    parser.add_argument("--lnc", type=int, default=1)
    parser.add_argument("--visible-cores", default="0")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--dge", action="store_true")
    parser.add_argument(
        "--inspect-dir",
        default="/mnt/trainium_artifacts/profiles/deltanet_fused_isolated",
    )
    parser.add_argument("--atol", type=float, default=3.0e-2)
    parser.add_argument("--rtol", type=float, default=3.0e-2)
    parser.add_argument("--value-scale", type=float, default=0.05)
    parser.add_argument("--state-scale", type=float, default=0.01)
    parser.add_argument("--gate-scale", type=float, default=0.01)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> Path:
    if args.seq_len <= 0 or args.seq_len % P_MAX != 0:
        raise ValueError("--seq-len must be a positive multiple of 128")
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", args.target)
    os.environ.setdefault("NEURON_CC_FLAGS", f"--target {args.target} --lnc {args.lnc}")
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", args.visible_cores)

    inspect_dir = Path(args.inspect_dir).expanduser().resolve()
    if args.inspect:
        inspect_dir.mkdir(parents=True, exist_ok=True)
        os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
        os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
        os.environ["NEURON_RT_INSPECT_SYSTEM_PROFILE"] = "0"
        os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = str(inspect_dir)
        os.environ["XLA_IR_DEBUG"] = "1"
        os.environ["XLA_HLO_DEBUG"] = "1"
        os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
    if args.dge:
        os.environ["NEURON_RT_ENABLE_DGE_NOTIFICATIONS"] = "1"
    return inspect_dir


def add_qwen_to_path() -> None:
    script_path = Path(__file__).resolve()
    qwen_root = script_path.parents[1]
    sys.path.insert(0, str(qwen_root))


def load_fused_kernel():
    kernel_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "nki_kernels"
        / "nki_deltanet_fused.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen36_nki_deltanet_fused_under_test",
        kernel_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.deltanet_fused_chunked_fwd


def make_inputs(torch: Any, args: argparse.Namespace) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    def randn(shape: tuple[int, ...], scale: float) -> Any:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * scale

    query = randn((args.seq_len, P_MAX), args.value_scale)
    key = randn((args.seq_len, P_MAX), args.value_scale)
    value = randn((args.seq_len, P_MAX), args.value_scale)
    state_in = randn((P_MAX, P_MAX), args.state_scale)

    query = torch.nn.functional.normalize(query, p=2, dim=-1) / math.sqrt(P_MAX)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)

    beta = torch.sigmoid(randn((args.seq_len, 1), 1.0))
    g_raw = -torch.nn.functional.softplus(randn((args.seq_len, 1), 1.0))
    g_raw = g_raw * args.gate_scale

    lower_mask = torch.tril(torch.ones((P_MAX, P_MAX), dtype=torch.float32), diagonal=-1)
    lower_mask_diag = torch.tril(torch.ones((P_MAX, P_MAX), dtype=torch.float32))
    identity = torch.eye(P_MAX, dtype=torch.float32)

    return {
        "query": query.contiguous(),
        "key": key.contiguous(),
        "value": value.contiguous(),
        "g_raw": g_raw.contiguous(),
        "beta": beta.contiguous(),
        "state_in": state_in.contiguous(),
        "lower_mask": lower_mask.contiguous(),
        "identity": identity.contiguous(),
        "lower_mask_diag": lower_mask_diag.contiguous(),
    }


def stable_causal_decay(torch: Any, gc: Any, mask: Any) -> Any:
    """Compute exp(gc[i] - gc[j]) only where the causal mask is active."""
    diff = gc - gc.T
    masked_diff = torch.where(mask.bool(), diff, torch.zeros_like(diff))
    return torch.exp(masked_diff) * mask


def reference_math(torch: Any, inputs: dict[str, Any]) -> tuple[Any, Any]:
    lower = inputs["lower_mask"]
    lower_diag = inputs["lower_mask_diag"]
    eye = inputs["identity"]
    state = inputs["state_in"].clone()
    outputs = []

    for start in range(0, inputs["query"].shape[0], P_MAX):
        end = start + P_MAX
        q = inputs["query"][start:end]
        k = inputs["key"][start:end]
        v = inputs["value"][start:end]
        g = inputs["g_raw"][start:end]
        beta = inputs["beta"][start:end]

        gc = torch.cumsum(g, dim=0)
        gl = gc[-1:]
        k_beta = k * beta
        v_beta = v * beta

        decay_strict = stable_causal_decay(torch, gc, lower)
        decay_diag = stable_causal_decay(torch, gc, lower_diag)

        qk_beta = k_beta @ k.T
        a_mat = -(qk_beta * decay_strict) * lower

        lhs = eye - a_mat

        exp_gc = torch.exp(gc)
        solve_rhs = v_beta - ((k_beta * exp_gc) @ state)
        v_new = torch.linalg.solve_triangular(lhs, solve_rhs, upper=False)
        attn_intra = (q @ k.T) * decay_diag

        chunk_out = ((q * exp_gc) @ state) + (attn_intra @ v_new)
        outputs.append(chunk_out)

        k_raw_decay = k * torch.exp(gl - gc)
        state = (state * torch.exp(gl)) + (k_raw_decay.T @ v_new)

    return torch.cat(outputs, dim=0).contiguous(), state.contiguous()


def tensor_metrics(torch: Any, actual: Any, expected: Any) -> dict[str, float | bool]:
    diff = actual - expected
    expected_norm = torch.linalg.vector_norm(expected).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    actual_flat = actual.reshape(-1).to(torch.float64)
    expected_flat = expected.reshape(-1).to(torch.float64)
    denom = torch.linalg.vector_norm(actual_flat) * torch.linalg.vector_norm(expected_flat)
    cosine = (
        float(torch.dot(actual_flat, expected_flat) / denom)
        if denom.item() != 0.0
        else float("nan")
    )
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "max_abs": float(torch.max(torch.abs(diff)).item()),
        "mean_abs": float(torch.mean(torch.abs(diff)).item()),
        "diff_norm": float(diff_norm),
        "expected_norm": float(expected_norm),
        "relative_norm": float(diff_norm / max(expected_norm, 1.0e-12)),
        "cosine": cosine,
    }


def main() -> int:
    args = parse_args()
    inspect_dir = configure_environment(args)
    add_qwen_to_path()

    import torch
    import torch_xla.core.xla_model as xm

    deltanet_fused_chunked_fwd = load_fused_kernel()

    inputs = make_inputs(torch, args)
    ref_out, ref_state = reference_math(torch, inputs)

    device = xm.xla_device()
    xla_inputs = {name: tensor.to(device=device) for name, tensor in inputs.items()}

    out_cpu = state_cpu = None
    for _ in range(args.runs):
        out_dev, state_dev = deltanet_fused_chunked_fwd(
            xla_inputs["query"],
            xla_inputs["key"],
            xla_inputs["value"],
            xla_inputs["g_raw"],
            xla_inputs["beta"],
            xla_inputs["state_in"],
            xla_inputs["lower_mask"],
            xla_inputs["identity"],
            xla_inputs["lower_mask_diag"],
        )
        xm.mark_step()
        out_cpu = out_dev.detach().cpu().float()
        state_cpu = state_dev.detach().cpu().float()

    assert out_cpu is not None
    assert state_cpu is not None

    output_close = torch.allclose(out_cpu, ref_out, atol=args.atol, rtol=args.rtol)
    state_close = torch.allclose(state_cpu, ref_state, atol=args.atol, rtol=args.rtol)
    output_finite = bool(torch.isfinite(out_cpu).all().item())
    state_finite = bool(torch.isfinite(state_cpu).all().item())
    passed = bool(output_close and state_close and output_finite and state_finite)

    result = {
        "passed": passed,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "runs": args.runs,
        "atol": args.atol,
        "rtol": args.rtol,
        "inspect": args.inspect,
        "dge": args.dge,
        "output_finite": output_finite,
        "state_finite": state_finite,
        "inspect_dir": str(inspect_dir),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_INSPECT_ENABLE",
                "NEURON_RT_ENABLE_DGE_NOTIFICATIONS",
            )
        },
        "nki_vs_reference": {
            "output": tensor_metrics(torch, out_cpu, ref_out),
            "state": tensor_metrics(torch, state_cpu, ref_state),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.fail_on_mismatch and not passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
