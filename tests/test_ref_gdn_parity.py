"""Validate ref_gdn against FLA.fused_recurrent_gated_delta_rule."""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/inferentia-gdn"))

import contextlib

import torch

# FLA 0.5.0 calls torch.cpu.device(tensor.device.index) for its custom-op wrapper.
# torch 2.9 removed torch.cpu.device; torch.cpu._device exists but rejects None
# (CPU tensors have device.index=None). On CPU we don't need a device context
# at all, so shim with a no-op context manager that accepts any arg.
if not hasattr(torch.cpu, "device"):
    torch.cpu.device = lambda *a, **k: contextlib.nullcontext()

from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

from ref.ref_gdn import ref_gated_delta_rule_recurrent


def run_parity(B, T, H, HV, K, V, dtype=torch.bfloat16, seed=42, tag=""):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, HV, V, dtype=dtype)
    g = (-torch.rand(B, T, HV, dtype=torch.float32)) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, HV, dtype=torch.float32))

    out_ref, state_ref = ref_gated_delta_rule_recurrent(
        q, k, v, g, beta, output_final_state=True,
    )
    out_fla, state_fla = fused_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, output_final_state=True,
    )

    out_ref_f = out_ref.float()
    out_fla_f = out_fla.float()
    max_abs = (out_ref_f - out_fla_f).abs().max().item()
    denom = out_fla_f.abs().max().item() + 1e-8
    max_rel = max_abs / denom

    verdict = "PASS" if max_abs < 5e-2 else "FAIL"
    print(f"[{tag}] shapes: q={tuple(q.shape)} v={tuple(v.shape)}")
    print(f"[{tag}] out shape ref={tuple(out_ref.shape)} fla={tuple(out_fla.shape)}")
    print(f"[{tag}] max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}")
    if state_ref is not None and state_fla is not None:
        s_abs = (state_ref.float() - state_fla.float()).abs().max().item()
        print(f"[{tag}] state max_abs_err={s_abs:.3e}  state shape={tuple(state_ref.shape)}")
    print(f"[{tag}] RESULT: {verdict}")
    print()
    return max_abs < 5e-2


if __name__ == "__main__":
    results = []
    results.append(run_parity(B=1, T=64,  H=16, HV=16, K=128, V=128, tag="tiny no-GVA"))
    results.append(run_parity(B=1, T=512, H=16, HV=16, K=128, V=128, tag="Qwen3.5-0.8B shape"))
    results.append(run_parity(B=1, T=512, H=16, HV=32, K=128, V=128, tag="Qwen3.5-4B  shape (GVA 1:2)"))
    results.append(run_parity(B=2, T=256, H=16, HV=16, K=128, V=128, tag="batch=2"))
    overall = "PASS" if all(results) else "FAIL"
    print(f"OVERALL: {overall}")
    sys.exit(0 if all(results) else 1)
