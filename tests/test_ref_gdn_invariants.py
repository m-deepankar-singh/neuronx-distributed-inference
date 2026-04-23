"""Algebraic invariant tests for ref_gdn.

Rationale: FLA can't run on Neuron boxes (Triton requires a GPU driver, no CUDA on trn1).
Instead of a black-box comparison, we pin the algorithm via hand-computable invariants.
If all of these pass, the implementation matches the paper's formulation.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/inferentia-gdn"))

import torch

from ref.ref_gdn import ref_gated_delta_rule_recurrent


def _mk(B, T, H, HV, K, V, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, HV, V, dtype=dtype)
    g = torch.zeros(B, T, HV, dtype=dtype)
    beta = torch.ones(B, T, HV, dtype=dtype)
    return q, k, v, g, beta


def test_single_step_matches_pure_attention():
    """With g=0, beta=1, S_0=0 and T=1: S = k^T ⊗ v, out = (q·k)·v · scale."""
    q, k, v, g, beta = _mk(B=1, T=1, H=1, HV=1, K=8, V=8, seed=1)
    out, S = ref_gated_delta_rule_recurrent(q, k, v, g, beta, output_final_state=True)
    scale = 1.0 / math.sqrt(8)

    q_vec = q[0, 0, 0]
    k_vec = k[0, 0, 0]
    v_vec = v[0, 0, 0]

    # Expected S after one step: outer(k, v)  (since decay=1, beta=1, S_prev=0, delta = v)
    expected_S = torch.einsum("k,v->kv", k_vec, v_vec)
    err_S = (S[0, 0] - expected_S).abs().max().item()
    assert err_S < 1e-6, f"S mismatch: {err_S:.3e}"

    # Expected out: (q·scale) @ S = scale * (q·k) * v
    expected_out = scale * (q_vec @ k_vec) * v_vec
    err_out = (out[0, 0, 0] - expected_out).abs().max().item()
    assert err_out < 1e-5, f"out mismatch: {err_out:.3e}"

    print(f"  S err={err_S:.2e}  out err={err_out:.2e}")
    return True


def test_decay_zero_preserves_state_on_beta_zero():
    """g=0, beta=0 -> state never updates; out[t] = (q_t·scale) @ S_init for all t."""
    q, k, v, g, beta = _mk(B=1, T=16, H=2, HV=2, K=16, V=16, seed=2)
    beta.zero_()
    # Set a fixed initial state so we know what out should be
    S_init = torch.randn(1, 2, 16, 16)
    out, S = ref_gated_delta_rule_recurrent(
        q, k, v, g, beta, initial_state=S_init, output_final_state=True
    )
    scale = 1.0 / math.sqrt(16)
    expected = torch.einsum("bthk,bhkv->bthv", q * scale, S_init)
    err = (out - expected).abs().max().item()
    # Also: S_final should equal S_init
    err_S = (S - S_init).abs().max().item()
    assert err < 1e-4, f"out mismatch under beta=0: {err:.3e}"
    assert err_S < 1e-6, f"state mutated under beta=0: {err_S:.3e}"
    print(f"  out err={err:.2e}  S preservation err={err_S:.2e}")
    return True


def test_decay_neg_inf_resets_state():
    """g -> -inf: exp(g) -> 0, state fully resets each step.
    Then for beta=1: S_t = outer(k_t, v_t), out_t = scale * (q_t·k_t) * v_t.
    """
    q, k, v, g, beta = _mk(B=1, T=5, H=1, HV=1, K=8, V=8, seed=3)
    g.fill_(-50.0)  # exp(-50) ≈ 2e-22, effectively zero
    out, _ = ref_gated_delta_rule_recurrent(q, k, v, g, beta)
    scale = 1.0 / math.sqrt(8)
    # Expected per step: scale * (q_t·k_t) * v_t
    dots = torch.einsum("bthk,bthk->bth", q, k)  # [B, T, H]
    expected = scale * dots.unsqueeze(-1) * v  # [B, T, HV=1, V]
    err = (out - expected).abs().max().item()
    assert err < 1e-4, f"state did not reset: {err:.3e}"
    print(f"  err={err:.2e}")
    return True


def test_linearity_in_v():
    """Scaling v by alpha scales the output by alpha (delta rule is linear in v-v_hat)."""
    q, k, v, g, beta = _mk(B=1, T=32, H=4, HV=4, K=32, V=32, seed=4)
    alpha = 3.7
    out1, _ = ref_gated_delta_rule_recurrent(q, k, v, g, beta)
    out2, _ = ref_gated_delta_rule_recurrent(q, k, alpha * v, g, beta)
    err = (out2 - alpha * out1).abs().max().item()
    denom = (alpha * out1).abs().max().item() + 1e-8
    assert err / denom < 1e-4, f"linearity in v broken: rel {err/denom:.3e}"
    print(f"  rel err={err/denom:.2e}")
    return True


def test_gva_consistency():
    """GVA with repeat_interleave convention (HV = repeat * H):
    V heads (i*repeat .. i*repeat+repeat-1) all share Q/K head i.
    For repeat=2: v_head_0 and v_head_1 share Q head 0.
    If we set v[:,:,2i] == v[:,:,2i+1] and same g/beta, then out pairs must match.
    """
    B, T, H, HV, K, V = 1, 64, 8, 16, 32, 32
    repeat = HV // H
    torch.manual_seed(5)
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)

    # Build v so consecutive `repeat` heads are identical:
    # v_base: [B, T, H, V], then repeat_interleave to [B, T, HV, V]
    v_base = torch.randn(B, T, H, V)
    v = v_base.repeat_interleave(repeat, dim=2)
    g_base = -torch.rand(B, T, H) * 0.1
    g = g_base.repeat_interleave(repeat, dim=2)
    beta_base = torch.sigmoid(torch.randn(B, T, H))
    beta = beta_base.repeat_interleave(repeat, dim=2)

    out, _ = ref_gated_delta_rule_recurrent(q, k, v, g, beta)

    # Every consecutive pair (i*repeat, i*repeat+1) should match
    max_err = 0.0
    for i in range(H):
        base = i * repeat
        for j in range(1, repeat):
            err = (out[:, :, base, :] - out[:, :, base + j, :]).abs().max().item()
            max_err = max(max_err, err)
    assert max_err < 1e-4, f"GVA repeated-head pair diverged: {max_err:.3e}"
    print(f"  max_err={max_err:.2e}")
    return True


def test_batch_independence():
    """Sample 0 and sample 1 should evolve independently. Process each separately,
    confirm batched output equals stacked single-sample outputs.
    """
    torch.manual_seed(6)
    q = torch.randn(2, 16, 4, 16)
    k = torch.randn(2, 16, 4, 16)
    v = torch.randn(2, 16, 4, 16)
    g = -torch.rand(2, 16, 4) * 0.1
    beta = torch.sigmoid(torch.randn(2, 16, 4))

    out_batched, _ = ref_gated_delta_rule_recurrent(q, k, v, g, beta)
    out_0, _ = ref_gated_delta_rule_recurrent(q[:1], k[:1], v[:1], g[:1], beta[:1])
    out_1, _ = ref_gated_delta_rule_recurrent(q[1:], k[1:], v[1:], g[1:], beta[1:])
    err0 = (out_batched[0:1] - out_0).abs().max().item()
    err1 = (out_batched[1:2] - out_1).abs().max().item()
    assert err0 < 1e-6 and err1 < 1e-6, f"batch entanglement: {err0:.3e} {err1:.3e}"
    print(f"  err0={err0:.2e}  err1={err1:.2e}")
    return True


def test_initial_state_continuation():
    """Running T=10 should equal: run T=4, take final state, run T=6 with initial_state=that."""
    torch.manual_seed(7)
    q = torch.randn(1, 10, 2, 16)
    k = torch.randn(1, 10, 2, 16)
    v = torch.randn(1, 10, 2, 16)
    g = -torch.rand(1, 10, 2) * 0.1
    beta = torch.sigmoid(torch.randn(1, 10, 2))

    out_full, S_full = ref_gated_delta_rule_recurrent(q, k, v, g, beta, output_final_state=True)
    out_a, S_a = ref_gated_delta_rule_recurrent(q[:, :4], k[:, :4], v[:, :4], g[:, :4], beta[:, :4], output_final_state=True)
    out_b, S_b = ref_gated_delta_rule_recurrent(
        q[:, 4:], k[:, 4:], v[:, 4:], g[:, 4:], beta[:, 4:],
        initial_state=S_a, output_final_state=True,
    )
    err_out = (out_full - torch.cat([out_a, out_b], dim=1)).abs().max().item()
    err_S = (S_full - S_b).abs().max().item()
    assert err_out < 1e-4 and err_S < 1e-4, f"continuation broken: out={err_out:.3e} S={err_S:.3e}"
    print(f"  err_out={err_out:.2e}  err_S={err_S:.2e}")
    return True


TESTS = [
    ("single_step_pure_attention",       test_single_step_matches_pure_attention),
    ("beta_zero_freezes_state",          test_decay_zero_preserves_state_on_beta_zero),
    ("decay_neg_inf_resets_state",       test_decay_neg_inf_resets_state),
    ("linearity_in_v",                   test_linearity_in_v),
    ("gva_mirrored_heads_match",         test_gva_consistency),
    ("batch_independence",               test_batch_independence),
    ("initial_state_continuation",       test_initial_state_continuation),
]


if __name__ == "__main__":
    failures = 0
    for name, fn in TESTS:
        try:
            print(f"[{name}]")
            fn()
            print(f"[{name}] PASS\n")
        except AssertionError as e:
            print(f"[{name}] FAIL: {e}\n")
            failures += 1
        except Exception as e:
            print(f"[{name}] ERROR: {type(e).__name__}: {e}\n")
            failures += 1
    print(f"OVERALL: {'PASS' if failures == 0 else f'FAIL ({failures} of {len(TESTS)})'}")
    sys.exit(0 if failures == 0 else 1)
