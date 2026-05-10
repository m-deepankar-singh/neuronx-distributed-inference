# Qwen3.6-27B GDN Recurrent-Core Rewrite Plan

Baseline: `qwen36-27b-vllm-apc-baseline-v3`

Measured bottleneck:
- 16K baseline prefill: ~418 tok/s, ~36.95s TTFT.
- Full GDN no-op: ~1292 tok/s, ~11.94s TTFT.
- GDN recurrent-core no-op: ~1238 tok/s, ~12.46s TTFT.
- Attention no-op: ~566 tok/s, ~27.26s TTFT.
- MLP no-op: ~433 tok/s, ~35.65s TTFT.

Conclusion: the recurrent DeltaNet chunk core is the primary prefill wall.
MLP and shallow wrapper work cannot move TTFT enough.

## Phase 1: Compact Kernel Inputs

Problem: `_nki_chunked_forward()` expands scalar per-token tensors to full
`[128, 128]` tiles before every microchunk call:

- `beta`: `[128] -> [128, 128]`
- `g_cumsum`: `[128] -> [128, 128]`
- `g_last`: scalar per chunk -> `[128, 128]`

The NKI kernel then reads only one column or uses the values as row-wise
broadcast operands. This burns HBM bandwidth and bloats the traced graph without
changing math.

Change:
- Pass `beta`, `g_cumsum`, and `g_last` as `[128, 1]`.
- Broadcast inside the kernel with `nisa.tensor_scalar`, matching existing
  `exp_gc_p`/`exp_gl_p` broadcast patterns.
- Keep q/k/v/state/output shapes unchanged.
- Keep the stable single-exp decay and forward-substitution solver unchanged.

Validation:
- `py_compile` locally.
- Compile one 128K/CTE512 artifact on Trn2.
- Greedy exact-token match vs baseline on short and 762-token prompts.
- 16K and 64K TTFT must improve or remain within noise.

Gate:
- If the compact-input artifact improves prefill by >= 5%, keep it.
- If it is neutral, keep only if it simplifies the next recurrent-core rewrite.
- If it regresses or mismatches tokens, revert.

## Phase 2: Remove Constant Mask HBM Inputs

Problem: every microchunk call receives `lower_mask`, `identity`, and
`lower_mask_diag` as `[128, 128]` HBM inputs. They are compile-time constants.

Change:
- Generate or embed masks in the kernel, or pass a smaller representation if NKI
  compile behavior allows.
- Keep this separate from Phase 1 so failures are easy to isolate.

Gate:
- Same correctness gates as Phase 1.
- Expected gain is modest by itself but reduces recurrent-core memory traffic.

## Phase 3: Recurrent-Core Dataflow Rewrite

Target only after Phase 1/2 establish a clean baseline.

Directions:
- Reduce transpose count around `N`, `q`, `k`, `k_cumdecay`, and `attn_intra`.
- Preserve the stable triangular solve until a replacement proves exact enough.
- Try a recurrence-oriented layout where the free dimension is reused more
  effectively across `value_corr`, `v_prime`, `attn_inter`, and state update.
- Do not revive the stale Neumann fused kernel.

Gate:
- Simulator or torch-reference equivalence where available.
- Greedy exact-token match vs baseline on short and 762-token prompts.
- 16-step logit cosine >= 0.999 if logits artifact is available.
- 16K prefill speedup target: >= 1.15x over baseline.
