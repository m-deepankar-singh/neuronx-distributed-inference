# Qwen3.6-27B DeltaNet Solve-Isolation Ablation

Baseline branch: `codex/qwen36-gdn-core-rewrite`

This branch is measurement-only. It intentionally replaces the stable
triangular solve:

```text
N = inv(I - A)
```

with:

```text
N = I
```

inside `nki_deltanet_chunked.py`.

The artifact is expected to produce invalid model-quality outputs. Only TTFT and
prefill throughput are meaningful.

Decision rule:

- If solve-noop approaches the recurrent-core no-op speed (~1238 tok/s at 16K),
  the triangular solve is the recurrent-core wall.
- If solve-noop is only modestly faster than baseline (~418 tok/s at 16K), the
  post-solve state/output interaction path dominates.
- If it lands between those points, both matter and the larger delta determines
  the next rewrite target.

## Result

Artifact:
`/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_solve_noop_run1`

| Artifact | 512 tok/s | 2K tok/s | 16K tok/s | 16K TTFT |
|---|---:|---:|---:|---:|
| Baseline v3 | ~406-418 | ~414 | ~418 | ~36.95s |
| Compact-input baseline | 412 | 413 | 417 | 37.04s |
| Solve no-op (`N = I`) | 1020 | 1039 | 1058 | 14.58s |
| GDN recurrent-core no-op | 1067 | 1219 | 1238 | 12.46s |

The solve-noop artifact is measurement-only and produces invalid text, but it
loads and runs without runtime errors.

## Interpretation

The triangular solve is the dominant recurrent-core bottleneck.

At 16K:

- Baseline TTFT: ~36.95s.
- Solve no-op TTFT: ~14.58s.
- Time removed by bypassing the solve: ~22.37s, or ~61% of baseline TTFT.
- Recurrent-core no-op removes ~24.49s, or ~66% of baseline TTFT.

So bypassing `N = inv(I - A)` accounts for roughly 91% of the measured
recurrent-core removable time. The post-solve state/output path is real, but
secondary.

## Next Target

Optimize the stable triangular solve. The next production candidate should not
change model math. It should reduce the cost of computing `N @ v_beta` and
`N @ (k_beta * exp(gc))`.

Candidate directions:

1. Avoid materializing full `N` when only `N @ B` is needed. Solve the two RHS
   blocks directly for `v_beta` and `k_beta * exp(gc)`.
2. Reuse one triangular-solve pass for the concatenated RHS
   `[v_beta, k_beta * exp(gc)]` if NKI tiling can handle it without SBUF spill.
3. Reduce row-by-row full-matrix matmul work in the current forward substitution.
   The current implementation repeatedly computes full `A_T @ P_acc` then masks
   one row; a direct block solve over RHS tiles should do less useless work.
4. Keep the stable single-exp decay and fp32 state. Do not return to Neumann
   power-doubling.
