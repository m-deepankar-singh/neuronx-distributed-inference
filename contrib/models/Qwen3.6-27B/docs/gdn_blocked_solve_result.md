# Qwen3.6-27B Blocked DeltaNet Solve Result

Branch: `codex/qwen36-gdn-blocked-solve`  
Baseline control: `qwen36-27b-vllm-apc-baseline-v3`  
Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_blocked_solve_run1`

## Change Tested

Added an opt-in `use_blocked_deltanet_solve` path for the 128-token DeltaNet
microchunk kernel. The experimental kernel computes the triangular inverse in
8 blocks of 16 rows:

- one Tensor Engine matmul per block for already-committed rows
- Vector Engine row-shuffle updates for dependencies inside the block
- full forward-substitution math, no Neumann approximation

The original kernel remains the default.

## Gates

| Gate | Result |
|---|---|
| Local `py_compile` | pass |
| Local blocked-solve CPU reference | pass, 3 tests |
| Remote `py_compile` | pass |
| Remote blocked-solve CPU reference | pass, 3 tests |
| HLO generation | pass, CTE traced at `[1, 512]` |
| NEFF compile | pass |
| Load-after-compile | pass |
| Proxy arithmetic smoke | pass: `391` |
| 762-token MGS prompt | coherent output |
| 16K prefill perf | fail: ~102.6 tok/s |

## Measurements

Proxy path, `max_tokens=1`, 16K prompt:

```text
prompt_tokens: 16396
latency_s:     159.8447
tok/s:         102.57
```

Baseline v3 on the same class of run is approximately 406-428 tok/s. This
blocked-solve implementation is therefore around 4x slower than baseline.

## Conclusion

This implementation is **not shippable**. The compiler accepts the NKI row
shuffle pattern, and the model can generate coherent text, but replacing 128
Tensor Engine matmuls with many Vector Engine row-shuffle/rank-update steps is
slower on Trainium2. The bottleneck is not just the count of Tensor Engine
matmuls; Vector Engine orchestration and row-shuffle traffic dominate this
version.

Do not continue toward 64K validation or exact-token comparison for this path.
Keep the branch as a negative result and return to baseline v3 for serving.

## Next Direction

The next GDN optimization should avoid row-by-row Vector Engine work. Viable
directions:

- a Tensor Engine block method that computes a local 16x16 or 32x32 inverse
  without per-row shuffles
- a direct RHS solve that preserves the baseline layout/order more closely
- head/group fusion only if profiling shows dispatch is still meaningful after
  the v3 sharding/FP8 baseline

