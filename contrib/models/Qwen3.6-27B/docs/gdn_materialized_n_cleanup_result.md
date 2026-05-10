# Qwen3.6-27B Materialized-N Cleanup Result

Branch: `codex/qwen36-gdn-materialized-n-cleanup`

Artifact:
`/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_materialized_n_cleanup_run1`

## Change

This branch kept the original materialized triangular inverse path:

```text
N = inv(I - A)
value_corr = N @ v_beta
k_cumdecay = N @ (k_beta * exp(gc))
v_new = value_corr - k_cumdecay @ state
```

It only attempted low-risk cleanup:

- Replace transpose-by-matmul-with-identity calls with `nisa.nc_transpose`.
- Remove the redundant second lower-triangular mask when building `A_mat`,
  because `decay_strict` already includes the strict lower mask.

## Validation

Local and remote CPU checks passed:

- `python3 -m py_compile` on touched files.
- `test_deltanet_materialized_n_cleanup.py`.
- `test_deltanet_compact_inputs.py`.

Compile and load passed:

- CTE HLO generation: ~14.5s.
- CTE NEFF compile: passed.
- Load-after-compile: passed.

## Performance

Direct vLLM measurements with unique prompts:

| Prompt | Tokens | Seconds | Tok/s |
|---|---:|---:|---:|
| Short smoke | 16 prompt / 8 completion | 1.51s | n/a |
| 512 | 512 | 1.20s | 426 |
| 2K | 2048 | 4.78s | 428 |
| 16K | 16000 | 38.21s | 419 |
| 64K | 64000 | 149.26s | 429 |

This is effectively the same as baseline v3. There is no material runtime
speedup.

## Correctness Gate

The 762-token repeated Metal Gear Solid greedy output differed from baseline v3.

Materialized cleanup output:

```text
 while

<think>

</think>

Solid Snake enters Shadow Moses to stop a nuclear threat while uncovering a deeper conspiracy involving FOXHOUND, Metal Gear REX,
```

Baseline v3 output from the same direct-vLLM comparison prompt:

```text
 while uncovering a deeper conspiracy involving FOXHOUND, Metal Gear REX, and his own origin. Solid Snake enters Shadow Moses to stop a nuclear threat while
```

The likely cause is that `nc_transpose` is not numerically/layout-equivalent to
the existing `nc_matmul(..., moving=eye)` transpose idiom for this kernel's
matmul orientation. Even if it were exact, this branch does not improve speed.

## Conclusion

Do not ship this branch.

The result closes off the simple materialized-`N` cleanup path. The real solve
bottleneck is not the surrounding transpose-by-identity calls or the redundant
mask multiply; it is the 128 repeated full-tile triangular-solve matmuls.

The next viable direction is a lower-level solve kernel that preserves the
existing arithmetic order closely enough to pass greedy matching, or a controlled
accuracy/performance policy where small greedy drift is explicitly accepted.
