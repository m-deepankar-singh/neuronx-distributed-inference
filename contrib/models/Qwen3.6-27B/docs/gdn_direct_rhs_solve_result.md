# Qwen3.6-27B Direct RHS Solve Result

Branch: `codex/qwen36-gdn-direct-rhs-solve`

Artifact:
`/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_direct_rhs_run2`

## Change

The chunked DeltaNet kernel no longer materializes the full triangular inverse
`N = inv(I - A)` before computing the value correction. Instead, it uses the
identity

```text
N @ v_beta - (N @ (k_beta * exp(gc))) @ state
  == N @ (v_beta - (k_beta * exp(gc)) @ state)
```

and solves a single RHS directly for `v_new`.

The first row-vector implementation failed Neuron compiler BIR verification
because `tensor_scalar` could not consume a partition-sliced `1x128` row with a
`1x1` coefficient from `A_mat`.

The second implementation compiled by keeping the compiler-safe full-matmul
plus row-selection pattern from the validated inverse solve.

## Validation

Local and remote CPU checks passed:

- `python3 -m py_compile` on touched files.
- `test_deltanet_direct_rhs_solve.py`.
- `test_deltanet_compact_inputs.py`.

Compile and load passed:

- CTE HLO generation: ~14.5s.
- CTE NEFF compile: passed.
- Load-after-compile: passed.

## Performance

Direct vLLM measurements with unique prompts:

| Prompt | Tokens | Seconds | Tok/s |
|---|---:|---:|---:|
| Short smoke | 16 prompt / 8 completion | 1.39s | n/a |
| 512 | 512 | 1.08s | 474 |
| 2K | 2048 | 3.22s | 635 |
| 762 repeat | 762 | 2.17s | 351 |
| 16K repeat | 16000 | 34.32s | 466 |
| 64K repeat | 64000 | 134.10s | 477 |
| 128-token decode smoke | 128 completion | 5.86s | 21.84 completion tok/s total |

Compared with baseline v3's ~418 tok/s long-context prefill, the direct RHS
path gives roughly 1.11-1.14x sustained long-context prefill speedup.

## Correctness Gate

The artifact fails the exact greedy token-match gate against baseline v3 on a
762-token repeated Metal Gear Solid prompt.

Baseline v3 output:

```text
 while uncovering a deeper conspiracy involving FOXHOUND, Metal Gear REX, and his own origin. Solid Snake enters Shadow Moses to stop a nuclear threat while
```

Direct RHS output:

```text
 while same sentence repeated many times

<think>

</think>

Solid Snake enters Shadow Moses to stop a nuclear threat while uncovering a deeper conspiracy involving FOXHOUND,
```

The output is coherent, but it is not exact. This branch is therefore not a
shipping candidate.

## Conclusion

The algebraic rewrite is mathematically valid and faster, but the changed
floating-point accumulation order is enough to alter greedy output on the
strict gate. Keep this branch as evidence that solving one RHS can buy real
speed, but do not promote it to baseline.

Next production direction: optimize the original materialized-`N` path without
changing the final arithmetic order, or add a lower-level row-reduction kernel
that preserves numerical behavior closely enough to pass the greedy gate.
