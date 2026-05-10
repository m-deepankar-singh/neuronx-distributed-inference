# Qwen3.6-27B Blocked Triangular Solve — Design

Branch target: `codex/qwen36-gdn-blocked-solve`
File target:   `contrib/models/Qwen3.6-27B/src/nki_kernels/nki_deltanet_chunked_blocked.py`
Baseline:      `qwen36-27b-vllm-apc-baseline-v3` (~418 tok/s @ 16K)

## Problem

`nki_deltanet_chunked.py` builds `N = inv(I - A_lower)` using two
`nl.static_range(64)` loops (lines 292, 327) that together do 128 iterations.
Each iteration calls `nisa.nc_matmul(stationary=A_T, moving=P_acc)` over the
full 128x128 SBUF tile, then masks down to a single committed row.

Per-microchunk solve cost: ~300 us, dominated by 128 full-size TE matmuls
where 127/128 output rows are discarded each iteration. From the solve no-op
ablation, this section accounts for ~91% of recurrent-core time and ~60% of
total 16K TTFT.

## Target

Reduce the 128 sequential outer matmuls to 8 by processing 16 rows per outer
iteration. Within each 16-row block, perform sequential rank-1 updates using
Vector Engine ops only (no fresh TE matmul per row). Preserve bit-exact match
against the existing forward-substitution math wherever the IEEE 754
accumulation order remains unchanged.

Estimated per-microchunk solve cost: ~80 us. Estimated 16K TTFT: ~21 s.
Estimated tok/s: ~780-830. Total prefill speedup: ~1.85-2.0x.

## Algorithm

`A_lower` is strict lower triangular. For block i covering rows
`base = i*BLOCK .. base+BLOCK-1` (BLOCK=16):

1. Compute `external_rows = A_T[base:base+BLOCK, :base] @ P_acc[:base, :]`
   in a single TE matmul. This produces the contribution to block rows from
   already-committed rows of `P_acc`.

2. Within the block, iterate sequentially `solve_inner = 0..BLOCK-1`. For
   each `row_idx = base + solve_inner`:
   - Compute `internal_contribution = sum over j < solve_inner of
     A_T[row_idx, base+j] * P_acc_block[j, :]` using Vector Engine ops only.
     The sum has exactly `solve_inner` terms.
   - Combine `external_rows[solve_inner] + internal_contribution + eye[row_idx]`
     to form the row update.
   - Apply `col_mask` (left for top half, right for bottom half) to mirror the
     existing two-loop column-masking discipline.
   - Commit to `P_acc[row_idx]` via a row-masked add.

3. After all 8 blocks complete, run the existing cross-block fixup
   (lines 364-399 in the current kernel) for the bottom-left N21 block. This
   step is unchanged.

## Bit-exactness analysis

The current kernel's per-row commit produces `P_acc[row] = (A_T @ P_acc)[row] +
eye[row]`, masked to the appropriate column half. The matmul is on the full
128x128 tile; only the row indexed by `row_mask` is committed.

The blocked variant's per-row commit produces `P_acc[row] =
external_rows[row - base] + internal_contribution + eye[row]`, where:

- `external_rows[row - base]` is one row of a smaller TE matmul. TE matmul is
  deterministic; output bits depend only on operand bits and tile sizes. If
  the operand SBUF tiles are aligned to the same 128-wide partition layout as
  today, output bits should match the corresponding row of the larger matmul.
  This is the main gate for bit-exactness.
- `internal_contribution` is built via Vector Engine `tensor_tensor` /
  `tensor_scalar` ops. These are the same ops used today for masking and
  combining; FMA accumulation order is the same per-row.
- The `+ eye[row]` and `* col_mask` steps are identical to today.

Risk areas:
- TE matmul output bits when the matmul shape changes (e.g. from full
  `(128,128) @ (128,128)` to `(16,128) @ (128,128)`). NeuronCore TE pipelines
  fixed 128-wide partition tiles; smaller logical shapes pad with zeros. The
  pad zeros do not affect output bits for the live rows, but compiler-side
  reordering of the inner dot product reduction is possible.
- The `internal_contribution` accumulation. If implemented as a sequential
  `tensor_tensor` chain `acc = acc + A * P_row`, the FMA order matches today.
  If implemented as a single small TE matmul `(1, BLOCK) @ (BLOCK, 128)`, the
  reduction order may differ.

Validation gate: `nki.simulate` against the existing
`deltanet_chunk_step` with random fp32 inputs. Target: max_abs_diff = 0
(bit-exact). If max_abs_diff is small but nonzero (~1e-7 to 1e-5), fall back
to cosine >= 0.999 as the gate; this is the same gate baseline-v1 used
against HF reference.

## Implementation outline

Create `nki_deltanet_chunked_blocked.py` as a copy of
`nki_deltanet_chunked.py`. Replace lines 256-399 (everything from `P_acc`
init through `N21` fixup) with the blocked variant. Keep all pre-solve
sections (lines 1-255) and post-solve sections (lines 401-552) verbatim.

Replace the two `static_range(64)` outer loops with one
`static_range(P_MAX // BLOCK)` outer loop containing:
- One TE matmul for the external contribution
- One `static_range(BLOCK)` inner loop for sequential rank-1 updates
- Column masking applied per-row to mirror the existing top/bottom split
  (rows 0-63 use `col_mask_left`, rows 64-127 use `col_mask_right`)

The cross-block fixup (lines 364-399) is unchanged. It can move outside the
outer loop or stay at the end of the solve section.

Hook into `_fused_chunked_forward` via a config flag
`use_blocked_deltanet_solve` (default False). Bind to
`_deltanet_fused_kernel` when flag is True.

## Validation plan

| Phase | Gate | Wall-clock budget |
|---|---|---|
| Simulator | `nki.simulate` max_abs_diff = 0 vs current kernel | 0.5 day |
| Compile | NEFF compile + load on hardware | 0.5 day |
| Greedy gate | 762-token Metal Gear Solid prompt token-exact match | 0.25 day |
| Diverse gate | 50-prompt set, >= 95% token agreement at top_k=1 | 0.5 day |
| Performance | vLLM 16K + 64K runs, target >= 1.7x baseline | 0.5 day |
| HBM check | neuron-monitor peak at 64K, target <= baseline + 5% | 0.25 day |

If simulator returns nonzero diff: investigate whether TE matmul shape change
caused reordering. If diff is bounded by ~1e-5 fp32, proceed to greedy gate
with cosine >= 0.999 fallback. If diff is large (>1e-3): bug; do not proceed.

## Expected performance

| Metric | Baseline v3 | Blocked solve target |
|---|---:|---:|
| 16K TTFT | 36.95 s | ~21 s |
| 16K tok/s | 418 | ~780 |
| Solve fraction of TTFT | 60% | ~25% |
| Outer TE matmuls per microchunk | 128 | 8 |
| Per-microchunk solve | 303 us | ~80 us |

Total prefill speedup: ~1.85-2.0x (16K), similar at 64K.

## Out of scope

- Direct triangular solve via NKI sequential_range with variable-length inner
  ops (NKI primitive limitations, attempted in `codex/qwen36-gdn-direct-rhs-solve`).
- Neumann power-doubling fp32 (correctness risk, separate optimization track).
- Surrounding GDN op fusion (l2norm, gate, projections — ablation showed <2%
  share, not worth the work).
- Cache-length attention bucketing (separate phase, lower ROI than solve).
- Head-batched dispatch (ablation showed dispatch is not the bottleneck).

## Stack with future work

This optimization is independent and stacks with:
- FP8 KV cache (decode bandwidth, ~1.2x decode)
- FP8 attention QKV/O weights (extends current FP8 MLP-only)
- Speculative decoding (decode tok/s, ~1.5-2x)

Combined ceiling after blocked solve + FP8 KV + spec decode:
- Prefill: ~800 tok/s (this work)
- Decode: ~50-60 tok/s (FP8 KV + spec)
