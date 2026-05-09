# Codex Prompt — Head-Batched DeltaNet Kernel

```
═══════════════════════════════════════════════════════════════════════
DELTANET HEAD-BATCHING — eliminate per-head dispatch loop
═══════════════════════════════════════════════════════════════════════

CONTEXT

Branch: codex/qwen36-27b-attn-cache-ablation (or main 64k branch)
Repo: ~/inferentia-gdn
Target: contrib/models/Qwen3.6-27B
Hardware: trn2.3xlarge, TP=4, BF16
Current baseline: 149.6 tok/s, 3.42s per CTE=512 chunk.

Identified redundancy (from code reading):
- modeling_qwen35.py:559 in _fused_chunked_forward
- Python loop "for bh in range(BH)" dispatches the NKI kernel
  _deltanet_fused_kernel 48 times per GDN layer per CTE chunk
- Total: 48 dispatches × 48 GDN layers = 2,304 NKI dispatches per chunk
- Each dispatch reloads state, sets up SBUF tiles, does launch overhead
- The kernel currently takes (S, 128) per-head shape, not (BH, S, 128)

Why the prior ablations missed this:
- attention ablation (16 attention layers × 1 batched call = 16 dispatches)
  only saved ~5%. DeltaNet's 2,304 dispatches dwarf that.
- CTE bucket scaling reduces chunk COUNT but per-chunk dispatch count
  is constant, hence the 8% cap.

GOAL: rewrite _deltanet_fused_kernel to take (BH, S, 128) shape and
process all 48 heads in one dispatch. Reduce dispatches from 2,304
per chunk to 48 per chunk (48× reduction).

OUT OF SCOPE — do NOT do any of these:
- Cache-length bucketing (Option B from earlier — confirmed not viable)
- Chunk_size=256 internal (separate work, lower priority now)
- FP8, speculation, vLLM
- Attention kernel changes

═══════════════════════════════════════════════════════════════════════
PHASE A — DISPATCH ABLATION (target: 1 day)
═══════════════════════════════════════════════════════════════════════

Before rewriting the kernel, confirm that dispatch overhead actually
dominates. Two cheap measurements:

A.1 Compile a "dispatch-stress" variant:
  - Modify _fused_chunked_forward temporarily so that each (B,H) iteration
    calls the kernel TWICE on the same data (output of second call discarded).
    This doubles dispatches without changing useful compute.
  - Compile this variant alongside the baseline.

A.2 Measure:
  - Baseline chunk time: 3.42s
  - 2x-dispatch variant chunk time: T

  The dispatch overhead per kernel call is:
    overhead_per_call = (T - 3.42s) / 48 / num_layers / num_dispatches_added

  Total dispatch overhead in baseline:
    total_overhead = overhead_per_call × 2304

  Dispatch fraction:
    dispatch_fraction = total_overhead / 3.42s

A.3 Decision gate:
    dispatch_fraction > 0.30  → PROCEED to Phase B (head-batching is
                                  high ROI; expected total speedup 1.2-1.5×)
    dispatch_fraction 0.15-0.30 → REPORT and decide; ROI moderate
    dispatch_fraction < 0.15  → ABORT; per-call overhead is small,
                                  head-batching gives <1.15× total

A.4 Output: ablation_dispatch.md with measurements.

HARD STOP. Wait for confirmation before Phase B.

═══════════════════════════════════════════════════════════════════════
PHASE B — HEAD-BATCHED KERNEL REWRITE (target: 1-1.5 weeks)
═══════════════════════════════════════════════════════════════════════

B.1 Read existing kernel:
  - contrib/models/Qwen3.6-27B/src/nki_kernels/nki_deltanet_fused.py
    (or whichever file contains _deltanet_fused_kernel)
  - Identify input shapes, SBUF tile structure, partition assignments
  - Identify state I/O (HBM read of initial state, HBM write of final
    state per (B,H) call)

B.2 Design the head-batched version:
  - Inputs: (BH, S, k_dim) instead of (S, k_dim) for Q/K/V
  - g, beta become (BH, S, 1) instead of (S, 1)
  - State input/output: (BH, k_dim, v_dim) instead of (k_dim, v_dim)
  - Tile heads across the partition dimension (P up to 128;
    BH=48 fits in 48 partitions natively)
  - Within each head's partition, the chunk math is unchanged

B.3 NKI implementation:
  - Replace per-head load+store with batched load+store
  - All 48 heads' Q/K/V live in SBUF simultaneously, each in its own
    partition group
  - Forward substitution + state update happens in parallel across
    all 48 partitions
  - Single output store at end

B.4 SBUF budget check:
  - Estimate SBUF usage at BH=48:
      Q tile: 48 partitions × 128 elements × 4 bytes = 24 KB per chunk
      K, V same
      State: 48 partitions × 128 × 128 × 4 = 3 MB
      Decay matrices, scratch: estimate
  - Total should fit in 24 MB SBUF per NeuronCore
  - If exceeds: tile heads in groups of 24 or 16 (still much better
    than 1 at a time)

B.5 CPU validation via nki.simulate:
  - Same input through old kernel (48 separate calls) vs new
    kernel (1 call with BH=48 dimension)
  - max_abs_diff < 1e-5 fp32
  - Token match exact under greedy decode

B.6 Hardware compile and validation:
  - Compile new artifact
  - Validation gates (each must pass before next):
    a. Compile succeeds (no NCC_EVRF009, no NCC_ITIN902)
    b. Loads on hardware
    c. 63-token prompt: coherent, valid IDs
    d. 762-token prompt: tokens match baseline exactly under greedy
    e. 16K prompt: ≥ 195 tok/s (target: 1.3× baseline)
    f. 64K prompt: ≥ 195 tok/s (target: 1.3× baseline)

B.7 Update _fused_chunked_forward:
  - Replace the for-loop with single batched call
  - Add config flag use_head_batched_deltanet (default False)
  - Default off until validation confirms throughput + correctness

B.8 Output phase_b_results.md:
  - Per-chunk timing with new kernel
  - Speedup vs baseline at 16K and 64K
  - HBM peak
  - Compile time and artifact size

═══════════════════════════════════════════════════════════════════════
HARD CONSTRAINTS
═══════════════════════════════════════════════════════════════════════

1. Phase A is a hard gate. dispatch_fraction < 0.15 → abort.
2. CPU correctness via nki.simulate before Neuron compile.
3. Greedy token match (B.6d) is exact, not approximate.
4. Commit + push after each step. Branch:
   chunked-prefill-deltanet-head-batched.

═══════════════════════════════════════════════════════════════════════
EXPECTED OUTCOME
═══════════════════════════════════════════════════════════════════════

Phase A finds dispatch_fraction ~0.30-0.50: head-batching gives
1.3-1.6× total speedup, ~195-240 tok/s sustained.

Phase A finds dispatch_fraction <0.15: ablation says head-batching
won't help much. The bottleneck is per-call compute, not per-call
overhead. In that case, ship at 150 or revisit chunk_size=256
internal (different bottleneck class).

Phase B failure modes:
- SBUF budget exceeded at BH=48: tile in groups (still beneficial)
- Numerical mismatch: kernel implementation bug, do not ship
- Compile fails: report tile sizes, may need different approach

Begin Phase A. Report after Phase A. Do not chain.
```
