# Codex Prompt — Option B Ablation + Implementation

```
═══════════════════════════════════════════════════════════════════════
ATTENTION OPTIMIZATION TASK — Qwen3.6-27B Active-Cache-Length Attention
═══════════════════════════════════════════════════════════════════════

CONTEXT

Branch: codex/qwen36-27b-64k-internal at e9118a2
Repo: ~/inferentia-gdn (commit profile reports already in branch)
Target: contrib/models/Qwen3.6-27B
Hardware: trn2.3xlarge, TP=4, BF16
Current baseline: 149.6 tok/s prefill (constant across 1K-64K),
  3.42s per CTE=512 call.

Profile finding (already documented in profile_decision.md):
- perform_qwen_chunked_prefill() uses cache_len = k_cache.shape[2] = 65536
- Every chunk pays full-cache attention cost regardless of cumulative
  position
- Throughput is flat because per-chunk attention work is constant at
  maximum, not because attention is cheap
- Runtime trace cannot break down 3.42s into DeltaNet vs attention vs MLP

GOAL: ablate to confirm attention is a meaningful fraction of per-chunk
time, then implement bucketed cache_len if justified.

OUT OF SCOPE — DO NOT DO ANY OF THESE:
- DeltaNet kernel rewrite at chunk_size=256
- FP8 quantization
- Speculative decoding
- vLLM integration
- head_dim=256 NKI flash attention surgery (separate phase if Option B
  bucketing isn't sufficient)

═══════════════════════════════════════════════════════════════════════
PHASE A — ABLATION (target: 0.5 day)
═══════════════════════════════════════════════════════════════════════

A.1 Locate perform_qwen_chunked_prefill() in modeling_qwen35.py.
  Confirm cache_len = k_cache.shape[2] is the actual code pattern.
  Record file:line.

A.2 Build a microbenchmark for individual layer types:

  Write contrib/models/Qwen3.6-27B/test/unit/ablate_chunk_layers.py
  that loads the existing 64K artifact and:
    - Times a single attention forward pass against full 65536 cache
    - Times a single DeltaNet chunk_step call
    - Times a single MLP forward
    - Times a single layer norm

  For each, run 100 iterations after 10 warmup iterations.
  Use torch_xla.core.xla_model.mark_step + time.perf_counter for sync.

A.3 Compute estimated per-chunk breakdown:

    estimated_total = (
        16 × attention_time +
        48 × deltanet_time +
        64 × mlp_time +
        128 × layernorm_time +
        cache_io_time
    )

  Compare to actual 3.42s observed per chunk.
  Compute % share for each layer category.

A.4 Output: ablation_phase_a.md with:
  - Per-layer-type single-call time
  - Estimated full-chunk breakdown
  - % share table
  - Whether estimate matches observed 3.42s within 20%
    (if not, instrumentation is wrong or there's a hidden cost)
  - DECISION GATE:
    * If attention % share >= 35%: PROCEED to Phase B
    * If attention % share 20-35%: REPORT and stop, decision needed
    * If attention % share < 20%: ABORT Option B, recommend ship at 150
      or revisit DeltaNet 256

HARD STOP after Phase A. Wait for human confirmation before Phase B.

═══════════════════════════════════════════════════════════════════════
PHASE B — BUCKETED CACHE_LEN (target: 1 week)
═══════════════════════════════════════════════════════════════════════

Only proceed if Phase A confirmed attention >= 35%.

B.1 Design

  Add 3 cache_len buckets to the artifact compile:
    bucket_0: cache_len = 4096   (covers chunks 0-7 at CTE=512)
    bucket_1: cache_len = 16384  (covers chunks 8-31)
    bucket_2: cache_len = 65536  (covers chunks 32-127)

  Each bucket compiles its own context_encoding_model variant with
  the right cache shape. Driver picks bucket per chunk based on
  current cumulative position.

  Output design_phase_b.md with:
    - Bucket sizes chosen and rationale
    - Driver dispatch logic
    - Estimated speedup per chunk position
    - Total expected speedup across 16K and 64K prompts

B.2 Implementation

  Modify the model config / compile flow to accept a list of
  cache_len_buckets instead of a single value.

  Modify perform_qwen_chunked_prefill() to:
    - Compute cumulative_pos = chunk_index × CTE_bucket_size
    - Pick the smallest cache_len bucket containing cumulative_pos
    - Dispatch attention to the corresponding context_encoding_model

  Keep the existing single-bucket code path as fallback (config flag
  use_bucketed_cache_attention, default False until validated).

B.3 CPU correctness validation BEFORE compile:

  Same prompt run two ways on CPU/torch:
    - Path 1: full 65536 cache, attention masked beyond cumulative_pos
    - Path 2: actual cache slice up to cumulative_pos, no mask needed

  max_abs_diff at fp32 < 1e-5

  HARD STOP if doesn't match: math is wrong somewhere.

B.4 Compile and load:

  Compile with cache_len_buckets=[4096, 16384, 65536].
  Expected compile time: ~50-70 min (3 context graphs to compile).
  Expected artifact size: ~150 GB (multiple context graphs share weights
  but still ~2× the single-bucket size).

  Verify it loads on hardware before any inference.

B.5 Validation gates (each must pass before next):
  a. Loads on hardware
  b. Short 63-token prompt: coherent output, no invalid IDs
  c. 762-token prompt: matches single-bucket artifact's output exactly
     (greedy decode, top_k=1)
  d. 16K prompt: ≥ 200 tok/s (target: 1.4× current)
  e. 64K prompt: ≥ 180 tok/s (target: 1.2× current — late chunks
     still pay full cost)

B.6 Characterization:
  - Per-chunk-position timing
  - Confirm chunks 0-7 are now much faster than current 3.42s
  - Confirm chunks 32+ are unchanged (~3.42s)
  - HBM peak

B.7 Output phase_b_results.md with measurements + speedup table.

═══════════════════════════════════════════════════════════════════════
HARD CONSTRAINTS
═══════════════════════════════════════════════════════════════════════

1. Phase A is a hard gate. If attention < 35%, do not proceed to Phase B.
2. Stop after each phase. Report. Wait for confirmation.
3. CPU correctness check (B.3) must pass before any Neuron compile.
4. Numerical correctness gate (B.5c) is non-negotiable. Must produce
   identical tokens vs single-bucket artifact under greedy decoding.
5. Commit + push after each step. Branch:
   chunked-prefill-bucketed-attention.

═══════════════════════════════════════════════════════════════════════
EXPECTED OUTCOME
═══════════════════════════════════════════════════════════════════════

Phase A success: attention is ~40-55% of chunk time, Phase B justified.
Phase A "no": attention < 35%, ship at 150 or revisit DeltaNet.
Phase B success: 200-260 tok/s sustained at 16K, 180-200 at 64K.
Phase B failure modes:
  - Compile too large for HBM at multi-bucket: drop to 2 buckets
  - Late-chunk perf doesn't improve (expected) — only early-chunk wins
  - Numerical mismatch vs baseline: math bug, do not ship

Begin with Phase A.1. Report after Phase A complete. Do not start B
without confirmation.
```
