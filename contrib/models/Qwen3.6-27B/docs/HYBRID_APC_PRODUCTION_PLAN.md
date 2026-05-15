# Qwen3.6 Hybrid APC Production Plan

## Build Order

```text
1. Production hybrid APC correctness
2. Dynamic CTE bucket serving
3. Block-size, bucket, and HBM tuning
4. GDN state dtype and memory optimization
5. Decode-side improvements
6. Kernel fusion and speculative decode
```

Do not start with FP8 recurrent cache, MTP, EAGLE, Medusa, flash decode, KV
tiling, or deeper GDN kernel fusion. Those add scheduler and rollback
complexity before the cache contract is correct.

## Target Cache Object

```text
HybridPrefixCheckpoint
  cumulative_prefix_hash
  token_ids_hash
  cache_salt / tenant key
  prefix_length_at_boundary

  attention:
    per-attention-layer KV block refs

  gdn:
    per-GDN-layer recurrent_state checkpoint
    per-GDN-layer conv_state checkpoint

  metadata:
    dtype
    layout_version
    model_revision
    ref_count
    last_access_time
    valid_state_mask
```

The usable hit is the deepest cumulative-prefix boundary where all required
state exists:

```text
usable_hit_len =
  intersection(
    attention_KV_full_block_hit,
    all_GDN_recurrent_prefix_checkpoint_hits,
    all_GDN_conv_prefix_checkpoint_hits
  )
```

If attention KV hits 16K but GDN state only hits 12K, suffix prefill must resume
from 12K.

## Qwen3.6 GDN State

At every reusable cumulative-prefix boundary, cache:

```text
recurrent_state: [num_local_value_heads, key_dim, value_dim]
conv_state:      [conv_dim, conv_kernel_size - 1]
```

Initial dtype policy:

```text
attention KV:        bfloat16
GDN conv_state:      bfloat16
GDN recurrent_state: float32
```

Conv state is small but correctness-critical. Recurrent state dominates GDN
cache memory and should remain FP32 until BF16 exactness is proven.

## Restore Flow

For prompt length `P` and hybrid hit length `H`:

```text
cached prefix:  tokens [0, H)
suffix prefill: tokens [H, P)
decode:         tokens [P, ...)
```

Serving path:

```text
1. vLLM hashes prompt blocks.
2. Hybrid APC computes usable H.
3. Restore attention block table for [0, H).
4. Restore GDN recurrent_state at H.
5. Restore GDN conv_state at H.
6. Send only suffix tokens [H, P) to Neuron CTE.
7. Position IDs start at H.
8. Attention suffix attends to cached KV plus new suffix KV.
9. GDN recurrence starts from restored recurrent_state.
10. GDN conv starts from restored conv_state.
11. Store new boundary checkpoints for newly completed blocks.
12. Decode uses final restored and updated state.
```

## Sprint Plan

### Sprint 1: Correctness Foundation

Build:

```text
HybridAPCManager
GDN recurrent/conv prefix-boundary checkpoint cache
hybrid hit intersection
partial-prefix restore path
FP32 recurrent cache option
correctness tests
```

Success criteria:

```text
warm full-prefix output == cold output
partial-prefix output == cold output
attention-only false hit cannot happen
concurrent requests do not leak state
```

Current v0 branch status:

```text
implemented:
  HybridAPCMetadataStore for cumulative-prefix checkpoint metadata
  bounded model-side HybridGDNCheckpointCache tensor bank
  model restore/commit slot inputs
  use_hybrid_apc_manager initialization without the old guard
  v0 launcher validation requiring checkpoint interval == block size
  async prefix-caching bridge for scheduler-supplied restore/commit tensors
  request finish/cancel lifecycle callbacks for checkpoint refcounts
  Trainium exactness and HBM validation harness

still required before production:
  vLLM scheduler integration that computes cumulative-prefix hashes and slots
  Trainium execution of cold/warm exactness harness on compiled artifacts
  production cancellation/eviction callback wiring from vLLM events
  long-context HBM sweep to choose checkpoint slot count and commit policy
```

### Sprint 2: Dynamic CTE Buckets

Build:

```text
multi-bucket CTE artifact path
runtime suffix bucket selection
262K TP=4 [256] artifact
block_size 128/256 comparison
```

Success criteria:

```text
short prompts retain 1.5x-2.3x latency gain
262K TP=4 [256] loads
TP=4 beats TP=8 unless TP=4 cannot load
```

### Sprint 3: Memory and HBM Tuning

Build:

```text
GDN recurrent state slot accounting
eviction/ref-count policy
FP32 vs BF16 recurrent experiment
attention KV memory report
hybrid cache memory dashboard
```

### Sprint 4: Decode Optimization

Build:

```text
lower-overhead GDN state gather/scatter
decode microbenchmarks
batch-slot reuse optimization
possibly fused recurrent step
```

## Test Matrix

Correctness:

```text
cold vs warm exact token IDs
partial-prefix exact match
non-block-aligned shared prefix floors to full block
attention hit with missing GDN state falls back
conv-state restore failure test by zeroing conv state
multi-hit chat simulation
mixed cold/warm continuous batching
long-context warm hit at 128K and 262K
```

Performance:

```text
Context length: 256, 512, 2K, 8K, 32K, 128K, 262K
Block size:    64, 128, 256
CTE buckets:   [256], [512], [256,512], [256,512,1024]
TP:            4, and 8 only if HBM/load requires it
Cache mode:    no APC, attention APC only, hybrid APC
GDN dtype:     recurrent FP32, recurrent BF16 experiment
Workloads:     single request, repeated system prompt, chat, long-doc QA
```

Immediate Trainium experiments:

```text
262K TP=4, block_size=256, CTE buckets [256]
262K TP=4, block_size=128, CTE buckets [256]
128K TP=4, block_size=128, CTE buckets [256,512]
128K TP=4, block_size=256, CTE buckets [256,512]
```
