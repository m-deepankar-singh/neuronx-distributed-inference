# Qwen3.6 27B Prefill Optimization Decision Matrix

Inputs:

- Driver profile: 149.47 tok/s at 1K, 149.70 tok/s at 16K, 149.56 tok/s at near-64K.
- Runtime profile: long `nrt_execute` calls dominate; host submit total is only `0.736s` over the 16K run.
- Monitor: active NeuronCore utilization is typically ~68-100%, max HBM allocation ~98.06 GB.
- Trace limitation: no source-level layer names; device time cannot be directly split into DeltaNet / attention / MLP from this trace alone.

## Bottleneck Identification

The workload is device-execution dominated, not Python dispatch dominated and not host-copy dominated. A 512-token CTE call costs ~3.42s whether it occurs at the beginning or end of a 64K prompt.

Important correction: flat throughput does not prove full attention is cheap. In the current `perform_qwen_chunked_prefill()` implementation, `cache_len = k_cache.shape[2]`, which is the compiled maximum sequence length (`65536`). Each chunk therefore computes attention against a full-size K/V cache tensor, with masking applied after the QK matmul. That produces a fixed per-chunk full-cache cost, independent of current prompt position.

## Optimization ROI Table

| Option | Assumed affected fraction | Kernel speedup | Expected total speedup | New tok/s from 149.6 | Effort | ROI | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| A. DeltaNet chunk 256 | 35-50% | 1.5x | 1.13-1.20x | 169-180 | 1-2 weeks | medium | Plausible, but not proven by the trace. Needs a DeltaNet-vs-attention ablation first. |
| B. Head-dim-256 prefix attention / active-cache attention | 30-60% | 2-4x | 1.23-1.82x | 184-272 | 1-2+ weeks | highest | Code proves current chunk attention pays fixed max-cache shape. This is the best candidate if pursuing a real speedup. |
| C. FP8 KV/cache I/O | <=20% upper bound | 1.7x | <=1.13x | <=169 | ~1 week | low | Runtime copy/write is visible but not dominant; trace does not show HBM cache I/O as the limiting envelope. |
| D. FP8 weights | 30-45% | 1.4x | 1.10-1.18x | 165-176 | ~1 week | medium-low | Could help MLP/projections, but not enough as first optimization unless quantization is needed anyway. |
| E. Reduce host chunk/dispatch overhead | <1% host submit, unknown in-NEFF dispatch | 2x | ~1.00x host-side | 150 | 0.5-1 week | low | Host submit is not the bottleneck. Bigger CTE buckets alone already showed limited benefit. |
| F. Ship current 150 tok/s | 100% | 1.0x | 149.6 | 0 | n/a | Current path is stable and usable at 64K. |

## Recommendation

Highest-ROI optimization is Option B: fix the full-attention chunked CTE path so it does not compute against the full 65,536-token cache for every chunk, ideally via a head_dim=256 prefix-attention kernel or an active-cache-length path that is mathematically exact.

Do not start the DeltaNet chunk_size=256 rewrite as the next major task unless a targeted ablation proves DeltaNet consumes more than ~50% of the 3.42s CTE call. The current profile does not prove that.

## Specific Blockers

- Existing NxDI `attention_cte` and `attention_tkg` kernels cap `head_dim <= 128`; Qwen3.6-27B uses `head_dim=256`.
- Naive head splitting is not mathematically equivalent because softmax must see the full QK score.
- A correct attention fix needs either custom head_dim=256 tiled attention or a way to express active-cache prefix attention with static compiled shapes.

## Ship/Stop Recommendation

If the goal is production usability rather than a research optimization push, ship the current ~150 tok/s 64K path and defer further speed work. If the target is a material speedup beyond 1.20x, pursue Option B first, not FP8 KV or host dispatch work.
