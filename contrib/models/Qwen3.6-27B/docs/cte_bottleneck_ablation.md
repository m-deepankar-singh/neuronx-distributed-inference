# Qwen3.6-27B CTE Bottleneck Ablation

Date: 2026-05-10

Baseline: `qwen36-27b-vllm-apc-baseline-v3`

Runtime shape:
- Model: Qwen3.6-27B
- Context: 128K
- CTE bucket: 512
- TP: 4
- vLLM APC enabled
- FP8 MLP-only checkpoint

## Result

Cold prefill is dominated by the GDN path.

| Artifact | 512 tok/s | 2K tok/s | 16K tok/s | 16K TTFT |
|---|---:|---:|---:|---:|
| Baseline v3 | ~406-418 | ~414 | ~418 | ~36.95s |
| GDN no-op | 1108 | 1270 | 1292 | 11.94s |
| GDN recurrent-core no-op | 1067 | 1219 | 1238 | 12.46s |
| Attention no-op | 526 | 559 | 566 | 27.26s |
| MLP no-op | 408 | 428 | 433 | 35.65s |

Approximate 16K time share from no-op deltas:

| Component | Time removed | Share of baseline TTFT |
|---|---:|---:|
| GDN path | ~25.0s | ~68% |
| GDN recurrent core | ~24.5s | ~66% |
| Attention path | ~9.7s | ~26% |
| MLP path | ~1.3s | ~4% |

These are measurement artifacts, not valid model-quality artifacts. The no-op
models intentionally produce broken text; only TTFT/prefill timing is used.

## Interpretation

The previous optimization attempts did not improve prefill because they targeted
the wrong level:

- `quantized_mlp_kernel_enabled` and the direct MLP NKI path did not matter
  because MLP contributes only a few percent of CTE latency.
- The head-loop wrapper around GDN did not matter because the traced Neuron graph
  is not dominated by Python loop dispatch. The time is inside the GDN device
  computation and surrounding dataflow.
- Attention is real but secondary. Removing attention core work improves 16K
  prefill from ~418 tok/s to ~566 tok/s, but removing GDN improves it to
  ~1292 tok/s.
- A narrower GDN recurrent-core no-op reaches ~1238 tok/s at 16K, nearly the
  same as the full GDN no-op. That means the recurrent DeltaNet chunk solve and
  state update are the dominant wall; the surrounding GDN projections/conv/output
  path is not the primary issue.

## Next Optimization

The next meaningful prefill speedup must change the GDN CTE implementation, not
MLP flags or shallow wrappers.

Highest-ROI direction:

1. Replace or substantially rewrite the recurrent DeltaNet chunk core first.
   The target is lower state traffic, fewer transposes/copies, and better SBUF
   reuse across the 128-token microchunk.
2. Only after that, consider fusing the surrounding QKV/Z/A/B projections,
   conv1d, norm/gate, and output projection into the same device-side dataflow.
3. Keep chunk size 128 initially. CTE=1024 did not help because it does not
   reduce the internal GDN work per 128-token microchunk.
4. Treat custom head_dim=256 attention as the second phase. Its maximum isolated
   win is meaningful but smaller than GDN.

Practical target:
- 1.3x total prefill improvement from serious recurrent-core dataflow work:
  ~540 tok/s.
- 2.0x total prefill improvement would require removing roughly half of GDN
  recurrent-core time, which likely means a true recurrent-core rewrite rather
  than a wrapper around the existing kernel.
