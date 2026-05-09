# Qwen3.6-27B Hybrid — Production Baseline

This document establishes the canonical production baseline for the Qwen3.6-27B
hybrid model on AWS Trainium2. All future optimization work must branch off
this baseline and report deltas relative to it.

## Baseline Reference

| Field | Value |
|---|---|
| **Tag** | `qwen36-27b-baseline-v1` |
| **Commit** | `99e24fc` |
| **Branch** | `codex/qwen36-27b-64k-clean-recovery` |
| **Date validated** | 2026-05-09 |
| **Hardware** | trn2.3xlarge, 4 NeuronCores, TP=4 |
| **Precision** | BF16 |
| **SDK / NKI** | Neuron SDK 2.29, NKI 0.3.0+ |

## Configuration

```python
NeuronConfig(
    tp_degree=4,
    batch_size=1,
    ctx_batch_size=1,
    tkg_batch_size=1,
    seq_len=65536,
    max_context_length=512,        # CRITICAL: not seq_len
    max_length=65536,
    context_encoding_buckets=[512],
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    enable_bucketing=False,
    flash_decoding_enabled=False,
    logical_nc_config=2,
    save_sharded_checkpoint=True,
)
```

Hybrid-specific config flags (passed via `Qwen35InferenceConfig`):

```python
use_hybrid_cache_manager = True
use_qwen_hybrid_chunked_prefill = True
use_qwen_hybrid_chunked_prefill_nki = True
```

## Validated Performance

| Prompt size | Prefill time | Tok/s sustained | Decode tok/s | Notes |
|---:|---:|---:|---:|---|
| 1K | 6.85s | 149.47 | 18.23 | 2 chunks |
| 16K | 109.4s | 149.70 | 18.14 | 32 chunks |
| 64K | 437.8s | 149.56 | 18.20 | 128 chunks |

- **Prefill rate is FLAT** across context sizes (per-chunk cost dominated by
  GDN compute, not cumulative cache attention).
- **Effective max prompt** = `seq_len - decode_budget`. For default decode
  budget of 16, max effective prompt is 65520 tokens.
- **Speedup vs TKG-prefill baseline**: 7.7× (18 → 150 tok/s prefill).

## Compile Profile

| Phase | Time |
|---|---|
| HLO generation (CTE + TKG) | 105.4s |
| TKG NEFF compile | 286.5s |
| Weight layout optimization | 17.6s |
| CTE NEFF compile | 309.6s |
| **Total build** | **1021.5s (17.0 min)** |
| Sharding weights (TP=4) | 157.0s |
| Weight load | 89.7s |
| Warmup | 3.8s |

Artifact size on disk: **82 GB** (sharded NEFF + weights).

## Architecture

- **64 hidden layers**: 48 GDN (linear attention) + 16 GQA (full attention)
- **head_dim**: 256 (full attention layers)
- **head_v_dim / head_k_dim**: 128 (DeltaNet linear attention)
- **48 value heads, 16 key heads** (DeltaNet)
- **24 Q heads, 4 KV heads** (full attention, 6:1 GQA)
- **partial_rotary_factor**: 0.25 (only first 64 of 256 head_dim get RoPE)

## Critical Implementation Details

### Custom NKI Kernel

The DeltaNet chunked kernel uses **forward substitution** math ported from the
validated 2B fused kernel. Replaced the original Neumann power-doubling
implementation that produced cosine 0.9156 in BF16. Current kernel:

- File: `contrib/models/Qwen3.6-27B/src/nki_kernels/nki_deltanet_chunked.py`
- Math: `exp(cumsum(g)_i - cumsum(g)_j)` (combined exp form, no split)
- Triangular solve: 64-block forward substitution
- Recurrent state: fp32 throughout
- Conv state: fp32

### HybridCacheManager

Per-layer-type cache allocation:
- GDN layers (48): recurrent_state `[B, H, K, V]` + conv_state `[B, channels, kernel-1]`
- Attention layers (16): full KV cache `[B, H_kv, seq, head_dim]`
- Layer-type aware to avoid dummy KV bloat for GDN layers

### Chunked Prefill Driver

Custom `perform_qwen_chunked_prefill` function:
- Splits prompt into `cte_bucket=512` chunks
- Threads GDN recurrent state across chunk boundaries
- Threads GDN conv state (last `kernel_size - 1` valid activations) across chunks
- Uses absolute position_ids for each chunk
- Cache writes go to shared 64K KV storage
- Attention compute uses full `cache_len = k_cache.shape[2]` (= 65536) per chunk
  (no active-cache optimization — confirmed not the bottleneck via ablation)

## Known Bottlenecks (do not chase without ablation)

| Bottleneck | Share of chunk time | Why |
|---|---|---|
| GDN per-chunk dispatch | dominant | `for bh in range(48)` in `_fused_chunked_forward` |
| MLP layers | ~20-25% | 64 MLP layers per chunk |
| Attention (full cache) | **~5%** | Confirmed by ablation (4% speedup at attn_cache=4096) |
| Layer norms + projections | ~10% | Constant per chunk |
| Per-chunk Python orchestration | ~5-10% | Dispatch overhead |

## Validated Quality

- "The capital of France is" → "Paris" ✓
- 762, 1K, 4K, 16K, 64K prompts: coherent, no invalid token IDs
- Token-level semantic coherence preserved (forward substitution improves
  precision over Neumann; cosine ≥0.999 vs HF reference)

## Compile Recipe

```bash
ssh -i trainium.pem ubuntu@<instance>
cd /home/ubuntu/inferentia-gdn
git checkout qwen36-27b-baseline-v1   # or codex/qwen36-27b-64k-clean-recovery
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

python /opt/dlami/nvme/qwen36_27b_compile_baseline.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-path /opt/dlami/nvme/qwen_artifacts/qwen36_27b_64k_baseline \
  --variant baseline \
  --seq-len 65536 \
  --cte-bucket 512 \
  --tp-degree 4 \
  --logical-nc-config 2 \
  --load-after-compile
```

Expected: ~17 min compile + ~5 min sharding/load = ~22 min end-to-end.

## Optimization Arc (Future Work)

All future improvements branch off `qwen36-27b-baseline-v1` and report
speedup vs this baseline. Documented arc:

| Stage | Optimization | Expected speedup | Effort |
|---|---|---|---|
| v1.1 | Config flags (async_mode, mlp_kernel, fused_qkv) | 1.15-1.30× | 1 day |
| v1.2 | FP8 weight quantization (NxDI native) | 1.40× prefill, 1.50× decode | 2 weeks |
| v1.3 | Head-batched GDN kernel (eliminate per-(B,H) loop) | 1.20-1.30× prefill | 1.5 weeks |
| v1.4 | FP8 KV cache | 1.15-1.20× prefill | 1 week |
| v1.5 | Speculative decoding (EAGLE/Medusa) | 1.5-2× decode | 2-3 weeks |
| v1.6 | Prefix caching (custom hybrid cache integration) | 5-50× for repeated prefixes | 1-2 weeks |

Realistic combined ceiling: ~300-400 tok/s prefill, ~50-60 tok/s decode.

## Branching Convention

```
qwen36-27b-baseline-v1                    ← immutable tag at 99e24fc
└── codex/qwen36-27b-64k-clean-recovery   ← canonical baseline branch
    ├── feature/fp8-weights               ← v1.2 work
    ├── feature/head-batched-gdn          ← v1.3 work
    ├── feature/fp8-kv-cache              ← v1.4 work
    └── feature/speculative-decoding      ← v1.5 work
```

Each feature branch must:
1. Branch off `qwen36-27b-baseline-v1` (or clean-recovery if no baseline drift)
2. Document config delta from baseline in feature commit
3. Report performance delta (tok/s, HBM, compile time) vs baseline
4. Pass cosine ≥0.999 vs baseline at first decode step (correctness gate)
5. Pass 64K end-to-end coherence test (no invalid IDs, valid output)
