# Qwen3.6 27B Driver-Level Prefill Profile

Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_hybrid_chunked_nki_cte512_micro128_tkg65536_run1`  
Log: `/home/ubuntu/validation_logs/qwen36_27b_profile_driver_20260507_1935.log`  
Branch: `codex/qwen36-27b-64k-internal`  
Commits: `c598ea2` baseline, `7c046c3` profiling runner  

## Summary

| Prompt tokens | Chunks | CTE seconds | tok/s | Avg chunk s | Min chunk s | Max chunk s | Decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 2 | 6.851 | 149.47 | 3.425416 | 3.424343 | 3.426489 | 18.23 |
| 16384 | 32 | 109.443 | 149.70 | 3.420094 | 3.419857 | 3.420430 | 18.14 |
| 65472 | 128 | 437.766 | 149.56 | 3.420049 | 3.419766 | 3.420364 | 18.20 |

The driver-level result is internally consistent: throughput stays flat from 1K to near-64K, and the 64K run varies by less than 1 ms across 128 chunks. This strongly argues that cumulative-cache attention scaling is not the dominant visible cost at the CTE-call level.

## Representative 16K Chunks

| Chunk | Cache pos | Valid | Chunk total s | Deviation from avg ms |
|---:|---:|---:|---:|---:|
| 0 | 0 | 512 | 3.420430 | 0.336 |
| 8 | 4096 | 512 | 3.420219 | 0.124 |
| 16 | 8192 | 512 | 3.420226 | 0.132 |
| 24 | 12288 | 512 | 3.420119 | 0.025 |
| 31 | 15872 | 512 | 3.419871 | -0.223 |

## Representative 64K Chunks

| Chunk | Cache pos | Valid | Chunk total s | Deviation from avg ms |
|---:|---:|---:|---:|---:|
| 0 | 0 | 512 | 3.420282 | 0.234 |
| 32 | 16384 | 512 | 3.420121 | 0.072 |
| 64 | 32768 | 512 | 3.420101 | 0.052 |
| 96 | 49152 | 512 | 3.420106 | 0.057 |
| 127 | 65024 | 448 | 3.419979 | -0.070 |

## 64K First vs Last Quarter

| Window | Chunks | Avg chunk s | Min chunk s | Max chunk s | tok/s |
|---|---:|---:|---:|---:|---:|
| first_quarter | 32 | 3.420044 | 3.419850 | 3.420364 | 149.71 |
| last_quarter | 32 | 3.420068 | 3.419802 | 3.420223 | 149.12 |

## Category Breakdown Status

The current artifact is already compiled, so Python timers inside `NeuronGatedDeltaNet`, attention, MLP, and cache-update methods would not execute inside the NEFF. This driver profile therefore measures the compiled context-call boundary only. Inner categories (`deltanet_kernel_total`, `attention_forward_total`, `cache_update_total`, `mlp_total`, `layer_norm_total`, `other`) must come from `neuron-profile` execution traces in Step 3, not from Python timers, unless we accept a recompile with explicit traced timing outputs.

## Immediate Interpretation

- CTE call granularity is 512 input tokens per host call.
- Each 512-token call takes about 3.420 seconds.
- Sustained ingest is about 149.6 tok/s across 1K, 16K, and near-64K.
- Decode remains about 18.1-18.2 tok/s.
- Last-quarter 64K chunks are not slower than first-quarter chunks, so full-attention cache-length growth is not visible as the dominant driver-level bottleneck.
