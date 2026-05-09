# Qwen3.6 27B Prefill Profile Report

Summary:

- Dominant bottleneck: compiled device execution per 512-token CTE call.
- Top recommended optimization: full-attention chunked CTE active-cache/head_dim=256 path.
- Expected speedup: 1.23-1.82x if attention is a 30-60% fraction.
- Effort: 1-2+ weeks.
- Ship at 150 tok/s instead: yes, if production stability matters more than another kernel project.

## Artifacts

- Driver report: `profile_driver.md`
- Runtime report: `profile_kernel.md`
- Decision matrix: `profile_decision.md`
- Artifact profiled: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_hybrid_chunked_nki_cte512_micro128_tkg65536_run1`
- Runtime trace: `/tmp/qwen36_prof_16k`

## Driver Findings

The driver profile measured 1K, 16K, and near-64K prompts using CTE bucket 512:

| Prompt tokens | Chunks | CTE seconds | tok/s | Avg chunk s | Decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 1024 | 2 | 6.851 | 149.47 | 3.425416 | 18.23 |
| 16384 | 32 | 109.443 | 149.70 | 3.420094 | 18.14 |
| 65472 | 128 | 437.766 | 149.56 | 3.420049 | 18.20 |

64K first and last quarters were essentially identical, so there is no visible cumulative-position slowdown at the driver level.

## Runtime Findings

`neuron-profile inspect` succeeded for the 16K run. Runtime event aggregation shows:

| Event | Count | Total s | Mean ms | Role |
|---|---:|---:|---:|---|
| `nrt_execute` | 156 | 453.534 | 2907.269 | Device execution envelope |
| `nc_exec_running` | 312 | 905.477 | 2902.169 | NeuronCore execution |
| `dmem_buf_copyin` | 12856 | 22.775 | 1.772 | Runtime copy traffic |
| `nrt_tensor_write` | 4968 | 22.516 | 4.532 | Runtime tensor writes |
| `nrt_model_submit` | 156 | 0.736 | 4.717 | Host submit overhead |

`nrt_execute` has 132 long executions averaging `3.421s`, matching the context execution envelope. Host submission is too small to explain the 3.42s chunk time.

`neuron-monitor` corroborates device-heavy execution: active NeuronCore utilization is generally ~68-100%, with max runtime HBM around 98.06 GB.

## Key Caveat

The runtime trace does not expose source-level layer names. It cannot directly say “DeltaNet is X%” or “attention is Y%.” The strongest actionable source-level finding comes from code inspection: current chunked full attention uses `cache_len = k_cache.shape[2]`, so for a 64K artifact it computes against the full 65,536-token cache every chunk, even early in the prompt.

## Decision

The next optimization should not be FP8 KV or host dispatch. Those are not the measured bottleneck.

If continuing speed work, prioritize the full-attention chunked CTE path: avoid full max-cache attention per chunk, or write/port a correct head_dim=256 prefix-attention kernel. DeltaNet chunk 256 remains plausible but should be gated by a targeted ablation before spending another week on kernel surgery.
