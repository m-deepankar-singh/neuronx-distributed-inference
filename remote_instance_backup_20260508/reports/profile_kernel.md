# Qwen3.6 27B Runtime/Profile Trace

Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_hybrid_chunked_nki_cte512_micro128_tkg65536_run1`  
Profile output: `/tmp/qwen36_prof_16k`  
Profile JSON: `/tmp/qwen36_prof_16k.json`  
Monitor log: `/tmp/qwen36_monitor_16k.log`  
Prompt: 16K, CTE bucket 512, max_new_tokens 5

## Capture Status

`neuron-profile inspect` succeeded after switching to absolute script paths. The summary export contains runtime event groups, not source-level layer or NKI kernel names. Therefore this report can classify runtime/device execution, copy, allocation, and system utilization, but it cannot directly name `deltanet_chunk_step` vs full-attention vs MLP from the trace alone.

## Driver Result Under Profiler

| Prompt tokens | Chunks | CTE seconds | tok/s | Avg chunk s | Decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 16384 | 32 | 109.454 | 149.69 | 3.420441 | 18.15 |

Profiler overhead did not materially change the driver-level result.

## Runtime Event Summary

| Event | Count | Total s | Mean ms | Max ms | Size GB | Interpreted role |
|---|---:|---:|---:|---:|---:|---|
| `nrt_execute` | 156 | 453.534 | 2907.269 | 3530.647 | 0.000 | Device execution envelope |
| `nc_exec_running` | 312 | 905.477 | 2902.169 | 3417.305 | 0.000 | NeuronCore execution, two logical NCs per run |
| `kmgr_exec_core` | 156 | 452.803 | 2902.585 | 3418.081 | 0.000 | Kernel manager execution |
| `kbl_exec_wait` | 156 | 452.770 | 2902.370 | 3417.385 | 0.000 | Host waiting for device execution |
| `dmem_buf_copyin` | 12856 | 22.775 | 1.772 | 269.683 | 94.010 | Runtime input/copy traffic |
| `nrt_tensor_write` | 4968 | 22.516 | 4.532 | 269.683 | 92.498 | Runtime tensor writes |
| `nrt_model_submit` | 156 | 0.736 | 4.717 | 114.019 | 0.000 | Host submit overhead |
| `nrt_tensor_read` | 86 | 0.004 | 0.046 | 1.556 | 0.022 | Runtime output reads |

`nrt_execute` distribution: 132 long executions with mean `3.421s`, 24 short executions with mean `0.082s`. The long group matches context execution envelopes across TP ranks plus warmup/context calls. The short group matches token generation and small auxiliary executions.

## Copy/Bandwidth Signals

| Event | Total GB | Total s | Effective GB/s |
|---|---:|---:|---:|
| `dmem_buf_copyin` | 94.010 | 22.775 | 4.13 |
| `nrt_tensor_write` | 92.498 | 22.516 | 4.11 |
| `dmem_buf_copyout` | 0.022 | 0.004 | 5.70 |
| `nrt_tensor_read` | 0.022 | 0.004 | 5.64 |

These are runtime copy signals, not full in-kernel HBM traffic. They do not indicate host copy is the dominant bottleneck: submit overhead is sub-second total, and device execution/wait time dominates.

## neuron-monitor Signals

| Metric | Value |
|---|---:|
| Max Neuron device memory | 98.06 GB |
| Max host memory attributed to runtime | 83.00 GB |
| Active average NeuronCore utilization | ~79.3% |
| Active utilization range | ~68.3-100% steady-state samples |
| Average effective FLOPs per active sample | ~4.0 TF/s per NeuronCore sample |
| Peak effective FLOPs per sample | ~5.7 TF/s |

The monitor corroborates that the workload is device-execution dominated, not host-submit dominated. Utilization is not near zero, so this is not primarily Python dispatch overhead.

## Important Interpretation Caveat

The flat 1K-to-64K driver throughput does **not** by itself prove attention is cheap. In the current chunked attention implementation, full-attention layers build `K_full` from the full cache tensor and use `cache_len = k_cache.shape[2]`, which is the compiled maximum sequence length (`65536`). That means each chunk can pay a fixed full-cache attention cost regardless of current cumulative position. Flat chunk time rules out growing cumulative-position overhead, but it does not rule out fixed max-cache attention work.

## Verdict

- Host submit/dispatch overhead is not the main bottleneck (`nrt_model_submit` total `0.736s` vs `109.454s` wall CTE).
- Runtime copy/write traffic is visible but not dominant enough to justify FP8 KV/cache work as the first optimization.
- Device execution is the dominant envelope (`~3.42s` per 512-token context call).
- The trace is insufficient to split device time between DeltaNet, full attention, and MLP by name. The code path plus flat timing make two fixed per-token/per-bucket suspects: internal DeltaNet microchunk work and full-cache attention over `65536` cache positions.
