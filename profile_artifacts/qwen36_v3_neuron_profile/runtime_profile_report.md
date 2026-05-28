# Qwen3.6-27B v3 Neuron Runtime Profile

## Environment

- Instance: trn2.48xlarge (`18.220.20.118`), 64 NeuronCores visible
- Repo: `/opt/dlami/nvme/qwen36_v3_profile/inferentia-gdn`
- Branch: `contrib/qwen36-27b-vllm-apc-pr @ 6d6ae62`
- Profile venv: `/opt/dlami/nvme/venvs/qwen36-v3-profile`
- Model: `/opt/dlami/nvme/models/Qwen3.6-27B`
- Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_v3_profile`
- Profile capture: `/opt/dlami/nvme/profiles/qwen36_v3/vllm_runtime_inspect_20260514_073005`

## Workload

- Runtime path: vLLM-Neuron using precompiled artifact
- Context: 128K-capable artifact (`seq_len=131072`), CTE bucket `512`, TP=4, LNC=2
- vLLM: prefix caching enabled, chunked prefill enabled, block size 256, mamba cache mode `align`
- Prompt tokens: 3445
- Decode tokens: 32
- Runs: cold request, then same prompt again for APC/warm request

## Runtime Results

| Run | Wall Time | vLLM Input Speed | vLLM Output Speed | Notes |
|---|---:|---:|---:|---|
| Cold | 11.238878 s | 306.77 tok/s | 2.85 tok/s | Full prefill + decode |
| Warm/APC | 2.457338 s | 1406.21 tok/s | 13.06 tok/s | Prefix-cache hit path |

- APC speedup on this 3445-token request: **4.574x**
- The warm and cold text snippets were coherent. This profiling workload did not enforce exact token equality; use the separate APC validation harness for correctness gates.

## Neuron Explorer System Trace

Exported files:

- `summary_system.txt`
- `summary_system.json`
- `system_profile.pftrace` (Perfetto UI trace, 25 MB)
- `system_profile.json` (raw trace JSON, 94 MB)
- raw protobufs: `ntrace.pb`, `trace_info.pb`, `host_mem.pb`, `cpu_util.pb`

System summary:

- Trace events: 138,892
- Runtime trace span: 31.900 s
- `nrt_execute`: 292 calls, total summed duration 61.563 s, average 210.831 ms, max 1654.788 ms
- `nrt_model_submit`: 292 calls, total 1.481 s, average 5.071 ms
- `nc_model_switch`: 56 calls
- `nrt_load`: 12 calls, total 42.577 s
- `nrt_load_collectives`: 8 calls, total 40.919 s
- `nrt_tensor_allocate`: 4,920 calls
- `nrt_tensor_free`: 41,388 calls
- `nrt_tensor_write`: 6,128 calls, total 10.350 s
- `dmem_buf_copyin`: 20,776 calls, total 11.108 s

Interpretation:

- The trace captured both engine startup/load and two inference requests. The large `nrt_load*` totals are load-time, not steady-state inference latency.
- For inference, the important counters are the 292 `nrt_execute`/`nrt_model_submit` calls and the tensor copy/write volume. These show the request is still dominated by many runtime executes plus tensor movement rather than a single long compute kernel.
- Warm/APC request reduced wall time from 11.24 s to 2.46 s for a 3445-token prompt, proving the vLLM prefix-cache path is active on this baseline.

## Static NEFF Profile References

Static compiler/device summaries captured earlier:

- CTE rank0: `/opt/dlami/nvme/profiles/qwen36_v3/context_encoding_model/summary_rank0.txt`
  - total_time: 1.448317723667 s
  - HBM read: 67,550,784,514 bytes
  - HBM write: 34,337,804,460 bytes
  - DMA active: 34.91%
  - tensor engine active: 18.23%
  - vector engine active: 27.65%
- TKG rank0: `/opt/dlami/nvme/profiles/qwen36_v3/token_generation_model/summary_rank0.txt`
  - total_time: 0.029915778256 s
  - HBM read: 10,785,258,698 bytes
  - HBM write: 61,167,932 bytes
  - DMA active: 89.30%

## Next Useful Profiling Runs

1. Repeat runtime inspect with a larger prompt (16K or 64K) and fewer startup effects by loading once, warming once, then profiling only the measured requests.
2. Capture concurrency profile with 2, 4, and 8 simultaneous requests to expose scheduler gaps and prefix-cache hit behavior under load.
3. Capture MTP/minimal-spec artifact separately and compare `nrt_execute` calls per generated token against baseline v3.
4. Use `system_profile.pftrace` in Perfetto to inspect host gaps between `nrt_execute` calls and identify scheduler/runtime idle regions.
