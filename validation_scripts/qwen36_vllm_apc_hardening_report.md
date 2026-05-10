# Qwen3.6-27B vLLM APC Hardening Report

## Setup

- Branch: `codex/qwen36-vllm-prefix-cache`
- Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1`
- Serving layout: vLLM backend on `127.0.0.1:8001`, guarded proxy on `:8000`
- Backend flags: `--enable-vllm-chunked-prefill --enable-prefix-caching --mamba-cache-mode align`
- Result files:
  - `validation_scripts/qwen36_vllm_apc_hardening_eval_20260510T101345Z.json`
  - `validation_scripts/qwen36_27b_vllm_long_chat_eval_20260510T101703Z.json`

## APC Correctness And Latency

| Case | Prompt tokens | Cold/first wall | Warm wall | Result |
|---|---:|---:|---:|---|
| Exact repeat | 10601 | 25.38s | 1.55s | exact text match, 16.35x |
| Cross-prefix A after B | 10376 | 25.17s | 1.36s | exact text match |

The cross-prefix case warms prefix A, warms a different prefix B, then reuses A.
This checks that cache reuse is not polluted by an unrelated prefix.

## Shared-Prefix Concurrency

All shared-prefix concurrent requests returned the requested marker exactly.
The backend is still compiled for `max_num_seqs=1`, so requests queue rather
than increase aggregate decode throughput.

| Concurrency | Success | Total wall | P50 wall | P95 wall | Aggregate prompt tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 1.48s | 1.48s | 1.48s | 7025.7 |
| 2 | 2/2 | 2.95s | 2.22s | 2.94s | 7047.7 |
| 4 | 4/4 | 5.84s | 3.66s | 5.84s | 7106.9 |

## Long-Context Quality

The APC-enabled server preserved the existing long-context quality gates:

| Case | Prompt tokens | Wall | Expected found |
|---|---:|---:|---|
| Math short chat | 41 | 1.37s | yes |
| Olympics short chat | 48 | 2.97s | yes |
| Needle 32K | 30638 | 72.15s | yes |
| Needle 64K | 61246 | 109.11s | yes |

## Verdict

APC passed the pre-baseline hardening gate:

- exact-repeat reuse is fast and output-stable;
- cross-prefix reuse remains isolated;
- shared-prefix concurrent requests are correct at concurrency 1/2/4;
- 32K and 64K long-context retrieval still pass.

This is enough to promote the APC-enabled vLLM setup to the next baseline,
with one clear caveat: throughput is still single-sequence queued because the
artifact is compiled with `max_num_seqs=1`.
