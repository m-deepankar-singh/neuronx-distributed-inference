# Qwen3.6-27B vLLM Concurrency Eval

## Setup

- Branch: `codex/qwen36-vllm-support`
- Baseline commit: `42b793c`
- Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1`
- Serving layout: proxy on `:8000`, vLLM backend on `127.0.0.1:8001`
- Backend config: `max-num-seqs=1`, `max-num-batched-tokens=512`, `block-size=256`

## Result

All requests completed successfully at concurrency `1`, `2`, and `4`; the proxy
kept chat formatting and request IDs correct. Throughput did not increase with
concurrency because the backend is intentionally configured for one active
sequence. Extra client requests queue behind the active request.

## Short-Answer 2K-Prompt Run

Result file:

```text
validation_scripts/qwen36_27b_vllm_concurrency_eval_20260510T093027Z.json
```

| Concurrency | Success | Total Wall | P50 Wall | P95 Wall | Prompt tok/s | Completion tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 6.97s | 6.97s | 6.97s | 347.3 | 3.9 |
| 2 | 2/2 | 13.35s | 10.02s | 13.35s | 362.5 | 3.0 |
| 4 | 4/4 | 26.22s | 16.89s | 26.21s | 369.3 | 2.6 |

## Decode-Heavy Run

Result file:

```text
validation_scripts/qwen36_27b_vllm_concurrency_eval_20260510T093227Z.json
```

| Concurrency | Success | Total Wall | P50 Wall | P95 Wall | Prompt tok/s | Completion tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 7.22s | 7.22s | 7.22s | 50.0 | 22.2 |
| 2 | 2/2 | 14.37s | 10.78s | 14.37s | 50.2 | 22.3 |
| 4 | 4/4 | 28.77s | 17.99s | 28.77s | 50.2 | 22.2 |

## Interpretation

The system is correct under concurrent client load but not throughput-scaled for
multiple simultaneous users on this artifact. The decode-heavy run is the
clearest signal: aggregate completion throughput stays flat at about `22.2
tok/s`, while tail latency scales linearly with the number of queued requests.

For real multi-user throughput, the next work is not another client/server
wrapper. It requires changing the serving/model setup: larger `max_num_seqs`
with a matching compiled artifact, continuous batching compatibility for the
hybrid cache, or multiple model replicas behind a load balancer.
