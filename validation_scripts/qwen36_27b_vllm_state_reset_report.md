# Qwen3.6-27B vLLM State Reset Validation

## Summary

Validated a production-safe Qwen3.6 vLLM serving path with:

- backend vLLM on `127.0.0.1:8001`;
- guard proxy on `0.0.0.0:8000`;
- `chat_template_kwargs={"enable_thinking": false}` forced for chat requests;
- raw `/v1/completions` rejected by default;
- DeltaNet recurrent/conv state reset at context position 0.

## Issue

The existing vLLM artifact could pass a first long request, then fail the next
short chat request with stale hybrid state symptoms (`</think>` loops or wrong
short answers). The root cause was that vLLM reuses `seq_id=0` across requests.
Attention KV is position-masked, but Qwen3.6 DeltaNet recurrent and conv states
were read from the prior request when a new context pass started at position 0.

## Fix

In `NeuronGatedDeltaNet.forward`, when hybrid chunked CTE receives cached
DeltaNet state and `position_ids[:, 0] == 0`, the initial recurrent state and
conv state are multiplied by zero before running the chunk. Later chunks keep
using the carried state normally.

Artifact compiled from the fix:

```text
/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1
```

Compile log:

```text
validation_scripts/qwen36_fp8_vllm_statereset_128k_run1.log
```

## Validation

Short gates through the proxy passed without callers providing
`chat_template_kwargs`:

| Case | Result |
|---|---:|
| Math, fresh server | `3151` |
| Olympics, fresh server | Tokyo/COVID answer |
| Raw `/v1/completions` | HTTP 400 |

State-reset regression:

| Sequence | Result |
|---|---:|
| Short math before long request | `3151` |
| 32K needle request | all 3 codes found |
| Short math after 32K request | `3151` |
| 64K needle request | all 3 codes found |
| Short math after 64K request | `3151` |

Long-context runs:

| Run | Prompt Tokens | Wall Time | Effective Prompt tok/s | Result |
|---|---:|---:|---:|---|
| 32K needle | 30,638 | 72.15s | 424.6 | pass |
| 64K needle | 61,246 | 143.63s | 426.4 | pass |

Result files:

```text
validation_scripts/qwen36_27b_vllm_long_chat_eval_20260510T091222Z.json
validation_scripts/qwen36_27b_vllm_long_chat_eval_20260510T091617Z.json
```

## Serving Command

Backend:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --block-size 256 \
  --enable-vllm-chunked-prefill \
  --port 8001
```

Proxy:

```bash
python contrib/models/Qwen3.6-27B/vllm/qwen36_chat_proxy.py \
  --backend-url http://127.0.0.1:8001 \
  --port 8000
```
