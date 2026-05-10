# Qwen3.6-27B vLLM Streaming Report

## Setup

- Branch: `codex/qwen36-vllm-prefix-cache`
- Server: baseline v3 backend on `127.0.0.1:8001`, guarded proxy on `:8000`
- Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1`

## Change

`qwen36_chat_proxy.py` now preserves streaming responses. For requests with
`"stream": true`, the proxy forwards upstream Server-Sent Events line by line
and flushes each chunk instead of buffering the whole response.

## Validation

Command run on the Trainium instance:

```bash
curl -NsS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/opt/dlami/nvme/models/Qwen3.6-27B",
    "messages": [{"role": "user", "content": "Count from 1 to 8, separated by commas."}],
    "max_tokens": 32,
    "temperature": 0,
    "top_k": 1,
    "stream": true
  }'
```

Observed output arrived as incremental SSE chunks:

```text
data: ... "content":"1" ...
data: ... "content":"," ...
data: ... "content":" " ...
data: ... "content":"2" ...
...
data: ... "content":"8" ...
data: [DONE]
```

Verdict: streaming works through the guarded proxy.
