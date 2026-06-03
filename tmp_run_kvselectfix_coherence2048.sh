#!/usr/bin/env bash
set -euo pipefail

RUN="/home/ubuntu/validation_logs/fp8_256k_decode_nki/kvselectfix_live_20260529T0835Z"
mkdir -p "${RUN}"
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

python - <<'PY'
import json
import time
import urllib.request
from pathlib import Path

run = Path("/home/ubuntu/validation_logs/fp8_256k_decode_nki/kvselectfix_live_20260529T0835Z")
payload = {
    "model": "/home/ubuntu/models/Qwen3.6-27B",
    "messages": [
        {
            "role": "user",
            "content": "What is life? Give a short final answer.",
        }
    ],
    "max_tokens": 2048,
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
    "enable_thinking": True,
}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
start = time.perf_counter()
first = None
parts = []
usage = None
events = 0
finish_reason = None
with urllib.request.urlopen(request, timeout=900) as response:
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        obj = json.loads(data)
        if isinstance(obj.get("usage"), dict):
            usage = obj["usage"]
        choices = obj.get("choices") or []
        if choices:
            finish_reason = choices[0].get("finish_reason") or finish_reason
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content:
                if first is None:
                    first = time.perf_counter() - start
                parts.append(content)
        events += 1
total = time.perf_counter() - start
text = "".join(parts)
final = text.split("</think>", 1)[1].strip() if "</think>" in text else ""
summary = {
    "events": events,
    "finish_reason": finish_reason,
    "ttft_seconds": first,
    "total_seconds": total,
    "usage": usage,
    "has_think_open": "<think>" in text,
    "has_think_close": "</think>" in text,
    "final_after_think_chars": len(final),
    "final_after_think_prefix": final[:300],
    "text_prefix": text[:500],
    "text_suffix": text[-500:],
}
(run / "coherence_thinking_2048_text.txt").write_text(text, encoding="utf-8")
(run / "coherence_thinking_2048_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True))
PY
