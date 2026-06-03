#!/usr/bin/env bash
set -euo pipefail

RUN="/home/ubuntu/validation_logs/fp8_256k_decode_nki/kvselectfix_live_20260529T0835Z"
MODEL="/home/ubuntu/models/Qwen3.6-27B"
BASE_URL="http://127.0.0.1:8000"

mkdir -p "${RUN}"
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
cd /home/ubuntu/inferentia-gdn

python validation_scripts/qwen36_chat_completion_context_bench.py \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --model-path "${MODEL}" \
  --lengths 1024,4096,8192,16384 \
  --repeats 1 \
  --turns 8 \
  --max-tokens 1 \
  --unique-per-request \
  --output-json "${RUN}/cold_prefill.json"

python validation_scripts/qwen36_chat_completion_context_bench.py \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --model-path "${MODEL}" \
  --lengths 8192 \
  --repeats 1 \
  --turns 8 \
  --max-tokens 128 \
  --ignore-eos \
  --unique-per-request \
  --output-json "${RUN}/decode_8k_128.json"

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
    "max_tokens": 512,
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
with urllib.request.urlopen(request, timeout=600) as response:
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
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content:
                if first is None:
                    first = time.perf_counter() - start
                parts.append(content)
        events += 1
total = time.perf_counter() - start
text = "".join(parts)
summary = {
    "events": events,
    "ttft_seconds": first,
    "total_seconds": total,
    "usage": usage,
    "has_think_open": "<think>" in text,
    "has_think_close": "</think>" in text,
    "final_after_think_chars": len(text.split("</think>", 1)[1].strip())
    if "</think>" in text
    else 0,
    "text_prefix": text[:500],
    "text_suffix": text[-500:],
}
(run / "coherence_thinking_text.txt").write_text(text, encoding="utf-8")
(run / "coherence_thinking_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True))
PY

python - <<'PY'
import json
from pathlib import Path

run = Path("/home/ubuntu/validation_logs/fp8_256k_decode_nki/kvselectfix_live_20260529T0835Z")
for name in ("cold_prefill", "decode_8k_128"):
    data = json.loads((run / f"{name}.json").read_text())
    print(f"SUMMARY {name}")
    for row in data["results"]:
        prompt = int(row["prompt_tokens"])
        ttft = row.get("ttft_seconds")
        total = row.get("total_seconds")
        tps = prompt / ttft if ttft else None
        print(
            json.dumps(
                {
                    "target_tokens": row["target_tokens"],
                    "prompt_tokens": prompt,
                    "ttft_seconds": ttft,
                    "prefill_tokens_per_second_by_ttft": tps,
                    "total_seconds": total,
                    "completion_tokens": row.get("completion_tokens"),
                    "completion_token_source": row.get("completion_token_source"),
                    "token_tpot_seconds": row.get("token_tpot_seconds"),
                    "decode_tokens_per_second": row.get("decode_tokens_per_second"),
                    "status": row.get("status"),
                },
                sort_keys=True,
            )
        )
PY
