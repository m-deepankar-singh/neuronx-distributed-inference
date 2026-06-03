#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z}
MODEL=${MODEL:-/home/ubuntu/models/Qwen3.6-27B}
MODEL_NAME=${MODEL_NAME:-${MODEL}}
ARTIFACT=${ARTIFACT:-/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_stable_probe32k_gdnrecbfloat16_sampletokens_outlogits_b256_cte512_pfx16k_slots64_async_cteargmaxsafe_20260601T124732Z}
ROOT=${ROOT:-/home/ubuntu/validation_logs/fp8_256k_decode_nki/outlogits_probe32k_coldprefill_$(date -u +%Y%m%dT%H%M%SZ)}
VENV=${VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}
BACKEND_HOST=${BACKEND_HOST:-127.0.0.1}
BACKEND_PORT=${BACKEND_PORT:-8001}
PROXY_HOST=${PROXY_HOST:-127.0.0.1}
PROXY_PORT=${PROXY_PORT:-8000}
LENGTHS=${LENGTHS:-16384}
MAX_TOKENS=${MAX_TOKENS:-8}
REPEATS=${REPEATS:-1}
TIMEOUT=${TIMEOUT:-1800}
RUN_COHERENCE=${RUN_COHERENCE:-1}
RUNTIME_CTE_BUCKETS=${RUNTIME_CTE_BUCKETS:-512}
RUNTIME_CONTEXT_ENCODING_BUCKET_PAIRS=${RUNTIME_CONTEXT_ENCODING_BUCKET_PAIRS:-512:256,512:512,512:1024,512:2048,512:4096,512:8192,512:16384}
RUNTIME_TOKEN_GENERATION_BUCKETS=${RUNTIME_TOKEN_GENERATION_BUCKETS:-512,16384,16640,32768}
RUNTIME_HYBRID_APC_PREFILL_CHUNK_TOKENS=${RUNTIME_HYBRID_APC_PREFILL_CHUNK_TOKENS:-}

mkdir -p "${ROOT}/server" "${ROOT}/bench" "${ROOT}/coherence"

log() {
  printf "[%s] %s\n" "$(date -Is)" "$*" | tee -a "${ROOT}/suite.log"
}

cleanup() {
  set +e
  if [[ -f "${ROOT}/server/proxy.pid" ]]; then
    kill "$(cat "${ROOT}/server/proxy.pid")" 2>/dev/null || true
  fi
  if [[ -f "${ROOT}/server/backend.pid" ]]; then
    kill "$(cat "${ROOT}/server/backend.pid")" 2>/dev/null || true
  fi
}
trap cleanup EXIT

old_pids="$(pgrep -f "qwen36_chat_proxy.py|serve_qwen36.py" || true)"
if [[ -n "${old_pids}" ]]; then
  log "STOP_OLD_PIDS ${old_pids}"
  kill ${old_pids} 2>/dev/null || true
  sleep 5
fi
leftover_pids="$(pgrep -f "qwen36_chat_proxy.py|serve_qwen36.py" || true)"
if [[ -n "${leftover_pids}" ]]; then
  log "FORCE_STOP_OLD_PIDS ${leftover_pids}"
  kill -9 ${leftover_pids} 2>/dev/null || true
  sleep 2
fi

cd "${REPO}"
source "${VENV}/bin/activate"
export PYTHONPATH="${REPO}/src:${REPO}/contrib/models/Qwen3.6-27B:${REPO}/contrib/models/Qwen3.6-27B/vllm:${REPO}/validation_scripts:${PYTHONPATH:-}"
export NEURON_RT_INSPECT_ENABLE=${NEURON_RT_INSPECT_ENABLE:-0}
export NEURON_RT_INSPECT_DEVICE_PROFILE=${NEURON_RT_INSPECT_DEVICE_PROFILE:-0}
export QWEN36_HYBRID_APC_DEBUG=${QWEN36_HYBRID_APC_DEBUG:-1}
export QWEN36_SAMPLE_LOGITS_COMPARE_JSONL=${QWEN36_SAMPLE_LOGITS_COMPARE_JSONL:-${ROOT}/server/sample_logits_compare.jsonl}

log "START backend artifact=${ARTIFACT}"
prefill_chunk_args=()
if [[ -n "${RUNTIME_HYBRID_APC_PREFILL_CHUNK_TOKENS}" ]]; then
  prefill_chunk_args=(--hybrid-apc-prefill-chunk-tokens "${RUNTIME_HYBRID_APC_PREFILL_CHUNK_TOKENS}")
fi
nohup bash contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path "${MODEL}" \
  --compiled-artifacts "${ARTIFACT}" \
  --seq-len 32768 \
  --max-model-len 32768 \
  --cte-buckets "${RUNTIME_CTE_BUCKETS}" \
  --context-encoding-bucket-pairs "${RUNTIME_CONTEXT_ENCODING_BUCKET_PAIRS}" \
  --token-generation-buckets "${RUNTIME_TOKEN_GENERATION_BUCKETS}" \
  --tensor-parallel-size 4 \
  --logical-nc-config 2 \
  --max-num-seqs 1 \
  --ctx-batch-size 1 \
  --async-mode \
  --enable-vllm-chunked-prefill \
  --enable-prefix-caching \
  --enable-hybrid-apc \
  --block-size 256 \
  --gdn-checkpoint-interval 256 \
  --max-gdn-checkpoint-slots 64 \
  --gdn-recurrent-cache-dtype bfloat16 \
  --gdn-conv-cache-dtype bfloat16 \
  --hybrid-gdn-recurrent-cache-dtype bfloat16 \
  --hybrid-gdn-conv-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype auto \
  --hybrid-cache-mode all \
  --hybrid-cache-prefix-boundary-only \
  --hybrid-apc-require-vllm-metadata \
  --hybrid-apc-enable-backed-prefix-reads \
  "${prefill_chunk_args[@]}" \
  --num-gpu-blocks-override 128 \
  --host "${BACKEND_HOST}" \
  --port "${BACKEND_PORT}" \
  >"${ROOT}/server/backend.log" 2>&1 &
echo "$!" >"${ROOT}/server/backend.pid"

backend_url="http://${BACKEND_HOST}:${BACKEND_PORT}"
backend_ready=0
for attempt in $(seq 1 240); do
  if curl -fsS "${backend_url}/v1/models" >"${ROOT}/server/backend_models.json" 2>"${ROOT}/server/backend_models.err"; then
    backend_ready=1
    break
  fi
  if ! kill -0 "$(cat "${ROOT}/server/backend.pid")" 2>/dev/null; then
    log "BACKEND_EXITED"
    tail -160 "${ROOT}/server/backend.log" || true
    exit 20
  fi
  if (( attempt % 12 == 0 )); then
    log "WAIT_BACKEND attempt=${attempt}"
    tail -40 "${ROOT}/server/backend.log" | sed 's/^/BACKEND_LOG /' | tee -a "${ROOT}/suite.log" || true
  fi
  sleep 5
done
if [[ "${backend_ready}" != "1" ]]; then
  log "BACKEND_READY_TIMEOUT"
  tail -160 "${ROOT}/server/backend.log" || true
  exit 21
fi
log "BACKEND_READY ${backend_url}"

log "START proxy"
nohup python contrib/models/Qwen3.6-27B/vllm/qwen36_chat_proxy.py \
  --backend-url "${backend_url}" \
  --host "${PROXY_HOST}" \
  --port "${PROXY_PORT}" \
  >"${ROOT}/server/proxy.log" 2>&1 &
echo "$!" >"${ROOT}/server/proxy.pid"

proxy_url="http://${PROXY_HOST}:${PROXY_PORT}"
proxy_ready=0
for attempt in $(seq 1 60); do
  if curl -fsS "${proxy_url}/v1/models" >"${ROOT}/server/proxy_models.json" 2>"${ROOT}/server/proxy_models.err"; then
    proxy_ready=1
    break
  fi
  if ! kill -0 "$(cat "${ROOT}/server/proxy.pid")" 2>/dev/null; then
    log "PROXY_EXITED"
    tail -120 "${ROOT}/server/proxy.log" || true
    exit 22
  fi
  sleep 2
done
if [[ "${proxy_ready}" != "1" ]]; then
  log "PROXY_READY_TIMEOUT"
  tail -120 "${ROOT}/server/proxy.log" || true
  exit 23
fi
log "PROXY_READY ${proxy_url}"

if [[ "${RUN_COHERENCE}" == "1" ]]; then
  log "START coherence"
  python - "${proxy_url}" "${MODEL_NAME}" "${ROOT}/coherence/response.json" <<'PY'
import json
import sys
import urllib.request

base_url, model, output_path = sys.argv[1:4]
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You answer directly and concisely."},
        {"role": "user", "content": "Name one practical reason to use smaller prefill chunks."},
    ],
    "max_tokens": 32,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
    "stream": False,
}
request = urllib.request.Request(
    base_url.rstrip("/") + "/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=900) as response:
    parsed = json.loads(response.read().decode("utf-8"))
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(parsed, handle, indent=2, sort_keys=True)
text = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
print(json.dumps({"content_prefix": text[:240], "usage": parsed.get("usage")}, sort_keys=True))
PY
else
  log "SKIP coherence"
fi

log "START cold_prefill_bench lengths=${LENGTHS} max_tokens=${MAX_TOKENS}"
python validation_scripts/qwen36_chat_completion_context_bench.py \
  --base-url "${proxy_url}" \
  --model "${MODEL_NAME}" \
  --model-path "${MODEL}" \
  --lengths "${LENGTHS}" \
  --repeats "${REPEATS}" \
  --concurrency 1 \
  --max-tokens "${MAX_TOKENS}" \
  --timeout "${TIMEOUT}" \
  --ignore-eos \
  --unique-per-request \
  --output-json "${ROOT}/bench/chat_cold_prefill.json" \
  >"${ROOT}/bench/chat_cold_prefill.log" 2>&1

python - "${ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
bench = json.loads((root / "bench/chat_cold_prefill.json").read_text())
summary = {
    "root": str(root),
    "passed": bench.get("passed"),
    "lengths": bench.get("lengths"),
    "max_tokens": bench.get("max_tokens"),
    "results": [],
}
for row in bench.get("results", []):
    summary["results"].append({
        "target_tokens": row.get("target_tokens"),
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "completion_token_source": row.get("completion_token_source"),
        "ttft_seconds": row.get("ttft_seconds"),
        "total_seconds": row.get("total_seconds"),
        "group_effective_prompt_tokens_per_second": row.get("group_effective_prompt_tokens_per_second"),
        "token_tpot_seconds": row.get("token_tpot_seconds"),
        "decode_tokens_per_second": row.get("decode_tokens_per_second"),
        "content_text_prefix": (row.get("content_text") or "")[:160],
        "error": row.get("error"),
    })
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

log "DONE"
