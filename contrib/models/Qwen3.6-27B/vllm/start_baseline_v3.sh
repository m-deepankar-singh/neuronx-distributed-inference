#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL_PATH="/opt/dlami/nvme/models/Qwen3.6-27B"
COMPILED_ARTIFACTS="/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1"
VENV="/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16"
BACKEND_HOST="127.0.0.1"
PROXY_HOST="0.0.0.0"
BACKEND_PORT="8001"
PROXY_PORT="8000"
LOG_DIR="/tmp"
RESTART="0"
INSTALL_REGISTRY="0"
RUN_SMOKE="1"
WAIT_SECONDS="900"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --compiled-artifacts) COMPILED_ARTIFACTS="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --backend-host) BACKEND_HOST="$2"; shift 2 ;;
    --proxy-host|--host) PROXY_HOST="$2"; shift 2 ;;
    --backend-port) BACKEND_PORT="$2"; shift 2 ;;
    --proxy-port) PROXY_PORT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --restart) RESTART="1"; shift ;;
    --install-registry) INSTALL_REGISTRY="1"; shift ;;
    --no-smoke) RUN_SMOKE="0"; shift ;;
    --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model path not found: ${MODEL_PATH}" >&2
  echo "Restore/download Qwen3.6-27B to this path or pass --model-path." >&2
  exit 2
fi

if [[ ! -d "${COMPILED_ARTIFACTS}" ]]; then
  echo "ERROR: compiled artifact not found: ${COMPILED_ARTIFACTS}" >&2
  echo "Restore/recompile the v3 artifact or pass --compiled-artifacts." >&2
  exit 2
fi

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "ERROR: vLLM Neuron venv not found: ${VENV}" >&2
  echo "Install/use the SDK 2.29 vLLM Neuron env or pass --venv." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

if [[ "${RESTART}" == "1" ]]; then
  pkill -f "serve_qwen36.py" 2>/dev/null || true
  pkill -f "qwen36_chat_proxy.py" 2>/dev/null || true
  sleep 3
fi

# shellcheck source=/dev/null
source "${VENV}/bin/activate"

if [[ "${INSTALL_REGISTRY}" == "1" ]]; then
  "${SCRIPT_DIR}/install_qwen36_vllm.sh" "${VENV}"
fi

BACKEND_LOG="${LOG_DIR}/qwen36_v3_backend.log"
PROXY_LOG="${LOG_DIR}/qwen36_v3_proxy.log"

echo "Starting Qwen3.6 vLLM APC baseline v3"
echo "repo=${REPO_ROOT}"
echo "model=${MODEL_PATH}"
echo "artifacts=${COMPILED_ARTIFACTS}"
echo "backend=http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "proxy=http://${PROXY_HOST}:${PROXY_PORT}"
echo "backend_log=${BACKEND_LOG}"
echo "proxy_log=${PROXY_LOG}"

nohup "${SCRIPT_DIR}/start_vllm_server.sh" \
  --model-path "${MODEL_PATH}" \
  --compiled-artifacts "${COMPILED_ARTIFACTS}" \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --block-size 256 \
  --enable-vllm-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --host "${BACKEND_HOST}" \
  --port "${BACKEND_PORT}" \
  >"${BACKEND_LOG}" 2>&1 &

BACKEND_PID="$!"
echo "backend_pid=${BACKEND_PID}"

deadline=$((SECONDS + WAIT_SECONDS))
until curl -sf "http://${BACKEND_HOST}:${BACKEND_PORT}/v1/models" >/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: backend did not become ready within ${WAIT_SECONDS}s" >&2
    tail -120 "${BACKEND_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done

nohup python "${SCRIPT_DIR}/qwen36_chat_proxy.py" \
  --backend-url "http://${BACKEND_HOST}:${BACKEND_PORT}" \
  --host "${PROXY_HOST}" \
  --port "${PROXY_PORT}" \
  >"${PROXY_LOG}" 2>&1 &

PROXY_PID="$!"
echo "proxy_pid=${PROXY_PID}"

deadline=$((SECONDS + 120))
until curl -sf "http://127.0.0.1:${PROXY_PORT}/v1/models" >/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: proxy did not become ready within 120s" >&2
    tail -80 "${PROXY_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done

if [[ "${RUN_SMOKE}" == "1" ]]; then
  SMOKE_PAYLOAD="$(
    MODEL_PATH="${MODEL_PATH}" python3 - <<'PY'
import json
import os

print(json.dumps({
    "model": os.environ["MODEL_PATH"],
    "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
    "max_tokens": 8,
    "temperature": 0,
    "top_k": 1,
}))
PY
  )"
  curl -sf "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "${SMOKE_PAYLOAD}" \
    | tee "${LOG_DIR}/qwen36_v3_smoke.json"
  echo
fi

echo "Qwen3.6 vLLM APC baseline v3 is ready on http://${PROXY_HOST}:${PROXY_PORT}"
