#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ubuntu/inferentia-gdn-prefill-speed-coherent}"
MODEL="${MODEL:-/home/ubuntu/models/Qwen3.6-27B}"
VENV="${VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}"
TOOL="${TOOL:-/opt/aws/neuron/bin/neuron-explorer}"
ARTIFACT="${ARTIFACT:-/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T152927Z_direct_scan0_recbf16fix}"
PROFILE_ROOT="${PROFILE_ROOT:-/home/ubuntu/validation_logs/fp8_256k_decode_nki/qkvnki_prefill_profile_$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${PORT:-8001}"
PROMPT_TOKENS="${PROMPT_TOKENS:-16384}"
MAX_TOKENS="${MAX_TOKENS:-1}"
RESTART_NORMAL="${RESTART_NORMAL:-1}"

mkdir -p "${PROFILE_ROOT}/summaries" "${PROFILE_ROOT}/analysis"
cd "${REPO}"
source "${VENV}/bin/activate"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${PROFILE_ROOT}/profile.log"
}

summarize_json() {
  local json_path="$1"
  python - "$json_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())

def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, key)
            if found is not None:
                return found
    return None

keys = [
    "total_time",
    "latency",
    "mfu_estimated_percent",
    "tensor_engine_active_time_percent",
    "vector_engine_active_time_percent",
    "scalar_engine_active_time_percent",
    "dma_active_time_percent",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "sbuf_read_bytes",
    "sbuf_write_bytes",
    "spill_reload_bytes",
    "spill_save_bytes",
    "dma_transfer_total_bytes",
    "inputs_outputs_weights_size_bytes",
    "hardware_flops",
    "transpose_flops",
    "mm_arithmetic_intensity",
    "peak_flops_bandwidth_ratio",
]

for key in keys:
    value = find_key(data, key)
    if value is not None:
        print(f"{key}={value}")
PY
}

run_prefill_request() {
  python - "${MODEL}" "${PORT}" "${PROMPT_TOKENS}" "${MAX_TOKENS}" "${PROFILE_ROOT}/prefill_request.json" <<'PY'
import json
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

model, port, prompt_tokens, max_tokens, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

def token_len(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

marker = f"profile-{int(time.time())}-{time.time_ns()}"
base = f"Profile prefill marker {marker}. Produce one ordinary continuation token after this unique context.\n"
filler = (
    "This unique profiling text keeps the request cold while preserving stable, ordinary English. "
    "The prefix cache, recurrent state, attention metadata, and segmented prefill path should remain aligned. "
)
prompt = base
while token_len(prompt) < prompt_tokens:
    prompt += filler
ids = tokenizer.encode(prompt, add_special_tokens=False)[:prompt_tokens]
prompt = tokenizer.decode(ids, skip_special_tokens=False)
actual = token_len(prompt)

payload = {
    "model": model,
    "prompt": prompt,
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
}

url = f"http://127.0.0.1:{port}/v1/completions"
t0 = time.time()
first = None
usage = None
chunks = 0
texts = []
with requests.post(url, json=payload, stream=True, timeout=1200) as response:
    status = response.status_code
    if status != 200:
        result = {"http": status, "error": response.text[:1000], "actual_prompt_tokens": actual}
        Path(out_path).write_text(json.dumps(result, indent=2) + "\n")
        raise SystemExit(2)
    for raw in response.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "):
            continue
        data = raw[6:]
        if data.strip() == "[DONE]":
            break
        now = time.time()
        obj = json.loads(data)
        chunks += 1
        if first is None and obj.get("choices"):
            first = now
        if obj.get("usage") is not None:
            usage = obj["usage"]
        for choice in obj.get("choices", []):
            text = choice.get("text") or ""
            if text:
                texts.append(text)

end = time.time()
ttft = None if first is None else first - t0
prompt_count = (usage or {}).get("prompt_tokens") or actual
result = {
    "http": 200,
    "target_prompt_tokens": prompt_tokens,
    "actual_prompt_tokens": actual,
    "usage": usage,
    "chunks": chunks,
    "ttft_seconds": ttft,
    "total_seconds": end - t0,
    "prefill_tok_s": None if not ttft else prompt_count / ttft,
    "text": "".join(texts),
}
Path(out_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY
}

summarize_profiles() {
  mapfile -t neffs < <(find "${PROFILE_ROOT}/inspect" -type f -name 'neff_*.neff' | sort)
  log "NEFF_COUNT=${#neffs[@]}"
  if [[ "${#neffs[@]}" -eq 0 ]]; then
    log "NO_NEFFS_FOUND"
    return 30
  fi

  for neff in "${neffs[@]}"; do
    local base pair_id ntff out view_rc
    base="$(basename "${neff}")"
    pair_id="${base#neff_}"
    pair_id="${pair_id%.neff}"
    ntff="$(find "${PROFILE_ROOT}/inspect" -type f -name "${pair_id}.ntff" | sort | head -n 1)"
    out="${PROFILE_ROOT}/summaries/${pair_id}"
    if [[ -z "${ntff}" || ! -f "${ntff}" ]]; then
      log "MISSING_NTFF pair=${pair_id}"
      continue
    fi
    log "VIEW_SUMMARY pair=${pair_id}"
    set +e
    "${TOOL}" view \
      -n "${neff}" \
      -s "${ntff}" \
      --output-format summary-json \
      --ignore-nc-buf-usage > "${out}.summary.json" 2> "${out}.summary.err"
    view_rc=$?
    if [[ "${view_rc}" -ne 0 ]]; then
      "${TOOL}" view \
        -n "${neff}" \
        -s "${ntff}" \
        --output-format summary-json > "${out}.summary.json" 2>> "${out}.summary.err"
      view_rc=$?
    fi
    set -e
    echo "${view_rc}" > "${out}.summary.exit"
    if [[ "${view_rc}" -eq 0 ]]; then
      summarize_json "${out}.summary.json" > "${out}.metrics.txt"
    else
      log "VIEW_SUMMARY_FAILED pair=${pair_id} rc=${view_rc}"
      tail -n 40 "${out}.summary.err" | tee -a "${PROFILE_ROOT}/profile.log" || true
    fi
  done

  python - "${PROFILE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "summaries").glob("*.metrics.txt")):
    row = {"pair": path.name.replace(".metrics.txt", "")}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        try:
            row[key] = float(raw)
        except ValueError:
            row[key] = raw
    rows.append(row)

rows.sort(key=lambda item: (item.get("total_time", item.get("latency", 0.0)), item.get("hbm_read_bytes", 0.0)), reverse=True)
(root / "analysis" / "top_profiles.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
for row in rows[:12]:
    print(json.dumps(row, sort_keys=True))
PY
}

cleanup_profile_server() {
  pkill -9 -f "[V]LLM::EngineCore" || true
  pkill -9 -f "[s]erve_qwen36.py" || true
  pkill -9 -f "[p]ython.*vllm" || true
}

log "PROFILE_ROOT=${PROFILE_ROOT}"
log "ARTIFACT=${ARTIFACT}"
log "MODEL=${MODEL}"
log "PORT=${PORT}"
log "PROMPT_TOKENS=${PROMPT_TOKENS}"
log "TOOL=$(${TOOL} --version 2>&1 | tr '\n' ' ')"

export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export NEURON_FRAMEWORK_DEBUG=1
export NEURON_RT_INSPECT_ENABLE=1
export NEURON_RT_INSPECT_DEVICE_PROFILE=1
export NEURON_RT_INSPECT_SYSTEM_PROFILE=0
export NEURON_RT_INSPECT_OUTPUT_DIR="${PROFILE_ROOT}/inspect"
export NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1
export QWEN36_HYBRID_APC_DEBUG=0
export QWEN36_VLLM_LOGITS_DEBUG=0

cleanup_profile_server

PROFILE_LOG="${PROFILE_ROOT}/server_profiled.log"
log "LAUNCH_PROFILE_SERVER log=${PROFILE_LOG}"
MAX_MODEL_LEN=32768 \
SEQ_LEN=32768 \
CTE_BUCKETS=2048 \
CONTEXT_ENCODING_BUCKET_PAIRS="2048:256 2048:512 2048:1024 2048:2048 2048:4096 2048:8192 2048:16384 2048:32768" \
TOKEN_GENERATION_BUCKETS="512 16384 16640 32768" \
GDN_RECURRENT_CACHE_DTYPE=bfloat16 \
GDN_CONV_CACHE_DTYPE=bfloat16 \
PORT="${PORT}" \
bash tmp_launch_qwen36_segcte2048.sh "${ARTIFACT}" "${PROFILE_LOG}" 2>&1 | tee "${PROFILE_ROOT}/launch_profiled.log"

log "RUN_PREFILL_REQUEST"
run_prefill_request 2>&1 | tee "${PROFILE_ROOT}/prefill_request.stdout"

log "STOP_PROFILE_SERVER"
cleanup_profile_server
sleep 5

log "SUMMARIZE_PROFILES"
summarize_profiles 2>&1 | tee "${PROFILE_ROOT}/summary.stdout"

if [[ "${RESTART_NORMAL}" == "1" ]]; then
  NORMAL_LOG="${PROFILE_ROOT}/server_normal_relaunch.log"
  log "RELAUNCH_NORMAL_SERVER log=${NORMAL_LOG}"
  unset XLA_IR_DEBUG XLA_HLO_DEBUG NEURON_FRAMEWORK_DEBUG
  unset NEURON_RT_INSPECT_ENABLE NEURON_RT_INSPECT_DEVICE_PROFILE NEURON_RT_INSPECT_SYSTEM_PROFILE
  unset NEURON_RT_INSPECT_OUTPUT_DIR NEURON_RT_ENABLE_DGE_NOTIFICATIONS
  MAX_MODEL_LEN=32768 \
  SEQ_LEN=32768 \
  CTE_BUCKETS=2048 \
  CONTEXT_ENCODING_BUCKET_PAIRS="2048:256 2048:512 2048:1024 2048:2048 2048:4096 2048:8192 2048:16384 2048:32768" \
  TOKEN_GENERATION_BUCKETS="512 16384 16640 32768" \
  GDN_RECURRENT_CACHE_DTYPE=bfloat16 \
  GDN_CONV_CACHE_DTYPE=bfloat16 \
  PORT="${PORT}" \
  bash tmp_launch_qwen36_segcte2048.sh "${ARTIFACT}" "${NORMAL_LOG}" 2>&1 | tee "${PROFILE_ROOT}/launch_normal.log"
fi

log "PROFILE_COMPLETE"
echo "PROFILE_ROOT=${PROFILE_ROOT}"
