#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ubuntu/inferentia-gdn-prefill-speed-coherent}"
ARTIFACT="${ARTIFACT:-/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T152927Z_direct_scan0_recbf16fix}"
PROFILE_ROOT="${PROFILE_ROOT:-/home/ubuntu/validation_logs/fp8_256k_decode_nki/qkvnki_prefill_profile_20260605T163533Z}"
OUT_ROOT="${OUT_ROOT:-${PROFILE_ROOT}/direct_context_captures_$(date -u +%Y%m%dT%H%M%SZ)}"
TOOL="${TOOL:-/opt/aws/neuron/bin/neuron-explorer}"
PORT="${PORT:-8001}"
RESTART_NORMAL="${RESTART_NORMAL:-1}"
ENABLE_DGE="${ENABLE_DGE:-0}"
CAPTURE_LABELS="${CAPTURE_LABELS:-context_bk0_pfx0 context_bk4_pfx2048 context_bk5_pfx4096 context_bk6_pfx8192 context_bk7_pfx16384}"

mkdir -p "${OUT_ROOT}/summaries"
cd "${REPO}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/capture.log"
}

cleanup_server() {
  pkill -9 -f "[V]LLM::EngineCore" || true
  pkill -9 -f "[s]erve_qwen36.py" || true
  pkill -9 -f "[p]ython.*vllm" || true
}

summarize_json() {
  local json_path="$1"
  python3 - "$json_path" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())

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

find_neff_by_sha() {
  local sha="$1"
  local path digest
  while IFS= read -r -d '' path; do
    digest="$(sha256sum "${path}" | awk '{print $1}')"
    if [[ "${digest}" == "${sha}" ]]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done < <(find "${PROFILE_ROOT}/inspect" -type f -name 'neff_*.neff' -print0)
  return 1
}

run_capture() {
  local label="$1"
  local sha="$2"
  local neff="$3"
  local out="${OUT_ROOT}/${label}"
  mkdir -p "${out}"
  log "CAPTURE_START label=${label} neff=${neff}"
  if [[ -z "${neff}" || ! -f "${neff}" ]]; then
    log "MISSING_NEFF label=${label} sha=${sha}"
    echo 2 > "${out}/capture.exit"
    return 2
  fi
  local dge_flags=()
  if [[ "${ENABLE_DGE}" == "1" ]]; then
    dge_flags+=(--enable-dge-notifs)
  fi

  set +e
  "${TOOL}" capture \
    -n "${neff}" \
    -s "${out}/profile.ntff" \
    --collectives-worker-start-id=0 \
    --collectives-worker-count=4 \
    --collectives-workers-per-node=4 \
    --collectives-profile-id=0 \
    --num-exec=2 \
    --profile-nth-exec=2 \
    --ignore-exec-errors \
    --io-from=neff \
    "${dge_flags[@]}" 2>&1 | tee "${out}/capture.raw.log"
  local capture_rc=${PIPESTATUS[0]}
  set -e
  echo "${capture_rc}" > "${out}/capture.exit"

  if [[ "${capture_rc}" -ne 0 ]] && grep -Eiq 'NRT_RESOURCE|out of memory|insufficient|allocated memory|failed to allocated resource' "${out}/capture.raw.log"; then
    log "CAPTURE_RETRY_SINGLE_IO label=${label}"
    set +e
    "${TOOL}" capture \
      -n "${neff}" \
      -s "${out}/profile_singleio.ntff" \
      --collectives-worker-start-id=0 \
      --collectives-worker-count=4 \
      --collectives-workers-per-node=4 \
      --collectives-profile-id=0 \
      --num-exec=2 \
      --profile-nth-exec=2 \
      --ignore-exec-errors \
      --io-from=neff \
      --single-io \
      "${dge_flags[@]}" 2>&1 | tee "${out}/capture_singleio.raw.log"
    capture_rc=${PIPESTATUS[0]}
    set -e
    echo "${capture_rc}" > "${out}/capture_singleio.exit"
  fi

  local ntff
  ntff="$(find "${out}" -maxdepth 1 -type f -name '*exec_2*.ntff' | sort | head -n 1)"
  if [[ -z "${ntff}" ]]; then
    ntff="$(find "${out}" -maxdepth 1 -type f -name '*.ntff' | sort | head -n 1)"
  fi
  if [[ -z "${ntff}" || ! -f "${ntff}" ]]; then
    log "NO_NTFF label=${label} rc=${capture_rc}"
    return "${capture_rc}"
  fi

  set +e
  "${TOOL}" view \
    -n "${neff}" \
    -s "${ntff}" \
    --output-format summary-json \
    --ignore-nc-buf-usage > "${out}/summary.json" 2> "${out}/summary.err"
  local view_rc=$?
  if [[ "${view_rc}" -ne 0 ]]; then
    "${TOOL}" view \
      -n "${neff}" \
      -s "${ntff}" \
      --output-format summary-json > "${out}/summary.json" 2>> "${out}/summary.err"
    view_rc=$?
  fi
  set -e
  echo "${view_rc}" > "${out}/summary.exit"
  if [[ "${view_rc}" -eq 0 ]]; then
    summarize_json "${out}/summary.json" > "${out}/metrics.txt"
  else
    log "VIEW_FAILED label=${label} rc=${view_rc}"
    tail -n 60 "${out}/summary.err" | tee -a "${OUT_ROOT}/capture.log" || true
  fi
  log "CAPTURE_END label=${label} capture_rc=${capture_rc} view_rc=${view_rc}"
  return 0
}

declare -A SHAS=(
  [context_bk0_pfx0]=b9d7d79fd5164c9c09b557b261934158e477031f71dfcf20406761bcf32ea82c
  [context_bk1_pfx256]=8634cbd341fff10f963da026027a6761f0d2bc04220e513f1b16ac58d6bc0ea5
  [context_bk2_pfx512]=20fd11252aa591341cdc9d341fbfe4368611023bfe38f796a4263de418690804
  [context_bk3_pfx1024]=1ab6c0e1f4eea0c695f3ec54b8f444a9232ed8b89bd11cca137b59ad5306a53a
  [context_bk4_pfx2048]=892c47c1d59c1e6c92b05a8c52cd41c50e50aced16e3b05241afdec005f09f40
  [context_bk5_pfx4096]=2a3de7991fa7bf3da659c941254f31a334b5dd2c17f3485c5187905e3f397122
  [context_bk6_pfx8192]=c8acb16ae5a0a690e137b4da0367be5cd58ce717245929ada4337666d5e43095
  [context_bk7_pfx16384]=cb80c3f7f7d2010a0384b204cd8dbf45385a72e4e7325d2e4f30b9e01104a9c4
  [context_bk8_pfx32768]=cae04c2fce0c71b5158abd2d3fde60a4d88cf549594968057791cbdeae18ee5d
)

log "OUT_ROOT=${OUT_ROOT}"
log "PROFILE_ROOT=${PROFILE_ROOT}"
log "CAPTURE_LABELS=${CAPTURE_LABELS}"
log "ENABLE_DGE=${ENABLE_DGE}"

cleanup_server
sleep 3

for label in ${CAPTURE_LABELS}; do
  sha="${SHAS[${label}]:-}"
  if [[ -z "${sha}" ]]; then
    log "UNKNOWN_LABEL label=${label}"
    continue
  fi
  neff="$(find_neff_by_sha "${sha}")"
  run_capture "${label}" "${sha}" "${neff}" || true
done

python3 - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for metrics_path in sorted(root.glob("context_*/metrics.txt")):
    row = {"label": metrics_path.parent.name}
    for line in metrics_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        try:
            row[key] = float(raw)
        except ValueError:
            row[key] = raw
    rows.append(row)
rows.sort(key=lambda row: row.get("total_time", row.get("latency", 0.0)), reverse=True)
(root / "summaries" / "context_capture_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
for row in rows:
    print(json.dumps(row, sort_keys=True))
PY

if [[ "${RESTART_NORMAL}" == "1" ]]; then
  log "RELAUNCH_NORMAL_SERVER"
  MAX_MODEL_LEN=32768 \
  SEQ_LEN=32768 \
  CTE_BUCKETS=2048 \
  CONTEXT_ENCODING_BUCKET_PAIRS="2048:256 2048:512 2048:1024 2048:2048 2048:4096 2048:8192 2048:16384 2048:32768" \
  TOKEN_GENERATION_BUCKETS="512 16384 16640 32768" \
  GDN_RECURRENT_CACHE_DTYPE=bfloat16 \
  GDN_CONV_CACHE_DTYPE=bfloat16 \
  PORT="${PORT}" \
  bash tmp_launch_qwen36_segcte2048.sh "${ARTIFACT}" "${OUT_ROOT}/server_normal_relaunch.log"
fi

log "DIRECT_CONTEXT_CAPTURE_COMPLETE"
echo "OUT_ROOT=${OUT_ROOT}"
