#!/usr/bin/env bash
set -uo pipefail

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

TOOL=${TOOL:-/opt/aws/neuron/bin/neuron-explorer}
BASE=${BASE:-/mnt/trainium_artifacts/qwen_artifacts/profile_neffs/outprojnki3/context_encoding_model}
ROOT=${ROOT:-/home/ubuntu/validation_logs/fp8_256k_decode_nki/outprojnki3_context_profile_$(date -u +%Y%m%dT%H%M%SZ)}
DATA_PATH="${ROOT}/neuron_profile_data"
ENABLE_DGE=${ENABLE_DGE:-1}
DO_INGEST=${DO_INGEST:-1}
VIEW_FAST_SUMMARY=${VIEW_FAST_SUMMARY:-0}

mkdir -p "${ROOT}/analysis" "${ROOT}/captures"

PAIRS_TEXT=${PAIRS_TEXT:-"1024:0 1024:256 1024:512 1024:1024 1024:2048 1024:4096 1024:8192 1024:16384"}
PROFILE_NAME_PREFIX=${PROFILE_NAME_PREFIX:-outprojnki3_context}

read -r -a PAIRS <<< "${PAIRS_TEXT}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${ROOT}/profile.log"
}

summarize_json() {
  local json_path="$1"
  python3 - "${json_path}" <<'PY'
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
    "total_exec_time",
    "total_active_time",
    "latency",
    "total_cycles",
    "neuroncore_cycle_count",
    "mfu_estimated_percent",
    "mbu_estimated_percent",
    "tensor_engine_active_time_percent",
    "vector_engine_active_time_percent",
    "scalar_engine_active_time_percent",
    "dma_active_time_percent",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "input_queue_bytes",
    "weight_queue_bytes",
    "output_queue_bytes",
    "spill_reload_bytes",
    "spill_save_bytes",
    "model_flops",
    "mm_arithmetic_intensity",
    "peak_flops_bandwidth_ratio",
]
for key in keys:
    value = find_key(data, key)
    if value is not None:
        print(f"{key}={value}")
PY
}

run_one() {
  local bucket_idx="$1"
  local bucket_pair="$2"
  local label="bk${bucket_idx}_active${bucket_pair%:*}_prefix${bucket_pair#*:}"
  local neff="${BASE}/_tp0_bk${bucket_idx}/graph.neff"
  local out="${ROOT}/captures/${label}"

  mkdir -p "${out}"
  log "CAPTURE_START label=${label} pair=${bucket_pair} neff=${neff}"
  if [[ ! -f "${neff}" ]]; then
    log "MISSING_NEFF label=${label} neff=${neff}"
    echo 2 > "${out}/capture.exit"
    return 2
  fi

  local dge_flags=()
  local dge_env=(env NEURON_RT_ENABLE_DGE_NOTIFICATIONS=0)
  if [[ "${ENABLE_DGE}" == "1" ]]; then
    dge_flags+=(--enable-dge-notifs)
    dge_env=(env NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1)
  fi

  set +e
  "${dge_env[@]}" timeout 900 "${TOOL}" capture \
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
    "${dge_flags[@]}" > "${out}/capture.log" 2>&1
  local capture_rc=$?
  echo "${capture_rc}" > "${out}/capture.exit"

  local ntff
  ntff="$(find "${out}" -maxdepth 1 -type f -name '*exec_2*.ntff' | sort | head -n 1)"
  if [[ -z "${ntff}" ]]; then
    ntff="$(find "${out}" -maxdepth 1 -type f -name '*.ntff' | sort | head -n 1)"
  fi
  if [[ -n "${ntff}" && -f "${ntff}" && "${ntff}" != "${out}/profile.ntff" ]]; then
    ln -sf "$(basename "${ntff}")" "${out}/profile.ntff"
  fi

  if [[ ${capture_rc} -ne 0 ]] && grep -Eiq 'NRT_RESOURCE|out of memory|insufficient|allocated memory|failed to allocated resource' "${out}/capture.log"; then
    log "CAPTURE_RETRY_SINGLE_IO label=${label} rc=${capture_rc}"
    "${dge_env[@]}" timeout 900 "${TOOL}" capture \
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
      "${dge_flags[@]}" > "${out}/capture_singleio.log" 2>&1
    capture_rc=$?
    echo "${capture_rc}" > "${out}/capture_singleio.exit"
    ntff="$(find "${out}" -maxdepth 1 -type f -name '*singleio*exec_2*.ntff' | sort | head -n 1)"
    if [[ -z "${ntff}" ]]; then
      ntff="$(find "${out}" -maxdepth 1 -type f -name 'profile_singleio*.ntff' | sort | head -n 1)"
    fi
    if [[ -n "${ntff}" && -f "${ntff}" ]]; then
      cp "${ntff}" "${out}/profile.ntff"
    fi
  fi
  set -e

  log "CAPTURE_EXIT label=${label} rc=${capture_rc}"
  if [[ ${capture_rc} -ne 0 || ! -f "${out}/profile.ntff" ]]; then
    tail -80 "${out}/capture.log" | tee -a "${ROOT}/profile.log" || true
    return "${capture_rc}"
  fi

  set +e
  local view_extra_flags=(--ignore-nc-buf-usage)
  if [[ "${VIEW_FAST_SUMMARY}" == "1" ]]; then
    view_extra_flags+=(
      --ignore-instruction-trace
      --ignore-event-trace
      --ignore-dma-trace
      --ignore-instruction-hierarchy
    )
  fi
  "${TOOL}" view \
    -n "${neff}" \
    -s "${out}/profile.ntff" \
    --output-format summary-json \
    "${view_extra_flags[@]}" > "${out}/summary.json" 2> "${out}/view.err"
  local view_rc=$?
  if [[ ${view_rc} -ne 0 ]]; then
    "${TOOL}" view \
      -n "${neff}" \
      -s "${out}/profile.ntff" \
      --output-format summary-json > "${out}/summary.json" 2>> "${out}/view.err"
    view_rc=$?
  fi
  echo "${view_rc}" > "${out}/view.exit"
  log "VIEW_EXIT label=${label} rc=${view_rc}"
  if [[ ${view_rc} -eq 0 ]]; then
    summarize_json "${out}/summary.json" | tee "${out}/summary_metrics.txt"
  else
    tail -80 "${out}/view.err" | tee -a "${ROOT}/profile.log" || true
  fi

  local ingest_rc="skipped"
  if [[ "${DO_INGEST}" == "1" ]]; then
    local profile_name="${PROFILE_NAME_PREFIX}_bk${bucket_idx}"
    "${TOOL}" view \
      -n "${neff}" \
      -s "${out}/profile.ntff" \
      --data-path "${DATA_PATH}" \
      --display-name "${profile_name}" \
      --ingest-only \
      --ignore-nc-buf-usage > "${out}/ingest.log" 2>&1
    ingest_rc=$?
    if [[ ${ingest_rc} -ne 0 ]]; then
      "${TOOL}" view \
        -n "${neff}" \
        -s "${out}/profile.ntff" \
        --data-path "${DATA_PATH}" \
        --display-name "${profile_name}" \
        --ingest-only >> "${out}/ingest.log" 2>&1
      ingest_rc=$?
    fi
  else
    echo "skipped" > "${out}/ingest.log"
  fi
  echo "${ingest_rc}" > "${out}/ingest.exit"
  log "INGEST_EXIT label=${label} rc=${ingest_rc}"
  set -e
}

log "PROFILE_ROOT=${ROOT}"
log "BASE=${BASE}"
log "ENABLE_DGE=${ENABLE_DGE}"
log "DO_INGEST=${DO_INGEST}"
log "VIEW_FAST_SUMMARY=${VIEW_FAST_SUMMARY}"
log "TOOL=$(${TOOL} --version 2>&1 | tr '\n' ' ')"

for idx in "${!PAIRS[@]}"; do
  run_one "${idx}" "${PAIRS[$idx]}"
done

python3 - "${ROOT}" "${PAIRS_TEXT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
pairs = sys.argv[2].split()

rows = []
for idx, pair in enumerate(pairs):
    parts = pair.split(":", 1)
    if len(parts) != 2:
        raise SystemExit(f"invalid bucket pair {pair!r}; expected ACTIVE:PREFIX")
    label = f"bk{idx}_active{parts[0]}_prefix{parts[1]}"
    out = root / "captures" / label
    metrics = {}
    metrics_path = out / "summary_metrics.txt"
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                metrics[key] = value
    row = {
        "bucket_index": idx,
        "bucket_pair": pair,
        "label": label,
        "capture_exit": (out / "capture.exit").read_text().strip() if (out / "capture.exit").exists() else None,
        "view_exit": (out / "view.exit").read_text().strip() if (out / "view.exit").exists() else None,
        "ingest_exit": (out / "ingest.exit").read_text().strip() if (out / "ingest.exit").exists() else None,
    }
    row.update(metrics)
    rows.append(row)

def as_float(value):
    try:
        return float(value)
    except Exception:
        return -1.0

def rank_value(row):
    for key in ("total_time", "total_exec_time", "total_active_time", "latency", "total_cycles", "neuroncore_cycle_count"):
        value = as_float(row.get(key))
        if value > 0:
            return value
    return -1.0

ranked = sorted(rows, key=rank_value, reverse=True)
(root / "analysis" / "bucket_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
(root / "analysis" / "bucket_ranked_by_time.json").write_text(json.dumps(ranked, indent=2, sort_keys=True) + "\n")

print("BUCKET_RANKED_BY_TIME")
for row in ranked:
    print(
        row["label"],
        f"pair={row['bucket_pair']}",
        f"capture={row['capture_exit']}",
        f"view={row['view_exit']}",
        f"ingest={row['ingest_exit']}",
        f"total_time={row.get('total_time')}",
        f"exec_time={row.get('total_exec_time')}",
        f"latency={row.get('latency')}",
        f"cycles={row.get('total_cycles') or row.get('neuroncore_cycle_count')}",
        f"tensor={row.get('tensor_engine_active_time_percent')}",
        f"dma={row.get('dma_active_time_percent')}",
        f"hbm_read={row.get('hbm_read_bytes')}",
        f"hbm_write={row.get('hbm_write_bytes')}",
    )
PY

log "PROFILE_COMPLETE root=${ROOT}"
