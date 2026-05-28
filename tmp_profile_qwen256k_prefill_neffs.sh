#!/usr/bin/env bash
set -uo pipefail

TS="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE_ROOT="/home/ubuntu/validation_logs/fp8_256k/prefill_profile_${TS}"
WORKDIR="/mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_256k_fp8_full_prod_pfx256k_segcte512stream_qpack4_boundfix_cte3072_pfx256k_pa1025_tkg262144_20260527T052822Z"
TOOL="/opt/aws/neuron/bin/neuron-explorer"
ENABLE_DGE="${ENABLE_DGE:-0}"

mkdir -p "${PROFILE_ROOT}"

echo "PROFILE_ROOT=${PROFILE_ROOT}"
echo "WORKDIR=${WORKDIR}"
echo "TOOL=$(${TOOL} --version 2>&1 | tr '\n' ' ')"
echo "ENABLE_DGE=${ENABLE_DGE}"
date -Is

summarize_json() {
  local json_path="$1"
  python3 - "$json_path" <<'PY'
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
    "mm_arithmetic_intensity",
    "peak_flops_bandwidth_ratio",
]

print("SUMMARY_METRICS_BEGIN")
for key in keys:
    value = find_key(data, key)
    if value is not None:
        print(f"{key}={value}")
print("SUMMARY_METRICS_END")
PY
}

run_capture() {
  local name="$1"
  local neff="$2"
  shift 2
  local extra_flags=("$@")
  local out="${PROFILE_ROOT}/${name}"

  mkdir -p "${out}"
  echo
  echo "=== CAPTURE_START ${name} ==="
  echo "NEFF=${neff}"
  echo "OUT=${out}"
  date -Is

  if [[ ! -f "${neff}" ]]; then
    echo "ERROR: missing NEFF ${neff}"
    echo 2 > "${out}/capture.exit"
    return 2
  fi

  set +e
  local dge_flags=()
  if [[ "${ENABLE_DGE}" == "1" ]]; then
    dge_flags+=(--enable-dge-notifs)
  fi
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
    "${dge_flags[@]}" \
    "${extra_flags[@]}" 2>&1 | tee "${out}/capture.log"
  local capture_rc=${PIPESTATUS[0]}
  set +e

  echo "${capture_rc}" > "${out}/capture.exit"
  echo "CAPTURE_EXIT ${name} ${capture_rc}"
  find "${out}" -maxdepth 1 -type f -name '*.ntff' -print | sort | tee "${out}/ntff_files.txt"

  local ntff
  ntff="$(find "${out}" -maxdepth 1 -type f -name '*exec_2*.ntff' | sort | head -n 1)"
  if [[ -z "${ntff}" ]]; then
    ntff="$(find "${out}" -maxdepth 1 -type f -name '*.ntff' | sort | head -n 1)"
  fi

  if [[ -n "${ntff}" && -f "${ntff}" ]]; then
    echo "VIEW_START ${name} NTFF=${ntff}"
    set +e
    "${TOOL}" view \
      -n "${neff}" \
      -s "${ntff}" \
      --output-format summary-json \
      --ignore-nc-buf-usage > "${out}/summary_rank0.json" 2> "${out}/view.err"
    local view_rc=$?
    if [[ ${view_rc} -ne 0 ]]; then
      "${TOOL}" view \
        -n "${neff}" \
        -s "${ntff}" \
        --output-format summary-json > "${out}/summary_rank0.json" 2>> "${out}/view.err"
      view_rc=$?
    fi
    set -e
    set +e
    echo "${view_rc}" > "${out}/view.exit"
    echo "VIEW_EXIT ${name} ${view_rc}"
    if [[ ${view_rc} -eq 0 ]]; then
      summarize_json "${out}/summary_rank0.json" | tee "${out}/summary_metrics.txt"
    else
      echo "VIEW_ERROR ${name}"
      tail -n 80 "${out}/view.err" || true
    fi
  else
    echo "ERROR: no NTFF generated for ${name}"
  fi

  echo "=== CAPTURE_END ${name} ==="
  date -Is
  return "${capture_rc}"
}

BK0="${WORKDIR}/context_encoding_model/_tp0_bk0/graph.neff"
BK1="${WORKDIR}/context_encoding_model/_tp0_bk1/graph.neff"

run_capture "context_bk0_dense_cte3072_pfx0" "${BK0}"
bk0_rc=$?

if [[ ${bk0_rc} -ne 0 ]] && grep -Eiq 'NRT_RESOURCE|out of memory|insufficient|allocated memory|failed to allocated resource' "${PROFILE_ROOT}/context_bk0_dense_cte3072_pfx0/capture.log"; then
  echo
  echo "BK0 normal capture hit a memory/resource error; retrying with --single-io for profiling-only capture."
  run_capture "context_bk0_dense_cte3072_pfx0_singleio" "${BK0}" --single-io
  bk0_retry_rc=$?
else
  bk0_retry_rc=0
fi

run_capture "context_bk1_segcte512_pfx256k" "${BK1}"
bk1_rc=$?

if [[ ${bk1_rc} -ne 0 ]] && grep -Eiq 'NRT_RESOURCE|out of memory|insufficient|allocated memory' "${PROFILE_ROOT}/context_bk1_segcte512_pfx256k/capture.log"; then
  echo
  echo "BK1 normal capture hit a memory/resource error; retrying with --single-io for profiling-only capture."
  run_capture "context_bk1_segcte512_pfx256k_singleio" "${BK1}" --single-io
  bk1_retry_rc=$?
else
  bk1_retry_rc=0
fi

echo
echo "PROFILE_COMPLETE_ROOT=${PROFILE_ROOT}"
echo "BK0_RC=${bk0_rc}"
echo "BK0_RETRY_RC=${bk0_retry_rc}"
echo "BK1_RC=${bk1_rc}"
echo "BK1_RETRY_RC=${bk1_retry_rc}"
date -Is

if [[ ${bk0_rc} -ne 0 && ${bk0_retry_rc} -ne 0 ]]; then
  exit "${bk0_rc}"
fi
if [[ ${bk1_rc} -ne 0 && ${bk1_retry_rc} -ne 0 ]]; then
  exit "${bk1_rc}"
fi
exit 0
