#!/usr/bin/env bash
set -uo pipefail

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate

TOOL=/opt/aws/neuron/bin/neuron-explorer
ROOT=${ROOT:-/home/ubuntu/validation_logs/fp8_256k_decode_nki/direct_neff_ab_context_$(date -u +%Y%m%dT%H%M%SZ)}
BASE=${BASE:-/mnt/trainium_artifacts/qwen_artifacts/profile_neffs/baseline_scan2powtskip/context_encoding_model}
COMP=${COMP:-/mnt/trainium_artifacts/qwen_artifacts/profile_neffs/compactautocp/context_encoding_model}

mkdir -p "${ROOT}"

summarize_json() {
  local json_path="$1"
  python3 - "${json_path}" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
d = json.loads(p.read_text())

def find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find(value, key)
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
for key in keys:
    value = find(d, key)
    if value is not None:
        print(f"{key}={value}")
PY
}

run_one() {
  local label="$1"
  local neff="$2"
  shift 2
  local out="${ROOT}/${label}"
  mkdir -p "${out}"
  echo "CAPTURE_START ${label} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "NEFF=${neff}"
  if [[ ! -f "${neff}" ]]; then
    echo "MISSING_NEFF ${label}"
    return 2
  fi

  set +e
  timeout 600 "${TOOL}" capture \
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
    "$@" > "${out}/capture.log" 2>&1
  local rc=$?
  echo "${rc}" > "${out}/capture.exit"

  local captured_ntff=""
  if [[ ! -f "${out}/profile.ntff" ]]; then
    captured_ntff="$(find "${out}" -maxdepth 1 -type f -name "*.ntff" | sort | head -n 1 || true)"
    if [[ -n "${captured_ntff}" ]]; then
      ln -sf "$(basename "${captured_ntff}")" "${out}/profile.ntff"
    fi
  fi

  if [[ ${rc} -ne 0 ]] && grep -Eiq "NRT_RESOURCE|out of memory|insufficient|allocated memory" "${out}/capture.log"; then
    echo "CAPTURE_RETRY_SINGLE_IO ${label} rc=${rc}"
    timeout 600 "${TOOL}" capture \
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
      --single-io > "${out}/capture_singleio.log" 2>&1
    rc=$?
    echo "${rc}" > "${out}/capture_singleio.exit"
    if [[ -f "${out}/profile_singleio.ntff" ]]; then
      cp "${out}/profile_singleio.ntff" "${out}/profile.ntff"
    else
      captured_ntff="$(find "${out}" -maxdepth 1 -type f -name "*.ntff" | sort | head -n 1 || true)"
      if [[ -n "${captured_ntff}" ]]; then
        ln -sf "$(basename "${captured_ntff}")" "${out}/profile.ntff"
      fi
    fi
  fi
  set -e

  echo "CAPTURE_EXIT ${label} ${rc}"
  if [[ ${rc} -eq 0 && -f "${out}/profile.ntff" ]]; then
    "${TOOL}" view \
      -n "${neff}" \
      -s "${out}/profile.ntff" \
      --output-format summary-json \
      --ignore-nc-buf-usage > "${out}/summary.json" 2> "${out}/view.err"
    local view_rc=$?
    echo "${view_rc}" > "${out}/view.exit"
    echo "VIEW_EXIT ${label} ${view_rc}"
    if [[ ${view_rc} -eq 0 ]]; then
      summarize_json "${out}/summary.json" | tee "${out}/summary_metrics.txt"
    else
      tail -40 "${out}/view.err" || true
    fi
  else
    tail -60 "${out}/capture.log" || true
  fi
  echo "CAPTURE_END ${label} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

for bk in 0 1 2 3 4 5 6 7; do
  run_one "baseline_bk${bk}" "${BASE}/_tp0_bk${bk}/graph.neff"
done

for bk in 0 1 2 3; do
  run_one "compact_bk${bk}" "${COMP}/_tp0_bk${bk}/graph.neff"
done

python3 - "${ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("AB_CONTEXT_SUMMARY")
for directory in sorted(path for path in root.iterdir() if path.is_dir()):
    metrics = {}
    metrics_path = directory / "summary_metrics.txt"
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                metrics[key] = value
    capture_rc = (
        (directory / "capture.exit").read_text().strip()
        if (directory / "capture.exit").exists()
        else "na"
    )
    view_rc = (
        (directory / "view.exit").read_text().strip()
        if (directory / "view.exit").exists()
        else "na"
    )
    print(
        directory.name,
        f"capture_rc={capture_rc}",
        f"view_rc={view_rc}",
        f"latency={metrics.get('latency')}",
        f"total_time={metrics.get('total_time')}",
        f"mfu={metrics.get('mfu_estimated_percent')}",
        f"tensor={metrics.get('tensor_engine_active_time_percent')}",
        f"dma={metrics.get('dma_active_time_percent')}",
        f"hbm_r={metrics.get('hbm_read_bytes')}",
        f"hbm_w={metrics.get('hbm_write_bytes')}",
    )
PY

echo "PROFILE_ROOT=${ROOT}"
