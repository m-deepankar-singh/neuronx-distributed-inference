#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ubuntu/inferentia-gdn-nki-decode-step-c49df2b-splitqkv}
MODEL=${MODEL:-/home/ubuntu/models/Qwen3.6-27B}
MODEL_NAME=${MODEL_NAME:-${MODEL}}
ARTIFACT=${ARTIFACT:-/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_splitqkv_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260528T074851Z}
ROOT=${ROOT:-/home/ubuntu/validation_logs/fp8_256k_decode_nki/splitqkv_rootcause_profile_$(date -u +%Y%m%dT%H%M%SZ)}
VENV=${VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}
TOOL=${TOOL:-/opt/aws/neuron/bin/neuron-explorer}
BACKEND_PORT=${BACKEND_PORT:-8011}
PROXY_PORT=${PROXY_PORT:-8010}
LENGTHS=${LENGTHS:-512}
MAX_TOKENS=${MAX_TOKENS:-64}
REPEATS=${REPEATS:-1}
TOKEN_GENERATION_BUCKETS=${TOKEN_GENERATION_BUCKETS:-512,768,1024,1280,2048,2304,4096,4352,8192,8448,16384,16640,24576,24832,32768,33024,65536,65792,131072,131328,262144}

mkdir -p "${ROOT}/profile_summaries" "${ROOT}/analysis" "${ROOT}/neuron_profile_data"
cd "${REPO}"
source "${VENV}/bin/activate"

export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export NEURON_FRAMEWORK_DEBUG=1
export NEURON_RT_INSPECT_ENABLE=1
export NEURON_RT_INSPECT_DEVICE_PROFILE=1
export NEURON_RT_INSPECT_SYSTEM_PROFILE=0
export NEURON_RT_INSPECT_OUTPUT_DIR="${ROOT}/inspect"
export NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1
export QWEN36_HYBRID_APC_DEBUG=0

log() {
  printf "[%s] %s\n" "$(date -Is)" "$*" | tee -a "${ROOT}/profile.log"
}

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
    if isinstance(obj, list):
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
    "sbuf_write_bytes",
    "spill_reload_bytes",
    "spill_save_bytes",
    "dma_transfer_total_bytes",
    "inputs_outputs_weights_size_bytes",
    "hardware_flops",
    "transpose_flops",
]

for key in keys:
    value = find_key(data, key)
    if value is not None:
        print(f"{key}={value}")
PY
}

log "ROOT=${ROOT}"
log "REPO=${REPO}"
log "ARTIFACT=${ARTIFACT}"
log "PROFILE_ENV inspect=${NEURON_RT_INSPECT_OUTPUT_DIR} dge=${NEURON_RT_ENABLE_DGE_NOTIFICATIONS}"

set +e
ROOT="${ROOT}/runtime" \
REPO="${REPO}" \
MODEL="${MODEL}" \
MODEL_NAME="${MODEL_NAME}" \
ARTIFACT="${ARTIFACT}" \
BACKEND_PORT="${BACKEND_PORT}" \
PROXY_PORT="${PROXY_PORT}" \
LENGTHS="${LENGTHS}" \
MAX_TOKENS="${MAX_TOKENS}" \
REPEATS="${REPEATS}" \
TOKEN_GENERATION_BUCKETS="${TOKEN_GENERATION_BUCKETS}" \
GDN_RECURRENT_CACHE_DTYPE=bfloat16 \
GDN_CONV_CACHE_DTYPE=bfloat16 \
./tmp_run_qwen256k_fp8_tighttkg_decode_eval.sh 2>&1 | tee "${ROOT}/runtime_profile.log"
run_rc=${PIPESTATUS[0]}
set -e
echo "${run_rc}" > "${ROOT}/runtime.exit"
log "RUNTIME_EXIT=${run_rc}"
if [[ "${run_rc}" -ne 0 ]]; then
  log "RUNTIME_FAILED"
  exit "${run_rc}"
fi

mapfile -t neffs < <(find "${NEURON_RT_INSPECT_OUTPUT_DIR}" -type f -name 'neff_*.neff' | sort)
log "NEFF_COUNT=${#neffs[@]}"
if [[ "${#neffs[@]}" -eq 0 ]]; then
  log "NO_NEFFS_FOUND"
  exit 30
fi

for neff in "${neffs[@]}"; do
  base="$(basename "${neff}")"
  pair_id="${base#neff_}"
  pair_id="${pair_id%.neff}"
  ntff="$(find "${NEURON_RT_INSPECT_OUTPUT_DIR}" -type f -name "${pair_id}.ntff" | sort | head -n 1)"
  out="${ROOT}/profile_summaries/${pair_id}"
  if [[ -z "${ntff}" || ! -f "${ntff}" ]]; then
    log "MISSING_NTFF pair=${pair_id} neff=${neff}"
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
  fi
done

python3 - "${ROOT}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for metrics_path in sorted((root / "profile_summaries").glob("*.metrics.txt")):
    values = {"pair": metrics_path.name.replace(".metrics.txt", "")}
    for line in metrics_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            values[key] = float(value)
        except ValueError:
            values[key] = value
    rows.append(values)

rows.sort(key=lambda item: (item.get("total_time", 0.0), item.get("hbm_read_bytes", 0.0)), reverse=True)
(root / "profile_summaries" / "top_pairs.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
for row in rows[:8]:
    print(json.dumps(row, sort_keys=True))
PY

mapfile -t top_pairs < <(python3 - "${ROOT}" <<'PY'
import json
import sys
from pathlib import Path

rows = json.loads((Path(sys.argv[1]) / "profile_summaries" / "top_pairs.json").read_text())
for row in rows[:4]:
    print(row["pair"])
PY
)

for pair_id in "${top_pairs[@]}"; do
  neff="$(find "${NEURON_RT_INSPECT_OUTPUT_DIR}" -type f -name "neff_${pair_id}.neff" | sort | head -n 1)"
  ntff="$(find "${NEURON_RT_INSPECT_OUTPUT_DIR}" -type f -name "${pair_id}.ntff" | sort | head -n 1)"
  profile_name="qwen_tkg_${pair_id}"
  log "INGEST pair=${pair_id} profile=${profile_name}"
  set +e
  "${TOOL}" view \
    -n "${neff}" \
    -s "${ntff}" \
    --data-path "${ROOT}/neuron_profile_data" \
    --display-name "${profile_name}" \
    --ingest-only \
    --ignore-nc-buf-usage > "${ROOT}/profile_summaries/${pair_id}.ingest.log" 2>&1
  ingest_rc=$?
  if [[ "${ingest_rc}" -ne 0 ]]; then
    "${TOOL}" view \
      -n "${neff}" \
      -s "${ntff}" \
      --data-path "${ROOT}/neuron_profile_data" \
      --display-name "${profile_name}" \
      --ingest-only >> "${ROOT}/profile_summaries/${pair_id}.ingest.log" 2>&1
    ingest_rc=$?
  fi
  set -e
  echo "${ingest_rc}" > "${ROOT}/profile_summaries/${pair_id}.ingest.exit"
  log "INGEST_EXIT pair=${pair_id} rc=${ingest_rc}"
done

python3 - "${ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = root / "neuron_profile_data" / "profiles" / "global"

try:
    import pandas as pd
except Exception as exc:
    (root / "analysis" / "analysis_error.txt").write_text(f"pandas_import_error: {exc}\n")
    raise

def read_table(profile_dir, table):
    path = profile_dir / f"{table}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)

def top_records(df, by, cols, n=20):
    if df is None or df.empty:
        return []
    work = df.copy()
    grouped = work.groupby(cols, dropna=False)[by].sum().reset_index()
    grouped = grouped.sort_values(by, ascending=False).head(n)
    return json.loads(grouped.to_json(orient="records"))

reports = []
for profile_dir in sorted(base.glob("qwen_tkg_*@latest")):
    report = {"profile": profile_dir.name.replace("@latest", "")}
    summary = read_table(profile_dir, "Summary")
    inst = read_table(profile_dir, "Instruction")
    dma = read_table(profile_dir, "DmaPacket")
    dma_agg = read_table(profile_dir, "DmaPacketAggregated")
    tensors = read_table(profile_dir, "TensorInfo")
    active = read_table(profile_dir, "ActiveTime")
    metadata = read_table(profile_dir, "Metadata")

    if summary is not None and not summary.empty:
        report["summary"] = json.loads(summary.iloc[0].to_json())
    report["tables"] = {
        "Instruction": 0 if inst is None else int(len(inst)),
        "DmaPacket": 0 if dma is None else int(len(dma)),
        "DmaPacketAggregated": 0 if dma_agg is None else int(len(dma_agg)),
        "TensorInfo": 0 if tensors is None else int(len(tensors)),
        "ActiveTime": 0 if active is None else int(len(active)),
        "Metadata": 0 if metadata is None else int(len(metadata)),
    }

    if inst is not None and not inst.empty:
        report["instruction_by_engine_opcode_duration_ns"] = top_records(
            inst, "duration_ns", ["engine", "opcode"], n=25
        )
        byte_cols = [col for col in ["hbm_read_bytes", "hbm_write_bytes", "spill_reload_bytes", "spill_save_bytes"] if col in inst.columns]
        for col in byte_cols:
            report[f"instruction_by_engine_opcode_{col}"] = top_records(inst, col, ["engine", "opcode"], n=20)
        source_col = "bir_debug_info_source_location" if "bir_debug_info_source_location" in inst.columns else None
        if source_col is not None:
            non_null = inst[inst[source_col].notna() & (inst[source_col].astype(str) != "")]
            report["source_location_rows"] = int(len(non_null))
            if not non_null.empty:
                report["source_hotspots_duration_ns"] = top_records(
                    non_null, "duration_ns", [source_col, "engine", "opcode"], n=25
                )
                if "hbm_read_bytes" in non_null.columns:
                    report["source_hotspots_hbm_read_bytes"] = top_records(
                        non_null, "hbm_read_bytes", [source_col, "engine", "opcode"], n=25
                    )

    if dma is not None and not dma.empty:
        report["dma_packet_by_queue_type_bytes"] = top_records(dma, "transfer_bytes", ["queue_type"], n=20)
        report["dma_packet_small_transfer_counts"] = {
            "le_64B": int((dma["transfer_bytes"] <= 64).sum()) if "transfer_bytes" in dma.columns else 0,
            "le_256B": int((dma["transfer_bytes"] <= 256).sum()) if "transfer_bytes" in dma.columns else 0,
            "le_4KB": int((dma["transfer_bytes"] <= 4096).sum()) if "transfer_bytes" in dma.columns else 0,
            "total": int(len(dma)),
        }

    if dma_agg is not None and not dma_agg.empty:
        group_cols = [col for col in ["queue_type", "op", "source", "dest", "variable"] if col in dma_agg.columns]
        if group_cols:
            report["dma_aggregated_top_transfer_bytes"] = top_records(
                dma_agg, "transfer_bytes", group_cols, n=40
            )
        if "is_transpose_mode" in dma_agg.columns:
            transpose = dma_agg[dma_agg["is_transpose_mode"] == True]
            report["dma_transpose_transfer_bytes"] = int(transpose["transfer_bytes"].sum()) if "transfer_bytes" in transpose.columns else 0

    if tensors is not None and not tensors.empty:
        if "load_to_sbuf_total_size_bytes" in tensors.columns:
            report["tensor_load_to_sbuf_top"] = json.loads(
                tensors.sort_values("load_to_sbuf_total_size_bytes", ascending=False)
                .head(40)
                .to_json(orient="records")
            )
        if "load_to_sbuf_repeat_factor" in tensors.columns:
            repeated = tensors.sort_values("load_to_sbuf_repeat_factor", ascending=False).head(40)
            report["tensor_load_repeat_top"] = json.loads(repeated.to_json(orient="records"))

    reports.append(report)

(root / "analysis" / "profile_report.json").write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
print(json.dumps(reports, indent=2, sort_keys=True)[:20000])
PY

log "PROFILE_COMPLETE root=${ROOT}"
