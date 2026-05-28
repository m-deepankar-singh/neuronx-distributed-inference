#!/usr/bin/env bash
set -euo pipefail

: "${WAIT_PID:?set WAIT_PID to the compile PID that must finish first}"

REPO=${REPO:-/home/ubuntu/inferentia-gdn-nki-decode-step-c49df2b-splitqkvgate}
LOGDIR=${LOGDIR:-/home/ubuntu/validation_logs/fp8_256k_decode_nki}
POLL_SECONDS=${POLL_SECONDS:-300}
VARIANT_TAG=${VARIANT_TAG:-splitqkvgate_mlpker}
COMPILE_SCRIPT=${COMPILE_SCRIPT:-tmp_compile_qwen256k_fp8_full_decode_splitqkv_mlpker_sampletokens.sh}
WATCH_SCRIPT=${WATCH_SCRIPT:-tmp_watch_splitqkv_transfer_eval.sh}

mkdir -p "${LOGDIR}"

timestamp() {
  date -Is
}

printf "[%s] DEFER_START wait_pid=%s repo=%s variant=%s\n" \
  "$(timestamp)" "${WAIT_PID}" "${REPO}" "${VARIANT_TAG}"

while kill -0 "${WAIT_PID}" 2>/dev/null; do
  printf "[%s] WAIT_PREVIOUS_COMPILE pid=%s\n" "$(timestamp)" "${WAIT_PID}"
  sleep "${POLL_SECONDS}"
done

printf "[%s] PREVIOUS_COMPILE_EXITED pid=%s\n" "$(timestamp)" "${WAIT_PID}"
cd "${REPO}"
chmod +x "${COMPILE_SCRIPT}" "${WATCH_SCRIPT}"

TS=$(date -u +%Y%m%dT%H%M%SZ)
printf "[%s] NEXT_COMPILE_LAUNCH ts=%s\n" "$(timestamp)" "${TS}"
launch_output=$(
  TS="${TS}" \
  REPO="${REPO}" \
  VARIANT_TAG="${VARIANT_TAG}" \
  "./${COMPILE_SCRIPT}"
)
printf "%s\n" "${launch_output}"

extract_value() {
  local key="$1"
  printf "%s\n" "${launch_output}" | awk -F= -v key="${key}" '$1 == key {print substr($0, length(key) + 2)}' | tail -n 1
}

PID=$(extract_value PID)
ART=$(extract_value ARTIFACT)
LOG=$(extract_value LOG)

if [[ -z "${PID}" || -z "${ART}" || -z "${LOG}" ]]; then
  printf "[%s] NEXT_COMPILE_PARSE_FAILED\n" "$(timestamp)"
  exit 20
fi

WATCH_LOG="${LOGDIR}/splitqkvgate_transfer_eval_${TS}.log"
EVAL_ROOT="${LOGDIR}/splitqkvgate_decode_eval_${TS}"
printf "[%s] NEXT_WATCH_LAUNCH pid=%s art=%s log=%s watch_log=%s eval_root=%s\n" \
  "$(timestamp)" "${PID}" "${ART}" "${LOG}" "${WATCH_LOG}" "${EVAL_ROOT}"

PID="${PID}" \
LOG="${LOG}" \
ART="${ART}" \
RUN_ID="${TS}" \
SRC_REPO="${REPO}" \
WATCH_LOG="${WATCH_LOG}" \
EVAL_ROOT="${EVAL_ROOT}" \
nohup "./${WATCH_SCRIPT}" >"${LOGDIR}/splitqkvgate_watch_${TS}.nohup.log" 2>&1 &

printf "[%s] NEXT_WATCH_PID pid=%s\n" "$(timestamp)" "$!"
