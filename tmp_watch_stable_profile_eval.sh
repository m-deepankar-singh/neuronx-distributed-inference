#!/usr/bin/env bash
set -euo pipefail

: "${WATCH_LOG:?set WATCH_LOG}"
: "${RUN_ID:?set RUN_ID}"
: "${ARTIFACT:?set ARTIFACT}"

TRN2=${TRN2:-ubuntu@172.31.46.160}
KEY=${KEY:-/home/ubuntu/.ssh/trn2_transfer_ed25519}
DEST_REPO=${DEST_REPO:-/home/ubuntu/inferentia-gdn}
PROFILE_SCRIPT=${PROFILE_SCRIPT:-tmp_profile_qwen256k_decode_tkg_dge.sh}
PROFILE_ROOT=${PROFILE_ROOT:-/home/ubuntu/validation_logs/fp8_256k_decode_nki/stable_decode_profile_${RUN_ID}}
PROFILE_WATCH_LOG=${PROFILE_WATCH_LOG:-/home/ubuntu/validation_logs/fp8_256k_decode_nki/stable_profile_watch_${RUN_ID}.log}
POLL_SECONDS=${POLL_SECONDS:-300}
PROFILE_LENGTHS=${PROFILE_LENGTHS:-512}
PROFILE_MAX_TOKENS=${PROFILE_MAX_TOKENS:-64}
PROFILE_REPEATS=${PROFILE_REPEATS:-1}

mkdir -p "$(dirname "${PROFILE_WATCH_LOG}")"
exec > >(tee -a "${PROFILE_WATCH_LOG}") 2>&1

timestamp() {
  date -Is
}

printf "[%s] PROFILE_WATCH_START watch_log=%s artifact=%s\n" "$(timestamp)" "${WATCH_LOG}" "${ARTIFACT}"
while true; do
  if grep -q "REMOTE_EVAL_DONE" "${WATCH_LOG}" 2>/dev/null; then
    break
  fi
  if grep -Eq "COMPILE_FAILED_OR_INCOMPLETE|REMOTE_EVAL_FAILED|BACKEND_EXITED|BACKEND_READY_TIMEOUT|PROXY_READY_TIMEOUT" "${WATCH_LOG}" 2>/dev/null; then
    printf "[%s] BASE_EVAL_FAILED_OR_INCOMPLETE\n" "$(timestamp)"
    tail -n 160 "${WATCH_LOG}" || true
    exit 20
  fi
  printf "[%s] WAIT_BASE_EVAL\n" "$(timestamp)"
  sleep "${POLL_SECONDS}"
done

printf "[%s] BASE_EVAL_DONE_CONFIRMED\n" "$(timestamp)"
printf "[%s] PROFILE_REMOTE_START root=%s\n" "$(timestamp)" "${PROFILE_ROOT}"
ssh -i "${KEY}" -o StrictHostKeyChecking=accept-new "${TRN2}" \
  "cd '${DEST_REPO}' && chmod +x '${PROFILE_SCRIPT}' && ARTIFACT='${ARTIFACT}' REPO='${DEST_REPO}' ROOT='${PROFILE_ROOT}' LENGTHS='${PROFILE_LENGTHS}' MAX_TOKENS='${PROFILE_MAX_TOKENS}' REPEATS='${PROFILE_REPEATS}' BACKEND_PORT=8011 PROXY_PORT=8010 ./'${PROFILE_SCRIPT}'"
printf "[%s] PROFILE_REMOTE_DONE root=%s\n" "$(timestamp)" "${PROFILE_ROOT}"
