#!/usr/bin/env bash
set -euo pipefail

: "${PID:?set PID}"
: "${LOG:?set LOG}"
: "${ART:?set ART}"
: "${RUN_ID:?set RUN_ID}"

SRC_REPO=${SRC_REPO:-/home/ubuntu/inferentia-gdn-nki-decode-step-c49df2b-splitqkv}
TRN2=${TRN2:-ubuntu@172.31.32.124}
KEY=${KEY:-/home/ubuntu/.ssh/trn2_transfer_ed25519}
DEST_REPO=${DEST_REPO:-/home/ubuntu/inferentia-gdn-nki-decode-step-c49df2b-splitqkv}
DEST_ART_ROOT=${DEST_ART_ROOT:-/mnt/trainium_artifacts/qwen_artifacts}
WATCH_LOG=${WATCH_LOG:-/home/ubuntu/validation_logs/fp8_256k_decode_nki/splitqkv_transfer_eval_${RUN_ID}.log}
EVAL_ROOT=${EVAL_ROOT:-/home/ubuntu/validation_logs/fp8_256k_decode_nki/splitqkv_decode_eval_${RUN_ID}}
POLL_SECONDS=${POLL_SECONDS:-300}

mkdir -p "$(dirname "${WATCH_LOG}")"
exec > >(tee -a "${WATCH_LOG}") 2>&1

timestamp() {
  date -Is
}

printf "[%s] WATCH_START pid=%s art=%s\n" "$(timestamp)" "${PID}" "${ART}"
while kill -0 "${PID}" 2>/dev/null; do
  printf "[%s] WAIT_COMPILE pid=%s\n" "$(timestamp)" "${PID}"
  sleep "${POLL_SECONDS}"
done

printf "[%s] COMPILE_PROCESS_EXITED pid=%s\n" "$(timestamp)" "${PID}"
if ! grep -q "COMPILE_DONE" "${LOG}"; then
  printf "[%s] COMPILE_FAILED_OR_INCOMPLETE\n" "$(timestamp)"
  tail -n 160 "${LOG}" || true
  exit 10
fi

printf "[%s] COMPILE_DONE_CONFIRMED\n" "$(timestamp)"
ssh -i "${KEY}" -o StrictHostKeyChecking=accept-new "${TRN2}" \
  "mkdir -p '${DEST_REPO}' '${DEST_ART_ROOT}' '${EVAL_ROOT}'"

printf "[%s] RSYNC_REPO_START\n" "$(timestamp)"
rsync -a --partial --info=progress2 \
  --exclude ".git" --exclude "__pycache__" --exclude "*.pyc" \
  -e "ssh -i ${KEY} -o StrictHostKeyChecking=accept-new" \
  "${SRC_REPO}/" "${TRN2}:${DEST_REPO}/"

printf "[%s] RSYNC_ARTIFACT_START\n" "$(timestamp)"
rsync -a --partial --info=progress2 \
  -e "ssh -i ${KEY} -o StrictHostKeyChecking=accept-new" \
  "${ART}/" "${TRN2}:${DEST_ART_ROOT}/$(basename "${ART}")/"

printf "[%s] RSYNC_DONE\n" "$(timestamp)"
printf "[%s] REMOTE_EVAL_START\n" "$(timestamp)"
ssh -i "${KEY}" -o StrictHostKeyChecking=accept-new "${TRN2}" \
  "cd '${DEST_REPO}' && chmod +x tmp_run_qwen256k_fp8_tighttkg_decode_eval.sh && ARTIFACT='${DEST_ART_ROOT}/$(basename "${ART}")' REPO='${DEST_REPO}' ROOT='${EVAL_ROOT}' GDN_RECURRENT_CACHE_DTYPE=bfloat16 GDN_CONV_CACHE_DTYPE=bfloat16 LENGTHS=512,16384 MAX_TOKENS=256 REPEATS=1 ./tmp_run_qwen256k_fp8_tighttkg_decode_eval.sh"

printf "[%s] REMOTE_EVAL_DONE root=%s\n" "$(timestamp)" "${EVAL_ROOT}"
