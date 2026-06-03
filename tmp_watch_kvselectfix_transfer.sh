#!/usr/bin/env bash
set -euo pipefail

PID=125870
LOG=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_stable_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260529T073442Z_kvselectfix_compile.log
ARTIFACT=/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_stable_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260529T073442Z_kvselectfix
DEST=ubuntu@172.31.46.160
KEY=/home/ubuntu/trainium.pem
WATCH_LOG=/home/ubuntu/validation_logs/fp8_256k_decode_nki/kvselectfix_transfer_20260529T073442Z.log

mkdir -p "$(dirname "$WATCH_LOG")"
exec >>"$WATCH_LOG" 2>&1

echo "WATCH_START $(date -Is) pid=$PID artifact=$ARTIFACT dest=$DEST"

while kill -0 "$PID" 2>/dev/null; do
  echo "WAIT_COMPILE $(date -Is) pid_alive=1"
  sleep 60
done

echo "PID_EXITED $(date -Is)"

if ! grep -q "COMPILE_DONE" "$LOG"; then
  echo "COMPILE_DONE_NOT_FOUND $(date -Is)"
  tail -120 "$LOG" || true
  exit 1
fi

echo "COMPILE_DONE_FOUND $(date -Is)"
du -sh "$ARTIFACT"

ssh -i "$KEY" -o StrictHostKeyChecking=no "$DEST" \
  "mkdir -p /mnt/trainium_artifacts/qwen_artifacts"

echo "RSYNC_START $(date -Is)"
rsync -aH --delete --info=progress2 \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$ARTIFACT/" "$DEST:$ARTIFACT/"
echo "RSYNC_DONE $(date -Is)"

ssh -i "$KEY" -o StrictHostKeyChecking=no "$DEST" "du -sh '$ARTIFACT'"
echo "TRANSFER_DONE $(date -Is)"
