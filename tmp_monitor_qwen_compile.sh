#!/usr/bin/env bash
set -euo pipefail

: "${PID:?PID is required}"
: "${LOG:?LOG is required}"
: "${ENVLOG:?ENVLOG is required}"
: "${ART:?ART is required}"
: "${WORK:?WORK is required}"

echo "PID_STATUS"
ps -p "${PID}" -o pid,ppid,etime,stat,cmd || true

echo "NEURONX_CC_CHILDREN"
pgrep -af neuronx-cc | head -20 || true

echo "ENV_EXPECTATIONS"
grep -E '^(QWEN36_DELTANET_CHUNK_SIZE|QWEN36_DELTANET_SOLVE_BLOCK_SIZE|QWEN36_DELTANET_SOLVE_SCAN_STEPS|CTE_BUCKETS|PREFIX_BUCKETS|TOKEN_GENERATION_BUCKETS|CONTEXT_ENCODING_BUCKET_PAIRS|PREFIX_CTE_ATTENTION_BACKEND|SEQ_LEN|MAX_CONTEXT_LENGTH|PA_NUM_BLOCKS|NEURON_CC_FLAGS|OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING|DISABLE_CONTEXT_ENCODING_ARGMAX_KERNEL)=' "${ENVLOG}" || true

echo "HARD_ERROR_SCAN"
if [ -f "${LOG}" ]; then
  grep -Ein 'Traceback|NkiValidationError|RuntimeError|No space left|OOM|FAILED|failed|Error|ERROR' "${LOG}" | tail -40 || true
else
  echo "log_missing=${LOG}"
fi

echo "LOG_TAIL"
if [ -f "${LOG}" ]; then
  tail -80 "${LOG}"
else
  echo "log_missing=${LOG}"
fi

echo "ARTIFACT_SIZE"
du -sh "${ART}" "${WORK}" 2>/dev/null || true

echo "DISK"
df -h / /mnt
