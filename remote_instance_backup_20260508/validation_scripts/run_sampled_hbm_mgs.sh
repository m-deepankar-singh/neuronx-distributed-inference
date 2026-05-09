#!/usr/bin/env bash
set -euo pipefail
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export PYTHONPATH=/home/ubuntu/inferentia-gdn/contrib/models/Qwen3.6-27B:/home/ubuntu/inferentia-gdn:${PYTHONPATH:-}
RUN_ID=$(date +%Y%m%d_%H%M%S)
MON_LOG=/home/ubuntu/validation_logs/qwen36_27b_mgs_hbm_sampled_${RUN_ID}.jsonl
RUN_LOG=/home/ubuntu/validation_logs/qwen36_27b_mgs_hbm_sampled_prompt_${RUN_ID}.log
python /home/ubuntu/validation_scripts/qwen36_27b_manual_chunk_runner.py \
  --compiled-path /opt/dlami/nvme/qwen_artifacts/qwen36_27b_hybrid_chunked_nki_cte128_tkg65536_run1 \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --contrib-root /home/ubuntu/inferentia-gdn/contrib/models/Qwen3.6-27B \
  --prompt "Question: In Metal Gear Solid, what themes make the story memorable? Answer in one paragraph.\nAnswer:" \
  --chunk-size 128 \
  --max-new-tokens 128 \
  --seq-len 65536 > "$RUN_LOG" 2>&1 &
RUN_PID=$!
while kill -0 "$RUN_PID" 2>/dev/null; do
  neuron-monitor >> "$MON_LOG" 2>> "${MON_LOG}.err" || true
  sleep 1
done
wait "$RUN_PID"
RUN_STATUS=$?
echo "RUN_LOG=$RUN_LOG"
echo "MON_LOG=$MON_LOG"
echo "MON_ERR=${MON_LOG}.err"
tail -80 "$RUN_LOG"
exit "$RUN_STATUS"
