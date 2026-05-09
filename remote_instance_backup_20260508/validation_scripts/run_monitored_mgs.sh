#!/usr/bin/env bash
set -euo pipefail
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export PYTHONPATH=/home/ubuntu/inferentia-gdn/contrib/models/Qwen3.6-27B:/home/ubuntu/inferentia-gdn:${PYTHONPATH:-}
RUN_ID=$(date +%Y%m%d_%H%M%S)
MON_CFG=/home/ubuntu/validation_logs/neuron_monitor_memory_${RUN_ID}.json
MON_LOG=/home/ubuntu/validation_logs/qwen36_27b_mgs_hbm_monitor_${RUN_ID}.jsonl
RUN_LOG=/home/ubuntu/validation_logs/qwen36_27b_mgs_hbm_prompt_${RUN_ID}.log
cat > "$MON_CFG" <<JSON
{
  "period": 1,
  "runtimes": [
    {
      "tag_filter": ".*",
      "metrics": [
        {"type": "memory", "period": 1},
        {"type": "neuroncore_counters", "period": 1}
      ]
    }
  ],
  "output_targets": [
    {"type": "json_stream"}
  ]
}
JSON
neuron-monitor -c "$MON_CFG" > "$MON_LOG" 2>&1 &
MON_PID=$!
sleep 2
python /home/ubuntu/validation_scripts/qwen36_27b_manual_chunk_runner.py \
  --compiled-path /opt/dlami/nvme/qwen_artifacts/qwen36_27b_hybrid_chunked_nki_cte128_tkg65536_run1 \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --contrib-root /home/ubuntu/inferentia-gdn/contrib/models/Qwen3.6-27B \
  --prompt "Question: In Metal Gear Solid, what themes make the story memorable? Answer in one paragraph.\nAnswer:" \
  --chunk-size 128 \
  --max-new-tokens 128 \
  --seq-len 65536 2>&1 | tee "$RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}
sleep 2
kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true
echo "RUN_LOG=$RUN_LOG"
echo "MON_LOG=$MON_LOG"
exit "$RUN_STATUS"
