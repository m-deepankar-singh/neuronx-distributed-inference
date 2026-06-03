#!/usr/bin/env bash
set -euo pipefail

ART="/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_stable_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260529T073442Z_kvselectfix"
RUN="/home/ubuntu/validation_logs/fp8_256k_decode_nki/kvselectfix_live_20260529T0835Z"

mkdir -p "${RUN}"

old_pids="$(pgrep -f "qwen36_chat_proxy.py|serve_qwen36.py" || true)"
if [[ -n "${old_pids}" ]]; then
  echo "STOPPING_OLD_PIDS ${old_pids}"
  kill ${old_pids} || true
  sleep 5
fi

leftover_pids="$(pgrep -f "qwen36_chat_proxy.py|serve_qwen36.py" || true)"
if [[ -n "${leftover_pids}" ]]; then
  echo "FORCE_STOPPING_OLD_PIDS ${leftover_pids}"
  kill -9 ${leftover_pids} || true
  sleep 2
fi

source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
cd /home/ubuntu/inferentia-gdn

nohup bash contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /home/ubuntu/models/Qwen3.6-27B \
  --compiled-artifacts "${ART}" \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 262144 \
  --seq-len 262144 \
  --cte-buckets 256,512 \
  --tensor-parallel-size 4 \
  --logical-nc-config 2 \
  --max-num-seqs 1 \
  --ctx-batch-size 1 \
  --async-mode \
  --enable-vllm-chunked-prefill \
  --enable-prefix-caching \
  --enable-hybrid-apc \
  --mamba-cache-mode all \
  --mamba-ssm-cache-dtype auto \
  --block-size 256 \
  --gdn-checkpoint-interval 256 \
  --max-gdn-checkpoint-slots 64 \
  --gdn-recurrent-cache-dtype bfloat16 \
  --gdn-conv-cache-dtype bfloat16 \
  --hybrid-gdn-recurrent-cache-dtype bfloat16 \
  --hybrid-gdn-conv-cache-dtype bfloat16 \
  --hybrid-cache-mode all \
  --hybrid-cache-prefix-boundary-only \
  --hybrid-apc-require-vllm-metadata \
  --hybrid-apc-enable-backed-prefix-reads \
  --num-gpu-blocks-override 1024 > "${RUN}/backend.log" 2>&1 &
backend_pid="$!"
echo "BACKEND_PID ${backend_pid}"

for i in {1..90}; do
  if curl -fsS http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    echo "BACKEND_READY attempt=${i}"
    break
  fi
  if ! kill -0 "${backend_pid}" >/dev/null 2>&1; then
    echo "BACKEND_EXITED"
    tail -80 "${RUN}/backend.log"
    exit 1
  fi
  if (( i % 10 == 0 )); then
    echo "WAIT_BACKEND attempt=${i}"
    tail -20 "${RUN}/backend.log" | sed 's/^/BACKEND_LOG /'
  fi
  sleep 10
done
curl -fsS http://127.0.0.1:8001/v1/models >/dev/null

nohup python contrib/models/Qwen3.6-27B/vllm/qwen36_chat_proxy.py \
  --backend-url http://127.0.0.1:8001 \
  --host 0.0.0.0 \
  --port 8000 \
  --allow-thinking \
  --default-thinking > "${RUN}/proxy.log" 2>&1 &
proxy_pid="$!"
echo "PROXY_PID ${proxy_pid}"

for i in {1..30}; do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "PROXY_READY attempt=${i}"
    break
  fi
  if ! kill -0 "${proxy_pid}" >/dev/null 2>&1; then
    echo "PROXY_EXITED"
    cat "${RUN}/proxy.log"
    exit 1
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null

echo "SERVER_READY_LOG_DIR ${RUN}"
