#!/usr/bin/env bash
set -eo pipefail

ARTIFACT="${1:?usage: tmp_launch_qwen36_segcte2048.sh ARTIFACT [LOG]}"
LOG="${2:-/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_segcte2048_runtime_$(date -u +%Y%m%dT%H%M%SZ).log}"
PIDFILE="${LOG%.log}.pid"
ENVLOG="${LOG%.log}_env.txt"
REPO="${REPO:-/home/ubuntu/inferentia-gdn-prefill-speed-coherent}"
MODEL="${MODEL:-/home/ubuntu/models/Qwen3.6-27B}"
VENV="${VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16}"
LAUNCH_DRY_RUN="${LAUNCH_DRY_RUN:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
SEQ_LEN="${SEQ_LEN:-${MAX_MODEL_LEN}}"
CTE_BUCKETS="${CTE_BUCKETS:-2048}"
CONTEXT_ENCODING_BUCKET_PAIRS="${CONTEXT_ENCODING_BUCKET_PAIRS:-2048:0 2048:256 2048:512 2048:1024 2048:2048 2048:4096 2048:8192 2048:16384 2048:32768 2048:65536 2048:131072 2048:262144}"
TOKEN_GENERATION_BUCKETS="${TOKEN_GENERATION_BUCKETS:-512 16384 16640 32768 65536 131072 262144}"
MAX_GDN_CHECKPOINT_SLOTS="${MAX_GDN_CHECKPOINT_SLOTS:-64}"
GDN_RECURRENT_CACHE_DTYPE="${GDN_RECURRENT_CACHE_DTYPE:-float32}"
GDN_CONV_CACHE_DTYPE="${GDN_CONV_CACHE_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
QWEN36_DELTANET_MULTIHEAD_CTE="${QWEN36_DELTANET_MULTIHEAD_CTE:-0}"
QWEN36_SPLIT_QKV_TKG_ROW_SCALES="${QWEN36_SPLIT_QKV_TKG_ROW_SCALES:-0}"
QWEN36_DELTANET_FUSED_SEGMENT_TOKENS="${QWEN36_DELTANET_FUSED_SEGMENT_TOKENS:-512}"
QWEN36_DELTANET_SOLVE_MODE="${QWEN36_DELTANET_SOLVE_MODE:-direct}"
QWEN36_DELTANET_SOLVE_SCAN_STEPS="${QWEN36_DELTANET_SOLVE_SCAN_STEPS:-0}"
DISABLE_HYBRID_KV_CACHE_MANAGER="${DISABLE_HYBRID_KV_CACHE_MANAGER:-0}"
if [[ "${DISABLE_HYBRID_KV_CACHE_MANAGER}" != "0" ]]; then
  echo "ERROR: DISABLE_HYBRID_KV_CACHE_MANAGER must remain 0/absent for Qwen hybrid KV budgeting" >&2
  exit 2
fi
GPU_MEMORY_ARGS=()
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  GPU_MEMORY_ARGS=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
NUM_GPU_BLOCKS_ARGS=()
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
  NUM_GPU_BLOCKS_ARGS=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
fi
KV_CACHE_DTYPE_ARGS=()
if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  KV_CACHE_DTYPE_ARGS=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
KV_CACHE_MEMORY_ARGS=()
if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]]; then
  KV_CACHE_MEMORY_ARGS=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
fi

mkdir -p "$(dirname "${LOG}")"

printf "%s\n" \
  "ARTIFACT=${ARTIFACT}" \
  "LOG=${LOG}" \
  "ENVLOG=${ENVLOG}" \
  "PIDFILE=${PIDFILE}" \
  "REPO=${REPO}" \
  "MODEL=${MODEL}" \
  "VENV=${VENV}" \
  "LAUNCH_DRY_RUN=${LAUNCH_DRY_RUN}" \
  "MAX_MODEL_LEN=${MAX_MODEL_LEN}" \
  "SEQ_LEN=${SEQ_LEN}" \
  "CTE_BUCKETS=${CTE_BUCKETS}" \
  "CONTEXT_ENCODING_BUCKET_PAIRS=${CONTEXT_ENCODING_BUCKET_PAIRS}" \
  "TOKEN_GENERATION_BUCKETS=${TOKEN_GENERATION_BUCKETS}" \
  "MAX_GDN_CHECKPOINT_SLOTS=${MAX_GDN_CHECKPOINT_SLOTS}" \
  "GDN_RECURRENT_CACHE_DTYPE=${GDN_RECURRENT_CACHE_DTYPE}" \
  "GDN_CONV_CACHE_DTYPE=${GDN_CONV_CACHE_DTYPE}" \
  "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-}" \
  "NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE:-}" \
  "KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-}" \
  "KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-}" \
  "HOST=${HOST}" \
  "PORT=${PORT}" \
  "QWEN36_DELTANET_MULTIHEAD_CTE=${QWEN36_DELTANET_MULTIHEAD_CTE}" \
  "QWEN36_SPLIT_QKV_TKG_ROW_SCALES=${QWEN36_SPLIT_QKV_TKG_ROW_SCALES}" \
  "QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=${QWEN36_DELTANET_FUSED_SEGMENT_TOKENS}" \
  "QWEN36_DELTANET_SOLVE_MODE=${QWEN36_DELTANET_SOLVE_MODE}" \
  "QWEN36_DELTANET_SOLVE_SCAN_STEPS=${QWEN36_DELTANET_SOLVE_SCAN_STEPS}" \
  "DISABLE_HYBRID_KV_CACHE_MANAGER=${DISABLE_HYBRID_KV_CACHE_MANAGER}" \
  "ENABLE_HYBRID_APC=1" \
  "ENABLE_PREFIX_CACHING=1" \
  "ENABLE_VLLM_CHUNKED_PREFILL=1" \
  "HYBRID_APC_REQUIRE_VLLM_METADATA=1" \
  "HYBRID_APC_ENABLE_BACKED_PREFIX_READS=1" \
  "BLOCK_SIZE=256" \
  "GDN_CHECKPOINT_INTERVAL=256" \
  "GPU_MEMORY_ARGS=${GPU_MEMORY_ARGS[*]}" \
  "NUM_GPU_BLOCKS_ARGS=${NUM_GPU_BLOCKS_ARGS[*]}" \
  "KV_CACHE_DTYPE_ARGS=${KV_CACHE_DTYPE_ARGS[*]}" \
  "KV_CACHE_MEMORY_ARGS=${KV_CACHE_MEMORY_ARGS[*]}" \
  >"${ENVLOG}"

if [[ "${LAUNCH_DRY_RUN}" == "1" ]]; then
  echo "ARTIFACT=${ARTIFACT}"
  echo "LOG=${LOG}"
  echo "ENVLOG=${ENVLOG}"
  echo "PIDFILE=${PIDFILE}"
  echo "LAUNCH_DRY_RUN=1"
  exit 0
elif [[ "${LAUNCH_DRY_RUN}" != "0" ]]; then
  echo "ERROR: LAUNCH_DRY_RUN must be 0 or 1, got ${LAUNCH_DRY_RUN}" >&2
  exit 2
fi

pkill -9 -f "[V]LLM::EngineCore" || true
pkill -9 -f "[s]erve_qwen36.py" || true
pkill -9 -f "[p]ython.*vllm" || true
sleep 2

cd "${REPO}/contrib/models/Qwen3.6-27B/vllm"
source "${VENV}/bin/activate"

nohup env \
  NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  NEURON_CC_FLAGS="--target trn2 --lnc 2" \
  NKI_LIBRARY_SRC=/home/ubuntu/nki-library-2.30/src/nkilib_src \
  USE_NKI_DECODE=1 \
  QWEN36_DELTANET_CTE_IMPL=current \
  QWEN36_DELTANET_MULTIHEAD_CTE="${QWEN36_DELTANET_MULTIHEAD_CTE}" \
  QWEN36_DELTANET_FUSED_SEGMENT_TOKENS="${QWEN36_DELTANET_FUSED_SEGMENT_TOKENS}" \
  QWEN36_DELTANET_CHUNK_SIZE=128 \
  QWEN36_DELTANET_SOLVE_BLOCK_SIZE=128 \
  QWEN36_DELTANET_SOLVE_MODE="${QWEN36_DELTANET_SOLVE_MODE}" \
  QWEN36_DELTANET_SOLVE_SCAN_STEPS="${QWEN36_DELTANET_SOLVE_SCAN_STEPS}" \
  QWEN36_PREFIX_ATTENTION_IMPL=expanded \
  QWEN36_SPLIT_QKV_TKG_ROW_SCALES="${QWEN36_SPLIT_QKV_TKG_ROW_SCALES}" \
  QWEN36_HYBRID_APC_DEBUG=1 \
  QWEN36_HYBRID_GDN_STATE_DEBUG="${QWEN36_HYBRID_GDN_STATE_DEBUG:-0}" \
  QWEN36_ZERO_HYBRID_GDN_RESTORE_MASK="${QWEN36_ZERO_HYBRID_GDN_RESTORE_MASK:-0}" \
  QWEN36_VLLM_LOGITS_DEBUG=1 \
  ./start_vllm_server.sh \
    --model-path "${MODEL}" \
    --compiled-artifacts "${ARTIFACT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --seq-len "${SEQ_LEN}" \
    --cte-buckets "${CTE_BUCKETS}" \
    --context-encoding-bucket-pairs "${CONTEXT_ENCODING_BUCKET_PAIRS}" \
    --token-generation-buckets "${TOKEN_GENERATION_BUCKETS}" \
    --tensor-parallel-size 4 \
    --logical-nc-config 2 \
    --max-num-seqs 1 \
    --ctx-batch-size 1 \
    --async-mode \
    --enable-vllm-chunked-prefill \
    --enable-prefix-caching \
    --enable-hybrid-apc \
    --block-size 256 \
    "${NUM_GPU_BLOCKS_ARGS[@]}" \
    --gdn-checkpoint-interval 256 \
    --max-gdn-checkpoint-slots "${MAX_GDN_CHECKPOINT_SLOTS}" \
    --gdn-recurrent-cache-dtype "${GDN_RECURRENT_CACHE_DTYPE}" \
    --gdn-conv-cache-dtype "${GDN_CONV_CACHE_DTYPE}" \
    --hybrid-cache-mode all \
    --hybrid-apc-require-vllm-metadata \
    --hybrid-apc-enable-backed-prefix-reads \
    "${GPU_MEMORY_ARGS[@]}" \
    "${KV_CACHE_DTYPE_ARGS[@]}" \
    "${KV_CACHE_MEMORY_ARGS[@]}" \
    --host "${HOST}" \
    --port "${PORT}" \
  >"${LOG}" 2>&1 &

echo "$!" >"${PIDFILE}"
echo "LOG=${LOG}"
echo "ENVLOG=${ENVLOG}"
echo "PIDFILE=${PIDFILE}"
echo "PID=$(cat "${PIDFILE}")"

for attempt in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "HEALTH_OK attempt=${attempt}"
    exit 0
  fi
  if ! kill -0 "$(cat "${PIDFILE}")" >/dev/null 2>&1; then
    echo "SERVER_EXITED_BEFORE_HEALTH log=${LOG}" >&2
    tail -n 120 "${LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "HEALTH_TIMEOUT log=${LOG}" >&2
tail -n 160 "${LOG}" >&2 || true
exit 1
