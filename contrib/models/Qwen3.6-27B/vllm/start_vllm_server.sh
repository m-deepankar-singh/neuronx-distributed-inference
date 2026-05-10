#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=""
COMPILED_ARTIFACTS=""
MAX_MODEL_LEN="512"
SEQ_LEN="512"
CTE_BUCKET="512"
TP_DEGREE="4"
LNC="2"
MAX_NUM_SEQS="1"
PORT="8000"
HOST="0.0.0.0"
ENABLE_CHUNKED_PREFILL="0"
BLOCK_SIZE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --compiled-artifacts) COMPILED_ARTIFACTS="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --cte-bucket) CTE_BUCKET="$2"; shift 2 ;;
    --tensor-parallel-size) TP_DEGREE="$2"; shift 2 ;;
    --logical-nc-config) LNC="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --enable-vllm-chunked-prefill) ENABLE_CHUNKED_PREFILL="1"; shift ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: --model-path is required" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTRIB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${CONTRIB_ROOT}:${PYTHONPATH:-}"
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export VLLM_PLUGINS="${VLLM_PLUGINS:-neuron}"

if [[ -n "${COMPILED_ARTIFACTS}" ]]; then
  export NEURON_COMPILED_ARTIFACTS="${COMPILED_ARTIFACTS}"
fi
if [[ -z "${BLOCK_SIZE}" ]]; then
  BLOCK_SIZE="256"
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  export DISABLE_NEURON_CUSTOM_SCHEDULER="1"
fi

ADDITIONAL_CONFIG="$(
  python3 - <<PY
import json
enable_chunked = "${ENABLE_CHUNKED_PREFILL}" == "1"
neuron_config = {
    "tp_degree": int("${TP_DEGREE}"),
    "batch_size": int("${MAX_NUM_SEQS}"),
    "ctx_batch_size": 1,
    "tkg_batch_size": int("${MAX_NUM_SEQS}"),
    "seq_len": int("${SEQ_LEN}"),
    "max_length": int("${SEQ_LEN}"),
    "max_context_length": int("${CTE_BUCKET}"),
    "context_encoding_buckets": [int("${CTE_BUCKET}")],
    "token_generation_buckets": [int("${SEQ_LEN}")],
    "enable_bucketing": False,
    "logical_nc_config": int("${LNC}"),
    "torch_dtype": "bfloat16",
    "save_sharded_checkpoint": True,
}
if enable_chunked:
    neuron_config.update({
        "is_block_kv_layout": True,
        "chunked_prefill_config": {
            "max_num_seqs": int("${MAX_NUM_SEQS}"),
            "tkg_model_enabled": True,
            "kernel_q_tile_size": 128,
            "kernel_kv_tile_size": 1024,
        },
    })
print(json.dumps({
    "max_prompt_length": int("${CTE_BUCKET}"),
    "override_neuron_config": neuron_config,
}))
PY
)"

echo "Starting vLLM for Qwen3.6-27B"
echo "MODEL_PATH=${MODEL_PATH}"
echo "NEURON_COMPILED_ARTIFACTS=${NEURON_COMPILED_ARTIFACTS:-}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "ADDITIONAL_CONFIG=${ADDITIONAL_CONFIG}"

VLLM_ARGS=(
  "${MODEL_PATH}"
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size "${TP_DEGREE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --generation-config vllm \
  --no-enable-prefix-caching \
  --additional-config "${ADDITIONAL_CONFIG}"
)
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  VLLM_ARGS+=(
    --enable-chunked-prefill
    --max-num-batched-tokens "${CTE_BUCKET}"
    --block-size "${BLOCK_SIZE}"
  )
else
  VLLM_ARGS+=(--no-enable-chunked-prefill)
fi

exec python "${SCRIPT_DIR}/serve_qwen36.py" "${VLLM_ARGS[@]}"
