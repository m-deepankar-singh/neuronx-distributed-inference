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
MAX_BATCH_SIZE=""
PORT="8000"
HOST="0.0.0.0"
ENABLE_CHUNKED_PREFILL="0"
ENABLE_PREFIX_CACHING="0"
ENABLE_ON_DEVICE_SAMPLING="0"
ASYNC_MODE="0"
OUTPUT_LOGITS="false"
TOKEN_GENERATION_BUCKETS=""
TOKEN_GENERATION_BATCHES=""
MAMBA_CACHE_MODE=""
MAMBA_CACHE_DTYPE=""
MAMBA_SSM_CACHE_DTYPE=""
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
    --max-batch-size) MAX_BATCH_SIZE="$2"; shift 2 ;;
    --enable-vllm-chunked-prefill) ENABLE_CHUNKED_PREFILL="1"; shift ;;
    --enable-prefix-caching) ENABLE_PREFIX_CACHING="1"; shift ;;
    --disable-prefix-caching|--no-enable-prefix-caching) ENABLE_PREFIX_CACHING="0"; shift ;;
    --enable-on-device-sampling) ENABLE_ON_DEVICE_SAMPLING="1"; shift ;;
    --disable-on-device-sampling|--no-enable-on-device-sampling) ENABLE_ON_DEVICE_SAMPLING="0"; shift ;;
    --async-mode) ASYNC_MODE="1"; shift ;;
    --no-async-mode) ASYNC_MODE="0"; shift ;;
    --output-logits) OUTPUT_LOGITS="$2"; shift 2 ;;
    --token-generation-buckets) TOKEN_GENERATION_BUCKETS="$2"; shift 2 ;;
    --token-generation-batches) TOKEN_GENERATION_BATCHES="$2"; shift 2 ;;
    --mamba-cache-mode) MAMBA_CACHE_MODE="$2"; shift 2 ;;
    --mamba-cache-dtype) MAMBA_CACHE_DTYPE="$2"; shift 2 ;;
    --mamba-ssm-cache-dtype) MAMBA_SSM_CACHE_DTYPE="$2"; shift 2 ;;
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
if [[ -z "${MAX_BATCH_SIZE}" ]]; then
  MAX_BATCH_SIZE="${MAX_NUM_SEQS}"
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  export DISABLE_NEURON_CUSTOM_SCHEDULER="1"
fi

ADDITIONAL_CONFIG="$(
  python3 - <<PY
import json


def parse_bool(name, value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be true or false, got {value!r}")


def parse_int_list(name, value):
    if not value:
        return None
    try:
        items = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated integer list") from exc
    if not items or any(item < 1 for item in items):
        raise SystemExit(f"{name} must contain positive integers")
    return items


enable_chunked = "${ENABLE_CHUNKED_PREFILL}" == "1"
enable_on_device_sampling = "${ENABLE_ON_DEVICE_SAMPLING}" == "1"
async_mode = "${ASYNC_MODE}" == "1"
output_logits = parse_bool("OUTPUT_LOGITS", "${OUTPUT_LOGITS}")
seq_len = int("${SEQ_LEN}")
max_num_seqs = int("${MAX_NUM_SEQS}")
max_batch_size = int("${MAX_BATCH_SIZE}")
token_generation_buckets = parse_int_list(
    "TOKEN_GENERATION_BUCKETS",
    "${TOKEN_GENERATION_BUCKETS}",
)
token_generation_batches = parse_int_list(
    "TOKEN_GENERATION_BATCHES",
    "${TOKEN_GENERATION_BATCHES}",
)
if token_generation_buckets is None:
    token_generation_buckets = [seq_len]
    enable_bucketing = False
else:
    token_generation_buckets = sorted(token_generation_buckets)
    if token_generation_buckets[-1] > seq_len:
        raise SystemExit(
            "TOKEN_GENERATION_BUCKETS cannot contain values greater than SEQ_LEN"
        )
    enable_bucketing = True
if token_generation_batches is not None:
    token_generation_batches = sorted(token_generation_batches)
    if token_generation_batches[-1] > max_num_seqs:
        raise SystemExit(
            "TOKEN_GENERATION_BATCHES cannot contain values greater than MAX_NUM_SEQS"
        )
neuron_config = {
    "tp_degree": int("${TP_DEGREE}"),
    "batch_size": max_num_seqs,
    "ctx_batch_size": 1,
    "tkg_batch_size": max_num_seqs,
    "max_batch_size": max_batch_size,
    "kv_cache_batch_size": max_num_seqs,
    "seq_len": seq_len,
    "max_length": seq_len,
    "max_context_length": int("${CTE_BUCKET}"),
    "context_encoding_buckets": [int("${CTE_BUCKET}")],
    "token_generation_buckets": token_generation_buckets,
    "enable_bucketing": enable_bucketing,
    "logical_nc_config": int("${LNC}"),
    "torch_dtype": "bfloat16",
    "output_logits": output_logits,
    "save_sharded_checkpoint": True,
}
if enable_on_device_sampling:
    neuron_config["on_device_sampling_config"] = {
        "do_sample": False,
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 1.0,
    }
if async_mode:
    neuron_config["async_mode"] = True
if token_generation_batches is not None:
    neuron_config["token_generation_batches"] = token_generation_batches
if enable_chunked:
    neuron_config.update({
        "is_block_kv_layout": True,
        "chunked_prefill_config": {
            "max_num_seqs": max_num_seqs,
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
echo "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
echo "ENABLE_ON_DEVICE_SAMPLING=${ENABLE_ON_DEVICE_SAMPLING}"
echo "ASYNC_MODE=${ASYNC_MODE}"
echo "OUTPUT_LOGITS=${OUTPUT_LOGITS}"
echo "TOKEN_GENERATION_BUCKETS=${TOKEN_GENERATION_BUCKETS:-}"
echo "TOKEN_GENERATION_BATCHES=${TOKEN_GENERATION_BATCHES:-}"
echo "MAX_BATCH_SIZE=${MAX_BATCH_SIZE}"
echo "MAMBA_CACHE_MODE=${MAMBA_CACHE_MODE:-}"
echo "MAMBA_CACHE_DTYPE=${MAMBA_CACHE_DTYPE:-}"
echo "MAMBA_SSM_CACHE_DTYPE=${MAMBA_SSM_CACHE_DTYPE:-}"
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
  --additional-config "${ADDITIONAL_CONFIG}"
)
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
else
  VLLM_ARGS+=(--no-enable-prefix-caching)
fi
if [[ -n "${MAMBA_CACHE_MODE}" ]]; then
  VLLM_ARGS+=(--mamba-cache-mode "${MAMBA_CACHE_MODE}")
fi
if [[ -n "${MAMBA_CACHE_DTYPE}" ]]; then
  VLLM_ARGS+=(--mamba-cache-dtype "${MAMBA_CACHE_DTYPE}")
fi
if [[ -n "${MAMBA_SSM_CACHE_DTYPE}" ]]; then
  VLLM_ARGS+=(--mamba-ssm-cache-dtype "${MAMBA_SSM_CACHE_DTYPE}")
fi
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
