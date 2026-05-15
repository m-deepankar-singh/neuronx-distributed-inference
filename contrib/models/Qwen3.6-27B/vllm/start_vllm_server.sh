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
ENABLE_PREFIX_CACHING="0"
ENABLE_HYBRID_APC="0"
MAMBA_CACHE_MODE=""
MAMBA_CACHE_DTYPE=""
MAMBA_SSM_CACHE_DTYPE=""
BLOCK_SIZE=""
GDN_CHECKPOINT_INTERVAL="256"
GDN_RECURRENT_CACHE_DTYPE="float32"
GDN_CONV_CACHE_DTYPE="bfloat16"
HYBRID_GDN_RECURRENT_CACHE_DTYPE=""
HYBRID_GDN_CONV_CACHE_DTYPE=""
HYBRID_CACHE_MODE="all"
HYBRID_CACHE_PREFIX_BOUNDARY_ONLY="1"
HYBRID_CACHE_VALIDATE_EXACT="0"
NUM_GPU_BLOCKS_OVERRIDE=""

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
    --enable-prefix-caching) ENABLE_PREFIX_CACHING="1"; shift ;;
    --disable-prefix-caching|--no-enable-prefix-caching) ENABLE_PREFIX_CACHING="0"; shift ;;
    --enable-hybrid-apc) ENABLE_HYBRID_APC="1"; shift ;;
    --mamba-cache-mode) MAMBA_CACHE_MODE="$2"; shift 2 ;;
    --mamba-cache-dtype) MAMBA_CACHE_DTYPE="$2"; shift 2 ;;
    --mamba-ssm-cache-dtype) MAMBA_SSM_CACHE_DTYPE="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --gdn-checkpoint-interval) GDN_CHECKPOINT_INTERVAL="$2"; shift 2 ;;
    --gdn-recurrent-cache-dtype) GDN_RECURRENT_CACHE_DTYPE="$2"; shift 2 ;;
    --gdn-conv-cache-dtype) GDN_CONV_CACHE_DTYPE="$2"; shift 2 ;;
    --hybrid-gdn-recurrent-cache-dtype) HYBRID_GDN_RECURRENT_CACHE_DTYPE="$2"; shift 2 ;;
    --hybrid-gdn-conv-cache-dtype) HYBRID_GDN_CONV_CACHE_DTYPE="$2"; shift 2 ;;
    --hybrid-cache-mode) HYBRID_CACHE_MODE="$2"; shift 2 ;;
    --hybrid-cache-prefix-boundary-only|--hybrid-cache-block-boundary-only) HYBRID_CACHE_PREFIX_BOUNDARY_ONLY="1"; shift ;;
    --no-hybrid-cache-prefix-boundary-only|--no-hybrid-cache-block-boundary-only) HYBRID_CACHE_PREFIX_BOUNDARY_ONLY="0"; shift ;;
    --hybrid-cache-validate-exact) HYBRID_CACHE_VALIDATE_EXACT="1"; shift ;;
    --num-gpu-blocks-override) NUM_GPU_BLOCKS_OVERRIDE="$2"; shift 2 ;;
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
  BLOCK_SIZE="128"
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  export DISABLE_NEURON_CUSTOM_SCHEDULER="1"
fi
if [[ -z "${HYBRID_GDN_RECURRENT_CACHE_DTYPE}" ]]; then
  HYBRID_GDN_RECURRENT_CACHE_DTYPE="${GDN_RECURRENT_CACHE_DTYPE}"
fi
if [[ -z "${HYBRID_GDN_CONV_CACHE_DTYPE}" ]]; then
  HYBRID_GDN_CONV_CACHE_DTYPE="${GDN_CONV_CACHE_DTYPE}"
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" || "${ENABLE_HYBRID_APC}" == "1" ]]; then
  ENABLE_PREFIX_CACHING="1"
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" && -z "${MAMBA_CACHE_MODE}" ]]; then
  MAMBA_CACHE_MODE="all"
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" && -z "${MAMBA_SSM_CACHE_DTYPE}" ]]; then
  MAMBA_SSM_CACHE_DTYPE="${HYBRID_GDN_RECURRENT_CACHE_DTYPE}"
fi

ADDITIONAL_CONFIG="$(
  python3 - <<PY
import json
enable_chunked = "${ENABLE_CHUNKED_PREFILL}" == "1"
enable_prefix_caching = "${ENABLE_PREFIX_CACHING}" == "1"
enable_hybrid_apc = "${ENABLE_HYBRID_APC}" == "1"
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
    "pa_block_size": int("${BLOCK_SIZE}"),
}
if enable_prefix_caching or enable_hybrid_apc or enable_chunked:
    neuron_config["is_block_kv_layout"] = True
if enable_prefix_caching or enable_hybrid_apc:
    neuron_config["is_prefix_caching"] = True
if enable_chunked:
    neuron_config.update({
        "chunked_prefill_config": {
            "max_num_seqs": int("${MAX_NUM_SEQS}"),
            "tkg_model_enabled": True,
            "kernel_q_tile_size": 128,
            "kernel_kv_tile_size": 1024,
        },
    })
print(json.dumps({
    "max_prompt_length": int("${CTE_BUCKET}"),
    "use_hybrid_apc_manager": enable_hybrid_apc,
    "gdn_checkpoint_interval": int("${GDN_CHECKPOINT_INTERVAL}"),
    "gdn_recurrent_cache_dtype": "${HYBRID_GDN_RECURRENT_CACHE_DTYPE}",
    "gdn_conv_cache_dtype": "${HYBRID_GDN_CONV_CACHE_DTYPE}",
    "hybrid_recurrent_cache_dtype": "${HYBRID_GDN_RECURRENT_CACHE_DTYPE}",
    "hybrid_conv_cache_dtype": "${HYBRID_GDN_CONV_CACHE_DTYPE}",
    "hybrid_cache_mode": "${HYBRID_CACHE_MODE}",
    "hybrid_cache_prefix_boundary_only": "${HYBRID_CACHE_PREFIX_BOUNDARY_ONLY}" == "1",
    "hybrid_cache_block_boundary_only": "${HYBRID_CACHE_PREFIX_BOUNDARY_ONLY}" == "1",
    "hybrid_cache_validate_exact": "${HYBRID_CACHE_VALIDATE_EXACT}" == "1",
    "override_neuron_config": neuron_config,
}))
PY
)"

echo "Starting vLLM for Qwen3.6-27B"
echo "MODEL_PATH=${MODEL_PATH}"
echo "NEURON_COMPILED_ARTIFACTS=${NEURON_COMPILED_ARTIFACTS:-}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
echo "ENABLE_HYBRID_APC=${ENABLE_HYBRID_APC}"
echo "MAMBA_CACHE_MODE=${MAMBA_CACHE_MODE:-}"
echo "MAMBA_CACHE_DTYPE=${MAMBA_CACHE_DTYPE:-}"
echo "MAMBA_SSM_CACHE_DTYPE=${MAMBA_SSM_CACHE_DTYPE:-}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "GDN_CHECKPOINT_INTERVAL=${GDN_CHECKPOINT_INTERVAL}"
echo "HYBRID_GDN_RECURRENT_CACHE_DTYPE=${HYBRID_GDN_RECURRENT_CACHE_DTYPE}"
echo "HYBRID_GDN_CONV_CACHE_DTYPE=${HYBRID_GDN_CONV_CACHE_DTYPE}"
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
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
  VLLM_ARGS+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" || "${ENABLE_HYBRID_APC}" == "1" || "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  VLLM_ARGS+=(--block-size "${BLOCK_SIZE}")
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  VLLM_ARGS+=(
    --enable-chunked-prefill
    --max-num-batched-tokens "${CTE_BUCKET}"
  )
else
  VLLM_ARGS+=(--no-enable-chunked-prefill)
fi

exec python "${SCRIPT_DIR}/serve_qwen36.py" "${VLLM_ARGS[@]}"
