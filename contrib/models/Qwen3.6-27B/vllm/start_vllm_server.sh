#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=""
COMPILED_ARTIFACTS=""
MAX_MODEL_LEN="512"
SEQ_LEN="512"
CTE_BUCKET="512"
CTE_BUCKETS=""
CTE_BUCKET_PROFILE="single"
TP_DEGREE="4"
LNC="2"
MAX_NUM_SEQS="1"
CTX_BATCH_SIZE="1"
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
MAX_GDN_CHECKPOINT_SLOTS="8"
GDN_RECURRENT_CACHE_DTYPE="float32"
GDN_CONV_CACHE_DTYPE="bfloat16"
HYBRID_GDN_RECURRENT_CACHE_DTYPE=""
HYBRID_GDN_CONV_CACHE_DTYPE=""
HYBRID_CACHE_MODE="all"
HYBRID_CACHE_PREFIX_BOUNDARY_ONLY="1"
HYBRID_CACHE_VALIDATE_EXACT="0"
HYBRID_APC_REQUIRE_VLLM_METADATA="1"
HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS="0"
HYBRID_APC_ENABLE_BACKED_PREFIX_READS="0"
NUM_GPU_BLOCKS_OVERRIDE=""
KERNEL_Q_TILE_SIZE="128"
KERNEL_KV_TILE_SIZE="1024"
TEXT_ONLY_CTE="1"
COMPACT_CTE_ATTENTION_MASK="1"
COLD_ZERO_CONV_FAST_PATH="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --compiled-artifacts) COMPILED_ARTIFACTS="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --cte-bucket) CTE_BUCKET="$2"; shift 2 ;;
    --cte-buckets) CTE_BUCKETS="$2"; shift 2 ;;
    --cte-bucket-profile) CTE_BUCKET_PROFILE="$2"; shift 2 ;;
    --tensor-parallel-size) TP_DEGREE="$2"; shift 2 ;;
    --logical-nc-config) LNC="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --ctx-batch-size) CTX_BATCH_SIZE="$2"; shift 2 ;;
    --enable-vllm-chunked-prefill) ENABLE_CHUNKED_PREFILL="1"; shift ;;
    --enable-prefix-caching) ENABLE_PREFIX_CACHING="1"; shift ;;
    --disable-prefix-caching|--no-enable-prefix-caching) ENABLE_PREFIX_CACHING="0"; shift ;;
    --enable-hybrid-apc) ENABLE_HYBRID_APC="1"; shift ;;
    --mamba-cache-mode) MAMBA_CACHE_MODE="$2"; shift 2 ;;
    --mamba-cache-dtype) MAMBA_CACHE_DTYPE="$2"; shift 2 ;;
    --mamba-ssm-cache-dtype) MAMBA_SSM_CACHE_DTYPE="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --gdn-checkpoint-interval) GDN_CHECKPOINT_INTERVAL="$2"; shift 2 ;;
    --max-gdn-checkpoint-slots) MAX_GDN_CHECKPOINT_SLOTS="$2"; shift 2 ;;
    --gdn-recurrent-cache-dtype) GDN_RECURRENT_CACHE_DTYPE="$2"; shift 2 ;;
    --gdn-conv-cache-dtype) GDN_CONV_CACHE_DTYPE="$2"; shift 2 ;;
    --hybrid-gdn-recurrent-cache-dtype) HYBRID_GDN_RECURRENT_CACHE_DTYPE="$2"; shift 2 ;;
    --hybrid-gdn-conv-cache-dtype) HYBRID_GDN_CONV_CACHE_DTYPE="$2"; shift 2 ;;
    --hybrid-cache-mode) HYBRID_CACHE_MODE="$2"; shift 2 ;;
    --hybrid-cache-prefix-boundary-only|--hybrid-cache-block-boundary-only) HYBRID_CACHE_PREFIX_BOUNDARY_ONLY="1"; shift ;;
    --no-hybrid-cache-prefix-boundary-only|--no-hybrid-cache-block-boundary-only) HYBRID_CACHE_PREFIX_BOUNDARY_ONLY="0"; shift ;;
    --hybrid-cache-validate-exact) HYBRID_CACHE_VALIDATE_EXACT="1"; shift ;;
    --hybrid-apc-require-vllm-metadata) HYBRID_APC_REQUIRE_VLLM_METADATA="1"; shift ;;
    --no-hybrid-apc-require-vllm-metadata|--allow-hybrid-apc-local-hash-fallback) HYBRID_APC_REQUIRE_VLLM_METADATA="0"; shift ;;
    --hybrid-apc-disable-unbacked-prefix-reads) HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS="1"; shift ;;
    --no-hybrid-apc-disable-unbacked-prefix-reads) HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS="0"; shift ;;
    --hybrid-apc-enable-backed-prefix-reads) HYBRID_APC_ENABLE_BACKED_PREFIX_READS="1"; shift ;;
    --no-hybrid-apc-enable-backed-prefix-reads) HYBRID_APC_ENABLE_BACKED_PREFIX_READS="0"; shift ;;
    --num-gpu-blocks-override) NUM_GPU_BLOCKS_OVERRIDE="$2"; shift 2 ;;
    --kernel-q-tile-size) KERNEL_Q_TILE_SIZE="$2"; shift 2 ;;
    --kernel-kv-tile-size) KERNEL_KV_TILE_SIZE="$2"; shift 2 ;;
    --text-only-cte) TEXT_ONLY_CTE="1"; shift ;;
    --no-text-only-cte|--multimodal-cte) TEXT_ONLY_CTE="0"; shift ;;
    --compact-cte-attention-mask) COMPACT_CTE_ATTENTION_MASK="1"; shift ;;
    --no-compact-cte-attention-mask) COMPACT_CTE_ATTENTION_MASK="0"; shift ;;
    --cold-zero-conv-fast-path) COLD_ZERO_CONV_FAST_PATH="1"; shift ;;
    --no-cold-zero-conv-fast-path) COLD_ZERO_CONV_FAST_PATH="0"; shift ;;
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
  export XLA_HANDLE_SPECIAL_SCALAR="${XLA_HANDLE_SPECIAL_SCALAR:-1}"
  export UNSAFE_FP8FNCAST="${UNSAFE_FP8FNCAST:-1}"
fi
if [[ -z "${BLOCK_SIZE}" ]]; then
  BLOCK_SIZE="128"
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  export DISABLE_NEURON_CUSTOM_SCHEDULER="1"
fi
if [[ "${ENABLE_HYBRID_APC}" == "1" ]]; then
  export QWEN36_HYBRID_APC_INSTALL_PATCH="1"
fi
if [[ "${HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS}" == "1" ]]; then
  export QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS="1"
fi
if [[ "${HYBRID_APC_ENABLE_BACKED_PREFIX_READS}" == "1" ]]; then
  export QWEN36_HYBRID_APC_ENABLE_BACKED_PREFIX_READS="1"
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
if [[ "${ENABLE_PREFIX_CACHING}" == "1" || "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  if [[ -z "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
    NUM_GPU_BLOCKS_OVERRIDE=$(( ((SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE) * MAX_NUM_SEQS ))
  fi
fi
if [[ "${ENABLE_HYBRID_APC}" == "1" ]]; then
  if [[ "${HYBRID_CACHE_MODE}" != "all" ]]; then
    echo "ERROR: --enable-hybrid-apc requires --hybrid-cache-mode all" >&2
    exit 2
  fi
  if [[ "${GDN_CHECKPOINT_INTERVAL}" != "${BLOCK_SIZE}" ]]; then
    echo "ERROR: --enable-hybrid-apc v0 requires --gdn-checkpoint-interval to equal --block-size" >&2
    exit 2
  fi
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" && -z "${MAMBA_CACHE_MODE}" ]]; then
  MAMBA_CACHE_MODE="all"
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" && -z "${MAMBA_SSM_CACHE_DTYPE}" ]]; then
  MAMBA_SSM_CACHE_DTYPE="${HYBRID_GDN_RECURRENT_CACHE_DTYPE}"
fi

CTE_BUCKETS_JSON="$(
  python3 - <<PY
import json

profiles = {
    "short": [128, 256, 512, 1024],
    "general": [256, 512, 1024, 2048],
    "long": [4096, 8192, 16384, 32768],
    "262k": [256],
}
profile = "${CTE_BUCKET_PROFILE}"
if profile != "single":
    if profile not in profiles:
        raise SystemExit(f"unknown --cte-bucket-profile: {profile}")
    buckets = profiles[profile]
else:
    raw = "${CTE_BUCKETS}".replace(",", " ").split()
    buckets = [int(x) for x in raw] if raw else [int("${CTE_BUCKET}")]
buckets = sorted(set(buckets))
if not buckets:
    raise SystemExit("at least one CTE bucket is required")
for bucket in buckets:
    if bucket <= 0:
        raise SystemExit(f"CTE buckets must be positive, got {bucket}")
    if bucket % 128 != 0:
        raise SystemExit(
            f"CTE bucket {bucket} is not 128-aligned; DeltaNet CTE uses 128-token chunks"
        )
if buckets[-1] > int("${SEQ_LEN}"):
    raise SystemExit(
        f"largest CTE bucket {buckets[-1]} exceeds --seq-len ${SEQ_LEN}"
    )
print(json.dumps(buckets))
PY
)"
MAX_CTE_BUCKET="$(
  python3 - <<PY
import json
print(json.loads('${CTE_BUCKETS_JSON}')[-1])
PY
)"

ADDITIONAL_CONFIG="$(
  python3 - <<PY
import json
enable_chunked = "${ENABLE_CHUNKED_PREFILL}" == "1"
enable_prefix_caching = "${ENABLE_PREFIX_CACHING}" == "1"
enable_hybrid_apc = "${ENABLE_HYBRID_APC}" == "1"
cte_buckets = json.loads('${CTE_BUCKETS_JSON}')
max_cte_bucket = cte_buckets[-1]
num_gpu_blocks_override = "${NUM_GPU_BLOCKS_OVERRIDE}"
pa_num_blocks = (
    int(num_gpu_blocks_override)
    if num_gpu_blocks_override
    else max(
        1,
        ((int("${SEQ_LEN}") + int("${BLOCK_SIZE}") - 1) // int("${BLOCK_SIZE}"))
        * int("${MAX_NUM_SEQS}"),
    )
)
neuron_config = {
    "tp_degree": int("${TP_DEGREE}"),
    "batch_size": int("${MAX_NUM_SEQS}"),
    "ctx_batch_size": int("${CTX_BATCH_SIZE}"),
    "tkg_batch_size": int("${MAX_NUM_SEQS}"),
    "seq_len": int("${SEQ_LEN}"),
    "max_length": int("${SEQ_LEN}"),
    "max_context_length": max_cte_bucket,
    "context_encoding_buckets": cte_buckets,
    "token_generation_buckets": [int("${SEQ_LEN}")],
    "enable_bucketing": len(cte_buckets) > 1,
    "logical_nc_config": int("${LNC}"),
    "torch_dtype": "bfloat16",
    "save_sharded_checkpoint": True,
    "pa_block_size": int("${BLOCK_SIZE}"),
    "pa_num_blocks": pa_num_blocks,
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
            "kernel_q_tile_size": int("${KERNEL_Q_TILE_SIZE}"),
            "kernel_kv_tile_size": int("${KERNEL_KV_TILE_SIZE}"),
        },
    })
print(json.dumps({
    "max_prompt_length": max_cte_bucket,
    "use_hybrid_apc_manager": enable_hybrid_apc,
    "use_text_only_cte_inputs": "${TEXT_ONLY_CTE}" == "1",
    "use_compact_cte_attention_mask": "${COMPACT_CTE_ATTENTION_MASK}" == "1",
    "use_cold_zero_conv_fast_path": "${COLD_ZERO_CONV_FAST_PATH}" == "1",
    "gdn_checkpoint_interval": int("${GDN_CHECKPOINT_INTERVAL}"),
    "max_gdn_checkpoint_slots": int("${MAX_GDN_CHECKPOINT_SLOTS}"),
    "gdn_recurrent_cache_dtype": "${HYBRID_GDN_RECURRENT_CACHE_DTYPE}",
    "gdn_conv_cache_dtype": "${HYBRID_GDN_CONV_CACHE_DTYPE}",
    "hybrid_recurrent_cache_dtype": "${HYBRID_GDN_RECURRENT_CACHE_DTYPE}",
    "hybrid_conv_cache_dtype": "${HYBRID_GDN_CONV_CACHE_DTYPE}",
    "hybrid_cache_mode": "${HYBRID_CACHE_MODE}",
    "hybrid_cache_prefix_boundary_only": "${HYBRID_CACHE_PREFIX_BOUNDARY_ONLY}" == "1",
    "hybrid_cache_block_boundary_only": "${HYBRID_CACHE_PREFIX_BOUNDARY_ONLY}" == "1",
    "hybrid_cache_validate_exact": "${HYBRID_CACHE_VALIDATE_EXACT}" == "1",
    "hybrid_apc_require_vllm_metadata": enable_hybrid_apc and "${HYBRID_APC_REQUIRE_VLLM_METADATA}" == "1",
    "hybrid_apc_allow_local_hash_fallback": not (enable_hybrid_apc and "${HYBRID_APC_REQUIRE_VLLM_METADATA}" == "1"),
    "hybrid_apc_require_attention_block_refs": enable_hybrid_apc and "${HYBRID_APC_REQUIRE_VLLM_METADATA}" == "1",
    "hybrid_apc_disable_unbacked_prefix_reads": enable_hybrid_apc and "${HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS}" == "1",
    "hybrid_apc_enable_backed_prefix_reads": enable_hybrid_apc and "${HYBRID_APC_ENABLE_BACKED_PREFIX_READS}" == "1",
    "use_qwen_hybrid_chunked_prefill": enable_chunked,
    "use_qwen_hybrid_chunked_prefill_nki": enable_chunked,
    "override_neuron_config": neuron_config,
}))
PY
)"

echo "Starting vLLM for Qwen3.6-27B"
echo "MODEL_PATH=${MODEL_PATH}"
echo "NEURON_COMPILED_ARTIFACTS=${NEURON_COMPILED_ARTIFACTS:-}"
echo "XLA_HANDLE_SPECIAL_SCALAR=${XLA_HANDLE_SPECIAL_SCALAR:-}"
echo "UNSAFE_FP8FNCAST=${UNSAFE_FP8FNCAST:-}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
echo "ENABLE_HYBRID_APC=${ENABLE_HYBRID_APC}"
echo "MAMBA_CACHE_MODE=${MAMBA_CACHE_MODE:-}"
echo "MAMBA_CACHE_DTYPE=${MAMBA_CACHE_DTYPE:-}"
echo "MAMBA_SSM_CACHE_DTYPE=${MAMBA_SSM_CACHE_DTYPE:-}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "CTE_BUCKETS=${CTE_BUCKETS_JSON}"
echo "CTX_BATCH_SIZE=${CTX_BATCH_SIZE}"
echo "KERNEL_Q_TILE_SIZE=${KERNEL_Q_TILE_SIZE}"
echo "KERNEL_KV_TILE_SIZE=${KERNEL_KV_TILE_SIZE}"
echo "TEXT_ONLY_CTE=${TEXT_ONLY_CTE}"
echo "COMPACT_CTE_ATTENTION_MASK=${COMPACT_CTE_ATTENTION_MASK}"
echo "COLD_ZERO_CONV_FAST_PATH=${COLD_ZERO_CONV_FAST_PATH}"
echo "GDN_CHECKPOINT_INTERVAL=${GDN_CHECKPOINT_INTERVAL}"
echo "MAX_GDN_CHECKPOINT_SLOTS=${MAX_GDN_CHECKPOINT_SLOTS}"
echo "HYBRID_GDN_RECURRENT_CACHE_DTYPE=${HYBRID_GDN_RECURRENT_CACHE_DTYPE}"
echo "HYBRID_GDN_CONV_CACHE_DTYPE=${HYBRID_GDN_CONV_CACHE_DTYPE}"
echo "HYBRID_APC_REQUIRE_VLLM_METADATA=${HYBRID_APC_REQUIRE_VLLM_METADATA}"
echo "HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS=${HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS}"
echo "HYBRID_APC_ENABLE_BACKED_PREFIX_READS=${HYBRID_APC_ENABLE_BACKED_PREFIX_READS}"
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
    --max-num-batched-tokens "${MAX_CTE_BUCKET}"
  )
else
  VLLM_ARGS+=(--no-enable-chunked-prefill)
fi

exec python "${SCRIPT_DIR}/serve_qwen36.py" "${VLLM_ARGS[@]}"
