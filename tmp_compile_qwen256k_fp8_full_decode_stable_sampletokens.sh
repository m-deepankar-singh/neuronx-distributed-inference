#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-${SCRIPT_DIR}}
MODEL=${MODEL:-/home/ubuntu/models/Qwen3.6-27B}
ART_ROOT=${ART_ROOT:-/mnt/trainium_artifacts/qwen_artifacts}
LOGDIR=${LOGDIR:-/home/ubuntu/validation_logs/fp8_256k_decode_nki}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
LOAD_AFTER_COMPILE=${LOAD_AFTER_COMPILE:-0}
GDN_RECURRENT_CACHE_DTYPE=${GDN_RECURRENT_CACHE_DTYPE:-bfloat16}
GDN_CONV_CACHE_DTYPE=${GDN_CONV_CACHE_DTYPE:-bfloat16}
ENABLE_KV_CACHE_QUANT=${ENABLE_KV_CACHE_QUANT:-1}
DISABLE_CONTEXT_ENCODING_ARGMAX_KERNEL=${DISABLE_CONTEXT_ENCODING_ARGMAX_KERNEL:-0}
OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=${OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING:-0}
FAST_COLD_PREFILL_PROBE=${FAST_COLD_PREFILL_PROBE:-0}
QWEN36_DELTANET_SOLVE_BLOCK_SIZE=${QWEN36_DELTANET_SOLVE_BLOCK_SIZE:-}
QWEN36_DELTANET_SOLVE_SCAN_STEPS=${QWEN36_DELTANET_SOLVE_SCAN_STEPS:-}
QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K=${QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K:-}
NKI_LIBRARY_SRC=${NKI_LIBRARY_SRC:-/home/ubuntu/nki-library-2.30/src/nkilib_src}
RECURRENT_DTYPE_TAG="gdnrec${GDN_RECURRENT_CACHE_DTYPE}"
KV_CACHE_TAG="kvbf16"
KV_CACHE_FLAGS=()
if [[ "${ENABLE_KV_CACHE_QUANT}" == "1" ]]; then
  KV_CACHE_TAG="kvfp8"
  KV_CACHE_FLAGS=(--enable-kv-cache-quant)
elif [[ "${ENABLE_KV_CACHE_QUANT}" != "0" ]]; then
  echo "ERROR: ENABLE_KV_CACHE_QUANT must be 0 or 1, got ${ENABLE_KV_CACHE_QUANT}" >&2
  exit 2
fi
ARGMAX_TAG=""
ARGMAX_FLAGS=()
if [[ "${DISABLE_CONTEXT_ENCODING_ARGMAX_KERNEL}" == "1" ]]; then
  ARGMAX_TAG="_cteargmaxsafe"
  ARGMAX_FLAGS=(--disable-context-encoding-argmax-kernel)
elif [[ "${DISABLE_CONTEXT_ENCODING_ARGMAX_KERNEL}" != "0" ]]; then
  echo "ERROR: DISABLE_CONTEXT_ENCODING_ARGMAX_KERNEL must be 0 or 1, got ${DISABLE_CONTEXT_ENCODING_ARGMAX_KERNEL}" >&2
  exit 2
fi
OUTPUT_LOGITS_TAG=""
OUTPUT_LOGITS_FLAGS=()
if [[ "${OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING}" == "1" ]]; then
  OUTPUT_LOGITS_TAG="_outlogits"
  OUTPUT_LOGITS_FLAGS=(--output-logits-with-on-device-sampling)
elif [[ "${OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING}" != "0" ]]; then
  echo "ERROR: OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING must be 0 or 1, got ${OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING}" >&2
  exit 2
fi

SEQ_LEN=262144
MAX_CONTEXT_LENGTH=262144
PA_NUM_BLOCKS=1024
PROBE_TAG=""
CTE_BUCKETS=(256 512)
PREFIX_BUCKETS=(256 512 1024 2048 4096 8192 16384)
TKG_BUCKETS=(
  512 768 1024 1280
  2048 2304 4096 4352
  8192 8448 16384 16640
  24576 24832 32768 33024
  65536 65792 131072 131328
  262144
)
PAIR_ARGS=(
  256:256 512:256
  256:512 512:512
  256:1024 512:1024
  256:2048 512:2048
  256:4096 512:4096
  256:8192 512:8192
  512:16384
)
if [[ "${FAST_COLD_PREFILL_PROBE}" == "1" ]]; then
  SEQ_LEN=32768
  MAX_CONTEXT_LENGTH=32768
  PA_NUM_BLOCKS=128
  PROBE_TAG="_probe32k"
  CTE_BUCKETS=(512)
  PREFIX_BUCKETS=(256 512 1024 2048 4096 8192 16384)
  PAIR_ARGS=(
    512:256
    512:512
    512:1024
    512:2048
    512:4096
    512:8192
    512:16384
  )
  TKG_BUCKETS=(512 16384 16640 32768)
elif [[ "${FAST_COLD_PREFILL_PROBE}" != "0" ]]; then
  echo "ERROR: FAST_COLD_PREFILL_PROBE must be 0 or 1, got ${FAST_COLD_PREFILL_PROBE}" >&2
  exit 2
fi

BASE="qwen36_27b_${SEQ_LEN}_fp8_full_lmheadfp8_${KV_CACHE_TAG}_hybrid_apc_nki_decode_stable${PROBE_TAG}_${RECURRENT_DTYPE_TAG}_sampletokens${OUTPUT_LOGITS_TAG}_b256_cte$(IFS=_; echo "${CTE_BUCKETS[*]}")_pfx16k_slots64_async${ARGMAX_TAG}_${TS}"
ART="${ART_ROOT}/${BASE}"
WORK="${ART_ROOT}/_nxd_model_workdir_${SEQ_LEN}_fp8_full_lmheadfp8_${KV_CACHE_TAG}_nki_decode_stable${PROBE_TAG}_${RECURRENT_DTYPE_TAG}_sampletokens${OUTPUT_LOGITS_TAG}_b256${ARGMAX_TAG}_${TS}"
QUANT="${ART_ROOT}/_quantized/qwen36_27b_fp8_full_lmheadfp8"
LOG="${LOGDIR}/${BASE}_compile.log"
PID="${LOGDIR}/${BASE}_compile.pid"

mkdir -p "${LOGDIR}"
cd "${REPO}"
export NEURON_PLATFORM_TARGET_OVERRIDE="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---target trn2}"
export USE_NKI_DECODE=1
if [[ -n "${QWEN36_DELTANET_SOLVE_BLOCK_SIZE}" ]]; then
  export QWEN36_DELTANET_SOLVE_BLOCK_SIZE
fi
if [[ -n "${QWEN36_DELTANET_SOLVE_SCAN_STEPS}" ]]; then
  export QWEN36_DELTANET_SOLVE_SCAN_STEPS
fi
if [[ -n "${QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K}" ]]; then
  export QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K
fi

printf "%s\n" \
  "BASE=${BASE}" \
  "ARTIFACT=${ART}" \
  "WORKDIR=${WORK}" \
  "QUANTIZED_CHECKPOINTS=${QUANT}" \
  "LOG=${LOG}" \
  "PIDFILE=${PID}" \
  "REPO=${REPO}" \
  "MODEL=${MODEL}" \
  "NEURON_PLATFORM_TARGET_OVERRIDE=${NEURON_PLATFORM_TARGET_OVERRIDE}" \
  "NEURON_CC_FLAGS=${NEURON_CC_FLAGS}" \
  "NKI_LIBRARY_SRC=${NKI_LIBRARY_SRC}" \
  "USE_NKI_DECODE=${USE_NKI_DECODE}" \
  "QWEN36_DELTANET_SOLVE_BLOCK_SIZE=${QWEN36_DELTANET_SOLVE_BLOCK_SIZE}" \
  "QWEN36_DELTANET_SOLVE_SCAN_STEPS=${QWEN36_DELTANET_SOLVE_SCAN_STEPS}" \
  "QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K=${QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K}" \
  "FAST_COLD_PREFILL_PROBE=${FAST_COLD_PREFILL_PROBE}" \
  "SEQ_LEN=${SEQ_LEN}" \
  "MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH}" \
  "PA_NUM_BLOCKS=${PA_NUM_BLOCKS}" \
  "CTE_BUCKETS=${CTE_BUCKETS[*]}" \
  "PREFIX_BUCKETS=${PREFIX_BUCKETS[*]}" \
  "TOKEN_GENERATION_BUCKETS=${TKG_BUCKETS[*]}" \
  "CONTEXT_ENCODING_BUCKET_PAIRS=${PAIR_ARGS[*]}" \
  "KERNELS=decode_deltanet,standard_qkv" \
  "SAMPLING=on_device_greedy_output_logits_${OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING}" \
  "MEMORY_FLAGS=lm_head_fp8,kv_cache_${KV_CACHE_TAG},no_tkg_checkpoint_commit,gdn_recurrent_${GDN_RECURRENT_CACHE_DTYPE},gdn_conv_${GDN_CONV_CACHE_DTYPE}" \
  "LOAD_AFTER_COMPILE=${LOAD_AFTER_COMPILE}" \
  "GDN_RECURRENT_CACHE_DTYPE=${GDN_RECURRENT_CACHE_DTYPE}" \
  "GDN_CONV_CACHE_DTYPE=${GDN_CONV_CACHE_DTYPE}" \
  "ENABLE_KV_CACHE_QUANT=${ENABLE_KV_CACHE_QUANT}" \
  "DISABLE_CONTEXT_ENCODING_ARGMAX_KERNEL=${DISABLE_CONTEXT_ENCODING_ARGMAX_KERNEL}" \
  "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=${OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING}" \
  >"${LOGDIR}/${BASE}_env.txt"

(
  set -euo pipefail
  cd "${REPO}"
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
  export PYTHONPATH="${NKI_LIBRARY_SRC}:${REPO}/src:${REPO}/contrib/models/Qwen3.6-27B:${REPO}/contrib/models/Qwen3.6-27B/vllm:${PYTHONPATH:-}"
  export NEURON_PLATFORM_TARGET_OVERRIDE="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"
  export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---target trn2}"
  export USE_NKI_DECODE=1
  if [[ -n "${QWEN36_DELTANET_SOLVE_BLOCK_SIZE}" ]]; then
    export QWEN36_DELTANET_SOLVE_BLOCK_SIZE
  fi
  if [[ -n "${QWEN36_DELTANET_SOLVE_SCAN_STEPS}" ]]; then
    export QWEN36_DELTANET_SOLVE_SCAN_STEPS
  fi
  if [[ -n "${QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K}" ]]; then
    export QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K
  fi
  load_after_compile_args=()
  if [[ "${LOAD_AFTER_COMPILE}" == "1" ]]; then
    load_after_compile_args=(--load-after-compile)
  fi

  python contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py \
    --repo-root "${REPO}" \
    --model-path "${MODEL}" \
    --compiled-path "${ART}" \
    --base-compile-work-dir "${WORK}" \
    --quantized-checkpoints-path "${QUANT}" \
    --weight-dtype fp8_full \
    --quantize-lm-head \
    --seq-len "${SEQ_LEN}" \
    --max-context-length "${MAX_CONTEXT_LENGTH}" \
    --cte-buckets "${CTE_BUCKETS[@]}" \
    --prefix-buckets "${PREFIX_BUCKETS[@]}" \
    --context-encoding-bucket-pairs "${PAIR_ARGS[@]}" \
    --token-generation-buckets "${TKG_BUCKETS[@]}" \
    --block-size 256 \
    --pa-num-blocks "${PA_NUM_BLOCKS}" \
    --tp-degree 4 \
    --logical-nc-config 2 \
    --max-num-seqs 1 \
    --ctx-batch-size 1 \
    --skip-warmup \
    --async-mode \
    --enable-prefix-caching \
    --enable-hybrid-apc \
    --enable-vllm-chunked-prefill \
    --enable-deltanet-decode-nki \
    "${KV_CACHE_FLAGS[@]}" \
    --deltanet-cte-backend fused \
    --gdn-checkpoint-interval 256 \
    --max-gdn-checkpoint-slots 64 \
    --gdn-recurrent-cache-dtype "${GDN_RECURRENT_CACHE_DTYPE}" \
    --gdn-conv-cache-dtype "${GDN_CONV_CACHE_DTYPE}" \
    --hybrid-cache-mode all \
    --hybrid-apc-require-vllm-metadata \
    --hybrid-apc-enable-backed-prefix-reads \
    "${OUTPUT_LOGITS_FLAGS[@]}" \
    "${ARGMAX_FLAGS[@]}" \
    "${load_after_compile_args[@]}"
) >"${LOG}" 2>&1 &

echo "$!" >"${PID}"
echo "BASE=${BASE}"
echo "ARTIFACT=${ART}"
echo "WORKDIR=${WORK}"
echo "QUANTIZED_CHECKPOINTS=${QUANT}"
echo "LOG=${LOG}"
echo "PIDFILE=${PID}"
echo "PID=$(cat "${PID}")"
