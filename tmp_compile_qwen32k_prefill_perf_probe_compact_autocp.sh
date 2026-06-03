#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-${SCRIPT_DIR}}
MODEL=${MODEL:-/home/ubuntu/models/Qwen3.6-27B}
ART_ROOT=${ART_ROOT:-/mnt/trainium_artifacts/qwen_artifacts}
LOGDIR=${LOGDIR:-/home/ubuntu/validation_logs/fp8_256k_decode_nki}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}

SEQ_LEN=32768
MAX_CONTEXT_LENGTH=32768
PA_NUM_BLOCKS=128
GDN_RECURRENT_CACHE_DTYPE=${GDN_RECURRENT_CACHE_DTYPE:-bfloat16}
GDN_CONV_CACHE_DTYPE=${GDN_CONV_CACHE_DTYPE:-bfloat16}
PREFIX_CTE_ATTENTION_BACKEND=${PREFIX_CTE_ATTENTION_BACKEND:-segmented_cte}
PREFIX_CTE_ATTENTION_SEGMENT_SIZE=${PREFIX_CTE_ATTENTION_SEGMENT_SIZE:-512}
QWEN36_DELTANET_SOLVE_BLOCK_SIZE=${QWEN36_DELTANET_SOLVE_BLOCK_SIZE:-128}
QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K=${QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K:-0}
QWEN36_DELTANET_AUTOCP_CTE=${QWEN36_DELTANET_AUTOCP_CTE:-1}
QWEN36_DELTANET_COMPACT_AUTOCP_CTE=${QWEN36_DELTANET_COMPACT_AUTOCP_CTE:-1}
QWEN36_DELTANET_AUTOCP_CP_CHUNKS=${QWEN36_DELTANET_AUTOCP_CP_CHUNKS:-4}
QWEN36_DELTANET_AUTOCP_LNC=${QWEN36_DELTANET_AUTOCP_LNC:-2}
QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE=${QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE:-2}
NKI_LIBRARY_SRC=${NKI_LIBRARY_SRC:-/home/ubuntu/nki-library-2.30/src/nkilib_src}

# Reduced 32k shape set for the 16k cold-prefill question.
CTE_BUCKETS=(3072)
PREFIX_BUCKETS=(4096 8192 16384)
PAIR_ARGS=(3072:4096 3072:8192 3072:16384)
TKG_BUCKETS=(512 16384 16640 32768)

AUTOCP_TAG="compactautocp_cp${QWEN36_DELTANET_AUTOCP_CP_CHUNKS}_lnc${QWEN36_DELTANET_AUTOCP_LNC}_rgrp${QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE}"
BASE="qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_prefillperf_probe32k_gdnrec${GDN_RECURRENT_CACHE_DTYPE}_sampletokens_outlogits_b256_cte3072_pfx16k_${PREFIX_CTE_ATTENTION_BACKEND}${PREFIX_CTE_ATTENTION_SEGMENT_SIZE}_solve${QWEN36_DELTANET_SOLVE_BLOCK_SIZE}_apfx${QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K}_${AUTOCP_TAG}_slots64_async_cteargmaxsafe_${TS}"
ART="${ART_ROOT}/${BASE}"
WORK="${ART_ROOT}/_nxd_model_workdir_32768_fp8_full_lmheadfp8_kvfp8_nki_prefillperf_probe32k_gdnrec${GDN_RECURRENT_CACHE_DTYPE}_sampletokens_outlogits_b256_cte3072_${PREFIX_CTE_ATTENTION_BACKEND}${PREFIX_CTE_ATTENTION_SEGMENT_SIZE}_solve${QWEN36_DELTANET_SOLVE_BLOCK_SIZE}_apfx${QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K}_${AUTOCP_TAG}_${TS}"
QUANT="${ART_ROOT}/_quantized/qwen36_27b_fp8_full_lmheadfp8"
LOG="${LOGDIR}/${BASE}_compile.log"
PID="${LOGDIR}/${BASE}_compile.pid"
ENVLOG="${LOGDIR}/${BASE}_env.txt"

mkdir -p "${LOGDIR}"
cd "${REPO}"
export NEURON_PLATFORM_TARGET_OVERRIDE="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---target trn2 --lnc 2}"
export USE_NKI_DECODE=1
export QWEN36_DELTANET_SOLVE_BLOCK_SIZE
export QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K
export QWEN36_DELTANET_AUTOCP_CTE
export QWEN36_DELTANET_COMPACT_AUTOCP_CTE
export QWEN36_DELTANET_AUTOCP_CP_CHUNKS
export QWEN36_DELTANET_AUTOCP_LNC
export QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE

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
  "QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K=${QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K}" \
  "QWEN36_DELTANET_AUTOCP_CTE=${QWEN36_DELTANET_AUTOCP_CTE}" \
  "QWEN36_DELTANET_COMPACT_AUTOCP_CTE=${QWEN36_DELTANET_COMPACT_AUTOCP_CTE}" \
  "QWEN36_DELTANET_AUTOCP_CP_CHUNKS=${QWEN36_DELTANET_AUTOCP_CP_CHUNKS}" \
  "QWEN36_DELTANET_AUTOCP_LNC=${QWEN36_DELTANET_AUTOCP_LNC}" \
  "QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE=${QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE}" \
  "SEQ_LEN=${SEQ_LEN}" \
  "MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH}" \
  "PA_NUM_BLOCKS=${PA_NUM_BLOCKS}" \
  "CTE_BUCKETS=${CTE_BUCKETS[*]}" \
  "PREFIX_BUCKETS=${PREFIX_BUCKETS[*]}" \
  "TOKEN_GENERATION_BUCKETS=${TKG_BUCKETS[*]}" \
  "CONTEXT_ENCODING_BUCKET_PAIRS=${PAIR_ARGS[*]}" \
  "PREFIX_CTE_ATTENTION_BACKEND=${PREFIX_CTE_ATTENTION_BACKEND}" \
  "PREFIX_CTE_ATTENTION_SEGMENT_SIZE=${PREFIX_CTE_ATTENTION_SEGMENT_SIZE}" \
  "MEMORY_FLAGS=lm_head_fp8,kv_cache_fp8,no_tkg_checkpoint_commit,gdn_recurrent_${GDN_RECURRENT_CACHE_DTYPE},gdn_conv_${GDN_CONV_CACHE_DTYPE}" \
  >"${ENVLOG}"

(
  set -euo pipefail
  cd "${REPO}"
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
  export PYTHONPATH="${NKI_LIBRARY_SRC}:${REPO}/src:${REPO}/contrib/models/Qwen3.6-27B:${REPO}/contrib/models/Qwen3.6-27B/vllm:${PYTHONPATH:-}"
  export NEURON_PLATFORM_TARGET_OVERRIDE="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"
  export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---target trn2 --lnc 2}"
  export USE_NKI_DECODE=1
  export QWEN36_DELTANET_SOLVE_BLOCK_SIZE
  export QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K
  export QWEN36_DELTANET_AUTOCP_CTE
  export QWEN36_DELTANET_COMPACT_AUTOCP_CTE
  export QWEN36_DELTANET_AUTOCP_CP_CHUNKS
  export QWEN36_DELTANET_AUTOCP_LNC
  export QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE

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
    --enable-kv-cache-quant \
    --deltanet-cte-backend fused \
    --gdn-checkpoint-interval 256 \
    --max-gdn-checkpoint-slots 64 \
    --gdn-recurrent-cache-dtype "${GDN_RECURRENT_CACHE_DTYPE}" \
    --gdn-conv-cache-dtype "${GDN_CONV_CACHE_DTYPE}" \
    --hybrid-cache-mode all \
    --hybrid-apc-require-vllm-metadata \
    --hybrid-apc-enable-backed-prefix-reads \
    --output-logits-with-on-device-sampling \
    --disable-context-encoding-argmax-kernel \
    --prefix-cte-attention-backend "${PREFIX_CTE_ATTENTION_BACKEND}" \
    --prefix-cte-attention-segment-size "${PREFIX_CTE_ATTENTION_SEGMENT_SIZE}"
) >"${LOG}" 2>&1 &

echo "$!" >"${PID}"
echo "BASE=${BASE}"
echo "ARTIFACT=${ART}"
echo "WORKDIR=${WORK}"
echo "QUANTIZED_CHECKPOINTS=${QUANT}"
echo "LOG=${LOG}"
echo "ENVLOG=${ENVLOG}"
echo "PIDFILE=${PID}"
echo "PID=$(cat "${PID}")"
