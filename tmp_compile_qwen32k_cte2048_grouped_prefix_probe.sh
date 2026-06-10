#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z}
MODEL=${MODEL:-/home/ubuntu/models/Qwen3.6-27B}
ART_ROOT=${ART_ROOT:-/mnt/trainium_artifacts/qwen_artifacts}
LOGDIR=${LOGDIR:-/home/ubuntu/validation_logs/fp8_256k_decode_nki}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}

SEQ_LEN=32768
MAX_CONTEXT_LENGTH=32768
PA_NUM_BLOCKS=128
GDN_RECURRENT_CACHE_DTYPE=${GDN_RECURRENT_CACHE_DTYPE:-bfloat16}
GDN_CONV_CACHE_DTYPE=${GDN_CONV_CACHE_DTYPE:-bfloat16}
QWEN36_DELTANET_CHUNK_SIZE=${QWEN36_DELTANET_CHUNK_SIZE:-128}
QWEN36_DELTANET_SOLVE_BLOCK_SIZE=${QWEN36_DELTANET_SOLVE_BLOCK_SIZE:-128}
QWEN36_DELTANET_SOLVE_SCAN_STEPS=${QWEN36_DELTANET_SOLVE_SCAN_STEPS:-2}
QWEN36_QK_NORM_ROPE_NKI=${QWEN36_QK_NORM_ROPE_NKI:-0}
QWEN36_OUTPUT_GATE_NKI=${QWEN36_OUTPUT_GATE_NKI:-0}
QWEN36_QKV_GATE_PACKED=${QWEN36_QKV_GATE_PACKED:-0}
QWEN36_GATED_OUT_PROJ_NKI=${QWEN36_GATED_OUT_PROJ_NKI:-0}
DELTANET_CTE_BACKEND=${DELTANET_CTE_BACKEND:-fused}
DELTANET_CTE_BACKEND_TAG=""
if [[ "${DELTANET_CTE_BACKEND}" != "fused" ]]; then
  DELTANET_CTE_BACKEND_TAG="_${DELTANET_CTE_BACKEND}"
fi
QK_NORM_ROPE_TAG=""
if [[ "${QWEN36_QK_NORM_ROPE_NKI}" == "1" ]]; then
  QK_NORM_ROPE_TAG="_qknormrope"
fi
OUTPUT_GATE_TAG=""
if [[ "${QWEN36_OUTPUT_GATE_NKI}" == "1" ]]; then
  OUTPUT_GATE_TAG="_gatenki"
fi
QKV_GATE_PACKED_TAG=""
if [[ "${QWEN36_QKV_GATE_PACKED}" == "1" ]]; then
  QKV_GATE_PACKED_TAG="_qkvgate"
fi
GATED_OUT_PROJ_TAG=""
if [[ "${QWEN36_GATED_OUT_PROJ_NKI}" == "1" ]]; then
  GATED_OUT_PROJ_TAG="_gatedoutproj"
fi
if [[ "${QWEN36_OUTPUT_GATE_NKI}" == "1" && "${QWEN36_QKV_GATE_PACKED}" == "1" ]]; then
  echo "QWEN36_OUTPUT_GATE_NKI and QWEN36_QKV_GATE_PACKED are mutually exclusive" >&2
  exit 2
fi

CTE_BUCKETS=(2048)
PREFIX_BUCKETS=(256 512 1024 2048 4096 8192 16384)
PAIR_ARGS=(2048:0 2048:256 2048:512 2048:1024 2048:2048 2048:4096 2048:8192 2048:16384)
TKG_BUCKETS=(512 16384 16640 32768)

BASE="qwen36_27b_32k_fp8_cte2048_gpfx2_b256_pfx16k_mlpcte_qkvnki_outprojnki${QK_NORM_ROPE_TAG}${OUTPUT_GATE_TAG}${QKV_GATE_PACKED_TAG}${GATED_OUT_PROJ_TAG}${DELTANET_CTE_BACKEND_TAG}_${TS}_scan2"
ART="${ART_ROOT}/${BASE}"
WORK="${ART_ROOT}/_nxd_workdir_32k_fp8_cte2048_gpfx2_mlpcte_qkvnki_outprojnki${QK_NORM_ROPE_TAG}${OUTPUT_GATE_TAG}${QKV_GATE_PACKED_TAG}${GATED_OUT_PROJ_TAG}${DELTANET_CTE_BACKEND_TAG}_${TS}_scan2"
QUANT="${ART_ROOT}/_quantized/qwen36_27b_fp8_full_lmheadfp8"
LOG="${LOGDIR}/${BASE}_compile.log"
PID="${LOGDIR}/${BASE}_compile.pid"
ENVLOG="${LOGDIR}/${BASE}_env.txt"
NKI_LIBRARY_SRC=${NKI_LIBRARY_SRC:-/home/ubuntu/nki-library-2.30/src/nkilib_src}

mkdir -p "${LOGDIR}"
cd "${REPO}"
export NKI_LIBRARY_SRC
export NEURON_PLATFORM_TARGET_OVERRIDE="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---target trn2 --lnc 2}"
export USE_NKI_DECODE=1
export QWEN36_DELTANET_CHUNK_SIZE
export QWEN36_DELTANET_SOLVE_BLOCK_SIZE
export QWEN36_DELTANET_SOLVE_SCAN_STEPS

printf "%s\n" \
  "BASE=${BASE}" \
  "ARTIFACT=${ART}" \
  "WORKDIR=${WORK}" \
  "QUANTIZED_CHECKPOINTS=${QUANT}" \
  "LOG=${LOG}" \
  "PIDFILE=${PID}" \
  "REPO=${REPO}" \
  "MODEL=${MODEL}" \
  "NKI_LIBRARY_SRC=${NKI_LIBRARY_SRC}" \
  "NEURON_PLATFORM_TARGET_OVERRIDE=${NEURON_PLATFORM_TARGET_OVERRIDE}" \
  "NEURON_CC_FLAGS=${NEURON_CC_FLAGS}" \
  "USE_NKI_DECODE=${USE_NKI_DECODE}" \
  "QWEN36_DELTANET_CHUNK_SIZE=${QWEN36_DELTANET_CHUNK_SIZE}" \
  "QWEN36_DELTANET_SOLVE_BLOCK_SIZE=${QWEN36_DELTANET_SOLVE_BLOCK_SIZE}" \
  "QWEN36_DELTANET_SOLVE_SCAN_STEPS=${QWEN36_DELTANET_SOLVE_SCAN_STEPS}" \
  "QWEN36_QK_NORM_ROPE_NKI=${QWEN36_QK_NORM_ROPE_NKI}" \
  "QWEN36_OUTPUT_GATE_NKI=${QWEN36_OUTPUT_GATE_NKI}" \
  "QWEN36_QKV_GATE_PACKED=${QWEN36_QKV_GATE_PACKED}" \
  "QWEN36_GATED_OUT_PROJ_NKI=${QWEN36_GATED_OUT_PROJ_NKI}" \
  "DELTANET_CTE_BACKEND=${DELTANET_CTE_BACKEND}" \
  "GROUPED_PREFIX_ATTENTION_PROBE=1" \
  "EXPECTED_RUNTIME_PREFILL_CHUNK_TOKENS=2048" \
  "SEQ_LEN=${SEQ_LEN}" \
  "MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH}" \
  "PA_NUM_BLOCKS=${PA_NUM_BLOCKS}" \
  "CTE_BUCKETS=${CTE_BUCKETS[*]}" \
  "PREFIX_BUCKETS=${PREFIX_BUCKETS[*]}" \
  "TOKEN_GENERATION_BUCKETS=${TKG_BUCKETS[*]}" \
  "CONTEXT_ENCODING_BUCKET_PAIRS=${PAIR_ARGS[*]}" \
  "PREFIX_CTE_ATTENTION_BACKEND=attention_cte" \
  "KERNELS=decode_deltanet,qkv_cte_nki,mlp_cte,out_proj_cte_nki" \
  "QKV_KERNEL=cte_nki_fp8_row_scale_no_rope_fusion" \
  "OUT_PROJ_KERNEL=cte_nki_fp8_row_quantized_attention" \
  "MLP_KERNEL=cte_nki_quantized_no_fused_cte_rmsnorm" \
  "SAMPLING=on_device_greedy_output_logits_1" \
  "MEMORY_FLAGS=lm_head_fp8,kv_cache_kvfp8,no_tkg_checkpoint_commit,gdn_recurrent_${GDN_RECURRENT_CACHE_DTYPE},gdn_conv_${GDN_CONV_CACHE_DTYPE}" \
  "LOAD_AFTER_COMPILE=0" \
  "GDN_RECURRENT_CACHE_DTYPE=${GDN_RECURRENT_CACHE_DTYPE}" \
  "GDN_CONV_CACHE_DTYPE=${GDN_CONV_CACHE_DTYPE}" \
  "ENABLE_KV_CACHE_QUANT=1" \
  "ENABLE_QKV_NKI_KERNELS=1" \
  "ENABLE_MLP_CTE_NKI_KERNEL=1" \
  "ENABLE_QUANTIZED_MLP_KERNEL=1" \
  "ENABLE_OUT_PROJ_NKI_KERNEL=1" \
  "ENABLE_QWEN_QK_NORM_ROPE_NKI_KERNEL=${QWEN36_QK_NORM_ROPE_NKI}" \
  "ENABLE_QWEN_OUTPUT_GATE_NKI_KERNEL=${QWEN36_OUTPUT_GATE_NKI}" \
  "ENABLE_QWEN_QKV_GATE_PACKED_KERNEL=${QWEN36_QKV_GATE_PACKED}" \
  "ENABLE_QWEN_GATED_OUT_PROJ_NKI_KERNEL=${QWEN36_GATED_OUT_PROJ_NKI}" \
  "DISABLE_CONTEXT_ENCODING_ARGMAX_KERNEL=1" \
  "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=1" \
  >"${ENVLOG}"

(
  set -euo pipefail
  cd "${REPO}"
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
  export NKI_LIBRARY_SRC
  export PYTHONPATH="${NKI_LIBRARY_SRC}:${REPO}/src:${REPO}/contrib/models/Qwen3.6-27B:${REPO}/contrib/models/Qwen3.6-27B/vllm:${PYTHONPATH:-}"
  export NEURON_PLATFORM_TARGET_OVERRIDE="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"
  export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---target trn2 --lnc 2}"
  export USE_NKI_DECODE=1
  export QWEN36_DELTANET_CHUNK_SIZE
  export QWEN36_DELTANET_SOLVE_BLOCK_SIZE
  export QWEN36_DELTANET_SOLVE_SCAN_STEPS
  export QWEN36_QKV_GATE_PACKED
  export QWEN36_GATED_OUT_PROJ_NKI

  EXTRA_COMPILE_ARGS=()
  if [[ "${QWEN36_QK_NORM_ROPE_NKI}" == "1" ]]; then
    EXTRA_COMPILE_ARGS+=(--enable-qwen-qk-norm-rope-nki-kernel)
  fi
  if [[ "${QWEN36_OUTPUT_GATE_NKI}" == "1" ]]; then
    EXTRA_COMPILE_ARGS+=(--enable-qwen-output-gate-nki-kernel)
  fi
  if [[ "${QWEN36_QKV_GATE_PACKED}" == "1" ]]; then
    EXTRA_COMPILE_ARGS+=(--enable-qwen-qkv-gate-packed-kernel)
  fi
  if [[ "${QWEN36_GATED_OUT_PROJ_NKI}" == "1" ]]; then
    EXTRA_COMPILE_ARGS+=(--enable-qwen-gated-o-proj-nki-kernel)
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
    --enable-kv-cache-quant \
    --enable-qkv-nki-kernels \
    --enable-mlp-cte-nki-kernel \
    --enable-quantized-mlp-kernel \
    --enable-out-proj-nki-kernel \
    --deltanet-cte-backend "${DELTANET_CTE_BACKEND}" \
    --gdn-checkpoint-interval 256 \
    --max-gdn-checkpoint-slots 64 \
    --gdn-recurrent-cache-dtype "${GDN_RECURRENT_CACHE_DTYPE}" \
    --gdn-conv-cache-dtype "${GDN_CONV_CACHE_DTYPE}" \
    --hybrid-cache-mode all \
    --hybrid-apc-require-vllm-metadata \
    --hybrid-apc-enable-backed-prefix-reads \
    --output-logits-with-on-device-sampling \
    --disable-context-encoding-argmax-kernel \
    --prefix-cte-attention-backend attention_cte \
    "${EXTRA_COMPILE_ARGS[@]}"
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
