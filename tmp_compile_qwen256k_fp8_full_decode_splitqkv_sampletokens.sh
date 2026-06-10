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
RECURRENT_DTYPE_TAG="gdnrec${GDN_RECURRENT_CACHE_DTYPE}"

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

BASE="qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_splitqkv_${RECURRENT_DTYPE_TAG}_sampletokens_b256_cte256_512_pfx16k_slots64_async_${TS}"
ART="${ART_ROOT}/${BASE}"
WORK="${ART_ROOT}/_nxd_model_workdir_256k_fp8_full_lmheadfp8_kvfp8_nki_decode_splitqkv_${RECURRENT_DTYPE_TAG}_sampletokens_b256_${TS}"
QUANT="${ART_ROOT}/_quantized/qwen36_27b_fp8_full_lmheadfp8"
LOG="${LOGDIR}/${BASE}_compile.log"
PID="${LOGDIR}/${BASE}_compile.pid"

mkdir -p "${LOGDIR}"
cd "${REPO}"
export NEURON_PLATFORM_TARGET_OVERRIDE="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---target trn2}"
export USE_NKI_DECODE=1

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
  "USE_NKI_DECODE=${USE_NKI_DECODE}" \
  "KERNELS=decode_deltanet,split_qkv_tkg" \
  "SAMPLING=on_device_greedy_no_host_logits" \
  "MEMORY_FLAGS=lm_head_fp8,kv_cache_fp8,no_tkg_checkpoint_commit,gdn_recurrent_${GDN_RECURRENT_CACHE_DTYPE},gdn_conv_${GDN_CONV_CACHE_DTYPE}" \
  "LOAD_AFTER_COMPILE=${LOAD_AFTER_COMPILE}" \
  "GDN_RECURRENT_CACHE_DTYPE=${GDN_RECURRENT_CACHE_DTYPE}" \
  "GDN_CONV_CACHE_DTYPE=${GDN_CONV_CACHE_DTYPE}" \
  >"${LOGDIR}/${BASE}_env.txt"

(
  set -euo pipefail
  cd "${REPO}"
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
  export PYTHONPATH="${REPO}/src:${REPO}/contrib/models/Qwen3.6-27B:${REPO}/contrib/models/Qwen3.6-27B/vllm:${PYTHONPATH:-}"
  export NEURON_PLATFORM_TARGET_OVERRIDE="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"
  export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---target trn2}"
  export USE_NKI_DECODE=1
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
    --seq-len 262144 \
    --max-context-length 262144 \
    --cte-buckets 256 512 \
    --prefix-buckets 256 512 1024 2048 4096 8192 16384 \
    --context-encoding-bucket-pairs "${PAIR_ARGS[@]}" \
    --token-generation-buckets "${TKG_BUCKETS[@]}" \
    --block-size 256 \
    --pa-num-blocks 1024 \
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
    --enable-split-qkv-tkg-nki-kernel \
    --enable-kv-cache-quant \
    --deltanet-cte-backend fused \
    --gdn-checkpoint-interval 256 \
    --max-gdn-checkpoint-slots 64 \
    --gdn-recurrent-cache-dtype "${GDN_RECURRENT_CACHE_DTYPE}" \
    --gdn-conv-cache-dtype "${GDN_CONV_CACHE_DTYPE}" \
    --hybrid-cache-mode all \
    --hybrid-apc-require-vllm-metadata \
    --hybrid-apc-enable-backed-prefix-reads \
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
