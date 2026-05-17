#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
orchestrator="${repo_root}/validation_scripts/qwen36_cold_prefill_strict_final.py"

required_env=(
  QWEN36_MODEL
  QWEN36_ARTIFACT_2K
  QWEN36_ARTIFACT_8K
  QWEN36_ARTIFACT_32K
  QWEN36_ARTIFACT_128K
  QWEN36_ARTIFACT_262K
  QWEN36_OUT
)

for name in "${required_env[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required env: ${name}" >&2
    exit 2
  fi
done

python_bin="${PYTHON:-python3}"
repetitions="${QWEN36_REPETITIONS:-3}"
strict_min_samples="${QWEN36_STRICT_MIN_SAMPLES:-3}"
expected_branch="${QWEN36_EXPECTED_BRANCH:-qwen36-cold-prefill-perf}"
output_prefix="${QWEN36_OUTPUT_PREFIX:-qwen36_cold_prefill}"

common_args=(
  --model-path "${QWEN36_MODEL}"
  --artifacts-2k "${QWEN36_ARTIFACT_2K}"
  --artifacts-8k "${QWEN36_ARTIFACT_8K}"
  --artifacts-32k "${QWEN36_ARTIFACT_32K}"
  --artifacts-128k "${QWEN36_ARTIFACT_128K}"
  --artifacts-262k "${QWEN36_ARTIFACT_262K}"
  --output-dir "${QWEN36_OUT}"
  --output-prefix "${output_prefix}"
  --python "${python_bin}"
  --expected-branch "${expected_branch}"
)

if [[ -n "${QWEN36_GDN_STATE_DIFF_JSON:-}" ]]; then
  common_args+=(--gdn-state-diff-json "${QWEN36_GDN_STATE_DIFF_JSON}")
fi

if [[ -n "${QWEN36_NUM_GPU_BLOCKS_OVERRIDE_BY_LEN:-}" ]]; then
  # shellcheck disable=SC2206
  block_overrides=(${QWEN36_NUM_GPU_BLOCKS_OVERRIDE_BY_LEN})
  common_args+=(--num-gpu-blocks-override-by-len "${block_overrides[@]}")
elif [[ -n "${QWEN36_NUM_GPU_BLOCKS_OVERRIDE:-}" ]]; then
  common_args+=(--num-gpu-blocks-override "${QWEN36_NUM_GPU_BLOCKS_OVERRIDE}")
else
  common_args+=(
    --num-gpu-blocks-override-by-len
    2048=16
    8192=64
    32768=256
    131072=1024
    262144=2048
  )
fi

if [[ "${QWEN36_FAIL_FAST:-0}" == "1" ]]; then
  common_args+=(--fail-fast)
fi

echo "== strict-final preflight =="
"${python_bin}" "${orchestrator}" \
  "${common_args[@]}" \
  --preflight

echo "== strict-final dry-run =="
"${python_bin}" "${orchestrator}" \
  "${common_args[@]}" \
  --dry-run

echo "== strict-final collection =="
"${python_bin}" "${orchestrator}" \
  "${common_args[@]}" \
  --repetitions "${repetitions}" \
  --strict-min-samples "${strict_min_samples}"

echo "== evidence check =="
"${python_bin}" "${orchestrator}" \
  --output-dir "${QWEN36_OUT}" \
  --output-prefix "${output_prefix}" \
  --evidence-check
