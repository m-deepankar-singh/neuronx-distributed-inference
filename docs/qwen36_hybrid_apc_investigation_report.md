# Qwen3.6 Hybrid APC FP8 Investigation Report

Date: 2026-05-17
Branch: experimental

## Goal

Make Qwen3.6-27B Hybrid APC production-ready enough for real vLLM/Neuron
validation: cold/warm exactness, non-dummy generated tokens, and sane FP8
runtime behavior.

## Relevant Changes

- Added a real-token validation gate to
  `validation_scripts/qwen36_hybrid_apc_validation.py`.
- Added JSON output support for the Hybrid APC validation report.
- Added FP8 runtime defaults for Qwen3.6 FP8 compile and vLLM entrypoints:
  `XLA_HANDLE_SPECIAL_SCALAR=1` and `UNSAFE_FP8FNCAST=1`.
- Changed Qwen3.6 FP8 compile for host-side sampling to emit logits
  (`output_logits=True`) instead of placeholder token ids.
- Fixed Qwen3.6 trace output alias accounting when `output_logits=True` is used
  without on-device sampling.
- Adjusted FP8 compile PA block planning to compile one extra physical block for
  the vLLM Neuron null block.
- Added a focused async execution regression test for Hybrid APC debug trace
  metadata.
- Added unit coverage for the FP8 compile config, Qwen alias accounting, and
  real-token validation gate.

## Working Evidence

Local checks passed:

- `python3 -m pytest contrib/models/Qwen3.6-27B/test/unit/test_qwen36_model_aliases.py contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py contrib/models/Qwen3.6-27B/test/unit/test_hybrid_apc_validation.py contrib/models/Qwen3.6-27B/test/unit/test_hybrid_apc_manager.py`
  - Result: `37 passed`
- `PYTHONPATH=src python3 -m pytest test/unit/modules/test_async_execution.py`
  - Result: `23 passed`
- Remote Trainium focused tests:
  - `test_qwen36_model_aliases.py`, `test_qwen36_compile_fp8_config.py`,
    `test_hybrid_apc_validation.py`
  - Result: `10 passed`, then `test_qwen36_compile_fp8_config.py` rerun after
    PA block fix with `4 passed`

Earlier Trainium Hybrid APC 2K exactness passed for cold/warm equality, but all
generated token ids were `0`, which is why the real-token gate was added.

## Trainium Findings

Remote instance:

- Host: `ubuntu@16.26.90.15`
- Repo: `/home/ubuntu/inferentia-gdn`
- Validation logs: `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/`
- Scripts: `/home/ubuntu/qwen_artifacts/`

Important runs:

- CPU/host sampling with `output_logits=True` compiled after the alias fix, but
  vLLM CPU sampling received all-NaN logits:
  - `loader.raw_output.logits finite=0/248320 nan=248320`
  - Generated tokens were all `0`.
- Adding `XLA_HANDLE_SPECIAL_SCALAR=1` and `UNSAFE_FP8FNCAST=1` did not fix the
  host-logits NaN path.
- On-device sampling matches the older working vLLM APC branch more closely, but
  first runs hit Neuron runtime out-of-bounds in `token_generation_model`.
- The vLLM Neuron loader increments `pa_num_blocks` by one for a null block.
  Compiling exactly 8 physical blocks while runtime used 9 caused mismatch risk.
- After patching compile to emit 9 physical blocks for user-intended 8 blocks,
  the PA9 artifact compiled and loaded, but generation still hit Neuron runtime
  OOB in `token_generation_model`.

Latest PA9 artifact run:

- Artifact path:
  `/dev/shm/qwen36_27b_2048_fp8_mlp_only_hybrid_apc_ondevice_pa9_specialscalar_20260517T084122Z`
- Compile log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/hybrid_apc_ondevice_pa9_specialscalar_2k_compile.log`
- Validation log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/hybrid_apc_ondevice_pa9_specialscalar_2k_validation.log`
- Result:
  - Compile succeeded.
  - vLLM loaded precompiled artifacts.
  - Hardware sampling was enabled.
  - Runtime failed with `NRT_EXEC_OOB` in token generation.

## External Reference

AWS Neuron documentation and the existing NxDI `inference_demo.py` both use
special FP8 environment handling for `f8e4m3`:

- `XLA_HANDLE_SPECIAL_SCALAR=1`
- `UNSAFE_FP8FNCAST=1`

Those are now set by the Qwen3.6 FP8 compile and precompiled-artifact serving
entrypoints.

## Open Questions

- Why does host-side `output_logits=True` return all-NaN logits even when FP8
  special scalar handling is set?
- Why does on-device sampling OOB in token generation even after the compile
  artifact and vLLM runtime both use 9 physical PA blocks?
- Whether the OOB is caused by token-generation block-table shape/indexing,
  sampler input layout, or another vLLM Neuron/NxDI contract mismatch.
- Whether the older `contrib/qwen36-27b-vllm-apc-pr` branch has a serving/runtime
  detail outside the Qwen compile path that avoids this OOB.

## Files To Review First

- `contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py`
- `contrib/models/Qwen3.6-27B/src/modeling_qwen35.py`
- `validation_scripts/qwen36_hybrid_apc_validation.py`
- `contrib/models/Qwen3.6-27B/vllm/run_offline_inference.py`
- `contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh`
- `test/unit/modules/test_async_execution.py`
- `contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py`
- `contrib/models/Qwen3.6-27B/test/unit/test_qwen36_model_aliases.py`
- `contrib/models/Qwen3.6-27B/test/unit/test_hybrid_apc_validation.py`
