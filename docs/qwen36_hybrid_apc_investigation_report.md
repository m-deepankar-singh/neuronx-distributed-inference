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
- Normalized flattened prefix-cache `slot_mapping` tensors in the Qwen custom
  runtime path before CTE/TKG batch chunking, and sliced flattened
  `slot_mapping` in Hybrid APC warm-suffix request prep.

## 2026-05-17 Follow-Up Instrumentation

Added repo-side checks for the next debugging pass:

- `QWEN36_LOGIT_STAGE_DEBUG=1` now prints finite/NaN/Inf summaries in
  `modeling_qwen35.py` at:
  - before final norm;
  - after final norm / final hidden selection;
  - selected hidden state before `lm_head`;
  - `lm_head.weight`;
  - raw `lm_head` output before float cast;
  - after `lm_head`;
  - after `mask_padded_logits`;
  - before logits are returned.
- `QWEN36_TKG_INPUT_DEBUG=1` now prints the decode-side
  `token_generation_model` inputs immediately before the call:
  `input_ids`, `position_ids`, `attention_mask`, `slot_mapping`, `block_table`,
  `num_queries`, `computed_context_lens`, `pa_num_blocks`, `block_size`,
  `seq_len`, and `max_model_len`.
- The compile harness supports a BF16 real-token control with
  `--weight-dtype bf16_control`, preserving the same Hybrid APC/2K shape knobs
  but compiling with `quantized=False`.
- The validation harness supports `--skip-fp8-env` so BF16 artifacts can run
  without `XLA_HANDLE_SPECIAL_SCALAR` / `UNSAFE_FP8FNCAST`.
- `QWEN36_TKG_LEGACY_ARGS=1` is an experimental compile-time and runtime mode
  for testing the older 24-tensor prefix-cache ABI. Both CTE and TKG use the
  prefix-cache/mRoPE/vision contract and omit Hybrid APC restore/commit tensors.
  This matches the currently compiled legacy artifacts; Neuron pruned the extra
  CTE Hybrid APC metadata inputs from the serialized trace.
- `NXDI_RAW_OUTPUT_DEBUG=1` now prints raw runtime output slot summaries before
  `CausalLMOutputWithPast` construction. The latest Hybrid APC BF16 host-logits
  validation showed exactly one raw output slot and that slot was already all
  NaN, so the NaNs were not hidden in a different output slot.
- `QWEN36_DISABLE_HYBRID_GDN_RESTORE=1`,
  `QWEN36_DISABLE_HYBRID_GDN_COMMIT=1`, and the combined
  `QWEN36_DISABLE_HYBRID_GDN_RESTORE_COMMIT=1` are no-recompile isolation
  switches. They preserve the traced argument positions but force the
  restore/commit masks to zero. Commit disabling also skips CPU metadata commit
  so an unwritten checkpoint slot is not marked reusable.

Minimal BF16 host-logits control to run on Trainium:

```bash
python3 contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py \
  --weight-dtype bf16_control \
  --model-path "$MODEL_PATH" \
  --compiled-path /dev/shm/qwen36_27b_2048_bf16_hybrid_apc_host_logits \
  --seq-len 2048 \
  --cte-buckets 256,512 \
  --enable-prefix-caching \
  --enable-hybrid-apc \
  --disable-on-device-sampling \
  --pa-num-blocks 8 \
  --load-after-compile

QWEN36_LOGIT_STAGE_DEBUG=1 \
python3 validation_scripts/qwen36_hybrid_apc_validation.py exactness \
  --model-path "$MODEL_PATH" \
  --compiled-artifacts /dev/shm/qwen36_27b_2048_bf16_hybrid_apc_host_logits \
  --max-model-len 2048 \
  --seq-len 2048 \
  --cte-buckets 256,512 \
  --enable-vllm-chunked-prefill \
  --require-real-tokens \
  --skip-fp8-env
```

PA null-block sweep to run only at 2K:

```bash
# user-intended 8 -> compile physical 9
--pa-num-blocks 8

# user-intended 9 -> compile physical 10
--pa-num-blocks 9

# user-intended 10 -> compile physical 11
--pa-num-blocks 10
```

Legacy TKG contract experiment:

```bash
QWEN36_TKG_LEGACY_ARGS=1 \
python3 contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py \
  --model-path "$MODEL_PATH" \
  --compiled-path /dev/shm/qwen36_27b_2048_fp8_hybrid_apc_ondevice_legacy_tkg \
  --quantized-checkpoints-path /dev/shm/qwen36_27b_2048_fp8_mlp_only_ckpt \
  --seq-len 2048 \
  --cte-buckets 256,512 \
  --enable-prefix-caching \
  --enable-hybrid-apc \
  --pa-num-blocks 8 \
  --load-after-compile

QWEN36_TKG_LEGACY_ARGS=1 \
QWEN36_TKG_INPUT_DEBUG=1 \
python3 validation_scripts/qwen36_hybrid_apc_validation.py exactness \
  --model-path "$MODEL_PATH" \
  --compiled-artifacts /dev/shm/qwen36_27b_2048_fp8_hybrid_apc_ondevice_legacy_tkg \
  --max-model-len 2048 \
  --seq-len 2048 \
  --cte-buckets 256,512 \
  --enable-vllm-chunked-prefill \
  --require-real-tokens
```

## Older APC PR Branch Comparison

Compared current `experimental` against
`contrib/qwen36-27b-vllm-apc-pr` for the token-generation contract:

- Older branch pads wrapper inputs to 24 positional args:
  base inputs, 14 empties, then mRoPE/vision tensors.
- Current branch pads wrapper inputs to 29 positional args by appending five
  Hybrid APC restore/commit tensors after mRoPE/vision.
- Older branch does not pass `slot_mapping`, `block_table`, `num_queries`, or
  `computed_context_lens` into `token_generation_model`; current branch passes
  those four tensors before six empty tensor placeholders.
- Current branch computes `computed_context_lens` and `num_queries` as
  `(batch, 1)` int32 matrices and pads CTE chunks, while older branch has no
  equivalent TKG prefix-cache tensors.
- Current vLLM runners force Hybrid APC prefix settings:
  `enable_prefix_caching=True`, `mamba_cache_mode="all"`,
  `mamba_ssm_cache_dtype=<GDN recurrent dtype>`, and
  `num_gpu_blocks_override` when prefix/APC is enabled. The older runner only
  forwards mamba cache settings when explicitly supplied.

Most suspicious contract risk: a positional-input mismatch in the expanded TKG
call. One shifted tensor would put block-table or prefix-length metadata in the
wrong Neuron input slot and can plausibly produce the observed token-generation
OOB.

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

Remote instances:

- Prior host: `ubuntu@16.26.90.15`
- Current host: `ubuntu@16.50.102.110`
- Current repo: `/home/ubuntu/inferentia-gdn-experimental-test`
- Current branch: `experimental`
- Current weights:
  `/home/ubuntu/models/Qwen3.6-27B`
  and `/opt/dlami/nvme/models/Qwen3.6-27B`
- Current validation logs:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/`
- Current artifacts:
  `/home/ubuntu/qwen_artifacts/`

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
- Minimal BF16 host-logits control with prefix caching, chunked prefill, and
  Hybrid APC disabled compiled, loaded, and produced finite real logits:
  - Artifact:
    `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_host_logits_no_prefix_353306f`
  - Compile log:
    `/home/ubuntu/validation_logs/host_logits_controls/bf16_host_logits_no_prefix_353306f_compile.log`
  - Smoke log:
    `/home/ubuntu/validation_logs/host_logits_controls/bf16_host_logits_no_prefix_353306f_smoke.log`
  - Raw output summary repeated on every step:
    `raw_output[0] shape=(1, 1, 248320) dtype=torch.float32 finite=248320/248320 nan=0`
  - Generated token ids:
    `[271, 248068, 271, 248069]`
  - Conclusion: base Qwen3.6 BF16 host logits and CPU sampling are healthy
    without prefix/Hybrid APC.
- Hybrid APC BF16 host-logits validation with raw-slot debug still produced
  all-NaN logits:
  - Artifact:
    `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_host_logits_decay_clamp_980b918`
  - Raw-slot validation log:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/legacy_tkg_bf16_host_logits_decay_clamp_980b918_rawslots_96e15f7_validation.log`
  - Raw output summary:
    `raw_output[0] shape=(1, 1, 248320) dtype=torch.float32 finite=0/248320 nan=248320`
  - TKG metadata remained in range with `arg_mode=prefix24_legacy`,
    `input_shape=(1, 1)`, `num_queries=[1]`, and `pa_num_blocks=9`.
  - Conclusion: the remaining all-NaN host-logits failure is tied to the
    prefix/Hybrid APC path, not FP8 and not raw-output slot selection.
- Restore/commit disabled isolation on commit `55ae0ca` kept the same compiled
  BF16 Hybrid APC artifact and forced all Hybrid GDN restore/commit masks to
  zero:
  - Log:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/restore_commit_disabled_55ae0ca_validation.log`
  - JSON:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/restore_commit_disabled_55ae0ca_validation.json`
  - CTE debug confirmed the switch worked:
    `restore_mask=[0]` and `commit_mask=[0]`.
  - Raw logits still stayed all NaN:
    `raw_output[0] shape=(1, 1, 248320) dtype=torch.float32 finite=0/248320 nan=248320`
  - The same log exposed the next concrete contract bug: CTE was called with
    hundreds of active tokens but only one slot mapping entry, for example
    `input_shape=(1, 463)` with `slot_shape=(1,)`.
  - Conclusion: restore/commit state is not the first NaN source for this
    artifact. The current leading suspect is malformed prefix-cache CTE
    `slot_mapping` before cache update.
- Slot-mapping fix on commit `5911e3d` corrected the runtime CTE slot contract
  for short prompts:
  - Hybrid APC single-prompt smoke log:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/slotfix_5911e3d_single_smoke.log`
  - CTE now enters with `slot_shape=(1, 16)`, pads to
    `padded_slot_shape=(1, 256)`, and the first raw logits are finite:
    `finite=248320/248320 nan=0`.
  - The first generated token was real: `TOKENS [271, 0, 0, 0]`.
  - TKG still returned all-NaN logits on the next decode step, so the
    slot-mapping patch fixed one CTE contract bug but not the full generation
    path.
- Longer BF16 controls show the remaining NaN is not specific to Hybrid APC:
  - Hybrid APC validation-style prompt after slot fix:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/slotfix_5911e3d_validation_prompt_single.log`
    - CTE slot mapping was correct: `slot_shape=(1, 463)` and padded to
      `padded_slot_shape=(1, 512)`.
    - Raw logits were still all NaN.
  - No-prefix BF16 artifact with the same validation-style prompt:
    `/home/ubuntu/validation_logs/host_logits_controls/bf16_no_prefix_validation_prompt_5911e3d_smoke.log`
    - Raw logits were also all NaN.
  - Conclusion: the current highest-priority blocker is long-prompt BF16 CTE
    numerical behavior. Hybrid APC should not be judged until a no-prefix
    BF16 long-prompt control produces finite logits.
- Boundary sweeps on commit `6f575ef` isolated the no-prefix BF16 failure:
  - Existing fused-CTE BF16 artifact:
    `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_host_logits_no_prefix_353306f`
  - Validation-style repeated prompt:
    `/home/ubuntu/validation_logs/host_logits_controls/bf16_no_prefix_length_sweep_boundary_6f575ef.log`
    - Finite through 99 tokens.
    - All-NaN beginning at 106 tokens.
  - Plain repeated `hello` prompt:
    `/home/ubuntu/validation_logs/host_logits_controls/bf16_no_prefix_plain_hello_sweep_6f575ef.log`
    - Finite at 97 tokens.
    - All-NaN beginning at 105 tokens.
  - Conclusion: the failure is not prompt-content specific. It appears around
    the first 128-token fused DeltaNet CTE chunk boundary.
- A no-prefix BF16 artifact compiled with legacy per-chunk NKI CTE fixed the
  CTE NaNs without changing APC:
  - Compile env: `USE_NKI_FUSED=0 USE_NKI_CHUNKED=1`
  - Artifact:
    `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_host_logits_no_prefix_nki_chunked_6f575ef`
  - Compile log:
    `/home/ubuntu/validation_logs/host_logits_controls/bf16_no_prefix_nki_chunked_6f575ef_compile.log`
    - `COMPILE_DONE`
    - `LOAD_AFTER_COMPILE_OK`
  - Boundary sweep:
    `/home/ubuntu/validation_logs/host_logits_controls/bf16_no_prefix_nki_chunked_boundary_6f575ef.log`
    - Finite through 127 tokens.
  - Long validation-style sweep:
    `/home/ubuntu/validation_logs/host_logits_controls/bf16_no_prefix_nki_chunked_long_6f575ef.log`
    - Finite through 463 tokens.
  - Conclusion: the immediate workaround is to compile Qwen3.6 CTE with
    `USE_NKI_FUSED=0 USE_NKI_CHUNKED=1`. The current fused DeltaNet CTE NKI
    kernel is the leading root cause for host-logits all-NaN CTE outputs.
- Hybrid APC BF16 host-logits artifact compiled with per-chunk NKI CTE:
  - Compile env: `USE_NKI_FUSED=0 USE_NKI_CHUNKED=1`
  - Artifact:
    `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_host_logits_nki_chunked_4434edf`
  - Compile log:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_4434edf_compile.log`
    - `COMPILE_DONE`
    - `LOAD_AFTER_COMPILE_OK`
  - CTE-only exactness gate on `4434edf`:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_4434edf_cte_only.json`
    - `full_prefix_exact=true`
    - `partial_prefix_exact=true`
    - `real_generated_tokens_passed=true`
  - First short decode gate on `4434edf` showed finite real tokens but failed
    exactness because warm suffix CTE received all-padding slot mappings:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_4434edf_decode4.log`
    - Warm suffix CTE had `slot_minmax=-1:-1`.
    - TKG started from missing suffix attention KV and diverged from cold
      decode after the second generated token.
- Commit `b59e3a2` fixed warm-suffix attention KV writes by synthesizing suffix
  `slot_mapping` from `block_table`, `restore_len`, and `block_size` when vLLM
  provides only `-1` padding slots for the replayed suffix:
  - Changed:
    `contrib/models/Qwen3.6-27B/src/hybrid_apc.py`
  - Unit test:
    `test_prefill_plan_synthesizes_padding_suffix_slots_from_block_table`
  - Local and remote focused tests:
    `python3 -m pytest contrib/models/Qwen3.6-27B/test/unit/test_hybrid_apc_manager.py`
    - `29 passed`
  - Reused the already compiled per-chunk Hybrid APC artifact; no recompile
    was needed because this changes runtime input preparation only.
  - Decode-4 exactness after the fix:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_b59e3a2_decode4.json`
    - `full_prefix_exact=true`
    - `partial_prefix_exact=true`
    - `real_generated_tokens_passed=true`
    - Warm suffix CTE slot mappings became positive physical slots, for
      example `slot_minmax=768:974` instead of `-1:-1`.
  - Decode-32 exactness after the fix:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_b59e3a2_decode32.json`
    - `real_generated_tokens_passed=true`
    - `full_prefix_exact=false`
    - `partial_prefix_exact=false`
    - Cold and warm outputs match for the early tokens but drift later in
      decode. This is now the leading remaining BF16 host-logits correctness
      issue; it is no longer a NaN/OOB or dummy-token failure.
  - Runtime-enabling `--enable-vllm-chunked-prefill` against the non-chunked
    compiled artifact is not a valid workaround:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_b59e3a2_chunked_runtime_decode4.json`
    - Real tokens were produced, but warm exactness failed immediately.

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

Legacy TKG arg experiment:

- Branch/artifact code under test:
  - Local/remote commit: `d0cdfc2 Add legacy Qwen TKG args experiment`
  - Remote clean worktree: `/home/ubuntu/inferentia-gdn-experimental-test`
- Remote focused unit tests passed in the vLLM/Neuron environment:
  - `test_qwen36_model_aliases.py`
  - `test_qwen36_compile_fp8_config.py`
  - `test_hybrid_apc_validation.py`
  - Result: `13 passed`
- FP8 legacy TKG compile attempt:
  - Log:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/legacy_tkg_2k_compile.log`
  - Result:
    - Legacy TKG trace/HLO/NEFF compilation reached the checkpoint-sharding
      stage.
    - Sharding failed because the supplied checkpoint path contained
      already-presharded Neuron runtime files
      (`tp*_sharded_checkpoint.safetensors`) instead of HF-style source
      checkpoint files (`model.safetensors`, `model.safetensors.index.json`, or
      `pytorch_model.bin`).
    - Conclusion: those existing `weights/` directories can be loaded as part
      of their original compiled artifact, but they are not valid
      `--quantized-checkpoints-path` inputs for a fresh FP8 compile.
- BF16 legacy TKG control compile:
  - Log:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/legacy_tkg_bf16_2k_compile.log`
  - Result:
    - TKG priority compile passed.
    - CTE HLO compilation passed.
    - Weight sharding completed.
    - `LOAD_AFTER_COMPILE_OK` was reached in-process.
  - Caveat:
    - The `/dev/shm/...` compiled artifact path was gone after the SSH process
      exited, so a later separate validation process could not load it.
    - A one-shot compile-then-validate wrapper was launched to keep validation
      in the same process lifetime:
      `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/legacy_tkg_bf16_inline_2k_compile_validate.log`
    - The inline run compiled and started validation, but failed before decode
      in context encoding:
      `RuntimeError: forward() expected at most 25 argument(s) but received 30 argument(s)`.
      The compiled CTE graph accepted 24 tensor arguments, while the runtime
      still passed the expanded 29-tensor/30-argument CTE input list.
    - Conclusion: the attempted legacy mode proved the failure class is a
      compiled-trace/runtime positional signature mismatch, but the patch was
      too broad or applied at the wrong wrapper boundary. CTE and TKG need
      independent explicit arg contracts; legacy TKG mode must not make the
      CTE trace/runtime disagree.

Follow-up ABI patch:

- Local patch after `5396ef3` makes the Qwen wrapper choose argument contracts
  by model tag instead of active-token length.
- With `QWEN36_TKG_LEGACY_ARGS=1`:
  - CTE and TKG trace/runtime use a 24-tensor prefix-cache contract: it keeps
    `slot_mapping`, `block_table`, `num_queries`, and
    `computed_context_lens`, but omits the five Hybrid APC restore/commit
    tensors.
  - Hard arg-count checks fail early if compile/runtime drift again.
- Remote BF16 compile retry on `ba65161` confirmed the original CTE mismatch
  is fixed: CTE HLO generation completed. It then failed during TKG tracing
  when the legacy path blanked prefix-cache metadata, because
  `BlockKVCacheManager` requires `active_block_table.shape`. The patch was
  corrected so legacy TKG remains 24 args but retains prefix-cache metadata.
- Remote BF16 compile on `fd47bd1` reached `COMPILE_DONE` and
  `LOAD_AFTER_COMPILE_OK`, but validation failed before decode because the
  serialized CTE trace exposed 24 tensor inputs while runtime still sent 29.
  The latest local patch makes legacy mode stage-consistent with that artifact,
  so the same artifact can be retested without recompiling.
- Remote BF16 on-device compile on `5a08328` reached `COMPILE_DONE` and
  `LOAD_AFTER_COMPILE_OK` with the stage-consistent 24-tensor ABI:
  `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_ondevice_legacy_tkg_5a08328`.
  Validation progressed past CTE and failed on the first TKG decode with
  `NRT_EXEC_OOB`.
  `QWEN36_TKG_INPUT_DEBUG=1` showed:
  `input_ids=[2143289344]`,
  `position_minmax=463:463`,
  `slot_minmax=719:719`,
  `block_minmax=1:2`,
  `computed_context_lens=[463]`,
  `pa_num_blocks=9`,
  `block_size=256`.
  Interpretation: the block table and slot mapping were in range for PA9, but
  the TKG token id was garbage/out-of-vocab before the embedding gather. The
  remaining on-device failure is now most likely the sampler/token handoff
  contract, not physical PA capacity.
- A BF16 host-logits control was launched on the same commit with
  `--disable-on-device-sampling`:
  `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_host_logits_legacy_tkg_5a08328`.
  Its compile log is
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/legacy_tkg_bf16_host_logits_2k_5a08328_compile.log`.
- That BF16 host-logits artifact reached `COMPILE_DONE` and
  `LOAD_AFTER_COMPILE_OK`. It produced four BF16 sharded checkpoint files and
  loaded/warmed successfully from the precompiled artifact.
- BF16 host-logits validation log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/legacy_tkg_bf16_host_logits_2k_5a08328_validation.log`.
  The run did not write the requested JSON report because the engine crashed
  before validation completed.
- Host-side sampling avoided the on-device garbage token/OOB failure. TKG input
  debug showed repeated valid token id `0` instead of `2143289344`, with
  in-range position, slot, and block metadata. Example first decode:
  `input_values=[0]`,
  `position_minmax=463:463`,
  `slot_minmax=719:719`,
  `block_minmax=1:2`,
  `computed_context_lens=[463]`,
  `pa_num_blocks=9`,
  `block_size=256`.
- However, the CPU sampler selected dummy token `0` repeatedly. Because the
  validation gate requires real generated tokens, this means the host-logits
  output path is still not producing a usable real-token distribution.
- The same host-logits validation then crashed on the warm/multi-request path in
  TKG prefix-cache input padding:
  `ValueError: Input len tensor([[207]], dtype=torch.int32) exceeds largest bucket (2048) for token_generation_model`.
  The stack is in
  `src/neuronx_distributed_inference/models/model_wrapper.py`
  `_pad_prefix_caching_inputs` -> `get_target_2d_bucket_for_prefix_caching`.
  This is a separate bucket/shape contract issue from the earlier physical PA
  block-capacity hypothesis.
- Patch `1e41faf` added explicit CTE/TKG arg builders, contract debug, a TKG
  token-id guard, and generic TKG prefix-cache padding protection so decode
  `num_queries` is derived from `input_ids.shape[-1]`, not from a bad
  context-length value.
- Patch `f283e93` fixed warm-prefix suffix classification: any multi-token Qwen
  request is routed to CTE even when `position_ids` starts at a nonzero restored
  prefix boundary. TKG remains a strict one-token decode path.
- Remote unit verification on `f283e93`:
  - Qwen focused unit suites:
    `test_qwen36_model_aliases.py`,
    `test_qwen36_compile_fp8_config.py`,
    `test_hybrid_apc_validation.py`
    - Result: `21 passed`.
  - Prefix-cache bucket suite:
    `test/unit/models/test_prefix_caching_bucket_selection.py`
    with `--import-mode=importlib`
    - Result: `28 passed`.
- BF16 host-logits validation on `f283e93` reused the existing
  `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_host_logits_legacy_tkg_5a08328`
  artifact and completed far enough to write JSON:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/legacy_tkg_bf16_host_logits_2k_f283e93_validation.json`.
  The previous warm TKG padding crash is gone. TKG debug now shows
  `input_shape=(1, 1)`, `num_queries=[1]`, and valid context metadata through
  cold and warm decode. The validation still fails because every generated
  token is dummy token `0`:
  `real_generated_tokens_passed=false`.
- BF16 on-device validation on `f283e93` reused the existing
  `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_ondevice_legacy_tkg_5a08328`
  artifact. The new guard fails before Neuron embedding/gather execution with:
  `Qwen3.6 TKG input_ids contains out-of-vocab token id 2143289344; vocab_size=248320`.
  This confirms the prior `NRT_EXEC_OOB` was downstream of a corrupt sampled
  token handoff, not a PA block-count issue.
- Current exact problem:
  - The stage-consistent 24-tensor legacy ABI fix compiles and loads.
  - Warm prefix-cache CTE/TKG length classification is fixed for the host-logits
    path.
  - A fresh BF16 host-logits artifact built on `980b918` with the fused
    DeltaNet CTE log-decay clamp reached `COMPILE_DONE` and
    `LOAD_AFTER_COMPILE_OK`, but still returned all-NaN logits at the NxDI
    output boundary. Every debug summary reported
    `finite=0/248320 nan=248320`.
  - On-device BF16 decode still fails because the sampler/token handoff feeds a
    corrupt out-of-vocab token id into TKG.
  - BF16 host-side sampling avoids that corrupt token and no longer crashes on
    the second/warm request, but NaN logits make CPU sampling collapse to dummy
    token `0`, so real generation is not proven.
- Fresh BF16 decay-clamp validation on `980b918`:
  - Compile artifact:
    `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_host_logits_decay_clamp_980b918`
  - Compile log:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/legacy_tkg_bf16_host_logits_decay_clamp_980b918_compile.log`
  - Validation log:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/legacy_tkg_bf16_host_logits_decay_clamp_980b918_validation.log`
  - Validation JSON:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/legacy_tkg_bf16_host_logits_decay_clamp_980b918_validation.json`
  - Result:
    - Compile succeeded and loaded after compile.
    - TKG debug stayed in the legacy 24-arg path with valid one-token decode:
      `input_shape=(1, 1)`, `num_queries=[1]`, in-range slots, and
      `pa_num_blocks=9`.
    - Host logits remained all NaN:
      `[nxdi_output_debug] name=logits shape=(1, 1, 248320) dtype=torch.float32 finite=0/248320 nan=248320`.
    - The real-token gate failed because cold/warm/full/partial outputs were
      all token `0`.
  - Interpretation:
    - The NaNs are not explained by FP8 MLP quantization, because the artifact
      is BF16/no-FP8 and still returns all-NaN host logits.
    - The fused DeltaNet log-decay clamp alone is insufficient. The remaining
      fault is at or before the traced model's `logits` output slot for the
      BF16 host-logits path, not only in vLLM CPU sampling.
- Raw-output slot validation on `96e15f7`:
  - Added env-gated wrapper debug:
    `NXDI_RAW_OUTPUT_DEBUG=1 NXDI_RAW_OUTPUT_DEBUG_LIMIT=6`.
  - Validation log:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/legacy_tkg_bf16_host_logits_decay_clamp_980b918_rawslots_96e15f7_validation.log`
  - Validation JSON:
    `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/legacy_tkg_bf16_host_logits_decay_clamp_980b918_rawslots_96e15f7_validation.json`
  - Result:
    - No recompile was needed; this reused the `980b918` BF16 artifact.
    - NxDI returned exactly one raw output slot:
      `[nxdi_raw_output_debug] count=1 limit=1`.
    - That sole raw output was the logits-shaped tensor and was already all
      NaN:
      `name=raw_output[0] shape=(1, 1, 248320) dtype=torch.float32 finite=0/248320 nan=248320`.
    - Therefore the failure is not a simple "finite logits are in another
      output slot" wrapper-order issue.
- Local verification:
  - `python3 -m py_compile contrib/models/Qwen3.6-27B/src/modeling_qwen35.py contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py validation_scripts/qwen36_hybrid_apc_validation.py`
  - `python3 -m pytest contrib/models/Qwen3.6-27B/test/unit/test_qwen36_model_aliases.py contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py contrib/models/Qwen3.6-27B/test/unit/test_hybrid_apc_validation.py`
  - Result: `21 passed` after `f283e93`.
  - After `980b918`, focused local and remote unit suites passed:
    `22 passed`.

## External Reference

AWS Neuron documentation and the existing NxDI `inference_demo.py` both use
special FP8 environment handling for `f8e4m3`:

- `XLA_HANDLE_SPECIAL_SCALAR=1`
- `UNSAFE_FP8FNCAST=1`

Those are now set by the Qwen3.6 FP8 compile and precompiled-artifact serving
entrypoints.

## Open Questions

- Why does on-device sampling feed an invalid token id into TKG
  (`2143289344` in the BF16 PA9 run) even though the TKG block metadata is
  in range?
- Why does the fused DeltaNet CTE NKI kernel return all-NaN logits around
  105-106 active tokens, while the legacy per-chunk NKI CTE backend stays
  finite through the 463-token validation prompt?
- Why do host-side BF16 logits/CPU sampling collapse to dummy token `0` after
  CTE/TKG active-length handling is fixed? The dummy token is downstream of
  all-NaN logits, not yet an independent sampler bug.
- Why does FP8 host-side `output_logits=True` return all-NaN logits even when
  special scalar handling is set?
- Whether the older `contrib/qwen36-27b-vllm-apc-pr` branch has a serving/runtime
  detail outside the Qwen compile path that avoids these sampler/logits issues.

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
