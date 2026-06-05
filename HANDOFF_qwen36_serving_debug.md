# Qwen3.6 Coherent Prefill-Speed Rebuild Notes

## 2026-06-05 coherent rebuild from decode-step baseline

Source branch: `codex/qwen36-prefill-speed-coherent`

Local clone: `/private/tmp/inferentia-gdn-prefill-speed-coherent`

Compile host: `ubuntu@16.26.135.243`

Runtime host target: `ubuntu@16.26.184.190`

Baseline source commit on compile host before rebuild patches: `834543e`

Ported rebuild commits:

- `157624a` remote / `4ee0abb` local: diagnostics and launch/compile plumbing
- `6be38e1` remote / `bac881e` local: Hybrid APC same-request active carry
- `dc54d22` remote / `0c0bf94` local: opt-in segmented DeltaNet prefill path
- `d00156d` remote / `f2c775c` local: FP8 compile policy flags

### Error log

1. Local unit-test invocation used an invalid dotted module path.
   - Command: `python3 -m unittest contrib.models.Qwen3.6-27B.test.unit.test_hybrid_apc_manager`
   - Error: `ModuleNotFoundError: No module named 'contrib.models.Qwen3'`
   - Context: local clean clone, after Hybrid APC active-carry edits.
   - Root cause: the `Qwen3.6-27B` directory name contains characters that cannot be addressed as a dotted Python module.
   - Mitigation: use unittest discovery with `PYTHONPATH=src`.
   - Verification: `PYTHONPATH=src python3 -m unittest discover -s contrib/models/Qwen3.6-27B/test/unit -p test_hybrid_apc_manager.py` passed 56 tests.

2. Local unit-test discovery initially missed the source path.
   - Command: `python3 -m unittest discover -s contrib/models/Qwen3.6-27B/test/unit -p test_hybrid_apc_manager.py`
   - Error: `ModuleNotFoundError: No module named 'neuronx_distributed_inference'`
   - Context: local clean clone, system Python.
   - Root cause: repo `src` was not on `PYTHONPATH`.
   - Mitigation: rerun with `PYTHONPATH=src`.
   - Verification: Hybrid APC and scheduler patch suites passed.

3. Remote tests under system Python did not have Torch.
   - Command: `python3 -m unittest discover ...` on `ubuntu@16.26.135.243`
   - Error: `ModuleNotFoundError: No module named 'torch'`
   - Context: compile host after applying the first three rebuild patches.
   - Root cause: system Python is not the Neuron/Torch environment.
   - Mitigation: use `/home/ubuntu/venvs/neuron_230_segmented_cte/bin/python`.
   - Verification: remote Hybrid APC suite passed 56 tests and scheduler suite passed 51 tests.

4. First remote compile launch could not execute the patch-created script.
   - Command: `./tmp_compile_qwen32k_segcte2048_gdnseg512.sh`
   - Error: `bash: line 1: ./tmp_compile_qwen32k_segcte2048_gdnseg512.sh: Permission denied`
   - Context: compile host source tree after `git am`; script file mode was not executable.
   - Root cause: patch-created scripts did not preserve executable mode.
   - Mitigation: invoke the driver as `bash tmp_compile_qwen32k_segcte2048_gdnseg512.sh`.
   - Verification: the next launch entered the Python compile entrypoint.

5. Second remote compile launch failed before compilation because the baseline compile harness lacked ported CLI flags.
   - Artifact base: `qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T131438Z_coherent_rebuild_direct_scan0`
   - Log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T131438Z_coherent_rebuild_direct_scan0_compile.log`
   - Error: `qwen36_27b_compile_fp8.py: error: unrecognized arguments: --fp8-quantize-linear-attn-gates --disable-context-encoding-argmax-kernel`
   - Root cause: the compile driver already requested old-policy/debug flags, but the decode-step baseline compile entrypoint did not parse them.
   - Mitigation: added `--fp8-quantize-linear-attn-gates`, `--fp8-exclude-groups`, `--postprocess-only`, `--disable-argmax-kernel`, and `--disable-context-encoding-argmax-kernel` support in `qwen36_27b_compile_fp8.py`.
   - Verification: local and remote Python compile checks passed; local and remote Hybrid APC/scheduler unit suites passed.

6. CTE2048 compile with stock QKV CTE NKI failed during HLO generation.
   - Artifact base: `qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132224Z_coherent_rebuild_direct_scan0`
   - PID: `46754`
   - Log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132224Z_coherent_rebuild_direct_scan0_compile.log`
   - Failing path: `src/neuronx_distributed_inference/modules/attention/gqa.py:_kernel_qkv_forward` calling `/home/ubuntu/nki-library-2.30/src/nkilib_src/nkilib/core/qkv/qkv_cte.py`
   - Error: `[NCC_INKI016] Kernel validation exception: [QKV CTE Kernel] weights.shape[1] must be <= 4096, but got 5120.`
   - Inputs/flags: `ENABLE_QKV_NKI_KERNELS=1`, `CTE_BUCKETS_RAW=2048`, `PREFIX_CTE_ATTENTION_BACKEND=segmented_cte`, `PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`, `ENABLE_KV_CACHE_QUANT=0`, `QUANTIZE_LM_HEAD=1`, `FP8_QUANTIZE_LINEAR_ATTN_GATES=1`.
   - Root cause hypothesis: the stock NKI Library QKV CTE kernel cannot handle Qwen3.6's 5120-wide input projection in this baseline stack. This is a compile-time kernel capability limit, not a model coherence result.
   - Mitigation: do not use stock `--enable-qkv-nki-kernels` for the first coherent-speed anchor. Relaunch the same CTE2048 segmented-attention/GDN candidate with `ENABLE_QKV_NKI_KERNELS=0`, then revisit QKV speed as a separate kernel-port task.
   - Verification: pending fallback compile.

7. Standard-QKV fallback compile exposed a remaining Python `raise` inside the segmented CTE NKI kernel.
   - Artifact base: `qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132507Z_coherent_rebuild_stdqkv_direct_scan0`
   - PID: `47806`
   - Log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132507Z_coherent_rebuild_stdqkv_direct_scan0_compile.log`
   - Failing path: `src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/attention_segmented_cte_256.py:180`
   - Error: `NKI does not support 'raise' statements; use 'if/else' control flow within kernels, or 'assert' for fatal errors`
   - Inputs/flags: `ENABLE_QKV_NKI_KERNELS=0`, `CTE_BUCKETS_RAW=2048`, `PREFIX_CTE_ATTENTION_BACKEND=segmented_cte`, `PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512`, `ENABLE_KV_CACHE_QUANT=0`, `QUANTIZE_LM_HEAD=1`, `FP8_QUANTIZE_LINEAR_ATTN_GATES=1`.
   - Root cause: Slice C replaced some NKI-hostile raises, but missed the `k_pre_transposed` guard in the segmented CTE load helper.
   - Mitigation: replace the `raise ValueError` with `kernel_assert(not k_pre_transposed, ...)`.
   - Verification: pending retry compile.

8. Completed standard-QKV CTE2048 compile produced BF16 recurrent checkpoint-bank slots.
   - Artifact base: `qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132739Z_coherent_rebuild_stdqkv2_direct_scan0`
   - Log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132739Z_coherent_rebuild_stdqkv2_direct_scan0_compile.log`
   - Evidence: `COMPILE_DONE` was present and safetensor inspection showed each TP shard had 48 recurrent checkpoint tensors with dtype `torch.bfloat16` and 48 conv checkpoint tensors with dtype `torch.bfloat16`.
   - Inputs/flags: `--gdn-recurrent-cache-dtype float32`, `--gdn-conv-cache-dtype bfloat16`, `ENABLE_QKV_NKI_KERNELS=0`, `CTE_BUCKETS_RAW=2048`, `PREFIX_CTE_ATTENTION_BACKEND=segmented_cte`.
   - Root cause: `_ensure_hybrid_checkpoint_weights` used `NeuronConfig.torch_dtype` for both recurrent and conv checkpoint-bank tensors and skipped existing checkpoint-bank keys, so postprocess-only could not correct an already-added BF16 recurrent bank.
   - Mitigation: make `_ensure_hybrid_checkpoint_weights` derive recurrent and conv dtypes from `gdn_recurrent_cache_dtype` / `gdn_conv_cache_dtype`, and rewrite existing checkpoint-bank tensors when their dtype does not match.
   - Verification: postprocess-only repair changed recurrent banks to `torch.float32`, but runtime load then failed because the compiled graph expected BF16 checkpoint-bank tensors.

9. Postprocess-only FP32 recurrent-bank repair made the completed artifact unloadable.
   - Runtime host: `ubuntu@16.26.184.190`
   - Runtime log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/coherent_rebuild_stdqkv2_runtime_20260605T141619Z.log`
   - Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132739Z_coherent_rebuild_stdqkv2_direct_scan0`
   - Error: `Incorrect data type for checkpoint key hybrid_gdn_checkpoint_cache.recurrent_slots.0: received float, expected c10::BFloat16`
   - Root cause: the compiled `model.pt` was traced with BF16 checkpoint-bank parameter dtype. Rewriting safetensors to FP32 after compile cannot change the traced model's expected parameter dtype.
   - Mitigation: for this artifact, restore recurrent checkpoint banks to BF16 and launch with `--gdn-recurrent-cache-dtype bfloat16`. A true FP32 recurrent checkpoint-bank artifact requires a source fix before compile so the traced checkpoint-bank parameters are FP32.
   - Verification: pending BF16 restore and runtime launch.

### Current next step

Do not postprocess-only a BF16-traced artifact to FP32 recurrent banks. For this completed artifact, launch with BF16 recurrent banks:

```bash
GDN_RECURRENT_CACHE_DTYPE=bfloat16 \
bash tmp_launch_qwen36_segcte2048.sh ...
```

Compile the same coherent-speed anchor with standard QKV after the segmented CTE `kernel_assert` fix:

```bash
ENABLE_QKV_NKI_KERNELS=0 \
CTE_BUCKETS_RAW=2048 \
PREFIX_CTE_ATTENTION_BACKEND=segmented_cte \
PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512 \
QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512 \
QWEN36_DELTANET_MULTIHEAD_CTE=0 \
ENABLE_KV_CACHE_QUANT=0 \
QUANTIZE_LM_HEAD=1 \
FP8_QUANTIZE_LINEAR_ATTN_GATES=1 \
bash tmp_compile_qwen32k_segcte2048_gdnseg512.sh
```
