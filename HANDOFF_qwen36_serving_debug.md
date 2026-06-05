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
   - Refined root cause after NKI docs/source inspection: this was not a hidden-width limit. AWS/NKI QKV documents `fused_qkv_weights` as `[H, I]` and QKV CTE validates `I <= 4096`; Qwen3.6 TP=4 has `H=5120` and local `I=(24/4 + 2*(4/4))*256 = 2048`. The observed `weights.shape[1]=5120` means the FP8 fused-QKV parameter reached the kernel as `[I, H]`, so quantization had dropped the transposed `[H, I]` loader/layout contract.
   - Secondary root cause: the wrapper did not pass `QuantizationType.ROW` or `qkv_w_scale` for FP8 per-channel QKV weights, even though the NKI QKV API supports row-scale dequantization.
   - Mitigation: do not use stock `--enable-qkv-nki-kernels` for the first coherent-speed anchor. After the coherent standard-QKV anchor is green, fix QKV NKI as one isolated source slice by preserving the transposed layout through quantized module creation and passing row scales to the kernel.
   - Verification: standard-QKV fallback compile/runtime became the coherent anchor; QKV NKI source fix pending compile validation.

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

10. First inline runtime probe used the wrong Python environment on the runtime host.
   - Runtime host: `ubuntu@16.26.184.190`
   - Command: `python3 - <<'PY' ... from transformers import AutoTokenizer ...`
   - Error: `ModuleNotFoundError: No module named 'transformers'`
   - Context: probing the live `coherent_rebuild_stdqkv2_direct_scan0` artifact through `http://127.0.0.1:8000/v1/chat/completions`.
   - Root cause: system Python on the runtime host is not the Neuron/vLLM environment.
   - Mitigation: use `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python` for tokenizer-backed probes.
   - Verification: exact-token probes then ran successfully.

11. CTE2048 artifact is coherent through 2048 tokens but crashes on the first cached continuation at 2049.
   - Runtime host: `ubuntu@16.26.184.190`
   - Runtime log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/coherent_rebuild_stdqkv2_bf16rec_runtime_20260605T142519Z.log`
   - Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132739Z_coherent_rebuild_stdqkv2_direct_scan0`
   - Inputs/flags: standard QKV, CTE2048, segmented CTE512 prefix attention, GDN segment 512, KV BF16, recurrent checkpoint/cache BF16, Hybrid APC enabled, chunked prefill enabled.
   - Passing probes: exact prompt lengths `146`, `160`, `485`, `505`, `526`, `1225`, and `2048` all returned coherent text with matching `usage.prompt_tokens`.
   - Failing probe: exact prompt length `2049`; subsequent `2500`, `4092`, and `4096` requests failed because the engine was already dead.
   - Exact runtime error: `RuntimeError: shape '[1, 1]' is invalid for input of size 0` at `src/neuronx_distributed_inference/models/model_wrapper.py:1874`, called from `_process_async_inputs`.
   - Log evidence immediately before crash: `tag=token_generation_model index=0 name=input_ids shape=(1, 0)`, `position_ids shape=(1, 1) min=2048`, `slot_mapping shape=(1, 1) min=8448`, `computed_context_lens min=2048`, and `num_queries min=0`. The dumped scheduler output showed `scheduled_cached_reqs`, `num_computed_tokens=[2048]`, `num_scheduled_tokens=1`, `request_prefix_len=2049`, `active_suffix_len=1`, and `full_input_ids` present in Hybrid APC metadata.
   - Root cause hypothesis: vLLM V1 cached/chunked-prefill continuation schedules one active suffix token after the 2048-token CTE chunk, but the Neuron token-generation input path does not merge the request metadata for non-context executions. The model wrapper receives an impossible tuple: empty `input_ids` but non-empty active `position_ids` and `slot_mapping`.
   - Mitigation applied: add `_repair_cached_chunked_prefill_tkg_inputs` in `async_execution.py`; for non-context execution, attach Hybrid APC owner metadata, reconstruct the active suffix from `full_input_ids[computed_context_lens:computed_context_lens + active_suffix_len]`, set `num_queries` to the suffix length, and then build inert Hybrid APC args.
   - Verification: local `PYTHONPATH=src python3 -m py_compile src/neuronx_distributed_inference/modules/async_execution.py test/unit/modules/test_async_execution.py` passed; local `PYTHONPATH=src python3 -m unittest test.unit.modules.test_async_execution.TestCachedChunkedPrefillTkgRepair` passed. Runtime verification is pending source sync and vLLM restart.

12. After fixing 2049, cold 2500 fails because Hybrid APC is prepared twice for the same chunk continuation.
   - Runtime host: `ubuntu@16.26.184.190`
   - Runtime log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/coherent_rebuild_stdqkv2_cachedrepair_cold_runtime_20260605T144421Z.log`
   - Passing probe after fix #11: exact prompt length `2049` returned coherent text with `usage.prompt_tokens=2049`.
   - Failing probe: exact prompt length `2500`, unique cold marker to avoid cross-request prefix reuse.
   - Error: `ValueError: hybrid APC received an attention prefix hit without a matching GDN checkpoint; scheduler must intersect attention KV hits with GDN checkpoint hits or disable prefix reuse for this request`
   - Log evidence: first chunk prepared and committed `commit_prefix_len=2048 commit_slot=0`; second chunk prepared by async execution with `attention_hit_len=2048`, `request_prefix_len=2500`, `prepared_shape=(1, 452)`, `computed=tensor([[2048]])`, `num_queries=tensor([[452]])`, `restore_mask=tensor([0])`, `commit_mask=tensor([0])`; then `modeling_qwen35.py:_prepare_hybrid_apc_pad_inputs` prepared the same row again inside `pad_inputs` and raised.
   - Root cause: `pad_inputs` prepares when both Hybrid APC masks are zero. That is correct for raw cold inputs, but wrong for an already-prepared same-request continuation whose masks are intentionally inert because active carry is authoritative.
   - Mitigation applied: async execution now sets `_qwen36_hybrid_apc_skip_pad_prepare_once` on the selected model wrapper whenever it has already prepared Hybrid APC inputs; `Qwen35ModelWrapper._prepare_hybrid_apc_pad_inputs` consumes that one-shot flag and returns the prepared args unchanged.
   - Verification: local `PYTHONPATH=src python3 -m py_compile src/neuronx_distributed_inference/modules/async_execution.py contrib/models/Qwen3.6-27B/src/modeling_qwen35.py` passed; local `PYTHONPATH=src python3 -m unittest test.unit.modules.test_async_execution.TestCachedChunkedPrefillTkgRepair` passed. Runtime verification after source sync passed exact prompt lengths `2500`, `4092`, `4096`, `8192`, and `16384` with coherent output and matching `usage.prompt_tokens`.

13. The coherent CTE2048 standard-QKV anchor is too slow for the prefill-speed target.
   - Runtime host: `ubuntu@16.26.184.190`
   - Runtime log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/coherent_rebuild_stdqkv2_doubleprepfix_runtime_20260605T144922Z.log`
   - Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132739Z_coherent_rebuild_stdqkv2_direct_scan0`
   - Inputs/flags: standard QKV, CTE2048, segmented CTE512 prefix attention, GDN segment 512, KV BF16, recurrent checkpoint/cache BF16, Hybrid APC enabled.
   - Coherence evidence: exact `8192` prompt returned coherent text with `usage.prompt_tokens=8192`; exact `16384` prompt with `max_tokens=1` returned a normal first token and `usage.prompt_tokens=16384`.
   - Speed evidence: 16k `max_tokens=1` non-streaming wall time was `26.353s`, so conservative prompt throughput was `16384 / 26.353 = 621.7 prompt tok/s`.
   - Root cause hypothesis, refined: this anchor deliberately disabled QKV NKI because the FP8 fused-QKV path lost its `[H, I]` parameter layout and row-scale contract after quantization. Coherence is restored, but the speed path is still missing a validated QKV NKI rebuild.
   - Mitigation: keep this artifact as the coherent CTE2048 anchor; next speed work should port/adapt the working QKV tiled path for 5120-wide Qwen instead of reintroducing unrelated dense moat kernels.
   - Verification: final serve-log scan after 2500/4092/4096/8192/16384 showed no `negative token_id`, `out-of-vocab token_id`, `fallback argmax`, `finite=0`, `nan=`, `NRT_RESOURCE`, `EngineDeadError`, `RuntimeError`, `ValueError`, `Traceback`, or `InternalServerError`; backend and proxy health were OK.

14. QKV NKI FP8 source audit found a missing quantized layout/scale contract.
   - Local source paths: `src/neuronx_distributed_inference/modules/attention/gqa.py`, `src/neuronx_distributed_inference/modules/attention/utils.py`.
   - Remote docs/source evidence: AWS QKV API documents `fused_qkv_weights` as `[H, I]`; compile-host NKI Library 2.30 `qkv_cte_utils.py` validates `_H == H` and `I <= 4096`; `qkv_cte.py` supports `QuantizationType.ROW` with `qkv_w_scale` shape `[1, I]` or `[128, I]`.
   - Inputs/flags that expose it: `ENABLE_QKV_NKI_KERNELS=1`, `weight_dtype=fp8_full`, `QUANTIZE_LM_HEAD=1`, `FP8_QUANTIZE_LINEAR_ATTN_GATES=1`, Qwen3.6 TP=4.
   - Root cause: `GroupQueryAttention_QKV.__init__` transposed non-quantized `Wqkv.weight`, but did not install a `post_create_quantized_module_hook`, so FP8 quantization replaced the module with default `[I, H]` weight and `[I, 1]` scale metadata. `_kernel_qkv_forward` then derived `fused_qkv_size` from `Wqkv.weight.shape[1]`, mistaking hidden size `5120` for output width, and passed no row scale.
   - Mitigation applied locally: added `preprocess_quantized_qkv_nki_layer`, FP8-preserving transposed state-dict loaders, a `[128, I]` QKV row-scale loader, explicit QKV layout/scale guards, and `QuantizationType.ROW` / `qkv_w_scale` kernel arguments for quantized QKV.
   - Verification: local `py_compile` passed; remote compile-host `test/unit/modules/attention/test_gqa.py` passed 11 tests under the Neuron venv with `PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin:$PATH NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_CC_FLAGS="--target trn2"`.

15. Local QKV unit-test run used a Python environment without Neuron/NxD.
   - Command: `PYTHONPATH=src python3 -m pytest -q test/unit/modules/attention/test_gqa.py`
   - Error: `ModuleNotFoundError: No module named 'neuronx_distributed'`
   - Context: local macOS/workspace Python after QKV NKI source edits.
   - Root cause: local Python lacks the Neuron/NxD package needed to import `neuronx_distributed_inference.modules.attention.gqa`.
   - Mitigation: use local `py_compile` for syntax and run the real unit test on `ubuntu@16.26.135.243` with `/home/ubuntu/venvs/neuron_230_segmented_cte/bin/python`.
   - Verification: local `PYTHONPATH=src python3 -m py_compile src/neuronx_distributed_inference/modules/attention/gqa.py test/unit/modules/attention/test_gqa.py` passed.

16. Remote QKV unit-test import exposed environment and circular-import issues before assertions ran.
   - Command 1: `PYTHONPATH=src /home/ubuntu/venvs/neuron_230_segmented_cte/bin/python -m pytest -q test/unit/modules/attention/test_gqa.py`
   - Error 1: `FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'`
   - Root cause 1: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin` was not on `PATH`.
   - Command 2: same command with that venv bin on `PATH`.
   - Error 2: `RuntimeError: Unsupported Platform - r7i.24xlarge`; compile host is CPU-only and needs an explicit target override.
   - Mitigation 2: rerun with `NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_CC_FLAGS="--target trn2"`.
   - Error 3 after the target override: `ImportError: cannot import name 'replicate_kv' from partially initialized module 'neuronx_distributed_inference.modules.attention.gqa'`; import path was `gqa.py -> lora_serving.__init__ -> lora_checkpoint.py -> gqa.replicate_kv`.
   - Root cause 3: `gqa.py` imported `is_lora_module` at module import time, which executes the LoRA package initializer before `replicate_kv` is defined.
   - Mitigation applied locally: replace the top-level LoRA helper import with a lazy `_is_lora_module()` wrapper and update QKV/O-proj call sites.
   - Verification: remote compile-host `PYTHONPATH=src /home/ubuntu/venvs/neuron_230_segmented_cte/bin/python -m py_compile src/neuronx_distributed_inference/modules/attention/gqa.py test/unit/modules/attention/test_gqa.py` passed with the corrected PATH/target override; remote `pytest -q test/unit/modules/attention/test_gqa.py` passed 11 tests with 11 warnings.

### Current next step

Use the live standard-QKV CTE2048 artifact as the coherent anchor. The next speed slice is the one-variable QKV NKI FP8 layout/scale fix; do not add packed qkvgate, output-proj NKI, quantized MLP NKI, or FP8 KV until QKV speed is isolated and the same coherence matrix stays green.

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
